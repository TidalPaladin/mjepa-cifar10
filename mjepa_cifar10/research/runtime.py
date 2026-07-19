from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import uuid
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any, Final, Mapping, Self, Sequence

from .models import RunSpec, RunState, StudySpec, StudyState


GIB: Final[int] = 1024**3
TERMINAL_STATUSES: Final = frozenset(("completed", "failed", "timed_out"))
ACTIVE_STATUSES: Final = frozenset(("launching", "running"))
TERMINAL_FILENAME: Final[str] = "terminal.json"
HEARTBEAT_FILENAME: Final[str] = "worker.json"
RETENTION_LOG_FILENAME: Final[str] = "retention.jsonl"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(value, temporary_file, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class StateStore(AbstractContextManager["StateStore"]):
    """Atomic state storage with a process lock for launch/recovery operations."""

    def __init__(self, study_dir: Path):
        self.study_dir = study_dir
        self.path = study_dir / "state.json"
        self.lock_path = study_dir / ".state.lock"
        self._lock_file: IO[str] | None = None

    def __enter__(self) -> Self:
        self.study_dir.mkdir(parents=True, exist_ok=True)
        self._lock_file = self.lock_path.open("a+")
        fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._lock_file is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def load(self) -> StudyState:
        return StudyState.from_dict(json.loads(self.path.read_text()))

    def save(self, state: StudyState) -> None:
        state.updated_at = utc_now()
        atomic_write_json(self.path, state.to_dict())

    def load_or_create(self, spec: StudySpec, spec_path: Path) -> StudyState:
        if self.path.is_file():
            state = self.load()
            if state.study_id != spec.id:
                raise ValueError(f"state study ID {state.study_id!r} does not match {spec.id!r}")
            return state
        now = utc_now()
        state = StudyState(
            study_id=spec.id,
            spec_path=str(spec_path.resolve()),
            created_at=now,
            updated_at=now,
            runs={run.id: RunState(run) for run in spec.initial_runs()},
        )
        self.save(state)
        return state


class GPULock(AbstractContextManager["GPULock"]):
    def __init__(self, physical_gpu: int, lock_root: Path = Path("/tmp/mjepa-cifar10-research")):
        self.physical_gpu = physical_gpu
        self.path = lock_root / f"gpu-{physical_gpu}.lock"
        self._file: IO[str] | None = None

    def acquire(self, *, blocking: bool = False) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a+")
        operation = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
        try:
            fcntl.flock(self._file.fileno(), operation)
        except BlockingIOError:
            self._file.close()
            self._file = None
            return False
        self._file.seek(0)
        self._file.truncate()
        self._file.write(f"pid={os.getpid()} host={socket.gethostname()} acquired={utc_now()}\n")
        self._file.flush()
        return True

    def __enter__(self) -> Self:
        if not self.acquire(blocking=False):
            raise RuntimeError(f"physical GPU {self.physical_gpu} is already locked")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def available_physical_gpus(
    physical_gpus: Sequence[int],
    *,
    reserved: Sequence[int] = (),
    lock_root: Path = Path("/tmp/mjepa-cifar10-research"),
) -> tuple[int, ...]:
    available: list[int] = []
    for physical_gpu in physical_gpus:
        if physical_gpu in reserved:
            continue
        lock = GPULock(physical_gpu, lock_root)
        if lock.acquire(blocking=False):
            available.append(physical_gpu)
            lock.__exit__(None, None, None)
    return tuple(available)


def study_directory(spec: StudySpec, repo_root: Path) -> Path:
    log_root = spec.log_root if spec.log_root.is_absolute() else repo_root / spec.log_root
    return (log_root / spec.id).resolve()


def estimate_checkpoint_size(log_roots: Sequence[Path], fallback_gib: int, model_class: str) -> int:
    checkpoints: list[Path] = []
    for root in log_roots:
        if root.is_dir():
            checkpoints.extend(
                path
                for path in root.rglob("checkpoint.pt")
                if path.is_file() and _checkpoint_matches_model_class(path, model_class)
            )
    if not checkpoints:
        return fallback_gib * GIB
    recent = sorted(checkpoints, key=lambda path: path.stat().st_mtime, reverse=True)[:20]
    return max(path.stat().st_size for path in recent)


def _checkpoint_matches_model_class(checkpoint: Path, model_class: str) -> bool:
    metadata_path = checkpoint.parent / "metadata.json"
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return metadata.get("model_class") == model_class


def required_free_bytes(minimum_free_gib: int, concurrent_jobs: int, estimated_checkpoint_size: int) -> int:
    return minimum_free_gib * GIB + 2 * concurrent_jobs * estimated_checkpoint_size


def storage_report(spec: StudySpec, repo_root: Path) -> dict[str, int | bool]:
    estimated_size = estimate_checkpoint_size(
        (repo_root / "logs",),
        spec.resources.fallback_checkpoint_gib,
        spec.model_class,
    )
    required = required_free_bytes(
        spec.resources.minimum_free_gib,
        spec.resources.max_concurrent_jobs,
        estimated_size,
    )
    free = shutil.disk_usage(repo_root).free
    return {
        "free_bytes": free,
        "estimated_checkpoint_bytes": estimated_size,
        "required_free_bytes": required,
        "sufficient": free >= required,
    }


def assert_storage_available(spec: StudySpec, repo_root: Path) -> dict[str, int | bool]:
    report = storage_report(spec, repo_root)
    if not report["sufficient"]:
        raise RuntimeError(
            "insufficient checkpoint space: "
            f"{report['free_bytes']} bytes free, {report['required_free_bytes']} bytes required"
        )
    return report


def build_run_command(
    spec: StudySpec,
    run: RunSpec,
    repo_root: Path,
    run_dir: Path,
    wandb_run_id: str,
    physical_gpu: int,
    provenance_file: Path,
) -> list[str]:
    if run.command is not None:
        return list(run.command)
    common_arguments = [
        str(run.config if run.config.is_absolute() else repo_root / run.config),
        str(spec.data if spec.data.is_absolute() else repo_root / spec.data),
        "--exact-log-dir",
        str(run_dir),
        "--local-rank",
        "0",
        "--seed",
        str(run.seed),
        "--wandb-run-id",
        wandb_run_id,
        "--wandb-project",
        spec.wandb_project,
        "--wandb-group",
        spec.wandb_group,
        "--study-id",
        spec.id,
        "--model-class",
        spec.model_class,
        "--variant",
        run.variant,
        "--physical-gpu",
        str(physical_gpu),
        "--provenance-file",
        str(provenance_file),
        "--name",
        run.id,
    ]
    if spec.wandb_entity:
        common_arguments.extend(("--wandb-entity", spec.wandb_entity))
    if run.evaluate_test:
        common_arguments.append("--evaluate-test")
    if run.kind == "pretrain":
        return [sys.executable, str(repo_root / "scripts" / "pretrain.py"), *common_arguments]
    if run.source_checkpoint is None:
        raise ValueError(f"SFT run {run.id} has no source checkpoint")
    command = [
        sys.executable,
        str(repo_root / "scripts" / "finetune.py"),
        *common_arguments,
        "--checkpoint",
        str(run.source_checkpoint),
        "--subset-seed",
        str(run.subset_seed if run.subset_seed is not None else run.seed),
    ]
    if run.shots_per_class is not None:
        command.extend(("--shots-per-class", str(run.shots_per_class)))
    return command


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_command_with_timeout(
    command: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_seconds: int,
    log_file: IO[str],
) -> tuple[int, bool]:
    process = subprocess.Popen(
        list(command),
        cwd=env["MJEPA_RESEARCH_REPO_ROOT"],
        env=dict(env),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=timeout_seconds), False
    except subprocess.TimeoutExpired:
        _terminate_process_group(process)
        return 124, True


def run_worker(
    spec_path: Path,
    run_id: str,
    physical_gpu: int,
    repo_root: Path,
    lock_root: Path = Path("/tmp/mjepa-cifar10-research"),
) -> int:
    spec = StudySpec.from_path(spec_path)
    study_dir = study_directory(spec, repo_root)
    with StateStore(study_dir) as store:
        state = store.load()
        run_state = state.runs[run_id]
    run_dir = Path(run_state.run_dir or study_dir / "runs" / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = run_dir / TERMINAL_FILENAME
    wandb_run_id = run_state.wandb_run_id or uuid.uuid4().hex[:8]
    started_at = utc_now()
    try:
        with GPULock(physical_gpu, lock_root):
            atomic_write_json(
                run_dir / HEARTBEAT_FILENAME,
                {"status": "running", "pid": os.getpid(), "started_at": started_at, "physical_gpu": physical_gpu},
            )
            from .provenance import collect_provenance

            provenance_path = run_dir / "provenance.json"
            provenance = collect_provenance(spec, repo_root).to_dict()
            command = build_run_command(
                spec,
                run_state.spec,
                repo_root,
                run_dir,
                wandb_run_id,
                physical_gpu,
                provenance_path,
            )
            provenance.update(
                {
                    "physical_gpu": physical_gpu,
                    "hostname": socket.gethostname(),
                    "command": command,
                    "config": str(run_state.spec.config),
                    "local_weight_disposition": "retained",
                }
            )
            atomic_write_json(provenance_path, provenance)
            environment = dict(os.environ)
            environment.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(physical_gpu),
                    "MJEPA_RESEARCH_REPO_ROOT": str(repo_root),
                    "PYTHONUNBUFFERED": "1",
                }
            )
            with (run_dir / "run.log").open("a", encoding="utf-8") as log_file:
                exit_code, timed_out = run_command_with_timeout(
                    command,
                    env=environment,
                    timeout_seconds=spec.resources.timeout_seconds,
                    log_file=log_file,
                )
            status = "timed_out" if timed_out else ("completed" if exit_code == 0 else "failed")
            terminal = {
                "status": status,
                "exit_code": exit_code,
                "started_at": started_at,
                "finished_at": utc_now(),
                "physical_gpu": physical_gpu,
                "wandb_run_id": wandb_run_id,
                "command": command,
                "error": "24-hour job timeout exceeded" if timed_out else None,
            }
    except Exception as error:
        terminal = {
            "status": "failed",
            "exit_code": 1,
            "started_at": started_at,
            "finished_at": utc_now(),
            "physical_gpu": physical_gpu,
            "wandb_run_id": wandb_run_id,
            "error": f"{type(error).__name__}: {error}",
        }
    atomic_write_json(terminal_path, terminal)
    atomic_write_json(
        run_dir / HEARTBEAT_FILENAME,
        {"status": terminal["status"], "finished_at": terminal["finished_at"]},
    )
    return int(terminal["exit_code"])


def reconcile_state(state: StudyState) -> bool:
    changed = False
    for run in state.runs.values():
        if run.run_dir is None:
            continue
        terminal_path = Path(run.run_dir) / TERMINAL_FILENAME
        if terminal_path.is_file() and run.status not in TERMINAL_STATUSES:
            terminal = json.loads(terminal_path.read_text())
            run.status = terminal["status"]
            run.exit_code = int(terminal["exit_code"])
            run.started_at = terminal.get("started_at", run.started_at)
            run.finished_at = terminal.get("finished_at")
            run.error = terminal.get("error")
            run.wandb_run_id = terminal.get("wandb_run_id", run.wandb_run_id)
            metadata_path = Path(run.run_dir) / "metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                run.wandb_run_id = metadata.get("wandb_run_id", run.wandb_run_id)
                run.wandb_url = metadata.get("wandb_url", run.wandb_url)
            run.decision = "retryable" if run.status in ("failed", "timed_out") else run.decision
            changed = True
            continue
        heartbeat_path = Path(run.run_dir) / HEARTBEAT_FILENAME
        if heartbeat_path.is_file() and run.status == "launching":
            heartbeat = json.loads(heartbeat_path.read_text())
            if heartbeat.get("status") == "running":
                run.status = "running"
                changed = True
        if run.status in ACTIVE_STATUSES and run.pid is not None and not _pid_is_alive(run.pid):
            run.status = "failed"
            run.decision = "retryable"
            run.exit_code = 1
            run.finished_at = utc_now()
            run.error = "supervisor exited without writing terminal state"
            changed = True
    return changed


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def launch_available_runs(
    spec: StudySpec,
    spec_path: Path,
    repo_root: Path,
    *,
    development: bool = False,
    lock_root: Path = Path("/tmp/mjepa-cifar10-research"),
) -> StudyState:
    if not development:
        from .provenance import assert_launch_provenance

        assert_launch_provenance(spec, repo_root)
    assert_storage_available(spec, repo_root)
    study_dir = study_directory(spec, repo_root)
    with StateStore(study_dir) as store:
        state = store.load_or_create(spec, spec_path)
        reconcile_state(state)
        active = [run for run in state.runs.values() if run.status in ACTIVE_STATUSES]
        capacity = max(0, spec.resources.max_concurrent_jobs - len(active))
        reserved = [run.physical_gpu for run in active if run.physical_gpu is not None]
        available = available_physical_gpus(spec.resources.physical_gpus, reserved=reserved, lock_root=lock_root)
        for run, physical_gpu in zip(
            (run for run in state.runs.values() if run.status == "pending"),
            available[:capacity],
            strict=False,
        ):
            assert_storage_available(spec, repo_root)
            run_dir = (study_dir / "runs" / run.spec.id).resolve()
            run_dir.mkdir(parents=True, exist_ok=True)
            run.status = "launching"
            run.physical_gpu = physical_gpu
            run.started_at = utc_now()
            run.run_dir = str(run_dir)
            run.wandb_run_id = run.wandb_run_id or uuid.uuid4().hex[:8]
            supervisor_log = (run_dir / "supervisor.log").open("a", encoding="utf-8")
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(repo_root / "scripts" / "research.py"),
                        "_worker",
                        str(spec_path.resolve()),
                        run.spec.id,
                        str(physical_gpu),
                        "--repo-root",
                        str(repo_root),
                        "--lock-root",
                        str(lock_root),
                    ],
                    cwd=repo_root,
                    stdout=supervisor_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    text=True,
                )
            finally:
                supervisor_log.close()
            run.pid = process.pid
            store.save(state)
        store.save(state)
        return state


def cleanup_run_weights(
    state: StudyState,
    run_id: str,
    study_dir: Path,
    *,
    study_close: bool = False,
) -> tuple[Path, ...]:
    run = state.runs[run_id]
    if run.status not in TERMINAL_STATUSES:
        raise ValueError(f"cannot clean nonterminal run {run_id}")
    if run.decision not in ("rejected", "retryable"):
        raise ValueError(f"cannot clean run {run_id} with decision {run.decision!r}")
    if run.decision == "retryable" and not study_close:
        raise ValueError("retryable failed-run checkpoints are retained until retry or study close")
    managed_runs_root = (study_dir / "runs").resolve()
    run_dir = Path(run.run_dir or "").resolve()
    if run_dir.parent != managed_runs_root or run_dir.name != run_id:
        raise ValueError(f"refusing cleanup outside exact managed run directory: {run_dir}")
    targets = [run_dir / "checkpoint.pt"]
    if study_close:
        targets.append(run_dir / "backbone.safetensors")
    deleted: list[Path] = []
    bytes_freed = 0
    for target in targets:
        if target.is_file():
            bytes_freed += target.stat().st_size
            target.unlink()
            deleted.append(target)
    run.bytes_freed += bytes_freed
    run.checkpoint_disposition = "deleted-not-recoverable" if deleted else run.checkpoint_disposition
    metadata_path = run_dir / "metadata.json"
    if deleted and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata["local_weight_disposition"] = run.checkpoint_disposition
        provenance = metadata.setdefault("provenance", {})
        provenance["provenance/local_weight_disposition"] = run.checkpoint_disposition
        atomic_write_json(metadata_path, metadata)
    retention_record = {
        "timestamp": utc_now(),
        "study_id": state.study_id,
        "run_id": run_id,
        "paths": [str(path) for path in deleted],
        "bytes_freed": bytes_freed,
        "recoverable": False,
    }
    with (study_dir / RETENTION_LOG_FILENAME).open("a", encoding="utf-8") as retention_log:
        retention_log.write(json.dumps(retention_record, sort_keys=True) + "\n")
    return tuple(deleted)
