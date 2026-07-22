from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable
from contextlib import AbstractContextManager, suppress
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import IO, Any, Final, Mapping, Self, Sequence

from .models import WANDB_LOCAL_MODES, RunSpec, RunState, StudySpec, StudyState


GIB: Final[int] = 1024**3
TERMINAL_STATUSES: Final = frozenset(("completed", "failed", "timed_out"))
ACTIVE_STATUSES: Final = frozenset(("launching", "running"))
TERMINAL_FILENAME: Final[str] = "terminal.json"
HEARTBEAT_FILENAME: Final[str] = "worker.json"
NOTIFICATION_FILENAME: Final[str] = "notification.json"
RETENTION_LOG_FILENAME: Final[str] = "retention.jsonl"
WANDB_SERVICE_ENVIRONMENT_VARIABLE: Final[str] = "WANDB_SERVICE"
MINIMUM_POLL_INTERVAL_SECONDS: Final[int] = 10 * 60
STEADY_STATE_POLL_INTERVAL_SECONDS: Final[int] = 30 * 60
MAX_ROUTINE_CHECKS: Final[int] = 5


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def schedule_monitor_check(run: RunState, *, now: datetime | None = None) -> None:
    """Record one coordinator check and schedule the next bounded poll."""
    if run.status not in ACTIVE_STATUSES:
        run.next_check_at = None
        run.next_check_reason = "terminal"
        return
    check_time = now or datetime.now(UTC)
    run.routine_check_count += 1
    run.last_check_at = check_time.isoformat()
    if run.routine_check_count >= MAX_ROUTINE_CHECKS:
        run.last_check_interval_seconds = None
        run.next_check_at = None
        run.next_check_reason = "routine-budget-exhausted; watchdog-only"
        return
    interval = MINIMUM_POLL_INTERVAL_SECONDS if run.routine_check_count < 2 else STEADY_STATE_POLL_INTERVAL_SECONDS
    run.last_check_interval_seconds = float(interval)
    run.next_check_at = (check_time + timedelta(seconds=interval)).isoformat()
    run.next_check_reason = "startup-check" if run.routine_check_count < 2 else "steady-state-check"


def schedule_due_monitor_checks(runs: Iterable[RunState], *, now: datetime | None = None) -> bool:
    """Advance only due active checks and clear obsolete terminal schedules."""
    check_time = now or datetime.now(UTC)
    if check_time.tzinfo is None or check_time.utcoffset() is None:
        raise ValueError("monitor check time must include a UTC offset")
    check_time = check_time.astimezone(UTC)
    changed = False
    for run in runs:
        if run.status in TERMINAL_STATUSES:
            if run.next_check_at is not None or run.next_check_reason != "terminal":
                schedule_monitor_check(run, now=check_time)
                changed = True
            continue
        if run.status not in ACTIVE_STATUSES:
            continue
        if run.next_check_at is None:
            due = run.routine_check_count < MAX_ROUTINE_CHECKS
        else:
            scheduled_at = datetime.fromisoformat(run.next_check_at.replace("Z", "+00:00"))
            if scheduled_at.tzinfo is None or scheduled_at.utcoffset() is None:
                raise ValueError(f"run {run.spec.id} next_check_at must include a UTC offset")
            due = scheduled_at.astimezone(UTC) <= check_time
        if due:
            schedule_monitor_check(run, now=check_time)
            changed = True
    return changed


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
        _fsync_directory(path.parent)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def persist_terminal_and_queue_notification(
    terminal_path: Path,
    terminal: Mapping[str, Any],
    managed_root: Path,
    *,
    study_id: str,
    run_id: str,
) -> str | None:
    """Persist terminal truth before queueing a recoverable Codex notification."""
    atomic_write_json(terminal_path, terminal)
    try:
        from .codex_notifications import queue_notification_from_terminal

        queue_notification_from_terminal(
            terminal_path,
            managed_root,
            study_id=study_id,
            run_id=run_id,
        )
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _fsync_directory(directory: Path) -> None:
    """Flush directory metadata after an atomic replacement."""
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def append_locked_text(path: Path, entry: str, operation_id: str, *, initial_text: str = "") -> bool:
    """Append one operation-deduplicated entry using an atomic replacement."""
    marker_prefix = "<!-- autoresearch-operation:"
    marker_suffix = " -->"
    if marker_prefix in entry:
        raise ValueError("research-log entry contains reserved metadata")
    if (
        not operation_id
        or operation_id.strip() != operation_id
        or any(character in operation_id for character in "\r\n<>")
    ):
        raise ValueError("operation_id must be a nonempty single-line identifier")
    entry_digest = hashlib.sha256(entry.encode()).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = path.read_text(encoding="utf-8") if path.is_file() else initial_text
        for line in existing.splitlines():
            if not line.startswith(marker_prefix) or not line.endswith(marker_suffix):
                continue
            serialized_metadata = line[len(marker_prefix) : -len(marker_suffix)]
            if not serialized_metadata.startswith("{"):
                if serialized_metadata == operation_id:
                    return False
                continue
            try:
                metadata = json.loads(serialized_metadata)
            except json.JSONDecodeError as error:
                raise ValueError("research log contains invalid operation metadata") from error
            if not isinstance(metadata, dict) or set(metadata) != {"content_sha256", "operation_id"}:
                raise ValueError("research log contains invalid operation metadata")
            stored_digest = metadata["content_sha256"]
            stored_operation_id = metadata["operation_id"]
            if (
                not isinstance(stored_digest, str)
                or len(stored_digest) != 64
                or any(character not in "0123456789abcdef" for character in stored_digest)
                or not isinstance(stored_operation_id, str)
                or not stored_operation_id
                or stored_operation_id.strip() != stored_operation_id
                or any(character in stored_operation_id for character in "\r\n<>")
            ):
                raise ValueError("research log contains invalid operation metadata")
            if stored_operation_id != operation_id:
                continue
            if stored_digest == entry_digest:
                return False
            raise ValueError(f"operation {operation_id!r} already exists with different content")
        marker_metadata = json.dumps(
            {"content_sha256": entry_digest, "operation_id": operation_id},
            separators=(",", ":"),
            sort_keys=True,
        )
        marker = f"{marker_prefix}{marker_metadata}{marker_suffix}"
        content = existing + ("\n" if existing and not existing.endswith("\n") else "") + marker + "\n" + entry
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
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
            _fsync_directory(path.parent)
            return True
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)


def append_locked_jsonl(path: Path, value: Mapping[str, Any], operation_id: str) -> None:
    """Append a durable JSONL record under a stable lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        existing = path.read_text(encoding="utf-8") if path.is_file() else ""
        if any(
            isinstance(record, dict) and record.get("operation_id") == operation_id
            for line in existing.splitlines()
            if line.strip()
            for record in (json.loads(line),)
        ):
            return
        serialized = json.dumps({**value, "operation_id": operation_id}, sort_keys=True)
        content = existing + serialized + "\n"
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
                temporary_file.write(content)
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, path)
            _fsync_directory(path.parent)
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


def validate_managed_paths(spec: StudySpec, repo_root: Path) -> None:
    """Reject broad or overlapping paths before a study can create artifacts."""
    repo_root = repo_root.resolve()
    log_root = spec.log_root if spec.log_root.is_absolute() else repo_root / spec.log_root
    if log_root.resolve() == repo_root:
        raise ValueError(f"managed log root must not be the repository root: {log_root}")
    managed_dir = study_directory(spec, repo_root)
    if managed_dir == repo_root or not managed_dir.is_relative_to(repo_root):
        raise ValueError(f"managed study directory must be a child of the repository: {managed_dir}")
    source_paths = [
        *(spec.baseline.config,),
        *(variant.config for variant in spec.variants),
        spec.data,
    ]
    for source in source_paths:
        resolved = (source if source.is_absolute() else repo_root / source).resolve()
        if resolved == managed_dir or resolved.is_relative_to(managed_dir) or managed_dir.is_relative_to(resolved):
            raise ValueError(f"managed study directory overlaps an input or source path: {managed_dir} / {resolved}")


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
    heartbeat_callback: Any | None = None,
    heartbeat_interval_seconds: float = 60.0,
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
    if heartbeat_callback is None:
        try:
            return process.wait(timeout=timeout_seconds), False
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return 124, True

    deadline = time.monotonic() + timeout_seconds
    next_heartbeat = time.monotonic()
    try:
        while True:
            return_code = process.poll()
            if return_code is not None:
                return return_code, False
            now = time.monotonic()
            if now >= deadline:
                _terminate_process_group(process)
                return 124, True
            if now >= next_heartbeat:
                heartbeat_callback()
                next_heartbeat = now + heartbeat_interval_seconds
            try:
                process.wait(timeout=min(heartbeat_interval_seconds, max(0.01, deadline - now)))
            except subprocess.TimeoutExpired:
                continue
    except BaseException:
        with suppress(Exception):
            _terminate_process_group(process)
        raise


def build_worker_environment(base: Mapping[str, str], physical_gpu: int, repo_root: Path) -> dict[str, str]:
    environment = dict(base)
    environment.pop(WANDB_SERVICE_ENVIRONMENT_VARIABLE, None)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "MJEPA_RESEARCH_REPO_ROOT": str(repo_root),
            "PYTHONUNBUFFERED": "1",
        }
    )
    return environment


def build_managed_worker_environment(
    base: Mapping[str, str], physical_gpu: int, repo_root: Path, spec: StudySpec
) -> dict[str, str]:
    """Build an isolated worker environment and enforce external-publication consent."""
    environment = build_worker_environment(base, physical_gpu, repo_root)
    decision = spec.wandb_operation_decision("launch", environment.get("WANDB_MODE", "online"))
    if decision.effective_mode == "local-only" and decision.requested_mode not in WANDB_LOCAL_MODES:
        environment["WANDB_MODE"] = "offline"
    return environment


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
    terminal_event_id = run_state.terminal_event_id or str(uuid.uuid4())
    originating_thread_id = run_state.originating_thread_id or os.environ.get("CODEX_THREAD_ID")
    started_at = utc_now()
    try:
        with GPULock(physical_gpu, lock_root):

            def write_heartbeat() -> None:
                heartbeat_at = utc_now()
                atomic_write_json(
                    run_dir / HEARTBEAT_FILENAME,
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "started_at": started_at,
                        "heartbeat_at": heartbeat_at,
                        "physical_gpu": physical_gpu,
                        "attempt": run_state.attempt,
                        "originating_thread_id": originating_thread_id,
                    },
                )

            write_heartbeat()
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
            tracker_decision = spec.wandb_operation_decision("launch", os.environ.get("WANDB_MODE", "online"))
            environment = build_managed_worker_environment(os.environ, physical_gpu, repo_root, spec)
            provenance.update(
                {
                    "physical_gpu": physical_gpu,
                    "hostname": socket.gethostname(),
                    "command": command,
                    "config": str(run_state.spec.config),
                    "local_weight_disposition": "retained",
                    "external_tracker": {
                        "provider": "wandb" if spec.wandb_entity else None,
                        "entity": spec.wandb_entity,
                        "project": spec.wandb_project,
                        "configured_authorization": spec.wandb_authorized,
                        **tracker_decision.to_dict(),
                    },
                }
            )
            atomic_write_json(provenance_path, provenance)
            with (run_dir / "run.log").open("a", encoding="utf-8") as log_file:
                exit_code, timed_out = run_command_with_timeout(
                    command,
                    env=environment,
                    timeout_seconds=spec.resources.timeout_seconds,
                    log_file=log_file,
                    heartbeat_callback=write_heartbeat,
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
                "attempt": run_state.attempt,
                "terminal_event_id": terminal_event_id,
                "originating_thread_id": originating_thread_id,
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
            "attempt": run_state.attempt,
            "terminal_event_id": terminal_event_id,
            "originating_thread_id": originating_thread_id,
        }
    notification_error = persist_terminal_and_queue_notification(
        terminal_path,
        terminal,
        study_dir.parent,
        study_id=spec.id,
        run_id=run_id,
    )
    atomic_write_json(
        run_dir / HEARTBEAT_FILENAME,
        {
            "status": terminal["status"],
            "finished_at": terminal["finished_at"],
            "attempt": run_state.attempt,
            "terminal_event_id": terminal_event_id,
            "notification_error": notification_error,
        },
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
            run.attempt = int(terminal.get("attempt", run.attempt))
            run.terminal_event_id = terminal.get("terminal_event_id", run.terminal_event_id)
            run.originating_thread_id = terminal.get("originating_thread_id", run.originating_thread_id)
            run.notification_state = "pending" if run.terminal_event_id else run.notification_state
            metadata_path = Path(run.run_dir) / "metadata.json"
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text())
                run.wandb_run_id = metadata.get("wandb_run_id", run.wandb_run_id)
                run.wandb_url = metadata.get("wandb_url", run.wandb_url)
            run.decision = "retryable" if run.status in ("failed", "timed_out") else run.decision
            changed = True
        notification_path = Path(run.run_dir) / NOTIFICATION_FILENAME
        if notification_path.is_file():
            from .codex_notifications import NotificationStateError, read_notification_event

            try:
                event = read_notification_event(notification_path, Path(run.run_dir).resolve().parents[2])
            except (OSError, NotificationStateError) as error:
                notification_error = f"{type(error).__name__}: {error}"
                if run.notification_state != "failed" or run.notification_last_error != notification_error:
                    run.notification_state = "failed"
                    run.notification_last_error = notification_error
                    changed = True
            else:
                notification_values = {
                    "terminal_event_id": event.event_id,
                    "notification_state": event.state,
                    "notification_attempts": event.attempt_count,
                    "notification_last_error": event.last_error,
                    "notification_next_attempt_at": (
                        event.next_attempt_at.isoformat() if event.next_attempt_at is not None else None
                    ),
                    "notification_accepted_at": (
                        event.accepted_at.isoformat() if event.accepted_at is not None else None
                    ),
                    "notification_accepted_rpc_method": event.accepted_rpc_method,
                    "notification_accepted_turn_id": event.accepted_turn_id,
                }
                for field_name, value in notification_values.items():
                    if getattr(run, field_name) != value:
                        setattr(run, field_name, value)
                        changed = True
        heartbeat_path = Path(run.run_dir) / HEARTBEAT_FILENAME
        if heartbeat_path.is_file():
            heartbeat = json.loads(heartbeat_path.read_text())
            run.heartbeat_at = heartbeat.get("heartbeat_at", run.heartbeat_at)
            if heartbeat.get("status") == "running" and run.status == "launching":
                run.status = "running"
                changed = True
        if run.status in ACTIVE_STATUSES and run.pid is not None and not _pid_is_alive(run.pid):
            run.status = "failed"
            run.decision = "retryable"
            run.exit_code = 1
            run.finished_at = utc_now()
            run.error = "supervisor exited without writing terminal state"
            run.notification_state = "not-requested"
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


def _pending_runs_in_study_order(state: StudyState, spec: StudySpec) -> tuple[RunState, ...]:
    """Return pending runs in protocol order, independent of JSON object-key order."""
    variant_order = {variant.id: index for index, variant in enumerate((spec.baseline, *spec.variants))}
    shot_order = {shots: index for index, shots in enumerate(spec.evaluation.shots_per_class)}
    unknown_variant = len(variant_order)
    unknown_shots = len(shot_order)

    def schedule_key(run: RunState) -> tuple[int, int, int, int, str]:
        variant_index = variant_order.get(run.spec.variant, unknown_variant)
        if run.spec.kind == "pretrain":
            return (0, run.spec.seed, 0, variant_index, run.spec.id)
        return (
            1,
            shot_order.get(run.spec.shots_per_class, unknown_shots),
            run.spec.seed,
            variant_index,
            run.spec.id,
        )

    pending = (run for run in state.runs.values() if run.status == "pending")
    return tuple(sorted(pending, key=schedule_key))


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
    from .codex_notifications import initialize_notification_root

    initialize_notification_root(study_dir.parent)
    with StateStore(study_dir) as store:
        state = store.load_or_create(spec, spec_path)
        reconcile_state(state)
        active = [run for run in state.runs.values() if run.status in ACTIVE_STATUSES]
        capacity = max(0, spec.resources.max_concurrent_jobs - len(active))
        reserved = [run.physical_gpu for run in active if run.physical_gpu is not None]
        available = available_physical_gpus(spec.resources.physical_gpus, reserved=reserved, lock_root=lock_root)
        for run, physical_gpu in zip(
            _pending_runs_in_study_order(state, spec),
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
            run.originating_thread_id = run.originating_thread_id or os.environ.get("CODEX_THREAD_ID")
            run.notification_state = "not-requested"
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
            launch_time = datetime.now(UTC)
            run.next_check_at = (launch_time + timedelta(seconds=MINIMUM_POLL_INTERVAL_SECONDS)).isoformat()
            run.last_check_interval_seconds = float(MINIMUM_POLL_INTERVAL_SECONDS)
            run.next_check_reason = "startup-check"
            store.save(state)
        store.save(state)
        return state


def prepare_retryable_runs(state: StudyState) -> int:
    retry_count = 0
    for run in state.runs.values():
        if run.status not in ("failed", "timed_out") or run.decision != "retryable":
            continue
        if run.run_dir is not None:
            run_dir = Path(run.run_dir)
            attempts_dir = run_dir / "attempts"
            for artifact_name in (TERMINAL_FILENAME, HEARTBEAT_FILENAME, NOTIFICATION_FILENAME):
                artifact = run_dir / artifact_name
                if artifact.is_file():
                    attempts_dir.mkdir(parents=True, exist_ok=True)
                    archived_path = attempts_dir / f"{artifact.stem}-{uuid.uuid4().hex}{artifact.suffix}"
                    os.replace(artifact, archived_path)
        run.status = "pending"
        run.decision = "pending"
        run.physical_gpu = None
        run.pid = None
        run.started_at = None
        run.finished_at = None
        run.exit_code = None
        run.error = None
        run.attempt += 1
        run.terminal_event_id = None
        run.notification_attempts = 0
        run.notification_last_error = None
        run.notification_next_attempt_at = None
        run.notification_accepted_at = None
        run.notification_accepted_rpc_method = None
        run.notification_accepted_turn_id = None
        run.notification_state = "not-requested"
        retry_count += 1
    return retry_count


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
    planned_targets = [(target, target.stat().st_size) for target in targets if target.is_file()]
    bytes_planned = sum(size for _, size in planned_targets)
    operation_id = hashlib.sha256(
        f"{state.study_id}:{run_id}:{run.attempt}:{study_close}:{bytes_planned}".encode()
    ).hexdigest()[:32]
    retention_path = study_dir / RETENTION_LOG_FILENAME
    append_locked_jsonl(
        retention_path,
        {
            "timestamp": utc_now(),
            "study_id": state.study_id,
            "run_id": run_id,
            "attempt": run.attempt,
            "operation_id": operation_id,
            "phase": "planned",
            "paths": [str(path) for path, _ in planned_targets],
            "bytes_planned": bytes_planned,
            "recoverable": False,
        },
        f"{operation_id}:planned",
    )
    deleted: list[Path] = []
    bytes_freed = 0
    for target, planned_size in planned_targets:
        if target.is_file():
            target.unlink()
            deleted.append(target)
            bytes_freed += planned_size
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
        "attempt": run.attempt,
        "operation_id": operation_id,
        "phase": "deleted",
        "paths": [str(path) for path in deleted],
        "bytes_freed": bytes_freed,
        "recoverable": False,
    }
    append_locked_jsonl(retention_path, retention_record, f"{operation_id}:deleted")
    return tuple(deleted)
