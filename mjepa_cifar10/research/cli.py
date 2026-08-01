from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO
from uuid import uuid4

from notify_wake import discover_daemon_socket

from .codex_notifications import (
    UnixWebSocketTransport,
    capture_wake_context,
    ensure_notification,
    enter_research_notify_wait,
    initialize_notification_root,
    next_notification_attempt_at,
    register_notification_root,
    sweep_notifications,
    unix_connector,
    validate_notification_root,
)
from .event_controller import DEFAULT_PROGRESS_TIMEOUT_SECONDS, serve_controller
from .event_controller import LOGGER as EVENT_CONTROLLER_LOGGER
from .inventory import index_local_runs, index_wandb_runs, inventory_counts, open_inventory
from .models import WANDB_LOCAL_MODES, StudySpec
from .provenance import assert_launch_provenance, collect_provenance
from .runtime import (
    StateStore,
    launch_available_runs,
    prepare_retryable_runs,
    reconcile_state,
    run_worker,
    schedule_due_monitor_checks,
    storage_report,
    study_directory,
    validate_managed_paths,
)
from .summary import append_research_log, apply_rejected_retention, summarize_study
from .wake_context import (
    CODEX_PERMISSION_PROFILE_ENVIRONMENT_VARIABLE,
    CODEX_THREAD_ENVIRONMENT_VARIABLE,
    WakeContext,
)


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("study", type=Path, help="Path to a committed research study YAML")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


def _add_event_controller_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("logs/research"))
    parser.add_argument(
        "--progress-timeout-seconds",
        type=int,
        default=DEFAULT_PROGRESS_TIMEOUT_SECONDS,
    )
    parser.add_argument("--socket", type=Path, default=None)
    parser.add_argument(
        "--study-id",
        dest="study_ids",
        action="append",
        help="Deliver only notifications for this study; repeat to select more than one",
    )
    parser.add_argument("--defer-until-socket-replaced", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent JEPA auto-research harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight", help="Validate data, GPUs, storage, code, and environment")
    _add_common_arguments(preflight_parser)
    preflight_parser.add_argument(
        "--development",
        action="store_true",
        help="Report provenance problems without failing",
    )

    launch_parser = subparsers.add_parser("launch", help="Launch pending managed runs on physical GPUs 1 and 2")
    _add_common_arguments(launch_parser)
    launch_parser.add_argument("--dry-run", action="store_true", help="Create recoverable state without starting jobs")
    launch_parser.add_argument("--retry-failed", action="store_true", help="Retry terminal runs marked retryable")

    status_parser = subparsers.add_parser("status", help="Read persistent study/run state")
    _add_common_arguments(status_parser)

    monitor_parser = subparsers.add_parser("monitor", help="Recover terminal state and launch the next bounded jobs")
    _add_common_arguments(monitor_parser)
    monitor_parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Read-only inspection; do not reconcile state or launch workers",
    )

    notify_parser = subparsers.add_parser("notify", help="Validate, recover, or explicitly requeue one notification")
    _add_common_arguments(notify_parser)
    notify_parser.add_argument("run_id", help="Managed run to notify about")
    notify_parser.add_argument("--requeue", action="store_true", help="Explicitly requeue a failed notification")

    notify_worker_parser = subparsers.add_parser("notify-worker", help="Deliver all due Codex notifications once")
    notify_worker_parser.add_argument("--once", action="store_true", required=True)
    notify_worker_parser.add_argument("--root", type=Path, default=Path("logs/research"))
    notify_worker_parser.add_argument("--socket", type=Path, default=None)
    notify_worker_parser.add_argument(
        "--study-id",
        dest="study_ids",
        action="append",
        help="Deliver only notifications for this study; repeat to select more than one",
    )

    event_controller_parser = subparsers.add_parser(
        "event-controller",
        help="Watch durable lifecycle events without model polling",
    )
    _add_event_controller_arguments(event_controller_parser)

    start_controller_parser = subparsers.add_parser(
        "start-controller",
        help="Start a detached, directly identifiable research event controller",
    )
    _add_event_controller_arguments(start_controller_parser)

    notify_wait_parser = subparsers.add_parser(
        "notify-wait",
        help="Bind the active goal wait to one verified research event controller",
    )
    notify_wait_parser.add_argument("--root", type=Path, default=Path("logs/research"))
    notify_wait_parser.add_argument("--socket", type=Path, default=None)
    notify_wait_parser.add_argument("--controller-pid", type=int, required=True)
    notify_wait_parser.add_argument("--controller-start-ticks", type=int, required=True)
    notify_wait_parser.add_argument(
        "--study-id",
        dest="study_ids",
        action="append",
        required=True,
        help="Study owned by the controller; repeat to select more than one",
    )

    register_root_parser = subparsers.add_parser(
        "register-root", help="Register one exact root for notification discovery"
    )
    register_root_parser.add_argument("--root", type=Path, required=True)

    summarize_parser = subparsers.add_parser(
        "summarize",
        help="Calculate convergence, promotion, and replication results",
    )
    _add_common_arguments(summarize_parser)
    summarize_parser.add_argument("--record", action="store_true", help="Append a result entry to research/LOG.md")
    summarize_parser.add_argument(
        "--apply-retention",
        action="store_true",
        help="Delete eligible rejected managed checkpoints after clean pushed result recording",
    )
    summarize_parser.add_argument("--study-close", action="store_true", help="Also apply end-of-study retention")

    inventory_parser = subparsers.add_parser("inventory", help="Index historical local and optional W&B runs")
    inventory_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    inventory_parser.add_argument("--database", type=Path, default=Path("logs/research/inventory.sqlite3"))
    inventory_parser.add_argument("--wandb-entity", type=str, default=None)
    inventory_parser.add_argument("--wandb-project", type=str, default="mjepa-cifar10")

    storage_parser = subparsers.add_parser(
        "storage-report",
        help="Report checkpoint storage estimate and safety margin",
    )
    _add_common_arguments(storage_parser)

    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("study", type=Path)
    worker_parser.add_argument("run_id")
    worker_parser.add_argument("physical_gpu", type=int)
    worker_parser.add_argument("--repo-root", type=Path, required=True)
    worker_parser.add_argument("--lock-root", type=Path, required=True)
    return parser


UNRESOLVED_ENVIRONMENT_VARIABLE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
EVENT_CONTROLLER_LOG_DIRECTORY = ".event-controller"
PROC_START_TICKS_SUFFIX_INDEX = 19
CONTROLLER_SWEEP_TIMEOUT_SECONDS = 60.0
CONTROLLER_SWEEP_RETRY_SECONDS = 5.0
CONTROLLER_STARTUP_TIMEOUT_SECONDS = 10.0
MAX_CONTROLLER_ERROR_LENGTH = 500


def _event_controller_log_path(
    root: Path,
    study_ids: frozenset[str] | None,
    *,
    pid: int | None = None,
) -> Path:
    managed_root = validate_notification_root(root)
    scope = _event_controller_scope(study_ids)
    log_directory = managed_root / EVENT_CONTROLLER_LOG_DIRECTORY
    log_directory.mkdir(exist_ok=True)
    if log_directory.is_symlink() or log_directory.resolve() != log_directory:
        raise ValueError(f"event controller log directory must not be a symlink: {log_directory}")
    log_path = log_directory / f"{scope}-{pid or os.getpid()}.jsonl"
    if log_path.is_symlink():
        raise ValueError(f"event controller log must not be a symlink: {log_path}")
    return log_path


def _event_controller_scope(study_ids: frozenset[str] | None) -> str:
    return "all" if study_ids is None else hashlib.sha256("\0".join(sorted(study_ids)).encode()).hexdigest()[:16]


def _append_controller_record(stream: TextIO, event: str, **payload: Any) -> None:
    record = {"event": event, "recorded_at": datetime.now(UTC).isoformat(), **payload}
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()


def _bounded_controller_error(error: BaseException) -> str:
    message = " ".join(str(error).split()) or type(error).__name__
    return f"{type(error).__name__}: {message}"[:MAX_CONTROLLER_ERROR_LENGTH]


def _parse_proc_start_ticks(stat: str) -> int:
    """Extract Linux field 22 while allowing spaces and parentheses in comm."""
    closing_parenthesis = stat.rfind(")")
    if closing_parenthesis < 0:
        raise ValueError("process stat has no closing command parenthesis")
    suffix = stat[closing_parenthesis + 1 :].split()
    if len(suffix) <= PROC_START_TICKS_SUFFIX_INDEX:
        raise ValueError("process stat is missing start ticks")
    start_ticks = int(suffix[PROC_START_TICKS_SUFFIX_INDEX])
    if start_ticks <= 0:
        raise ValueError("process start ticks must be positive")
    return start_ticks


def _process_start_ticks(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    if pid <= 0:
        raise ValueError("controller PID must be positive")
    return _parse_proc_start_ticks((proc_root / str(pid) / "stat").read_text(encoding="utf-8"))


def _command_flag_values(command: tuple[str, ...], flag: str) -> tuple[str, ...]:
    values: list[str] = []
    for index, argument in enumerate(command):
        if argument == flag:
            if index + 1 >= len(command):
                return ()
            values.append(command[index + 1])
        elif argument.startswith(f"{flag}="):
            values.append(argument.removeprefix(f"{flag}="))
    return tuple(values)


def _event_controller_identity_matches(
    root: Path,
    *,
    pid: int,
    start_ticks: int,
    study_ids: frozenset[str],
    proc_root: Path = Path("/proc"),
) -> bool:
    """Verify the live direct controller and its durable startup record."""
    try:
        managed_root = validate_notification_root(root)
        if _process_start_ticks(pid, proc_root=proc_root) != start_ticks:
            return False
        command = tuple(
            part.decode("utf-8") for part in (proc_root / str(pid) / "cmdline").read_bytes().split(b"\0") if part
        )
        if len(command) < 3 or not Path(command[0]).name.startswith("python"):
            return False
        expected_script = Path(__file__).parents[2] / "scripts" / "research.py"
        if Path(command[1]).expanduser().resolve(strict=False) != expected_script.resolve():
            return False
        if command[2] != "event-controller":
            return False
        roots = _command_flag_values(command, "--root")
        if len(roots) != 1 or Path(roots[0]).expanduser().resolve(strict=False) != managed_root:
            return False
        if frozenset(_command_flag_values(command, "--study-id")) != study_ids:
            return False

        log_path = _event_controller_log_path(managed_root, study_ids, pid=pid)
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
        started = [record for record in records if record.get("event") == "controller_started"][-1]
        return (
            started.get("pid") == pid
            and started.get("start_ticks") == start_ticks
            and started.get("root") == str(managed_root)
            and started.get("study_ids") == sorted(study_ids)
        )
    except (IndexError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
        return False


def _find_active_event_controller(
    root: Path,
    study_ids: frozenset[str],
) -> tuple[int, int, Path] | None:
    log_directory = root / EVENT_CONTROLLER_LOG_DIRECTORY
    scope = _event_controller_scope(study_ids)
    for log_path in sorted(log_directory.glob(f"{scope}-*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True):
        try:
            pid = int(log_path.stem.rsplit("-", 1)[1])
            records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
            started = [record for record in records if record.get("event") == "controller_started"][-1]
            start_ticks = int(started["start_ticks"])
        except (IndexError, KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
        if _event_controller_identity_matches(
            root,
            pid=pid,
            start_ticks=start_ticks,
            study_ids=study_ids,
        ):
            return pid, start_ticks, log_path
    return None


def _terminate_controller_process_group(pid: int, start_ticks: int) -> None:
    try:
        if _process_start_ticks(pid) == start_ticks:
            os.killpg(pid, signal.SIGTERM)
    except (OSError, ValueError):
        return


class _ControllerJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "event": "controller_log",
                "level": record.levelname,
                "message": record.getMessage(),
                "recorded_at": datetime.now(UTC).isoformat(),
            },
            sort_keys=True,
        )


def _load_spec(args: argparse.Namespace, *, require_data: bool = True) -> tuple[StudySpec, Path]:
    repo_root = args.repo_root.resolve()
    spec = StudySpec.from_path(args.study.resolve())
    spec.validate(repo_root)
    validate_managed_paths(spec, repo_root)
    if require_data:
        unresolved = sorted(
            {
                match.group("braced") or match.group("plain")
                for match in UNRESOLVED_ENVIRONMENT_VARIABLE.finditer(str(spec.data))
            }
        )
        if unresolved:
            raise EnvironmentError(
                "study data path references unset environment "
                f"{'variable' if len(unresolved) == 1 else 'variables'}: {', '.join(unresolved)}"
            )
        data_path = spec.data if spec.data.is_absolute() else repo_root / spec.data
        if not data_path.is_dir():
            raise NotADirectoryError(data_path)
    return spec, repo_root


def _gpu_inventory() -> list[dict[str, str]]:
    result = subprocess.run(
        ("nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"),
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "nvidia-smi failed")
    inventory = []
    for line in result.stdout.splitlines():
        index, name, memory_mib = (part.strip() for part in line.split(",", maxsplit=2))
        inventory.append({"index": index, "name": name, "memory_mib": memory_mib})
    return inventory


def _wandb_viewer() -> dict[str, Any]:
    import wandb

    viewer = wandb.Api().viewer
    return {
        "username": getattr(viewer, "username", None),
        "email": getattr(viewer, "email", None),
        "entity": getattr(viewer, "entity", None),
    }


def wandb_preflight_errors(spec: StudySpec, environment: Mapping[str, str]) -> list[str]:
    mode = environment.get("WANDB_MODE", "online").strip().lower()
    if mode in WANDB_LOCAL_MODES:
        return ["authorized W&B study requires online tracking"] if spec.wandb_authorized else []
    decision = spec.wandb_operation_decision("launch", mode)
    errors: list[str] = []
    if not spec.wandb_entity:
        errors.append("online W&B requires an explicit destination entity")
    if not spec.wandb_authorized:
        errors.append("external tracker is configured without explicit study authorization")
    if not spec.wandb_manifests_explicit:
        errors.append("online W&B requires explicit emitted-data manifests in the study specification")
    if decision.missing_data_classes:
        errors.append(
            f"external tracker authorization is missing emitted data classes: {list(decision.missing_data_classes)}"
        )
    return errors


def preflight_payload(spec: StudySpec, repo_root: Path, *, development: bool) -> dict[str, Any]:
    storage = storage_report(spec, repo_root)
    provenance = collect_provenance(spec, repo_root)
    gpu_inventory = _gpu_inventory()
    wandb_errors = wandb_preflight_errors(spec, os.environ)
    wandb_mode = os.environ.get("WANDB_MODE", "online").strip().lower()
    wandb_decision = spec.wandb_operation_decision("launch", wandb_mode)
    if wandb_mode in WANDB_LOCAL_MODES:
        wandb_viewer = {"mode": wandb_mode}
    else:
        try:
            wandb_viewer = _wandb_viewer()
        except Exception as error:
            wandb_viewer = {"error": f"{type(error).__name__}: {error}"}
    present_gpu_indices = {int(gpu["index"]) for gpu in gpu_inventory}
    missing_gpus = set(spec.resources.physical_gpus) - present_gpu_indices
    errors = list(provenance.errors)
    if missing_gpus:
        errors.append(f"configured physical GPUs are unavailable: {sorted(missing_gpus)}")
    if not storage["sufficient"]:
        errors.append("checkpoint storage margin is insufficient")
    if "error" in wandb_viewer:
        errors.append(f"W&B authentication check failed: {wandb_viewer['error']}")
    errors.extend(wandb_errors)
    payload = {
        "study_id": spec.id,
        "ok": not errors,
        "development": development,
        "errors": errors,
        "storage": storage,
        "gpus": gpu_inventory,
        "wandb": {"viewer": wandb_viewer, "launch_gate": wandb_decision.to_dict()},
        "provenance": provenance.to_dict(),
    }
    if errors and not development:
        raise RuntimeError("preflight failed:\n- " + "\n- ".join(errors))
    return payload


def command_preflight(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args, require_data=True)
    print(json.dumps(preflight_payload(spec, repo_root, development=args.development), indent=2))
    return 0


def command_launch(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args, require_data=not args.dry_run)
    if args.dry_run:
        study_dir = study_directory(spec, repo_root)
        initialize_notification_root(study_dir.parent)
        with StateStore(study_dir) as store:
            state = store.load_or_create(spec, args.study)
        payload = {"study_id": spec.id, "dry_run": True, "state": str(store.path), "runs": list(state.runs)}
    else:
        preflight_payload(spec, repo_root, development=False)
        wake_context = capture_launch_wake_context()
        if args.retry_failed:
            with StateStore(study_directory(spec, repo_root)) as store:
                retry_state = store.load()
                prepare_retryable_runs(retry_state)
                store.save(retry_state)
        state = launch_available_runs(
            spec,
            args.study,
            repo_root,
            wake_context=wake_context,
        )
        payload = _state_payload(state)
    print(json.dumps(payload, indent=2))
    return 0


def _state_payload(state) -> dict[str, Any]:
    return {
        "study_id": state.study_id,
        "phase": state.phase,
        "winner": state.winner,
        "runs": {
            run_id: {
                "status": run.status,
                "decision": run.decision,
                "physical_gpu": run.physical_gpu,
                "pid": run.pid,
                "run_dir": run.run_dir,
                "wandb_run_id": run.wandb_run_id,
                "error": run.error,
                "attempt": run.attempt,
                "heartbeat_at": run.heartbeat_at,
                "current_progress": run.current_progress,
                "next_check_at": run.next_check_at,
                "next_check_reason": run.next_check_reason,
                "notification_state": run.notification_state,
            }
            for run_id, run in state.runs.items()
        },
    }


def command_status(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args, require_data=False)
    with StateStore(study_directory(spec, repo_root)) as store:
        state = store.load()
    print(json.dumps(_state_payload(state), indent=2))
    return 0


def command_monitor(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args, require_data=not args.no_launch)
    study_dir = study_directory(spec, repo_root)
    if args.no_launch:
        with StateStore(study_dir) as store:
            state = store.load()
        print(json.dumps(_state_payload(state), indent=2))
        return 0
    with StateStore(study_dir) as store:
        state = store.load()
        changed = reconcile_state(state)
        changed = schedule_due_monitor_checks(state.runs.values()) or changed
        if changed:
            store.save(state)
    if not args.no_launch:
        preflight_payload(spec, repo_root, development=False)
        state = launch_available_runs(
            spec,
            args.study,
            repo_root,
            wake_context=capture_launch_wake_context(),
        )
    print(json.dumps(_state_payload(state), indent=2))
    return 0


def command_notify(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args, require_data=False)
    with StateStore(study_directory(spec, repo_root)) as store:
        state = store.load()
        run = state.runs[args.run_id]
        if run.run_dir is None:
            raise ValueError(f"run {args.run_id} has no managed run directory")
        terminal_path = Path(run.run_dir) / "terminal.json"
        if not terminal_path.is_file():
            raise ValueError(f"run {args.run_id} has no terminal state")
        managed_root = study_directory(spec, repo_root).parent
        initialize_notification_root(managed_root)
        event = ensure_notification(terminal_path, managed_root, requeue=args.requeue)
        run.terminal_event_id = event.event_id
        run.notification_state = event.state
        run.notification_attempts = event.attempt_count
        run.notification_last_error = event.last_error
        run.notification_next_attempt_at = (
            event.next_attempt_at.isoformat() if event.next_attempt_at is not None else None
        )
        run.notification_accepted_at = event.accepted_at.isoformat() if event.accepted_at is not None else None
        run.notification_accepted_rpc_method = event.accepted_rpc_method
        run.notification_accepted_turn_id = event.accepted_turn_id
        store.save(state)
    print(json.dumps(event.to_dict(), indent=2, sort_keys=True))
    return 1 if event.state == "failed" else 0


def command_notify_worker(args: argparse.Namespace) -> int:
    socket_path = resolve_event_controller_socket(args.socket)
    connector = unix_connector(socket_path)
    study_ids = frozenset(args.study_ids) if args.study_ids else None
    result = asyncio.run(sweep_notifications(args.root, connect=connector, study_ids=study_ids))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return result.exit_code


def resolve_event_controller_socket(explicit_socket: Path | None) -> Path:
    """Resolve the control socket used for transport readiness events."""
    if explicit_socket is not None:
        return explicit_socket.expanduser().resolve(strict=False)
    return discover_daemon_socket()


def capture_launch_wake_context() -> WakeContext:
    """Capture the effective originating-thread permissions for new runs."""
    thread_id = os.environ.get(CODEX_THREAD_ENVIRONMENT_VARIABLE)
    requested_permission_profile = os.environ.get(CODEX_PERMISSION_PROFILE_ENVIRONMENT_VARIABLE)
    if not thread_id:
        raise RuntimeError(f"{CODEX_THREAD_ENVIRONMENT_VARIABLE} is required for a managed launch")
    if requested_permission_profile == "":
        raise RuntimeError(f"{CODEX_PERMISSION_PROFILE_ENVIRONMENT_VARIABLE} must be non-empty when set")
    socket_path = resolve_event_controller_socket(None)

    async def capture() -> WakeContext:
        transport = await UnixWebSocketTransport.connect(socket_path)
        return await capture_wake_context(
            thread_id=thread_id,
            requested_permission_profile=requested_permission_profile,
            transport=transport,
        )

    return asyncio.run(capture())


def command_event_controller(args: argparse.Namespace) -> int:
    if args.progress_timeout_seconds <= 0:
        raise ValueError("--progress-timeout-seconds must be positive")
    socket_path = resolve_event_controller_socket(args.socket)
    connector = unix_connector(socket_path)
    study_ids = frozenset(args.study_ids) if args.study_ids else None
    log_path = _event_controller_log_path(args.root, study_ids)
    controller_pid = os.getpid()
    controller_start_ticks = _process_start_ticks(controller_pid)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(_ControllerJsonFormatter())
    EVENT_CONTROLLER_LOGGER.addHandler(file_handler)
    try:
        with log_path.open("a", encoding="utf-8") as log_stream:
            _append_controller_record(
                log_stream,
                "controller_started",
                pid=controller_pid,
                start_ticks=controller_start_ticks,
                root=str(args.root.expanduser().resolve(strict=False)),
                socket_path=str(socket_path),
                study_ids=sorted(study_ids) if study_ids is not None else None,
            )
            controller_retry_at: datetime | None = None

            def deliver_once() -> None:
                nonlocal controller_retry_at

                async def bounded_sweep():
                    return await asyncio.wait_for(
                        sweep_notifications(args.root, connect=connector, study_ids=study_ids),
                        timeout=CONTROLLER_SWEEP_TIMEOUT_SECONDS,
                    )

                try:
                    result = asyncio.run(bounded_sweep())
                except Exception as error:
                    controller_retry_at = datetime.now(UTC) + timedelta(seconds=CONTROLLER_SWEEP_RETRY_SECONDS)
                    _append_controller_record(
                        log_stream,
                        "notification_sweep_failed",
                        error=_bounded_controller_error(error),
                        retry_at=controller_retry_at.isoformat(),
                    )
                    return
                controller_retry_at = None
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True), flush=True)
                _append_controller_record(log_stream, "notification_sweep", **result.to_dict())

            def next_delivery_deadline() -> datetime | None:
                persisted = next_notification_attempt_at(args.root, study_ids=study_ids)
                candidates = tuple(value for value in (persisted, controller_retry_at) if value is not None)
                return min(candidates) if candidates else None

            try:
                serve_controller(
                    args.root,
                    progress_timeout=timedelta(seconds=args.progress_timeout_seconds),
                    deliver=deliver_once,
                    next_delivery_at=next_delivery_deadline,
                    socket_path=socket_path,
                    defer_until_socket_replaced=args.defer_until_socket_replaced,
                    study_ids=study_ids,
                )
            except BaseException as error:
                _append_controller_record(
                    log_stream,
                    "controller_failed",
                    error=f"{type(error).__name__}: {error}",
                )
                raise
            else:
                _append_controller_record(log_stream, "controller_stopped")
    finally:
        EVENT_CONTROLLER_LOGGER.removeHandler(file_handler)
        file_handler.close()
    return 0


def command_start_controller(args: argparse.Namespace) -> int:
    if args.progress_timeout_seconds <= 0:
        raise ValueError("--progress-timeout-seconds must be positive")
    if not args.study_ids:
        raise ValueError("start-controller requires at least one --study-id")
    if len(args.study_ids) != len(set(args.study_ids)):
        raise ValueError("--study-id values must be unique")
    managed_root = validate_notification_root(args.root)
    study_ids = frozenset(args.study_ids)
    if active := _find_active_event_controller(managed_root, study_ids):
        pid, start_ticks, controller_log = active
        print(
            json.dumps(
                {
                    "pid": pid,
                    "start_ticks": start_ticks,
                    "root": str(managed_root),
                    "study_ids": sorted(study_ids),
                    "controller_log": str(controller_log),
                    "reused": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    script = Path(__file__).parents[2] / "scripts" / "research.py"
    command = [
        sys.executable,
        str(script),
        "event-controller",
        "--root",
        str(managed_root),
        "--progress-timeout-seconds",
        str(args.progress_timeout_seconds),
    ]
    if args.socket is not None:
        command.extend(("--socket", str(args.socket.expanduser().resolve(strict=False))))
    for study_id in sorted(study_ids):
        command.extend(("--study-id", study_id))
    if args.defer_until_socket_replaced:
        command.append("--defer-until-socket-replaced")

    log_directory = _event_controller_log_path(managed_root, study_ids, pid=os.getpid()).parent
    process_log_path = log_directory / f"launcher-{uuid4()}.log"
    with process_log_path.open("ab", buffering=0) as process_log:
        process = subprocess.Popen(
            command,
            cwd=Path(__file__).parents[2],
            stdin=subprocess.DEVNULL,
            stdout=process_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    try:
        start_ticks = _process_start_ticks(process.pid)
    except (OSError, ValueError) as error:
        raise RuntimeError(f"event controller exited before identity capture; see {process_log_path}") from error

    deadline = time.monotonic() + CONTROLLER_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _event_controller_identity_matches(
            managed_root,
            pid=process.pid,
            start_ticks=start_ticks,
            study_ids=study_ids,
        ):
            controller_log = _event_controller_log_path(managed_root, study_ids, pid=process.pid)
            print(
                json.dumps(
                    {
                        "pid": process.pid,
                        "start_ticks": start_ticks,
                        "root": str(managed_root),
                        "study_ids": sorted(study_ids),
                        "controller_log": str(controller_log),
                        "process_log": str(process_log_path),
                        "reused": False,
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if process.poll() is not None:
            raise RuntimeError(f"event controller failed during startup; see {process_log_path}")
        time.sleep(0.01)
    _terminate_controller_process_group(process.pid, start_ticks)
    raise RuntimeError(f"event controller did not become ready; see {process_log_path}")


def command_notify_wait(args: argparse.Namespace) -> int:
    if args.controller_pid <= 0:
        raise ValueError("--controller-pid must be positive")
    if args.controller_start_ticks <= 0:
        raise ValueError("--controller-start-ticks must be positive")
    if len(args.study_ids) != len(set(args.study_ids)):
        raise ValueError("--study-id values must be unique")
    study_ids = frozenset(args.study_ids)
    managed_root = validate_notification_root(args.root)
    loop_id = f"research-event-controller:{args.controller_pid}:{args.controller_start_ticks}"
    source_ids = tuple(f"study:{study_id}" for study_id in sorted(study_ids))

    def verify_loop_identity(candidate_loop_id: str, candidate_source_ids: tuple[str, ...]) -> bool:
        return (
            candidate_loop_id == loop_id
            and candidate_source_ids == source_ids
            and _event_controller_identity_matches(
                managed_root,
                pid=args.controller_pid,
                start_ticks=args.controller_start_ticks,
                study_ids=study_ids,
            )
        )

    if not verify_loop_identity(loop_id, source_ids):
        raise RuntimeError("research event controller identity or durable startup record does not match")
    context = capture_launch_wake_context()
    socket_path = resolve_event_controller_socket(args.socket)

    async def enter_wait():
        transport = await UnixWebSocketTransport.connect(socket_path)
        try:
            return await enter_research_notify_wait(
                managed_root,
                context=context,
                loop_id=loop_id,
                source_ids=source_ids,
                transport=transport,
                verify_loop_identity=verify_loop_identity,
            )
        finally:
            await transport.close()

    lease = asyncio.run(enter_wait())
    print(json.dumps(lease.to_dict(), indent=2, sort_keys=True))
    return 0


def command_register_root(args: argparse.Namespace) -> int:
    registration = register_notification_root(args.root)
    print(json.dumps(registration.to_dict(), indent=2, sort_keys=True))
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args, require_data=False)
    summary = summarize_study(spec, args.study, repo_root)
    if args.record:
        summary["research_log_appended"] = append_research_log(spec, summary, repo_root)
    if args.apply_retention:
        assert_launch_provenance(spec, repo_root)
        summary["deleted_weights"] = apply_rejected_retention(spec, repo_root, study_close=args.study_close)
    print(json.dumps(summary, indent=2))
    return 0


def command_inventory(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    database = args.database if args.database.is_absolute() else repo_root / args.database
    with closing(open_inventory(database)) as connection:
        local_count = index_local_runs(repo_root, connection)
        remote_count = 0
        if args.wandb_entity:
            remote_count = index_wandb_runs(args.wandb_entity, args.wandb_project, connection)
        counts = inventory_counts(connection)
    print(
        json.dumps(
            {"database": str(database), "local_indexed": local_count, "remote_indexed": remote_count, "counts": counts},
            indent=2,
        )
    )
    return 0


def command_storage_report(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args, require_data=False)
    print(json.dumps(storage_report(spec, repo_root), indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "preflight": command_preflight,
        "launch": command_launch,
        "status": command_status,
        "monitor": command_monitor,
        "notify": command_notify,
        "notify-worker": command_notify_worker,
        "event-controller": command_event_controller,
        "start-controller": command_start_controller,
        "notify-wait": command_notify_wait,
        "register-root": command_register_root,
        "summarize": command_summarize,
        "inventory": command_inventory,
        "storage-report": command_storage_report,
    }
    if args.command == "_worker":
        return run_worker(args.study, args.run_id, args.physical_gpu, args.repo_root, args.lock_root)
    return commands[args.command](args)
