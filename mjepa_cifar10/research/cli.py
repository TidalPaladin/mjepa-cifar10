from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any, Mapping, Sequence

from .codex_notifications import (
    ensure_notification,
    initialize_notification_root,
    register_notification_root,
    stdio_connector,
    sweep_notifications,
    unix_connector,
)
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


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("study", type=Path, help="Path to a committed research study YAML")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())


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
    notify_worker_parser.add_argument("--transport", choices=("stdio", "unix"), default="stdio")
    notify_worker_parser.add_argument("--socket", type=Path, default=None)

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


def _load_spec(args: argparse.Namespace) -> tuple[StudySpec, Path]:
    repo_root = args.repo_root.resolve()
    spec = StudySpec.from_path(args.study.resolve())
    spec.validate(repo_root)
    validate_managed_paths(spec, repo_root)
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
        return []
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
    spec, repo_root = _load_spec(args)
    print(json.dumps(preflight_payload(spec, repo_root, development=args.development), indent=2))
    return 0


def command_launch(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args)
    if args.dry_run:
        study_dir = study_directory(spec, repo_root)
        initialize_notification_root(study_dir.parent)
        with StateStore(study_dir) as store:
            state = store.load_or_create(spec, args.study)
        payload = {"study_id": spec.id, "dry_run": True, "state": str(store.path), "runs": list(state.runs)}
    else:
        preflight_payload(spec, repo_root, development=False)
        if args.retry_failed:
            with StateStore(study_directory(spec, repo_root)) as store:
                retry_state = store.load()
                prepare_retryable_runs(retry_state)
                store.save(retry_state)
        state = launch_available_runs(spec, args.study, repo_root)
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
    spec, repo_root = _load_spec(args)
    with StateStore(study_directory(spec, repo_root)) as store:
        state = store.load()
    print(json.dumps(_state_payload(state), indent=2))
    return 0


def command_monitor(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args)
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
        state = launch_available_runs(spec, args.study, repo_root)
    print(json.dumps(_state_payload(state), indent=2))
    return 0


def command_notify(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args)
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
    if args.transport == "unix":
        if args.socket is None:
            raise ValueError("--socket is required with --transport unix")
        connector = unix_connector(args.socket)
    else:
        connector = stdio_connector(args.socket)
    result = asyncio.run(sweep_notifications(args.root, connect=connector))
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return result.exit_code


def command_register_root(args: argparse.Namespace) -> int:
    registration = register_notification_root(args.root)
    print(json.dumps(registration.to_dict(), indent=2, sort_keys=True))
    return 0


def command_summarize(args: argparse.Namespace) -> int:
    spec, repo_root = _load_spec(args)
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
    spec, repo_root = _load_spec(args)
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
        "register-root": command_register_root,
        "summarize": command_summarize,
        "inventory": command_inventory,
        "storage-report": command_storage_report,
    }
    if args.command == "_worker":
        return run_worker(args.study, args.run_id, args.physical_gpu, args.repo_root, args.lock_root)
    return commands[args.command](args)
