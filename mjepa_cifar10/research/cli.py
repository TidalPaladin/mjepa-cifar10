from __future__ import annotations

import argparse
import json
import subprocess
from contextlib import closing
from pathlib import Path
from typing import Any, Sequence

from .inventory import index_local_runs, index_wandb_runs, inventory_counts, open_inventory
from .models import StudySpec
from .provenance import assert_launch_provenance, collect_provenance
from .runtime import (
    StateStore,
    launch_available_runs,
    prepare_retryable_runs,
    reconcile_state,
    run_worker,
    storage_report,
    study_directory,
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
    monitor_parser.add_argument("--no-launch", action="store_true", help="Only reconcile completed workers")

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


def preflight_payload(spec: StudySpec, repo_root: Path, *, development: bool) -> dict[str, Any]:
    storage = storage_report(spec, repo_root)
    provenance = collect_provenance(spec, repo_root)
    gpu_inventory = _gpu_inventory()
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
    payload = {
        "study_id": spec.id,
        "ok": not errors,
        "development": development,
        "errors": errors,
        "storage": storage,
        "gpus": gpu_inventory,
        "wandb": wandb_viewer,
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
    with StateStore(study_dir) as store:
        state = store.load()
        changed = reconcile_state(state)
        if changed:
            store.save(state)
    if not args.no_launch:
        preflight_payload(spec, repo_root, development=False)
        state = launch_available_runs(spec, args.study, repo_root)
    print(json.dumps(_state_payload(state), indent=2))
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
        "summarize": command_summarize,
        "inventory": command_inventory,
        "storage-report": command_storage_report,
    }
    if args.command == "_worker":
        return run_worker(args.study, args.run_id, args.physical_gpu, args.repo_root, args.lock_root)
    return commands[args.command](args)
