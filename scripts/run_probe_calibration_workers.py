#!/usr/bin/env python3

import json
import os
import subprocess
import sys
import tempfile
import time
from argparse import ArgumentParser, Namespace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import yaml


POLL_INTERVAL_SECONDS: Final[float] = 0.2
GRACEFUL_SHUTDOWN_SECONDS: Final[float] = 10.0


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run sharded frozen-probe calibration workers")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("data", type=Path)
    return parser.parse_args()


def _load_manifest_resources(manifest_path: Path) -> tuple[str, Path, list[int]]:
    manifest = yaml.safe_load(manifest_path.read_text())
    if not isinstance(manifest, dict):
        raise TypeError("probe calibration manifest must contain a mapping")
    resources = manifest.get("resources")
    if not isinstance(resources, dict):
        raise TypeError("probe calibration manifest must define resources")
    physical_gpus = resources.get("physical_gpus")
    if (
        not isinstance(physical_gpus, list)
        or not physical_gpus
        or not all(isinstance(gpu, int) for gpu in physical_gpus)
    ):
        raise ValueError("resources.physical_gpus must be a non-empty integer list")
    if len(set(physical_gpus)) != len(physical_gpus):
        raise ValueError("resources.physical_gpus must not contain duplicates")
    workers = resources.get("workers")
    if workers != len(physical_gpus):
        raise ValueError("resources.workers must match the number of physical GPUs")

    repo_root = Path(__file__).resolve().parents[1]
    log_root = repo_root / str(manifest["log_root"])
    return str(manifest["id"]), log_root, physical_gpus


def _build_worker_commands(
    manifest_path: Path,
    data_path: Path,
    physical_gpus: list[int],
) -> list[tuple[list[str], dict[str, str]]]:
    worker_script = Path(__file__).with_name("calibrate_probes.py")
    worker_count = len(physical_gpus)
    commands: list[tuple[list[str], dict[str, str]]] = []
    for worker_index, physical_gpu in enumerate(physical_gpus):
        command = [
            sys.executable,
            str(worker_script),
            str(manifest_path.resolve()),
            str(data_path.resolve()),
            "--worker-index",
            str(worker_index),
            "--num-workers",
            str(worker_count),
            "--local-rank",
            "0",
            "--physical-gpu",
            str(physical_gpu),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
        commands.append((command, environment))
    return commands


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _stop_workers(processes: list[subprocess.Popen[bytes]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=GRACEFUL_SHUTDOWN_SECONDS)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_workers(manifest_path: Path, data_path: Path) -> int:
    study_id, log_root, physical_gpus = _load_manifest_resources(manifest_path)
    commands = _build_worker_commands(manifest_path, data_path, physical_gpus)
    terminal_path = log_root / study_id / "workers-terminal.json"
    started_at = time.perf_counter()
    started_utc = datetime.now(UTC).isoformat()
    processes: list[subprocess.Popen[bytes]] = []
    exit_codes: dict[int, int] = {}
    status = "failed"
    try:
        for command, environment in commands:
            processes.append(subprocess.Popen(command, env=environment))

        while len(exit_codes) < len(processes):
            for worker_index, process in enumerate(processes):
                if worker_index in exit_codes:
                    continue
                exit_code = process.poll()
                if exit_code is None:
                    continue
                exit_codes[worker_index] = exit_code
                if exit_code != 0:
                    _stop_workers(processes)
                    for pending_index, pending_process in enumerate(processes):
                        if pending_index not in exit_codes:
                            exit_codes[pending_index] = pending_process.returncode
                    return_code = exit_code
                    break
            else:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue
            break
        else:
            return_code = 0
            status = "completed"
    except BaseException:
        _stop_workers(processes)
        raise
    finally:
        _write_json_atomic(
            terminal_path,
            {
                "active_seconds": time.perf_counter() - started_at,
                "completed_at": datetime.now(UTC).isoformat(),
                "data": str(data_path.resolve()),
                "exit_codes": {str(index): code for index, code in sorted(exit_codes.items())},
                "manifest": str(manifest_path.resolve()),
                "physical_gpus": physical_gpus,
                "started_at": started_utc,
                "status": status,
                "study_id": study_id,
            },
        )
    return return_code


def main() -> int:
    args = parse_args()
    return run_workers(args.manifest, args.data)


if __name__ == "__main__":
    raise SystemExit(main())
