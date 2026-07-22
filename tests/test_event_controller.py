from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mjepa_cifar10.research.cli import resolve_event_controller_socket
from mjepa_cifar10.research.codex_notifications import initialize_notification_root, read_notification_event
from mjepa_cifar10.research.event_controller import is_controller_source, reconcile_managed_root
from mjepa_cifar10.research.lifecycle_events import persist_first_cycle_event


NOW = datetime(2026, 7, 22, 14, 0, tzinfo=UTC)
THREAD_ID = "019f876b-21ff-7463-994e-46b075537a5a"
DAEMON_SOCKET = Path("/home/research/.codex/app-server-control/app-server-control.sock")


def make_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    run_dir = root / "study-a" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    return root, run_dir


def test_controller_discovers_stdio_daemon_socket_for_readiness_events() -> None:
    output = json.dumps({"status": "running", "socketPath": str(DAEMON_SOCKET)})

    resolved = resolve_event_controller_socket(None, "stdio", daemon_version=lambda: output)

    assert resolved == DAEMON_SOCKET


def test_controller_sources_exclude_notification_retry_writes() -> None:
    assert is_controller_source("terminal.json")
    assert is_controller_source("first-cycle.json")
    assert is_controller_source("worker.json")
    assert is_controller_source("progress.json")
    assert not is_controller_source("notification.json")
    assert not is_controller_source("first-cycle.notification.json")


def test_reconcile_queues_existing_lifecycle_event_once(tmp_path: Path) -> None:
    root, run_dir = make_root(tmp_path)
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.touch()
    persist_first_cycle_event(
        run_dir,
        study_id="study-a",
        run_id="run-a",
        attempt=1,
        occurred_at=NOW,
        originating_thread_id=THREAD_ID,
        epoch=0,
        optimizer_step=10,
        active_seconds=12.5,
        checkpoint_path=checkpoint,
    )

    first = reconcile_managed_root(
        root,
        now=NOW,
        progress_timeout=timedelta(minutes=30),
        pid_is_alive=lambda _pid: True,
    )
    second = reconcile_managed_root(
        root,
        now=NOW,
        progress_timeout=timedelta(minutes=30),
        pid_is_alive=lambda _pid: True,
    )

    assert first.queued == 1
    assert second.queued == 0
    notification = read_notification_event(run_dir / "first-cycle.notification.json", root)
    assert notification.event_kind == "first_cycle_completed"


def test_reconcile_isolates_malformed_legacy_terminal_state(tmp_path: Path) -> None:
    root, run_dir = make_root(tmp_path)
    (run_dir / "terminal.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "exit_code": 0,
                "started_at": (NOW - timedelta(minutes=1)).isoformat(),
                "finished_at": NOW.isoformat(),
                "physical_gpu": 1,
                "wandb_run_id": "legacy-run",
                "error": None,
                "attempt": 1,
                "terminal_event_id": "legacy-non-uuid",
                "originating_thread_id": THREAD_ID,
            }
        ),
        encoding="utf-8",
    )

    result = reconcile_managed_root(
        root,
        now=NOW,
        progress_timeout=timedelta(minutes=30),
        pid_is_alive=lambda _pid: False,
    )

    assert result.queued == 0
    assert len(result.problems) == 1
    assert "event_id must be a UUID string" in result.problems[0]
    assert not (run_dir / "notification.json").exists()


def test_reconcile_queues_supervisor_loss_without_changing_terminal_truth(tmp_path: Path) -> None:
    root, run_dir = make_root(tmp_path)
    (run_dir / "worker.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": 4242,
                "started_at": (NOW - timedelta(minutes=10)).isoformat(),
                "heartbeat_at": (NOW - timedelta(minutes=1)).isoformat(),
                "attempt": 1,
                "originating_thread_id": THREAD_ID,
            }
        ),
        encoding="utf-8",
    )

    result = reconcile_managed_root(
        root,
        now=NOW,
        progress_timeout=timedelta(minutes=30),
        pid_is_alive=lambda _pid: False,
    )

    assert result.created_kinds == ("supervisor_lost",)
    assert result.queued == 1
    assert not (run_dir / "terminal.json").exists()
    notification = read_notification_event(run_dir / "supervisor-lost.notification.json", root)
    assert notification.event_kind == "supervisor_lost"
