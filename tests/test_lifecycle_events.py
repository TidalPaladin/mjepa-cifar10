from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mjepa_cifar10.research.lifecycle_events import (
    FIRST_CYCLE_FILENAME,
    PROGRESS_STALLED_FILENAME,
    SUPERVISOR_LOST_FILENAME,
    ProgressState,
    RunLifecycleReporter,
    persist_first_cycle_event,
    read_lifecycle_event,
    read_progress_state,
    reconcile_run_safety_events,
    write_progress_state,
)


NOW = datetime(2026, 7, 22, 14, 0, tzinfo=UTC)
THREAD_ID = "019f876b-21ff-7463-994e-46b075537a5a"


def make_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "logs" / "research" / "study-a" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    return run_dir


def test_progress_state_round_trips_latest_trainer_progress(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    progress = ProgressState(
        study_id="study-a",
        run_id="run-a",
        attempt=2,
        updated_at=NOW,
        phase="training",
        epoch=3,
        optimizer_step=125,
        active_seconds=91.5,
        originating_thread_id=THREAD_ID,
    )

    write_progress_state(run_dir, progress)

    assert read_progress_state(run_dir) == progress
    assert not tuple(run_dir.glob(".progress.json.*.tmp"))


def test_first_cycle_event_is_one_shot_and_stable_across_resume(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.touch()

    first = persist_first_cycle_event(
        run_dir,
        study_id="study-a",
        run_id="run-a",
        attempt=2,
        occurred_at=NOW,
        originating_thread_id=THREAD_ID,
        epoch=3,
        optimizer_step=125,
        active_seconds=91.5,
        checkpoint_path=checkpoint,
    )
    second = persist_first_cycle_event(
        run_dir,
        study_id="study-a",
        run_id="run-a",
        attempt=2,
        occurred_at=NOW + timedelta(hours=1),
        originating_thread_id=THREAD_ID,
        epoch=4,
        optimizer_step=175,
        active_seconds=151.5,
        checkpoint_path=checkpoint,
    )

    assert second == first
    assert first.kind == "first_cycle_completed"
    assert first.event_id == second.event_id
    assert read_lifecycle_event(run_dir / FIRST_CYCLE_FILENAME) == first


def test_run_reporter_persists_progress_and_first_cycle(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.touch()
    reporter = RunLifecycleReporter(
        run_dir=run_dir,
        study_id="study-a",
        run_id="run-a",
        attempt=2,
        originating_thread_id=THREAD_ID,
        now=lambda: NOW,
    )

    reporter.progress("validation", epoch=3, optimizer_step=125, active_seconds=90.0)
    event = reporter.first_cycle(
        epoch=3,
        optimizer_step=125,
        active_seconds=91.5,
        checkpoint_path=checkpoint,
    )

    assert read_progress_state(run_dir).phase == "validation"
    assert read_lifecycle_event(run_dir / FIRST_CYCLE_FILENAME) == event


def test_reconcile_synthesizes_supervisor_loss_once(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
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

    first = reconcile_run_safety_events(
        run_dir,
        now=NOW,
        progress_timeout=timedelta(minutes=30),
        pid_is_alive=lambda _pid: False,
    )
    second = reconcile_run_safety_events(
        run_dir,
        now=NOW + timedelta(minutes=1),
        progress_timeout=timedelta(minutes=30),
        pid_is_alive=lambda _pid: False,
    )

    assert [event.kind for event in first] == ["supervisor_lost"]
    assert second == ()
    assert read_lifecycle_event(run_dir / SUPERVISOR_LOST_FILENAME) == first[0]


def test_reconcile_synthesizes_progress_stall_only_for_live_supervisor(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    (run_dir / "worker.json").write_text(
        json.dumps(
            {
                "status": "running",
                "pid": 4242,
                "started_at": (NOW - timedelta(hours=1)).isoformat(),
                "heartbeat_at": NOW.isoformat(),
                "attempt": 1,
                "originating_thread_id": THREAD_ID,
            }
        ),
        encoding="utf-8",
    )
    write_progress_state(
        run_dir,
        ProgressState(
            study_id="study-a",
            run_id="run-a",
            attempt=1,
            updated_at=NOW - timedelta(minutes=31),
            phase="training",
            epoch=2,
            optimizer_step=100,
            active_seconds=600.0,
            originating_thread_id=THREAD_ID,
        ),
    )

    events = reconcile_run_safety_events(
        run_dir,
        now=NOW,
        progress_timeout=timedelta(minutes=30),
        pid_is_alive=lambda _pid: True,
    )

    assert [event.kind for event in events] == ["progress_stalled"]
    stalled = read_lifecycle_event(run_dir / PROGRESS_STALLED_FILENAME)
    assert stalled.details["optimizer_step"] == 100
    assert stalled.details["phase"] == "training"


def test_terminal_state_suppresses_safety_event_synthesis(tmp_path: Path) -> None:
    run_dir = make_run_dir(tmp_path)
    (run_dir / "worker.json").write_text(
        json.dumps({"status": "running", "pid": 4242, "attempt": 1}),
        encoding="utf-8",
    )
    (run_dir / "terminal.json").write_text('{"status":"completed"}', encoding="utf-8")

    assert (
        reconcile_run_safety_events(
            run_dir,
            now=NOW,
            progress_timeout=timedelta(minutes=30),
            pid_is_alive=lambda _pid: False,
        )
        == ()
    )
