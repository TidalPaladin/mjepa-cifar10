from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any

import pytest
from notify_wake import MessageTransport, WakeContext

from mjepa_cifar10.research.cli import capture_launch_wake_context
from mjepa_cifar10.research.codex_notifications import (
    NotificationEvent,
    NotificationStateError,
    build_wake_prompt,
    enter_research_notify_wait,
    goal_wait_path,
    initialize_notification_root,
    notification_namespace,
    notification_path_for_event,
    persist_wake_context,
    queue_notification_from_lifecycle,
    queue_notification_from_terminal,
    read_notification_event,
    sweep_notifications,
)
from mjepa_cifar10.research.lifecycle_events import persist_first_cycle_event
from mjepa_cifar10.research.runtime import atomic_write_json


NOW = datetime(2026, 7, 29, 19, 0, tzinfo=UTC)
THREAD_ID = "019fa9c6-3613-7e60-a328-bf6f5c62c7bd"
EVENT_ID = "22345678-1234-5678-9234-567812345678"
PERMISSION_PROFILE = ":danger-full-access"


def run(coroutine: Any) -> Any:
    return asyncio.run(coroutine)


class ScriptedTransport(MessageTransport):
    def __init__(
        self,
        handler: Callable[[dict[str, Any]], list[dict[str, Any]]],
        *,
        fail_after_method: str | None = None,
    ) -> None:
        self._handler = handler
        self._fail_after_method = fail_after_method
        self._responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.sent: list[dict[str, Any]] = []
        self.closed = False

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        for response in self._handler(message):
            self._responses.put_nowait(response)
        if message.get("method") == self._fail_after_method:
            raise ConnectionError("lost acknowledgment")

    async def receive(self) -> dict[str, Any]:
        return await self._responses.get()

    async def close(self) -> None:
        self.closed = True


def context() -> WakeContext:
    return WakeContext(
        thread_id=THREAD_ID,
        permission_profile=PERMISSION_PROFILE,
        approval_policy="never",
        captured_at=NOW,
        goal_snapshot=None,
    )


def goal(*, status: str, updated_at: int) -> dict[str, Any]:
    return {
        "threadId": THREAD_ID,
        "objective": "wait for the research controller",
        "status": status,
        "tokenBudget": 10_000,
        "tokensUsed": 100,
        "timeUsedSeconds": 10,
        "createdAt": 1,
        "updatedAt": updated_at,
    }


def handler(
    *,
    selected_goal: dict[str, Any] | None = None,
    history_event_id: str | None = None,
) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    current_goal = None if selected_goal is None else dict(selected_goal)

    def respond(message: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal current_goal
        if "id" not in message:
            return []
        request_id = message["id"]
        method = message.get("method")
        if method == "initialize":
            return [{"id": request_id, "result": {"userAgent": "fake"}}]
        if method == "thread/resume":
            return [
                {
                    "id": request_id,
                    "result": {
                        "thread": {
                            "id": THREAD_ID,
                            "status": {"type": "idle"},
                            "turns": [],
                        },
                        "activePermissionProfile": {"id": PERMISSION_PROFILE},
                        "approvalPolicy": "never",
                    },
                }
            ]
        if method == "thread/goal/get":
            return [{"id": request_id, "result": {"goal": current_goal}}]
        if method == "thread/goal/set":
            assert current_goal is not None
            current_goal = {
                **current_goal,
                "status": message["params"]["status"],
                "updatedAt": current_goal["updatedAt"] + 1,
            }
            return [{"id": request_id, "result": {"goal": current_goal}}]
        if method == "thread/read":
            turns = []
            if history_event_id is not None:
                turns = [
                    {
                        "id": "history-turn",
                        "status": "completed",
                        "items": [
                            {
                                "type": "userMessage",
                                "clientId": history_event_id,
                                "content": [],
                            }
                        ],
                    }
                ]
            return [
                {
                    "id": request_id,
                    "result": {
                        "thread": {
                            "id": THREAD_ID,
                            "status": {"type": "idle"},
                            "turns": turns,
                        }
                    },
                }
            ]
        if method == "turn/start":
            return [{"id": request_id, "result": {"turn": {"id": "wake-turn"}}}]
        raise AssertionError(f"unexpected method: {method}")

    return respond


def prepared_event(tmp_path: Path) -> tuple[Path, Path, NotificationEvent]:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    run_dir = root / "study-a" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    persist_wake_context(run_dir, root, context())
    terminal_path = run_dir / "terminal.json"
    atomic_write_json(
        terminal_path,
        {
            "notify_wake_contract_version": 2,
            "status": "completed",
            "exit_code": 0,
            "started_at": (NOW - timedelta(minutes=1)).isoformat(),
            "finished_at": NOW.isoformat(),
            "attempt": 1,
            "terminal_event_id": EVENT_ID,
            "originating_thread_id": THREAD_ID,
        },
    )
    event = queue_notification_from_terminal(
        terminal_path,
        root,
        study_id="study-a",
        run_id="run-a",
    )
    return root, run_dir, event


def test_new_events_use_only_version_two_namespace(tmp_path: Path) -> None:
    root, run_dir, event = prepared_event(tmp_path)
    path = notification_path_for_event(root, EVENT_ID)

    assert notification_namespace(root) == root / ".notify-wake" / "v2"
    assert path.is_file()
    assert event.state == "pending"
    assert not (run_dir / "notification.json").exists()
    assert not (run_dir / "wake-context.json").exists()


def test_sweep_starts_root_without_model_override(tmp_path: Path) -> None:
    root, _run_dir, _event = prepared_event(tmp_path)
    transport = ScriptedTransport(handler())

    result = run(
        sweep_notifications(
            root,
            connect=lambda: asyncio.sleep(0, result=transport),
            now=lambda: NOW,
            random=Random(0),
        )
    )

    assert result.accepted == 1
    persisted = read_notification_event(
        notification_path_for_event(root, EVENT_ID),
        root,
    )
    assert persisted.state == "accepted"
    start = next(message for message in transport.sent if message.get("method") == "turn/start")
    assert "model" not in start["params"]
    assert "effort" not in start["params"]


def test_lost_acknowledgment_reconciles_without_duplicate_start(tmp_path: Path) -> None:
    root, _run_dir, _event = prepared_event(tmp_path)
    first = ScriptedTransport(handler(), fail_after_method="turn/start")

    uncertain = run(
        sweep_notifications(
            root,
            connect=lambda: asyncio.sleep(0, result=first),
            now=lambda: NOW,
            random=Random(0),
        )
    )
    assert uncertain.retrying == 1
    persisted = read_notification_event(
        notification_path_for_event(root, EVENT_ID),
        root,
    )
    assert persisted.state == "uncertain"

    history = ScriptedTransport(handler(history_event_id=EVENT_ID))
    accepted = run(
        sweep_notifications(
            root,
            connect=lambda: asyncio.sleep(0, result=history),
            now=lambda: NOW + timedelta(seconds=1),
            random=Random(0),
        )
    )
    assert accepted.accepted == 1
    assert not any(message.get("method") == "turn/start" for message in history.sent)


def test_manually_blocked_goal_is_preserved(tmp_path: Path) -> None:
    root, _run_dir, _event = prepared_event(tmp_path)
    transport = ScriptedTransport(handler(selected_goal=goal(status="blocked", updated_at=3)))

    result = run(
        sweep_notifications(
            root,
            connect=lambda: asyncio.sleep(0, result=transport),
            now=lambda: NOW,
            random=Random(0),
        )
    )

    assert result.failed == 1
    assert not any(message.get("method") == "thread/goal/set" for message in transport.sent)


def test_owned_notify_wait_is_durable_under_v2_namespace(tmp_path: Path) -> None:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    transport = ScriptedTransport(handler(selected_goal=goal(status="active", updated_at=2)))

    lease = run(
        enter_research_notify_wait(
            root,
            context=context(),
            loop_id="research:study-a",
            source_ids=("controller:study-a",),
            transport=transport,
            verify_loop_identity=lambda loop_id, source_ids: (
                loop_id == "research:study-a" and source_ids == ("controller:study-a",)
            ),
        )
    )

    assert lease.state == "owned"
    assert goal_wait_path(root, THREAD_ID).is_file()


def test_version_one_terminal_and_notification_are_not_consumed(tmp_path: Path) -> None:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    run_dir = root / "study-a" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    terminal_path = run_dir / "terminal.json"
    atomic_write_json(
        terminal_path,
        {
            "status": "completed",
            "finished_at": NOW.isoformat(),
            "attempt": 1,
            "terminal_event_id": EVENT_ID,
            "originating_thread_id": THREAD_ID,
        },
    )
    legacy_notification = run_dir / "notification.json"
    legacy_notification.write_text('{"schema_version":1,"state":"pending"}\n', encoding="utf-8")

    with pytest.raises(NotificationStateError, match="cutover required"):
        queue_notification_from_terminal(
            terminal_path,
            root,
            study_id="study-a",
            run_id="run-a",
        )

    result = run(
        sweep_notifications(
            root,
            connect=lambda: (_ for _ in ()).throw(AssertionError("must not connect")),
            now=lambda: NOW,
        )
    )
    assert result.discovered == 0
    assert legacy_notification.is_file()


def test_lifecycle_events_queue_in_version_two_namespace(tmp_path: Path) -> None:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    run_dir = root / "study-a" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    persist_wake_context(run_dir, root, context())
    checkpoint = run_dir / "checkpoint.pt"
    checkpoint.touch()
    lifecycle = persist_first_cycle_event(
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

    event = queue_notification_from_lifecycle(Path(lifecycle.event_state_path), root)

    assert event.event_kind == "first_cycle_completed"
    assert "Event: first_cycle_completed" in build_wake_prompt(event)
    assert notification_path_for_event(root, event.event_id).is_file()
    assert not (run_dir / "first-cycle.notification.json").exists()


def test_launch_capture_uses_shared_context_capture(
    mocker: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transport = object()
    monkeypatch.setenv("CODEX_THREAD_ID", THREAD_ID)
    monkeypatch.delenv("CODEX_PERMISSION_PROFILE", raising=False)
    mocker.patch(
        "mjepa_cifar10.research.cli.resolve_event_controller_socket",
        return_value=tmp_path / "app-server.sock",
    )
    connect = mocker.patch(
        "mjepa_cifar10.research.cli.UnixWebSocketTransport.connect",
        new=mocker.AsyncMock(return_value=transport),
    )
    capture = mocker.patch(
        "mjepa_cifar10.research.cli.capture_wake_context",
        new=mocker.AsyncMock(return_value=context()),
    )

    assert capture_launch_wake_context() == context()
    connect.assert_awaited_once_with(tmp_path / "app-server.sock")
    capture.assert_awaited_once_with(
        thread_id=THREAD_ID,
        requested_permission_profile=None,
        transport=transport,
    )


def test_wake_context_version_one_shape_is_rejected() -> None:
    payload = context().to_dict()
    payload["schema_version"] = 1

    with pytest.raises(ValueError, match="cutover required"):
        WakeContext.from_dict(payload)


def test_context_recapture_preserves_original_v2_record(tmp_path: Path) -> None:
    root, run_dir, event = prepared_event(tmp_path)
    context_path = notification_namespace(root) / "contexts" / event.study_id / event.run_id / "wake-context.json"
    original = context_path.read_bytes()

    assert (
        persist_wake_context(
            run_dir,
            root,
            replace(context(), captured_at=NOW + timedelta(hours=1)),
        )
        == context_path
    )
    assert context_path.read_bytes() == original
