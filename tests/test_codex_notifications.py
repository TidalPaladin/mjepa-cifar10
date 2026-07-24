from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from functools import wraps
from pathlib import Path
from random import Random
from typing import Any, ParamSpec

import pytest
from websockets.asyncio.server import ServerConnection, unix_serve

from mjepa_cifar10.research.cli import capture_launch_wake_context
from mjepa_cifar10.research.codex_notifications import (
    APP_SERVER_MESSAGE_LIMIT_BYTES,
    MANAGED_ROOT_SCHEMA_VERSION,
    MAX_DELIVERY_ATTEMPTS,
    AppServerProtocolError,
    NotificationEvent,
    NotificationStateError,
    RpcClient,
    UnixWebSocketTransport,
    build_wake_prompt,
    capture_wake_context,
    deliver_notification,
    ensure_notification,
    initialize_notification_root,
    notification_lock_path,
    persist_wake_context,
    queue_notification_from_lifecycle,
    queue_notification_from_terminal,
    read_notification_event,
    register_notification_root,
    sweep_notifications,
    validate_notification_root,
    write_notification_event,
)
from mjepa_cifar10.research.lifecycle_events import persist_first_cycle_event
from mjepa_cifar10.research.runtime import atomic_write_json
from mjepa_cifar10.research.wake_context import WakeContext


EVENT_ID = "12345678-1234-5678-9234-567812345678"
THREAD_ID = "019f8098-aa66-7011-bc23-c3b3a78f7501"
EXPECTED_WAKE_MODEL = "gpt-5.6-luna"
EXPECTED_WAKE_EFFORT = "medium"
PERMISSION_PROFILE = ":danger-full-access"
APPROVAL_POLICY = "never"
NOW = datetime(2026, 7, 20, 12, 0, tzinfo=UTC)
WAKE_CONTEXT = WakeContext(
    thread_id=THREAD_ID,
    permission_profile=PERMISSION_PROFILE,
    approval_policy=APPROVAL_POLICY,
    captured_at=NOW,
)
P = ParamSpec("P")


def run_async(function: Callable[P, Coroutine[Any, Any, None]]) -> Callable[P, None]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> None:
        asyncio.run(function(*args, **kwargs))

    return wrapped


@run_async
async def test_unix_transport_uses_app_server_compatible_handshake(tmp_path: Path) -> None:
    socket_path = tmp_path / "fake-app-server.sock"
    request_headers: dict[str, str] = {}
    request_received = asyncio.Event()

    async def handler(connection: ServerConnection) -> None:
        assert connection.request is not None
        request_headers.update(connection.request.headers)
        request_received.set()
        await connection.send(json.dumps({"payload": "x" * (APP_SERVER_MESSAGE_LIMIT_BYTES // 4)}))
        await connection.wait_closed()

    server = await unix_serve(
        handler,
        path=str(socket_path),
        compression=None,
        server_header=None,
    )
    try:
        transport = await UnixWebSocketTransport.connect(socket_path)
        await request_received.wait()
        message = await transport.receive()
        await transport.close()
    finally:
        server.close()
        await server.wait_closed()

    assert "sec-websocket-extensions" not in request_headers
    assert "user-agent" not in request_headers
    assert len(message["payload"]) == APP_SERVER_MESSAGE_LIMIT_BYTES // 4


class ScriptedTransport:
    def __init__(self, handler: Callable[[dict[str, Any]], list[dict[str, Any]]]) -> None:
        self.handler = handler
        self.sent: list[dict[str, Any]] = []
        self.incoming: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.closed = False

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)
        for response in self.handler(message):
            await self.incoming.put(response)

    async def receive(self) -> dict[str, Any]:
        return await self.incoming.get()

    async def close(self) -> None:
        self.closed = True


class GatedStartTransport(ScriptedTransport):
    def __init__(self, started: asyncio.Event, release: asyncio.Event) -> None:
        super().__init__(app_server_handler("idle", []))
        self.started = started
        self.release = release

    async def send(self, message: dict[str, Any]) -> None:
        if message.get("method") == "turn/start":
            self.sent.append(message)
            self.started.set()
            await self.release.wait()
            await self.incoming.put({"id": message["id"], "result": {"turn": {"id": "new-turn"}}})
            return
        await super().send(message)


def app_server_handler(
    status: str,
    turns: list[dict[str, Any]],
    *,
    goal_status: str = "active",
    permission_profile: str | None = PERMISSION_PROFILE,
    approval_policy: str = APPROVAL_POLICY,
) -> Callable[[dict[str, Any]], list[dict[str, Any]]]:
    experimental_api = False

    def handle(message: dict[str, Any]) -> list[dict[str, Any]]:
        nonlocal experimental_api
        if "id" not in message:
            return []
        request_id = message["id"]
        method = message["method"]
        thread = {"id": THREAD_ID, "status": {"type": status}, "turns": turns}
        if method == "initialize":
            experimental_api = message["params"].get("capabilities", {}).get("experimentalApi") is True
            result: dict[str, Any] = {}
        elif method == "thread/resume" and message["params"].get("permissions") and not experimental_api:
            return [
                {
                    "id": request_id,
                    "error": {
                        "code": -32600,
                        "message": "thread/resume.permissions requires experimentalApi capability",
                    },
                }
            ]
        elif method == "thread/resume":
            result = {
                "thread": thread,
                "activePermissionProfile": ({"id": permission_profile} if permission_profile is not None else None),
                "approvalPolicy": approval_policy,
            }
        elif method == "thread/read":
            result = {"thread": thread}
        elif method == "thread/goal/get":
            result = {"goal": {"threadId": THREAD_ID, "status": goal_status}}
        elif method == "thread/goal/set":
            result = {"goal": {"threadId": THREAD_ID, "status": message["params"]["status"]}}
        elif method == "turn/start":
            result = {"turn": {"id": "new-turn"}}
        elif method == "turn/steer":
            result = {"turnId": "active-turn"}
        else:
            raise AssertionError(method)
        return [{"id": request_id, "result": result}]

    return handle


@run_async
async def test_capture_wake_context_records_effective_profile_and_approval_policy() -> None:
    transport = ScriptedTransport(app_server_handler("active", []))

    context = await capture_wake_context(
        thread_id=THREAD_ID,
        requested_permission_profile=PERMISSION_PROFILE,
        transport=transport,
        captured_at=NOW,
    )

    assert context == WAKE_CONTEXT
    initialize = next(message for message in transport.sent if message.get("method") == "initialize")
    assert initialize["params"]["capabilities"] == {"experimentalApi": True}
    resume = next(message for message in transport.sent if message.get("method") == "thread/resume")
    assert resume["params"] == {
        "threadId": THREAD_ID,
        "permissions": PERMISSION_PROFILE,
    }


@run_async
async def test_capture_wake_context_discovers_implicit_permission_profile() -> None:
    transport = ScriptedTransport(app_server_handler("active", []))

    context = await capture_wake_context(
        thread_id=THREAD_ID,
        requested_permission_profile=None,
        transport=transport,
        captured_at=NOW,
    )

    assert context == WAKE_CONTEXT
    resume = next(message for message in transport.sent if message.get("method") == "thread/resume")
    assert resume["params"] == {"threadId": THREAD_ID}


@run_async
async def test_capture_wake_context_rejects_unselectable_permission_profile() -> None:
    transport = ScriptedTransport(app_server_handler("active", [], permission_profile=None))

    with pytest.raises(AppServerProtocolError, match="selectable effective permission profile") as error:
        await capture_wake_context(
            thread_id=THREAD_ID,
            requested_permission_profile=None,
            transport=transport,
            captured_at=NOW,
        )

    assert error.value.permanent
    resume = next(message for message in transport.sent if message.get("method") == "thread/resume")
    assert resume["params"] == {"threadId": THREAD_ID}


@run_async
async def test_capture_wake_context_rejects_missing_permission_profile_field() -> None:
    base_handler = app_server_handler("active", [], permission_profile=None)

    def handler(message: dict[str, Any]) -> list[dict[str, Any]]:
        responses = base_handler(message)
        if message.get("method") == "thread/resume":
            del responses[0]["result"]["activePermissionProfile"]
        return responses

    with pytest.raises(AppServerProtocolError, match="missing the effective permission profile") as error:
        await capture_wake_context(
            thread_id=THREAD_ID,
            requested_permission_profile=None,
            transport=ScriptedTransport(handler),
            captured_at=NOW,
        )

    assert error.value.permanent


def test_launch_capture_discovers_profile_when_environment_has_no_profile_name(
    mocker,
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
        new=mocker.AsyncMock(return_value=WAKE_CONTEXT),
    )

    assert capture_launch_wake_context() == WAKE_CONTEXT
    connect.assert_awaited_once_with(tmp_path / "app-server.sock")
    capture.assert_awaited_once_with(
        thread_id=THREAD_ID,
        requested_permission_profile=None,
        transport=transport,
    )


def test_launch_capture_rejects_empty_requested_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CODEX_THREAD_ID", THREAD_ID)
    monkeypatch.setenv("CODEX_PERMISSION_PROFILE", "")

    with pytest.raises(RuntimeError, match="CODEX_PERMISSION_PROFILE must be non-empty"):
        capture_launch_wake_context()


@run_async
async def test_blocked_goal_is_resumed_before_wake(tmp_path: Path) -> None:
    _root, event = queued_notification(tmp_path)
    transport = ScriptedTransport(app_server_handler("idle", [], goal_status="blocked"))

    await deliver_notification(event, transport)

    methods = [message.get("method") for message in transport.sent]
    assert methods.index("thread/resume") < methods.index("thread/goal/set")
    resume = next(message for message in transport.sent if message.get("method") == "thread/resume")
    assert resume["params"] == {
        "threadId": THREAD_ID,
        "permissions": PERMISSION_PROFILE,
        "approvalPolicy": APPROVAL_POLICY,
    }
    goal_set = next(message for message in transport.sent if message.get("method") == "thread/goal/set")
    assert goal_set["params"] == {"threadId": THREAD_ID, "status": "active"}


def queued_notification(tmp_path: Path, *, thread_id: str | None = THREAD_ID) -> tuple[Path, NotificationEvent]:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    run_dir = root / "study-a" / "runs" / "run-a"
    if thread_id is not None:
        persist_wake_context(run_dir, root, WAKE_CONTEXT)
    terminal_path = run_dir / "terminal.json"
    atomic_write_json(
        terminal_path,
        {
            "status": "completed",
            "exit_code": 0,
            "started_at": (NOW - timedelta(minutes=1)).isoformat(),
            "finished_at": NOW.isoformat(),
            "physical_gpu": 1,
            "wandb_run_id": "wandb-1",
            "error": None,
            "attempt": 1,
            "terminal_event_id": EVENT_ID,
            "originating_thread_id": thread_id,
        },
    )
    event = queue_notification_from_terminal(
        terminal_path,
        root,
        study_id="study-a",
        run_id="run-a",
    )
    return root, event


def test_wake_prompt_contains_only_validated_terminal_identifiers(tmp_path: Path) -> None:
    _root, event = queued_notification(tmp_path)
    prompt = build_wake_prompt(event)

    assert "Study: study-a" in prompt
    assert "Run: run-a" in prompt
    assert "Status: completed" in prompt
    assert event.terminal_state_path in prompt
    assert "wandb-1" not in prompt


def test_persisted_wake_context_cannot_be_replaced(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)

    with pytest.raises(NotificationStateError, match="different immutable wake context"):
        persist_wake_context(
            Path(event.terminal_state_path).parent,
            root,
            WakeContext(
                thread_id=THREAD_ID,
                permission_profile=":workspace",
                approval_policy=APPROVAL_POLICY,
                captured_at=NOW,
            ),
        )


def test_new_wake_context_requires_selectable_permission_profile(tmp_path: Path) -> None:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    legacy_context = WakeContext(
        thread_id=THREAD_ID,
        permission_profile=None,
        approval_policy=APPROVAL_POLICY,
        captured_at=NOW,
    )

    with pytest.raises(NotificationStateError, match="selectable permission profile"):
        persist_wake_context(root / "study-a" / "runs" / "run-a", root, legacy_context)


def test_lifecycle_wake_prompt_contains_only_validated_event_identifiers(tmp_path: Path) -> None:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    run_dir = root / "study-a" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    persist_wake_context(run_dir, root, WAKE_CONTEXT)
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

    prompt = build_wake_prompt(event)

    assert "Event: first_cycle_completed" in prompt
    assert "Study: study-a" in prompt
    assert "Run: run-a" in prompt
    assert lifecycle.event_state_path in prompt
    assert "checkpoint.pt" not in prompt


@run_async
async def test_sweep_accepts_lifecycle_event_once(tmp_path: Path) -> None:
    root = tmp_path / "logs" / "research"
    initialize_notification_root(root)
    run_dir = root / "study-a" / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    persist_wake_context(run_dir, root, WAKE_CONTEXT)
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
    calls = 0

    async def connect() -> ScriptedTransport:
        nonlocal calls
        calls += 1
        return ScriptedTransport(app_server_handler("idle", []))

    first = await sweep_notifications(root, connect=connect, now=lambda: NOW)
    second = await sweep_notifications(root, connect=connect, now=lambda: NOW)
    persisted = read_notification_event(
        Path(event.terminal_state_path).with_name("first-cycle.notification.json"),
        root,
    )

    assert first.accepted == 1
    assert second.due == 0
    assert calls == 1
    assert persisted.state == "accepted"


@run_async
async def test_idle_thread_starts_turn(tmp_path: Path) -> None:
    _root, event = queued_notification(tmp_path)
    transport = ScriptedTransport(app_server_handler("idle", []))

    acceptance = await deliver_notification(event, transport)

    assert acceptance.rpc_method == "turn/start"
    assert acceptance.turn_id == "new-turn"
    start = next(message for message in transport.sent if message.get("method") == "turn/start")
    assert start["params"]["clientUserMessageId"] == EVENT_ID
    assert start["params"]["model"] == EXPECTED_WAKE_MODEL
    assert start["params"]["effort"] == EXPECTED_WAKE_EFFORT
    assert start["params"]["permissions"] == PERMISSION_PROFILE
    assert start["params"]["approvalPolicy"] == APPROVAL_POLICY


@run_async
async def test_profile_mismatch_fails_before_goal_reactivation(tmp_path: Path) -> None:
    _root, event = queued_notification(tmp_path)
    transport = ScriptedTransport(
        app_server_handler(
            "idle",
            [],
            goal_status="blocked",
            permission_profile=":workspace",
        )
    )

    with pytest.raises(AppServerProtocolError, match="permission profile mismatch") as error:
        await deliver_notification(event, transport)

    assert error.value.permanent
    assert not any(
        message.get("method") in {"thread/goal/set", "thread/read", "turn/start", "turn/steer"}
        for message in transport.sent
    )


@run_async
async def test_legacy_unnamed_profile_requires_explicit_recovery(tmp_path: Path) -> None:
    _root, event = queued_notification(tmp_path)
    legacy_context = WakeContext(
        thread_id=THREAD_ID,
        permission_profile=None,
        approval_policy=APPROVAL_POLICY,
        captured_at=NOW,
    )
    transport = ScriptedTransport(app_server_handler("idle", []))

    with pytest.raises(AppServerProtocolError, match="legacy wake context") as error:
        await deliver_notification(replace(event, wake_context=legacy_context), transport)

    assert error.value.permanent
    assert transport.closed
    assert not transport.sent


@run_async
async def test_active_thread_steers_existing_turn(tmp_path: Path) -> None:
    _root, event = queued_notification(tmp_path)
    turns = [{"id": "active-turn", "status": "inProgress"}]
    transport = ScriptedTransport(app_server_handler("active", turns))

    acceptance = await deliver_notification(event, transport)

    assert acceptance.rpc_method == "turn/steer"
    steer = next(message for message in transport.sent if message.get("method") == "turn/steer")
    assert steer["params"]["expectedTurnId"] == "active-turn"
    assert "model" not in steer["params"]
    assert "effort" not in steer["params"]


@run_async
async def test_active_thread_steers_newest_turn_when_history_contains_stale_in_progress_turn(
    tmp_path: Path,
) -> None:
    _root, event = queued_notification(tmp_path)
    turns = [
        {"id": "stale-turn", "status": "inProgress"},
        {"id": "completed-turn", "status": "completed"},
        {"id": "active-turn", "status": "inProgress"},
    ]
    transport = ScriptedTransport(app_server_handler("active", turns))

    acceptance = await deliver_notification(event, transport)

    assert acceptance.rpc_method == "turn/steer"
    assert acceptance.turn_id == "active-turn"
    steer = next(message for message in transport.sent if message.get("method") == "turn/steer")
    assert steer["params"]["expectedTurnId"] == "active-turn"


@pytest.mark.parametrize(
    ("status", "turns"),
    (
        ("notLoaded", []),
        ("active", []),
        ("active", [{"id": "a", "status": "completed"}, {"id": "b", "status": "completed"}]),
        ("idle", [{"id": "a", "status": "inProgress"}]),
    ),
)
@run_async
async def test_racy_or_nonsteerable_state_is_not_accepted(
    tmp_path: Path, status: str, turns: list[dict[str, Any]]
) -> None:
    _root, event = queued_notification(tmp_path)
    transport = ScriptedTransport(app_server_handler(status, turns))

    with pytest.raises(AppServerProtocolError):
        await deliver_notification(event, transport, request_timeout=0.2)

    assert not any(message.get("method") in {"turn/start", "turn/steer"} for message in transport.sent)


@run_async
async def test_rpc_client_rejects_server_requests_without_approval() -> None:
    def handler(message: dict[str, Any]) -> list[dict[str, Any]]:
        if message.get("method") != "initialize":
            return []
        return [
            {"id": "approval-1", "method": "item/commandExecution/requestApproval", "params": {}},
            {"id": message["id"], "result": {}},
        ]

    transport = ScriptedTransport(handler)
    async with RpcClient(transport, request_timeout=0.2) as client:
        await client.request("initialize", {})

    rejection = next(message for message in transport.sent if message.get("id") == "approval-1")
    assert rejection["error"]["code"] == -32601
    assert "result" not in rejection


@run_async
async def test_missing_thread_is_permanent_delivery_failure(tmp_path: Path) -> None:
    _root, event = queued_notification(tmp_path, thread_id=None)
    transport = ScriptedTransport(lambda _message: [])

    with pytest.raises(AppServerProtocolError, match="no originating") as error:
        await deliver_notification(event, transport)

    assert error.value.permanent
    assert transport.closed


@run_async
async def test_sweep_accepts_once_and_deduplicates_event(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)
    calls = 0

    async def connect() -> ScriptedTransport:
        nonlocal calls
        calls += 1
        return ScriptedTransport(app_server_handler("idle", []))

    first = await sweep_notifications(root, connect=connect, now=lambda: NOW, random=Random(0))
    second = await sweep_notifications(root, connect=connect, now=lambda: NOW, random=Random(0))
    persisted = read_notification_event(Path(event.terminal_state_path).with_name("notification.json"), root)

    assert first.accepted == 1
    assert second.due == 0
    assert calls == 1
    assert persisted.state == "accepted"
    assert notification_lock_path(root, THREAD_ID).is_file()


@run_async
async def test_sweep_can_isolate_one_study_from_unrelated_notification_failures(tmp_path: Path) -> None:
    root, study_a_event = queued_notification(tmp_path)
    run_dir = root / "study-b" / "runs" / "run-b"
    persist_wake_context(
        run_dir,
        root,
        WakeContext(
            thread_id=THREAD_ID,
            permission_profile=PERMISSION_PROFILE,
            approval_policy=APPROVAL_POLICY,
            captured_at=NOW,
        ),
    )
    terminal_path = run_dir / "terminal.json"
    atomic_write_json(
        terminal_path,
        {
            "status": "completed",
            "exit_code": 0,
            "started_at": (NOW - timedelta(minutes=1)).isoformat(),
            "finished_at": NOW.isoformat(),
            "physical_gpu": 2,
            "wandb_run_id": "wandb-2",
            "error": None,
            "attempt": 1,
            "terminal_event_id": "87654321-4321-5678-9234-567812345678",
            "originating_thread_id": THREAD_ID,
        },
    )
    study_b_event = queue_notification_from_terminal(
        terminal_path,
        root,
        study_id="study-b",
        run_id="run-b",
    )
    calls = 0

    async def connect() -> ScriptedTransport:
        nonlocal calls
        calls += 1
        return ScriptedTransport(app_server_handler("idle", []))

    result = await sweep_notifications(
        root,
        connect=connect,
        now=lambda: NOW,
        study_ids=frozenset({"study-b"}),
    )

    assert result.discovered == 1
    assert result.accepted == 1
    assert calls == 1
    assert (
        read_notification_event(Path(study_a_event.terminal_state_path).with_name("notification.json"), root).state
        == "pending"
    )
    assert (
        read_notification_event(Path(study_b_event.terminal_state_path).with_name("notification.json"), root).state
        == "accepted"
    )


@run_async
async def test_acceptance_is_persisted_only_after_server_response(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)
    path = Path(event.terminal_state_path).with_name("notification.json")
    started = asyncio.Event()
    release = asyncio.Event()

    async def connect() -> ScriptedTransport:
        return GatedStartTransport(started, release)

    sweep = asyncio.create_task(sweep_notifications(root, connect=connect, now=lambda: NOW))
    await asyncio.wait_for(started.wait(), timeout=1)
    assert read_notification_event(path, root).state == "pending"
    release.set()
    result = await asyncio.wait_for(sweep, timeout=1)

    assert result.accepted == 1
    assert read_notification_event(path, root).state == "accepted"


@run_async
async def test_concurrent_sweeps_serialize_delivery_per_thread(tmp_path: Path) -> None:
    root, _event = queued_notification(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def connect() -> ScriptedTransport:
        nonlocal calls
        calls += 1
        return GatedStartTransport(started, release)

    first = asyncio.create_task(sweep_notifications(root, connect=connect, now=lambda: NOW))
    await asyncio.wait_for(started.wait(), timeout=1)
    second = asyncio.create_task(sweep_notifications(root, connect=connect, now=lambda: NOW))
    await asyncio.sleep(0.02)
    release.set()
    first_result, second_result = await asyncio.wait_for(asyncio.gather(first, second), timeout=1)

    assert first_result.accepted + second_result.accepted == 1
    assert calls == 1


@run_async
async def test_sweep_records_jittered_retry(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)

    async def connect() -> ScriptedTransport:
        return ScriptedTransport(lambda _message: [])

    result = await sweep_notifications(
        root,
        connect=connect,
        now=lambda: NOW,
        random=Random(0),
        request_timeout=0.01,
    )
    persisted = read_notification_event(Path(event.terminal_state_path).with_name("notification.json"), root)

    assert result.retrying == 1
    assert persisted.attempt_count == 1
    assert NOW < persisted.next_attempt_at <= NOW + timedelta(seconds=5)  # type: ignore[operator]


@run_async
async def test_sweep_fails_after_maximum_attempts(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)
    path = Path(event.terminal_state_path).with_name("notification.json")
    current = event
    for index in range(MAX_DELIVERY_ATTEMPTS - 1):
        current = current.with_delivery_failure(
            attempted_at=NOW - timedelta(minutes=10 - index),
            error="retry",
            next_attempt_at=NOW,
            exhausted=False,
        )
    write_notification_event(current, root)

    async def connect() -> ScriptedTransport:
        return ScriptedTransport(lambda _message: [])

    result = await sweep_notifications(root, connect=connect, now=lambda: NOW, request_timeout=0.01)

    assert result.failed == 1
    assert read_notification_event(path, root).state == "failed"


@run_async
async def test_sweep_replaces_invalid_notification_with_failed_state(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)
    path = Path(event.terminal_state_path).with_name("notification.json")
    path.write_text("not-json", encoding="utf-8")

    async def connect() -> ScriptedTransport:
        raise AssertionError("invalid state must not connect")

    result = await sweep_notifications(root, connect=connect, now=lambda: NOW)
    persisted = read_notification_event(path, root)

    assert result.failed == 1
    assert persisted.state == "failed"
    assert persisted.last_error is not None


@run_async
async def test_sweep_validates_root_and_clock(tmp_path: Path) -> None:
    async def connect() -> ScriptedTransport:
        raise AssertionError("no delivery should be attempted")

    assert (await sweep_notifications(tmp_path / "missing", connect=connect)).discovered == 0
    with pytest.raises(NotificationStateError, match="filesystem root"):
        await sweep_notifications(Path("/"), connect=connect)
    initialize_notification_root(tmp_path)
    with pytest.raises(NotificationStateError, match="offset-aware"):
        await sweep_notifications(tmp_path, connect=connect, now=lambda: datetime(2026, 7, 20, 12, 0))


@run_async
async def test_sweep_rejects_an_existing_unregistered_root(tmp_path: Path) -> None:
    broad_root = tmp_path / "workspace"
    broad_root.mkdir()

    async def connect() -> ScriptedTransport:
        raise AssertionError("unregistered roots must not connect")

    with pytest.raises(NotificationStateError, match="registered managed research root"):
        await sweep_notifications(broad_root, connect=connect)


def test_failed_notification_requires_explicit_requeue(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)
    failed = event.with_delivery_failure(
        attempted_at=NOW,
        error="delivery failed",
        next_attempt_at=None,
        exhausted=True,
    )
    write_notification_event(failed, root)

    unchanged = ensure_notification(Path(event.terminal_state_path), root, requeue=False)
    assert unchanged.state == "failed"

    requeued = ensure_notification(Path(event.terminal_state_path), root, requeue=True)
    assert requeued.state == "pending"
    assert requeued.attempt_count == 0
    with pytest.raises(NotificationStateError, match="only failed"):
        ensure_notification(Path(event.terminal_state_path), root, requeue=True)


def test_notification_rejects_terminal_mismatch(tmp_path: Path) -> None:
    root, event = queued_notification(tmp_path)
    terminal_path = Path(event.terminal_state_path)
    terminal = json.loads(terminal_path.read_text())
    terminal["terminal_event_id"] = "22345678-1234-5678-9234-567812345678"
    terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        read_notification_event(terminal_path.with_name("notification.json"), root)


def test_registration_binds_marker_to_exact_root_and_migrates_legacy_marker(tmp_path: Path) -> None:
    root = tmp_path / "managed" / "logs"
    first = register_notification_root(root)
    second = register_notification_root(root)

    assert first.created
    assert not second.created
    assert validate_notification_root(root) == root.resolve()
    assert json.loads(first.marker.read_text(encoding="utf-8")) == {
        "kind": "mjepa-cifar10-managed-research-root",
        "root_path": str(root.resolve()),
        "schema_version": MANAGED_ROOT_SCHEMA_VERSION,
    }

    first.marker.write_text(
        json.dumps(
            {
                "schema_version": MANAGED_ROOT_SCHEMA_VERSION,
                "kind": "mjepa-cifar10-managed-research-root",
            }
        ),
        encoding="utf-8",
    )
    migrated = register_notification_root(root)
    assert not migrated.created
    assert json.loads(first.marker.read_text(encoding="utf-8"))["root_path"] == str(root.resolve())


@pytest.mark.parametrize("root", [Path("/"), Path("/tmp"), Path.home(), Path.cwd(), Path.cwd().parent])
def test_registration_rejects_broad_roots(root: Path) -> None:
    with pytest.raises(NotificationStateError, match=r"root|home|directory|repository|broad"):
        register_notification_root(root)


def test_registration_rejects_symlinked_roots_markers_and_repository_roots(tmp_path: Path) -> None:
    target = tmp_path / "target" / "logs"
    target.mkdir(parents=True)
    linked_root = tmp_path / "linked-logs"
    linked_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(NotificationStateError, match="symlink"):
        register_notification_root(linked_root)

    root = tmp_path / "managed" / "logs"
    registration = register_notification_root(root)
    outside_marker = tmp_path / "outside-marker.json"
    outside_marker.write_text(registration.marker.read_text(encoding="utf-8"), encoding="utf-8")
    registration.marker.unlink()
    registration.marker.symlink_to(outside_marker)
    with pytest.raises(NotificationStateError, match="symlink"):
        validate_notification_root(root)

    repository_root = tmp_path / "repository"
    (repository_root / ".git").mkdir(parents=True)
    with pytest.raises(NotificationStateError, match="repository root"):
        register_notification_root(repository_root)
