"""Deliver durable research terminal events to a persistent Codex app-server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from websockets.asyncio.client import ClientConnection, unix_connect

from .runtime import atomic_write_json


SCHEMA_VERSION = 1
TERMINAL_FILENAME = "terminal.json"
NOTIFICATION_FILENAME = "notification.json"
NOTIFICATION_LOCK_FILENAME = ".notification.lock"
MANAGED_ROOT_MARKER_FILENAME = ".mjepa-research-root.json"
MANAGED_ROOT_KIND = "mjepa-cifar10-managed-research-root"
MANAGED_ROOT_SCHEMA_VERSION = 1
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TERMINAL_STATUSES = frozenset({"completed", "failed", "crashed", "timed_out", "cancelled"})
DELIVERY_STATES = frozenset({"pending", "accepted", "failed"})
MAX_LAST_ERROR_LENGTH = 500
APP_SERVER_BASELINE = "0.144.5"
CLIENT_NAME = "mjepa_cifar10_autoresearch"
CLIENT_TITLE = "MJEPA CIFAR-10 Autoresearch"
CLIENT_VERSION = "1.0.0"
DEFAULT_REQUEST_TIMEOUT = 15.0
RETRY_BASE_SECONDS = 5.0
RETRY_FACTOR = 2.0
RETRY_CAP_SECONDS = 300.0
MAX_DELIVERY_ATTEMPTS = 8
SERVER_REQUEST_REJECTION_CODE = -32601
SERVER_REQUEST_REJECTION_MESSAGE = "This client does not handle server requests"
CONTROL_CHARACTERS = re.compile(r"[\x00-\x1f\x7f]+")

TerminalStatus = Literal["completed", "failed", "crashed", "timed_out", "cancelled"]
DeliveryState = Literal["pending", "accepted", "failed"]
JsonObject = dict[str, Any]


class NotificationStateError(ValueError):
    """Persisted notification state violates the schema or managed-path contract."""


class AppServerProtocolError(RuntimeError):
    """A connection or protocol outcome was not accepted by app-server."""

    def __init__(self, message: str, *, permanent: bool = False) -> None:
        super().__init__(message)
        self.permanent = permanent


@dataclass(frozen=True, slots=True)
class ManagedRootRegistration:
    root: Path
    marker: Path
    created: bool

    def to_dict(self) -> dict[str, object]:
        return {"root": str(self.root), "marker": str(self.marker), "created": self.created}


def _validate_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER_PATTERN.fullmatch(value):
        raise NotificationStateError(f"{field_name} is not a safe identifier: {value!r}")
    return value


def _validate_thread_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 512:
        raise NotificationStateError("originating_thread_id must be a non-empty string or null")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise NotificationStateError("originating_thread_id must not contain control characters")
    return value


def _validate_event_id(value: object) -> str:
    if not isinstance(value, str):
        raise NotificationStateError("event_id must be a UUID string")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise NotificationStateError("event_id must be a UUID string") from error
    if str(parsed) != value.lower():
        raise NotificationStateError("event_id must use canonical UUID text")
    return value.lower()


def _validate_attempt(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise NotificationStateError("attempt must be a positive integer")
    return value


def _parse_datetime(value: object, field_name: str, *, optional: bool = False) -> datetime | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise NotificationStateError(f"{field_name} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise NotificationStateError(f"{field_name} must be an ISO 8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NotificationStateError(f"{field_name} must include a UTC offset")
    return parsed.astimezone(UTC)


def _isoformat(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


def _managed_path(path: Path, root: Path, field_name: str) -> Path:
    managed_root = root.expanduser().resolve(strict=False)
    if managed_root == Path(managed_root.anchor):
        raise NotificationStateError("notification root must not be a filesystem root")
    resolved = path.expanduser().resolve(strict=False)
    if resolved == managed_root or not resolved.is_relative_to(managed_root):
        raise NotificationStateError(f"{field_name} must remain inside the managed root {managed_root}")
    return resolved


def _load_json(path: Path) -> JsonObject:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NotificationStateError(f"state in {path} is not valid JSON: {error}") from error
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise NotificationStateError(f"state in {path} must be a JSON object")
    return cast(JsonObject, value)


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(path.expanduser()))


def _validate_root_location(root: Path) -> Path:
    lexical = _lexical_absolute(root)
    managed_root = lexical.resolve(strict=False)
    if lexical != managed_root:
        raise NotificationStateError("notification root must not contain symlink path components")
    if managed_root == Path(managed_root.anchor):
        raise NotificationStateError("notification root must not be a filesystem root")
    if managed_root.parent == Path(managed_root.anchor):
        raise NotificationStateError("notification root must not be a top-level directory")

    home = Path.home().resolve(strict=False)
    if managed_root == home or home.is_relative_to(managed_root):
        raise NotificationStateError("notification root must not be a home directory or its parent")

    current = Path.cwd().resolve(strict=False)
    if managed_root == current or current.is_relative_to(managed_root):
        if (managed_root / ".git").exists():
            raise NotificationStateError("notification root must not be a repository root")
        raise NotificationStateError("notification root must not be a broad working-directory parent")
    if (managed_root / ".git").exists():
        raise NotificationStateError("notification root must not be a repository root")
    if managed_root.exists() and not managed_root.is_dir():
        raise NotificationStateError("notification root must be a directory")
    return managed_root


def _root_marker_payload(managed_root: Path) -> dict[str, object]:
    return {
        "schema_version": MANAGED_ROOT_SCHEMA_VERSION,
        "kind": MANAGED_ROOT_KIND,
        "root_path": str(managed_root),
    }


def register_notification_root(root: Path) -> ManagedRootRegistration:
    """Atomically register one exact safe root, migrating the legacy marker when present."""
    managed_root = _validate_root_location(root)
    managed_root.mkdir(parents=True, exist_ok=True)
    managed_root = _validate_root_location(managed_root)
    marker_path = managed_root / MANAGED_ROOT_MARKER_FILENAME
    expected = _root_marker_payload(managed_root)
    legacy = {"schema_version": MANAGED_ROOT_SCHEMA_VERSION, "kind": MANAGED_ROOT_KIND}
    if marker_path.is_symlink():
        raise NotificationStateError("managed research root marker must not be a symlink")
    if marker_path.exists():
        if not marker_path.is_file():
            raise NotificationStateError(f"managed research root marker must be a file: {marker_path}")
        payload = _load_json(marker_path)
        if payload == legacy:
            atomic_write_json(marker_path, expected)
        elif payload != expected:
            raise NotificationStateError(f"managed research root marker is invalid: {marker_path}")
        return ManagedRootRegistration(managed_root, marker_path, created=False)
    atomic_write_json(marker_path, expected)
    return ManagedRootRegistration(managed_root, marker_path, created=True)


def initialize_notification_root(root: Path) -> Path:
    """Register one exact managed root before creating study state or events."""
    return register_notification_root(root).root


def validate_notification_root(root: Path) -> Path:
    managed_root = _validate_root_location(root)
    marker_path = managed_root / MANAGED_ROOT_MARKER_FILENAME
    if marker_path.is_symlink():
        raise NotificationStateError("managed research root marker must not be a symlink")
    if not marker_path.is_file():
        raise NotificationStateError(f"notification root is not a registered managed research root: {managed_root}")
    expected = _root_marker_payload(managed_root)
    if _load_json(marker_path) != expected:
        raise NotificationStateError(f"managed research root marker is invalid: {marker_path}")
    return managed_root


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    event_id: str
    study_id: str
    run_id: str
    attempt: int
    status: TerminalStatus
    occurred_at: datetime
    originating_thread_id: str | None
    terminal_state_path: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _validate_event_id(self.event_id))
        object.__setattr__(self, "study_id", _validate_identifier(self.study_id, "study id"))
        object.__setattr__(self, "run_id", _validate_identifier(self.run_id, "run id"))
        object.__setattr__(self, "attempt", _validate_attempt(self.attempt))
        if self.status not in TERMINAL_STATUSES:
            raise NotificationStateError(f"invalid terminal status: {self.status!r}")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise NotificationStateError("occurred_at must include a UTC offset")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))
        object.__setattr__(self, "originating_thread_id", _validate_thread_id(self.originating_thread_id))
        if not isinstance(self.terminal_state_path, str):
            raise NotificationStateError("terminal_state_path must be a string")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "study_id": self.study_id,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "status": self.status,
            "occurred_at": _isoformat(self.occurred_at),
            "originating_thread_id": self.originating_thread_id,
            "terminal_state_path": self.terminal_state_path,
        }


@dataclass(frozen=True, slots=True)
class NotificationEvent:
    event_id: str
    study_id: str
    run_id: str
    attempt: int
    status: TerminalStatus
    occurred_at: datetime
    originating_thread_id: str | None
    terminal_state_path: str
    state: DeliveryState = "pending"
    attempt_count: int = 0
    last_attempt_at: datetime | None = None
    next_attempt_at: datetime | None = None
    last_error: str | None = None
    accepted_at: datetime | None = None
    accepted_rpc_method: str | None = None
    accepted_turn_id: str | None = None

    def __post_init__(self) -> None:
        terminal = self.as_terminal()
        if self.state not in DELIVERY_STATES:
            raise NotificationStateError(f"invalid delivery state: {self.state!r}")
        if not isinstance(self.attempt_count, int) or isinstance(self.attempt_count, bool) or self.attempt_count < 0:
            raise NotificationStateError("attempt_count must be a non-negative integer")
        for field_name in ("last_attempt_at", "next_attempt_at", "accepted_at"):
            value = getattr(self, field_name)
            if value is not None:
                if value.tzinfo is None or value.utcoffset() is None:
                    raise NotificationStateError(f"{field_name} must include a UTC offset")
                object.__setattr__(self, field_name, value.astimezone(UTC))
        if self.last_error is not None and (
            not self.last_error
            or len(self.last_error) > MAX_LAST_ERROR_LENGTH
            or CONTROL_CHARACTERS.search(self.last_error)
        ):
            raise NotificationStateError("last_error must be sanitized and at most 500 characters")
        if self.state == "accepted":
            if (
                self.accepted_at is None
                or self.accepted_rpc_method not in {"turn/start", "turn/steer"}
                or not isinstance(self.accepted_turn_id, str)
                or not self.accepted_turn_id
                or self.last_attempt_at != self.accepted_at
                or self.next_attempt_at is not None
                or self.last_error is not None
            ):
                raise NotificationStateError("accepted notification has inconsistent delivery metadata")
        elif any(value is not None for value in (self.accepted_at, self.accepted_rpc_method, self.accepted_turn_id)):
            raise NotificationStateError("unaccepted notification must not contain acceptance metadata")
        if self.state == "failed" and (
            self.attempt_count < 1
            or self.last_attempt_at is None
            or self.next_attempt_at is not None
            or not self.last_error
        ):
            raise NotificationStateError("failed notification has inconsistent delivery metadata")
        if self.state == "pending":
            retry_metadata = (self.last_attempt_at, self.next_attempt_at, self.last_error)
            if self.attempt_count == 0 and any(value is not None for value in retry_metadata):
                raise NotificationStateError("new pending notification must not contain retry metadata")
            if self.attempt_count > 0 and any(value is None for value in retry_metadata):
                raise NotificationStateError("retried pending notification has incomplete retry metadata")
        del terminal

    @classmethod
    def from_terminal(cls, terminal: TerminalEvent) -> NotificationEvent:
        return cls(
            event_id=terminal.event_id,
            study_id=terminal.study_id,
            run_id=terminal.run_id,
            attempt=terminal.attempt,
            status=terminal.status,
            occurred_at=terminal.occurred_at,
            originating_thread_id=terminal.originating_thread_id,
            terminal_state_path=terminal.terminal_state_path,
        )

    def as_terminal(self) -> TerminalEvent:
        return TerminalEvent(
            event_id=self.event_id,
            study_id=self.study_id,
            run_id=self.run_id,
            attempt=self.attempt,
            status=self.status,
            occurred_at=self.occurred_at,
            originating_thread_id=self.originating_thread_id,
            terminal_state_path=self.terminal_state_path,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **self.as_terminal().to_dict(),
            "state": self.state,
            "attempt_count": self.attempt_count,
            "last_attempt_at": _isoformat(self.last_attempt_at),
            "next_attempt_at": _isoformat(self.next_attempt_at),
            "last_error": self.last_error,
            "accepted_at": _isoformat(self.accepted_at),
            "accepted_rpc_method": self.accepted_rpc_method,
            "accepted_turn_id": self.accepted_turn_id,
        }

    @classmethod
    def from_dict(cls, payload: JsonObject) -> NotificationEvent:
        expected = {
            "schema_version",
            "event_id",
            "study_id",
            "run_id",
            "attempt",
            "status",
            "occurred_at",
            "originating_thread_id",
            "terminal_state_path",
            "state",
            "attempt_count",
            "last_attempt_at",
            "next_attempt_at",
            "last_error",
            "accepted_at",
            "accepted_rpc_method",
            "accepted_turn_id",
        }
        if set(payload) != expected or payload.get("schema_version") != SCHEMA_VERSION:
            raise NotificationStateError("notification state has invalid fields or schema version")
        status = payload["status"]
        state = payload["state"]
        if not isinstance(status, str) or status not in TERMINAL_STATUSES:
            raise NotificationStateError(f"invalid terminal status: {status!r}")
        if not isinstance(state, str) or state not in DELIVERY_STATES:
            raise NotificationStateError(f"invalid delivery state: {state!r}")
        occurred_at = _parse_datetime(payload["occurred_at"], "occurred_at")
        assert occurred_at is not None
        return cls(
            event_id=cast(str, payload["event_id"]),
            study_id=cast(str, payload["study_id"]),
            run_id=cast(str, payload["run_id"]),
            attempt=cast(int, payload["attempt"]),
            status=cast(TerminalStatus, status),
            occurred_at=occurred_at,
            originating_thread_id=cast(str | None, payload["originating_thread_id"]),
            terminal_state_path=cast(str, payload["terminal_state_path"]),
            state=cast(DeliveryState, state),
            attempt_count=cast(int, payload["attempt_count"]),
            last_attempt_at=_parse_datetime(payload["last_attempt_at"], "last_attempt_at", optional=True),
            next_attempt_at=_parse_datetime(payload["next_attempt_at"], "next_attempt_at", optional=True),
            last_error=cast(str | None, payload["last_error"]),
            accepted_at=_parse_datetime(payload["accepted_at"], "accepted_at", optional=True),
            accepted_rpc_method=cast(str | None, payload["accepted_rpc_method"]),
            accepted_turn_id=cast(str | None, payload["accepted_turn_id"]),
        )

    def with_delivery_failure(
        self,
        *,
        attempted_at: datetime,
        error: str,
        next_attempt_at: datetime | None,
        exhausted: bool,
    ) -> NotificationEvent:
        return replace(
            self,
            state="failed" if exhausted else "pending",
            attempt_count=self.attempt_count + 1,
            last_attempt_at=attempted_at,
            next_attempt_at=None if exhausted else next_attempt_at,
            last_error=error,
            accepted_at=None,
            accepted_rpc_method=None,
            accepted_turn_id=None,
        )

    def with_acceptance(self, *, accepted_at: datetime, rpc_method: str, turn_id: str) -> NotificationEvent:
        return replace(
            self,
            state="accepted",
            attempt_count=self.attempt_count + 1,
            last_attempt_at=accepted_at,
            next_attempt_at=None,
            last_error=None,
            accepted_at=accepted_at,
            accepted_rpc_method=rpc_method,
            accepted_turn_id=turn_id,
        )

    def requeued(self) -> NotificationEvent:
        if self.state != "failed":
            raise NotificationStateError("only failed notifications can be requeued")
        return replace(
            self,
            state="pending",
            attempt_count=0,
            last_attempt_at=None,
            next_attempt_at=None,
            last_error=None,
            accepted_at=None,
            accepted_rpc_method=None,
            accepted_turn_id=None,
        )


def read_terminal_event(path: Path, root: Path) -> TerminalEvent:
    resolved = _managed_path(path.absolute(), root, "terminal state path")
    if resolved.name != TERMINAL_FILENAME or resolved.parent.parent.name != "runs":
        raise NotificationStateError("terminal state path is not an exact managed run terminal")
    payload = _load_json(resolved)
    status = payload.get("status")
    if not isinstance(status, str) or status not in TERMINAL_STATUSES:
        raise NotificationStateError(f"invalid terminal status: {status!r}")
    occurred_at = _parse_datetime(payload.get("finished_at"), "finished_at")
    assert occurred_at is not None
    return TerminalEvent(
        event_id=_validate_event_id(payload.get("terminal_event_id")),
        study_id=resolved.parent.parent.parent.name,
        run_id=resolved.parent.name,
        attempt=_validate_attempt(payload.get("attempt")),
        status=cast(TerminalStatus, status),
        occurred_at=occurred_at,
        originating_thread_id=_validate_thread_id(payload.get("originating_thread_id")),
        terminal_state_path=str(resolved),
    )


def read_notification_event(path: Path, root: Path) -> NotificationEvent:
    resolved = _managed_path(path.absolute(), root, "notification path")
    if resolved.name != NOTIFICATION_FILENAME or resolved.parent.parent.name != "runs":
        raise NotificationStateError("notification path is not an exact managed run notification")
    event = NotificationEvent.from_dict(_load_json(resolved))
    terminal = read_terminal_event(resolved.with_name(TERMINAL_FILENAME), root)
    if event.as_terminal() != terminal:
        raise NotificationStateError(f"notification in {resolved} does not match terminal state")
    return event


def write_notification_event(event: NotificationEvent, root: Path) -> None:
    terminal_path = _managed_path(Path(event.terminal_state_path), root, "terminal state path")
    atomic_write_json(terminal_path.with_name(NOTIFICATION_FILENAME), event.to_dict())


def queue_notification_from_terminal(
    terminal_path: Path,
    root: Path,
    *,
    study_id: str,
    run_id: str,
) -> NotificationEvent:
    """Create or recover a pending notification after terminal state is durable."""
    terminal = read_terminal_event(terminal_path, root)
    if terminal.study_id != _validate_identifier(study_id, "study id") or terminal.run_id != _validate_identifier(
        run_id, "run id"
    ):
        raise NotificationStateError("terminal identifiers do not match the requested run")
    resolved_terminal_path = Path(terminal.terminal_state_path)
    notification_path = resolved_terminal_path.with_name(NOTIFICATION_FILENAME)
    with FileLock(str(resolved_terminal_path.parent / NOTIFICATION_LOCK_FILENAME)):
        if notification_path.is_file():
            current = read_notification_event(notification_path, root)
            if current.event_id != terminal.event_id:
                raise NotificationStateError("current notification belongs to a different terminal event")
            return current
        event = NotificationEvent.from_terminal(terminal)
        write_notification_event(event, root)
        return event


def ensure_notification(terminal_path: Path, root: Path, *, requeue: bool = False) -> NotificationEvent:
    terminal = read_terminal_event(terminal_path, root)
    event = queue_notification_from_terminal(
        terminal_path,
        root,
        study_id=terminal.study_id,
        run_id=terminal.run_id,
    )
    if not requeue:
        return event
    resolved_terminal_path = Path(terminal.terminal_state_path)
    with FileLock(str(resolved_terminal_path.parent / NOTIFICATION_LOCK_FILENAME)):
        current = read_notification_event(resolved_terminal_path.with_name(NOTIFICATION_FILENAME), root)
        requeued = current.requeued()
        write_notification_event(requeued, root)
        return requeued


class MessageTransport(Protocol):
    async def send(self, message: JsonObject) -> None: ...

    async def receive(self) -> JsonObject: ...

    async def close(self) -> None: ...


class JsonlStdioTransport:
    """JSONL transport through ``codex app-server proxy`` to an existing daemon."""

    command: tuple[str, ...] = ("codex", "app-server", "proxy")

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self._process = process

    @classmethod
    async def connect(cls, socket_path: Path | None = None) -> JsonlStdioTransport:
        command = [*cls.command]
        if socket_path is not None:
            command.extend(("--sock", str(socket_path.expanduser().resolve(strict=False))))
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as error:
            raise AppServerProtocolError(f"could not start app-server proxy: {error}") from error
        if process.stdin is None or process.stdout is None:
            process.kill()
            await process.wait()
            raise AppServerProtocolError("app-server proxy did not expose stdin and stdout")
        return cls(process)

    async def send(self, message: JsonObject) -> None:
        stream = self._process.stdin
        if stream is None or stream.is_closing():
            raise AppServerProtocolError("app-server proxy stdin is closed")
        try:
            stream.write(json.dumps(message, separators=(",", ":")).encode() + b"\n")
            await stream.drain()
        except (BrokenPipeError, ConnectionError, OSError) as error:
            raise AppServerProtocolError(f"could not write to app-server proxy: {error}") from error

    async def receive(self) -> JsonObject:
        stream = self._process.stdout
        if stream is None:
            raise AppServerProtocolError("app-server proxy stdout is unavailable")
        line = await stream.readline()
        if not line:
            detail = ""
            if self._process.stderr is not None:
                try:
                    raw_error = await asyncio.wait_for(self._process.stderr.read(2048), timeout=0.1)
                    detail = raw_error.decode(errors="replace").strip()
                except TimeoutError:
                    pass
            raise AppServerProtocolError(f"app-server proxy closed the connection{f': {detail}' if detail else ''}")
        return _decode_message(line)

    async def close(self) -> None:
        stream = self._process.stdin
        if stream is not None and not stream.is_closing():
            stream.close()
            with suppress(BrokenPipeError, ConnectionError):
                await stream.wait_closed()
        try:
            await asyncio.wait_for(self._process.wait(), timeout=1.0)
        except TimeoutError:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=1.0)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()


class UnixWebSocketTransport:
    """One-JSON-message-per-frame transport over a local Unix socket."""

    def __init__(self, connection: ClientConnection) -> None:
        self._connection = connection

    @classmethod
    async def connect(cls, socket_path: Path) -> UnixWebSocketTransport:
        path = socket_path.expanduser().resolve(strict=False)
        try:
            connection = await unix_connect(path=str(path), uri="ws://localhost")
        except (OSError, ValueError) as error:
            raise AppServerProtocolError(f"could not connect to app-server Unix socket {path}: {error}") from error
        return cls(connection)

    async def send(self, message: JsonObject) -> None:
        try:
            await self._connection.send(json.dumps(message, separators=(",", ":")))
        except (ConnectionError, OSError) as error:
            raise AppServerProtocolError(f"could not write to app-server socket: {error}") from error

    async def receive(self) -> JsonObject:
        try:
            return _decode_message(await self._connection.recv())
        except (ConnectionError, OSError) as error:
            raise AppServerProtocolError(f"app-server socket read failed: {error}") from error

    async def close(self) -> None:
        await self._connection.close()


def _decode_message(message: str | bytes) -> JsonObject:
    try:
        decoded = json.loads(message)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AppServerProtocolError(f"app-server returned invalid JSON: {error}") from error
    if not isinstance(decoded, dict) or not all(isinstance(key, str) for key in decoded):
        raise AppServerProtocolError("app-server message must be a JSON object")
    return cast(JsonObject, decoded)


class RpcClient:
    def __init__(self, transport: MessageTransport, *, request_timeout: float) -> None:
        self._transport = transport
        self._request_timeout = request_timeout
        self._next_request_id = 1
        self._pending: dict[int, asyncio.Future[JsonObject]] = {}
        self._reader: asyncio.Task[None] | None = None
        self._send_lock = asyncio.Lock()
        self._reader_error: AppServerProtocolError | None = None

    async def __aenter__(self) -> RpcClient:
        self._reader = asyncio.create_task(self._read_messages())
        return self

    async def __aexit__(self, *_args: object) -> None:
        if self._reader is not None:
            self._reader.cancel()
            with suppress(asyncio.CancelledError):
                await self._reader
        await self._transport.close()

    async def request(self, method: str, params: JsonObject) -> JsonObject:
        if self._reader_error is not None:
            raise self._reader_error
        request_id = self._next_request_id
        self._next_request_id += 1
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            await self._send({"id": request_id, "method": method, "params": params})
            response = await asyncio.wait_for(future, timeout=self._request_timeout)
        except TimeoutError as error:
            raise AppServerProtocolError(f"{method} timed out") from error
        finally:
            self._pending.pop(request_id, None)
        if "error" in response:
            rpc_error = response["error"]
            code = rpc_error.get("code", "unknown") if isinstance(rpc_error, dict) else "unknown"
            message = rpc_error.get("message", rpc_error) if isinstance(rpc_error, dict) else rpc_error
            raise AppServerProtocolError(f"{method} failed ({code}): {message}")
        result = response.get("result")
        if not isinstance(result, dict):
            raise AppServerProtocolError(f"{method} returned a non-object result")
        return cast(JsonObject, result)

    async def notify(self, method: str, params: JsonObject) -> None:
        await self._send({"method": method, "params": params})

    async def _send(self, message: JsonObject) -> None:
        async with self._send_lock:
            await self._transport.send(message)

    async def _read_messages(self) -> None:
        try:
            while True:
                message = await self._transport.receive()
                message_id = message.get("id")
                method = message.get("method")
                if message_id is not None and isinstance(method, str):
                    await self._send(
                        {
                            "id": message_id,
                            "error": {
                                "code": SERVER_REQUEST_REJECTION_CODE,
                                "message": SERVER_REQUEST_REJECTION_MESSAGE,
                            },
                        }
                    )
                    continue
                if isinstance(message_id, int):
                    future = self._pending.get(message_id)
                    if future is not None and not future.done():
                        future.set_result(message)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            protocol_error = (
                error
                if isinstance(error, AppServerProtocolError)
                else AppServerProtocolError(f"app-server dispatcher failed: {error}")
            )
            self._reader_error = protocol_error
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(protocol_error)


@dataclass(frozen=True, slots=True)
class Acceptance:
    rpc_method: str
    turn_id: str


def build_wake_prompt(event: NotificationEvent) -> str:
    return (
        "Research run completed.\n"
        f"Study: {event.study_id}\n"
        f"Run: {event.run_id}\n"
        f"Status: {event.status}\n"
        f"Terminal state: {event.terminal_state_path}\n\n"
        "Inspect the terminal state and continue the study protocol."
    )


def _thread_from_result(result: JsonObject, method: str, expected_thread_id: str) -> JsonObject:
    thread = result.get("thread")
    if not isinstance(thread, dict) or thread.get("id") != expected_thread_id:
        raise AppServerProtocolError(f"{method} response is missing or returned an unexpected thread")
    return cast(JsonObject, thread)


async def deliver_notification(
    event: NotificationEvent,
    transport: MessageTransport,
    *,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> Acceptance:
    thread_id = event.originating_thread_id
    if thread_id is None:
        await transport.close()
        raise AppServerProtocolError("notification has no originating Codex thread ID", permanent=True)
    if event.state != "pending":
        await transport.close()
        raise AppServerProtocolError("only pending notifications can be delivered", permanent=True)
    async with RpcClient(transport, request_timeout=request_timeout) as client:
        await client.request(
            "initialize",
            {"clientInfo": {"name": CLIENT_NAME, "title": CLIENT_TITLE, "version": CLIENT_VERSION}},
        )
        await client.notify("initialized", {})
        resumed = await client.request("thread/resume", {"threadId": thread_id})
        _thread_from_result(resumed, "thread/resume", thread_id)
        fresh = await client.request("thread/read", {"threadId": thread_id, "includeTurns": True})
        thread = _thread_from_result(fresh, "thread/read", thread_id)
        status = thread.get("status")
        turns = thread.get("turns")
        if not isinstance(status, dict) or not isinstance(status.get("type"), str):
            raise AppServerProtocolError("thread/read returned an unknown thread status")
        if not isinstance(turns, list):
            raise AppServerProtocolError("thread/read response is missing turns")
        in_progress = [
            turn
            for turn in turns
            if isinstance(turn, dict) and turn.get("status") == "inProgress" and isinstance(turn.get("id"), str)
        ]
        input_items = [{"type": "text", "text": build_wake_prompt(event)}]
        if status["type"] == "idle":
            if in_progress:
                raise AppServerProtocolError("thread status changed while preparing turn/start")
            result = await client.request(
                "turn/start",
                {"threadId": thread_id, "input": input_items, "clientUserMessageId": event.event_id},
            )
            turn = result.get("turn")
            if not isinstance(turn, dict) or not isinstance(turn.get("id"), str):
                raise AppServerProtocolError("turn/start response is missing the accepted turn ID")
            return Acceptance("turn/start", turn["id"])
        if status["type"] == "active":
            if len(in_progress) != 1:
                raise AppServerProtocolError("active thread does not have exactly one steerable in-progress turn")
            expected_turn_id = cast(str, in_progress[0]["id"])
            result = await client.request(
                "turn/steer",
                {
                    "threadId": thread_id,
                    "input": input_items,
                    "expectedTurnId": expected_turn_id,
                    "clientUserMessageId": event.event_id,
                },
            )
            if result.get("turnId") != expected_turn_id:
                raise AppServerProtocolError("turn/steer returned an unexpected turn ID")
            return Acceptance("turn/steer", expected_turn_id)
        raise AppServerProtocolError(f"thread is not deliverable in state {status['type']!r}")


@dataclass(frozen=True, slots=True)
class SweepResult:
    discovered: int = 0
    due: int = 0
    accepted: int = 0
    retrying: int = 0
    failed: int = 0
    skipped: int = 0
    problems: tuple[str, ...] = ()

    @property
    def exit_code(self) -> int:
        return 1 if self.failed or self.retrying or self.problems else 0

    def to_dict(self) -> dict[str, object]:
        return {
            "discovered": self.discovered,
            "due": self.due,
            "accepted": self.accepted,
            "retrying": self.retrying,
            "failed": self.failed,
            "skipped": self.skipped,
            "problems": list(self.problems),
        }


def notification_lock_path(root: Path, thread_id: str) -> Path:
    digest = hashlib.sha256(thread_id.encode()).hexdigest()
    return root.expanduser().resolve(strict=False) / ".notification-locks" / f"{digest}.lock"


def _accepted_ledger_path(lock_path: Path) -> Path:
    return lock_path.with_suffix(".accepted.json")


def _read_accepted_ledger(path: Path, thread_id: str) -> dict[str, JsonObject]:
    if not path.exists():
        return {}
    payload = _load_json(path)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("thread_id") != thread_id
        or not isinstance(payload.get("events"), dict)
    ):
        raise NotificationStateError(f"accepted-event ledger is invalid: {path}")
    events = payload["events"]
    if not all(isinstance(key, str) and isinstance(value, dict) for key, value in events.items()):
        raise NotificationStateError(f"accepted-event ledger contains invalid entries: {path}")
    return cast(dict[str, JsonObject], events)


def _write_accepted_ledger(path: Path, thread_id: str, events: dict[str, JsonObject]) -> None:
    atomic_write_json(path, {"schema_version": SCHEMA_VERSION, "thread_id": thread_id, "events": events})


def _sanitize_error(error: BaseException) -> str:
    text = " ".join(CONTROL_CHARACTERS.sub(" ", str(error)).split())
    return (text or error.__class__.__name__)[:MAX_LAST_ERROR_LENGTH]


def _is_due(event: NotificationEvent, now: datetime) -> bool:
    return event.state == "pending" and (event.next_attempt_at is None or event.next_attempt_at <= now)


@asynccontextmanager
async def _async_file_lock(path: Path) -> AsyncIterator[None]:
    lock = FileLock(str(path), thread_local=False)
    while True:
        try:
            await asyncio.to_thread(lock.acquire, timeout=0)
            break
        except FileLockTimeout:
            await asyncio.sleep(0.01)
    try:
        yield
    finally:
        lock.release()


async def _deliver_path(
    path: Path,
    root: Path,
    *,
    connect: Callable[[], Awaitable[MessageTransport]],
    now: datetime,
    random: Random,
    request_timeout: float,
) -> tuple[str, str | None]:
    try:
        initial = read_notification_event(path, root)
    except (OSError, NotificationStateError) as error:
        problem = f"{path}: {_sanitize_error(error)}"
        try:
            terminal_path = path.with_name(TERMINAL_FILENAME)
            terminal = read_terminal_event(terminal_path, root)
            if Path(terminal.terminal_state_path) != terminal_path.resolve(strict=False):
                raise NotificationStateError("terminal_state_path does not identify the current terminal file")
            failed = NotificationEvent.from_terminal(terminal).with_delivery_failure(
                attempted_at=now,
                error=_sanitize_error(error),
                next_attempt_at=None,
                exhausted=True,
            )
            with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
                write_notification_event(failed, root)
        except (OSError, NotificationStateError):
            return "failed", problem
        return "failed", problem
    if initial.state == "accepted":
        return "skipped", None
    if initial.state == "failed":
        return "failed", f"{path}: notification requires explicit requeue"
    if not _is_due(initial, now):
        return "skipped", None
    if initial.originating_thread_id is None:
        failed = initial.with_delivery_failure(
            attempted_at=now,
            error="notification has no originating Codex thread ID",
            next_attempt_at=None,
            exhausted=True,
        )
        with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
            write_notification_event(failed, root)
        return "failed", f"{path}: notification has no originating Codex thread ID"
    thread_id = initial.originating_thread_id
    lock_path = notification_lock_path(root, thread_id)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    async with _async_file_lock(lock_path):
        try:
            with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
                event = read_notification_event(path, root)
            if event.state != "pending" or not _is_due(event, now):
                return "skipped", None
            ledger_path = _accepted_ledger_path(lock_path)
            ledger = _read_accepted_ledger(ledger_path, thread_id)
            prior = ledger.get(event.event_id)
            if prior is not None:
                accepted_at = _parse_datetime(prior.get("accepted_at"), "accepted_at")
                rpc_method = prior.get("rpc_method")
                turn_id = prior.get("turn_id")
                if accepted_at is None or not isinstance(rpc_method, str) or not isinstance(turn_id, str):
                    raise NotificationStateError("accepted-event ledger entry is invalid")
                accepted = replace(
                    event,
                    state="accepted",
                    last_attempt_at=accepted_at,
                    next_attempt_at=None,
                    last_error=None,
                    accepted_at=accepted_at,
                    accepted_rpc_method=rpc_method,
                    accepted_turn_id=turn_id,
                )
                with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
                    write_notification_event(accepted, root)
                return "accepted", None
            acceptance = await deliver_notification(event, await connect(), request_timeout=request_timeout)
            accepted = event.with_acceptance(
                accepted_at=now,
                rpc_method=acceptance.rpc_method,
                turn_id=acceptance.turn_id,
            )
            ledger[event.event_id] = {
                "accepted_at": now.isoformat(),
                "rpc_method": acceptance.rpc_method,
                "turn_id": acceptance.turn_id,
            }
            _write_accepted_ledger(ledger_path, thread_id, ledger)
            with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
                write_notification_event(accepted, root)
            return "accepted", None
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            current = read_notification_event(path, root)
            attempt_count = current.attempt_count + 1
            permanent = isinstance(error, NotificationStateError) or (
                isinstance(error, AppServerProtocolError) and error.permanent
            )
            exhausted = permanent or attempt_count >= MAX_DELIVERY_ATTEMPTS
            delay_cap = min(RETRY_CAP_SECONDS, RETRY_BASE_SECONDS * (RETRY_FACTOR**current.attempt_count))
            next_attempt_at = None if exhausted else now + timedelta(seconds=random.uniform(0.0, delay_cap))
            updated = current.with_delivery_failure(
                attempted_at=now,
                error=_sanitize_error(error),
                next_attempt_at=next_attempt_at,
                exhausted=exhausted,
            )
            with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
                write_notification_event(updated, root)
            return ("failed" if exhausted else "retrying"), f"{path}: {updated.last_error}"


def _notification_paths(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob(NOTIFICATION_FILENAME)
        if path.parent.parent.name == "runs" and "attempts" not in path.parts
    )


async def sweep_notifications(
    root: Path,
    *,
    connect: Callable[[], Awaitable[MessageTransport]],
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    random: Random | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> SweepResult:
    managed_root = root.expanduser().resolve(strict=False)
    if managed_root == Path(managed_root.anchor):
        raise NotificationStateError("notification root must not be a filesystem root")
    if not managed_root.exists():
        return SweepResult()
    managed_root = validate_notification_root(managed_root)
    selected_now = now()
    if selected_now.tzinfo is None or selected_now.utcoffset() is None:
        raise NotificationStateError("worker clock must return an offset-aware datetime")
    selected_now = selected_now.astimezone(UTC)
    generator = random or Random()
    paths = _notification_paths(managed_root)
    counts = {"accepted": 0, "retrying": 0, "failed": 0, "skipped": 0}
    due = 0
    problems: list[str] = []
    for path in paths:
        try:
            if _is_due(read_notification_event(path, managed_root), selected_now):
                due += 1
        except (OSError, NotificationStateError):
            due += 1
        outcome, problem = await _deliver_path(
            path,
            managed_root,
            connect=connect,
            now=selected_now,
            random=generator,
            request_timeout=request_timeout,
        )
        counts[outcome] += 1
        if problem is not None:
            problems.append(problem)
    return SweepResult(
        discovered=len(paths),
        due=due,
        accepted=counts["accepted"],
        retrying=counts["retrying"],
        failed=counts["failed"],
        skipped=counts["skipped"],
        problems=tuple(problems),
    )


def stdio_connector(socket_path: Path | None = None) -> Callable[[], Awaitable[MessageTransport]]:
    async def connect() -> MessageTransport:
        return await JsonlStdioTransport.connect(socket_path)

    return connect


def unix_connector(socket_path: Path) -> Callable[[], Awaitable[MessageTransport]]:
    async def connect() -> MessageTransport:
        return await UnixWebSocketTransport.connect(socket_path)

    return connect
