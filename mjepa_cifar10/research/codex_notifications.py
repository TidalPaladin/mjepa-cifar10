"""MJEPA event production and shared notify-wake v2 delivery."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from random import Random
from typing import Any, Literal, cast
from uuid import UUID

from filelock import FileLock
from filelock import Timeout as FileLockTimeout
from notify_wake import (
    AppServerError,
    DeliveryState,
    MessageTransport,
    NotifyWaitLease,
    UnixWebSocketTransport,
    WakeContext,
    WakeRequest,
    capture_wake_context,
    deliver_wake,
    enter_notify_wait,
    reconcile_wake,
)
from notify_wake.models import NotificationRecord, normalize_datetime

from .lifecycle_events import LifecycleEvent, LifecycleKind, read_lifecycle_event
from .runtime import atomic_write_json
from .wake_context import WAKE_CONTEXT_FILENAME, WakeContextValidationError


SCHEMA_VERSION = 2
NOTIFY_WAKE_DIRECTORY = ".notify-wake"
NOTIFY_WAKE_CONTRACT = "mjepa-cifar10-notify-wake-v2"
NOTIFY_WAKE_ROOT_MARKER = ".notify-wake-root.json"
TERMINAL_FILENAME = "terminal.json"
NOTIFICATION_FILENAME = "notification.json"
NOTIFICATION_LOCK_FILENAME = ".notification.lock"
MANAGED_ROOT_MARKER_FILENAME = ".mjepa-research-root.json"
MANAGED_ROOT_KIND = "mjepa-cifar10-managed-research-root"
MANAGED_ROOT_SCHEMA_VERSION = 1
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
TERMINAL_STATUSES = frozenset({"completed", "failed", "crashed", "timed_out", "cancelled"})
DELIVERY_STATES = frozenset({"pending", "in_flight", "uncertain", "retry_due", "accepted", "blocked"})
MAX_LAST_ERROR_LENGTH = 500
APP_SERVER_BASELINE = "0.146.0"
DEFAULT_REQUEST_TIMEOUT = 15.0
RETRY_BASE_SECONDS = 5.0
RETRY_FACTOR = 2.0
RETRY_CAP_SECONDS = 300.0
GOAL_WAIT_DIRECTORY = "goal-waits"
TERMINAL_CONTRACT_FIELD = "notify_wake_contract_version"

TerminalStatus = Literal["completed", "failed", "crashed", "timed_out", "cancelled"]
EventKind = Literal["terminal", "first_cycle_completed", "supervisor_lost", "progress_stalled"]
JsonObject = dict[str, Any]

LIFECYCLE_STATUS_BY_KIND: dict[LifecycleKind, str] = {
    "first_cycle_completed": "completed",
    "supervisor_lost": "detected",
    "progress_stalled": "detected",
}


class NotificationStateError(ValueError):
    """Persisted notification state violates the v2 managed-path contract."""


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


def _parse_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise NotificationStateError(f"{field_name} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise NotificationStateError(f"{field_name} must be an ISO 8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise NotificationStateError(f"{field_name} must include a UTC offset")
    return parsed.astimezone(UTC)


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


def notification_namespace(root: Path) -> Path:
    return root.expanduser().resolve(strict=False) / NOTIFY_WAKE_DIRECTORY / f"v{SCHEMA_VERSION}"


def _register_notification_namespace(root: Path) -> Path:
    namespace = notification_namespace(root)
    namespace.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(namespace, 0o700)
    marker = namespace / NOTIFY_WAKE_ROOT_MARKER
    expected: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "kind": NOTIFY_WAKE_CONTRACT,
        "root_path": str(namespace),
    }
    if marker.exists():
        payload = _load_json(marker)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise NotificationStateError("unsupported notify-wake contract; cutover required")
        if payload != expected:
            raise NotificationStateError("notify-wake root marker is invalid")
    else:
        atomic_write_json(marker, expected)
    return namespace


def _validate_notification_namespace(root: Path) -> Path:
    namespace = notification_namespace(root)
    marker = namespace / NOTIFY_WAKE_ROOT_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise NotificationStateError("version-2 notify-wake root is not registered")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "kind": NOTIFY_WAKE_CONTRACT,
        "root_path": str(namespace),
    }
    payload = _load_json(marker)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise NotificationStateError("unsupported notify-wake contract; cutover required")
    if payload != expected:
        raise NotificationStateError("notify-wake root marker is invalid")
    return namespace


def notification_path_for_event(root: Path, event_id: str) -> Path:
    return notification_namespace(root) / "events" / _validate_event_id(event_id) / NOTIFICATION_FILENAME


def _context_path(root: Path, study_id: str, run_id: str) -> Path:
    return (
        notification_namespace(root)
        / "contexts"
        / _validate_identifier(study_id, "study id")
        / _validate_identifier(run_id, "run id")
        / WAKE_CONTEXT_FILENAME
    )


def _root_marker_payload(managed_root: Path) -> dict[str, object]:
    return {
        "schema_version": MANAGED_ROOT_SCHEMA_VERSION,
        "kind": MANAGED_ROOT_KIND,
        "root_path": str(managed_root),
    }


def register_notification_root(root: Path) -> ManagedRootRegistration:
    """Register one exact root and its empty v2 notification namespace."""

    managed_root = _validate_root_location(root)
    managed_root.mkdir(parents=True, exist_ok=True)
    managed_root = _validate_root_location(managed_root)
    marker_path = managed_root / MANAGED_ROOT_MARKER_FILENAME
    expected = _root_marker_payload(managed_root)
    if marker_path.is_symlink():
        raise NotificationStateError("managed research root marker must not be a symlink")
    if marker_path.exists():
        if not marker_path.is_file():
            raise NotificationStateError(f"managed research root marker must be a file: {marker_path}")
        if _load_json(marker_path) != expected:
            raise NotificationStateError(f"managed research root marker is invalid: {marker_path}")
        created = False
    else:
        atomic_write_json(marker_path, expected)
        created = True
    _register_notification_namespace(managed_root)
    return ManagedRootRegistration(managed_root, marker_path, created=created)


def initialize_notification_root(root: Path) -> Path:
    return register_notification_root(root).root


def validate_notification_root(root: Path) -> Path:
    managed_root = _validate_root_location(root)
    marker_path = managed_root / MANAGED_ROOT_MARKER_FILENAME
    if marker_path.is_symlink():
        raise NotificationStateError("managed research root marker must not be a symlink")
    if not marker_path.is_file():
        raise NotificationStateError(f"notification root is not a registered managed research root: {managed_root}")
    if _load_json(marker_path) != _root_marker_payload(managed_root):
        raise NotificationStateError(f"managed research root marker is invalid: {marker_path}")
    _validate_notification_namespace(managed_root)
    return managed_root


def _run_identity(run_dir: Path, root: Path) -> tuple[Path, str, str]:
    managed = _managed_path(run_dir.absolute(), root, "managed run directory")
    if managed.parent.name != "runs":
        raise NotificationStateError("run directory must be <root>/<study>/runs/<run>")
    return (
        managed,
        _validate_identifier(managed.parent.parent.name, "study id"),
        _validate_identifier(
            managed.name,
            "run id",
        ),
    )


def _read_wake_context(root: Path, study_id: str, run_id: str) -> WakeContext | None:
    path = _context_path(root, study_id, run_id)
    if path.is_symlink():
        raise NotificationStateError("wake context must not be a symlink")
    if not path.exists():
        return None
    if not path.is_file():
        raise NotificationStateError("wake context must be a file")
    try:
        return WakeContext.from_dict(_load_json(path))
    except WakeContextValidationError as error:
        raise NotificationStateError(f"wake context in {path} is invalid: {error}") from error


def persist_wake_context(run_dir: Path, root: Path, context: WakeContext) -> Path:
    """Persist one immutable v2 authority context before run dispatch."""

    managed_root = validate_notification_root(root)
    managed_run_dir, study_id, run_id = _run_identity(run_dir, managed_root)
    managed_run_dir.mkdir(parents=True, exist_ok=True)
    context_path = _context_path(managed_root, study_id, run_id)
    with FileLock(str(context_path.parent / NOTIFICATION_LOCK_FILENAME)):
        current = _read_wake_context(managed_root, study_id, run_id)
        if current is not None:
            if (
                current.thread_id != context.thread_id
                or current.permission_profile != context.permission_profile
                or current.approval_policy != context.approval_policy
            ):
                raise NotificationStateError("managed run already has a different immutable wake context")
            return context_path
        atomic_write_json(context_path, context.to_dict())
    return context_path


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
        object.__setattr__(
            self,
            "occurred_at",
            normalize_datetime(self.occurred_at, "occurred_at"),
        )
        object.__setattr__(
            self,
            "originating_thread_id",
            _validate_thread_id(self.originating_thread_id),
        )
        if not isinstance(self.terminal_state_path, str):
            raise NotificationStateError("terminal_state_path must be a string")


EVENT_FIELDS = frozenset(
    {
        "schema_version",
        "event_id",
        "event_kind",
        "study_id",
        "run_id",
        "attempt",
        "status",
        "occurred_at",
        "originating_thread_id",
        "event_state_path",
        "delivery",
    }
)


@dataclass(frozen=True, slots=True)
class NotificationEvent:
    event_id: str
    event_kind: EventKind
    study_id: str
    run_id: str
    attempt: int
    status: str
    occurred_at: datetime
    originating_thread_id: str | None
    terminal_state_path: str
    delivery: NotificationRecord
    wake_context: WakeContext | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_id", _validate_event_id(self.event_id))
        object.__setattr__(self, "study_id", _validate_identifier(self.study_id, "study id"))
        object.__setattr__(self, "run_id", _validate_identifier(self.run_id, "run id"))
        object.__setattr__(self, "attempt", _validate_attempt(self.attempt))
        object.__setattr__(
            self,
            "occurred_at",
            normalize_datetime(self.occurred_at, "occurred_at"),
        )
        object.__setattr__(
            self,
            "originating_thread_id",
            _validate_thread_id(self.originating_thread_id),
        )
        if self.event_kind == "terminal":
            if self.status not in TERMINAL_STATUSES:
                raise NotificationStateError(f"invalid terminal status: {self.status!r}")
        elif self.event_kind not in LIFECYCLE_STATUS_BY_KIND:
            raise NotificationStateError(f"invalid event kind: {self.event_kind!r}")
        elif self.status != LIFECYCLE_STATUS_BY_KIND[cast(LifecycleKind, self.event_kind)]:
            raise NotificationStateError(f"invalid status for lifecycle event {self.event_kind!r}: {self.status!r}")
        if not isinstance(self.terminal_state_path, str) or not self.terminal_state_path:
            raise NotificationStateError("event state path must be a non-empty string")
        if self.delivery.schema_version != SCHEMA_VERSION:
            raise NotificationStateError("unsupported notify-wake contract; cutover required")
        if self.delivery.watch_id != self.event_id or self.delivery.event_id != self.event_id:
            raise NotificationStateError("delivery identity does not match the research event")
        if self.delivery.thread_id != self.originating_thread_id:
            raise NotificationStateError("delivery thread does not match the research event")
        if self.wake_context is not None and self.wake_context.thread_id != self.originating_thread_id:
            raise NotificationStateError("wake context thread does not match the originating thread")

    @property
    def state(self) -> str:
        return self.delivery.state

    @property
    def attempt_count(self) -> int:
        return self.delivery.attempt_count

    @property
    def last_attempt_at(self) -> datetime | None:
        return self.delivery.last_attempt_at

    @property
    def next_attempt_at(self) -> datetime | None:
        return self.delivery.next_attempt_at

    @property
    def last_error(self) -> str | None:
        return self.delivery.last_error

    @property
    def accepted_at(self) -> datetime | None:
        return self.delivery.accepted_at

    @property
    def accepted_rpc_method(self) -> str | None:
        return self.delivery.accepted_rpc_method

    @property
    def accepted_turn_id(self) -> str | None:
        return self.delivery.accepted_turn_id

    @classmethod
    def from_terminal(
        cls,
        terminal: TerminalEvent,
        *,
        wake_context: WakeContext | None = None,
    ) -> NotificationEvent:
        return cls._pending(
            event_id=terminal.event_id,
            event_kind="terminal",
            study_id=terminal.study_id,
            run_id=terminal.run_id,
            attempt=terminal.attempt,
            status=terminal.status,
            occurred_at=terminal.occurred_at,
            originating_thread_id=terminal.originating_thread_id,
            event_state_path=terminal.terminal_state_path,
            wake_context=wake_context,
        )

    @classmethod
    def from_lifecycle(
        cls,
        event: LifecycleEvent,
        *,
        wake_context: WakeContext | None = None,
    ) -> NotificationEvent:
        return cls._pending(
            event_id=event.event_id,
            event_kind=cast(EventKind, event.kind),
            study_id=event.study_id,
            run_id=event.run_id,
            attempt=event.attempt,
            status=LIFECYCLE_STATUS_BY_KIND[event.kind],
            occurred_at=event.occurred_at,
            originating_thread_id=event.originating_thread_id,
            event_state_path=event.event_state_path,
            wake_context=wake_context,
        )

    @classmethod
    def _pending(
        cls,
        *,
        event_id: str,
        event_kind: EventKind,
        study_id: str,
        run_id: str,
        attempt: int,
        status: str,
        occurred_at: datetime,
        originating_thread_id: str | None,
        event_state_path: str,
        wake_context: WakeContext | None,
    ) -> NotificationEvent:
        if originating_thread_id is None:
            raise NotificationStateError("version-2 notifications require an originating Codex thread ID")
        return cls(
            event_id=event_id,
            event_kind=event_kind,
            study_id=study_id,
            run_id=run_id,
            attempt=attempt,
            status=status,
            occurred_at=occurred_at,
            originating_thread_id=originating_thread_id,
            terminal_state_path=event_state_path,
            delivery=NotificationRecord.pending(
                watch_id=event_id,
                event_id=event_id,
                thread_id=originating_thread_id,
            ),
            wake_context=wake_context,
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        source: Path,
        root: Path,
    ) -> NotificationEvent:
        if set(payload) != EVENT_FIELDS:
            raise NotificationStateError(f"notification in {source} has invalid version-2 fields")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise NotificationStateError("unsupported notify-wake contract; cutover required")
        try:
            delivery = NotificationRecord.from_dict(payload["delivery"])
        except (TypeError, ValueError) as error:
            raise NotificationStateError(
                f"notification in {source} has invalid shared delivery state: {error}"
            ) from error
        event_kind = payload["event_kind"]
        if not isinstance(event_kind, str):
            raise NotificationStateError("event_kind must be a string")
        event_path = payload["event_state_path"]
        if not isinstance(event_path, str):
            raise NotificationStateError("event_state_path must be a string")
        occurred_at = _parse_datetime(payload["occurred_at"], "occurred_at")
        return cls(
            event_id=cast(str, payload["event_id"]),
            event_kind=cast(EventKind, event_kind),
            study_id=cast(str, payload["study_id"]),
            run_id=cast(str, payload["run_id"]),
            attempt=cast(int, payload["attempt"]),
            status=cast(str, payload["status"]),
            occurred_at=occurred_at,
            originating_thread_id=cast(str | None, payload["originating_thread_id"]),
            terminal_state_path=str(_managed_path(Path(event_path), root, "event_state_path")),
            delivery=delivery,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "event_id": self.event_id,
            "event_kind": self.event_kind,
            "study_id": self.study_id,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "status": self.status,
            "occurred_at": self.occurred_at.isoformat(),
            "originating_thread_id": self.originating_thread_id,
            "event_state_path": self.terminal_state_path,
            "delivery": self.delivery.to_dict(),
        }

    def with_delivery_failure(
        self,
        *,
        attempted_at: datetime,
        error: str,
        next_attempt_at: datetime | None,
        exhausted: bool,
    ) -> NotificationEvent:
        if exhausted:
            delivery = self.delivery.mark_blocked(
                attempted_at=attempted_at,
                error=error,
            )
        else:
            if next_attempt_at is None:
                raise NotificationStateError("retryable delivery requires next_attempt_at")
            delivery = self.delivery.schedule_retry(
                attempted_at=attempted_at,
                error=error,
                next_attempt_at=next_attempt_at,
                increment_attempt=self.state not in {"in_flight", "uncertain"},
            )
        return replace(self, delivery=delivery)

    def with_acceptance(
        self,
        *,
        accepted_at: datetime,
        rpc_method: str,
        turn_id: str,
    ) -> NotificationEvent:
        return replace(
            self,
            delivery=self.delivery.mark_accepted(
                accepted_at=accepted_at,
                rpc_method=rpc_method,
                turn_id=turn_id,
            ),
        )

    def requeued(self) -> NotificationEvent:
        if self.state != "blocked":
            raise NotificationStateError("only blocked notifications can be requeued")
        assert self.originating_thread_id is not None
        return replace(
            self,
            delivery=NotificationRecord.pending(
                watch_id=self.event_id,
                event_id=self.event_id,
                thread_id=self.originating_thread_id,
            ),
        )


def read_terminal_event(path: Path, root: Path) -> TerminalEvent:
    resolved = _managed_path(path.absolute(), root, "terminal path")
    if resolved.name != TERMINAL_FILENAME or resolved.parent.parent.name != "runs":
        raise NotificationStateError("terminal path is not an exact managed run terminal")
    payload = _load_json(resolved)
    if payload.get(TERMINAL_CONTRACT_FIELD) != SCHEMA_VERSION:
        raise NotificationStateError("unsupported notify-wake contract; cutover required")
    status = payload.get("status")
    if not isinstance(status, str) or status not in TERMINAL_STATUSES:
        raise NotificationStateError(f"invalid terminal status: {status!r}")
    event_id = payload.get("terminal_event_id")
    finished_at = payload.get("finished_at")
    return TerminalEvent(
        event_id=_validate_event_id(event_id),
        study_id=_validate_identifier(resolved.parent.parent.parent.name, "study id"),
        run_id=_validate_identifier(resolved.parent.name, "run id"),
        attempt=_validate_attempt(payload.get("attempt")),
        status=cast(TerminalStatus, status),
        occurred_at=_parse_datetime(finished_at, "finished_at"),
        originating_thread_id=_validate_thread_id(payload.get("originating_thread_id")),
        terminal_state_path=str(resolved),
    )


def read_notification_event(path: Path, root: Path) -> NotificationEvent:
    resolved_root = validate_notification_root(root)
    resolved = _managed_path(path.absolute(), resolved_root, "notification path")
    payload = _load_json(resolved)
    event = NotificationEvent.from_dict(payload, resolved, resolved_root)
    if resolved != notification_path_for_event(resolved_root, event.event_id):
        raise NotificationStateError("notification path does not match its version-2 event identity")
    return replace(
        event,
        wake_context=_read_wake_context(
            resolved_root,
            event.study_id,
            event.run_id,
        ),
    )


def write_notification_event(event: NotificationEvent, root: Path) -> None:
    managed_root = validate_notification_root(root)
    expected = notification_path_for_event(managed_root, event.event_id)
    _managed_path(Path(event.terminal_state_path), managed_root, "event_state_path")
    atomic_write_json(expected, event.to_dict())


def _queue_event(event: NotificationEvent, root: Path) -> NotificationEvent:
    path = notification_path_for_event(root, event.event_id)
    with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
        if path.exists():
            current = read_notification_event(path, root)
            current_identity = {key: value for key, value in current.to_dict().items() if key != "delivery"}
            event_identity = {key: value for key, value in event.to_dict().items() if key != "delivery"}
            if current_identity != event_identity:
                raise NotificationStateError("current notification belongs to a different research event")
            return current
        write_notification_event(event, root)
        return event


def queue_notification_from_terminal(
    terminal_path: Path,
    root: Path,
    *,
    study_id: str,
    run_id: str,
) -> NotificationEvent:
    """Create a v2 notification from explicitly v2-marked terminal truth."""

    managed_root = validate_notification_root(root)
    terminal = read_terminal_event(terminal_path, managed_root)
    if (terminal.study_id, terminal.run_id) != (
        _validate_identifier(study_id, "study id"),
        _validate_identifier(run_id, "run id"),
    ):
        raise NotificationStateError("terminal identifiers do not match the requested study and run")
    event = NotificationEvent.from_terminal(
        terminal,
        wake_context=_read_wake_context(
            managed_root,
            terminal.study_id,
            terminal.run_id,
        ),
    )
    return _queue_event(event, managed_root)


def queue_notification_from_lifecycle(event_path: Path, root: Path) -> NotificationEvent:
    """Create a v2 notification from a version-2 lifecycle source."""

    managed_root = validate_notification_root(root)
    resolved = _managed_path(event_path.absolute(), managed_root, "lifecycle event path")
    event = read_lifecycle_event(resolved)
    notification = NotificationEvent.from_lifecycle(
        event,
        wake_context=_read_wake_context(
            managed_root,
            event.study_id,
            event.run_id,
        ),
    )
    return _queue_event(notification, managed_root)


def ensure_notification(
    terminal_path: Path,
    root: Path,
    *,
    requeue: bool = False,
) -> NotificationEvent:
    terminal = read_terminal_event(terminal_path, root)
    event = queue_notification_from_terminal(
        terminal_path,
        root,
        study_id=terminal.study_id,
        run_id=terminal.run_id,
    )
    if requeue:
        event = event.requeued()
        write_notification_event(event, root)
    return event


def build_wake_prompt(event: NotificationEvent) -> str:
    if event.event_kind == "terminal":
        heading = "Research run completed."
    else:
        heading = "Research lifecycle event requires attention."
    return (
        f"{heading}\n"
        f"Event: {event.event_kind}\n"
        f"Study: {event.study_id}\n"
        f"Run: {event.run_id}\n"
        f"Status: {event.status}\n"
        f"Event state: {event.terminal_state_path}\n\n"
        "Inspect the event state and continue the study protocol."
    )


def notification_lock_path(root: Path, thread_id: str) -> Path:
    digest = hashlib.sha256(thread_id.encode()).hexdigest()
    return notification_namespace(root) / ".thread-locks" / f"{digest}.lock"


def goal_wait_path(root: Path, thread_id: str) -> Path:
    digest = hashlib.sha256(thread_id.encode()).hexdigest()
    return notification_namespace(root) / GOAL_WAIT_DIRECTORY / f"{digest}.json"


def _read_goal_wait(root: Path, thread_id: str) -> NotifyWaitLease | None:
    path = goal_wait_path(root, thread_id)
    if not path.exists():
        return None
    try:
        return NotifyWaitLease.from_dict(_load_json(path))
    except ValueError as error:
        raise NotificationStateError(f"goal-wait lease is invalid: {error}") from error


def _write_goal_wait(root: Path, lease: NotifyWaitLease) -> None:
    atomic_write_json(goal_wait_path(root, lease.thread_id), lease.to_dict())


async def enter_research_notify_wait(
    root: Path,
    *,
    context: WakeContext,
    loop_id: str,
    source_ids: tuple[str, ...],
    transport: MessageTransport,
    verify_loop_identity: Callable[[str, tuple[str, ...]], bool],
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> NotifyWaitLease:
    managed_root = validate_notification_root(root)
    async with _async_file_lock(notification_lock_path(managed_root, context.thread_id)):
        return await enter_notify_wait(
            context=context,
            loop_id=loop_id,
            source_ids=source_ids,
            transport=transport,
            persist_lease=lambda lease: _write_goal_wait(managed_root, lease),
            verify_loop_identity=verify_loop_identity,
            request_timeout=request_timeout,
        )


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


def _sanitize_error(error: BaseException) -> str:
    text = " ".join(str(error).split())
    return (text or error.__class__.__name__)[:MAX_LAST_ERROR_LENGTH]


def _is_due(event: NotificationEvent, now: datetime) -> bool:
    if event.state in {"in_flight", "uncertain"}:
        return True
    return event.state in {"pending", "retry_due"} and (event.next_attempt_at is None or event.next_attempt_at <= now)


@asynccontextmanager
async def _async_file_lock(path: Path) -> AsyncIterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _retry_event(
    event: NotificationEvent,
    *,
    attempted_at: datetime,
    error: str,
    random: Random,
) -> NotificationEvent:
    projected_attempts = event.attempt_count + (0 if event.state in {"in_flight", "uncertain"} else 1)
    exponent = max(projected_attempts - 1, 0)
    delay_cap = min(RETRY_CAP_SECONDS, RETRY_BASE_SECONDS * (RETRY_FACTOR**exponent))
    next_attempt_at = attempted_at + timedelta(seconds=random.uniform(0.0, delay_cap))
    return replace(
        event,
        delivery=event.delivery.schedule_retry(
            attempted_at=attempted_at,
            error=error,
            next_attempt_at=next_attempt_at,
            increment_attempt=event.state not in {"in_flight", "uncertain"},
        ),
    )


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
        return "failed", f"{path}: {_sanitize_error(error)}"
    if initial.state == "accepted":
        return "skipped", None
    if initial.state == "blocked":
        return "failed", f"{path}: notification requires explicit requeue"
    if not _is_due(initial, now):
        return "skipped", None
    if initial.wake_context is None:
        blocked = replace(
            initial,
            delivery=initial.delivery.mark_blocked(
                attempted_at=now,
                error="notification has no version-2 wake context",
            ),
        )
        write_notification_event(blocked, root)
        return "failed", f"{path}: notification has no version-2 wake context"

    thread_id = initial.delivery.thread_id
    async with _async_file_lock(notification_lock_path(root, thread_id)):
        event = read_notification_event(path, root)
        if not _is_due(event, now):
            return "skipped", None
        try:
            lease = _read_goal_wait(root, thread_id)
            transport = await connect()

            def persist_boundary(rpc_method: str, sent_at: datetime) -> None:
                current = read_notification_event(path, root)
                updated = replace(
                    current,
                    delivery=current.delivery.mark_in_flight(
                        sent_at,
                        rpc_method=rpc_method,
                    ),
                )
                with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
                    write_notification_event(updated, root)

            request = WakeRequest(
                event_id=event.event_id,
                prompt=build_wake_prompt(event),
                context=cast(WakeContext, event.wake_context),
            )
            if event.delivery.requires_history_reconciliation:
                attempted_rpc_method = event.delivery.attempted_rpc_method
                if attempted_rpc_method is None:
                    raise NotificationStateError("uncertain notification lacks attempted_rpc_method")
                outcome = await reconcile_wake(
                    request,
                    transport,
                    attempted_rpc_method=attempted_rpc_method,
                    lease=lease,
                    persist_lease=lambda selected: _write_goal_wait(root, selected),
                    now=lambda: now,
                    request_timeout=request_timeout,
                )
            else:
                outcome = await deliver_wake(
                    request,
                    transport,
                    persist_request_boundary=persist_boundary,
                    lease=lease,
                    persist_lease=lambda selected: _write_goal_wait(root, selected),
                    now=lambda: now,
                    request_timeout=request_timeout,
                )

            current = read_notification_event(path, root)
            if outcome.state == DeliveryState.ACCEPTED:
                if outcome.rpc_method is None or outcome.turn_id is None:
                    raise NotificationStateError("accepted wake lacks turn metadata")
                updated = replace(
                    current,
                    delivery=current.delivery.mark_accepted(
                        accepted_at=now,
                        rpc_method=outcome.rpc_method,
                        turn_id=outcome.turn_id,
                    ),
                )
                result = "accepted"
                problem = None
            elif outcome.state == DeliveryState.UNCERTAIN:
                sent_at = current.delivery.request_sent_at or now
                updated = replace(
                    current,
                    delivery=current.delivery.mark_uncertain(
                        sent_at=sent_at,
                        reason=outcome.error or "wake acknowledgment is uncertain",
                    ),
                )
                result = "retrying"
                problem = f"{path}: {outcome.error}"
            elif outcome.state == DeliveryState.BLOCKED:
                if current.delivery.requires_history_reconciliation:
                    delivery = current.delivery.mark_reconciliation_blocked(
                        attempted_at=now,
                        error=outcome.error or "history reconciliation is blocked",
                    )
                else:
                    delivery = current.delivery.mark_blocked(
                        attempted_at=now,
                        error=outcome.error or "wake delivery is blocked",
                    )
                updated = replace(current, delivery=delivery)
                result = "failed"
                problem = f"{path}: {delivery.last_error}"
            else:
                updated = _retry_event(
                    current,
                    attempted_at=now,
                    error=outcome.error or "wake delivery retry is due",
                    random=random,
                )
                result = "failed" if updated.state == "blocked" else "retrying"
                problem = f"{path}: {updated.last_error}"
        except BaseException as error:
            if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            current = read_notification_event(path, root)
            if isinstance(error, AppServerError) and error.permanent:
                updated = replace(
                    current,
                    delivery=current.delivery.mark_blocked(
                        attempted_at=now,
                        error=_sanitize_error(error),
                    ),
                )
            else:
                updated = _retry_event(
                    current,
                    attempted_at=now,
                    error=_sanitize_error(error),
                    random=random,
                )
            result = "failed" if updated.state == "blocked" else "retrying"
            problem = f"{path}: {updated.last_error}"
        with FileLock(str(path.parent / NOTIFICATION_LOCK_FILENAME)):
            write_notification_event(updated, root)
        return result, problem


def _notification_paths(
    root: Path,
    study_ids: frozenset[str] | None = None,
) -> list[Path]:
    events_root = notification_namespace(root) / "events"
    if not events_root.exists():
        return []
    paths = sorted(events_root.glob(f"*/{NOTIFICATION_FILENAME}"))
    if study_ids is None:
        return paths
    selected: list[Path] = []
    for path in paths:
        try:
            event = read_notification_event(path, root)
        except (OSError, NotificationStateError):
            selected.append(path)
        else:
            if event.study_id in study_ids:
                selected.append(path)
    return selected


def next_notification_attempt_at(
    root: Path,
    *,
    study_ids: frozenset[str] | None = None,
) -> datetime | None:
    requested_root = root.expanduser()
    if not requested_root.exists() and not requested_root.is_symlink():
        return None
    managed_root = validate_notification_root(requested_root)
    retry_deadlines: list[datetime] = []
    for path in _notification_paths(managed_root, study_ids):
        try:
            event = read_notification_event(path, managed_root)
        except (OSError, NotificationStateError):
            continue
        if event.state == "retry_due" and event.next_attempt_at is not None:
            retry_deadlines.append(event.next_attempt_at)
    return min(retry_deadlines) if retry_deadlines else None


async def sweep_notifications(
    root: Path,
    *,
    connect: Callable[[], Awaitable[MessageTransport]],
    study_ids: frozenset[str] | None = None,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    random: Random | None = None,
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
) -> SweepResult:
    requested_root = root.expanduser()
    if not requested_root.exists() and not requested_root.is_symlink():
        return SweepResult()
    managed_root = validate_notification_root(requested_root)
    selected_now = normalize_datetime(now(), "worker clock")
    generator = random or Random()
    paths = _notification_paths(managed_root, study_ids)
    counts = {"accepted": 0, "retrying": 0, "failed": 0, "skipped": 0}
    due = 0
    problems: list[str] = []
    for path in paths:
        try:
            event = read_notification_event(path, managed_root)
            if _is_due(event, selected_now):
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


def unix_connector(socket_path: Path) -> Callable[[], Awaitable[MessageTransport]]:
    async def connect() -> MessageTransport:
        return await UnixWebSocketTransport.connect(socket_path)

    return connect


__all__ = [
    "APP_SERVER_BASELINE",
    "DELIVERY_STATES",
    "MANAGED_ROOT_SCHEMA_VERSION",
    "ManagedRootRegistration",
    "MessageTransport",
    "NotificationEvent",
    "NotificationStateError",
    "SCHEMA_VERSION",
    "SweepResult",
    "TERMINAL_CONTRACT_FIELD",
    "TerminalEvent",
    "UnixWebSocketTransport",
    "build_wake_prompt",
    "capture_wake_context",
    "ensure_notification",
    "enter_research_notify_wait",
    "goal_wait_path",
    "initialize_notification_root",
    "next_notification_attempt_at",
    "notification_lock_path",
    "notification_namespace",
    "notification_path_for_event",
    "persist_wake_context",
    "queue_notification_from_lifecycle",
    "queue_notification_from_terminal",
    "read_notification_event",
    "read_terminal_event",
    "register_notification_root",
    "sweep_notifications",
    "unix_connector",
    "validate_notification_root",
    "write_notification_event",
]
