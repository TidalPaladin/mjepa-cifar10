"""Durable, local lifecycle events for managed research runs."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal, cast
from uuid import NAMESPACE_URL, UUID, uuid5

from .runtime import atomic_write_json


LIFECYCLE_SCHEMA_VERSION: Final[int] = 2
PROGRESS_FILENAME: Final[str] = "progress.json"
FIRST_CYCLE_FILENAME: Final[str] = "first-cycle.json"
SUPERVISOR_LOST_FILENAME: Final[str] = "supervisor-lost.json"
PROGRESS_STALLED_FILENAME: Final[str] = "progress-stalled.json"
LIFECYCLE_FILENAMES: Final[frozenset[str]] = frozenset(
    (FIRST_CYCLE_FILENAME, SUPERVISOR_LOST_FILENAME, PROGRESS_STALLED_FILENAME)
)
EVENT_FILENAME_BY_KIND: Final[dict[str, str]] = {
    "first_cycle_completed": FIRST_CYCLE_FILENAME,
    "supervisor_lost": SUPERVISOR_LOST_FILENAME,
    "progress_stalled": PROGRESS_STALLED_FILENAME,
}
KIND_BY_EVENT_FILENAME: Final[dict[str, str]] = {value: key for key, value in EVENT_FILENAME_BY_KIND.items()}
LIFECYCLE_EVENT_NAMESPACE: Final[UUID] = uuid5(
    NAMESPACE_URL,
    "openai.com/autoresearch/lifecycle-event/v2",
)

LifecycleKind = Literal["first_cycle_completed", "supervisor_lost", "progress_stalled"]
ProgressPhase = Literal["training", "validation", "checkpointing", "checkpointed"]
DetailValue = str | int | float | bool | None
DETAIL_VALUE_TYPES: Final = (str, int, float, bool, type(None))


class LifecycleStateError(ValueError):
    """A lifecycle event or progress record violates the managed-run contract."""


def _parse_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, str):
        raise LifecycleStateError(f"{field_name} must be an ISO 8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise LifecycleStateError(f"{field_name} must be an ISO 8601 string") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LifecycleStateError(f"{field_name} must include a UTC offset")
    return parsed.astimezone(UTC)


def _validate_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LifecycleStateError(f"{field_name} must include a UTC offset")
    return value.astimezone(UTC)


def _validate_positive_attempt(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise LifecycleStateError("attempt must be a positive integer")
    return value


def _validate_nonnegative_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LifecycleStateError(f"{field_name} must be a non-negative integer")
    return value


def _validate_nonnegative_float(value: object, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool) or value < 0:
        raise LifecycleStateError(f"{field_name} must be a non-negative number")
    return float(value)


def _validate_thread_id(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > 512:
        raise LifecycleStateError("originating_thread_id must be a non-empty string or null")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise LifecycleStateError("originating_thread_id must not contain control characters")
    return value


def _load_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LifecycleStateError(f"state in {path} is not valid JSON: {error}") from error
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise LifecycleStateError(f"state in {path} must be a JSON object")
    return cast(dict[str, object], payload)


def _run_identity(run_dir: Path) -> tuple[Path, str, str]:
    resolved = run_dir.expanduser().resolve(strict=False)
    if resolved.parent.name != "runs" or not resolved.name or not resolved.parent.parent.name:
        raise LifecycleStateError("run directory must be <managed-root>/<study>/runs/<run>")
    if resolved.is_symlink():
        raise LifecycleStateError("run directory must not be a symlink")
    return resolved, resolved.parent.parent.name, resolved.name


def lifecycle_event_id(study_id: str, run_id: str, attempt: int, kind: LifecycleKind) -> str:
    """Return the stable identifier for one lifecycle kind in one run attempt."""
    return str(uuid5(LIFECYCLE_EVENT_NAMESPACE, f"{study_id}/{run_id}/{attempt}/{kind}"))


@dataclass(frozen=True, slots=True)
class ProgressState:
    study_id: str
    run_id: str
    attempt: int
    updated_at: datetime
    phase: ProgressPhase
    epoch: int
    optimizer_step: int
    active_seconds: float
    originating_thread_id: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "attempt", _validate_positive_attempt(self.attempt))
        object.__setattr__(self, "updated_at", _validate_datetime(self.updated_at, "updated_at"))
        if self.phase not in {"training", "validation", "checkpointing", "checkpointed"}:
            raise LifecycleStateError(f"invalid progress phase: {self.phase!r}")
        object.__setattr__(self, "epoch", _validate_nonnegative_integer(self.epoch, "epoch"))
        object.__setattr__(
            self,
            "optimizer_step",
            _validate_nonnegative_integer(self.optimizer_step, "optimizer_step"),
        )
        object.__setattr__(
            self,
            "active_seconds",
            _validate_nonnegative_float(self.active_seconds, "active_seconds"),
        )
        object.__setattr__(self, "originating_thread_id", _validate_thread_id(self.originating_thread_id))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "study_id": self.study_id,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "updated_at": self.updated_at.isoformat(),
            "phase": self.phase,
            "epoch": self.epoch,
            "optimizer_step": self.optimizer_step,
            "active_seconds": self.active_seconds,
            "originating_thread_id": self.originating_thread_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ProgressState:
        expected = {
            "schema_version",
            "study_id",
            "run_id",
            "attempt",
            "updated_at",
            "phase",
            "epoch",
            "optimizer_step",
            "active_seconds",
            "originating_thread_id",
        }
        if set(payload) != expected or payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
            raise LifecycleStateError("progress state has invalid fields or schema version")
        phase = payload["phase"]
        if not isinstance(phase, str):
            raise LifecycleStateError("phase must be a string")
        return cls(
            study_id=cast(str, payload["study_id"]),
            run_id=cast(str, payload["run_id"]),
            attempt=cast(int, payload["attempt"]),
            updated_at=_parse_datetime(payload["updated_at"], "updated_at"),
            phase=cast(ProgressPhase, phase),
            epoch=cast(int, payload["epoch"]),
            optimizer_step=cast(int, payload["optimizer_step"]),
            active_seconds=cast(float, payload["active_seconds"]),
            originating_thread_id=cast(str | None, payload["originating_thread_id"]),
        )


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    event_id: str
    kind: LifecycleKind
    study_id: str
    run_id: str
    attempt: int
    occurred_at: datetime
    originating_thread_id: str | None
    event_state_path: str
    details: dict[str, DetailValue]

    def __post_init__(self) -> None:
        expected_id = lifecycle_event_id(self.study_id, self.run_id, self.attempt, self.kind)
        if self.event_id != expected_id:
            raise LifecycleStateError("event_id does not match the lifecycle event identity")
        if self.kind not in EVENT_FILENAME_BY_KIND:
            raise LifecycleStateError(f"invalid lifecycle event kind: {self.kind!r}")
        object.__setattr__(self, "attempt", _validate_positive_attempt(self.attempt))
        object.__setattr__(self, "occurred_at", _validate_datetime(self.occurred_at, "occurred_at"))
        object.__setattr__(self, "originating_thread_id", _validate_thread_id(self.originating_thread_id))
        if not isinstance(self.event_state_path, str) or not self.event_state_path:
            raise LifecycleStateError("event_state_path must be a non-empty string")
        if not all(
            isinstance(key, str) and isinstance(value, DETAIL_VALUE_TYPES) for key, value in self.details.items()
        ):
            raise LifecycleStateError("event details must contain string keys and scalar JSON values")
        object.__setattr__(self, "details", dict(self.details))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": LIFECYCLE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "kind": self.kind,
            "study_id": self.study_id,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "occurred_at": self.occurred_at.isoformat(),
            "originating_thread_id": self.originating_thread_id,
            "event_state_path": self.event_state_path,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> LifecycleEvent:
        expected = {
            "schema_version",
            "event_id",
            "kind",
            "study_id",
            "run_id",
            "attempt",
            "occurred_at",
            "originating_thread_id",
            "event_state_path",
            "details",
        }
        if set(payload) != expected or payload.get("schema_version") != LIFECYCLE_SCHEMA_VERSION:
            raise LifecycleStateError("lifecycle event has invalid fields or schema version")
        kind = payload["kind"]
        details = payload["details"]
        if not isinstance(kind, str) or kind not in EVENT_FILENAME_BY_KIND:
            raise LifecycleStateError(f"invalid lifecycle event kind: {kind!r}")
        if not isinstance(details, dict):
            raise LifecycleStateError("event details must be a JSON object")
        return cls(
            event_id=cast(str, payload["event_id"]),
            kind=cast(LifecycleKind, kind),
            study_id=cast(str, payload["study_id"]),
            run_id=cast(str, payload["run_id"]),
            attempt=cast(int, payload["attempt"]),
            occurred_at=_parse_datetime(payload["occurred_at"], "occurred_at"),
            originating_thread_id=cast(str | None, payload["originating_thread_id"]),
            event_state_path=cast(str, payload["event_state_path"]),
            details=cast(dict[str, DetailValue], details),
        )


@dataclass(frozen=True, slots=True)
class RunLifecycleReporter:
    """Persist trainer-owned progress and one-shot lifecycle milestones."""

    run_dir: Path
    study_id: str
    run_id: str
    attempt: int
    originating_thread_id: str | None
    now: Callable[[], datetime] = lambda: datetime.now(UTC)

    def progress(
        self,
        phase: ProgressPhase,
        epoch: int,
        optimizer_step: int,
        active_seconds: float,
    ) -> Path:
        return write_progress_state(
            self.run_dir,
            ProgressState(
                study_id=self.study_id,
                run_id=self.run_id,
                attempt=self.attempt,
                updated_at=self.now(),
                phase=phase,
                epoch=epoch,
                optimizer_step=optimizer_step,
                active_seconds=active_seconds,
                originating_thread_id=self.originating_thread_id,
            ),
        )

    def first_cycle(
        self,
        epoch: int,
        optimizer_step: int,
        active_seconds: float,
        checkpoint_path: Path,
    ) -> LifecycleEvent:
        return persist_first_cycle_event(
            self.run_dir,
            study_id=self.study_id,
            run_id=self.run_id,
            attempt=self.attempt,
            occurred_at=self.now(),
            originating_thread_id=self.originating_thread_id,
            epoch=epoch,
            optimizer_step=optimizer_step,
            active_seconds=active_seconds,
            checkpoint_path=checkpoint_path,
        )


def write_progress_state(run_dir: Path, progress: ProgressState) -> Path:
    resolved, study_id, run_id = _run_identity(run_dir)
    if (progress.study_id, progress.run_id) != (study_id, run_id):
        raise LifecycleStateError("progress identifiers do not match the managed run directory")
    path = resolved / PROGRESS_FILENAME
    atomic_write_json(path, progress.to_dict())
    return path


def read_progress_state(run_dir: Path) -> ProgressState:
    resolved, study_id, run_id = _run_identity(run_dir)
    progress = ProgressState.from_dict(_load_json(resolved / PROGRESS_FILENAME))
    if (progress.study_id, progress.run_id) != (study_id, run_id):
        raise LifecycleStateError("progress identifiers do not match the managed run directory")
    return progress


def _persist_event(
    run_dir: Path,
    *,
    kind: LifecycleKind,
    study_id: str,
    run_id: str,
    attempt: int,
    occurred_at: datetime,
    originating_thread_id: str | None,
    details: dict[str, DetailValue],
) -> LifecycleEvent:
    resolved, expected_study_id, expected_run_id = _run_identity(run_dir)
    if (study_id, run_id) != (expected_study_id, expected_run_id):
        raise LifecycleStateError("event identifiers do not match the managed run directory")
    event_path = resolved / EVENT_FILENAME_BY_KIND[kind]
    if event_path.exists():
        current = read_lifecycle_event(event_path)
        if current.attempt != attempt or current.kind != kind:
            raise LifecycleStateError("existing lifecycle event belongs to another run attempt")
        return current
    event = LifecycleEvent(
        event_id=lifecycle_event_id(study_id, run_id, attempt, kind),
        kind=kind,
        study_id=study_id,
        run_id=run_id,
        attempt=attempt,
        occurred_at=occurred_at,
        originating_thread_id=originating_thread_id,
        event_state_path=str(event_path),
        details=details,
    )
    atomic_write_json(event_path, event.to_dict())
    return event


def read_lifecycle_event(path: Path) -> LifecycleEvent:
    resolved = path.expanduser().resolve(strict=True)
    run_dir, study_id, run_id = _run_identity(resolved.parent)
    if resolved.name not in LIFECYCLE_FILENAMES:
        raise LifecycleStateError("path is not a recognized lifecycle event")
    event = LifecycleEvent.from_dict(_load_json(resolved))
    if event.kind != KIND_BY_EVENT_FILENAME[resolved.name]:
        raise LifecycleStateError("event kind does not match its filename")
    if (event.study_id, event.run_id) != (study_id, run_id):
        raise LifecycleStateError("event identifiers do not match the managed run directory")
    if Path(event.event_state_path).resolve(strict=False) != resolved:
        raise LifecycleStateError("event_state_path does not identify the current lifecycle event")
    del run_dir
    return event


def persist_first_cycle_event(
    run_dir: Path,
    *,
    study_id: str,
    run_id: str,
    attempt: int,
    occurred_at: datetime,
    originating_thread_id: str | None,
    epoch: int,
    optimizer_step: int,
    active_seconds: float,
    checkpoint_path: Path,
) -> LifecycleEvent:
    resolved, _, _ = _run_identity(run_dir)
    checkpoint = checkpoint_path.expanduser().resolve(strict=True)
    if checkpoint != resolved / "checkpoint.pt":
        raise LifecycleStateError("first-cycle checkpoint must be the managed run checkpoint")
    return _persist_event(
        resolved,
        kind="first_cycle_completed",
        study_id=study_id,
        run_id=run_id,
        attempt=attempt,
        occurred_at=occurred_at,
        originating_thread_id=originating_thread_id,
        details={
            "epoch": _validate_nonnegative_integer(epoch, "epoch"),
            "optimizer_step": _validate_nonnegative_integer(optimizer_step, "optimizer_step"),
            "active_seconds": _validate_nonnegative_float(active_seconds, "active_seconds"),
            "checkpoint_path": str(checkpoint),
        },
    )


def _persist_supervisor_lost_event(
    run_dir: Path,
    *,
    study_id: str,
    run_id: str,
    attempt: int,
    occurred_at: datetime,
    originating_thread_id: str | None,
    supervisor_pid: int,
    last_heartbeat_at: str | None,
) -> LifecycleEvent:
    return _persist_event(
        run_dir,
        kind="supervisor_lost",
        study_id=study_id,
        run_id=run_id,
        attempt=attempt,
        occurred_at=occurred_at,
        originating_thread_id=originating_thread_id,
        details={"supervisor_pid": supervisor_pid, "last_heartbeat_at": last_heartbeat_at},
    )


def _persist_progress_stalled_event(
    run_dir: Path,
    *,
    progress: ProgressState,
    occurred_at: datetime,
    progress_timeout: timedelta,
) -> LifecycleEvent:
    return _persist_event(
        run_dir,
        kind="progress_stalled",
        study_id=progress.study_id,
        run_id=progress.run_id,
        attempt=progress.attempt,
        occurred_at=occurred_at,
        originating_thread_id=progress.originating_thread_id,
        details={
            "last_progress_at": progress.updated_at.isoformat(),
            "timeout_seconds": progress_timeout.total_seconds(),
            "phase": progress.phase,
            "epoch": progress.epoch,
            "optimizer_step": progress.optimizer_step,
        },
    )


def reconcile_run_safety_events(
    run_dir: Path,
    *,
    now: datetime,
    progress_timeout: timedelta,
    pid_is_alive: Callable[[int], bool] = lambda pid: _pid_is_alive(pid),
) -> tuple[LifecycleEvent, ...]:
    """Create missing exceptional events without changing run terminal truth."""
    resolved, study_id, run_id = _run_identity(run_dir)
    if (resolved / "terminal.json").exists() or not (resolved / "worker.json").is_file():
        return ()
    selected_now = _validate_datetime(now, "now")
    if progress_timeout <= timedelta(0):
        raise LifecycleStateError("progress_timeout must be positive")
    worker = _load_json(resolved / "worker.json")
    if worker.get("status") != "running":
        return ()
    supervisor_pid = worker.get("pid")
    if not isinstance(supervisor_pid, int) or isinstance(supervisor_pid, bool) or supervisor_pid < 1:
        raise LifecycleStateError("running worker PID must be a positive integer")
    attempt = _validate_positive_attempt(worker.get("attempt"))
    thread_id = _validate_thread_id(worker.get("originating_thread_id"))
    if not pid_is_alive(supervisor_pid):
        if (resolved / SUPERVISOR_LOST_FILENAME).exists():
            return ()
        return (
            _persist_supervisor_lost_event(
                resolved,
                study_id=study_id,
                run_id=run_id,
                attempt=attempt,
                occurred_at=selected_now,
                originating_thread_id=thread_id,
                supervisor_pid=supervisor_pid,
                last_heartbeat_at=cast(str | None, worker.get("heartbeat_at")),
            ),
        )
    progress_path = resolved / PROGRESS_FILENAME
    if not progress_path.is_file():
        return ()
    progress = read_progress_state(resolved)
    if progress.attempt != attempt or selected_now - progress.updated_at <= progress_timeout:
        return ()
    if (resolved / PROGRESS_STALLED_FILENAME).exists():
        return ()
    return (
        _persist_progress_stalled_event(
            resolved,
            progress=progress,
            occurred_at=selected_now,
            progress_timeout=progress_timeout,
        ),
    )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
