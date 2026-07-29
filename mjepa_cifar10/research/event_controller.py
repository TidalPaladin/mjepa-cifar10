"""Event-driven lifecycle supervision for managed research roots."""

from __future__ import annotations

import ctypes
import json
import logging
import os
import selectors
import struct
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final

from .codex_notifications import (
    NOTIFICATION_FILENAME,
    queue_notification_from_lifecycle,
    queue_notification_from_terminal,
    validate_notification_root,
)
from .lifecycle_events import (
    LIFECYCLE_FILENAMES,
    PROGRESS_FILENAME,
    PROGRESS_STALLED_FILENAME,
    LifecycleKind,
    read_progress_state,
    reconcile_run_safety_events,
)


WORKER_FILENAME: Final[str] = "worker.json"
TERMINAL_FILENAME: Final[str] = "terminal.json"
IN_CLOSE_WRITE: Final[int] = 0x00000008
IN_MOVED_TO: Final[int] = 0x00000080
IN_CREATE: Final[int] = 0x00000100
IN_ISDIR: Final[int] = 0x40000000
INOTIFY_MASK: Final[int] = IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE
INOTIFY_EVENT_HEADER: Final[struct.Struct] = struct.Struct("iIII")
INOTIFY_READ_BYTES: Final[int] = 64 * 1024
SOURCE_FILENAMES: Final[frozenset[str]] = frozenset(
    (WORKER_FILENAME, PROGRESS_FILENAME, TERMINAL_FILENAME, *LIFECYCLE_FILENAMES)
)
DELIVERY_SOURCE_FILENAMES: Final[frozenset[str]] = frozenset((TERMINAL_FILENAME, *LIFECYCLE_FILENAMES))
DEFAULT_PROGRESS_TIMEOUT_SECONDS: Final[int] = 30 * 60
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ControllerReconciliation:
    created_kinds: tuple[LifecycleKind, ...]
    queued: int
    active_pids: tuple[int, ...]
    problems: tuple[str, ...] = ()


def is_controller_source(filename: str) -> bool:
    """Return whether a durable write should cause local controller reconciliation."""
    return filename in SOURCE_FILENAMES


def _managed_run_directories(root: Path) -> tuple[Path, ...]:
    run_directories: list[Path] = []
    for study_dir in root.iterdir():
        runs_dir = study_dir / "runs"
        if study_dir.is_symlink() or not runs_dir.is_dir() or runs_dir.is_symlink():
            continue
        run_directories.extend(
            run_dir for run_dir in runs_dir.iterdir() if run_dir.is_dir() and not run_dir.is_symlink()
        )
    return tuple(sorted(run_directories))


def _running_supervisor_pid(run_dir: Path) -> int | None:
    worker_path = run_dir / WORKER_FILENAME
    terminal_path = run_dir / TERMINAL_FILENAME
    if terminal_path.exists() or not worker_path.is_file():
        return None
    try:
        worker = json.loads(worker_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    pid = worker.get("pid") if isinstance(worker, dict) and worker.get("status") == "running" else None
    return pid if isinstance(pid, int) and not isinstance(pid, bool) and pid > 0 else None


def reconcile_managed_root(
    root: Path,
    *,
    now: datetime | None = None,
    progress_timeout: timedelta,
    pid_is_alive: Callable[[int], bool] = lambda pid: _pid_is_alive(pid),
) -> ControllerReconciliation:
    """Recover lifecycle events and queues without changing run terminal status."""
    managed_root = validate_notification_root(root)
    selected_now = (now or datetime.now(UTC)).astimezone(UTC)
    created_kinds: list[LifecycleKind] = []
    queued = 0
    active_pids: list[int] = []
    problems: list[str] = []
    for run_dir in _managed_run_directories(managed_root):
        try:
            for event in reconcile_run_safety_events(
                run_dir,
                now=selected_now,
                progress_timeout=progress_timeout,
                pid_is_alive=pid_is_alive,
            ):
                created_kinds.append(event.kind)
        except (OSError, ValueError) as error:
            problems.append(f"{run_dir}: {type(error).__name__}: {error}")
        terminal_path = run_dir / TERMINAL_FILENAME
        terminal_notification_path = run_dir / NOTIFICATION_FILENAME
        if terminal_path.is_file() and not terminal_notification_path.exists():
            try:
                queue_notification_from_terminal(
                    terminal_path,
                    managed_root,
                    study_id=run_dir.parent.parent.name,
                    run_id=run_dir.name,
                )
                queued += 1
            except (OSError, ValueError) as error:
                problems.append(f"{terminal_path}: {type(error).__name__}: {error}")
        for event_filename in sorted(LIFECYCLE_FILENAMES):
            event_path = run_dir / event_filename
            notification_path = run_dir / f"{event_path.stem}.notification.json"
            if event_path.is_file() and not notification_path.exists():
                try:
                    queue_notification_from_lifecycle(event_path, managed_root)
                    queued += 1
                except (OSError, ValueError) as error:
                    problems.append(f"{event_path}: {type(error).__name__}: {error}")
        if (pid := _running_supervisor_pid(run_dir)) is not None and pid_is_alive(pid):
            active_pids.append(pid)
    return ControllerReconciliation(
        tuple(created_kinds),
        queued,
        tuple(sorted(set(active_pids))),
        tuple(problems),
    )


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class InotifyTree:
    """Minimal recursive inotify source used only for durable local state writes."""

    def __init__(self, root: Path, socket_path: Path | None = None) -> None:
        self.root = root
        self.socket_path = socket_path
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.fd = self.libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        if self.fd < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number))
        self.watches: dict[int, Path] = {}
        self._add_tree(root)
        if socket_path is not None:
            self._add_directory(socket_path.parent)

    def _add_directory(self, directory: Path) -> None:
        watch = self.libc.inotify_add_watch(self.fd, os.fsencode(directory), INOTIFY_MASK)
        if watch < 0:
            error_number = ctypes.get_errno()
            raise OSError(error_number, os.strerror(error_number), directory)
        self.watches[watch] = directory

    def _add_tree(self, root: Path) -> None:
        for directory, child_directories, _files in os.walk(root):
            path = Path(directory)
            self._add_directory(path)
            child_directories[:] = [name for name in child_directories if not (path / name).is_symlink()]

    def read(self) -> tuple[Path, ...]:
        payload = os.read(self.fd, INOTIFY_READ_BYTES)
        paths: list[Path] = []
        offset = 0
        while offset < len(payload):
            watch, mask, _cookie, name_length = INOTIFY_EVENT_HEADER.unpack_from(payload, offset)
            offset += INOTIFY_EVENT_HEADER.size
            name = payload[offset : offset + name_length].split(b"\0", 1)[0]
            offset += name_length
            parent = self.watches.get(watch)
            if parent is None:
                continue
            path = parent / os.fsdecode(name)
            if mask & IN_ISDIR and mask & (IN_CREATE | IN_MOVED_TO):
                self._add_tree(path)
            elif mask & (IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE):
                paths.append(path)
        return tuple(paths)

    def close(self) -> None:
        os.close(self.fd)


def _next_progress_deadline(root: Path, progress_timeout: timedelta, now: datetime) -> float | None:
    waits: list[float] = []
    for run_dir in _managed_run_directories(root):
        if _running_supervisor_pid(run_dir) is None or (run_dir / PROGRESS_STALLED_FILENAME).exists():
            continue
        try:
            progress = read_progress_state(run_dir)
        except (OSError, ValueError):
            continue
        waits.append(max(0.0, (progress.updated_at + progress_timeout - now).total_seconds()))
    return min(waits) if waits else None


def serve_controller(
    root: Path,
    *,
    progress_timeout: timedelta,
    deliver: Callable[[], None],
    next_delivery_at: Callable[[], datetime | None] = lambda: None,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    socket_path: Path | None = None,
    defer_until_socket_replaced: bool = False,
) -> None:
    """Serve lifecycle events through inotify, pidfds, and deadline timers."""
    managed_root = validate_notification_root(root)
    source = InotifyTree(managed_root, socket_path)
    selector = selectors.DefaultSelector()
    selector.register(source.fd, selectors.EVENT_READ, ("inotify", None))
    registered_pidfds: dict[int, int] = {}
    reported_problems: set[str] = set()
    delivery_deferred = defer_until_socket_replaced

    def report_new_problems(problems: tuple[str, ...]) -> None:
        for problem in problems:
            if problem not in reported_problems:
                LOGGER.warning("event controller isolated invalid state: %s", problem)
                reported_problems.add(problem)

    def refresh_pidfds(pids: tuple[int, ...]) -> None:
        desired = set(pids)
        for pid, pidfd in tuple(registered_pidfds.items()):
            if pid not in desired:
                selector.unregister(pidfd)
                os.close(pidfd)
                del registered_pidfds[pid]
        if not hasattr(os, "pidfd_open"):
            return
        for pid in desired - registered_pidfds.keys():
            try:
                pidfd = os.pidfd_open(pid, 0)
            except (OSError, ProcessLookupError):
                continue
            registered_pidfds[pid] = pidfd
            selector.register(pidfd, selectors.EVENT_READ, ("pid", pid))

    try:
        reconciliation = reconcile_managed_root(managed_root, progress_timeout=progress_timeout)
        report_new_problems(reconciliation.problems)
        refresh_pidfds(reconciliation.active_pids)
        if not delivery_deferred:
            deliver()
        while True:
            now = clock().astimezone(UTC)
            progress_wait = _next_progress_deadline(managed_root, progress_timeout, now)
            delivery_deadline = None if delivery_deferred else next_delivery_at()
            delivery_wait = (
                None
                if delivery_deadline is None
                else max(0.0, (delivery_deadline.astimezone(UTC) - now).total_seconds())
            )
            waits = tuple(wait for wait in (progress_wait, delivery_wait) if wait is not None)
            timeout = min(waits) if waits else None
            ready = selector.select(timeout)
            deadline_due = delivery_deadline is not None and delivery_deadline <= clock().astimezone(UTC)
            progress_due = (
                not ready and progress_wait is not None and (delivery_wait is None or progress_wait <= delivery_wait)
            )
            reconcile = progress_due
            delivery_trigger = False
            socket_replaced = False
            for key, _mask in ready:
                source_kind, _identity = key.data
                if source_kind == "pid":
                    reconcile = True
                    delivery_trigger = True
                    continue
                for path in source.read():
                    if socket_path is not None and path == socket_path:
                        socket_replaced = True
                    if is_controller_source(path.name):
                        reconcile = True
                        delivery_trigger = delivery_trigger or path.name in DELIVERY_SOURCE_FILENAMES
            if not reconcile and not socket_replaced and not deadline_due:
                continue
            if reconcile:
                reconciliation = reconcile_managed_root(managed_root, progress_timeout=progress_timeout)
                report_new_problems(reconciliation.problems)
                refresh_pidfds(reconciliation.active_pids)
                delivery_trigger = delivery_trigger or bool(reconciliation.created_kinds or reconciliation.queued)
            if socket_replaced:
                delivery_deferred = False
            if not delivery_deferred and (socket_replaced or delivery_trigger or deadline_due):
                deliver()
    finally:
        for pidfd in registered_pidfds.values():
            os.close(pidfd)
        selector.close()
        source.close()
