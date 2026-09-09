from __future__ import annotations

import asyncio
import contextlib
import faulthandler
import os
import secrets
import signal
import stat
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

ENABLED_ENV = "LK_DUMP_STACK_TRACES"
TRUE_VALUES = frozenset({"1", "true", "yes"})
STACK_DUMP_LEAD_SECONDS = 0.5
MAX_STACK_DUMP_BYTES = 64 * 1024
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


@dataclass(frozen=True)
class StackDumpInit:
    enabled: bool = False
    directory_path: str = ""
    directory_device: int = 0
    directory_inode: int = 0
    directory_owner_uid: int = 0
    directory_mode: int = 0
    episode_token: str = ""


@dataclass(frozen=True)
class StackDumpReady:
    ready: bool = False
    child_pid: int = 0
    episode_token: str = ""
    relative_basename: str = ""
    directory_device: int = 0
    directory_inode: int = 0
    file_device: int = 0
    file_inode: int = 0
    owner_uid: int = 0
    mode: int = 0
    link_count: int = 0
    failure_class: str | None = None

    @classmethod
    def disabled(cls, failure_class: str | None = None) -> StackDumpReady:
        return cls(failure_class=failure_class)


@dataclass(frozen=True)
class StackDumpRequestRecord:
    child_pid: int
    episode_token: str
    requested_at_unix_ms: int
    last_pong_age_ms: int
    sent: bool
    failure_class: str | None = None


@dataclass(frozen=True)
class StackDumpCollection:
    stack_text: str | None = None
    bytes_read: int = 0
    truncated: bool = False
    failure_class: str | None = None


@dataclass
class _StackDumpProducer:
    stream: TextIO
    directory_fd: int
    ready: StackDumpReady
    created: bool


_producer: _StackDumpProducer | None = None


def stack_dump_enabled() -> bool:
    return os.getenv(ENABLED_ENV, "").strip().lower() in TRUE_VALUES


def stack_dump_signal_handler_ready() -> bool:
    return _producer is not None


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _file_flags(access: int) -> int:
    return access | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


def _read_file_flags() -> int:
    return _file_flags(os.O_RDONLY) | getattr(os, "O_NONBLOCK", 0)


def _valid_directory(info: os.stat_result, init: StackDumpInit) -> bool:
    return (
        stat.S_ISDIR(info.st_mode)
        and info.st_dev == init.directory_device
        and info.st_ino == init.directory_inode
        and info.st_uid == init.directory_owner_uid
        and stat.S_IMODE(info.st_mode) == PRIVATE_DIRECTORY_MODE
        and init.directory_mode == PRIVATE_DIRECTORY_MODE
    )


def _valid_file(info: os.stat_result, ready: StackDumpReady) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_dev == ready.file_device
        and info.st_ino == ready.file_inode
        and info.st_uid == ready.owner_uid
        and stat.S_IMODE(info.st_mode) == PRIVATE_FILE_MODE
        and ready.mode == PRIVATE_FILE_MODE
        and info.st_nlink == 1
        and ready.link_count == 1
    )


def create_stack_dump_init(base_directory: str = "/dev/shm") -> tuple[StackDumpInit, int]:
    token = secrets.token_hex(16)
    directory_path = tempfile.mkdtemp(prefix=f"livekit-stack-{token}-", dir=base_directory)
    directory_fd: int | None = None
    try:
        os.chmod(directory_path, PRIVATE_DIRECTORY_MODE)
        directory_fd = os.open(directory_path, _directory_flags())
        info = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.geteuid()
            or stat.S_IMODE(info.st_mode) != PRIVATE_DIRECTORY_MODE
        ):
            raise PermissionError("invalid stack dump directory ownership")
        return (
            StackDumpInit(
                enabled=True,
                directory_path=directory_path,
                directory_device=info.st_dev,
                directory_inode=info.st_ino,
                directory_owner_uid=info.st_uid,
                directory_mode=stat.S_IMODE(info.st_mode),
                episode_token=token,
            ),
            directory_fd,
        )
    except BaseException:
        if directory_fd is not None:
            os.close(directory_fd)
        with contextlib.suppress(OSError):
            os.rmdir(directory_path)
        raise


def install_stack_dump_signal_handler(init: StackDumpInit) -> StackDumpReady:
    global _producer

    if not init.enabled or not stack_dump_enabled():
        return StackDumpReady.disabled("disabled")
    if sys.platform == "win32" or not hasattr(signal, "SIGUSR1"):
        return StackDumpReady.disabled("unsupported_platform")
    if _producer is not None:
        return StackDumpReady.disabled("already_installed")

    directory_fd: int | None = None
    artifact_fd: int | None = None
    stream: TextIO | None = None
    ready: StackDumpReady | None = None
    created = False
    basename = f"stack-{os.getpid()}-{init.episode_token}.dump"
    failure_class = "directory_invalid"
    try:
        directory_fd = os.open(init.directory_path, _directory_flags())
        directory_info = os.fstat(directory_fd)
        if not _valid_directory(directory_info, init):
            raise PermissionError("stack dump directory identity mismatch")

        failure_class = "file_open_failed"
        artifact_fd = os.open(
            basename,
            _file_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
            PRIVATE_FILE_MODE,
            dir_fd=directory_fd,
        )
        created = True
        file_info = os.fstat(artifact_fd)
        ready = StackDumpReady(
            ready=True,
            child_pid=os.getpid(),
            episode_token=init.episode_token,
            relative_basename=basename,
            directory_device=directory_info.st_dev,
            directory_inode=directory_info.st_ino,
            file_device=file_info.st_dev,
            file_inode=file_info.st_ino,
            owner_uid=file_info.st_uid,
            mode=stat.S_IMODE(file_info.st_mode),
            link_count=file_info.st_nlink,
        )
        failure_class = "file_invalid"
        if not _valid_file(file_info, ready) or file_info.st_uid != os.geteuid():
            raise PermissionError("stack dump file ownership mismatch")

        failure_class = "file_wrap_failed"
        stream = os.fdopen(artifact_fd, "w", encoding="utf-8", buffering=1)
        artifact_fd = None
        failure_class = "handler_registration_failed"
        faulthandler.register(signal.SIGUSR1, file=stream, all_threads=True, chain=False)
        _producer = _StackDumpProducer(
            stream=stream,
            directory_fd=directory_fd,
            ready=ready,
            created=created,
        )
        directory_fd = None
        stream = None
        return ready
    except (OSError, RuntimeError, ValueError):
        if stream is not None:
            stream.close()
        if artifact_fd is not None:
            os.close(artifact_fd)
        if created and directory_fd is not None and ready is not None:
            _unlink_owned_file(directory_fd, ready)
        if directory_fd is not None:
            os.close(directory_fd)
        return StackDumpReady.disabled(failure_class)


def _unlink_owned_file(directory_fd: int, ready: StackDumpReady) -> bool:
    try:
        info = os.stat(ready.relative_basename, dir_fd=directory_fd, follow_symlinks=False)
        if not _valid_file(info, ready):
            return False
        os.unlink(ready.relative_basename, dir_fd=directory_fd)
        return True
    except OSError:
        return False


def close_stack_dump_signal_handler(*, unlink: bool) -> None:
    global _producer

    producer = _producer
    _producer = None
    if producer is None:
        return

    with contextlib.suppress(OSError, RuntimeError):
        faulthandler.unregister(signal.SIGUSR1)
    producer.stream.close()
    if unlink and producer.created:
        _unlink_owned_file(producer.directory_fd, producer.ready)
    os.close(producer.directory_fd)


def validate_stack_dump_ready(
    init: StackDumpInit,
    ready: StackDumpReady,
    *,
    expected_pid: int,
    directory_fd: int,
) -> StackDumpReady:
    expected_basename = f"stack-{expected_pid}-{init.episode_token}.dump"
    if (
        not init.enabled
        or not ready.ready
        or ready.child_pid != expected_pid
        or ready.episode_token != init.episode_token
        or ready.relative_basename != expected_basename
        or ready.directory_device != init.directory_device
        or ready.directory_inode != init.directory_inode
        or ready.owner_uid != init.directory_owner_uid
    ):
        return StackDumpReady.disabled("identity_mismatch")

    try:
        if not _valid_directory(os.fstat(directory_fd), init):
            return StackDumpReady.disabled("identity_mismatch")
        artifact_fd = os.open(
            ready.relative_basename,
            _read_file_flags(),
            dir_fd=directory_fd,
        )
        try:
            if not _valid_file(os.fstat(artifact_fd), ready):
                return StackDumpReady.disabled("identity_mismatch")
        finally:
            os.close(artifact_fd)
    except OSError:
        return StackDumpReady.disabled("identity_mismatch")
    return ready


def collect_stack_dump_artifact(
    init: StackDumpInit,
    ready: StackDumpReady,
    directory_fd: int,
) -> StackDumpCollection:
    artifact_fd: int | None = None
    identity_validated = False
    try:
        if not _valid_directory(os.fstat(directory_fd), init):
            return StackDumpCollection(failure_class="directory_identity_mismatch")
        artifact_fd = os.open(
            ready.relative_basename,
            _read_file_flags(),
            dir_fd=directory_fd,
        )
        if not _valid_file(os.fstat(artifact_fd), ready):
            return StackDumpCollection(failure_class="artifact_identity_mismatch")
        identity_validated = True

        chunks: list[bytes] = []
        remaining = MAX_STACK_DUMP_BYTES + 1
        while remaining:
            chunk = os.read(artifact_fd, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
        truncated = len(raw) > MAX_STACK_DUMP_BYTES
        retained = raw[:MAX_STACK_DUMP_BYTES]
        return StackDumpCollection(
            stack_text=retained.decode("utf-8", errors="replace"),
            bytes_read=len(retained),
            truncated=truncated,
        )
    except FileNotFoundError:
        return StackDumpCollection(failure_class="artifact_missing")
    except OSError:
        return StackDumpCollection(failure_class="artifact_read_failed")
    finally:
        if artifact_fd is not None:
            os.close(artifact_fd)
        if identity_validated:
            _unlink_owned_file(directory_fd, ready)


def close_stack_dump_directory(init: StackDumpInit, directory_fd: int) -> None:
    try:
        info = os.fstat(directory_fd)
    except OSError:
        return
    finally:
        with contextlib.suppress(OSError):
            os.close(directory_fd)

    try:
        path_info = os.stat(init.directory_path, follow_symlinks=False)
        if _valid_directory(info, init) and _valid_directory(path_info, init):
            os.rmdir(init.directory_path)
    except OSError:
        pass


class PongStallDumpTrigger:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        ping_timeout: float,
        request_dump: Callable[[], None],
    ) -> None:
        self._loop = loop
        self._delay = ping_timeout - STACK_DUMP_LEAD_SECONDS
        self._request_dump = request_dump
        self._handle: asyncio.TimerHandle | None = None
        self._closed = False
        self._requested = False

    @property
    def requested(self) -> bool:
        return self._requested

    def arm(self) -> None:
        if not self._closed and not self._requested and self._delay > 0:
            self._replace_timer()

    def pong(self) -> None:
        if self._closed or self._requested or self._delay <= 0:
            return
        self._replace_timer()

    def close(self) -> None:
        self._closed = True
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def _replace_timer(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
        self._handle = self._loop.call_later(self._delay, self._fire)

    def _fire(self) -> None:
        self._handle = None
        if self._closed or self._requested:
            return
        self._requested = True
        try:
            self._request_dump()
        except Exception:
            pass
