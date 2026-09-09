from __future__ import annotations

import array
import asyncio
import contextlib
import faulthandler
import json
import os
import secrets
import signal
import socket
import stat
import struct
import sys
import weakref
from collections.abc import Callable
from dataclasses import asdict, dataclass

ENABLED_ENV = "LK_DUMP_STACK_TRACES"
TRUE_VALUES = frozenset({"1", "true", "yes"})
STACK_DUMP_LEAD_SECONDS = 0.5
MAX_STACK_DUMP_BYTES = 64 * 1024
PRIVATE_FILE_MODE = 0o600
MAX_TRANSFER_BYTES = 1024


@dataclass(frozen=True)
class StackDumpInit:
    enabled: bool = False
    episode_token: str = ""


@dataclass(frozen=True)
class StackDumpReady:
    ready: bool = False
    child_pid: int = 0
    episode_token: str = ""
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


_producer_fd: int | None = None
_parent_fds: set[int] = set()
_transfer_sockets: weakref.WeakSet[socket.socket] = weakref.WeakSet()


def close_inherited_stack_dump_resources(keep: socket.socket | None) -> None:
    """A forked job must not retain other jobs' diagnostic resources."""
    for sock in tuple(_transfer_sockets):
        if sock is not keep:
            sock.close()
    _transfer_sockets.clear()
    for fd in tuple(_parent_fds):
        close_stack_dump_fd(fd)


def close_stack_dump_fd(fd: int) -> None:
    _parent_fds.discard(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


def stack_dump_enabled() -> bool:
    return os.getenv(ENABLED_ENV, "").strip().lower() in TRUE_VALUES


def stack_dump_supported() -> bool:
    return sys.platform == "linux" and all(
        hasattr(module, name)
        for module, name in (
            (os, "O_TMPFILE"),
            (signal, "SIGUSR1"),
            (socket, "SCM_RIGHTS"),
            (socket, "SCM_CREDENTIALS"),
            (socket, "SO_PASSCRED"),
            (socket, "MSG_CMSG_CLOEXEC"),
        )
    )


def stack_dump_signal_handler_ready() -> bool:
    return _producer_fd is not None


def create_stack_dump_channel() -> tuple[StackDumpInit, socket.socket, socket.socket]:
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 1)
        parent.setblocking(False)
        child.setblocking(False)
        _transfer_sockets.update((parent, child))
        return StackDumpInit(enabled=True, episode_token=secrets.token_hex(16)), parent, child
    except BaseException:
        parent.close()
        child.close()
        raise


def _valid_file(info: os.stat_result, ready: StackDumpReady) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_dev == ready.file_device
        and info.st_ino == ready.file_inode
        and info.st_uid == ready.owner_uid == os.geteuid()
        and stat.S_IMODE(info.st_mode) == ready.mode == PRIVATE_FILE_MODE
        and info.st_nlink == ready.link_count == 0
    )


def _valid_token(token: str) -> bool:
    return len(token) == 32 and all(char in "0123456789abcdef" for char in token)


def install_stack_dump_signal_handler(
    init: StackDumpInit, transfer_socket: socket.socket | None
) -> StackDumpReady:
    global _producer_fd
    fd: int | None = None
    registered = False
    failure_class = "file_open_failed"
    try:
        if not init.enabled or not stack_dump_enabled():
            return StackDumpReady.disabled("disabled")
        if not stack_dump_supported():
            return StackDumpReady.disabled("unsupported_platform")
        if transfer_socket is None or not _valid_token(init.episode_token):
            return StackDumpReady.disabled("transfer_unavailable")
        if _producer_fd is not None:
            return StackDumpReady.disabled("already_installed")
        # Child-created anonymous inode; O_EXCL forbids later linking. No fallback.
        fd = os.open("/dev/shm", os.O_TMPFILE | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)
        os.fchmod(fd, PRIVATE_FILE_MODE)
        info = os.fstat(fd)
        ready = StackDumpReady(
            ready=True,
            child_pid=os.getpid(),
            episode_token=init.episode_token,
            file_device=info.st_dev,
            file_inode=info.st_ino,
            owner_uid=info.st_uid,
            mode=stat.S_IMODE(info.st_mode),
            link_count=info.st_nlink,
        )
        failure_class = "file_invalid"
        if not _valid_file(info, ready):
            return StackDumpReady.disabled(failure_class)
        failure_class = "handler_registration_failed"
        faulthandler.register(signal.SIGUSR1, file=fd, all_threads=True, chain=False)
        registered = True
        failure_class = "transfer_failed"
        payload = json.dumps(asdict(ready), separators=(",", ":")).encode("utf-8")
        if len(payload) > MAX_TRANSFER_BYTES:
            return StackDumpReady.disabled(failure_class)
        transferred = transfer_socket.sendmsg(
            [payload],
            [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))],
            socket.MSG_DONTWAIT,
        )
        if transferred != len(payload):
            return StackDumpReady.disabled(failure_class)
        _producer_fd, fd = fd, None
        return ready
    except (OSError, RuntimeError, ValueError):
        return StackDumpReady.disabled(failure_class)
    finally:
        if fd is not None:
            if registered:
                with contextlib.suppress(OSError, RuntimeError):
                    faulthandler.unregister(signal.SIGUSR1)
            os.close(fd)
        if transfer_socket is not None:
            transfer_socket.close()


def close_stack_dump_signal_handler() -> None:
    global _producer_fd
    fd, _producer_fd = _producer_fd, None
    if fd is not None:
        with contextlib.suppress(OSError, RuntimeError):
            faulthandler.unregister(signal.SIGUSR1)
        os.close(fd)


def receive_stack_dump_fd(
    transfer_socket: socket.socket,
    init: StackDumpInit,
    ready: StackDumpReady,
    *,
    expected_pid: int,
) -> tuple[StackDumpReady, int | None]:
    """Adopt one FD after ordinary readiness; never wait for ancillary data."""
    fds: list[int] = []
    try:
        if (
            not stack_dump_supported()
            or not init.enabled
            or not ready.ready
            or ready.child_pid != expected_pid
            or ready.episode_token != init.episode_token
            or not _valid_token(init.episode_token)
        ):
            return StackDumpReady.disabled(ready.failure_class or "identity_mismatch"), None
        data, ancillary, flags, _ = transfer_socket.recvmsg(
            MAX_TRANSFER_BYTES,
            socket.CMSG_SPACE(8 * array.array("i").itemsize) + socket.CMSG_SPACE(12),
            socket.MSG_DONTWAIT | socket.MSG_CMSG_CLOEXEC,
        )
        credentials = []
        invalid_ancillary = False
        for level, kind, raw in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                values = array.array("i")
                values.frombytes(raw[: len(raw) - len(raw) % values.itemsize])
                fds.extend(values)
                invalid_ancillary |= len(raw) % values.itemsize != 0
            elif level == socket.SOL_SOCKET and kind == socket.SCM_CREDENTIALS:
                if len(raw) == 12:
                    credentials.append(struct.unpack("3i", raw))
                else:
                    invalid_ancillary = True
            else:
                invalid_ancillary = True
        if (
            flags & (socket.MSG_TRUNC | socket.MSG_CTRUNC)
            or invalid_ancillary
            or len(fds) != 1
            or credentials != [(expected_pid, os.getuid(), os.getgid())]
            or json.loads(data) != asdict(ready)
        ):
            return StackDumpReady.disabled("transfer_invalid"), None
        import fcntl

        if (
            not _valid_file(os.fstat(fds[0]), ready)
            or os.get_inheritable(fds[0])
            or fcntl.fcntl(fds[0], fcntl.F_GETFL) & os.O_ACCMODE not in (os.O_RDONLY, os.O_RDWR)
        ):
            return StackDumpReady.disabled("identity_mismatch"), None
        fd = fds.pop()
        _parent_fds.add(fd)
        return ready, fd
    except (OSError, RuntimeError, ValueError, TypeError):
        return StackDumpReady.disabled("transfer_failed"), None
    finally:
        for fd in fds:
            os.close(fd)
        # Closing also releases descriptors still queued by failed initializers.
        transfer_socket.close()


def collect_stack_dump_fd(fd: int, ready: StackDumpReady) -> StackDumpCollection:
    """Called only after child join. Consume ownership even when validation/read fails."""
    try:
        if not _valid_file(os.fstat(fd), ready):
            return StackDumpCollection(failure_class="artifact_identity_mismatch")
        chunks: list[bytes] = []
        offset = 0
        while offset <= MAX_STACK_DUMP_BYTES:
            chunk = os.pread(fd, min(8192, MAX_STACK_DUMP_BYTES + 1 - offset), offset)
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        raw = b"".join(chunks)
        retained = raw[:MAX_STACK_DUMP_BYTES]
        return StackDumpCollection(
            stack_text=retained.decode("utf-8", errors="replace"),
            bytes_read=len(retained),
            truncated=len(raw) > MAX_STACK_DUMP_BYTES,
        )
    except OSError:
        return StackDumpCollection(failure_class="artifact_read_failed")
    finally:
        close_stack_dump_fd(fd)


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
