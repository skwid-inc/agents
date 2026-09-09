from __future__ import annotations

import array
import asyncio
import contextlib
import errno
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import select
import signal
import socket
import stat
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from livekit.agents.ipc import channel, proto, stack_dump, supervised_proc
from livekit.agents.ipc.job_proc_executor import ProcJobExecutor
from livekit.agents.utils.aio import duplex_unix

FIXTURE = Path(__file__).parent / "fixtures" / "stack_dump_child.py"
LINUX = stack_dump.stack_dump_supported()
linux = pytest.mark.skipif(not LINUX, reason="requires Linux anonymous-FD diagnostics")


def test_required_linux_gate_is_not_silently_skipped():
    if os.getenv("STACK_DUMP_LINUX_REQUIRED") == "1":
        assert LINUX and sys.version_info[:2] == (3, 13)
        assert __import__("platform").machine() == "aarch64"


def test_descriptor_transport_has_no_pathname_cleanup():
    """The former strict-XFAIL race has no pathname operation to interleave."""
    source = inspect.getsource(stack_dump)
    for forbidden in ("os.unlink", "os.rmdir", "mkdtemp", "relative_basename", "directory_path"):
        assert forbidden not in source
    assert set(stack_dump.StackDumpInit.__dataclass_fields__) == {"enabled", "episode_token"}


def _round_trip(message):
    return channel._read_message(channel._write_message(message), proto.IPC_MESSAGES)


def test_initialize_protocol_round_trips_complete_stack_dump_identity():
    init = stack_dump.StackDumpInit(enabled=True, episode_token="a" * 32)
    ready = stack_dump.StackDumpReady(
        ready=True,
        child_pid=404,
        episode_token=init.episode_token,
        file_device=505,
        file_inode=606,
        owner_uid=303,
        mode=0o600,
        link_count=0,
    )
    assert _round_trip(proto.InitializeRequest(stack_dump_init=init)).stack_dump_init == init
    assert _round_trip(proto.InitializeResponse(stack_dump_ready=ready)).stack_dump_ready == ready


def test_initialize_protocol_defaults_and_exact_base_bytes_are_disabled():
    import io

    # Base has only these fields and ignores trailing bytes from the new encoder.
    request = io.BytesIO()
    channel.write_int(request, 0)
    channel.write_bool(request, False)
    for _ in range(3):
        channel.write_float(request, 0)
    response = io.BytesIO()
    channel.write_int(response, 1)
    channel.write_string(response, "")
    assert channel._read_message(request.getvalue(), proto.IPC_MESSAGES).stack_dump_init is None
    assert (
        channel._read_message(response.getvalue(), proto.IPC_MESSAGES).stack_dump_ready
        == stack_dump.StackDumpReady.disabled()
    )
    assert channel._write_message(proto.InitializeRequest()).startswith(request.getvalue())
    assert channel._write_message(proto.InitializeResponse()).startswith(response.getvalue())


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes "])
def test_flag_accepts_only_explicit_true_values(monkeypatch, value):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, value)
    assert stack_dump.stack_dump_enabled()


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "on", "enabled"])
def test_flag_rejects_other_values(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(stack_dump.ENABLED_ENV, raising=False)
    else:
        monkeypatch.setenv(stack_dump.ENABLED_ENV, value)
    assert not stack_dump.stack_dump_enabled()


def test_disabled_and_unsupported_setup_never_open_a_sink(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("unexpected diagnostic open")

    monkeypatch.setattr(stack_dump.os, "open", unexpected)
    monkeypatch.delenv(stack_dump.ENABLED_ENV, raising=False)
    assert not stack_dump.install_stack_dump_signal_handler(stack_dump.StackDumpInit(), None).ready
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    monkeypatch.setattr(stack_dump, "stack_dump_supported", lambda: False)
    assert (
        stack_dump.install_stack_dump_signal_handler(
            stack_dump.StackDumpInit(enabled=True), None
        ).failure_class
        == "unsupported_platform"
    )


def _fd_count():
    return len(os.listdir("/proc/self/fd"))


def _anonymous():
    return os.open("/dev/shm", os.O_TMPFILE | os.O_EXCL | os.O_RDWR | os.O_CLOEXEC, 0o600)


def _ready(fd, token):
    info = os.fstat(fd)
    return stack_dump.StackDumpReady(
        ready=True,
        child_pid=os.getpid(),
        episode_token=token,
        file_device=info.st_dev,
        file_inode=info.st_ino,
        owner_uid=info.st_uid,
        mode=stat.S_IMODE(info.st_mode),
        link_count=info.st_nlink,
    )


@linux
def test_child_anonymous_sink_registration_transfer_and_close(monkeypatch):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    before = _fd_count()
    names = os.listdir("/dev/shm")
    init, parent, child = stack_dump.create_stack_dump_channel()
    ready = stack_dump.install_stack_dump_signal_handler(init, child)
    fd = None
    try:
        assert ready.ready and ready.link_count == 0 and ready.mode == 0o600
        validated, fd = stack_dump.receive_stack_dump_fd(
            parent, init, ready, expected_pid=os.getpid()
        )
        assert validated == ready and fd is not None
        assert not os.get_inheritable(fd)
        os.kill(os.getpid(), signal.SIGUSR1)
        stack_dump.close_stack_dump_signal_handler()
        collected = stack_dump.collect_stack_dump_fd(fd, ready)
        fd = None
        assert "test_child_anonymous_sink_registration_transfer_and_close" in collected.stack_text
        assert collected.bytes_read > 0
    finally:
        stack_dump.close_stack_dump_signal_handler()
        parent.close()
        child.close()
        if fd is not None:
            os.close(fd)
    assert os.listdir("/dev/shm") == names
    assert _fd_count() == before


@linux
@pytest.mark.parametrize("failure", ["open", "register", "full", "closed"])
def test_producer_failure_closes_every_fd_and_unregisters(monkeypatch, failure):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    before = _fd_count()
    for _ in range(5):
        init, parent, child = stack_dump.create_stack_dump_channel()
        with monkeypatch.context() as patch:
            if failure == "open":

                def fail_open(*_args, **_kwargs):
                    raise OSError(errno.EOPNOTSUPP, "unsupported mount")

                patch.setattr(stack_dump.os, "open", fail_open)
            if failure == "register":

                def fail_register(*_args, **_kwargs):
                    raise RuntimeError("registration failed")

                patch.setattr(stack_dump.faulthandler, "register", fail_register)
            if failure == "full":
                with contextlib.suppress(BlockingIOError):
                    while True:
                        child.send(b"x")
            if failure == "closed":
                parent.close()
            result = stack_dump.install_stack_dump_signal_handler(init, child)
            assert not result.ready and not stack_dump.stack_dump_signal_handler_ready()
        parent.close()
        child.close()
    assert _fd_count() == before


@linux
@pytest.mark.parametrize(
    "error", [errno.EINVAL, errno.EISDIR, errno.ENOSYS, errno.ENOSPC, errno.EACCES]
)
def test_unsupported_kernel_or_mount_has_no_fallback(monkeypatch, error):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, parent, child = stack_dump.create_stack_dump_channel()
    calls = []

    def fail_open(path, flags, mode):
        calls.append((path, flags, mode))
        raise OSError(error, "injected")

    monkeypatch.setattr(stack_dump.os, "open", fail_open)
    assert not stack_dump.install_stack_dump_signal_handler(init, child).ready
    assert len(calls) == 1 and calls[0][0] == "/dev/shm"
    parent.close()


@linux
@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "two",
        "many",
        "truncated",
        "json",
        "pid",
        "token",
        "inode",
        "mode",
        "linked",
        "pipe",
        "credentials",
        "writeonly",
    ],
)
def test_receiver_rejects_bad_transfer_without_leaks(tmp_path, case):
    before = _fd_count()
    for _ in range(3):
        init, parent, child = stack_dump.create_stack_dump_channel()
        fd = _anonymous()
        extra = None
        if case == "pipe":
            os.close(fd)
            fd, extra = os.pipe()
        if case in ("linked", "writeonly"):
            os.close(fd)
            fd = os.open(tmp_path / "fixture", os.O_CREAT | os.O_RDWR, 0o600)
            if case == "writeonly":
                os.close(fd)
                fd = os.open(tmp_path / "fixture", os.O_WRONLY)
                os.unlink(tmp_path / "fixture")
        ready = _ready(fd, init.episode_token)
        if case == "inode":
            ready = replace(ready, file_inode=ready.file_inode + 1)
        if case == "mode":
            os.fchmod(fd, 0o640)
        if case == "pid":
            ready = replace(ready, child_pid=os.getpid() + 1)
        if case == "token":
            ready = replace(ready, episode_token="b" * 32)
        payload = json.dumps(asdict(ready)).encode()
        if case == "json":
            payload = b"not json"
        if case == "truncated":
            payload += b" " * 2000
        if case == "credentials":
            parent.setsockopt(socket.SOL_SOCKET, socket.SO_PASSCRED, 0)
        if case != "missing":
            count = 2 if case == "two" else 64 if case == "many" else 1
            child.sendmsg(
                [payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd] * count))]
            )
        validated, received = stack_dump.receive_stack_dump_fd(
            parent, init, ready, expected_pid=os.getpid()
        )
        assert not validated.ready and received is None
        os.close(fd)
        if extra is not None:
            os.close(extra)
        child.close()
    assert _fd_count() == before


@linux
def test_pread_bound_shared_offset_and_read_failure_cleanup(monkeypatch):
    fd = _anonymous()
    os.write(fd, b"x" * (3 * stack_dump.MAX_STACK_DUMP_BYTES))
    ready = _ready(fd, "a" * 32)
    duplicate = os.dup(fd)
    original_offset = os.lseek(duplicate, 0, os.SEEK_CUR)
    actual_pread = os.pread
    reads = []

    def measured_read(actual_fd, size, offset):
        value = actual_pread(actual_fd, size, offset)
        reads.append(len(value))
        return value

    monkeypatch.setattr(stack_dump.os, "pread", measured_read)
    result = stack_dump.collect_stack_dump_fd(fd, ready)
    assert result.bytes_read == 65536 and result.truncated and sum(reads) == 65537
    assert os.lseek(duplicate, 0, os.SEEK_CUR) == original_offset
    with pytest.raises(OSError):
        os.fstat(fd)

    def fail_read(*_args):
        raise OSError(errno.EIO, "injected")

    monkeypatch.setattr(stack_dump.os, "pread", fail_read)
    assert (
        stack_dump.collect_stack_dump_fd(duplicate, ready).failure_class == "artifact_read_failed"
    )
    with pytest.raises(OSError):
        os.fstat(duplicate)


class _FakeHandle:
    def __init__(self, callback):
        self.callback, self.cancelled = callback, False

    def cancel(self):
        self.cancelled = True


class _FakeLoop:
    def __init__(self):
        self.calls = []

    def call_later(self, delay, callback):
        handle = _FakeHandle(callback)
        self.calls.append((delay, handle))
        return handle


def test_trigger_one_request_recovery_close_and_short_deadline():
    loop = _FakeLoop()
    requests = []
    trigger = stack_dump.PongStallDumpTrigger(
        loop, ping_timeout=60, request_dump=lambda: requests.append(1)
    )
    trigger.arm()
    first = loop.calls[-1][1]
    trigger.pong()
    assert first.cancelled and loop.calls[-1][0] == 59.5
    loop.calls[-1][1].callback()
    trigger.pong()
    trigger.arm()
    assert requests == [1] and len(loop.calls) == 2
    second = stack_dump.PongStallDumpTrigger(
        loop, ping_timeout=60, request_dump=lambda: requests.append(2)
    )
    second.arm()
    second.close()
    loop.calls[-1][1].callback()
    assert requests == [1]
    short = stack_dump.PongStallDumpTrigger(loop, ping_timeout=0.5, request_dump=lambda: None)
    short.arm()
    assert len(loop.calls) == 3


def test_protected_kill_method_remains_synchronous_and_unchanged():
    import ast

    source = inspect.getsource(supervised_proc)
    node = next(
        n
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == "_send_kill_signal"
    )
    digest = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
    assert digest == "cc435cedf3ac7483e84b93cdbe9ffd1cb1ba2d08debfa8c85c1b05ab8905a0ee"
    assert not inspect.iscoroutinefunction(supervised_proc.SupervisedProc._send_kill_signal)


def _event(process, timeout=2):
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    if not readable:
        raise TimeoutError("fixture emitted no event")
    line = process.stdout.readline()
    if not line:
        raise EOFError(f"fixture exited: {process.poll()}")
    return json.loads(line)


def _command(process, command):
    process.stdin.write(command + "\n")
    process.stdin.flush()


@contextlib.contextmanager
def _fixture(label="alpha", *, saturate=False, heartbeat=False, stderr_sink=False):
    init, parent, child = stack_dump.create_stack_dump_channel()
    command = [
        sys.executable,
        str(FIXTURE),
        "--init",
        json.dumps(asdict(init)),
        "--label",
        label,
        "--transfer-fd",
        str(child.fileno()),
    ]
    if saturate:
        command.append("--saturate-stderr")
    if heartbeat:
        command.append("--heartbeat")
    if stderr_sink:
        command.append("--stderr-sink")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if saturate else subprocess.DEVNULL,
        text=True,
        bufsize=1,
        pass_fds=(child.fileno(),),
        env={**os.environ, stack_dump.ENABLED_ENV: "true"},
    )
    child.close()
    state = {"fd": None}
    try:
        event = _event(process)
        assert event["event"] == "ready", event
        ready = stack_dump.StackDumpReady(**event["identity"]) if not stderr_sink else None
        if ready is not None:
            ready, state["fd"] = stack_dump.receive_stack_dump_fd(
                parent, init, ready, expected_pid=process.pid
            )
            assert ready.ready
        yield process, state, ready, event
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=3)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        if state["fd"] is not None:
            os.close(state["fd"])
        parent.close()


def _collect_fixture(process, state, ready, *, kill=False):
    if kill:
        process.kill()
    else:
        _command(process, "stop")
    process.wait(timeout=3)
    fd, state["fd"] = state["fd"], None
    return stack_dump.collect_stack_dump_fd(fd, ready)


@linux
def test_real_child_recovers_with_saturated_stderr():
    with _fixture(saturate=True) as (process, state, ready, event):
        started = time.monotonic()
        os.kill(process.pid, signal.SIGUSR1)
        _command(process, "recover")
        assert _event(process, 0.25)["event"] == "pong"
        assert time.monotonic() - started < 0.25
        result = _collect_fixture(process, state, ready)
        assert event["stderr_filled_bytes"] > 0
        assert result.stack_text.count("alpha_unique_anchor") >= 3
        assert "Current thread" not in process.stderr.read()


@linux
def test_inherited_stderr_control_blocks_recovery_when_pipe_is_full():
    with _fixture(saturate=True, stderr_sink=True) as (process, _state, _ready, event):
        assert event["stderr_filled_bytes"] > 0
        os.kill(process.pid, signal.SIGUSR1)
        _command(process, "recover")
        with pytest.raises(TimeoutError):
            _event(process, 0.1)


@linux
def test_two_children_attributed_while_third_child_progresses():
    with (
        _fixture("alpha") as alpha,
        _fixture("beta") as beta,
        _fixture("progress", heartbeat=True) as third,
    ):
        for process, _state, _ready, _event_data in (alpha, beta):
            os.kill(process.pid, signal.SIGUSR1)
            _command(process, "recover")
            assert _event(process, 0.25)["event"] == "pong"
        a = _collect_fixture(*alpha[:3])
        emitted_at = time.monotonic_ns()
        for _ in range(30):
            if _event(third[0], 0.25)["monotonic_ns"] > emitted_at:
                break
        else:
            raise AssertionError("unrelated child failed to progress")
        b = _collect_fixture(*beta[:3])
        c = _collect_fixture(*third[:3])
        assert "alpha_unique_anchor" in a.stack_text and "beta_unique_anchor" not in a.stack_text
        assert "beta_unique_anchor" in b.stack_text and "alpha_unique_anchor" not in b.stack_text
        assert c.bytes_read == 0


@linux
def test_real_held_interpreter_dump_survives_sigkill():
    with _fixture() as (process, state, ready, _event_data):
        _command(process, "hold")
        assert _event(process)["event"] == "holding"
        os.kill(process.pid, signal.SIGUSR1)
        time.sleep(0.1)
        result = _collect_fixture(process, state, ready, kill=True)
        assert "hold_interpreter" in result.stack_text


@linux
def test_real_capture_latency_budget():
    count = int(os.getenv("STACK_DUMP_STRESS_COUNT", "5"))
    before = _fd_count()
    for index in range(count):
        with _fixture(f"stress-{index}") as (process, state, ready, _event_data):
            started = time.monotonic()
            os.kill(process.pid, signal.SIGUSR1)
            _command(process, "recover")
            assert _event(process, 0.25)["event"] == "pong"
            assert time.monotonic() - started < 0.25
            assert "shared_anchor" in _collect_fixture(process, state, ready).stack_text
    assert _fd_count() == before


def _executor(
    monkeypatch, *, enabled=True, context="spawn", initializer="initialize_job_process", timeout=5
):
    monkeypatch.syspath_prepend(str(FIXTURE.parent))
    import stack_dump_child

    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true" if enabled else "false")
    return ProcJobExecutor(
        initialize_process_fnc=getattr(stack_dump_child, initializer),
        job_entrypoint_fnc=stack_dump_child.unused_job_entrypoint,
        inference_executor=None,
        initialize_timeout=timeout,
        close_timeout=1,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=0.05,
        ping_timeout=1.2,
        high_ping_threshold=1,
        mp_ctx=mp.get_context(context),
        loop=asyncio.get_running_loop(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_process_job_disabled_or_unsupported_is_inert(monkeypatch, enabled):
    if enabled and LINUX:
        monkeypatch.setattr(
            "livekit.agents.ipc.job_proc_executor.stack_dump_supported", lambda: False
        )
    executor = _executor(monkeypatch, enabled=enabled)
    await executor.start()
    try:
        await executor.initialize()
        assert executor._stack_dump_init is None
        assert (
            executor._stack_dump_pch is executor._stack_dump_cch is executor._stack_dump_fd is None
        )
        assert executor._stack_dump_trigger is executor._stack_dump_request_record is None
        executor._request_stack_dump()
        assert executor._stack_dump_request_record is None
    finally:
        await asyncio.wait_for(executor.kill(), 5)


@linux
@pytest.mark.asyncio
@pytest.mark.parametrize("context", ["spawn", "forkserver", "fork"])
async def test_real_process_job_transfer_one_request_post_exit_record(monkeypatch, caplog, context):
    executor = _executor(monkeypatch, context=context)
    await executor.start()
    try:
        await executor.initialize()
        await asyncio.sleep(0.1)
        assert (
            executor._stack_dump_ready.ready
            and executor._stack_dump_ready.child_pid == executor.pid
        )
        assert executor._stack_dump_pch is executor._stack_dump_cch is None
        assert executor._stack_dump_fd is not None
        with caplog.at_level("WARNING", logger="livekit.agents"):
            executor._request_stack_dump()
            first = executor._stack_dump_request_record
            executor._request_stack_dump()
            assert executor._stack_dump_request_record is first and first.sent
            await asyncio.sleep(0.1)
            assert not any(
                getattr(r, "diagnostic_event", "") == "job_stack_dump_collected"
                for r in caplog.records
            )
            await executor.kill()
        records = [
            r
            for r in caplog.records
            if getattr(r, "diagnostic_event", "") == "job_stack_dump_collected"
        ]
        assert len(records) == 1 and records[0].child_pid == executor.pid
        assert "Current thread" in records[0].stack_text and records[0].bytes_read <= 65536
        assert executor._stack_dump_fd is None
        executor._collect_and_emit_stack_dump()
        assert (
            len(
                [
                    r
                    for r in caplog.records
                    if getattr(r, "diagnostic_event", "") == "job_stack_dump_collected"
                ]
            )
            == 1
        )
    finally:
        if executor.exitcode is None:
            await asyncio.wait_for(executor.kill(), 5)


@linux
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["cancel", "timeout", "crash", "error", "no_init"])
async def test_real_process_job_init_failure_releases_queued_fd(monkeypatch, failure):
    # Warm multiprocessing's persistent resource helper before measuring.
    warm = _executor(monkeypatch, enabled=False)
    await warm.start()
    await warm.initialize()
    await warm.kill()
    before = _fd_count()
    for _ in range(3):
        initializer = {
            "cancel": "stall_job_process",
            "timeout": "stall_job_process",
            "crash": "crash_job_process",
            "error": "fail_job_process",
        }.get(failure, "initialize_job_process")
        executor = _executor(
            monkeypatch, initializer=initializer, timeout=0.5 if failure == "timeout" else 5
        )
        await executor.start()
        if failure == "no_init":
            await asyncio.wait_for(executor.kill(), 5)
        else:
            receiver = executor._stack_dump_pch
            task = asyncio.create_task(executor.initialize())
            if failure == "cancel":
                for _ in range(400):
                    if select.select([receiver], [], [], 0)[0]:
                        break
                    await asyncio.sleep(0.005)
                else:
                    raise AssertionError("child never queued the descriptor")
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                expected = (
                    asyncio.TimeoutError
                    if failure == "timeout"
                    else (duplex_unix.DuplexClosed if failure == "crash" else RuntimeError)
                )
                with pytest.raises(expected):
                    await task
            await asyncio.wait_for(executor.kill(), 5)
        assert (
            executor._stack_dump_pch is executor._stack_dump_cch is executor._stack_dump_fd is None
        )
    assert _fd_count() == before


@linux
@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["full", "closed", "setup"])
async def test_real_process_job_side_failure_preserves_normal_initialization(monkeypatch, failure):
    from livekit.agents.ipc import job_proc_executor

    real_create = job_proc_executor.create_stack_dump_channel

    def create():
        if failure == "setup":
            raise OSError(errno.EMFILE, "injected socketpair failure")
        init, parent, child = real_create()
        if failure == "full":
            with contextlib.suppress(BlockingIOError):
                while True:
                    child.send(b"x")
        return init, parent, child

    monkeypatch.setattr(job_proc_executor, "create_stack_dump_channel", create)
    executor = _executor(monkeypatch)
    await executor.start()
    try:
        if failure == "closed":
            executor._stack_dump_pch.close()
        await executor.initialize()
        await asyncio.sleep(0.2)
        assert not executor._stack_dump_ready.ready
        assert executor._stack_dump_fd is None and not executor.killed
        await executor.aclose()
        assert executor.exitcode == 0
    finally:
        if executor.exitcode is None:
            await executor.kill()


@linux
@pytest.mark.asyncio
async def test_forked_job_does_not_retain_other_child_descriptor(monkeypatch):
    first = _executor(monkeypatch)
    await first.start()
    second = None
    try:
        await first.initialize()
        info = os.fstat(first._stack_dump_fd)
        second = _executor(monkeypatch, context="fork")
        await second.start()
        await second.initialize()
        inherited = []
        for entry in Path(f"/proc/{second.pid}/fd").iterdir():
            with contextlib.suppress(FileNotFoundError):
                actual = entry.stat()
                if (actual.st_dev, actual.st_ino) == (info.st_dev, info.st_ino):
                    inherited.append(entry.name)
        assert inherited == []
        assert first._stack_dump_fd is not None
    finally:
        if second is not None:
            await second.kill()
        await first.kill()


@linux
@pytest.mark.asyncio
async def test_real_supervisor_recovery_and_original_second_stall_deadline(monkeypatch, caplog):
    executor = _executor(monkeypatch)
    await executor.start()
    try:
        await executor.initialize()
        await asyncio.sleep(0.15)
        os.kill(executor.pid, signal.SIGSTOP)
        for _ in range(100):
            if executor._stack_dump_request_record is not None:
                break
            await asyncio.sleep(0.01)
        assert executor._stack_dump_request_record.sent
        os.kill(executor.pid, signal.SIGCONT)
        await asyncio.sleep(0.6)
        assert not executor.killed and executor.exitcode is None
        first = executor._stack_dump_request_record
        os.kill(executor.pid, signal.SIGSTOP)
        started = time.monotonic()
        with caplog.at_level("WARNING", logger="livekit.agents"):
            await asyncio.wait_for(executor.join(), 2)
        elapsed = time.monotonic() - started
        assert 0.95 < elapsed < 1.45
        assert executor.killed and executor._stack_dump_request_record is first
        assert any(
            getattr(r, "diagnostic_event", "") == "job_stack_dump_collected" for r in caplog.records
        )
    finally:
        if executor.exitcode is None:
            await asyncio.wait_for(executor.kill(), 5)
