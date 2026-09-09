from __future__ import annotations

import inspect
import json
import os
import select
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from livekit.agents.ipc import channel, proc_client, proto, stack_dump, supervised_proc
from livekit.agents.ipc.job_proc_executor import ProcJobExecutor
from livekit.agents.utils.aio import duplex_unix

FIXTURE = Path(__file__).parent / "fixtures" / "stack_dump_child.py"


def _round_trip(message):
    return channel._read_message(channel._write_message(message), proto.IPC_MESSAGES)


def test_initialize_protocol_round_trips_complete_stack_dump_identity(tmp_path):
    request = proto.InitializeRequest(
        stack_dump_init=stack_dump.StackDumpInit(
            enabled=True,
            directory_path=str(tmp_path),
            directory_device=101,
            directory_inode=202,
            directory_owner_uid=303,
            directory_mode=0o700,
            episode_token="a" * 32,
        )
    )
    decoded_request = _round_trip(request)
    assert decoded_request.stack_dump_init == request.stack_dump_init

    response = proto.InitializeResponse(
        stack_dump_ready=stack_dump.StackDumpReady(
            ready=True,
            child_pid=404,
            episode_token="a" * 32,
            relative_basename=f"stack-404-{'a' * 32}.dump",
            directory_device=101,
            directory_inode=202,
            file_device=505,
            file_inode=606,
            owner_uid=303,
            mode=0o600,
            link_count=1,
        )
    )
    decoded_response = _round_trip(response)
    assert decoded_response.stack_dump_ready == response.stack_dump_ready


def test_initialize_protocol_defaults_are_disabled():
    request = _round_trip(proto.InitializeRequest())
    response = _round_trip(proto.InitializeResponse())
    assert request.stack_dump_init is None
    assert response.stack_dump_ready == stack_dump.StackDumpReady.disabled()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes "])
def test_stack_dump_flag_accepts_only_explicit_true_values(monkeypatch, value):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, value)
    assert stack_dump.stack_dump_enabled()


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "on", "enabled"])
def test_stack_dump_flag_rejects_every_other_value(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(stack_dump.ENABLED_ENV, raising=False)
    else:
        monkeypatch.setenv(stack_dump.ENABLED_ENV, value)
    assert not stack_dump.stack_dump_enabled()


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_child_file_is_exclusive_owned_and_collectable(tmp_path, monkeypatch):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    ready = stack_dump.install_stack_dump_signal_handler(init)
    try:
        assert ready.ready
        assert ready.child_pid == os.getpid()
        assert ready.episode_token == init.episode_token
        assert ready.relative_basename == f"stack-{os.getpid()}-{init.episode_token}.dump"
        assert ready.directory_device == init.directory_device
        assert ready.directory_inode == init.directory_inode
        assert ready.owner_uid == os.geteuid()
        assert ready.mode == 0o600
        assert ready.link_count == 1

        validated = stack_dump.validate_stack_dump_ready(
            init, ready, expected_pid=os.getpid(), directory_fd=directory_fd
        )
        assert validated.ready

        os.kill(os.getpid(), signal.SIGUSR1)
    finally:
        stack_dump.close_stack_dump_signal_handler(unlink=False)

    collected = stack_dump.collect_stack_dump_artifact(init, ready, directory_fd)
    assert collected.failure_class is None
    assert collected.stack_text is not None
    assert "test_child_file_is_exclusive_owned_and_collectable" in collected.stack_text
    assert collected.bytes_read > 0
    assert not collected.truncated
    assert not Path(init.directory_path, ready.relative_basename).exists()
    stack_dump.close_stack_dump_directory(init, directory_fd)
    assert not Path(init.directory_path).exists()


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_preexisting_file_is_never_unlinked(tmp_path, monkeypatch):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    sentinel = Path(init.directory_path, f"stack-{os.getpid()}-{init.episode_token}.dump")
    sentinel.write_text("keep me")
    sentinel.chmod(0o600)

    ready = stack_dump.install_stack_dump_signal_handler(init)
    assert not ready.ready
    assert ready.failure_class == "file_open_failed"
    assert sentinel.read_text() == "keep me"

    sentinel.unlink()
    stack_dump.close_stack_dump_directory(init, directory_fd)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
@pytest.mark.parametrize(
    ("target", "failure_class"),
    [("fdopen", "file_wrap_failed"), ("register", "handler_registration_failed")],
)
def test_partial_producer_failure_unlinks_only_created_file(
    tmp_path, monkeypatch, target, failure_class
):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    if target == "fdopen":
        monkeypatch.setattr(
            stack_dump.os,
            "fdopen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError()),
        )
    else:
        monkeypatch.setattr(
            stack_dump.faulthandler,
            "register",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()),
        )

    ready = stack_dump.install_stack_dump_signal_handler(init)

    assert not ready.ready
    assert ready.failure_class == failure_class
    assert list(Path(init.directory_path).iterdir()) == []
    stack_dump.close_stack_dump_directory(init, directory_fd)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_producer_does_not_unlink_an_inode_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    ready = stack_dump.install_stack_dump_signal_handler(init)
    assert ready.ready
    artifact = Path(init.directory_path, ready.relative_basename)
    original = artifact.with_suffix(".original")
    artifact.rename(original)
    artifact.write_text("replacement")
    artifact.chmod(0o600)

    stack_dump.close_stack_dump_signal_handler(unlink=True)
    assert artifact.read_text() == "replacement"

    artifact.unlink()
    original.unlink()
    stack_dump.close_stack_dump_directory(init, directory_fd)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_collector_rejects_and_preserves_an_inode_replacement(tmp_path, monkeypatch):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    ready = stack_dump.install_stack_dump_signal_handler(init)
    assert ready.ready
    stack_dump.close_stack_dump_signal_handler(unlink=False)
    artifact = Path(init.directory_path, ready.relative_basename)
    original = artifact.with_suffix(".original")
    artifact.rename(original)
    artifact.write_text("replacement")
    artifact.chmod(0o600)

    collected = stack_dump.collect_stack_dump_artifact(init, ready, directory_fd)
    assert collected.failure_class == "artifact_identity_mismatch"
    assert artifact.read_text() == "replacement"

    artifact.unlink()
    original.unlink()
    stack_dump.close_stack_dump_directory(init, directory_fd)


def test_ready_validation_rejects_wrong_pid_without_removing_artifact(tmp_path, monkeypatch):
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("requires SIGUSR1")
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    ready = stack_dump.install_stack_dump_signal_handler(init)
    assert ready.ready
    try:
        rejected = stack_dump.validate_stack_dump_ready(
            init, ready, expected_pid=os.getpid() + 1, directory_fd=directory_fd
        )
        assert not rejected.ready
        assert rejected.failure_class == "identity_mismatch"
        assert Path(init.directory_path, ready.relative_basename).exists()
    finally:
        stack_dump.close_stack_dump_signal_handler(unlink=True)
        stack_dump.close_stack_dump_directory(init, directory_fd)


class _FakeTimerHandle:
    def __init__(self, callback):
        self.callback = callback
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


class _FakeLoop:
    def __init__(self):
        self.calls = []

    def call_later(self, delay, callback):
        handle = _FakeTimerHandle(callback)
        self.calls.append((delay, handle))
        return handle


def test_trigger_allows_only_one_request_per_child():
    loop = _FakeLoop()
    requests = []
    trigger = stack_dump.PongStallDumpTrigger(
        loop, ping_timeout=60.0, request_dump=lambda: requests.append("sent")
    )

    trigger.arm()
    first = loop.calls[-1][1]
    trigger.pong()
    assert first.cancelled
    assert len(loop.calls) == 2

    loop.calls[-1][1].callback()
    assert requests == ["sent"]
    trigger.pong()
    trigger.arm()
    assert len(loop.calls) == 2


def test_trigger_close_cancels_pending_request():
    loop = _FakeLoop()
    requests = []
    trigger = stack_dump.PongStallDumpTrigger(
        loop, ping_timeout=60.0, request_dump=lambda: requests.append("sent")
    )
    trigger.arm()
    handle = loop.calls[-1][1]
    trigger.close()
    handle.callback()
    assert handle.cancelled
    assert requests == []


def test_trigger_is_inert_when_lead_time_is_not_available():
    loop = _FakeLoop()
    trigger = stack_dump.PongStallDumpTrigger(loop, ping_timeout=0.5, request_dump=lambda: None)
    trigger.arm()
    assert loop.calls == []


async def _unused_main_task(_receiver):
    return None


def _initialize_client(init, initialize_fnc):
    parent_socket, child_socket = socket.socketpair()
    parent = duplex_unix._Duplex.open(parent_socket)
    client = proc_client._ProcClient(
        child_socket,
        None,
        initialize_fnc,
        _unused_main_task,
    )
    failure = []

    def run():
        try:
            client.initialize()
        except Exception as error:
            failure.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    channel.send_message(parent, proto.InitializeRequest(stack_dump_init=init))
    response = channel.recv_message(parent, proto.IPC_MESSAGES)
    thread.join(timeout=1)
    assert not thread.is_alive()
    parent.close()
    return response, failure


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_proc_client_reports_ready_only_after_handler_install(tmp_path, monkeypatch):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    callback_saw_handler = []

    def initialize(_request, _client):
        callback_saw_handler.append(stack_dump.stack_dump_signal_handler_ready())

    response, failure = _initialize_client(init, initialize)
    try:
        assert failure == []
        assert response.error == ""
        assert response.stack_dump_ready.ready
        assert callback_saw_handler == [True]
    finally:
        stack_dump.close_stack_dump_signal_handler(unlink=True)
        stack_dump.close_stack_dump_directory(init, directory_fd)


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_proc_client_init_failure_removes_only_its_owned_artifact(tmp_path, monkeypatch):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))

    def initialize(_request, _client):
        raise RuntimeError("user init failed")

    response, failure = _initialize_client(init, initialize)
    assert response.error == "user init failed"
    assert response.stack_dump_ready == stack_dump.StackDumpReady.disabled()
    assert len(failure) == 1
    assert list(Path(init.directory_path).iterdir()) == []
    stack_dump.close_stack_dump_directory(init, directory_fd)


class _AliveProcess:
    def is_alive(self):
        return True


class _TestSupervisedProc(supervised_proc.SupervisedProc):
    def _create_process(self, _cch, _log_cch):
        raise NotImplementedError

    async def _main_task(self, _ipc_ch):
        return None


@pytest.mark.asyncio
async def test_supervisor_requests_at_most_one_dump_per_child(monkeypatch):
    loop = __import__("asyncio").get_running_loop()
    process = _TestSupervisedProc(
        initialize_timeout=1,
        close_timeout=1,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=2.5,
        ping_timeout=60,
        high_ping_threshold=0.5,
        mp_ctx=None,
        loop=loop,
    )
    process._proc = _AliveProcess()
    process._pid = 1234
    process._last_pong_monotonic = time.monotonic() - 59.5
    process._stack_dump_ready = stack_dump.StackDumpReady(
        ready=True,
        child_pid=1234,
        episode_token="b" * 32,
        relative_basename=f"stack-1234-{'b' * 32}.dump",
    )
    signals = []
    monkeypatch.setattr(supervised_proc.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    process._request_stack_dump()
    process._request_stack_dump()

    assert signals == [(1234, signal.SIGUSR1)]
    assert process._stack_dump_request_record is not None
    assert process._stack_dump_request_record.child_pid == 1234
    assert process._stack_dump_request_record.episode_token == "b" * 32
    assert process._stack_dump_request_record.sent
    assert process._stack_dump_request_record.failure_class is None


@pytest.mark.asyncio
async def test_supervisor_never_signals_without_positive_readiness(monkeypatch):
    loop = __import__("asyncio").get_running_loop()
    process = _TestSupervisedProc(
        initialize_timeout=1,
        close_timeout=1,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=2.5,
        ping_timeout=60,
        high_ping_threshold=0.5,
        mp_ctx=None,
        loop=loop,
    )
    process._proc = _AliveProcess()
    process._pid = 1234
    process._last_pong_monotonic = time.monotonic()
    signals = []
    monkeypatch.setattr(supervised_proc.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    process._request_stack_dump()

    assert signals == []
    assert process._stack_dump_request_record is None


def _new_test_supervisor(loop):
    return _TestSupervisedProc(
        initialize_timeout=1,
        close_timeout=1,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=2.5,
        ping_timeout=60,
        high_ping_threshold=0.5,
        mp_ctx=None,
        loop=loop,
    )


@pytest.mark.asyncio
async def test_initialize_sends_sink_identity_and_requires_validated_readiness(monkeypatch):
    loop = __import__("asyncio").get_running_loop()
    process = _new_test_supervisor(loop)
    process._proc = SimpleNamespace(name="job_proc")
    process._pid = 1234
    process._pch = object()
    init = stack_dump.StackDumpInit(
        enabled=True,
        directory_path="/dev/shm/private",
        directory_device=10,
        directory_inode=11,
        directory_owner_uid=os.geteuid(),
        directory_mode=0o700,
        episode_token="c" * 32,
    )
    ready = stack_dump.StackDumpReady(
        ready=True,
        child_pid=1234,
        episode_token="c" * 32,
        relative_basename=f"stack-1234-{'c' * 32}.dump",
        directory_device=10,
        directory_inode=11,
        file_device=12,
        file_inode=13,
        owner_uid=os.geteuid(),
        mode=0o600,
        link_count=1,
    )
    sent = []
    monkeypatch.setattr(supervised_proc, "stack_dump_enabled", lambda: True)
    monkeypatch.setattr(supervised_proc, "create_stack_dump_init", lambda: (init, 99))
    monkeypatch.setattr(
        supervised_proc,
        "validate_stack_dump_ready",
        lambda actual_init, actual_ready, **kwargs: ready,
    )

    async def send(_channel, message):
        sent.append(message)

    async def receive(_channel, _messages):
        return proto.InitializeResponse(stack_dump_ready=ready)

    monkeypatch.setattr(supervised_proc.channel, "asend_message", send)
    monkeypatch.setattr(supervised_proc.channel, "arecv_message", receive)

    await process.initialize()

    assert sent[0].stack_dump_init == init
    assert process._stack_dump_init == init
    assert process._stack_dump_directory_fd == 99
    assert process._stack_dump_ready == ready


@pytest.mark.asyncio
async def test_initialize_disabled_creates_no_sink(monkeypatch):
    loop = __import__("asyncio").get_running_loop()
    process = _new_test_supervisor(loop)
    process._proc = SimpleNamespace(name="job_proc")
    process._pid = 1234
    process._pch = object()
    sent = []
    monkeypatch.setattr(supervised_proc, "stack_dump_enabled", lambda: False)

    def unexpected_create():
        raise AssertionError("disabled mode created a sink")

    monkeypatch.setattr(supervised_proc, "create_stack_dump_init", unexpected_create)

    async def send(_channel, message):
        sent.append(message)

    async def receive(_channel, _messages):
        return proto.InitializeResponse()

    monkeypatch.setattr(supervised_proc.channel, "asend_message", send)
    monkeypatch.setattr(supervised_proc.channel, "arecv_message", receive)

    await process.initialize()

    assert sent[0].stack_dump_init is None
    assert process._stack_dump_init is None
    assert process._stack_dump_directory_fd is None
    assert process._stack_dump_ready == stack_dump.StackDumpReady.disabled()


@pytest.mark.asyncio
async def test_supervisor_arms_only_after_readiness_and_never_rearms_after_request(monkeypatch):
    real_loop = __import__("asyncio").get_running_loop()
    process = _new_test_supervisor(real_loop)
    fake_loop = _FakeLoop()
    process._loop = fake_loop
    process._proc = _AliveProcess()
    process._pid = 1234
    process._stack_dump_ready = stack_dump.StackDumpReady(
        ready=True,
        child_pid=1234,
        episode_token="d" * 32,
        relative_basename=f"stack-1234-{'d' * 32}.dump",
    )
    signals = []
    monkeypatch.setattr(supervised_proc.os, "kill", lambda pid, sig: signals.append((pid, sig)))

    process._arm_stack_dump_trigger()
    assert fake_loop.calls[0][0] == 59.5
    fake_loop.calls[0][1].callback()
    assert signals == [(1234, signal.SIGUSR1)]

    process._on_stack_dump_pong()
    process._arm_stack_dump_trigger()
    assert len(fake_loop.calls) == 1


@pytest.mark.asyncio
async def test_post_exit_collection_emits_bounded_attributed_record(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    ready = stack_dump.install_stack_dump_signal_handler(init)
    assert ready.ready
    os.kill(os.getpid(), signal.SIGUSR1)
    stack_dump.close_stack_dump_signal_handler(unlink=False)

    loop = __import__("asyncio").get_running_loop()
    process = _new_test_supervisor(loop)
    process._pid = os.getpid()
    process._stack_dump_init = init
    process._stack_dump_directory_fd = directory_fd
    process._stack_dump_ready = ready
    process._stack_dump_request_record = stack_dump.StackDumpRequestRecord(
        child_pid=os.getpid(),
        episode_token=init.episode_token,
        requested_at_unix_ms=123456,
        last_pong_age_ms=59500,
        sent=True,
    )

    with caplog.at_level("WARNING", logger="livekit.agents"):
        process._collect_and_emit_stack_dump()

    events = {getattr(record, "diagnostic_event", None): record for record in caplog.records}
    assert "job_stack_dump_requested" in events
    collected = events["job_stack_dump_collected"]
    assert collected.child_pid == os.getpid()
    assert collected.episode_token == init.episode_token
    assert collected.dump_requested_at_unix_ms == 123456
    assert collected.bytes_read <= 65_536
    assert "test_post_exit_collection_emits_bounded_attributed_record" in collected.stack_text
    assert process._stack_dump_directory_fd is None
    assert not Path(init.directory_path).exists()


def _read_fixture_event(process, timeout=2.0):
    assert process.stdout is not None
    readable, _, _ = select.select([process.stdout], [], [], timeout)
    if not readable:
        raise TimeoutError("fixture emitted no event")
    line = process.stdout.readline()
    if not line:
        raise EOFError(f"fixture exited with {process.poll()}")
    return json.loads(line)


def _send_fixture_command(process, command):
    assert process.stdin is not None
    process.stdin.write(command + "\n")
    process.stdin.flush()


def _start_fixture(tmp_path, label="alpha", *, saturate_stderr=False, heartbeat=False):
    init, directory_fd = stack_dump.create_stack_dump_init(str(tmp_path))
    command = [
        sys.executable,
        str(FIXTURE),
        "--init",
        json.dumps(asdict(init)),
        "--label",
        label,
    ]
    if saturate_stderr:
        command.append("--saturate-stderr")
    if heartbeat:
        command.append("--heartbeat")
    environment = os.environ.copy()
    environment[stack_dump.ENABLED_ENV] = "true"
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if saturate_stderr else subprocess.DEVNULL,
        text=True,
        bufsize=1,
        env=environment,
    )
    event = _read_fixture_event(process)
    assert event["event"] == "ready"
    ready = stack_dump.StackDumpReady(**event["identity"])
    ready = stack_dump.validate_stack_dump_ready(
        init, ready, expected_pid=process.pid, directory_fd=directory_fd
    )
    assert ready.ready
    return process, init, directory_fd, ready, event


def _wait_for_complete_dump(init, ready, expected_frame, timeout=0.25):
    deadline = time.monotonic() + timeout
    artifact = Path(init.directory_path, ready.relative_basename)
    while time.monotonic() < deadline:
        text = artifact.read_text(errors="replace")
        if "Current thread" in text and expected_frame in text:
            return
        time.sleep(0.001)
    raise TimeoutError("real child stack dump did not complete")


def _stop_and_collect(process, init, directory_fd, ready):
    _send_fixture_command(process, "stop")
    process.wait(timeout=2)
    assert process.returncode == 0
    collected = stack_dump.collect_stack_dump_artifact(init, ready, directory_fd)
    stack_dump.close_stack_dump_directory(init, directory_fd)
    return collected


def test_real_child_recovers_with_saturated_stderr(tmp_path):
    process, init, directory_fd, ready, event = _start_fixture(tmp_path, saturate_stderr=True)
    started = time.monotonic_ns()
    os.kill(process.pid, signal.SIGUSR1)
    _wait_for_complete_dump(init, ready, "alpha_unique_anchor")
    _send_fixture_command(process, "recover")
    pong = _read_fixture_event(process, timeout=0.25)
    latency_ms = (time.monotonic_ns() - started) / 1_000_000
    assert pong["event"] == "pong"
    assert latency_ms < 250
    collected = _stop_and_collect(process, init, directory_fd, ready)
    assert event["stderr_filled_bytes"] > 0
    assert collected.failure_class is None
    assert "alpha_unique_anchor" in collected.stack_text
    assert collected.stack_text.count("alpha_unique_anchor") >= 3
    assert process.stderr is not None
    stderr = process.stderr.read()
    assert "Current thread" not in stderr


def test_inherited_stderr_control_blocks_recovery_when_pipe_is_full():
    environment = os.environ.copy()
    environment[stack_dump.ENABLED_ENV] = "true"
    process = subprocess.Popen(
        [
            sys.executable,
            str(FIXTURE),
            "--init",
            "{}",
            "--label",
            "stderr-control",
            "--saturate-stderr",
            "--stderr-sink",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=environment,
    )
    event = _read_fixture_event(process)
    assert event["event"] == "ready"
    assert event["stderr_filled_bytes"] > 0
    os.kill(process.pid, signal.SIGUSR1)
    _send_fixture_command(process, "recover")
    with pytest.raises(TimeoutError):
        _read_fixture_event(process, timeout=0.1)
    os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=2)


def test_two_real_children_are_attributed_while_third_child_keeps_progress(tmp_path):
    alpha = _start_fixture(tmp_path, "alpha")
    beta = _start_fixture(tmp_path, "beta")
    progress = _start_fixture(tmp_path, "progress", heartbeat=True)
    alpha_process, alpha_init, alpha_fd, alpha_ready, _ = alpha
    beta_process, beta_init, beta_fd, beta_ready, _ = beta
    progress_process, progress_init, progress_fd, progress_ready, _ = progress

    os.kill(alpha_process.pid, signal.SIGUSR1)
    os.kill(beta_process.pid, signal.SIGUSR1)
    _wait_for_complete_dump(alpha_init, alpha_ready, "alpha_unique_anchor")
    _wait_for_complete_dump(beta_init, beta_ready, "beta_unique_anchor")
    _send_fixture_command(alpha_process, "recover")
    _send_fixture_command(beta_process, "recover")
    assert _read_fixture_event(alpha_process, 0.25)["event"] == "pong"
    assert _read_fixture_event(beta_process, 0.25)["event"] == "pong"

    alpha_collection = _stop_and_collect(alpha_process, alpha_init, alpha_fd, alpha_ready)
    emitted_at = time.monotonic_ns()
    progress_after_emission = False
    for _ in range(20):
        progress_event = _read_fixture_event(progress_process, 0.25)
        if progress_event["event"] == "pong" and progress_event["monotonic_ns"] > emitted_at:
            progress_after_emission = True
            break
    beta_collection = _stop_and_collect(beta_process, beta_init, beta_fd, beta_ready)
    progress_collection = _stop_and_collect(
        progress_process, progress_init, progress_fd, progress_ready
    )

    assert progress_after_emission
    assert "alpha_unique_anchor" in alpha_collection.stack_text
    assert "beta_unique_anchor" not in alpha_collection.stack_text
    assert "beta_unique_anchor" in beta_collection.stack_text
    assert "alpha_unique_anchor" not in beta_collection.stack_text
    assert progress_collection.bytes_read == 0


def test_real_held_interpreter_produces_a_dump(tmp_path):
    process, init, directory_fd, ready, _ = _start_fixture(tmp_path)
    _send_fixture_command(process, "hold")
    assert _read_fixture_event(process)["event"] == "holding"
    os.kill(process.pid, signal.SIGUSR1)
    _wait_for_complete_dump(init, ready, "hold_interpreter", timeout=10)
    os.kill(process.pid, signal.SIGKILL)
    process.wait(timeout=2)
    collected = stack_dump.collect_stack_dump_artifact(init, ready, directory_fd)
    stack_dump.close_stack_dump_directory(init, directory_fd)
    assert collected.failure_class is None
    assert "hold_interpreter" in collected.stack_text


def test_real_capture_latency_budget(tmp_path):
    count = int(os.getenv("STACK_DUMP_STRESS_COUNT", "5"))
    latencies = []
    for index in range(count):
        process, init, directory_fd, ready, _ = _start_fixture(tmp_path, f"stress-{index}")
        started = time.monotonic_ns()
        os.kill(process.pid, signal.SIGUSR1)
        _wait_for_complete_dump(init, ready, "shared_anchor")
        _send_fixture_command(process, "recover")
        assert _read_fixture_event(process, 0.25)["event"] == "pong"
        latencies.append((time.monotonic_ns() - started) / 1_000_000)
        _stop_and_collect(process, init, directory_fd, ready)
    assert len(latencies) == count
    assert max(latencies) < 250


def test_protected_kill_method_remains_synchronous_and_unchanged():
    source = inspect.getsource(supervised_proc.SupervisedProc._send_kill_signal)
    assert not inspect.iscoroutinefunction(supervised_proc.SupervisedProc._send_kill_signal)
    assert "SIGUSR1" not in source
    assert "sleep" not in source


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="requires Linux /dev/shm")
async def test_process_job_disabled_mode_starts_without_diagnostic_resources(monkeypatch):
    fixtures = str(FIXTURE.parent)
    monkeypatch.syspath_prepend(fixtures)
    from stack_dump_child import initialize_job_process, unused_job_entrypoint

    monkeypatch.delenv(stack_dump.ENABLED_ENV, raising=False)
    loop = __import__("asyncio").get_running_loop()
    executor = ProcJobExecutor(
        initialize_process_fnc=initialize_job_process,
        job_entrypoint_fnc=unused_job_entrypoint,
        inference_executor=None,
        initialize_timeout=5,
        close_timeout=1,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=0.1,
        ping_timeout=2,
        high_ping_threshold=1,
        mp_ctx=__import__("multiprocessing").get_context("spawn"),
        loop=loop,
    )
    await executor.start()
    try:
        await executor.initialize()
        assert executor._stack_dump_init is None
        assert executor._stack_dump_directory_fd is None
        assert executor._stack_dump_trigger is None
        assert executor._stack_dump_request_record is None
    finally:
        await executor.kill()


@pytest.mark.asyncio
@pytest.mark.skipif(not Path("/dev/shm").is_dir(), reason="requires Linux /dev/shm")
async def test_process_job_initialization_round_trips_real_handler(monkeypatch):
    fixtures = str(FIXTURE.parent)
    monkeypatch.syspath_prepend(fixtures)
    from stack_dump_child import initialize_job_process, unused_job_entrypoint

    monkeypatch.setenv(stack_dump.ENABLED_ENV, "true")
    loop = __import__("asyncio").get_running_loop()
    executor = ProcJobExecutor(
        initialize_process_fnc=initialize_job_process,
        job_entrypoint_fnc=unused_job_entrypoint,
        inference_executor=None,
        initialize_timeout=5,
        close_timeout=1,
        memory_warn_mb=0,
        memory_limit_mb=0,
        ping_interval=0.1,
        ping_timeout=2,
        high_ping_threshold=1,
        mp_ctx=__import__("multiprocessing").get_context("spawn"),
        loop=loop,
    )
    await executor.start()
    try:
        await executor.initialize()
        assert executor._stack_dump_ready.ready
        assert executor._stack_dump_ready.child_pid == executor.pid
    finally:
        await executor.kill()
    assert executor._stack_dump_request_record is None
