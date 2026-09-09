from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing.context import BaseContext
from typing import Any

import psutil

from ..log import logger
from ..utils import aio, log_exceptions, time_ms
from ..utils.aio import duplex_unix
from . import channel, proto
from .log_queue import LogQueueListener
from .stack_dump import (
    PongStallDumpTrigger,
    StackDumpInit,
    StackDumpReady,
    StackDumpRequestRecord,
    close_stack_dump_fd,
    collect_stack_dump_fd,
    receive_stack_dump_fd,
)


@dataclass
class _ProcOpts:
    initialize_timeout: float
    close_timeout: float
    memory_warn_mb: float
    memory_limit_mb: float
    ping_interval: float
    ping_timeout: float
    high_ping_threshold: float


class SupervisedProc(ABC):
    def __init__(
        self,
        *,
        initialize_timeout: float,
        close_timeout: float,
        memory_warn_mb: float,
        memory_limit_mb: float,
        ping_interval: float,
        ping_timeout: float,
        high_ping_threshold: float,
        mp_ctx: BaseContext,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._loop = loop
        self._mp_ctx = mp_ctx
        self._opts = _ProcOpts(
            initialize_timeout=initialize_timeout,
            close_timeout=close_timeout,
            memory_warn_mb=memory_warn_mb,
            memory_limit_mb=memory_limit_mb,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            high_ping_threshold=high_ping_threshold,
        )

        self._exitcode: int | None = None
        self._pid: int | None = None

        self._supervise_atask: asyncio.Task[None] | None = None
        self._closing = False
        self._kill_sent = False
        self._initialize_fut = asyncio.Future[None]()
        self._lock = asyncio.Lock()
        self._stack_dump_ready = StackDumpReady.disabled()
        self._stack_dump_init: StackDumpInit | None = None
        self._stack_dump_fd: int | None = None
        self._stack_dump_pch: socket.socket | None = None
        self._stack_dump_cch: socket.socket | None = None
        self._stack_dump_collected = False
        self._stack_dump_trigger: PongStallDumpTrigger | None = None
        self._stack_dump_request_record: StackDumpRequestRecord | None = None
        self._stack_dump_setup_failure: str | None = None
        self._last_pong_monotonic: float | None = None
        self._pong_timeout_fired = False

    @abstractmethod
    def _create_process(self, cch: socket.socket, log_cch: socket.socket) -> mp.Process: ...

    @abstractmethod
    async def _main_task(self, ipc_ch: aio.ChanReceiver[channel.Message]) -> None: ...

    @property
    def exitcode(self) -> int | None:
        return self._exitcode

    @property
    def killed(self) -> bool:
        return self._kill_sent

    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def started(self) -> bool:
        return self._supervise_atask is not None

    async def start(self) -> None:
        """start the supervised process"""
        if self.started:
            raise RuntimeError("process already started")

        if self._closing:
            raise RuntimeError("process is closed")

        await asyncio.shield(self._start())

    async def _start(self) -> None:
        def _add_proc_ctx_log(record: logging.LogRecord) -> None:
            extra = self.logging_extra()
            for key, value in extra.items():
                setattr(record, key, value)

        async with self._lock:
            mp_pch, mp_cch = socket.socketpair()
            mp_log_pch, mp_log_cch = socket.socketpair()

            self._pch = await duplex_unix._AsyncDuplex.open(mp_pch)

            log_pch = duplex_unix._Duplex.open(mp_log_pch)
            log_listener = LogQueueListener(log_pch, _add_proc_ctx_log)
            log_listener.start()

            try:
                self._proc = self._create_process(mp_cch, mp_log_cch)
                await self._loop.run_in_executor(None, self._proc.start)
            except BaseException:
                self._close_stack_dump_channels()
                raise
            finally:
                if self._stack_dump_cch is not None:
                    self._stack_dump_cch.close()
                    self._stack_dump_cch = None
            mp_log_cch.close()
            mp_cch.close()

            self._pid = self._proc.pid
            self._join_fut = asyncio.Future[None]()

            def _sync_run():
                self._proc.join()
                log_listener.stop()
                try:
                    self._loop.call_soon_threadsafe(self._join_fut.set_result, None)
                except RuntimeError:
                    pass

            thread = threading.Thread(target=_sync_run, name="proc_join_thread")
            thread.start()
            self._supervise_atask = asyncio.create_task(self._supervise_task())

    async def join(self) -> None:
        """wait for the process to finish"""
        if not self.started:
            raise RuntimeError("process not started")

        if self._supervise_atask:
            await asyncio.shield(self._supervise_atask)

    async def initialize(self) -> None:
        """initialize the process, this is sending a InitializeRequest message and waiting for a
        InitializeResponse with a timeout"""
        # wait for the process to become ready
        try:
            await channel.asend_message(
                self._pch,
                proto.InitializeRequest(
                    asyncio_debug=self._loop.get_debug(),
                    ping_interval=self._opts.ping_interval,
                    ping_timeout=self._opts.ping_timeout,
                    high_ping_threshold=self._opts.high_ping_threshold,
                    stack_dump_init=self._stack_dump_init,
                ),
            )
            init_res = await asyncio.wait_for(
                channel.arecv_message(self._pch, proto.IPC_MESSAGES),
                timeout=self._opts.initialize_timeout,
            )
            assert isinstance(init_res, proto.InitializeResponse), (
                "first message must be InitializeResponse"
            )

            if init_res.error:
                logger.error(
                    f"process initialization failed: {init_res.error}",
                    extra=self.logging_extra(),
                )
                raise RuntimeError(f"process initialization failed: {init_res.error}")
            else:
                if (
                    self._stack_dump_init is not None
                    and self._stack_dump_pch is not None
                    and self._pid is not None
                ):
                    self._stack_dump_ready, self._stack_dump_fd = receive_stack_dump_fd(
                        self._stack_dump_pch,
                        self._stack_dump_init,
                        init_res.stack_dump_ready,
                        expected_pid=self._pid,
                    )
                self._initialize_fut.set_result(None)

        except asyncio.TimeoutError:
            self._initialize_fut.set_exception(
                asyncio.TimeoutError("process initialization timed out")
            )
            logger.error("initialization timed out, killing process", extra=self.logging_extra())
            self._send_kill_signal()
            raise
        except asyncio.CancelledError:
            if not self._initialize_fut.done():
                self._initialize_fut.set_exception(RuntimeError("process initialization cancelled"))
            raise
        except Exception as e:  # should be channel.ChannelClosed most of the time
            if not self._initialize_fut.done():
                self._initialize_fut.set_exception(e)
            raise
        finally:
            self._close_stack_dump_channels()

    def _close_stack_dump_channels(self) -> None:
        for name in ("_stack_dump_pch", "_stack_dump_cch"):
            sock = getattr(self, name)
            if sock is not None:
                sock.close()
                setattr(self, name, None)

    async def aclose(self) -> None:
        """attempt to gracefully close the supervised process"""
        if not self.started:
            return

        self._closing = True
        self._close_stack_dump_trigger()
        self._close_stack_dump_channels()
        if not self._initialize_fut.done():
            self._initialize_fut.set_exception(RuntimeError("process closed before initialization"))
        with contextlib.suppress(duplex_unix.DuplexClosed):
            await channel.asend_message(self._pch, proto.ShutdownRequest())

        try:
            if self._supervise_atask:
                await asyncio.wait_for(
                    asyncio.shield(self._supervise_atask),
                    timeout=self._opts.close_timeout,
                )
        except asyncio.TimeoutError:
            logger.error(
                "process did not exit in time, killing process",
                extra=self.logging_extra(),
            )
            self._send_kill_signal()

        async with self._lock:
            if self._supervise_atask:
                await asyncio.shield(self._supervise_atask)

    async def kill(self) -> None:
        """forcefully kill the supervised process"""
        if not self.started:
            raise RuntimeError("process not started")

        self._closing = True
        self._close_stack_dump_trigger()
        self._close_stack_dump_channels()
        if not self._initialize_fut.done():
            self._initialize_fut.set_exception(RuntimeError("process killed before initialization"))
        self._send_kill_signal()

        async with self._lock:
            if self._supervise_atask:
                await asyncio.shield(self._supervise_atask)

    def _request_stack_dump(self) -> None:
        if (
            not self._stack_dump_ready.ready
            or self._stack_dump_request_record is not None
            or self._kill_sent
            or self._pid is None
            or self._last_pong_monotonic is None
            or sys.platform == "win32"
            or not hasattr(signal, "SIGUSR1")
        ):
            return
        try:
            if not self._proc.is_alive():
                return
        except ValueError:
            return

        requested_at_unix_ms = time.time_ns() // 1_000_000
        last_pong_age_ms = round((time.monotonic() - self._last_pong_monotonic) * 1000)
        try:
            os.kill(self._pid, signal.SIGUSR1)
        except (OSError, ValueError) as error:
            self._stack_dump_request_record = StackDumpRequestRecord(
                child_pid=self._pid,
                episode_token=self._stack_dump_ready.episode_token,
                requested_at_unix_ms=requested_at_unix_ms,
                last_pong_age_ms=last_pong_age_ms,
                sent=False,
                failure_class=type(error).__name__,
            )
        else:
            self._stack_dump_request_record = StackDumpRequestRecord(
                child_pid=self._pid,
                episode_token=self._stack_dump_ready.episode_token,
                requested_at_unix_ms=requested_at_unix_ms,
                last_pong_age_ms=last_pong_age_ms,
                sent=True,
            )

    def _arm_stack_dump_trigger(self) -> None:
        if not self._stack_dump_ready.ready:
            return
        if self._stack_dump_trigger is None:
            self._last_pong_monotonic = time.monotonic()
            self._stack_dump_trigger = PongStallDumpTrigger(
                self._loop,
                ping_timeout=self._opts.ping_timeout,
                request_dump=self._request_stack_dump,
            )
        self._stack_dump_trigger.arm()

    def _on_stack_dump_pong(self) -> None:
        self._last_pong_monotonic = time.monotonic()
        if self._stack_dump_trigger is not None:
            self._stack_dump_trigger.pong()

    def _close_stack_dump_trigger(self) -> None:
        if self._stack_dump_trigger is not None:
            self._stack_dump_trigger.close()

    def _collect_and_emit_stack_dump(self) -> None:
        if self._stack_dump_collected:
            return
        self._stack_dump_collected = True
        init = self._stack_dump_init
        fd, self._stack_dump_fd = self._stack_dump_fd, None
        self._close_stack_dump_channels()
        collection = None
        if fd is not None:
            if self._stack_dump_request_record is not None and self._stack_dump_request_record.sent:
                collection = collect_stack_dump_fd(fd, self._stack_dump_ready)
            else:
                close_stack_dump_fd(fd)

        request = self._stack_dump_request_record
        if request is None:
            if self._pong_timeout_fired and (
                init is not None or self._stack_dump_setup_failure is not None
            ):
                with contextlib.suppress(Exception):
                    logger.warning(
                        "job stack dump unavailable",
                        extra={
                            "diagnostic_event": "job_stack_dump_unavailable",
                            "diagnostic_version": 1,
                            "failure_class": self._stack_dump_setup_failure
                            or self._stack_dump_ready.failure_class
                            or "handler_not_ready",
                            **self.logging_extra(),
                        },
                    )
            return

        request_extra = {
            "diagnostic_version": 1,
            "child_pid": request.child_pid,
            "episode_token": request.episode_token,
            "dump_requested_at_unix_ms": request.requested_at_unix_ms,
            "last_pong_age_ms": request.last_pong_age_ms,
            "ping_timeout_ms": round(self._opts.ping_timeout * 1000),
            "dump_lead_ms": round(0.5 * 1000),
            **self.logging_extra(),
        }
        with contextlib.suppress(Exception):
            logger.warning(
                "job stack dump requested" if request.sent else "job stack dump request failed",
                extra={
                    **request_extra,
                    "diagnostic_event": (
                        "job_stack_dump_requested"
                        if request.sent
                        else "job_stack_dump_request_failed"
                    ),
                    "failure_class": request.failure_class,
                },
            )

        if not request.sent:
            return
        if collection is None or collection.stack_text is None or collection.bytes_read == 0:
            with contextlib.suppress(Exception):
                logger.warning(
                    "job stack dump was not collected",
                    extra={
                        **request_extra,
                        "diagnostic_event": "job_stack_dump_collection_failed",
                        "failure_class": (
                            collection.failure_class
                            if collection is not None and collection.failure_class
                            else "request_without_dump"
                        ),
                    },
                )
            return

        with contextlib.suppress(Exception):
            logger.warning(
                "job stack dump collected",
                extra={
                    **request_extra,
                    "diagnostic_event": "job_stack_dump_collected",
                    "bytes_read": collection.bytes_read,
                    "truncated": collection.truncated,
                    "stack_text": collection.stack_text,
                },
            )

    def _send_kill_signal(self) -> None:
        """forcefully kill the process"""
        try:
            if not self._proc.is_alive():
                return
        except ValueError:
            return

        logger.info("killing process", extra=self.logging_extra())
        if sys.platform == "win32":
            self._proc.terminate()
        else:
            self._proc.kill()

        self._kill_sent = True

    @log_exceptions(logger=logger)
    async def _supervise_task(self) -> None:
        try:
            await self._initialize_fut
        except asyncio.TimeoutError:
            pass  # this happens when the initialization takes longer than self._initialize_timeout
        except Exception:
            pass  # initialization failed

        self._arm_stack_dump_trigger()

        # the process is killed if it doesn't respond to ping requests
        pong_timeout = aio.sleep(self._opts.ping_timeout)

        ipc_ch = aio.Chan[channel.Message]()

        main_task = asyncio.create_task(self._main_task(ipc_ch))
        read_ipc_task = asyncio.create_task(self._read_ipc_task(ipc_ch, pong_timeout))
        ping_task = asyncio.create_task(self._ping_pong_task(pong_timeout))
        read_ipc_task.add_done_callback(lambda _: ipc_ch.close())

        memory_monitor_task: asyncio.Task[None] | None = None
        if self._opts.memory_limit_mb > 0 or self._opts.memory_warn_mb > 0:
            memory_monitor_task = asyncio.create_task(self._memory_monitor_task())

        await self._join_fut
        self._close_stack_dump_trigger()
        self._exitcode = self._proc.exitcode
        self._collect_and_emit_stack_dump()
        self._proc.close()
        await aio.cancel_and_wait(ping_task, read_ipc_task, main_task)

        if memory_monitor_task is not None:
            await aio.cancel_and_wait(memory_monitor_task)

        with contextlib.suppress(duplex_unix.DuplexClosed):
            await self._pch.aclose()

        if self._exitcode != 0 and not self._kill_sent:
            logger.error(
                f"process exited with non-zero exit code {self.exitcode}",
                extra=self.logging_extra(),
            )

    @log_exceptions(logger=logger)
    async def _read_ipc_task(
        self, ipc_ch: aio.Chan[channel.Message], pong_timeout: aio.Sleep
    ) -> None:
        while True:
            try:
                msg = await channel.arecv_message(self._pch, proto.IPC_MESSAGES)
            except duplex_unix.DuplexClosed:
                self._close_stack_dump_trigger()
                break

            if isinstance(msg, proto.PongResponse):
                delay = time_ms() - msg.timestamp
                if delay > self._opts.high_ping_threshold * 1000:
                    logger.warning(
                        "process is unresponsive",
                        extra={"delay": delay, **self.logging_extra()},
                    )

                with contextlib.suppress(aio.SleepFinished):
                    pong_timeout.reset()
                self._on_stack_dump_pong()

            if isinstance(msg, proto.Exiting):
                logger.info(
                    "process exiting",
                    extra={"reason": msg.reason, **self.logging_extra()},
                )

            ipc_ch.send_nowait(msg)

    @log_exceptions(logger=logger)
    async def _ping_pong_task(self, pong_timeout: aio.Sleep) -> None:
        ping_interval = aio.interval(self._opts.ping_interval)

        async def _send_ping_co():
            while True:
                await ping_interval.tick()
                try:
                    await channel.asend_message(self._pch, proto.PingRequest(timestamp=time_ms()))
                except duplex_unix.DuplexClosed:
                    break

        async def _pong_timeout_co():
            await pong_timeout
            self._pong_timeout_fired = True
            logger.error("process is unresponsive, killing process", extra=self.logging_extra())
            self._send_kill_signal()

        tasks = [
            asyncio.create_task(_send_ping_co()),
            asyncio.create_task(_pong_timeout_co()),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            await aio.cancel_and_wait(*tasks)

    @log_exceptions(logger=logger)
    async def _memory_monitor_task(self) -> None:
        """Monitor memory usage and kill the process if it exceeds the limit."""
        while not self._closing and not self._kill_sent:
            try:
                if not self._pid:
                    await asyncio.sleep(5)
                    continue

                # get process memory info
                process = psutil.Process(self._pid)
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB

                if self._opts.memory_limit_mb > 0 and memory_mb > self._opts.memory_limit_mb:
                    logger.error(
                        "process exceeded memory limit, killing process",
                        extra={
                            "memory_usage_mb": memory_mb,
                            "memory_limit_mb": self._opts.memory_limit_mb,
                            **self.logging_extra(),
                        },
                    )
                    self._close_stack_dump_trigger()
                    self._send_kill_signal()
                elif self._opts.memory_warn_mb > 0 and memory_mb > self._opts.memory_warn_mb:
                    logger.warning(
                        "process memory usage is high",
                        extra={
                            "memory_usage_mb": memory_mb,
                            "memory_warn_mb": self._opts.memory_warn_mb,
                            "memory_limit_mb": self._opts.memory_limit_mb,
                            **self.logging_extra(),
                        },
                    )

            except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
                if self._closing or self._kill_sent:
                    return

                logger.warning(
                    "Failed to get memory info for process",
                    extra=self.logging_extra(),
                    exc_info=e,
                )
                # don't bother rechecking if we cannot get process info
                return
            except Exception:
                if self._closing or self._kill_sent:
                    return

                logger.exception(
                    "Error in memory monitoring task",
                    extra=self.logging_extra(),
                )

            await asyncio.sleep(5)  # check every 5 seconds

    def logging_extra(self):
        extra: dict[str, Any] = {
            "pid": self.pid,
        }

        return extra
