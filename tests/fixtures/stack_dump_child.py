from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import asdict

from livekit.agents.ipc import stack_dump

_output_lock = threading.Lock()


def _emit(**fields) -> None:
    with _output_lock:
        print(json.dumps(fields, sort_keys=True), flush=True)


def shared_anchor(stop: threading.Event) -> None:
    stop.wait()


def alpha_unique_anchor(stop: threading.Event) -> None:
    shared_anchor(stop)


def beta_unique_anchor(stop: threading.Event) -> None:
    shared_anchor(stop)


def hold_interpreter() -> None:
    math.factorial(700_000)


def initialize_job_process(_process) -> None:
    return None


async def unused_job_entrypoint(_context) -> None:
    return None


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--saturate-stderr", action="store_true")
    parser.add_argument("--heartbeat", action="store_true")
    parser.add_argument("--stderr-sink", action="store_true")
    args = parser.parse_args()

    stderr_filled_bytes = 0
    if args.saturate_stderr:
        os.set_blocking(2, False)
        block = b"B" * 4096
        while True:
            try:
                stderr_filled_bytes += os.write(2, block)
            except BlockingIOError:
                break
        os.set_blocking(2, True)

    if args.stderr_sink:
        faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True, chain=False)
        ready = None
    else:
        init = stack_dump.StackDumpInit(**json.loads(args.init))
        ready = stack_dump.install_stack_dump_signal_handler(init)
        if not ready.ready:
            _emit(event="not_ready", identity=asdict(ready))
            return 3

    stop = threading.Event()
    target = (
        alpha_unique_anchor
        if args.label == "alpha"
        else beta_unique_anchor
        if args.label == "beta"
        else shared_anchor
    )
    anchors = [
        threading.Thread(target=target, args=(stop,), name=f"{args.label}-{index}")
        for index in range(3)
    ]
    for anchor in anchors:
        anchor.start()

    heartbeat = None
    if args.heartbeat:

        def _heartbeat() -> None:
            while not stop.wait(0.02):
                _emit(event="pong", monotonic_ns=time.monotonic_ns(), pid=os.getpid())

        heartbeat = threading.Thread(target=_heartbeat, name="heartbeat")
        heartbeat.start()

    _emit(
        event="ready",
        identity=asdict(ready) if ready is not None else None,
        stderr_filled_bytes=stderr_filled_bytes,
    )
    for line in sys.stdin:
        command = line.strip()
        if command == "recover":
            _emit(event="pong", monotonic_ns=time.monotonic_ns(), pid=os.getpid())
        elif command == "hold":
            _emit(event="holding", pid=os.getpid())
            hold_interpreter()
        elif command == "stop":
            break

    stop.set()
    for anchor in anchors:
        anchor.join(timeout=1)
    if heartbeat is not None:
        heartbeat.join(timeout=1)
    if args.stderr_sink:
        faulthandler.unregister(signal.SIGUSR1)
    else:
        stack_dump.close_stack_dump_signal_handler(unlink=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
