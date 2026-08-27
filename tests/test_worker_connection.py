import asyncio

import pytest

from livekit.agents.worker import WORKER_WS_HEARTBEAT_INTERVAL, Worker, WorkerOptions
from livekit.protocol import agent


async def _entrypoint(_ctx):
    return None


class _FakeWebSocket:
    def __init__(self):
        register = agent.ServerMessage()
        register.register.worker_id = "AW_test"
        self._register_response = register.SerializeToString()
        self.closed = False

    async def send_bytes(self, _data):
        return None

    async def receive_bytes(self):
        return self._register_response

    async def close(self):
        self.closed = True


class _FakeSession:
    def __init__(self, ws):
        self.ws = ws
        self.kwargs = None

    async def ws_connect(self, _url, **kwargs):
        self.kwargs = kwargs
        return self.ws


@pytest.mark.asyncio
async def test_worker_connection_uses_heartbeat_and_emits_disconnect(monkeypatch):
    worker = Worker(
        WorkerOptions(
            entrypoint_fnc=_entrypoint,
            ws_url="ws://livekit.test",
            api_key="key",
            api_secret="secret",
            max_retry=0,
        ),
        devmode=False,
        loop=asyncio.get_running_loop(),
    )
    ws = _FakeWebSocket()
    session = _FakeSession(ws)
    worker._http_session = session
    worker._closed = False

    async def fail_after_registration(_ws):
        raise ConnectionError("black-holed worker socket")

    monkeypatch.setattr(worker, "_run_ws", fail_after_registration)
    disconnected = asyncio.Event()
    worker.on("worker_disconnected", disconnected.set)

    with pytest.raises(RuntimeError, match="failed to connect to livekit"):
        await worker._connection_task()

    assert session.kwargs["autoping"] is True
    assert session.kwargs["heartbeat"] == WORKER_WS_HEARTBEAT_INTERVAL
    assert ws.closed is True
    assert disconnected.is_set()
