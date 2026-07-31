import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from livekit.agents import stt
from livekit.agents.voice.agent import ModelSettings
from livekit.agents.voice.agent_activity import AgentActivity
from livekit.agents.voice.agent_session import AgentSession
from livekit.agents.voice.audio_recognition import AudioRecognition
from livekit.agents.voice.speech_handle import SpeechHandle


class _OneEventStream:
    def __init__(self) -> None:
        self._yielded = False
        self.close_calls = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._yielded:
            await asyncio.Event().wait()

        self._yielded = True
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT)

    async def aclose(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
async def test_audio_recognition_closes_stt_node_when_cancelled_during_event_handling():
    recognition = AudioRecognition.__new__(AudioRecognition)
    stream = _OneEventStream()
    handling_event = asyncio.Event()

    async def stt_node(_audio_input, _model_settings: ModelSettings):
        return stream

    async def on_stt_event(_event: stt.SpeechEvent) -> None:
        handling_event.set()
        await asyncio.Event().wait()

    recognition._on_stt_event = on_stt_event
    task = asyncio.create_task(recognition._stt_task(stt_node, object(), None))
    await handling_event.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert stream.close_calls == 1


@pytest.mark.asyncio
async def test_agent_activity_force_interrupts_speech_and_closes_recognition():
    activity = AgentActivity.__new__(AgentActivity)
    current_speech = SpeechHandle.create(allow_interruptions=False)
    queued_speech = SpeechHandle.create(allow_interruptions=False)
    speech_task = asyncio.create_task(asyncio.Event().wait())
    main_task = asyncio.create_task(asyncio.Event().wait())
    recognition = SimpleNamespace(aclose=AsyncMock())
    agent = SimpleNamespace(_activity=activity)

    activity._lock = asyncio.Lock()
    activity._closed = False
    activity._draining = False
    activity._started = True
    activity._current_speech = current_speech
    activity._speech_q = [(SpeechHandle.SPEECH_PRIORITY_NORMAL, 0.0, queued_speech)]
    activity._speech_tasks = [speech_task]
    activity._q_updated = asyncio.Event()
    activity._rt_session = None
    activity._audio_recognition = recognition
    activity._main_atask = main_task
    activity._agent = agent

    await activity.aclose()
    await activity.aclose()

    assert current_speech.interrupted
    assert queued_speech.interrupted
    assert speech_task.cancelled()
    assert main_task.cancelled()
    recognition.aclose.assert_awaited_once()
    assert agent._activity is None


@pytest.mark.asyncio
async def test_agent_session_closes_activity_before_room_io_once():
    close_order = []

    async def close_activity() -> None:
        close_order.append("activity")

    async def close_room() -> None:
        close_order.append("room")

    session = AgentSession(stt=None, vad=None, llm=None, tts=None)
    session._started = True
    session._activity = SimpleNamespace(aclose=AsyncMock(side_effect=close_activity))
    session._next_activity = None
    session._room_io = SimpleNamespace(aclose=AsyncMock(side_effect=close_room))

    await session.aclose()
    await session.aclose()

    assert close_order == ["activity", "room"]
