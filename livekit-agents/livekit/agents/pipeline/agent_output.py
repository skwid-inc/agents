from __future__ import annotations

import asyncio
import inspect
import re
import time
import uuid
from typing import Any, AsyncIterable, Awaitable, Callable, Union

from app_config import AppConfig
from livekit import rtc

from .. import llm, tokenize, utils
from .. import transcription as agent_transcription
from .. import tts as text_to_speech
from .agent_playout import AgentPlayout, PlayoutHandle
from custom_logger import log
from custom_tokenize import StreamFlusher

SpeechSource = Union[AsyncIterable[str], str, Awaitable[str]]


class SynthesisHandle:
    def __init__(
        self,
        *,
        speech_id: str,
        tts_source: SpeechSource,
        transcript_source: SpeechSource,
        agent_playout: AgentPlayout,
        tts: text_to_speech.TTS,
        transcription_fwd: agent_transcription.TTSSegmentsForwarder,
    ) -> None:
        (
            self._tts_source,
            self._transcript_source,
            self._agent_playout,
            self._tts,
            self._tr_fwd,
        ) = (
            tts_source,
            transcript_source,
            agent_playout,
            tts,
            transcription_fwd,
        )
        self._buf_ch = utils.aio.Chan[rtc.AudioFrame]()
        self._play_handle: PlayoutHandle | None = None
        self._interrupt_fut = asyncio.Future[None]()
        self._speech_id = speech_id

    @property
    def speech_id(self) -> str:
        return self._speech_id

    @property
    def tts_forwarder(self) -> agent_transcription.TTSSegmentsForwarder:
        return self._tr_fwd

    @property
    def validated(self) -> bool:
        return self._play_handle is not None

    @property
    def interrupted(self) -> bool:
        return self._interrupt_fut.done()

    @property
    def play_handle(self) -> PlayoutHandle | None:
        return self._play_handle

    def play(self) -> PlayoutHandle:
        """Validate the speech for playout"""
        if self.interrupted:
            raise RuntimeError("synthesis was interrupted")

        self._play_handle = self._agent_playout.play(
            self._speech_id, self._buf_ch, transcription_fwd=self._tr_fwd
        )
        return self._play_handle

    def interrupt(self) -> None:
        """Interrupt the speech"""
        if self.interrupted:
            return

        log.pipeline(
            "agent interrupted",
            extra={"speech_id": self.speech_id},
        )
        log.pipeline(f"AGENT INTERRUPTED TEXT: {self.tts_forwarder.played_text}")
        if (
            self.tts_forwarder.played_text
            and self.tts_forwarder.played_text.strip() != ""
        ):
            AppConfig().call_metadata.update({"agent_has_been_interrupted": True})
        else:
            AppConfig().is_human_interrupted = True

        if self._play_handle is not None:
            self._play_handle.interrupt()

        self._interrupt_fut.set_result(None)


class AgentOutput:
    def __init__(
        self,
        *,
        room: rtc.Room,
        agent_playout: AgentPlayout,
        llm: llm.LLM,
        tts: text_to_speech.TTS,
    ) -> None:
        self._room, self._agent_playout, self._llm, self._tts = (
            room,
            agent_playout,
            llm,
            tts,
        )
        self._tasks = set[asyncio.Task[Any]]()

    @property
    def playout(self) -> AgentPlayout:
        return self._agent_playout

    async def aclose(self) -> None:
        for task in self._tasks:
            task.cancel()

        await asyncio.gather(*self._tasks, return_exceptions=True)

    def synthesize(
        self,
        *,
        speech_id: str,
        tts_source: SpeechSource,
        transcript_source: SpeechSource,
        transcription: bool,
        transcription_speed: float,
        sentence_tokenizer: tokenize.SentenceTokenizer,
        word_tokenizer: tokenize.WordTokenizer,
        hyphenate_word: Callable[[str], list[str]],
    ) -> SynthesisHandle:
        log.pipeline("synthesizing speech")
        log.pipeline(f"tts_source: {tts_source}")
        log.pipeline(f"transcript_source: {transcript_source}")
        log.pipeline(f"transcription: {transcription}")
        log.pipeline(f"transcription_speed: {transcription_speed}")
        log.pipeline(f"sentence_tokenizer: {sentence_tokenizer}")
        log.pipeline(f"word_tokenizer: {word_tokenizer}")
        log.pipeline(f"hyphenate_word: {hyphenate_word}")

        def _before_forward(
            fwd: agent_transcription.TTSSegmentsForwarder,
            rtc_transcription: rtc.Transcription,
        ):
            if not transcription:
                rtc_transcription.segments = []

            return rtc_transcription

        transcription_fwd = agent_transcription.TTSSegmentsForwarder(
            room=self._room,
            participant=self._room.local_participant,
            speed=transcription_speed,
            sentence_tokenizer=sentence_tokenizer,
            word_tokenizer=word_tokenizer,
            hyphenate_word=hyphenate_word,
            before_forward_cb=_before_forward,
        )

        handle = SynthesisHandle(
            tts_source=tts_source,
            transcript_source=transcript_source,
            agent_playout=self._agent_playout,
            tts=self._tts,
            transcription_fwd=transcription_fwd,
            speech_id=speech_id,
        )

        synthesize_task_id = f"Synthesize-{str(uuid.uuid4())}"
        log.verbose(f"Starting task: {synthesize_task_id}")
        pending_tasks = AppConfig().get_call_metadata().get("pending_livekit_tasks", {})
        pending_tasks[synthesize_task_id] = time.time()
        task = asyncio.create_task(self._synthesize_task(handle))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.remove)

        def _post_task_callback(_) -> None:
            log.verbose(f"Task completed: {synthesize_task_id}")
            pending_tasks.pop(synthesize_task_id, None)

        task.add_done_callback(_post_task_callback)
        return handle

    @utils.log_exceptions(logger=log)
    async def _synthesize_task(self, handle: SynthesisHandle) -> None:
        """Synthesize speech from the source"""
        tts_source = handle._tts_source
        transcript_source = handle._transcript_source

        if isinstance(tts_source, Awaitable):
            tts_source = await tts_source
        if isinstance(transcript_source, Awaitable):
            transcript_source = await transcript_source

        tts_stream: AsyncIterable[str] | None = None
        log.pipeline(f"tts_source: {tts_source}")
        log.pipeline(f"type of tts_source: {type(tts_source)}")
        log.pipeline(f"transcript_source: {transcript_source}")
        log.pipeline(f"type of transcript_source: {type(transcript_source)}")
        if isinstance(tts_source, str):
            # wrap in async iterator
            log.pipeline(f"wrapping tts_source in async iterator: {tts_source}")

            async def string_to_stream(text: str):
                yield text

            tts_stream = string_to_stream(tts_source)
        else:
            tts_stream = tts_source
        co = self._stream_synthesis_task(tts_stream, transcript_source, handle)

        stream_synthesis_task_id = f"StreamSynthesis-{str(uuid.uuid4())}"
        log.verbose(f"Starting task: {stream_synthesis_task_id}")
        pending_tasks = AppConfig().get_call_metadata().get("pending_livekit_tasks", {})
        pending_tasks[stream_synthesis_task_id] = time.time()
        synth = asyncio.create_task(co)
        synth.add_done_callback(lambda _: handle._buf_ch.close())

        def _post_task_callback(_) -> None:
            log.verbose(f"Task completed: {stream_synthesis_task_id}")
            pending_tasks.pop(stream_synthesis_task_id, None)

        synth.add_done_callback(_post_task_callback)

        try:
            _ = await asyncio.wait(
                [synth, handle._interrupt_fut], return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            await utils.aio.gracefully_cancel(synth)

    @utils.log_exceptions(logger=log)
    async def _read_transcript_task(
        self, transcript_source: AsyncIterable[str] | str, handle: SynthesisHandle
    ) -> None:
        try:
            if isinstance(transcript_source, str):
                handle._tr_fwd.push_text(transcript_source)
            else:
                async for seg in transcript_source:
                    if not handle._tr_fwd.closed:
                        handle._tr_fwd.push_text(seg)

            if not handle.tts_forwarder.closed:
                handle.tts_forwarder.mark_text_segment_end()
        finally:
            if inspect.isasyncgen(transcript_source):
                await transcript_source.aclose()

    @utils.log_exceptions(logger=log)
    async def _stream_synthesis_task(
        self,
        tts_source: AsyncIterable[str],
        transcript_source: AsyncIterable[str] | str,
        handle: SynthesisHandle,
    ) -> None:
        """synthesize speech from streamed text"""

        @utils.log_exceptions(logger=log)
        async def _read_generated_audio_task(
            tts_stream: text_to_speech.SynthesizeStream,
        ) -> None:
            try:
                async for audio in tts_stream:
                    if not handle._tr_fwd.closed:
                        handle._tr_fwd.push_audio(audio.frame)

                    handle._buf_ch.send_nowait(audio.frame)
            finally:
                if handle._tr_fwd and not handle._tr_fwd.closed:
                    handle._tr_fwd.mark_audio_segment_end()

                await tts_stream.aclose()

        tts_stream: text_to_speech.SynthesizeStream | None = None
        read_tts_atask: asyncio.Task | None = None
        read_transcript_atask: asyncio.Task | None = None

        try:
            async for seg in tts_source:
                log.pipeline(f"segment: {seg}")
                if tts_stream is None:
                    log.pipeline("creating new tts stream")
                    tts_stream = handle._tts.stream()
                    flusher = StreamFlusher(tts_stream)

                    read_tts_atask_id = f"ReadTTS-{str(uuid.uuid4())}"
                    log.verbose(f"Starting task: {read_tts_atask_id}")
                    pending_tasks = (
                        AppConfig().get_call_metadata().get("pending_livekit_tasks", {})
                    )
                    pending_tasks[read_tts_atask_id] = time.time()
                    read_tts_atask = asyncio.create_task(
                        _read_generated_audio_task(tts_stream)
                    )

                    def _post_task_callback_1(_) -> None:
                        log.verbose(f"Task completed: {read_tts_atask_id}")
                        pending_tasks.pop(read_tts_atask_id, None)

                    read_tts_atask.add_done_callback(_post_task_callback_1)

                    read_transcript_atask_id = f"ReadTranscript-{str(uuid.uuid4())}"
                    log.verbose(f"Starting task: {read_transcript_atask_id}")
                    pending_tasks = (
                        AppConfig().get_call_metadata().get("pending_livekit_tasks", {})
                    )
                    pending_tasks[read_transcript_atask_id] = time.time()

                    read_transcript_atask = asyncio.create_task(
                        self._read_transcript_task(transcript_source, handle)
                    )

                    def _post_task_callback_2(_) -> None:
                        log.verbose(f"Task completed: {read_transcript_atask_id}")
                        pending_tasks.pop(read_transcript_atask_id, None)

                    read_transcript_atask.add_done_callback(_post_task_callback_2)
                
                log.pipeline(f"Agent Output seg: {seg}")

                flusher.push_text(seg)


            if tts_stream is not None:
                log.pipeline("ending tts stream")
                tts_stream.end_input()
                assert read_transcript_atask and read_tts_atask
                log.pipeline("waiting for read_tts_atask")
                await read_tts_atask
                log.pipeline("waiting for read_transcript_atask")
                await read_transcript_atask

        finally:
            if read_tts_atask is not None:
                assert read_transcript_atask is not None
                await utils.aio.gracefully_cancel(read_tts_atask, read_transcript_atask)

            if inspect.isasyncgen(tts_source):
                await tts_source.aclose()
