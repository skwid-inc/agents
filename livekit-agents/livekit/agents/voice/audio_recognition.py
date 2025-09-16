from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable
from typing import Literal, Protocol

from livekit import rtc

from .. import llm, metrics, stt, utils, vad
from ..stt import SpeechEventType
from ..debug import tracing
from ..log import logger
from ..utils import aio
from . import io
from .agent import ModelSettings


class _TurnDetector(Protocol):
    # TODO: Move those two functions to EOU ctor (capabilities dataclass)
    def unlikely_threshold(self) -> float: ...
    def supports_language(self, language: str | None) -> bool: ...

    async def predict_end_of_turn(self, chat_ctx: llm.ChatContext) -> float: ...


class RecognitionHooks(Protocol):
    def on_start_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_vad_inference_done(self, ev: vad.VADEvent) -> None: ...
    def on_end_of_speech(self, ev: vad.VADEvent) -> None: ...
    def on_interim_transcript(self, ev: stt.SpeechEvent) -> None: ...
    def on_final_transcript(self, ev: stt.SpeechEvent) -> None: ...
    async def on_end_of_turn(self, new_transcript: str) -> None: ...

    def retrieve_chat_ctx(self) -> llm.ChatContext: ...


class AudioRecognition(rtc.EventEmitter[Literal["metrics_collected"]]):
    def __init__(
        self,
        *,
        hooks: RecognitionHooks,
        stt: io.STTNode | None,
        vad: vad.VAD | None,
        turn_detector: _TurnDetector | None,
        min_endpointing_delay: float,
        max_endpointing_delay: float,
    ) -> None:
        super().__init__()
        self._hooks = hooks
        self._audio_input_atask: asyncio.Task[None] | None = None
        self._stt_atask: asyncio.Task[None] | None = None
        self._vad_atask: asyncio.Task[None] | None = None
        self._end_of_turn_task: asyncio.Task[None] | None = None
        self._min_endpointing_delay = min_endpointing_delay
        self._max_endpointing_delay = max_endpointing_delay
        self._turn_detector = turn_detector
        self._stt = stt
        self._vad = vad

        # Variables to support fast interim transcript.
        self._previous_transcript_type_sent_to_llm: SpeechEventType = SpeechEventType.FINAL_TRANSCRIPT
        self._current_llm_transcript_type_sent_to_llm: SpeechEventType = SpeechEventType.FINAL_TRANSCRIPT
        self._most_recent_transcript_type: SpeechEventType = SpeechEventType.FINAL_TRANSCRIPT
        self._previous_transcript_sent_to_llm = ""
        self._committed_final_transcript = ""
        self._previous_committed_final_transcript = ""
        self._current_interim_transcript = ""
        self._interim_conf_threshold: float = 0.7

        self._speaking = False
        self._last_speaking_time: float = 0
        self._last_transcript_arrival_time: float = 0
        self._audio_transcript = ""
        self._last_language: str | None = None
        self._audio_stream_start_time: float | None = None
        self._audio_stream_start_time_history: list[float] = []
        self._last_transcript_end_time: float = 0
        self._vad_graph = tracing.Tracing.add_graph(
            title="vad",
            x_label="time",
            y_label="speech_probability",
            x_type="time",
            y_range=(0, 1),
            max_data_points=int(30 * 30),
        )

        self._stt_ch: aio.Chan[rtc.AudioFrame] | None = None
        self._vad_ch: aio.Chan[rtc.AudioFrame] | None = None

    def start(self) -> None:
        self.update_stt(self._stt)
        self.update_vad(self._vad)

    def stop(self) -> None:
        self.update_stt(None)
        self.update_vad(None)

    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._audio_stream_start_time is None:
            self._audio_stream_start_time = time.time()
            self._audio_stream_start_time_history.append(self._audio_stream_start_time)
            logger.info(f"Pushing audio, setting audio stream start time to {self._audio_stream_start_time}")
        if self._stt_ch is not None:
            self._stt_ch.send_nowait(frame)

        if self._vad_ch is not None:
            self._vad_ch.send_nowait(frame)

    async def aclose(self) -> None:
        if self._stt_atask is not None:
            await aio.cancel_and_wait(self._stt_atask)

        if self._vad_atask is not None:
            await aio.cancel_and_wait(self._vad_atask)

        if self._end_of_turn_task is not None:
            await aio.cancel_and_wait(self._end_of_turn_task)

    def update_stt(self, stt: io.STTNode | None) -> None:
        self._stt = stt
        if stt:
            logger.info(f"Updating STT, resetting audio stream start time at {time.time()}")
            self._audio_stream_start_time = None  # Reset when STT is updated
            self._stt_ch = aio.Chan[rtc.AudioFrame]()
            self._stt_atask = asyncio.create_task(self._stt_task(stt, self._stt_ch, self._stt_atask))
        elif self._stt_atask is not None:
            self._stt_atask.cancel()
            self._stt_atask = None
            self._stt_ch = None

    def update_vad(self, vad: vad.VAD | None) -> None:
        self._vad = vad
        if vad:
            self._vad_ch = aio.Chan[rtc.AudioFrame]()
            self._vad_atask = asyncio.create_task(self._vad_task(vad, self._vad_ch, self._vad_atask))
        elif self._vad_atask is not None:
            self._vad_atask.cancel()
            self._vad_atask = None
            self._vad_ch = None

    def _estimate_actual_speech_end_time(self) -> float:
        # This is the wall clock time in epoch seconds when the user stopped speaking.
        # This is calculated by adding the time of the last transcript end time(DG clock) with the
        # audio stream start time (wall clock).
        # Ex: 8s + 1750184202.385735 = 1750184210.385735
        # _audio_stream_start_time is set by us above when we receive the first audio frame, it's reset on a Language switch.
        return self._last_transcript_end_time + self._audio_stream_start_time

    async def _on_stt_event(self, ev: stt.SpeechEvent) -> None:
        if ev.type == SpeechEventType.FINAL_TRANSCRIPT:
            self._most_recent_transcript_type = SpeechEventType.FINAL_TRANSCRIPT
            self._hooks.on_final_transcript(ev)
            transcript = ev.alternatives[0].text
            self._last_language = ev.alternatives[0].language
            if not transcript:
                return

            logger.info(f"_on_stt_event: received final user transcript: {transcript}")

            tracing.Tracing.log_event(
                "user transcript",
                {
                    "transcript": transcript,
                    "buffered_transcript": self._audio_transcript,
                },
            )

            self._last_transcript_arrival_time = time.time()
            self._audio_transcript += f" {transcript}"
            self._audio_transcript = self._audio_transcript.lstrip()

            if hasattr(ev.alternatives[0], "end_time") and ev.alternatives[0].end_time > 0:
                self._last_transcript_end_time = ev.alternatives[0].end_time

            self._committed_final_transcript = (self._committed_final_transcript + " " + transcript).strip()

            # Final transcripts resets any existing interim transcripts.
            if self._current_interim_transcript:
                self._current_interim_transcript = ""

            self._audio_transcript = self._committed_final_transcript

            if not self._speaking:
                if not self._vad:
                    # vad disabled, use stt timestamp
                    # TODO: this would screw up transcription latency metrics
                    # but we'll live with it for now.
                    # the correct way is to ensure STT fires SpeechEventType.END_OF_SPEECH
                    # and using that timestamp for _last_speaking_time
                    self._last_speaking_time = time.time()

                chat_ctx = self._hooks.retrieve_chat_ctx().copy()
                await self._run_eou_detection(chat_ctx)

        elif ev.type == SpeechEventType.INTERIM_TRANSCRIPT:
            self._most_recent_transcript_type = SpeechEventType.INTERIM_TRANSCRIPT

            self._hooks.on_interim_transcript(ev)
            # Allow high-confidence interim to advance the buffer.
            if not ev.alternatives:
                return
            transcript_alternative = ev.alternatives[0]
            text = getattr(transcript_alternative, "text", "") or ""
            if not text:
                return
            confidence = float(getattr(transcript_alternative, "confidence", 0.0) or 0.0)
            if confidence >= self._interim_conf_threshold:
                self._current_interim_transcript = text
                self._last_language = getattr(transcript_alternative, "language", self._last_language)

                # Build the transcript buffer with latest commited final + latest interim.
                self._audio_transcript = (
                    self._committed_final_transcript + " " + self._current_interim_transcript
                ).strip()

                # Set the _last_transcript_end_time for this interim transcript.
                if hasattr(ev.alternatives[0], "end_time") and ev.alternatives[0].end_time > 0:
                    self._last_transcript_end_time = ev.alternatives[0].end_time
                self._last_transcript_arrival_time = time.time()

                logger.debug(
                    f"confident interim accepted with confidence={confidence}\n"
                    f"committed_transcript={self._committed_final_transcript}\n"
                    f"current_interim_transcript={self._current_interim_transcript}\n"
                    f"audio_transcript={self._audio_transcript}"
                )

    async def _on_vad_event(self, ev: vad.VADEvent) -> None:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            self._hooks.on_start_of_speech(ev)
            self._speaking = True

            if self._end_of_turn_task is not None:
                self._end_of_turn_task.cancel()

        elif ev.type == vad.VADEventType.INFERENCE_DONE:
            self._vad_graph.plot(ev.timestamp, ev.probability)
            self._hooks.on_vad_inference_done(ev)

        elif ev.type == vad.VADEventType.END_OF_SPEECH:
            self._hooks.on_end_of_speech(ev)
            self._speaking = False
            # when VAD fires END_OF_SPEECH, it already waited for the silence_duration
            self._last_speaking_time = time.time() - ev.silence_duration

            chat_ctx = self._hooks.retrieve_chat_ctx().copy()
            await self._run_eou_detection(chat_ctx)

    async def _run_eou_detection(self, chat_ctx: llm.ChatContext) -> None:
        # If should_force_final_transcript is on, then we need to only use final transcript.
        if (
            self._hooks._agent.should_force_final_transcript
            and self._most_recent_transcript_type == SpeechEventType.INTERIM_TRANSCRIPT
        ):
            logger.info(
                "Force final transcript is enabled, setting audio transcript from interim: "
                f"{self._audio_transcript} to final: {self._committed_final_transcript}"
            )
            self._most_recent_transcript_type = SpeechEventType.FINAL_TRANSCRIPT
            self._audio_transcript = self._committed_final_transcript

        if self._stt and not self._audio_transcript:
            # stt enabled but no transcript yet
            return

        self._previous_transcript_type_sent_to_llm = self._current_llm_transcript_type_sent_to_llm
        self._current_llm_transcript_type_sent_to_llm = self._most_recent_transcript_type

        logger.info(
            f"previous_transcript_type_sent_to_llm: {self._previous_transcript_type_sent_to_llm}\n"
            f"current_llm_transcript_type_sent_to_llm: {self._current_llm_transcript_type_sent_to_llm}"
        )

        # The algorithm below describes the process of leveraging interim transcripts
        # to reduce uneccessary LLM calls from final transcripts. The first thing we need
        # to know if if the last LLM call is done using an interim or final transcript:
        #
        # - If the previous LLM request is an interim transcript, two things can happen:
        #   1) current transcript sent to LLM is the same as the previous transcript
        #      sent to LLM. If so: skip the LLM call. Because the exact same call is
        #      already made.
        #   2) current transcript sent to LLM  is not the same as the previous transcript
        #      sent to the LLM. In this case, delete the latest user turn in chat context,
        #      and issue a brand new LLM request. This in affect, "updates" the previous user chat
        #      message into a more up to date message, and issue a new LLM call.
        # - If the previous LLM request is final, we skip this optimization and proceed with
        # LLM generation directly, because any new transcript after Final is guaranteed to be
        # covering a new audio segment.
        # Note that this optimization should be skipped entirely if should_force_final_transcript is set to True.
        #   This is because we won't be deleting any interims.
        if (
            chat_ctx.is_previous_message_user_role()
            and self._previous_transcript_type_sent_to_llm == SpeechEventType.INTERIM_TRANSCRIPT
            and not self._hooks._agent.should_force_final_transcript
        ):
            if self._previous_transcript_sent_to_llm == self._audio_transcript:
                logger.info(f"Skipping LLM call for same transcript: {self._audio_transcript}")
                # We need to reset the transcript buffer, because the exact same LLM request already
                # happened and added to chat_ctx.
                self._audio_transcript = ""
                self._committed_final_transcript = ""
                self._current_interim_transcript = ""
                # Reseting previous committed transcript to "" because when we have to "rollback" the next Chat_ctx,
                # We *don't* want to include this committed transcript, since it would duplicate the prior interim.
                self._previous_committed_final_transcript = ""
                self._committed_final_transcript = ""
                return

            # Delete the previous user's chat_message which we already know came from an interim transcript.
            did_delete = chat_ctx.maybe_delete_latest_user_interim_message()
            self._hooks._session._chat_ctx.maybe_delete_latest_user_interim_message()
            if did_delete:
                # A portion of the deleted interim message could be final. If so,
                # We need to add the final transcript back to the buffer.
                self._audio_transcript = (
                    self._previous_committed_final_transcript + " " + self._audio_transcript
                ).strip()
            await self._hooks.update_chat_ctx(chat_ctx)

        if self._current_llm_transcript_type_sent_to_llm == SpeechEventType.FINAL_TRANSCRIPT:
            # If the current transcript is final, we want to clear the recent commited buffer so if we need to rollback,
            # We never include this.
            # With this logic, _committed_final_transcript and _previous_committed_final_transcript will only contain info
            # if the previous LLM is done on an interim transcript.
            self._previous_committed_final_transcript = ""
            self._committed_final_transcript = ""

        logger.info(
            f"eou_detection dump:\n _audio_transcript: {self._audio_transcript}\n"
            f"_previous_committed_final_transcript: {self._previous_committed_final_transcript}\n"
            f"_committed_final_transcript: {self._committed_final_transcript}\n"
            f"_current_interim_transcript: {self._current_interim_transcript}\n"
        )
        self._previous_transcript_sent_to_llm = self._audio_transcript

        chat_ctx = chat_ctx.copy()
        chat_ctx.add_message(role="user", content=self._audio_transcript)
        turn_detector = self._turn_detector if self._audio_transcript else None

        @utils.log_exceptions(logger=logger)
        async def _bounce_eou_task() -> None:
            endpointing_delay = self._min_endpointing_delay

            if turn_detector is not None and turn_detector.supports_language(self._last_language):
                end_of_turn_probability = await turn_detector.predict_end_of_turn(chat_ctx)
                tracing.Tracing.log_event(
                    "end of user turn probability",
                    {"probability": end_of_turn_probability},
                )
                unlikely_threshold = turn_detector.unlikely_threshold()
                if end_of_turn_probability < unlikely_threshold:
                    endpointing_delay = self._max_endpointing_delay

            await asyncio.sleep(
                max(
                    self._last_speaking_time + endpointing_delay - time.time(),
                    0,
                )
            )

            tracing.Tracing.log_event("end of user turn", {"transcript": self._audio_transcript})

            actual_speech_end_time = self._estimate_actual_speech_end_time()
            # _last_final_transcript_time is set by us above when we receive the final transcript from DG
            transcription_delay = max(self._last_transcript_arrival_time - actual_speech_end_time, 0)
            end_of_utterance_delay = max(time.time() - actual_speech_end_time, 0)

            # These are just debugging logs to help us understand the flow of the code. Not used anywhere.
            logger.info(
                f"Debug transcription delay calculation: "
                f"audio_stream_start={self._audio_stream_start_time},"
                f"last_transcript_end_time={self._last_transcript_end_time}, "
                f"actual_speech_end_time={actual_speech_end_time}, "
                f"transcription_delay={transcription_delay}, "
                f"last_transcript_arrival_time={self._last_transcript_arrival_time}, "
                f"last_speaking_time_vad={self._last_speaking_time}"
                f"stream history: {self._audio_stream_start_time_history}"
            )

            # We inject [beep detected] transcripts manually in voice detection. If this type of transcript is found, do not attempt to emit metrics as it will distort EOU/Transcript delay measurements.
            if "[beep detected]" not in self._audio_transcript:
                # These numbers are emitted to taylor fresh and used to calculate the turn latency.
                eou_metrics = metrics.EOUMetrics(
                    timestamp=time.time(),
                    end_of_utterance_delay=end_of_utterance_delay,
                    transcription_delay=transcription_delay,
                )
                self.emit("metrics_collected", eou_metrics)
            else:
                logger.info("Skipping EOU metrics emission for [beep detected] transcript")

            try:
                await self._hooks.on_end_of_turn(self._audio_transcript)
            except Exception:
                # When the exception above triggers, we still need to reset the buffers.
                self._audio_transcript = ""
                self._previous_committed_final_transcript = self._committed_final_transcript
                self._committed_final_transcript = ""
                self._current_interim_transcript = ""
                raise

            self._audio_transcript = ""
            self._previous_committed_final_transcript = self._committed_final_transcript
            self._committed_final_transcript = ""
            self._current_interim_transcript = ""

        if self._end_of_turn_task is not None:
            self._end_of_turn_task.cancel()

        self._end_of_turn_task = asyncio.create_task(_bounce_eou_task())

    @utils.log_exceptions(logger=logger)
    async def _stt_task(
        self,
        stt_node: io.STTNode,
        audio_input: io.AudioInput,
        task: asyncio.Task[None] | None,
    ) -> None:
        if task is not None:
            await aio.cancel_and_wait(task)

        node = stt_node(audio_input, ModelSettings())
        if asyncio.iscoroutine(node):
            node = await node

        if node is None:
            return

        if isinstance(node, AsyncIterable):
            async for ev in node:
                assert isinstance(ev, stt.SpeechEvent), "STT node must yield SpeechEvent"
                await self._on_stt_event(ev)

    @utils.log_exceptions(logger=logger)
    async def _vad_task(self, vad: vad.VAD, audio_input: io.AudioInput, task: asyncio.Task[None] | None) -> None:
        if task is not None:
            await aio.cancel_and_wait(task)

        stream = vad.stream()

        @utils.log_exceptions(logger=logger)
        async def _forward() -> None:
            async for frame in audio_input:
                stream.push_frame(frame)

        forward_task = asyncio.create_task(_forward())

        try:
            async for ev in stream:
                await self._on_vad_event(ev)
        finally:
            await aio.cancel_and_wait(forward_task)
            await stream.aclose()
