from __future__ import annotations

import asyncio
import contextvars
import time
import uuid
from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterable,
    Awaitable,
    Callable,
    Literal,
    Optional,
    Protocol,
    Union,
)

from app_config import AppConfig
from livekit import rtc

from .. import metrics, stt, tokenize, tts, utils, vad
from ..llm import LLM, ChatContext, ChatMessage, FunctionContext, LLMStream
from ..types import ATTRIBUTE_AGENT_STATE, AgentState
from .agent_output import AgentOutput, SpeechSource, SynthesisHandle
from .agent_playout import AgentPlayout
from .human_input import HumanInput
from custom_logger import log
from .plotter import AssistantPlotter
from .speech_handle import SpeechHandle

BeforeLLMCallback = Callable[
    ["VoicePipelineAgent", ChatContext],
    Union[
        Optional[LLMStream],
        Awaitable[Optional[LLMStream]],
        Literal[False],
        Awaitable[Literal[False]],
    ],
]

WillSynthesizeAssistantReply = BeforeLLMCallback

BeforeTTSCallback = Callable[
    ["VoicePipelineAgent", Union[str, AsyncIterable[str]]],
    SpeechSource,
]


EventTypes = Literal[
    "user_started_speaking",
    "user_stopped_speaking",
    "agent_started_speaking",
    "agent_stopped_speaking",
    "user_speech_committed",
    "agent_speech_committed",
    "agent_speech_interrupted",
    "function_calls_collected",
    "function_calls_finished",
    "metrics_collected",
]

_CallContextVar = contextvars.ContextVar["AgentCallContext"](
    "voice_assistant_contextvar"
)


class AgentCallContext:
    def __init__(self, assistant: "VoicePipelineAgent", llm_stream: LLMStream) -> None:
        self._assistant = assistant
        self._metadata = dict[str, Any]()
        self._llm_stream = llm_stream
        self._extra_chat_messages: list[ChatMessage] = []

    @staticmethod
    def get_current() -> "AgentCallContext":
        return _CallContextVar.get()

    @property
    def agent(self) -> "VoicePipelineAgent":
        return self._assistant

    @property
    def chat_ctx(self) -> ChatContext:
        return self._llm_stream.chat_ctx

    def store_metadata(self, key: str, value: Any) -> None:
        self._metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def llm_stream(self) -> LLMStream:
        return self._llm_stream

    def add_extra_chat_message(self, message: ChatMessage) -> None:
        """Append chat message to the end of function outputs for the answer LLM call"""
        self._extra_chat_messages.append(message)

    @property
    def extra_chat_messages(self) -> list[ChatMessage]:
        return self._extra_chat_messages


def _default_before_llm_cb(
    agent: VoicePipelineAgent, chat_ctx: ChatContext
) -> LLMStream:
    return agent.llm.chat(
        chat_ctx=chat_ctx,
        fnc_ctx=agent.fnc_ctx,
    )


@dataclass
class SpeechData:
    sequence_id: str


SpeechDataContextVar = contextvars.ContextVar[SpeechData]("voice_assistant_speech_data")


def _default_before_tts_cb(
    agent: VoicePipelineAgent, text: str | AsyncIterable[str]
) -> str | AsyncIterable[str]:
    return text


@dataclass(frozen=True)
class _ImplOptions:
    allow_interruptions: bool
    int_speech_duration: float
    int_min_words: int
    min_endpointing_delay: float
    max_endpointing_delay: float
    max_nested_fnc_calls: int
    preemptive_synthesis: bool
    before_llm_cb: BeforeLLMCallback
    before_tts_cb: BeforeTTSCallback
    plotting: bool
    transcription: AgentTranscriptionOptions


@dataclass(frozen=True)
class AgentTranscriptionOptions:
    user_transcription: bool = True
    """Whether to forward the user transcription to the client"""
    agent_transcription: bool = True
    """Whether to forward the agent transcription to the client"""
    agent_transcription_speed: float = 1.0
    """The speed at which the agent's speech transcription is forwarded to the client.
    We try to mimic the agent's speech speed by adjusting the transcription speed."""
    sentence_tokenizer: tokenize.SentenceTokenizer = tokenize.basic.SentenceTokenizer()
    """The tokenizer used to split the speech into sentences.
    This is used to decide when to mark a transcript as final for the agent transcription."""
    word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
        ignore_punctuation=False
    )
    """The tokenizer used to split the speech into words.
    This is used to simulate the "interim results" of the agent transcription."""
    hyphenate_word: Callable[[str], list[str]] = tokenize.basic.hyphenate_word
    """A function that takes a string (word) as input and returns a list of strings,
    representing the hyphenated parts of the word."""


class _TurnDetector(Protocol):
    # When endpoint probability is below this threshold we think the user is not finished speaking
    # so we will use a long delay
    def unlikely_threshold(self) -> float: ...
    def supports_language(self, language: str | None) -> bool: ...
    async def predict_end_of_turn(self, chat_ctx: ChatContext) -> float: ...


class VoicePipelineAgent(utils.EventEmitter[EventTypes]):
    """
    A pipeline agent (VAD + STT + LLM + TTS) implementation.
    """

    MIN_TIME_PLAYED_FOR_COMMIT = 0.1
    """Minimum time played for the user speech to be committed to the chat context"""

    def __init__(
        self,
        *,
        vad: vad.VAD,
        stt: stt.STT,
        llm: LLM,
        tts: tts.TTS,
        noise_cancellation: rtc.NoiseCancellationOptions | None = None,
        turn_detector: _TurnDetector | None = None,
        chat_ctx: ChatContext | None = None,
        fnc_ctx: FunctionContext | None = None,
        allow_interruptions: bool = True,
        interrupt_speech_duration: float = 0.5,
        interrupt_min_words: int = 0,
        min_endpointing_delay: float = 0.5,
        max_endpointing_delay: float = 6.0,
        max_nested_fnc_calls: int = 1,
        preemptive_synthesis: bool = False,
        transcription: AgentTranscriptionOptions = AgentTranscriptionOptions(),
        before_llm_cb: BeforeLLMCallback = _default_before_llm_cb,
        before_tts_cb: BeforeTTSCallback = _default_before_tts_cb,
        plotting: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        # backward compatibility
        will_synthesize_assistant_reply: WillSynthesizeAssistantReply | None = None,
    ) -> None:
        """
        Create a new VoicePipelineAgent.

        Args:
            vad: Voice Activity Detection (VAD) instance.
            stt: Speech-to-Text (STT) instance.
            llm: Large Language Model (LLM) instance.
            tts: Text-to-Speech (TTS) instance.
            chat_ctx: Chat context for the assistant.
            fnc_ctx: Function context for the assistant.
            allow_interruptions: Whether to allow the user to interrupt the assistant.
            interrupt_speech_duration: Minimum duration of speech to consider for interruption.
            interrupt_min_words: Minimum number of words to consider for interruption.
                Defaults to 0 as this may increase the latency depending on the STT.
            min_endpointing_delay: Delay to wait before considering the user finished speaking.
            max_nested_fnc_calls: Maximum number of nested function calls allowed for chaining
                function calls (e.g functions that depend on each other).
            preemptive_synthesis: Whether to preemptively synthesize responses.
            transcription: Options for assistant transcription.
            before_llm_cb: Callback called when the assistant is about to synthesize a reply.
                This can be used to customize the reply (e.g: inject context/RAG).

                Returning None will create a default LLM stream. You can also return your own llm
                stream by calling the llm.chat() method.

                Returning False will cancel the synthesis of the reply.
            before_tts_cb: Callback called when the assistant is about to
                synthesize a speech. This can be used to customize text before the speech synthesis.
                (e.g: editing the pronunciation of a word).
            plotting: Whether to enable plotting for debugging. matplotlib must be installed.
            loop: Event loop to use. Default to asyncio.get_event_loop().
        """
        super().__init__()
        self._loop = loop or asyncio.get_event_loop()

        if will_synthesize_assistant_reply is not None:
            log.warning(
                "will_synthesize_assistant_reply is deprecated and will be removed in 1.5.0, use before_llm_cb instead",
            )
            before_llm_cb = will_synthesize_assistant_reply

        self._opts = _ImplOptions(
            plotting=plotting,
            allow_interruptions=allow_interruptions,
            int_speech_duration=interrupt_speech_duration,
            int_min_words=interrupt_min_words,
            min_endpointing_delay=min_endpointing_delay,
            max_endpointing_delay=max_endpointing_delay,
            max_nested_fnc_calls=max_nested_fnc_calls,
            preemptive_synthesis=preemptive_synthesis,
            transcription=transcription,
            before_llm_cb=before_llm_cb,
            before_tts_cb=before_tts_cb,
        )
        self._plotter = AssistantPlotter(self._loop)

        # wrap with StreamAdapter automatically when streaming is not supported on a specific TTS/STT.
        # To override StreamAdapter options, create the adapter manually.

        if not tts.capabilities.streaming:
            from .. import tts as text_to_speech

            tts = text_to_speech.StreamAdapter(
                tts=tts, sentence_tokenizer=tokenize.basic.SentenceTokenizer()
            )

        if not stt.capabilities.streaming:
            from .. import stt as speech_to_text

            stt = speech_to_text.StreamAdapter(
                stt=stt,
                vad=vad,
            )

        self._stt, self._vad, self._llm, self._tts = stt, vad, llm, tts
        self._turn_detector = turn_detector
        self._chat_ctx = chat_ctx or ChatContext()
        self._fnc_ctx = fnc_ctx
        self._started, self._closed = False, False

        self._human_input: HumanInput | None = None
        self._agent_output: AgentOutput | None = None

        # done when the agent output track is published
        self._track_published_fut = asyncio.Future[None]()

        self._pending_agent_reply: SpeechHandle | None = None
        self._agent_reply_task: asyncio.Task[None] | None = None

        self._playing_speech: SpeechHandle | None = None
        self._transcribed_text, self._transcribed_interim_text = "", ""

        self._deferred_validation = _DeferredReplyValidation(
            self._validate_reply_if_possible,
            min_endpointing_delay=self._opts.min_endpointing_delay,
            max_endpointing_delay=self._opts.max_endpointing_delay,
            turn_detector=self._turn_detector,
            agent=self,
        )

        self._speech_q: list[SpeechHandle] = []
        self._speech_q_changed = asyncio.Event()

        self._update_state_task: asyncio.Task | None = None

        self._last_final_transcript_time: float | None = None
        self._last_speech_time: float | None = None

        self._noise_cancellation = noise_cancellation

    @property
    def fnc_ctx(self) -> FunctionContext | None:
        return self._fnc_ctx

    @fnc_ctx.setter
    def fnc_ctx(self, fnc_ctx: FunctionContext | None) -> None:
        self._fnc_ctx = fnc_ctx

    @property
    def chat_ctx(self) -> ChatContext:
        return self._chat_ctx

    @property
    def llm(self) -> LLM:
        return self._llm

    @property
    def tts(self) -> tts.TTS:
        return self._tts

    @property
    def stt(self) -> stt.STT:
        return self._stt

    @property
    def vad(self) -> vad.VAD:
        return self._vad

    def start(
        self, room: rtc.Room, participant: rtc.RemoteParticipant | str | None = None
    ) -> None:
        """Start the voice assistant

        Args:
            room: the room to use
            participant: the participant to listen to, can either be a participant or a participant identity
                If None, the first participant found in the room will be selected
        """
        if self._started:
            raise RuntimeError("voice assistant already started")

        @self._stt.on("metrics_collected")
        def _on_stt_metrics(stt_metrics: metrics.STTMetrics) -> None:
            self.emit(
                "metrics_collected",
                metrics.PipelineSTTMetrics(
                    **stt_metrics.__dict__,
                ),
            )

        @self._tts.on("metrics_collected")
        def _on_tts_metrics(tts_metrics: metrics.TTSMetrics) -> None:
            speech_data = SpeechDataContextVar.get(None)
            if speech_data is None:
                return

            self.emit(
                "metrics_collected",
                metrics.PipelineTTSMetrics(
                    **tts_metrics.__dict__,
                    sequence_id=speech_data.sequence_id,
                ),
            )

        @self._llm.on("metrics_collected")
        def _on_llm_metrics(llm_metrics: metrics.LLMMetrics) -> None:
            speech_data = SpeechDataContextVar.get(None)
            if speech_data is None:
                return
            self.emit(
                "metrics_collected",
                metrics.PipelineLLMMetrics(
                    **llm_metrics.__dict__,
                    sequence_id=speech_data.sequence_id,
                ),
            )

        @self._vad.on("metrics_collected")
        def _on_vad_metrics(vad_metrics: vad.VADMetrics) -> None:
            self.emit(
                "metrics_collected", metrics.PipelineVADMetrics(**vad_metrics.__dict__)
            )

        room.on("participant_connected", self._on_participant_connected)
        self._room, self._participant = room, participant

        if participant is not None:
            if isinstance(participant, rtc.RemoteParticipant):
                self._link_participant(participant.identity)
            else:
                self._link_participant(participant)
        else:
            # no participant provided, try to find the first participant in the room
            for participant in self._room.remote_participants.values():
                self._link_participant(participant.identity)
                break

        self._main_atask = asyncio.create_task(self._main_task())

    def on(self, event: EventTypes, callback: Callable[[Any], None] | None = None):
        """Register a callback for an event

        Args:
            event: the event to listen to (see EventTypes)
                - user_started_speaking: the user started speaking
                - user_stopped_speaking: the user stopped speaking
                - agent_started_speaking: the agent started speaking
                - agent_stopped_speaking: the agent stopped speaking
                - user_speech_committed: the user speech was committed to the chat context
                - agent_speech_committed: the agent speech was committed to the chat context
                - agent_speech_interrupted: the agent speech was interrupted
                - function_calls_collected: received the complete set of functions to be executed
                - function_calls_finished: all function calls have been completed
            callback: the callback to call when the event is emitted
        """
        return super().on(event, callback)

    async def say(
        self,
        source: str | LLMStream | AsyncIterable[str],
        *,
        allow_interruptions: bool = True,
        add_to_chat_ctx: bool = True,
    ) -> SpeechHandle:
        """
        Play a speech source through the voice assistant.

        Args:
            source: The source of the speech to play.
                It can be a string, an LLMStream, or an asynchronous iterable of strings.
            allow_interruptions: Whether to allow interruptions during the speech playback.
            add_to_chat_ctx: Whether to add the speech to the chat context.

        Returns:
            The speech handle for the speech that was played, can be used to
            wait for the speech to finish.
        """
        await self._track_published_fut

        call_ctx = None
        fnc_source: str | AsyncIterable[str] | None = None
        if add_to_chat_ctx:
            try:
                call_ctx = AgentCallContext.get_current()
            except LookupError:
                # no active call context, ignore
                pass
            else:
                if isinstance(source, LLMStream):
                    log.warning(
                        "LLMStream will be ignored for function call chat context"
                    )
                elif isinstance(source, AsyncIterable):
                    source, fnc_source = utils.aio.itertools.tee(source, 2)  # type: ignore
                else:
                    fnc_source = source

        new_handle = SpeechHandle.create_assistant_speech(
            allow_interruptions=allow_interruptions, add_to_chat_ctx=add_to_chat_ctx
        )
        synthesis_handle = self._synthesize_agent_speech(new_handle.id, source)
        new_handle.initialize(source=source, synthesis_handle=synthesis_handle)

        if self._playing_speech and not self._playing_speech.nested_speech_done:
            self._playing_speech.add_nested_speech(new_handle)
        elif self._speech_q:
            self._speech_q[0].add_nested_speech(new_handle)
        else:
            self._add_speech_for_playout(new_handle)

        # add the speech to the function call context if needed
        if call_ctx is not None and fnc_source is not None:
            if isinstance(fnc_source, AsyncIterable):
                text = ""
                async for chunk in fnc_source:
                    text += chunk
            else:
                text = fnc_source

            call_ctx.add_extra_chat_message(
                ChatMessage.create(text=text, role="assistant")
            )
            log.debug(
                "added speech to function call chat context",
                extra={"text": text},
            )

        return new_handle

    def interrupt(self, interrupt_all: bool = True) -> None:
        """Interrupt the current speech

        Args:
            interrupt_all: Whether to interrupt all pending speech
        """
        if interrupt_all:
            # interrupt all pending speech
            if self._pending_agent_reply is not None:
                self._pending_agent_reply.cancel(cancel_nested=True)
            for speech in self._speech_q:
                speech.cancel(cancel_nested=True)

        # interrupt the playing speech
        if self._playing_speech is not None:
            self._playing_speech.cancel(cancel_nested=True)

        # Stop current LLM stream
        # log.pipeline(f"cancelling agent reply task: {self._agent_reply_task}")
        # if self._agent_reply_task is not None:
        #     self._agent_reply_task.cancel()
        # if self._pending_agent_reply is not None:
        #     self._pending_agent_reply.cancel()

    def _update_state(self, state: AgentState, delay: float = 0.0):
        """Set the current state of the agent"""

        @utils.log_exceptions(logger=log)
        async def _run_task(delay: float) -> None:
            await asyncio.sleep(delay)

            if self._room.isconnected():
                await self._room.local_participant.set_attributes(
                    {ATTRIBUTE_AGENT_STATE: state}
                )

        if self._update_state_task is not None:
            self._update_state_task.cancel()

        self._update_state_task = asyncio.create_task(_run_task(delay))

    async def aclose(self) -> None:
        """Close the voice assistant"""
        if not self._started:
            return

        self._room.off("participant_connected", self._on_participant_connected)
        await self._deferred_validation.aclose()

    def _on_participant_connected(self, participant: rtc.RemoteParticipant):
        if self._human_input is not None:
            return

        self._link_participant(participant.identity)

    def _link_participant(self, identity: str) -> None:
        participant = self._room.remote_participants.get(identity)
        if participant is None:
            log.error("_link_participant must be called with a valid identity")
            return

        self._human_input = HumanInput(
            room=self._room,
            vad=self._vad,
            stt=self._stt,
            participant=participant,
            transcription=self._opts.transcription.user_transcription,
            noise_cancellation=self._noise_cancellation,
        )

        def _on_start_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_started_speaking")
            self.emit("user_started_speaking")
            log.pipeline("User started speaking so starting validation")
            self._deferred_validation.on_human_start_of_speech(ev)

        def _on_vad_inference_done(ev: vad.VADEvent) -> None:
            if not self._track_published_fut.done():
                return

            assert self._agent_output is not None

            tv = 1.0
            if self._opts.allow_interruptions:
                tv = max(0.0, 1.0 - ev.probability)
                self._agent_output.playout.target_volume = tv

            smoothed_tv = self._agent_output.playout.smoothed_volume

            self._plotter.plot_value("raw_vol", tv)
            self._plotter.plot_value("smoothed_vol", smoothed_tv)
            self._plotter.plot_value("vad_probability", ev.probability)

            if ev.speech_duration >= self._opts.int_speech_duration:
                self._interrupt_if_possible()

            if ev.raw_accumulated_speech > 0.0:
                self._last_speech_time = (
                    time.perf_counter() - ev.raw_accumulated_silence
                )

        def _on_end_of_speech(ev: vad.VADEvent) -> None:
            self._plotter.plot_event("user_stopped_speaking")
            self.emit("user_stopped_speaking")
            log.pipeline("User stopped speaking so starting validation")
            self._deferred_validation.on_human_end_of_speech(ev)

        def _on_interim_transcript(ev: stt.SpeechEvent) -> None:
            self._transcribed_interim_text = ev.alternatives[0].text

        def _on_final_transcript(ev: stt.SpeechEvent) -> None:
            new_transcript = ev.alternatives[0].text
            if not new_transcript:
                return

            log.debug(
                f"received user transcript - {new_transcript}",
                extra={"user_transcript": new_transcript},
            )
            AppConfig().received_user_transcript_timestamp = time.time()

            self._last_final_transcript_time = time.perf_counter()

            self._transcribed_text += (
                " " if self._transcribed_text else ""
            ) + new_transcript

            if self._opts.preemptive_synthesis:
                if (
                    self._playing_speech is None
                    or self._playing_speech.allow_interruptions
                ):
                    self._synthesize_agent_reply()

            log.pipeline(
                f"Validating final transcript and starting deferred validation"
            )
            self._deferred_validation.on_human_final_transcript(
                new_transcript, ev.alternatives[0].language
            )

            words = self._opts.transcription.word_tokenizer.tokenize(
                text=new_transcript
            )
            if len(words) >= 3:
                # VAD can sometimes not detect that the human is speaking
                # to make the interruption more reliable, we also interrupt on the final transcript.
                self._interrupt_if_possible()

        self._human_input.on("start_of_speech", _on_start_of_speech)
        self._human_input.on("vad_inference_done", _on_vad_inference_done)
        self._human_input.on("end_of_speech", _on_end_of_speech)
        self._human_input.on("interim_transcript", _on_interim_transcript)
        self._human_input.on("final_transcript", _on_final_transcript)

    @utils.log_exceptions(logger=log)
    async def _main_task(self) -> None:
        if self._opts.plotting:
            await self._plotter.start()

        self._update_state("initializing")
        audio_source = rtc.AudioSource(self._tts.sample_rate, self._tts.num_channels)
        track = rtc.LocalAudioTrack.create_audio_track("assistant_voice", audio_source)
        self._agent_publication = await self._room.local_participant.publish_track(
            track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        )

        agent_playout = AgentPlayout(audio_source=audio_source)
        self._agent_output = AgentOutput(
            room=self._room,
            agent_playout=agent_playout,
            llm=self._llm,
            tts=self._tts,
        )

        def _on_playout_started() -> None:
            self._plotter.plot_event("agent_started_speaking")
            self.emit("agent_started_speaking")
            self._update_state("speaking")

        def _on_playout_stopped(interrupted: bool) -> None:
            self._plotter.plot_event("agent_stopped_speaking")
            self.emit("agent_stopped_speaking")
            self._update_state("listening")

        agent_playout.on("playout_started", _on_playout_started)
        agent_playout.on("playout_stopped", _on_playout_stopped)

        self._track_published_fut.set_result(None)

        while True:
            await self._speech_q_changed.wait()

            while self._speech_q:
                speech = self._speech_q[0]
                self._playing_speech = speech
                await self._play_speech(speech)
                self._speech_q.pop(0)  # Remove the element only after playing
                self._playing_speech = None

            self._speech_q_changed.clear()

    def _synthesize_agent_reply(self):
        """Synthesize the agent reply to the user question, also make sure only one reply
        is synthesized/played at a time"""
        log.pipeline("inside _synthesize_agent_reply")
        if self._pending_agent_reply is not None:
            self._pending_agent_reply.cancel()

        if self._human_input is not None and not self._human_input.speaking:
            self._update_state("thinking", 0.2)

        self._pending_agent_reply = new_handle = SpeechHandle.create_assistant_reply(
            allow_interruptions=self._opts.allow_interruptions,
            add_to_chat_ctx=True,
            user_question=self._transcribed_text,
        )

        agent_reply_task_id = f"AgentReply-{str(uuid.uuid4())}"
        log.verbose(f"Starting task: {agent_reply_task_id}")
        pending_tasks = AppConfig().get_call_metadata().get("pending_livekit_tasks", {})
        pending_tasks[agent_reply_task_id] = time.time()
        log.pipeline("pending task - synthesizing agent reply")
        log.pipeline(pending_tasks)
        self._agent_reply_task = asyncio.create_task(
            self._synthesize_answer_task(self._agent_reply_task, new_handle)
        )
        self._agent_reply_task.add_done_callback(
            lambda t: new_handle.cancel() if t.cancelled() else None
        )

        def _post_task_callback(_) -> None:
            log.verbose(f"Task completed: {agent_reply_task_id}")
            pending_tasks.pop(agent_reply_task_id, None)
            log.pipeline(pending_tasks)

        self._agent_reply_task.add_done_callback(_post_task_callback)

    @utils.log_exceptions(logger=log)
    async def _synthesize_answer_task(
        self, old_task: asyncio.Task[None], handle: SpeechHandle
    ) -> None:
        if old_task is not None:
            await utils.aio.gracefully_cancel(old_task)

        copied_ctx = self._chat_ctx.copy()
        playing_speech = self._playing_speech
        if playing_speech is not None and playing_speech.initialized:
            if (
                not playing_speech.user_question or playing_speech.user_committed
            ) and not playing_speech.speech_committed:
                # the speech is playing but not committed yet, add it to the chat context for this new reply synthesis
                # First add the previous function call message if any
                if playing_speech.extra_tools_messages:
                    if playing_speech.fnc_text_message_id is not None:
                        # there is a message alongside the function calls
                        msgs = copied_ctx.messages
                        if msgs and msgs[-1].id == playing_speech.fnc_text_message_id:
                            # replace it with the tool call message if it's the last in the ctx
                            msgs.pop()
                    copied_ctx.messages.extend(playing_speech.extra_tools_messages)

                # Then add the previous assistant message
                copied_ctx.messages.append(
                    ChatMessage.create(
                        text=playing_speech.synthesis_handle.tts_forwarder.played_text,
                        role="assistant",
                    )
                )

        # when user_question is empty, it's due to a false positive interruption
        # when this happens, we'd want to add a continue marker to the chat context.
        # while some LLMs could deal with empty content during an inference request
        # others would fail.
        user_input = handle.user_question
        if not user_input.strip():
            user_input = "<continue>"
        copied_ctx.messages.append(ChatMessage.create(text=user_input, role="user"))

        tk = SpeechDataContextVar.set(SpeechData(sequence_id=handle.id))
        try:
            llm_stream = self._opts.before_llm_cb(self, copied_ctx)
            if asyncio.iscoroutine(llm_stream):
                llm_stream = await llm_stream

            if llm_stream is False:
                # user chose not to synthesize an answer, so we do not want to
                # leave the same question in chat context. otherwise it would be
                # unintentionally committed when the next set of speech comes in.
                if len(self._transcribed_text) >= len(handle.user_question):
                    self._transcribed_text = self._transcribed_text[
                        len(handle.user_question) :
                    ]
                log.verbose(
                    "Cancelling handle due to user not wanting to synthesize an answer"
                )
                handle.cancel()
                return

            # fallback to default impl if no custom/user stream is returned
            if not isinstance(llm_stream, LLMStream):
                llm_stream = _default_before_llm_cb(self, chat_ctx=copied_ctx)

            if handle.interrupted:
                log.verbose("Returning early due to handle.interrupted")
                return

            synthesis_handle = self._synthesize_agent_speech(handle.id, llm_stream)
            handle.initialize(source=llm_stream, synthesis_handle=synthesis_handle)
        finally:
            SpeechDataContextVar.reset(tk)

    def _commit_user_question(self) -> None:
        speech_handle = self._playing_speech
        user_question = speech_handle.user_question

        user_msg = ChatMessage.create(text=user_question, role="user")
        self._chat_ctx.messages.append(user_msg)
        self.emit("user_speech_committed", user_msg)

        self._transcribed_text = self._transcribed_text[len(user_question) :]
        speech_handle.mark_user_committed()

    def _get_spoken_text_at_time(self, elapsed_ms: float) -> str:
        """
        Get the text that has been spoken up to a specific elapsed time.

        Args:
            elapsed_ms: Time elapsed since playout started in milliseconds

        Returns:
            Tuple containing:
            - The text spoken up to that time
            - The index of the last spoken character
        """
        spoken_chars = []
        duration_sum = 0

        log.verbose(
            f"inside _get_spoken_text_at_time, playout_buffer: {AppConfig().playout_buffer}",
            categories=["pipeline"],
        )
        # Find the last character that should have been spoken
        for i, char in enumerate(AppConfig().playout_buffer):
            log.verbose(f"char: {char}", categories=["pipeline"])
            duration = AppConfig().char_timings[i]
            duration_sum += duration
            # If this character's end time is beyond our elapsed time, we've found our cutoff
            if duration_sum > elapsed_ms:
                break

            spoken_chars.append(char)

        return "".join(spoken_chars)

    def _get_current_spoken_text(self) -> str:
        """
        Get the text that has been spoken so far based on current time.

        Returns:
            Tuple containing:
            - The text spoken up to current time
            - The index of the last spoken character
        """
        app_config = AppConfig()
        if not app_config.playout_start_time:
            log.pipeline("No playout start time found")
            return ""

        log.pipeline(f"playout_start_time: {app_config.playout_start_time}")
        elapsed_ms = (time.time() - app_config.playout_start_time) * 1000
        log.pipeline(f"elapsed_ms: {elapsed_ms}")

        log.pipeline(
            f"Inside _get_current_spoken_text, about to return: {self._get_spoken_text_at_time(elapsed_ms)}"
        )
        return self._get_spoken_text_at_time(elapsed_ms)

    async def _play_speech(self, speech_handle: SpeechHandle) -> None:
        await self._agent_publication.wait_for_subscription()

        fnc_done_fut = asyncio.Future[None]()
        playing_lock = asyncio.Lock()
        nested_speech_played = asyncio.Event()

        async def _play_nested_speech():
            speech_handle._nested_speech_done_fut = asyncio.Future[None]()
            while not speech_handle.nested_speech_done:
                nesting_changed = asyncio.create_task(
                    speech_handle.nested_speech_changed.wait()
                )
                nesting_done_fut: asyncio.Future = speech_handle._nested_speech_done_fut
                await asyncio.wait(
                    [nesting_changed, nesting_done_fut, fnc_done_fut],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not nesting_changed.done():
                    nesting_changed.cancel()

                while speech_handle.nested_speech_handles:
                    nested_speech_played.clear()
                    speech = speech_handle.nested_speech_handles[0]
                    if speech_handle.nested_speech_done:
                        # in case tool speech is added after nested speech done
                        speech.cancel(cancel_nested=True)
                        speech_handle.nested_speech_handles.pop(0)
                        continue

                    async with playing_lock:
                        self._playing_speech = speech
                        await self._play_speech(speech)
                        speech_handle.nested_speech_handles.pop(0)
                        self._playing_speech = speech_handle

                nested_speech_played.set()
                speech_handle.nested_speech_changed.clear()
                # break if the function calls task is done
                if fnc_done_fut.done():
                    speech_handle.mark_nested_speech_done()

        nested_speech_task = asyncio.create_task(_play_nested_speech())

        async def _stop_nesting_speech():
            fnc_done_fut.set_result(None)
            await nested_speech_task

        try:
            await speech_handle.wait_for_initialization()
        except asyncio.CancelledError:
            await _stop_nesting_speech()
            return

        # wait for all pre-added nested speech to be played
        while speech_handle.nested_speech_handles:
            await nested_speech_played.wait()

        await playing_lock.acquire()
        synthesis_handle = speech_handle.synthesis_handle
        if synthesis_handle.interrupted:
            playing_lock.release()
            await _stop_nesting_speech()
            return

        user_question = speech_handle.user_question

        play_handle = synthesis_handle.play()
        join_fut = play_handle.join()

        def _commit_user_question_if_needed() -> None:
            if (
                not user_question
                or synthesis_handle.interrupted
                or speech_handle.user_committed
            ):
                return

            is_using_tools = isinstance(speech_handle.source, LLMStream) and len(
                speech_handle.source.function_calls
            )

            # make sure at least some speech was played before committing the user message
            # since we try to validate as fast as possible it is possible the agent gets interrupted
            # really quickly (barely audible), we don't want to mark this question as "answered".
            spoken_text = speech_handle.synthesis_handle.tts_forwarder.played_text
            if (
                speech_handle.allow_interruptions
                and not is_using_tools
                and (
                    not spoken_text  # Don't commit if nothing was actually said
                    or not spoken_text.strip()  # Don't commit if only whitespace
                    or play_handle.time_played < self.MIN_TIME_PLAYED_FOR_COMMIT
                    and not join_fut.done()
                )
            ):
                return

            user_msg = ChatMessage.create(text=user_question, role="user")
            self._chat_ctx.messages.append(user_msg)
            self.emit("user_speech_committed", user_msg)

            self._transcribed_text = self._transcribed_text[len(user_question) :]
            speech_handle.mark_user_committed()

        # wait for the play_handle to finish and check every 1s if the user question should be committed
        _commit_user_question_if_needed()

        while not join_fut.done():
            await asyncio.wait(
                [join_fut], return_when=asyncio.FIRST_COMPLETED, timeout=0.2
            )

            _commit_user_question_if_needed()

            if speech_handle.interrupted:
                break

        _commit_user_question_if_needed()

        collected_text = speech_handle.synthesis_handle.tts_forwarder.played_text
        interrupted = speech_handle.interrupted
        is_using_tools = isinstance(speech_handle.source, LLMStream) and len(
            speech_handle.source.function_calls
        )

        log.pipeline(
            f"collected_text: {collected_text}\ninterrupted: {interrupted}\nis_using_tools: {is_using_tools}"
        )

        # add tool calls and text message to the chat context
        message_id_committed: str | None = None
        log.pipeline(f"add_to_chat_ctx: {speech_handle.add_to_chat_ctx}")
        log.pipeline(f"not user_question: {not user_question}")
        log.pipeline(f"user_committed: {speech_handle.user_committed}")
        if speech_handle.add_to_chat_ctx and (
            not user_question or speech_handle.user_committed
        ):
            if speech_handle.extra_tools_messages:
                if speech_handle.fnc_text_message_id is not None:
                    # there is a message alongside the function calls
                    msgs = self._chat_ctx.messages
                    if msgs and msgs[-1].id == speech_handle.fnc_text_message_id:
                        # replace it with the tool call message if it's the last in the ctx
                        msgs.pop()
                    elif speech_handle.extra_tools_messages[0].tool_calls:
                        # remove the content of the tool call message
                        speech_handle.extra_tools_messages[0].content = ""
                self._chat_ctx.messages.extend(speech_handle.extra_tools_messages)

            if collected_text:
                log.pipeline(
                    f"collected_text: {collected_text}, last_llm_message: {AppConfig().last_llm_message}"
                )
                if interrupted:
                    log.pipeline("interrupted=True")
                    # if collected_text in (
                    #     AppConfig().call_metadata.get("agent_interrupted_text") or ""
                    # ):
                    AppConfig().agent_interrupted = True
                    # app_config_text = AppConfig().call_metadata.get(
                    #     "agent_interrupted_text"
                    # )
                    current_text = self._get_current_spoken_text() or ""
                    log.pipeline(f"current_text: {current_text}")
                    if (
                        collected_text.replace(" ", "").lower()
                        in current_text.replace(" ", "").lower()
                        and collected_text.replace(" ", "").lower()
                        != current_text.replace(" ", "").lower()
                    ):
                        log.pipeline(
                            f"Replacing interrupted text=`{collected_text}` with `{current_text}`"
                        )
                        collected_text = current_text + "..."
                        log.pipeline(
                            f"inside interrupt case, about to clear playout_buffer: {AppConfig().playout_buffer}"
                        )
                        AppConfig().playout_buffer = ""
                        AppConfig().char_timings = []
                else:
                    log.pipeline("interrupted=False")
                    if (
                        collected_text.replace(" ", "").lower()
                        == AppConfig().last_llm_message.replace(" ", "").lower()
                    ):
                        log.pipeline(
                            f"inside collected_text == AppConfig().last_llm_message, about to clear playout_buffer: {AppConfig().playout_buffer}"
                        )
                        AppConfig().playout_buffer = ""
                        AppConfig().char_timings = []

                msg = ChatMessage.create(text=collected_text, role="assistant")
                self._chat_ctx.messages.append(msg)
                message_id_committed = msg.id
                speech_handle.mark_speech_committed()

                if interrupted:
                    self.emit("agent_speech_interrupted", msg)
                else:
                    self.emit("agent_speech_committed", msg)

                log.debug(
                    "committed agent speech",
                    extra={
                        "agent_transcript": collected_text,
                        "interrupted": interrupted,
                        "speech_id": speech_handle.id,
                    },
                )

        AppConfig().call_metadata.update({"updated_chat_ctx_with_collected_text": True})
        playing_lock.release()

        @utils.log_exceptions(logger=log)
        async def _execute_function_calls() -> None:
            nonlocal interrupted, collected_text

            # if the answer is using tools, execute the functions and automatically generate
            # a response to the user question from the returned values
            if not is_using_tools or interrupted:
                return

            if speech_handle.fnc_nested_depth >= self._opts.max_nested_fnc_calls:
                log.warning(
                    "max function calls nested depth reached",
                    extra={
                        "speech_id": speech_handle.id,
                        "fnc_nested_depth": speech_handle.fnc_nested_depth,
                    },
                )
                return

            assert isinstance(speech_handle.source, LLMStream)
            assert (
                not user_question or speech_handle.user_committed
            ), "user speech should have been committed before using tools"

            llm_stream = speech_handle.source

            # execute functions
            call_ctx = AgentCallContext(self, llm_stream)
            tk = _CallContextVar.set(call_ctx)

            new_function_calls = llm_stream.function_calls

            self.emit("function_calls_collected", new_function_calls)

            called_fncs = []
            for fnc in new_function_calls:
                called_fnc = fnc.execute()
                called_fncs.append(called_fnc)
                log.debug(
                    "executing ai function",
                    extra={
                        "function": fnc.function_info.name,
                        "speech_id": speech_handle.id,
                    },
                )
                try:
                    await called_fnc.task
                except Exception as e:
                    log.exception(
                        "error executing ai function",
                        extra={
                            "function": fnc.function_info.name,
                            "speech_id": speech_handle.id,
                        },
                        exc_info=e,
                    )

            tool_calls_info = []
            tool_calls_results = []

            for called_fnc in called_fncs:
                # ignore the function calls that returns None
                if called_fnc.result is None and called_fnc.exception is None:
                    continue

                tool_calls_info.append(called_fnc.call_info)
                tool_calls_results.append(
                    ChatMessage.create_tool_from_called_function(called_fnc)
                )

            if not tool_calls_info:
                return

            # create a nested speech handle
            extra_tools_messages = [
                ChatMessage.create_tool_calls(tool_calls_info, text=collected_text)
            ]
            extra_tools_messages.extend(tool_calls_results)

            new_speech_handle = SpeechHandle.create_tool_speech(
                allow_interruptions=speech_handle.allow_interruptions,
                add_to_chat_ctx=speech_handle.add_to_chat_ctx,
                extra_tools_messages=extra_tools_messages,
                fnc_nested_depth=speech_handle.fnc_nested_depth + 1,
                fnc_text_message_id=message_id_committed,
            )

            # synthesize the tool speech with the chat ctx from llm_stream
            chat_ctx = call_ctx.chat_ctx.copy()
            chat_ctx.messages.extend(extra_tools_messages)
            chat_ctx.messages.extend(call_ctx.extra_chat_messages)
            fnc_ctx = self.fnc_ctx
            if (
                fnc_ctx
                and new_speech_handle.fnc_nested_depth
                >= self._opts.max_nested_fnc_calls
                and not self._llm.capabilities.requires_persistent_functions
            ):
                if len(fnc_ctx.ai_functions) > 1:
                    log.pipeline(
                        "max function calls nested depth reached, dropping function context. increase max_nested_fnc_calls to enable additional nesting.",
                        extra={
                            "speech_id": speech_handle.id,
                            "fnc_nested_depth": speech_handle.fnc_nested_depth,
                        },
                    )
                fnc_ctx = None

            answer_llm_stream = self._llm.chat(
                chat_ctx=chat_ctx,
                fnc_ctx=fnc_ctx,
            )

            synthesis_handle = self._synthesize_agent_speech(
                new_speech_handle.id, answer_llm_stream
            )
            new_speech_handle.initialize(
                source=answer_llm_stream, synthesis_handle=synthesis_handle
            )
            speech_handle.add_nested_speech(new_speech_handle)

            self.emit("function_calls_finished", called_fncs)
            _CallContextVar.reset(tk)

        if not is_using_tools:
            # skip the function calls execution
            await _stop_nesting_speech()
            speech_handle._set_done()
            return

        fnc_task = asyncio.create_task(_execute_function_calls())
        fnc_task.add_done_callback(lambda _: fnc_done_fut.set_result(None))
        await nested_speech_task

        if not fnc_task.done():
            log.debug(
                "cancelling function calls task", extra={"speech_id": speech_handle.id}
            )
            fnc_task.cancel()

        # mark the speech as done
        speech_handle._set_done()

    def _synthesize_agent_speech(
        self,
        speech_id: str,
        source: str | LLMStream | AsyncIterable[str],
    ) -> SynthesisHandle:
        assert (
            self._agent_output is not None
        ), "agent output should be initialized when ready"

        tk = SpeechDataContextVar.set(SpeechData(speech_id))

        async def _llm_stream_to_str_generator(
            stream: LLMStream,
        ) -> AsyncGenerator[str]:
            try:
                async for chunk in stream:
                    if not chunk.choices:
                        continue

                    content = chunk.choices[0].delta.content
                    if content is None:
                        continue

                    yield content
            finally:
                await stream.aclose()

        if isinstance(source, LLMStream):
            source = _llm_stream_to_str_generator(source)

        og_source = source
        transcript_source = source
        if isinstance(og_source, AsyncIterable):
            og_source, transcript_source = utils.aio.itertools.tee(og_source, 2)

        tts_source = self._opts.before_tts_cb(self, og_source)
        if tts_source is None:
            raise ValueError("before_tts_cb must return str or AsyncIterable[str]")

        try:
            return self._agent_output.synthesize(
                speech_id=speech_id,
                tts_source=tts_source,
                transcript_source=transcript_source,
                transcription=self._opts.transcription.agent_transcription,
                transcription_speed=self._opts.transcription.agent_transcription_speed,
                sentence_tokenizer=self._opts.transcription.sentence_tokenizer,
                word_tokenizer=self._opts.transcription.word_tokenizer,
                hyphenate_word=self._opts.transcription.hyphenate_word,
            )
        finally:
            SpeechDataContextVar.reset(tk)

    def _validate_reply_if_possible(self) -> None:
        """Check if the new agent speech should be played"""

        log.pipeline(f"Validating reply - {self._transcribed_text}")

        if not self._transcribed_text.strip():
            log.pipeline("Transcribed text is empty, skipping validation")
            return

        if "potential_user_question" not in AppConfig().call_metadata:
            AppConfig().call_metadata["potential_user_question"] = ""
        AppConfig().call_metadata["potential_user_question"] += self._transcribed_text

        if AppConfig().get_call_metadata().get("is_payment_processing"):
            log.pipeline(
                f"Skipping validation because payment is processing - {self._transcribed_text}"
            )
            self._transcribed_text = ""
            return

        if not AppConfig().call_metadata.get("initial_greeting_delivered"):
            log.pipeline(
                f"Skipping validation because the initial greeting was not delivered - {self._transcribed_text}"
            )
            self._transcribed_text = ""
            return

        if AppConfig().get_call_metadata().get("is_speaking_uninterruptible_message"):
            log.pipeline(
                f"_validate_reply_if_possible: Skipping validation because the agent is speaking an uninterruptible message - {self._transcribed_text}"
            )
            self._transcribed_text = ""
            return

        if self._playing_speech and not self._playing_speech.interrupted:
            should_ignore_input = False
            if not self._playing_speech.allow_interruptions:
                should_ignore_input = True
                log.debug(
                    "skipping validation, agent is speaking and does not allow interruptions",
                    extra={"speech_id": self._playing_speech.id},
                )
            elif not self._should_interrupt():
                should_ignore_input = True
                log.debug(
                    "interrupt threshold is not met",
                    extra={"speech_id": self._playing_speech.id},
                )

            if should_ignore_input:
                self._transcribed_text = ""
                return

        if self._pending_agent_reply is None:
            if self._opts.preemptive_synthesis:
                return

            # as long as we don't have a pending reply, we need to synthesize it
            # in order to keep the conversation flowing.
            # transcript could be empty at this moment, if the user interrupted the agent
            # but did not generate any transcribed text.
            self._synthesize_agent_reply()

        assert self._pending_agent_reply is not None

        # due to timing, we could end up with two pushed agent replies inside the speech queue.
        # so make sure we directly interrupt every reply when validating a new one
        for speech in self._speech_q:
            if not speech.is_reply:
                continue

            if speech.allow_interruptions:
                speech.interrupt()

        log.debug(
            "validated agent reply",
            extra={
                "speech_id": self._pending_agent_reply.id,
                "text": self._transcribed_text,
            },
        )

        if self._last_speech_time is not None:
            time_since_last_speech = time.perf_counter() - self._last_speech_time
            transcription_delay = max(
                (self._last_final_transcript_time or 0) - self._last_speech_time, 0
            )

            eou_metrics = metrics.PipelineEOUMetrics(
                timestamp=time.time(),
                sequence_id=self._pending_agent_reply.id,
                end_of_utterance_delay=time_since_last_speech,
                transcription_delay=transcription_delay,
            )
            self.emit("metrics_collected", eou_metrics)

        self._add_speech_for_playout(self._pending_agent_reply)
        self._pending_agent_reply = None
        self._transcribed_interim_text = ""
        # self._transcribed_text is reset after MIN_TIME_PLAYED_FOR_COMMIT, see self._play_speech

    def _interrupt_if_possible(self) -> None:
        """Check whether the current assistant speech should be interrupted"""
        if self._should_interrupt():
            self.interrupt()

    def _should_interrupt(self) -> bool:
        if self._playing_speech is None:
            log.pipeline("No playing speech, skipping interrupt")
            return False

        if "potential_user_question" not in AppConfig().call_metadata:
            AppConfig().call_metadata["potential_user_question"] = ""
        AppConfig().call_metadata["potential_user_question"] += self._transcribed_text

        if AppConfig().get_call_metadata().get("is_speaking_uninterruptible_message"):
            log.pipeline(
                f"_should_interrupt: Skipping validation because the agent is speaking an uninterruptible message - {self._transcribed_text}"
            )
            self._transcribed_text = ""
            return False

        if (
            not self._playing_speech.allow_interruptions
            or self._playing_speech.interrupted
        ):
            log.pipeline(
                "Skipping interrupt because the speech is not allowed to be interrupted or is already interrupted"
            )
            return False

        # We should only have the check for min words when we are actively speaking, if not we should interrupt the agent process
        # Also if we hit this block then it is 100% a human self interruption
        try:
            spoken_text = (
                self._playing_speech.synthesis_handle.tts_forwarder.played_text
            )
            if spoken_text is None:
                log.pipeline(
                    "Interrupting the speech because the agent is not actively speaking"
                )
                AppConfig().is_human_interrupted = True
                return True
        except:
            log.pipeline(
                "Interrupting the speech because the agent is not actively speaking"
            )
            AppConfig().is_human_interrupted = True
            return True

        if self._opts.int_min_words != 0:
            text = (
                self._transcribed_interim_text
                if self._transcribed_interim_text is not None
                and len(self._transcribed_interim_text)
                > len(self._transcribed_text or "")
                else self._transcribed_text or ""
            )
            interim_words = self._opts.transcription.word_tokenizer.tokenize(text=text)
            if len(interim_words) < self._opts.int_min_words:
                log.pipeline(
                    "Skipping interrupt because the number of interim words is less than the minimum number of words"
                )
                return False

        log.pipeline("Interrupting the speech because the interrupt threshold is met")
        return True

    def _add_speech_for_playout(self, speech_handle: SpeechHandle) -> None:
        self._speech_q.append(speech_handle)
        self._speech_q_changed.set()


class _DeferredReplyValidation:
    """This class is used to try to find the best time to validate the agent reply."""

    # if the STT gives us punctuation, we can try validate the reply faster.
    PUNCTUATION = ".!?"
    PUNCTUATION_REDUCE_FACTOR = 0.75

    FINAL_TRANSCRIPT_TIMEOUT = 5

    def __init__(
        self,
        validate_fnc: Callable[[], None],
        min_endpointing_delay: float,
        max_endpointing_delay: float,
        turn_detector: _TurnDetector | None,
        agent: VoicePipelineAgent,
    ) -> None:
        self._turn_detector = turn_detector
        self._validate_fnc = validate_fnc
        self._validating_task: asyncio.Task | None = None
        self._last_final_transcript: str = ""
        self._last_language: str | None = None
        self._last_recv_start_of_speech_time: float = 0.0
        self._last_recv_end_of_speech_time: float = 0.0
        self._last_recv_transcript_time: float = 0.0
        self._speaking = False

        self._agent = agent
        self._end_of_speech_delay = min_endpointing_delay
        self._max_endpointing_delay = max_endpointing_delay

    @property
    def validating(self) -> bool:
        return self._validating_task is not None and not self._validating_task.done()

    def _compute_delay(self) -> float | None:
        """Computes the amount of time to wait before validating the agent reply.

        This function should be called after the agent has received final transcript, or after VAD
        """
        # never interrupt the user while they are speaking
        if self._speaking:
            return None

        # if STT doesn't give us the final transcript after end of speech, we'll still validate the reply
        # to prevent the agent from getting "stuck"
        # in this case, the agent will not have final transcript, so it'll trigger the user input with empty
        if not self._last_final_transcript:
            return self.FINAL_TRANSCRIPT_TIMEOUT

        delay = self._end_of_speech_delay
        if self._end_with_punctuation():
            delay = delay * self.PUNCTUATION_REDUCE_FACTOR

        # the delay should be computed from end of earlier timestamp, that's the true end of user speech
        end_of_speech_time = self._last_recv_end_of_speech_time
        if (
            self._last_recv_transcript_time > 0
            and self._last_recv_transcript_time > self._last_recv_start_of_speech_time
            and self._last_recv_transcript_time < end_of_speech_time
        ):
            end_of_speech_time = self._last_recv_transcript_time

        elapsed_time = time.perf_counter() - end_of_speech_time
        if elapsed_time < delay:
            delay -= elapsed_time
        else:
            delay = 0
        return delay

    def on_human_final_transcript(self, transcript: str, language: str | None) -> None:
        self._last_final_transcript += " " + transcript.strip()  # type: ignore
        self._last_language = language
        self._last_recv_transcript_time = time.perf_counter()

        delay = self._compute_delay()
        if delay is not None:
            self._run(delay)

    def on_human_start_of_speech(self, ev: vad.VADEvent) -> None:
        self._speaking = True
        self._last_recv_start_of_speech_time = time.perf_counter()
        if self.validating:
            assert self._validating_task is not None
            self._validating_task.cancel()

    def on_human_end_of_speech(self, ev: vad.VADEvent) -> None:
        self._speaking = False
        self._last_recv_end_of_speech_time = time.perf_counter()

        delay = self._compute_delay()
        if delay is not None:
            self._run(delay)

    async def aclose(self) -> None:
        if self._validating_task is not None:
            await utils.aio.gracefully_cancel(self._validating_task)

    def _end_with_punctuation(self) -> bool:
        return (
            len(self._last_final_transcript) > 0
            and self._last_final_transcript[-1] in self.PUNCTUATION
        )

    def _reset_states(self) -> None:
        self._last_final_transcript = ""
        self._last_recv_end_of_speech_time = 0.0
        self._last_recv_transcript_time = 0.0

    def _run(self, delay: float) -> None:
        @utils.log_exceptions(logger=log)
        async def _run_task(chat_ctx: ChatContext, delay: float) -> None:
            use_turn_detector = self._last_final_transcript and not self._speaking
            if (
                use_turn_detector
                and self._turn_detector is not None
                and self._turn_detector.supports_language(self._last_language)
            ):
                start_time = time.perf_counter()
                try:
                    eot_prob = await self._turn_detector.predict_end_of_turn(chat_ctx)
                    unlikely_threshold = self._turn_detector.unlikely_threshold()
                    elasped = time.perf_counter() - start_time
                    if eot_prob < unlikely_threshold:
                        delay = self._max_endpointing_delay
                    delay = max(0, delay - elasped)
                except (TimeoutError, AssertionError):
                    pass  # inference process is unresponsive

            await asyncio.sleep(delay)

            self._reset_states()
            self._validate_fnc()

        if self._validating_task is not None:
            self._validating_task.cancel()

        detect_ctx = self._agent._chat_ctx.copy()
        detect_ctx.messages.append(
            ChatMessage.create(text=self._agent._transcribed_text, role="user")
        )
        self._validating_task = asyncio.create_task(_run_task(detect_ctx, delay))
