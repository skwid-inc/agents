# Copyright 2023 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any, List, Optional

import aiohttp
import numpy as np
import soundfile as sf
from app_config import AppConfig
from filler_phrases import get_wav_if_available
from livekit import rtc
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from scipy import signal

from .log import logger
from .models import TTSEncoding, TTSModels

_DefaultEncoding: TTSEncoding = "mp3_44100"


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_44100
    return int(split[1])


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: float | None = None  # [0.0 - 1.0]
    speed: float | None = 1.0  # [0.8 - 1.2]
    use_speaker_boost: bool | None = False


@dataclass
class Voice:
    id: str
    name: str
    category: str
    settings: VoiceSettings | None = None


DEFAULT_VOICE = Voice(
    id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    category="premade",
    settings=VoiceSettings(
        stability=0.71,
        speed=1.0,
        similarity_boost=0.5,
        style=0.0,
        use_speaker_boost=True,
    ),
)

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
WS_INACTIVITY_TIMEOUT = 300
NUM_CHANNELS = 1


@dataclass
class _TTSOptions:
    api_key: str
    voice: Voice
    model: TTSModels | str
    language: str | None
    base_url: str
    encoding: TTSEncoding
    sample_rate: int
    streaming_latency: int
    word_tokenizer: tokenize.WordTokenizer
    chunk_length_schedule: list[int]
    enable_ssml_parsing: bool
    inactivity_timeout: int


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model: TTSModels | str = "eleven_flash_v2_5",
        api_key: str | None = None,
        base_url: str | None = None,
        streaming_latency: int = 3,
        inactivity_timeout: int = WS_INACTIVITY_TIMEOUT,
        word_tokenizer: Optional[tokenize.WordTokenizer] = None,
        enable_ssml_parsing: bool = False,
        chunk_length_schedule: list[int] = [80, 120, 200, 260],  # range is [50, 500]
        http_session: aiohttp.ClientSession | None = None,
        # deprecated
        model_id: TTSModels | str | None = None,
        language: str | None = None,
    ) -> None:
        """
        Create a new instance of ElevenLabs TTS.

        Args:
            voice (Voice): Voice configuration. Defaults to `DEFAULT_VOICE`.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            api_key (str | None): ElevenLabs API key. Can be set via argument or `ELEVEN_API_KEY` environment variable.
            base_url (str | None): Custom base URL for the API. Optional.
            streaming_latency (int): Latency in seconds for streaming. Defaults to 3.
            inactivity_timeout (int): Inactivity timeout in seconds for the websocket connection. Defaults to 300.
            word_tokenizer (tokenize.WordTokenizer): Tokenizer for processing text. Defaults to basic WordTokenizer.
            enable_ssml_parsing (bool): Enable SSML parsing for input text. Defaults to False.
            chunk_length_schedule (list[int]): Schedule for chunk lengths, ranging from 50 to 500. Defaults to [80, 120, 200, 260].
            http_session (aiohttp.ClientSession | None): Custom HTTP session for API requests. Optional.
            language (str | None): Language code for the TTS model, as of 10/24/24 only valid for "eleven_turbo_v2_5". Optional.
        """

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
            ),
            sample_rate=_sample_rate_from_format(_DefaultEncoding),
            num_channels=1,
        )

        if model_id is not None:
            logger.warning(
                "model_id is deprecated and will be removed in 1.5.0, use model instead",
            )
            model = model_id

        api_key = api_key or os.environ.get("ELEVEN_API_KEY")
        if not api_key:
            raise ValueError(
                "ElevenLabs API key is required, either as argument or set ELEVEN_API_KEY environmental variable"
            )

        if word_tokenizer is None:
            word_tokenizer = tokenize.basic.WordTokenizer(
                ignore_punctuation=False  # punctuation can help for intonation
            )

        self._opts = _TTSOptions(
            voice=voice,
            model=model,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            encoding=_DefaultEncoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            language=language,
            inactivity_timeout=inactivity_timeout,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=inactivity_timeout,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        return await asyncio.wait_for(
            session.ws_connect(
                _stream_url(self._opts),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
            ),
            self._conn_options.timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    async def list_voices(self) -> List[Voice]:
        async with self._ensure_session().get(
            f"{self._opts.base_url}/voices",
            headers={AUTHORIZATION_HEADER: self._opts.api_key},
        ) as resp:
            return _dict_to_voices_list(await resp.json())

    def update_options(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model: TTSModels | str = "eleven_turbo_v2_5",
        language: str | None = None,
    ) -> None:
        """
        Args:
            voice (Voice): Voice configuration. Defaults to `DEFAULT_VOICE`.
            model (TTSModels | str): TTS model to use. Defaults to "eleven_turbo_v2_5".
            language (str | None): Language code for the TTS model. Optional.
        """
        self._opts.model = model or self._opts.model
        self._opts.voice = voice or self._opts.voice
        self._opts.language = language or self._opts.language

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> "ChunkedStream":
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

    async def _play_presynthesized_audio(
        self, wav_path: str, event_ch, input_text: str
    ) -> None:
        logger.info(f"Playing Presynthesized Audio: {input_text}")
        request_id = utils.shortuuid()

        # Read the WAV file
        wav_path = get_wav_if_available(input_text)
        audio_array, file_sample_rate = sf.read(str(wav_path), dtype="int16")

        logger.info(f"File sample rate: {file_sample_rate}")
        logger.info(
            f"WAV file channels: {1 if audio_array.ndim == 1 else audio_array.shape[1]}"
        )
        logger.info(f"Target sample rate: {self._opts.sample_rate}")

        # Convert stereo to mono if needed
        if audio_array.ndim == 2 and audio_array.shape[1] == 2:
            logger.info("Converting stereo to mono")
            audio_array = audio_array.mean(axis=1)

        # Resample if needed
        if file_sample_rate != self._opts.sample_rate:
            logger.info(
                f"Resampling from {file_sample_rate} to {self._opts.sample_rate}"
            )
            # Calculate number of samples for the target sample rate
            num_samples = int(
                len(audio_array) * self._opts.sample_rate / file_sample_rate
            )
            audio_array = signal.resample(audio_array, num_samples)

        # Process audio in chunks
        samples_per_channel = 960  # Standard frame size for audio processing
        for i in range(0, len(audio_array), samples_per_channel):
            chunk = audio_array[i : i + samples_per_channel]

            # Pad the last chunk if needed
            if len(chunk) < samples_per_channel:
                chunk = np.pad(chunk, (0, samples_per_channel - len(chunk)))

            # Apply soft clipping and ensure int16 format
            chunk = np.tanh(chunk / 32768.0) * 32768.0
            chunk = np.round(chunk).astype(np.int16)

            # Create and send the audio frame with matching parameters
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self._opts.sample_rate,  # Use the TTS instance's sample rate
                samples_per_channel=samples_per_channel,
                num_channels=NUM_CHANNELS,  # Make sure this matches the expected number of channels
            )
            event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    frame=frame,
                )
            )

        logger.info(f"Sent Presynthesized Audio to event channel")
        return

    def stream(
        self, *, conn_options: Optional[APIConnectOptions] = None
    ) -> "SynthesizeStream":
        stream = SynthesizeStream(tts=self, pool=self._pool, opts=self._opts)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()
        await super().aclose()


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self,
        *,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        conn_options: Optional[APIConnectOptions] = None,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        voice_settings = (
            _strip_nones(dataclasses.asdict(self._opts.voice.settings))
            if self._opts.voice.settings
            else None
        )
        data = {
            "text": self._input_text,
            "model_id": self._opts.model,
            "voice_settings": voice_settings,
        }

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
        )

        decode_task: asyncio.Task | None = None
        try:
            async with self._session.post(
                _synthesize_url(self._opts),
                headers={AUTHORIZATION_HEADER: self._opts.api_key},
                json=data,
            ) as resp:
                if not resp.content_type.startswith("audio/"):
                    content = await resp.text()
                    logger.error("11labs returned non-audio data: %s", content)
                    return

                async def _decode_loop():
                    try:
                        async for bytes_data, _ in resp.content.iter_chunks():
                            decoder.push(bytes_data)
                    finally:
                        decoder.end_input()

                decode_task = asyncio.create_task(_decode_loop())
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=None,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()


class SynthesizeStream(tts.SynthesizeStream):
    """Streamed API using websockets"""

    def __init__(
        self,
        *,
        tts: TTS,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
        opts: _TTSOptions,
    ):
        super().__init__(tts=tts)
        self._opts, self._pool = opts, pool

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    # Check for filler phrases
                    filler_phrase_wav = get_wav_if_available(input)
                    if filler_phrase_wav:
                        logger.info(f"Playing presynthesized audio for: {input}")
                        await self._tts._play_presynthesized_audio(
                            filler_phrase_wav, self._event_ch, input
                        )
                        continue
                    if word_stream is None:
                        # new segment (after flush for e.g)
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)

                    word_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if word_stream is not None:
                        word_stream.end_input()
                    word_stream = None
            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _process_segments():
            async for word_stream in self._segments_ch:
                await self._run_ws(word_stream, request_id)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        word_stream: tokenize.WordStream,
        request_id: str,
    ) -> None:
        async with self._pool.connection() as ws_conn:
            segment_id = utils.shortuuid()
            expected_text = ""  # accumulate all tokens sent

            decoder = utils.codecs.AudioStreamDecoder(
                sample_rate=self._opts.sample_rate,
                num_channels=1,
            )

            # 11labs protocol expects the first message to be an "init msg"
            init_pkt = dict(
                text=" ",
                voice_settings=(
                    _strip_nones(dataclasses.asdict(self._opts.voice.settings))
                    if self._opts.voice.settings
                    else None
                ),
                generation_config=dict(
                    chunk_length_schedule=self._opts.chunk_length_schedule
                ),
            )
            await ws_conn.send_str(json.dumps(init_pkt))

            @utils.log_exceptions(logger=logger)
            async def send_task():
                nonlocal expected_text
                xml_content = []
                async for data in word_stream:
                    text = data.token
                    expected_text += text
                    # send the xml phoneme in one go
                    if (
                        self._opts.enable_ssml_parsing
                        and data.token.startswith("<phoneme")
                        or xml_content
                    ):
                        xml_content.append(text)
                        if text.find("</phoneme>") > -1:
                            text = self._opts.word_tokenizer.format_words(xml_content)
                            xml_content = []
                        else:
                            continue

                    data_pkt = dict(
                        text=f"{text.strip()} "
                    )  # must always end with a space

                    # if any(text.strip().endswith(p) for p in [".", "?"]):
                    #     data_pkt = dict(text=text.strip())
                    self._mark_started()
                    logger.info(f"data_pkt: {data_pkt}")
                    await ws_conn.send_str(json.dumps(data_pkt))
                    if any(text.strip().endswith(p) for p in [".", "?", "!"]):
                        if not AppConfig().call_metadata.get(
                            "first_sentence_synthesis_start_time"
                        ):
                            AppConfig().call_metadata[
                                "first_sentence_synthesis_start_time"
                            ] = time.time()

                        logger.info("Sending flush due to sentence-ending punctuation")
                        await ws_conn.send_str(json.dumps({"flush": True}))
                if xml_content:
                    logger.warning("11labs stream ended with incomplete xml content")
                await ws_conn.send_str(json.dumps({"flush": True}))

            # consumes from decoder and generates events
            @utils.log_exceptions(logger=logger)
            async def generate_task():
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                    segment_id=segment_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()

            # receives from ws and decodes audio
            @utils.log_exceptions(logger=logger)
            async def recv_task():
                nonlocal expected_text
                received_text = ""

                while True:
                    msg = await ws_conn.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIStatusError(
                            "11labs connection closed unexpectedly, not all tokens have been consumed",
                            request_id=request_id,
                        )

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning("unexpected 11labs message type %s", msg.type)
                        continue

                    data = json.loads(msg.data)
                    if data.get("audio"):
                        b64data = base64.b64decode(data["audio"])
                        decoder.push(b64data)

                        if alignment := data.get("normalizedAlignment"):
                            received_text += "".join(
                                alignment.get("chars", [])
                            ).replace(" ", "")
                            expected_text_without_spaces = expected_text.replace(
                                " ", ""
                            )
                            logger.info(f"received_text: {received_text}")
                            logger.info(f"expected_text: {expected_text}")
                            logger.info(
                                f"expected_text_without_spaces: {expected_text_without_spaces}"
                            )
                            if received_text == expected_text_without_spaces:
                                decoder.end_input()
                                break
                    elif data.get("error"):
                        raise APIStatusError(
                            message=data["error"],
                            status_code=500,
                            request_id=request_id,
                            body=None,
                        )
                    else:
                        raise APIStatusError(
                            message=f"unexpected 11labs message {data}",
                            status_code=500,
                            request_id=request_id,
                            body=None,
                        )

            tasks = [
                asyncio.create_task(send_task()),
                asyncio.create_task(recv_task()),
                asyncio.create_task(generate_task()),
            ]
            try:
                await asyncio.gather(*tasks)
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except aiohttp.ClientResponseError as e:
                raise APIStatusError(
                    message=e.message,
                    status_code=e.status,
                    request_id=request_id,
                    body=None,
                ) from e
            except APIStatusError:
                raise
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                await utils.aio.gracefully_cancel(*tasks)
                await decoder.aclose()


def _dict_to_voices_list(data: dict[str, Any]):
    voices: List[Voice] = []
    for voice in data["voices"]:
        voices.append(
            Voice(
                id=voice["voice_id"],
                name=voice["name"],
                category=voice["category"],
                settings=None,
            )
        )
    return voices


def _strip_nones(data: dict[str, Any]):
    return {k: v for k, v in data.items() if v is not None}


def _synthesize_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice.id
    model_id = opts.model
    output_format = opts.encoding
    latency = opts.streaming_latency
    return (
        f"{base_url}/text-to-speech/{voice_id}/stream?"
        f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}"
    )


def _stream_url(opts: _TTSOptions) -> str:
    base_url = opts.base_url
    voice_id = opts.voice.id
    model_id = opts.model
    output_format = opts.encoding
    latency = opts.streaming_latency
    enable_ssml = str(opts.enable_ssml_parsing).lower()
    language = opts.language
    inactivity_timeout = opts.inactivity_timeout
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream-input?"
        f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}&"
        f"enable_ssml_parsing={enable_ssml}&inactivity_timeout={inactivity_timeout}"
    )
    if language is not None:
        url += f"&language_code={language}"
    return url
