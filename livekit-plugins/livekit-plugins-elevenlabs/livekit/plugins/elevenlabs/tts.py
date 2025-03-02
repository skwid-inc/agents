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
import re
import weakref
from dataclasses import dataclass
from typing import Any, List, Literal, Optional

import aiohttp
import numpy as np
import soundfile as sf
from app_config import AppConfig
from filler_phrases import get_wav_if_available
from helpers import normalize_text_for_elevenlabs
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

_Encoding = Literal["mp3", "pcm"]


def _sample_rate_from_format(output_format: TTSEncoding) -> int:
    split = output_format.split("_")  # e.g: mp3_22050_32
    return int(split[1])


def _encoding_from_format(output_format: TTSEncoding) -> _Encoding:
    if output_format.startswith("mp3"):
        return "mp3"
    elif output_format.startswith("pcm"):
        return "pcm"

    raise ValueError(f"Unknown format: {output_format}")


@dataclass
class VoiceSettings:
    stability: float  # [0.0 - 1.0]
    similarity_boost: float  # [0.0 - 1.0]
    style: float | None = None  # [0.0 - 1.0]
    use_speaker_boost: bool | None = False
    speed: float | None = None  # [0.7 - 1.2]


@dataclass
class Voice:
    id: str
    name: str
    category: str
    settings: VoiceSettings | None = None
    speed: float | None = None


DEFAULT_VOICE = Voice(
    id="EXAVITQu4vr4xnSDxMaL",
    name="Bella",
    category="premade",
    settings=VoiceSettings(
        stability=0.71,
        similarity_boost=0.5,
        style=0.0,
        use_speaker_boost=True,
        speed=1.0,
    ),
)

API_BASE_URL_V1 = "https://api.elevenlabs.io/v1"
AUTHORIZATION_HEADER = "xi-api-key"
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


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: Voice = DEFAULT_VOICE,
        model: TTSModels | str = "eleven_flash_v2_5",
        api_key: str | None = None,
        base_url: str | None = None,
        encoding: TTSEncoding = "mp3_22050_32",
        streaming_latency: int = 3,
        word_tokenizer: tokenize.WordTokenizer = tokenize.basic.WordTokenizer(
            ignore_punctuation=False  # punctuation can help for intonation
        ),
        enable_ssml_parsing: bool = False,
        chunk_length_schedule: list[int] = [20, 60, 120, 200],  # range is [50, 500]
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
            encoding (TTSEncoding): Audio encoding format. Defaults to "mp3_22050_32".
            streaming_latency (int): Latency in seconds for streaming. Defaults to 3.
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
            sample_rate=_sample_rate_from_format(encoding),
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

        self._opts = _TTSOptions(
            voice=voice,
            model=model,
            api_key=api_key,
            base_url=base_url or API_BASE_URL_V1,
            encoding=encoding,
            sample_rate=self.sample_rate,
            streaming_latency=streaming_latency,
            word_tokenizer=word_tokenizer,
            chunk_length_schedule=chunk_length_schedule,
            enable_ssml_parsing=enable_ssml_parsing,
            language=language,
        )
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
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
        voice: Voice | None = None,
        model: TTSModels | str | None = None,
        language: str | None = None,
        speed: float | None = None,
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
        if speed is not None:
            self._opts.voice.settings.speed = speed

    def synthesize(
        self,
        text: str,
        *,
        conn_options: Optional[APIConnectOptions] = None,
    ) -> "ChunkedStream":
        logger.info(f"Synthesize called with text: {text}")

        if (
            "Exeter Finance LLC" in text
            and "Dallas" not in text
            and "Carrollton" not in text
        ):
            self.update_options(speed=1.2)
        elif "Por favor diga español" in text:
            self.update_options(
                voice="db832ebd-3cb6-42e7-9d47-912b425adbaa",
                model="sonic-multilingual",
                language="es",
            )
        elif "Carrollton" in text or "Dallas" in text:
            logger.info("Carrollton or Dallas detected, setting speed to slowest")
            self.update_options(speed=0.7)
        else:
            self.update_options(speed=0.95)

        raw_input_text = text
        text = normalize_text_for_elevenlabs(text)

        logger.info(f"Processed text: {text}")
        return ChunkedStream(
            tts=self,
            input_text=text,
            raw_input_text=raw_input_text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
        )

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
        raw_input_text: str,
        opts: _TTSOptions,
        conn_options: Optional[APIConnectOptions] = None,
        session: aiohttp.ClientSession,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts, self._session = opts, session
        self._raw_input_text = raw_input_text
        if _encoding_from_format(self._opts.encoding) == "mp3":
            self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        logger.info(f"ChunkedStream _run with input text: {self._input_text}")
        print(f"get_wav_if_available for {self._raw_input_text}")
        filler_phrase_wav = get_wav_if_available(self._raw_input_text)
        print(f"filler_phrase_wav: {filler_phrase_wav}")
        if filler_phrase_wav:
            await self._play_presynthesized_audio(filler_phrase_wav)
            return

        request_id = utils.shortuuid()
        bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate, num_channels=1
        )

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

                encoding = _encoding_from_format(self._opts.encoding)
                if encoding == "mp3":
                    async for bytes_data, _ in resp.content.iter_chunks():
                        for frame in self._mp3_decoder.decode_chunk(bytes_data):
                            for frame in bstream.write(frame.data.tobytes()):
                                self._event_ch.send_nowait(
                                    tts.SynthesizedAudio(
                                        request_id=request_id,
                                        frame=frame,
                                    )
                                )
                else:
                    async for bytes_data, _ in resp.content.iter_chunks():
                        for frame in bstream.write(bytes_data):
                            self._event_ch.send_nowait(
                                tts.SynthesizedAudio(
                                    request_id=request_id,
                                    frame=frame,
                                )
                            )

                for frame in bstream.flush():
                    self._event_ch.send_nowait(
                        tts.SynthesizedAudio(request_id=request_id, frame=frame)
                    )

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

    async def _play_presynthesized_audio(self, wav_path: str) -> None:
        logger.info(
            f"ChunkedStream _run with input text for Presynthesized Audio: {self._input_text}"
        )
        request_id = utils.shortuuid()

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
            self._event_ch.send_nowait(
                tts.SynthesizedAudio(
                    request_id=request_id,
                    frame=frame,
                )
            )

        logger.info(f"ChunkedStream _run completed for Presynthesized Audio")


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
        self._mp3_decoder = utils.codecs.Mp3StreamDecoder()

    async def _run(self) -> None:
        request_id = utils.shortuuid()
        self._segments_ch = utils.aio.Chan[tokenize.WordStream]()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            word_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    print(f"inside _tokenize_input, input: {input}")

                    # pattern = r"([a-zA-Z])([^\w\s])([a-zA-Z])"

                    # Replace with the first alphabet character, the punctuation, a space, and the second alphabet character
                    # input = re.sub(pattern, r"\1\2 \3", input)
                    # if input.endswith("."):
                    #     input += " "
                    # print(f"inside _tokenize_input, input after regex: {input}")

                    # input = "test "
                    # input = input.replace(".", "dot")
                    if "$" in input or "." in input:
                        input = f" {input.strip()}"
                    else:
                        input = f" {input.strip()} "
                    if word_stream is None:
                        print(f"word stream is None, creating new word stream")
                        # new segment (after flush for e.g)
                        word_stream = self._opts.word_tokenizer.stream()
                        self._segments_ch.send_nowait(word_stream)

                    print(f"pushing text to word stream: {input}")
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
            print(f"init_pkt: {init_pkt}")
            await ws_conn.send_str(json.dumps(init_pkt))

            async def send_task():

                nonlocal expected_text
                xml_content = []
                # word_count = 0  # Initialize word counter

                async for data in word_stream:
                    print(f"inside send_task, data: {data}")
                    text = data.token
                    expected_text += text

                    # handle XML phoneme content
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

                    # Add space after punctuation to separate from next word
                    if text in [".", ",", "!", "?", ";", ":", "$"]:
                        data_pkt = dict(text=f"{text}")
                    else:
                        data_pkt = dict(text=f"{text} ")
                    logger.info(
                        f"about to send text to elevenlabs, data_pkt: {data_pkt}"
                    )
                    self._mark_started()
                    await ws_conn.send_str(json.dumps(data_pkt))

                    # Increment word counter and flush every 8 words or after punctuation
                    # word_count += 1
                    if any(
                        text.strip().endswith(punctuation)
                        for punctuation in [
                            ".",
                            ",",
                            "!",
                            "?",
                            ";",
                            ":",
                        ]
                    ):
                        logger.info(
                            "Elevenlabs: Sending flush after sentence-ending punctuation"
                        )
                        await ws_conn.send_str(json.dumps({"flush": True}))

                if xml_content:
                    logger.warning("11labs stream ended with incomplete xml content")

                # Final flush at the end
                logger.info("Elevenlabs: Sending final flush")
                await ws_conn.send_str(json.dumps({"flush": True}))

            async def recv_task():
                nonlocal expected_text
                received_text = ""
                audio_bstream = utils.audio.AudioByteStream(
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                )
                last_frame: rtc.AudioFrame | None = None

                def _send_last_frame(*, segment_id: str, is_final: bool) -> None:
                    nonlocal last_frame
                    if last_frame is not None:
                        self._event_ch.send_nowait(
                            tts.SynthesizedAudio(
                                request_id=request_id,
                                segment_id=segment_id,
                                frame=last_frame,
                                is_final=is_final,
                            )
                        )
                        last_frame = None

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
                    encoding = _encoding_from_format(self._opts.encoding)
                    if data.get("audio"):
                        b64data = base64.b64decode(data["audio"])
                        if encoding == "mp3":
                            for frame in self._mp3_decoder.decode_chunk(b64data):
                                for frame in audio_bstream.write(frame.data.tobytes()):
                                    _send_last_frame(
                                        segment_id=segment_id, is_final=False
                                    )
                                    last_frame = frame
                        else:
                            for frame in audio_bstream.write(b64data):
                                _send_last_frame(segment_id=segment_id, is_final=False)
                                last_frame = frame
                    elif data.get("isFinal"):
                        for frame in audio_bstream.flush():
                            _send_last_frame(segment_id=segment_id, is_final=False)
                            last_frame = frame
                        _send_last_frame(segment_id=segment_id, is_final=True)
                        break
                    elif data.get("error"):
                        logger.error("11labs reported an error: %s", data["error"])
                    else:
                        logger.error("unexpected 11labs message %s", data)

                    # print(f"Elevenlabs: data: {data}")
                    if alignment := data.get("normalizedAlignment"):
                        received_text += "".join(alignment.get("chars", [])).replace(
                            " ", ""
                        )
                        logger.info(f"Elevenlabs: received_text: {received_text}")
                        logger.info(f"Elevenlabs: expected_text: {expected_text}")
                        # if received_text == expected_text:
                        for frame in audio_bstream.flush():
                            _send_last_frame(segment_id=segment_id, is_final=False)
                            last_frame = frame
                        _send_last_frame(segment_id=segment_id, is_final=True)
                        # break

            tasks = [
                asyncio.create_task(send_task()),
                asyncio.create_task(recv_task()),
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
    synthesize_url = f"{base_url}/text-to-speech/{voice_id}/stream?"
    synthesize_url += f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}&auto_mode=false"
    print(f"synthesize_url: {synthesize_url}")
    return synthesize_url


def _stream_url(opts: _TTSOptions) -> str:

    base_url = opts.base_url
    voice_id = opts.voice.id
    model_id = opts.model
    output_format = opts.encoding
    latency = opts.streaming_latency
    enable_ssml = str(opts.enable_ssml_parsing).lower()
    language = opts.language
    url = (
        f"{base_url}/text-to-speech/{voice_id}/stream-input?"
        f"model_id={model_id}&output_format={output_format}&optimize_streaming_latency={latency}&"
        f"enable_ssml_parsing={enable_ssml}"
    )
    if language is not None:
        url += f"&language_code={language}"
    print(f"stream_url: {url}")
    return url
