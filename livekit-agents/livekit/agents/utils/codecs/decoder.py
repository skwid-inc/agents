# Copyright 2024 LiveKit, Inc.
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

import asyncio
import io
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

from livekit.agents.utils import aio

try:
    # preload to ensure faster startup
    import av  # noqa
except ImportError:
    pass
import logging
import threading

from livekit import rtc

logger = logging.getLogger(__name__)


class StreamBuffer:
    """
    A thread-safe buffer that behaves like an IO stream.
    Allows writing from one thread and reading from another.
    """

    def __init__(self):
        self._buffer = io.BytesIO()
        self._lock = threading.Lock()
        self._data_available = threading.Condition(self._lock)
        self._eof = False  # EOF flag to signal no more writes

    def write(self, data: bytes):
        """Write data to the buffer from a writer thread."""
        with self._data_available:  # Lock and notify readers
            self._buffer.seek(0, io.SEEK_END)  # Move to the end
            self._buffer.write(data)
            self._data_available.notify_all()  # Notify waiting readers

    def read(self, size: int = -1) -> bytes:
        """Read data from the buffer in a reader thread."""

        if self._buffer.closed:
            return b""

        with self._data_available:
            while True:
                self._buffer.seek(0)  # Rewind for reading
                data = self._buffer.read(size)

                # If data is available, return it
                if data:
                    # Shrink the buffer to remove already-read data
                    remaining = self._buffer.read()
                    self._buffer = io.BytesIO(remaining)
                    return data

                # If EOF is signaled and no data remains, return EOF
                if self._eof:
                    return b""

                # Wait for more data
                self._data_available.wait()

    def end_input(self):
        """Signal that no more data will be written."""
        with self._data_available:
            self._eof = True
            self._data_available.notify_all()

    def force_notify(self):
        logger.info("StreamBuffer: Forcing notify")
        with self._data_available:
            self._data_available.notify_all()

    def close(self):
        self._buffer.close()


class AudioStreamDecoder:
    """A class that can be used to decode audio stream into PCM AudioFrames.

    Decoders are stateful, and it should not be reused across multiple streams. Each decoder
    is designed to decode a single stream.
    """

    _max_workers: int = 10
    _executor: Optional[ThreadPoolExecutor] = None

    def __init__(self, *, sample_rate: int = 48000, num_channels: int = 1):
        try:
            import av  # noqa
        except ImportError:
            raise ImportError(
                "You haven't included the 'codecs' optional dependencies. Please install the 'codecs' extra by running `pip install livekit-agents[codecs]`"
            )

        logger.info(
            f"Initializing decoder with sample rate: {sample_rate} and num channels: {num_channels}"
        )

        self._sample_rate = sample_rate
        self._layout = "mono"
        if num_channels == 2:
            self._layout = "stereo"
        elif num_channels != 1:
            raise ValueError(f"Invalid number of channels: {num_channels}")

        self._output_ch = aio.Chan[rtc.AudioFrame]()
        self._closed = False
        self._started = False
        self._input_buf = StreamBuffer()
        self._loop = asyncio.get_event_loop()
        if self.__class__._executor is None:
            # each decoder instance will submit jobs to the shared pool
            self.__class__._executor = ThreadPoolExecutor(max_workers=self.__class__._max_workers)

    def push(self, chunk: bytes):
        self._input_buf.write(chunk)
        if not self._started:
            logger.info("Starting decode loop")
            self._started = True
            self._loop.run_in_executor(self.__class__._executor, self._decode_loop)

    def end_input(self):
        self._input_buf.end_input()

    def force_notify(self):
        logger.info("AudioStreamDecoder: Forcing notify")
        self._input_buf.force_notify()

    def _decode_loop(self):
        logger.info("IN decode loop")
        container = av.open(self._input_buf)
        audio_stream = next(s for s in container.streams if s.type == "audio")
        resampler = av.AudioResampler(
            # convert to signed 16-bit little endian
            format="s16",
            layout=self._layout,
            rate=self._sample_rate,
        )
        try:
            # TODO: handle error where audio stream isn't found
            if not audio_stream:
                logger.info("No audio stream found, returning")
                return
            for frame in container.decode(audio_stream):
                logging.info("Decoding frame")
                if self._closed:
                    logger.info("Decoder closed, returning")
                    return
                for resampled_frame in resampler.resample(frame):
                    nchannels = len(resampled_frame.layout.channels)
                    data = resampled_frame.to_ndarray().tobytes()
                    logger.info("Sending frame to output channel")
                    self._output_ch.send_nowait(
                        rtc.AudioFrame(
                            data=data,
                            num_channels=nchannels,
                            sample_rate=int(resampled_frame.sample_rate),
                            samples_per_channel=int(resampled_frame.samples / nchannels),
                        )
                    )
        finally:
            logger.info("Closing output channel")
            self._output_ch.close()

    def __aiter__(self) -> AsyncIterator[rtc.AudioFrame]:
        return self

    async def __anext__(self) -> rtc.AudioFrame:
        try:
            return await self._output_ch.recv()
        except aio.ChanClosed:
            raise StopAsyncIteration

    async def aclose(self):
        import traceback

        logger.info(f"Closing decoder")
        logger.info(traceback.format_exc())
        if self._closed:
            return
        self._closed = True
        self.end_input()
        self._input_buf.close()
        # wait for decode loop to finish
        try:
            await self._output_ch.recv()
        except aio.ChanClosed:
            pass
