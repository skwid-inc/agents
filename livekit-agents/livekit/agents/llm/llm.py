from __future__ import annotations

import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, AsyncIterable, AsyncIterator, Generic, Literal, TypeVar, Union

from app_config import AppConfig
from livekit import rtc
from livekit.agents._exceptions import APIConnectionError, APIError

from .. import utils
from ..log import logger
from ..metrics import LLMMetrics
from ..types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from ..utils import aio
from . import function_context
from .chat_context import ChatContext, ChatRole
from custom_logger import log


@dataclass
class ChoiceDelta:
    role: ChatRole
    content: str | None = None
    tool_calls: list[function_context.FunctionCallInfo] | None = None


@dataclass
class CompletionUsage:
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


@dataclass
class Choice:
    delta: ChoiceDelta
    index: int = 0


@dataclass
class LLMCapabilities:
    supports_choices_on_int: bool = True
    """check whether the LLM supports integer enums choices as function arguments"""
    requires_persistent_functions: bool = False
    """if the LLM requires function definition when previous function calls exist in chat context"""


@dataclass
class ChatChunk:
    request_id: str
    choices: list[Choice] = field(default_factory=list)
    usage: CompletionUsage | None = None


@dataclass
class ToolChoice:
    type: Literal["function"]
    name: str


TEvent = TypeVar("TEvent")


class LLM(
    ABC,
    rtc.EventEmitter[Union[Literal["metrics_collected"], TEvent]],
    Generic[TEvent],
):
    def __init__(self, *, capabilities: LLMCapabilities | None = None) -> None:
        super().__init__()
        if capabilities is None:
            capabilities = LLMCapabilities()
        self._capabilities = capabilities
        self._label = f"{type(self).__module__}.{type(self).__name__}"

    @property
    def label(self) -> str:
        return self._label

    @abstractmethod
    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx: function_context.FunctionContext | None = None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice: (
            Union[ToolChoice, Literal["auto", "required", "none"]] | None
        ) = None,
    ) -> "LLMStream": ...

    @property
    def capabilities(self) -> LLMCapabilities:
        return self._capabilities

    async def aclose(self) -> None: ...

    async def __aenter__(self) -> LLM:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()


class LLMStream(ABC):
    def __init__(
        self,
        llm: LLM,
        *,
        chat_ctx: ChatContext,
        fnc_ctx: function_context.FunctionContext | None,
        conn_options: APIConnectOptions,
    ) -> None:
        self._llm = llm
        self._chat_ctx = chat_ctx
        self._fnc_ctx = fnc_ctx
        self._conn_options = conn_options

        self._event_ch = aio.Chan[ChatChunk]()
        self._event_aiter, monitor_aiter = aio.itertools.tee(self._event_ch, 2)
        self._metrics_task = asyncio.create_task(
            self._metrics_monitor_task(monitor_aiter), name="LLM._metrics_task"
        )

        llm_stream_task_id = f"LLMStream-{str(uuid.uuid4())}"
        logger.info(f"Starting task: {llm_stream_task_id}")
        pending_tasks = AppConfig().get_call_metadata().get("pending_livekit_tasks", {})
        pending_tasks[llm_stream_task_id] = time.time()
        log.pipeline("pending task - LLM stream")
        log.pipeline(pending_tasks)
        self._task = asyncio.create_task(self._main_task())

        self._task.add_done_callback(lambda _: self._event_ch.close())

        def _post_task_callback(_) -> None:
            log.pipeline(f"Task completed: {llm_stream_task_id}")
            pending_tasks.pop(llm_stream_task_id, None)
            log.pipeline(pending_tasks)

        self._task.add_done_callback(_post_task_callback)

        self._function_calls_info: list[function_context.FunctionCallInfo] = []
        self._function_tasks = set[asyncio.Task[Any]]()

    @abstractmethod
    async def _run(self) -> None: ...

    async def _main_task(self) -> None:
        for i in range(self._conn_options.max_retry + 1):
            try:
                return await self._run()
            except APIError as e:
                retry_interval = self._conn_options._interval_for_retry(i)
                if self._conn_options.max_retry == 0 or not e.retryable:
                    raise
                elif i == self._conn_options.max_retry:
                    raise APIConnectionError(
                        f"failed to generate LLM completion after {self._conn_options.max_retry + 1} attempts",
                    ) from e
                else:
                    logger.warning(
                        f"failed to generate LLM completion, retrying in {retry_interval}s",
                        exc_info=e,
                        extra={
                            "llm": self._llm._label,
                            "attempt": i + 1,
                        },
                    )

                await asyncio.sleep(retry_interval)

    @utils.log_exceptions(logger=logger)
    async def _metrics_monitor_task(
        self, event_aiter: AsyncIterable[ChatChunk]
    ) -> None:
        start_time = time.perf_counter()
        ttft = -1.0
        request_id = ""
        usage: CompletionUsage | None = None

        async for ev in event_aiter:
            request_id = ev.request_id
            if ttft == -1.0:
                ttft = time.perf_counter() - start_time

            if ev.usage is not None:
                usage = ev.usage

        duration = time.perf_counter() - start_time
        metrics = LLMMetrics(
            timestamp=time.time(),
            request_id=request_id,
            ttft=ttft,
            duration=duration,
            cancelled=self._task.cancelled(),
            label=self._llm._label,
            completion_tokens=usage.completion_tokens if usage else 0,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
            tokens_per_second=usage.completion_tokens / duration if usage else 0.0,
            error=None,
        )
        self._llm.emit("metrics_collected", metrics)

    @property
    def function_calls(self) -> list[function_context.FunctionCallInfo]:
        """List of called functions from this stream."""
        return self._function_calls_info

    @property
    def chat_ctx(self) -> ChatContext:
        """The initial chat context of this stream."""
        return self._chat_ctx

    @property
    def fnc_ctx(self) -> function_context.FunctionContext | None:
        """The function context of this stream."""
        return self._fnc_ctx

    def execute_functions(self) -> list[function_context.CalledFunction]:
        """Execute all functions concurrently of this stream."""
        called_functions: list[function_context.CalledFunction] = []
        for fnc_info in self._function_calls_info:
            called_fnc = fnc_info.execute()
            self._function_tasks.add(called_fnc.task)
            called_fnc.task.add_done_callback(self._function_tasks.remove)
            called_functions.append(called_fnc)

        return called_functions

    async def aclose(self) -> None:
        await aio.gracefully_cancel(self._task)
        await utils.aio.gracefully_cancel(*self._function_tasks)
        await self._metrics_task

    async def __anext__(self) -> ChatChunk:
        try:
            val = await self._event_aiter.__anext__()
        except StopAsyncIteration:
            if not self._task.cancelled() and (exc := self._task.exception()):
                raise exc from None

            raise StopAsyncIteration

        return val

    def __aiter__(self) -> AsyncIterator[ChatChunk]:
        return self

    async def __aenter__(self) -> LLMStream:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.aclose()
