"""Reusable, bounded, Session-independent Agent Runner execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Protocol

from loguru import logger

from myclaw.agent.run import (
    _append_run_message,
    _assistant_run_message,
    _close_iterator,
    _externalize_tool_result,
    _identity_tool_result,
    _model_failure,
    _repair_cancelled_messages,
    _repair_failed_messages,
    _tool_run_message,
    _ToolCallState,
    build_assistant_repair_message,
)
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelCompleted,
    ModelContinuation,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningDelta,
    TextDelta,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import (
    ConfirmationRequester,
    ModelToolCall,
    ToolGateway,
    ToolResult,
)

type AgentRunnerRoute = Literal["chat", "schedule"]
type AgentRunnerSegment = Literal["reasoning", "response"]
type AgentRunnerFinishReason = Literal["completed", "failed", "cancelled", "max_iterations"]
type AgentRunnerOutput = (
    ReasoningDelta | TextDelta | AgentRunnerResponseSegmentEnd | AgentRunnerToolCallStarted
)

_USAGE_KEYS = frozenset({"model_calls", "input_tokens", "output_tokens", "total_tokens"})
_MAX_ITERATIONS_MESSAGE = (
    "MyClaw 本轮对话已经达到最大循环次数，仍没有输出最终结果。"  # noqa: RUF001
    "可以再次尝试本次请求或者尝试给出更明确的任务目标。"
)
_CANCELLED_MESSAGE = "MyClaw 已取消本轮对话。"


@dataclass(frozen=True, slots=True)
class AgentRunnerResponseSegmentEnd:
    """End of one contiguous Provider-visible reasoning or response segment."""

    type: ClassVar[Literal["segment_end"]] = "segment_end"
    segment: AgentRunnerSegment


@dataclass(frozen=True, slots=True)
class AgentRunnerToolCallStarted:
    """A Tool call immediately before it is handed to the Tool Gateway."""

    type: ClassVar[Literal["tool_call_started"]] = "tool_call_started"
    tool_call_id: str
    tool_name: str
    arguments: str


class AgentRunnerOutputCallback(Protocol):
    async def __call__(self, event: AgentRunnerOutput) -> None: ...


class AgentRunnerRouter(Protocol):
    def stream(
        self,
        route: AgentRunnerRoute,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(
        self,
        route: AgentRunnerRoute,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
        continuation: ModelContinuation | None = None,
    ) -> ModelResponse: ...


def _empty_usage() -> dict[str, int]:
    return {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


@dataclass(slots=True)
class AgentRunnerResult:
    messages: list[dict[str, Any]]
    final_content: str
    usage: dict[str, int] = field(default_factory=_empty_usage)
    finish_reason: AgentRunnerFinishReason = "completed"
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.final_content, str):
            raise TypeError("final_content must be a string")
        if set(self.usage) != _USAGE_KEYS:
            raise ValueError("usage must contain exactly the four Agent Runner usage fields")
        for value in self.usage.values():
            if type(value) is not int or value < 0:
                raise ValueError("usage values must be nonnegative integers")
        if self.usage["total_tokens"] != (self.usage["input_tokens"] + self.usage["output_tokens"]):
            raise ValueError("usage total_tokens must equal input_tokens + output_tokens")

        if self.finish_reason not in {"completed", "failed", "cancelled", "max_iterations"}:
            raise ValueError("finish_reason is invalid")
        if self.finish_reason == "completed" and self.error is not None:
            raise ValueError("completed Agent Runner results cannot contain an error")
        if self.finish_reason == "failed" and not isinstance(self.error, ErrorInfo):
            raise ValueError("failed Agent Runner results require an ErrorInfo")
        if self.finish_reason == "cancelled":
            if not isinstance(self.error, ErrorInfo) or self.error.code != "turn_cancelled":
                raise ValueError("cancelled Agent Runner results require turn_cancelled")
        if self.finish_reason == "max_iterations":
            if not isinstance(self.error, ErrorInfo) or self.error.code != "agent_iteration_limit":
                raise ValueError(
                    "maximum-iteration Agent Runner results require agent_iteration_limit"
                )


class _CallbackFailure(Exception):
    def __init__(self, error: BaseException) -> None:
        self.error = error
        super().__init__("Agent Runner output callback failed")


class AgentRunner:
    """Run bounded ReAct execution while owning only a Model Router reference."""

    def __init__(self, model_router: AgentRunnerRouter) -> None:
        self._model_router = model_router

    async def run(
        self,
        initial_messages: Sequence[dict[str, Any]],
        *,
        model: AgentRunnerRoute,
        tool_gateway: ToolGateway | None,
        on_output: AgentRunnerOutputCallback,
        confirmation: ConfirmationRequester | None,
        externalize_result: Callable[[ToolResult], ToolResult] | None,
        cancel_requested: Callable[[], bool] | None,
        max_iterations: int,
    ) -> AgentRunnerResult:
        if model not in {"chat", "schedule"}:
            raise ValueError("Agent Runner model route must be chat or schedule")
        _validate_max_iterations(max_iterations)

        runtime_messages = deepcopy(list(initial_messages))
        increment: list[dict[str, Any]] = []
        pending_tool_calls: list[ModelToolCall] = []
        partial_content: list[str] = []
        usage = _empty_usage()
        continuation: ModelContinuation | None = None
        segment: AgentRunnerSegment | None = None
        events: AsyncIterator[ModelStreamEvent] | None = None
        is_cancel_requested = cancel_requested or _never_cancel
        externalize = externalize_result or _identity_tool_result

        async def emit(event: AgentRunnerOutput) -> None:
            try:
                await on_output(event)
            except asyncio.CancelledError as error:
                task = asyncio.current_task()
                if task is not None and task.cancelling():
                    raise
                raise _CallbackFailure(error) from error
            except BaseException as error:
                raise _CallbackFailure(error) from error

        async def close_segment() -> None:
            nonlocal segment
            if segment is None:
                return
            closing = segment
            segment = None
            await emit(AgentRunnerResponseSegmentEnd(segment=closing))

        async def close_segment_for_error() -> None:
            try:
                await close_segment()
            except _CallbackFailure as failure:
                raise failure.error from failure

        async def start_segment(next_segment: AgentRunnerSegment) -> None:
            nonlocal segment
            if segment != next_segment:
                await close_segment()
                segment = next_segment

        def finish_cancelled(*, final_content: str) -> AgentRunnerResult:
            _repair_cancelled_messages(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
            )
            return _cancelled_result(increment, usage, final_content=final_content)

        if is_cancel_requested():
            return finish_cancelled(final_content="")

        try:
            while True:
                partial_content.clear()
                usage["model_calls"] += 1
                response: ModelResponse | None = None
                if model == "chat":
                    events = self._model_router.stream(
                        model,
                        messages=deepcopy(runtime_messages),
                        tools=() if tool_gateway is None else tuple(tool_gateway.schemas),
                        continuation=continuation,
                    )
                    try:
                        async for event in events:
                            if isinstance(event, ReasoningDelta):
                                await start_segment("reasoning")
                                await emit(event)
                                if is_cancel_requested():
                                    await close_segment()
                                    return finish_cancelled(final_content="")
                                continue
                            if isinstance(event, TextDelta):
                                await start_segment("response")
                                partial_content.append(event.delta)
                                await emit(event)
                                if is_cancel_requested():
                                    cancelled_content = "".join(partial_content)
                                    await close_segment()
                                    return finish_cancelled(final_content=cancelled_content)
                                continue
                            if not isinstance(event, ModelCompleted):
                                raise _model_failure()
                            response = event.response
                            await close_segment()
                            break
                        if response is None:
                            raise _model_failure()
                    finally:
                        await _close_iterator(events)
                        events = None
                else:
                    response = await self._model_router.complete(
                        model,
                        messages=deepcopy(runtime_messages),
                        tools=() if tool_gateway is None else tuple(tool_gateway.schemas),
                        continuation=continuation,
                    )

                _add_usage(usage, response.usage)
                _append_run_message(runtime_messages, increment, _assistant_run_message(response))
                partial_content.clear()
                pending_tool_calls = (
                    list(response.message.tool_calls) if tool_gateway is not None else []
                )
                continuation_for_next_call = response.continuation if pending_tool_calls else None

                if is_cancel_requested():
                    return finish_cancelled(final_content="")

                if not pending_tool_calls:
                    return AgentRunnerResult(
                        messages=increment,
                        final_content=response.message.content,
                        usage=usage,
                    )

                assert tool_gateway is not None
                for tool_call in response.message.tool_calls:
                    if is_cancel_requested():
                        return finish_cancelled(final_content="")
                    await emit(
                        AgentRunnerToolCallStarted(
                            tool_call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                        )
                    )

                    state = _ToolCallState()
                    result: ToolResult | None = None
                    try:
                        try:
                            result = await _await_tool_call(
                                tool_gateway,
                                tool_call,
                                confirmation,
                                state,
                            )
                        except BaseException as failure:
                            if not isinstance(failure, Exception) and state.result is not None:
                                try:
                                    recovered = _externalize_tool_result(state.result, externalize)
                                    _append_run_message(
                                        runtime_messages,
                                        increment,
                                        _tool_run_message(recovered),
                                    )
                                except Exception:
                                    pass
                                else:
                                    pending_tool_calls.pop(0)
                            raise
                        if result is None:
                            raise RuntimeError("Tool Gateway ended without a result")
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        result = ToolResult(
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                            status="error",
                            content=f"{tool_call.name} could not complete the request.",
                            artifact=None,
                        )

                    result = _externalize_tool_result(result, externalize)
                    _append_run_message(runtime_messages, increment, _tool_run_message(result))
                    pending_tool_calls.pop(0)
                    if is_cancel_requested():
                        return finish_cancelled(final_content="")

                if usage["model_calls"] >= max_iterations:
                    limit_error = ErrorInfo("agent_iteration_limit", _MAX_ITERATIONS_MESSAGE)
                    _append_run_message(
                        runtime_messages,
                        increment,
                        build_assistant_repair_message(
                            content=_MAX_ITERATIONS_MESSAGE,
                            status="error",
                            error=limit_error,
                            model_calls=0,
                        ),
                    )
                    return AgentRunnerResult(
                        messages=increment,
                        final_content=_MAX_ITERATIONS_MESSAGE,
                        usage=usage,
                        finish_reason="max_iterations",
                        error=limit_error,
                    )

                continuation = continuation_for_next_call
        except _CallbackFailure as failure:
            raise failure.error from failure
        except ModelCallError as failure:
            await close_segment_for_error()
            if failure.error.code == "turn_cancelled" or is_cancel_requested():
                cancelled_content = "".join(partial_content)
                return finish_cancelled(final_content=cancelled_content)
            _log_legacy_agent_failure(failure)
            failed_content = "".join(partial_content) if model == "chat" else ""
            _repair_failed_messages(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
                stream=model == "chat",
                failure=failure,
            )
            return AgentRunnerResult(
                messages=increment,
                final_content=failed_content,
                usage=usage,
                finish_reason="failed",
                error=failure.error,
            )
        except asyncio.CancelledError:
            await close_segment_for_error()
            if is_cancel_requested():
                cancelled_content = "".join(partial_content)
                return finish_cancelled(final_content=cancelled_content)
            try:
                _repair_cancelled_messages(
                    runtime_messages,
                    increment,
                    partial_content,
                    pending_tool_calls,
                )
            except BaseException:
                pass
            raise
        except Exception:
            await close_segment_for_error()
            if is_cancel_requested():
                cancelled_content = "".join(partial_content)
                return finish_cancelled(final_content=cancelled_content)
            failed_content = "".join(partial_content) if model == "chat" else ""
            generic_failure = _model_failure()
            _repair_failed_messages(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
                stream=model == "chat",
                failure=generic_failure,
            )
            return AgentRunnerResult(
                messages=increment,
                final_content=failed_content,
                usage=usage,
                finish_reason="failed",
                error=generic_failure.error,
            )
        finally:
            await _close_iterator(events)


def _validate_max_iterations(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 50:
        raise ValueError("max_iterations must be an integer at least 50")


def _never_cancel() -> bool:
    return False


async def _await_tool_call(
    gateway: ToolGateway,
    tool_call: ModelToolCall,
    confirmation: ConfirmationRequester | None,
    state: _ToolCallState,
) -> ToolResult:
    operation = asyncio.create_task(gateway.call(tool_call, confirmation=confirmation))
    try:
        result = await operation
        state.result = result
        return result
    finally:
        if not operation.done():
            operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        if state.result is None and operation.done() and not operation.cancelled():
            try:
                state.result = operation.result()
            except BaseException:
                pass


def _add_usage(total: dict[str, int], usage: ModelUsage) -> None:
    total["input_tokens"] += usage.input_tokens
    total["output_tokens"] += usage.output_tokens
    total["total_tokens"] += usage.total_tokens


def _cancelled_result(
    messages: list[dict[str, Any]],
    usage: dict[str, int],
    *,
    final_content: str,
) -> AgentRunnerResult:
    return AgentRunnerResult(
        messages=messages,
        final_content=final_content,
        usage=usage,
        finish_reason="cancelled",
        error=ErrorInfo("turn_cancelled", _CANCELLED_MESSAGE),
    )


__all__ = [
    "AgentRunner",
    "AgentRunnerFinishReason",
    "AgentRunnerOutput",
    "AgentRunnerOutputCallback",
    "AgentRunnerResponseSegmentEnd",
    "AgentRunnerResult",
    "AgentRunnerRoute",
    "AgentRunnerRouter",
    "AgentRunnerSegment",
    "AgentRunnerToolCallStarted",
]


def _log_legacy_agent_failure(failure: ModelCallError) -> None:
    def set_legacy_name(record: Any) -> None:
        record["name"] = "myclaw.session.conversation"

    logger.patch(set_legacy_name).opt(exception=failure).error(
        "Agent Run failed code={} type={}",
        failure.error.code,
        type(failure).__name__,
    )
