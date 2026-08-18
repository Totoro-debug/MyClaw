"""Lane-neutral Agent Run contract and Runtime Core execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol

from loguru import logger

from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelCompleted,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    TextDelta,
)
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import (
    ConfirmationChannel as ToolConfirmationChannel,
)
from myclaw.tools.tool_gateway import (
    ConfirmationRequest,
    ModelToolCall,
    ToolGateway,
    ToolResult,
    ToolResultStatus,
)

type AgentRunRoute = Literal["chat", "schedule"]
type ConfirmationChannel = ToolConfirmationChannel
type ToolResultExternalizer = Callable[[ToolResult], ToolResult]
type AgentRunContinuationPreparer = Callable[
    [Sequence[dict[str, Any]], Sequence[dict[str, Any]]],
    Awaitable[list[dict[str, Any]]],
]


@dataclass(frozen=True, slots=True)
class AgentRunStartedPayload:
    type: ClassVar[Literal["started"]] = "started"


@dataclass(frozen=True, slots=True)
class AgentRunTextDeltaPayload:
    type: ClassVar[Literal["text_delta"]] = "text_delta"
    delta: str

    def __post_init__(self) -> None:
        if not self.delta:
            raise ValueError("delta must not be empty")


@dataclass(frozen=True, slots=True)
class AgentRunToolStartedPayload:
    type: ClassVar[Literal["tool_started"]] = "tool_started"
    tool_call_id: str
    tool_name: str
    summary: str

    def __post_init__(self) -> None:
        _require_summary(self.summary)


@dataclass(frozen=True, slots=True)
class AgentRunConfirmationRequestedPayload:
    type: ClassVar[Literal["confirmation_requested"]] = "confirmation_requested"
    request: ConfirmationRequest

    def __post_init__(self) -> None:
        if not isinstance(self.request, ConfirmationRequest):
            raise TypeError("confirmation request payload requires a ConfirmationRequest")


@dataclass(frozen=True, slots=True)
class AgentRunToolCompletedPayload:
    type: ClassVar[Literal["tool_completed"]] = "tool_completed"
    tool_call_id: str
    tool_name: str
    status: ToolResultStatus
    summary: str

    def __post_init__(self) -> None:
        _require_summary(self.summary)


@dataclass(frozen=True, slots=True)
class AgentRunModelCallCompletedPayload:
    """Complete text and phase classification for one nonterminal model call."""

    type: ClassVar[Literal["model_call_completed"]] = "model_call_completed"
    content: str
    continues_with_tools: bool


@dataclass(frozen=True, slots=True)
class AgentRunCompletedPayload:
    type: ClassVar[Literal["completed"]] = "completed"
    content: str
    usage: ModelUsage


@dataclass(frozen=True, slots=True)
class AgentRunFailedPayload:
    type: ClassVar[Literal["failed"]] = "failed"
    error: ErrorInfo
    cause: BaseException | None = None


@dataclass(frozen=True, slots=True)
class AgentRunCancelledPayload:
    type: ClassVar[Literal["cancelled"]] = "cancelled"
    partial_content: str


type AgentRunPayload = (
    AgentRunStartedPayload
    | AgentRunTextDeltaPayload
    | AgentRunToolStartedPayload
    | AgentRunConfirmationRequestedPayload
    | AgentRunToolCompletedPayload
    | AgentRunModelCallCompletedPayload
    | AgentRunCompletedPayload
    | AgentRunFailedPayload
    | AgentRunCancelledPayload
)


class AgentRunEmitter(Protocol):
    """Awaitable sink for ordered Agent Run progress payloads."""

    async def emit(self, payload: AgentRunPayload) -> None: ...


class AgentRunRouter(Protocol):
    """Direct-call Model Router boundary used by the awaitable Agent Run."""

    def stream(
        self,
        route: AgentRunRoute,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
    ) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(
        self,
        route: AgentRunRoute,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[OpenAIToolSchema],
    ) -> ModelResponse: ...


@dataclass(slots=True)
class _ToolCallState:
    result: ToolResult | None = None


class AgentRun:
    """Execute the shared model and Tool loop for foreground and Schedule callers."""

    def __init__(
        self,
        *,
        model: AgentRunRouter,
        tool_gateway: ToolGateway | None = None,
        externalize_result: Callable[[ToolResult], ToolResult] | None = None,
        on_artifact_failure: Callable[[Exception, str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._model = model
        self._tool_gateway = tool_gateway
        self._externalize_result = externalize_result
        self._on_artifact_failure = on_artifact_failure
        self._cancel_requested = cancel_requested or (lambda: False)

    async def run(
        self,
        messages: Sequence[dict[str, Any]],
        current_user: dict[str, Any],
        *,
        route: AgentRunRoute,
        emitter: AgentRunEmitter,
        confirmation: ConfirmationChannel | None = None,
        externalize_result: ToolResultExternalizer | None = None,
        continuation_preparer: AgentRunContinuationPreparer | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Run the model and Tool loop without owning Conversation Session state.

        ``messages`` is the complete model-visible context, including the current
        user message with Runtime Context already applied. ``current_user`` is the
        raw message that the caller will later append to its Session. The two
        message lists maintained here deliberately cross an isolated copy boundary
        so a provider or caller cannot mutate the returned Session increment.
        """
        if route not in {"chat", "schedule"}:
            raise ValueError("Agent Run route must be chat or schedule")
        if not isinstance(current_user, dict):
            raise TypeError("current_user must be a dictionary")

        runtime_messages = deepcopy(list(messages))
        increment = [deepcopy(current_user)]
        partial_content: list[str] = []
        pending_tool_calls: list[ModelToolCall] = []
        events: AsyncIterator[ModelStreamEvent] | None = None
        started_emitted = False
        terminal_emitted = False
        stream = route == "chat"
        preparing_continuation = False
        is_cancel_requested = cancel_requested or self._cancel_requested

        try:
            gateway = self._tool_gateway
            frozen_tools = () if gateway is None else tuple(gateway.schemas)
            externalize_result_for_run = (
                externalize_result or self._externalize_result or _identity_tool_result
            )

            await _emit_agent_run_payload(emitter, AgentRunStartedPayload())
            started_emitted = True
            if is_cancel_requested():
                cancelled_content = "".join(partial_content)
                self._repair_awaitable_cancelled(
                    runtime_messages,
                    increment,
                    partial_content,
                    pending_tool_calls,
                )
                await _emit_agent_run_payload(
                    emitter,
                    AgentRunCancelledPayload(partial_content=cancelled_content),
                )
                terminal_emitted = True
                return increment

            while True:
                if preparing_continuation and continuation_preparer is not None:
                    prepared = await continuation_preparer(
                        deepcopy(runtime_messages),
                        deepcopy(increment),
                    )
                    if not isinstance(prepared, list):
                        raise TypeError("Agent Run continuation preparer must return a list")
                    runtime_messages = deepcopy(prepared)
                partial_content.clear()
                events = (
                    self._model.stream(
                        route,
                        messages=runtime_messages,
                        tools=frozen_tools,
                    )
                    if stream
                    else self._direct_complete_events(
                        route,
                        runtime_messages,
                        frozen_tools,
                    )
                )
                model_completed = False
                async for event in events:
                    if isinstance(event, TextDelta):
                        partial_content.append(event.delta)
                        await _emit_agent_run_payload(
                            emitter,
                            AgentRunTextDeltaPayload(delta=event.delta),
                        )
                        if is_cancel_requested():
                            cancelled_content = "".join(partial_content)
                            await _close_iterator(events)
                            events = None
                            self._repair_awaitable_cancelled(
                                runtime_messages,
                                increment,
                                partial_content,
                                pending_tool_calls,
                            )
                            await _emit_agent_run_payload(
                                emitter,
                                AgentRunCancelledPayload(partial_content=cancelled_content),
                            )
                            terminal_emitted = True
                            return increment
                        continue

                    if not isinstance(event, ModelCompleted):
                        raise _model_failure()
                    model_completed = True
                    response = event.response
                    assistant = _assistant_run_message(response)
                    _append_run_message(runtime_messages, increment, assistant)
                    continues_with_tools = bool(response.message.tool_calls and gateway is not None)
                    partial_content.clear()
                    if continues_with_tools:
                        pending_tool_calls = list(response.message.tool_calls)
                    await _close_iterator(events)
                    events = None
                    await _emit_agent_run_payload(
                        emitter,
                        AgentRunModelCallCompletedPayload(
                            content=response.message.content,
                            continues_with_tools=continues_with_tools,
                        ),
                    )
                    if continues_with_tools:
                        assert gateway is not None
                        for tool_call in response.message.tool_calls:
                            await _emit_agent_run_payload(
                                emitter,
                                AgentRunToolStartedPayload(
                                    tool_call_id=tool_call.id,
                                    tool_name=tool_call.name,
                                    summary=_tool_summary("Running", tool_call.name),
                                ),
                            )
                            if is_cancel_requested():
                                cancelled_content = "".join(partial_content)
                                self._repair_awaitable_cancelled(
                                    runtime_messages,
                                    increment,
                                    partial_content,
                                    pending_tool_calls,
                                )
                                await _emit_agent_run_payload(
                                    emitter,
                                    AgentRunCancelledPayload(partial_content=cancelled_content),
                                )
                                terminal_emitted = True
                                return increment

                            result: ToolResult | None = None
                            tool_state = _ToolCallState()
                            try:
                                try:
                                    tool_outcomes = self._call_tool(
                                        gateway,
                                        tool_call,
                                        confirmation,
                                        tool_state,
                                    )
                                    try:
                                        async for tool_outcome in tool_outcomes:
                                            if isinstance(tool_outcome, ConfirmationRequest):
                                                await _emit_agent_run_payload(
                                                    emitter,
                                                    AgentRunConfirmationRequestedPayload(
                                                        request=tool_outcome
                                                    ),
                                                )
                                            else:
                                                result = tool_outcome
                                    finally:
                                        await _close_iterator(tool_outcomes)
                                except BaseException as failure:
                                    if (
                                        not isinstance(failure, Exception)
                                        and tool_state.result is not None
                                    ):
                                        try:
                                            result = self._externalize_awaitable_result(
                                                tool_state.result,
                                                externalize_result_for_run,
                                            )
                                            _append_run_message(
                                                runtime_messages,
                                                increment,
                                                _tool_run_message(result),
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
                            result = self._externalize_awaitable_result(
                                result,
                                externalize_result_for_run,
                            )
                            _append_run_message(
                                runtime_messages, increment, _tool_run_message(result)
                            )
                            pending_tool_calls.pop(0)
                            await _emit_agent_run_payload(
                                emitter,
                                AgentRunToolCompletedPayload(
                                    tool_call_id=result.tool_call_id,
                                    tool_name=result.name,
                                    status=result.status,
                                    summary=_tool_completion_summary(result),
                                ),
                            )
                            if is_cancel_requested():
                                cancelled_content = "".join(partial_content)
                                self._repair_awaitable_cancelled(
                                    runtime_messages,
                                    increment,
                                    partial_content,
                                    pending_tool_calls,
                                )
                                await _emit_agent_run_payload(
                                    emitter,
                                    AgentRunCancelledPayload(partial_content=cancelled_content),
                                )
                                terminal_emitted = True
                                return increment
                        preparing_continuation = True
                        continue

                    await _emit_agent_run_payload(
                        emitter,
                        AgentRunCompletedPayload(
                            content=response.message.content,
                            usage=response.usage,
                        ),
                    )
                    terminal_emitted = True
                    return increment

                if not model_completed:
                    raise _model_failure()
        except ModelCallError as failure:
            await _close_iterator(events)
            events = None
            if failure.error.code == "turn_cancelled" or is_cancel_requested():
                cancelled_content = "".join(partial_content)
                self._repair_awaitable_cancelled(
                    runtime_messages,
                    increment,
                    partial_content,
                    pending_tool_calls,
                )
                if not started_emitted:
                    await _emit_agent_run_payload(emitter, AgentRunStartedPayload())
                    started_emitted = True
                await _emit_agent_run_payload(
                    emitter,
                    AgentRunCancelledPayload(partial_content=cancelled_content),
                )
                terminal_emitted = True
                return increment
            self._repair_awaitable_failed(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
                stream=stream,
                failure=failure,
            )
            if not started_emitted:
                await _emit_agent_run_payload(emitter, AgentRunStartedPayload())
                started_emitted = True
            await _emit_agent_run_payload(
                emitter,
                AgentRunFailedPayload(error=failure.error, cause=failure),
            )
            terminal_emitted = True
            return increment
        except Exception:
            await _close_iterator(events)
            events = None
            if is_cancel_requested():
                cancelled_content = "".join(partial_content)
                self._repair_awaitable_cancelled(
                    runtime_messages,
                    increment,
                    partial_content,
                    pending_tool_calls,
                )
                if not started_emitted:
                    await _emit_agent_run_payload(emitter, AgentRunStartedPayload())
                    started_emitted = True
                await _emit_agent_run_payload(
                    emitter,
                    AgentRunCancelledPayload(partial_content=cancelled_content),
                )
                terminal_emitted = True
                return increment
            generic_failure = _model_failure()
            self._repair_awaitable_failed(
                runtime_messages,
                increment,
                partial_content,
                pending_tool_calls,
                stream=stream,
                failure=generic_failure,
            )
            if not started_emitted:
                await _emit_agent_run_payload(emitter, AgentRunStartedPayload())
                started_emitted = True
            await _emit_agent_run_payload(
                emitter,
                AgentRunFailedPayload(error=generic_failure.error, cause=generic_failure),
            )
            terminal_emitted = True
            return increment
        except asyncio.CancelledError:
            await _close_iterator(events)
            events = None
            if is_cancel_requested():
                cancelled_content = "".join(partial_content)
                self._repair_awaitable_cancelled(
                    runtime_messages,
                    increment,
                    partial_content,
                    pending_tool_calls,
                )
                if not started_emitted:
                    await _emit_agent_run_payload(emitter, AgentRunStartedPayload())
                    started_emitted = True
                await _emit_agent_run_payload(
                    emitter,
                    AgentRunCancelledPayload(partial_content=cancelled_content),
                )
                terminal_emitted = True
                return increment
            try:
                self._repair_awaitable_cancelled(
                    runtime_messages,
                    increment,
                    partial_content,
                    pending_tool_calls,
                )
            except BaseException:
                pass
            raise
        except BaseException:
            await _close_iterator(events)
            if not terminal_emitted:
                try:
                    self._repair_awaitable_cancelled(
                        runtime_messages,
                        increment,
                        partial_content,
                        pending_tool_calls,
                    )
                except BaseException:
                    pass
            raise

    async def _direct_complete_events(
        self,
        route: AgentRunRoute,
        messages: list[dict[str, Any]],
        tools: tuple[OpenAIToolSchema, ...],
    ) -> AsyncGenerator[ModelStreamEvent, None]:
        response = await self._model.complete(route, messages=messages, tools=tools)
        yield ModelCompleted(response=response)

    def _externalize_awaitable_result(
        self,
        result: ToolResult,
        externalize_result: ToolResultExternalizer,
    ) -> ToolResult:
        try:
            return externalize_result(result)
        except Exception as failure:
            if self._on_artifact_failure is not None:
                self._on_artifact_failure(failure, result.name)
            return ToolResult(
                tool_call_id=result.tool_call_id,
                name=result.name,
                status="error",
                content=f"{result.name} result could not be stored.",
                artifact=None,
                confirmation=result.confirmation,
            )

    def _repair_awaitable_cancelled(
        self,
        runtime_messages: list[dict[str, Any]],
        increment: list[dict[str, Any]],
        partial_content: list[str],
        pending_tool_calls: list[ModelToolCall],
    ) -> None:
        if partial_content:
            _append_run_message(
                runtime_messages,
                increment,
                build_assistant_repair_message(
                    content="".join(partial_content),
                    status="interrupted",
                    error=ErrorInfo(
                        code="turn_cancelled",
                        message="Turn interrupted by user.",
                    ),
                ),
            )
        for tool_call in pending_tool_calls:
            _append_run_message(
                runtime_messages,
                increment,
                _tool_run_message(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        status="error",
                        content="Tool call interrupted because the turn was cancelled.",
                        artifact=None,
                    )
                ),
            )
        pending_tool_calls.clear()
        partial_content.clear()

    def _repair_awaitable_failed(
        self,
        runtime_messages: list[dict[str, Any]],
        increment: list[dict[str, Any]],
        partial_content: list[str],
        pending_tool_calls: list[ModelToolCall],
        *,
        stream: bool,
        failure: ModelCallError,
    ) -> None:
        for tool_call in pending_tool_calls:
            _append_run_message(
                runtime_messages,
                increment,
                _tool_run_message(
                    ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        status="error",
                        content="Tool call interrupted because the Agent Run failed.",
                        artifact=None,
                    )
                ),
            )
        pending_tool_calls.clear()
        _append_run_message(
            runtime_messages,
            increment,
            build_assistant_repair_message(
                content="".join(partial_content) if stream else "",
                status="error",
                error=failure.error,
            ),
        )
        partial_content.clear()

    @staticmethod
    async def _call_tool(
        gateway: ToolGateway,
        tool_call: ModelToolCall,
        confirmation: ConfirmationChannel | None,
        state: _ToolCallState,
    ) -> AsyncGenerator[ConfirmationRequest | ToolResult, None]:
        operation = asyncio.create_task(gateway.call(tool_call, confirmation=confirmation))
        if confirmation is None:
            try:
                result = await operation
                state.result = result
                yield result
                return
            finally:
                if not operation.done():
                    operation.cancel()
                await asyncio.gather(operation, return_exceptions=True)

        notification = asyncio.create_task(confirmation.next_request())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {operation, notification},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if notification in done:
                    request = notification.result()
                    notification = asyncio.create_task(confirmation.next_request())
                    yield request
                    continue
                result = operation.result()
                state.result = result
                yield result
                return
        finally:
            if not operation.done():
                confirmation.close()
                operation.cancel()
            if not notification.done():
                notification.cancel()
            await asyncio.gather(operation, notification, return_exceptions=True)
            if state.result is None and operation.done() and not operation.cancelled():
                try:
                    state.result = operation.result()
                except BaseException:
                    pass


def _model_failure() -> ModelCallError:
    return ModelCallError(ErrorInfo("model_failed", "The model request failed."))


async def _emit_agent_run_payload(
    emitter: AgentRunEmitter,
    payload: AgentRunPayload,
) -> None:
    await emitter.emit(payload)


def _append_run_message(
    runtime_messages: list[dict[str, Any]],
    increment: list[dict[str, Any]],
    message: dict[str, Any],
) -> None:
    runtime_messages.append(deepcopy(message))
    increment.append(deepcopy(message))


def _assistant_run_message(response: ModelResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.message.content,
        "tool_calls": [call.to_dict() for call in response.message.tool_calls],
        "status": "completed",
        "error": None,
        "token_usage": {"model_calls": 1, **response.usage.to_dict()},
    }


def build_assistant_repair_message(
    *,
    content: str,
    status: Literal["interrupted", "error"],
    error: ErrorInfo,
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
        "status": status,
        "error": {"code": error.code, "message": error.message},
        "token_usage": {
            "model_calls": 1,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }


def _tool_run_message(result: ToolResult) -> dict[str, Any]:
    return {"role": "tool", **result.to_dict()}


def _identity_tool_result(result: ToolResult) -> ToolResult:
    return result


def _require_summary(value: str) -> None:
    if len(value) > 240:
        raise ValueError("summary must not exceed 240 characters")


def _tool_summary(action: str, tool_name: str) -> str:
    return " ".join(f"{action} {tool_name}".split())[:240]


def _tool_completion_summary(result: ToolResult) -> str:
    if result.status == "error":
        return " ".join(result.content.split())[:240]
    return _tool_summary("Finished", result.name)


async def _close_iterator(iterator: AsyncIterator[object] | None) -> None:
    if iterator is None:
        return
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except BaseException:
        pass


def _log_artifact_failure(failure: Exception, *, tool_name: str) -> None:
    logger.opt(exception=failure).error(
        "Tool Artifact persistence failed code=persistence_error tool={} type={}",
        tool_name,
        type(failure).__name__,
    )


__all__ = [
    "AgentRun",
    "AgentRunCancelledPayload",
    "AgentRunCompletedPayload",
    "AgentRunConfirmationRequestedPayload",
    "AgentRunContinuationPreparer",
    "AgentRunEmitter",
    "AgentRunFailedPayload",
    "AgentRunModelCallCompletedPayload",
    "AgentRunPayload",
    "AgentRunRoute",
    "AgentRunRouter",
    "AgentRunStartedPayload",
    "AgentRunTextDeltaPayload",
    "AgentRunToolCompletedPayload",
    "AgentRunToolStartedPayload",
    "build_assistant_repair_message",
]
