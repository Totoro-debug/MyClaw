"""Lane-neutral Agent Run contract and Runtime Core execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal, Protocol, cast, runtime_checkable
from uuid import UUID, uuid4

from loguru import logger

from myclaw.agent.prompts import current_user_input, interrupted_assistant_content
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    DirectModelProvider,
    ModelCompleted,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
    ToolModelMessage,
    UserModelMessage,
    accepts_direct_provider_call,
    legacy_request_from_direct,
)
from myclaw.session.session import Session
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
type SummaryPreparer = Callable[
    [Session, AgentRunRoute, str, tuple[OpenAIToolSchema, ...]],
    Awaitable[Session],
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


@runtime_checkable
class AgentRunRouter(Protocol):
    """Direct-call Model Router boundary used by the awaitable Agent Run."""

    def route_status(self, route: AgentRunRoute) -> object: ...

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


type AgentRunProvider = ModelProvider | DirectModelProvider
type _AgentRunModelClient = AgentRunProvider | AgentRunRouter


@dataclass(frozen=True, slots=True)
class AgentRunModelSettings:
    """Provider-neutral budget and model settings for one Agent Run route."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


@dataclass(slots=True)
class _ToolCallState:
    result: ToolResult | None = None


class AgentRunInterface(Protocol):
    """Submit one complete Agent Run without exposing Session ownership."""

    def run_agent(
        self,
        session: Session,
        input: str,
        route: AgentRunRoute,
        stream: bool,
        confirmation: ConfirmationChannel | None = None,
    ) -> AsyncIterator[AgentRunPayload]: ...


class AgentRun:
    """Execute the shared model and Tool loop for foreground and Schedule callers."""

    def __init__(
        self,
        *,
        provider: _AgentRunModelClient | Mapping[AgentRunRoute, AgentRunProvider],
        settings: AgentRunModelSettings
        | Mapping[AgentRunRoute, AgentRunModelSettings]
        | None = None,
        now: Callable[[], datetime] | None = None,
        new_uuid: Callable[[], UUID] | None = None,
        system_prompt: str = "",
        tool_gateway: ToolGateway | None = None,
        externalize_result: Callable[[ToolResult], ToolResult] | None = None,
        externalize_result_for: Callable[[Session], Callable[[ToolResult], ToolResult]]
        | None = None,
        memory_snapshot: Callable[[], str] | None = None,
        system_prompt_for_memory: Callable[[str], str] | None = None,
        summary_preparer: SummaryPreparer | None = None,
        after_user_published: Callable[[Session], None] | None = None,
        on_terminal_failure: Callable[[BaseException], None] | None = None,
        on_artifact_failure: Callable[[Exception, str], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._provider = provider
        self._settings = settings
        self._now = datetime.now().astimezone if now is None else now
        self._new_uuid = uuid4 if new_uuid is None else new_uuid
        self._system_prompt = system_prompt
        self._tool_gateway = tool_gateway
        self._externalize_result = externalize_result
        self._externalize_result_for = externalize_result_for
        self._memory_snapshot = memory_snapshot
        self._system_prompt_for_memory = system_prompt_for_memory
        self._summary_preparer = summary_preparer
        self._after_user_published = after_user_published
        self._on_terminal_failure = on_terminal_failure
        self._on_artifact_failure = on_artifact_failure
        self._cancel_requested = cancel_requested or (lambda: False)

    def run_agent(
        self,
        session: Session,
        input: str,
        route: AgentRunRoute,
        stream: bool,
        confirmation: ConfirmationChannel | None = None,
    ) -> AsyncGenerator[AgentRunPayload, None]:
        return self._run_agent(
            session,
            input,
            route=route,
            stream=stream,
            confirmation=confirmation,
        )

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
            model_client = self._model_client(route)
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
                    self._direct_stream(
                        model_client,
                        route,
                        runtime_messages,
                        frozen_tools,
                    )
                    if stream
                    else self._direct_complete_events(
                        model_client,
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
            self._capture_terminal_failure(failure)
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
            await _emit_agent_run_payload(emitter, AgentRunFailedPayload(error=failure.error))
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
            self._capture_terminal_failure(generic_failure)
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
                AgentRunFailedPayload(error=generic_failure.error),
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

    def _direct_stream(
        self,
        model_client: _AgentRunModelClient,
        route: AgentRunRoute,
        messages: list[dict[str, Any]],
        tools: tuple[OpenAIToolSchema, ...],
    ) -> AsyncIterator[ModelStreamEvent]:
        if isinstance(model_client, AgentRunRouter):
            return model_client.stream(
                route,
                messages=messages,
                tools=tools,
            )
        settings = self._route_settings(route)
        method = cast(Any, model_client).stream
        if accepts_direct_provider_call(method):
            return cast(DirectModelProvider, model_client).stream(
                messages=messages,
                tools=tools,
                model=settings.model,
                max_output=settings.max_output,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                timeout=settings.timeout_seconds,
            )
        return cast(ModelProvider, model_client).stream(
            legacy_request_from_direct(
                route=route,
                messages=messages,
                tools=tools,
                model=settings.model,
                max_output=settings.max_output,
                temperature=settings.temperature,
                reasoning_effort=settings.reasoning_effort,
                timeout=settings.timeout_seconds,
                stream=True,
            )
        )

    async def _direct_complete_events(
        self,
        model_client: _AgentRunModelClient,
        route: AgentRunRoute,
        messages: list[dict[str, Any]],
        tools: tuple[OpenAIToolSchema, ...],
    ) -> AsyncGenerator[ModelStreamEvent, None]:
        if isinstance(model_client, AgentRunRouter):
            response = await model_client.complete(
                route,
                messages=messages,
                tools=tools,
            )
        else:
            settings = self._route_settings(route)
            method = cast(Any, model_client).complete
            if accepts_direct_provider_call(method):
                response = await cast(DirectModelProvider, model_client).complete(
                    messages=messages,
                    tools=tools,
                    model=settings.model,
                    max_output=settings.max_output,
                    temperature=settings.temperature,
                    reasoning_effort=settings.reasoning_effort,
                    timeout=settings.timeout_seconds,
                )
            else:
                response = await cast(ModelProvider, model_client).complete(
                    legacy_request_from_direct(
                        route=route,
                        messages=messages,
                        tools=tools,
                        model=settings.model,
                        max_output=settings.max_output,
                        temperature=settings.temperature,
                        reasoning_effort=settings.reasoning_effort,
                        timeout=settings.timeout_seconds,
                        stream=False,
                    )
                )
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
                _assistant_repair_message(
                    content="".join(partial_content),
                    status="interrupted",
                    error={
                        "code": "turn_cancelled",
                        "message": "Turn interrupted by user.",
                    },
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
            _assistant_repair_message(
                content="".join(partial_content) if stream else "",
                status="error",
                error={"code": failure.error.code, "message": failure.error.message},
            ),
        )
        partial_content.clear()

    async def _run_agent(
        self,
        session: Session,
        input: str,
        *,
        route: AgentRunRoute,
        stream: bool,
        confirmation: ConfirmationChannel | None,
    ) -> AsyncGenerator[AgentRunPayload, None]:
        if route not in {"chat", "schedule"}:
            raise ValueError("Agent Run route must be chat or schedule")
        if stream != (route == "chat"):
            raise ValueError("chat Agent Runs must stream and schedule Agent Runs must not stream")

        partial_content: list[str] = []
        pending_tool_calls: list[ModelToolCall] = []
        persisted = False
        events: AsyncIterator[ModelStreamEvent] | None = None
        started_emitted = False
        terminal_emitted = False
        user_published = False

        try:
            settings = self._route_settings(route)
            provider = self._route_provider(route)
            system_prompt = self._system_prompt
            memory: str | None = None
            if self._memory_snapshot is not None:
                memory = self._memory_snapshot()
                if self._system_prompt_for_memory is None:
                    raise RuntimeError("Memory snapshot requires a System Prompt factory")
                system_prompt = self._system_prompt_for_memory(memory)
            gateway = self._tool_gateway
            frozen_tools = () if gateway is None else tuple(gateway.schemas)
            externalize_result = (
                self._externalize_result_for(session)
                if self._externalize_result_for is not None
                else self._externalize_result
            )
            if externalize_result is None:
                externalize_result = _identity_tool_result
            yield AgentRunStartedPayload()
            started_emitted = True
            session.add_message("user", input)
            user_published = True
            if self._after_user_published is not None:
                self._after_user_published(session)
            current_user = session.messages[-1]
            if self._cancel_requested():
                cancelled_content = "".join(partial_content)
                persisted = self._repair_cancelled(
                    session, partial_content, pending_tool_calls, persisted=persisted
                )
                terminal_emitted = True
                yield AgentRunCancelledPayload(partial_content=cancelled_content)
                return
            while True:
                partial_content.clear()
                prepared_session = await self._prepare_summary(
                    session,
                    route,
                    system_prompt,
                    frozen_tools,
                )
                if prepared_session is not session:
                    raise RuntimeError("Conversation Summary replaced the active Session")
                request = self._request(
                    session,
                    current_user,
                    route=route,
                    stream=stream,
                    settings=settings,
                    tools=frozen_tools,
                    system_prompt=system_prompt,
                )
                if stream:
                    events = self._stream(provider, request)
                else:
                    events = self._complete(provider, request)
                model_completed = False
                async for event in events:
                    if isinstance(event, TextDelta):
                        partial_content.append(event.delta)
                        yield AgentRunTextDeltaPayload(delta=event.delta)
                        if self._cancel_requested():
                            cancelled_content = "".join(partial_content)
                            await _close_iterator(events)
                            events = None
                            persisted = self._repair_cancelled(
                                session,
                                partial_content,
                                pending_tool_calls,
                                persisted=persisted,
                            )
                            terminal_emitted = True
                            yield AgentRunCancelledPayload(partial_content=cancelled_content)
                            return
                        continue
                    if not isinstance(event, ModelCompleted):
                        raise _model_failure()
                    model_completed = True
                    response = event.response
                    session.add_message(
                        "assistant",
                        response.message.content,
                        tool_calls=[call.to_dict() for call in response.message.tool_calls],
                        status="completed",
                        error=None,
                        token_usage={"model_calls": 1, **response.usage.to_dict()},
                    )
                    continues_with_tools = bool(response.message.tool_calls and gateway is not None)
                    partial_content.clear()
                    if continues_with_tools:
                        pending_tool_calls = list(response.message.tool_calls)
                    await _close_iterator(events)
                    events = None
                    yield AgentRunModelCallCompletedPayload(
                        content=response.message.content,
                        continues_with_tools=continues_with_tools,
                    )
                    if continues_with_tools:
                        assert gateway is not None
                        for tool_call in response.message.tool_calls:
                            yield AgentRunToolStartedPayload(
                                tool_call_id=tool_call.id,
                                tool_name=tool_call.name,
                                summary=_tool_summary("Running", tool_call.name),
                            )
                            if self._cancel_requested():
                                cancelled_content = "".join(partial_content)
                                persisted = self._repair_cancelled(
                                    session,
                                    partial_content,
                                    pending_tool_calls,
                                    persisted=persisted,
                                )
                                terminal_emitted = True
                                yield AgentRunCancelledPayload(partial_content=cancelled_content)
                                return
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
                                                yield AgentRunConfirmationRequestedPayload(
                                                    request=tool_outcome
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
                                            self._record_tool_result(
                                                session,
                                                tool_state.result,
                                                externalize_result,
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
                            result = self._record_tool_result(
                                session,
                                result,
                                externalize_result,
                            )
                            pending_tool_calls.pop(0)
                            yield AgentRunToolCompletedPayload(
                                tool_call_id=result.tool_call_id,
                                tool_name=result.name,
                                status=result.status,
                                summary=_tool_completion_summary(result),
                            )
                            if self._cancel_requested():
                                cancelled_content = "".join(partial_content)
                                persisted = self._repair_cancelled(
                                    session,
                                    partial_content,
                                    pending_tool_calls,
                                    persisted=persisted,
                                )
                                terminal_emitted = True
                                yield AgentRunCancelledPayload(partial_content=cancelled_content)
                                return
                        break
                    persisted = self._request_persist(session, persisted=persisted)
                    terminal_emitted = True
                    yield AgentRunCompletedPayload(
                        content=response.message.content,
                        usage=response.usage,
                    )
                    return
                if not model_completed:
                    raise _model_failure()
        except ModelCallError as failure:
            await _close_iterator(events)
            events = None
            if failure.error.code == "turn_cancelled":
                cancelled_content = "".join(partial_content)
                persisted = self._repair_cancelled(
                    session,
                    partial_content,
                    pending_tool_calls,
                    persisted=persisted,
                )
                if not started_emitted:
                    yield AgentRunStartedPayload()
                    started_emitted = True
                terminal_emitted = True
                yield AgentRunCancelledPayload(partial_content=cancelled_content)
                return
            self._capture_terminal_failure(failure)
            self._repair_failed(session, pending_tool_calls)
            if user_published:
                self._safe_add_failed_assistant(session, partial_content, stream, failure)
            partial_content.clear()
            persisted = self._request_persist(session, persisted=persisted)
            if not started_emitted:
                yield AgentRunStartedPayload()
                started_emitted = True
            terminal_emitted = True
            yield AgentRunFailedPayload(error=failure.error)
            return
        except Exception:
            await _close_iterator(events)
            events = None
            generic_failure = _model_failure()
            self._capture_terminal_failure(generic_failure)
            self._repair_failed(session, pending_tool_calls)
            if user_published:
                self._safe_add_failed_assistant(session, partial_content, stream, generic_failure)
            partial_content.clear()
            persisted = self._request_persist(session, persisted=persisted)
            if not started_emitted:
                yield AgentRunStartedPayload()
                started_emitted = True
            terminal_emitted = True
            yield AgentRunFailedPayload(error=generic_failure.error)
            return
        except asyncio.CancelledError:
            await _close_iterator(events)
            events = None
            if self._cancel_requested():
                cancelled_content = "".join(partial_content)
                persisted = self._repair_cancelled(
                    session,
                    partial_content,
                    pending_tool_calls,
                    persisted=persisted,
                )
                if not started_emitted:
                    yield AgentRunStartedPayload()
                    started_emitted = True
                terminal_emitted = True
                yield AgentRunCancelledPayload(partial_content=cancelled_content)
                return
            try:
                self._repair_cancelled(
                    session,
                    partial_content,
                    pending_tool_calls,
                    persisted=persisted,
                )
            except BaseException:
                pass
            raise
        except BaseException:
            await _close_iterator(events)
            if not terminal_emitted:
                try:
                    self._repair_cancelled(
                        session,
                        partial_content,
                        pending_tool_calls,
                        persisted=persisted,
                    )
                except BaseException:
                    pass
            raise

    def _capture_terminal_failure(self, failure: BaseException) -> None:
        if self._on_terminal_failure is not None:
            self._on_terminal_failure(failure)

    def _route_provider(self, route: AgentRunRoute) -> ModelProvider:
        return cast(ModelProvider, self._model_client(route))

    def _model_client(self, route: AgentRunRoute) -> _AgentRunModelClient:
        if isinstance(self._provider, Mapping):
            return self._provider[route]
        return self._provider

    def _route_settings(self, route: AgentRunRoute) -> AgentRunModelSettings:
        if isinstance(self._settings, Mapping):
            return self._settings[route]
        if self._settings is None:
            raise TypeError("Agent Run model settings are required for this Provider")
        return self._settings

    def _request(
        self,
        session: Session,
        current_user: dict[str, Any],
        *,
        route: AgentRunRoute,
        stream: bool,
        settings: AgentRunModelSettings,
        tools: tuple[OpenAIToolSchema, ...],
        system_prompt: str,
    ) -> ModelRequest:
        messages: list[ModelMessage] = []
        for message in session.messages[session.last_consolidated :]:
            if message is current_user:
                timestamp = message.get("timestamp")
                content = message.get("content")
                if not isinstance(timestamp, str) or not isinstance(content, str):
                    raise TypeError("Session user message is malformed")
                messages.append(
                    UserModelMessage(
                        content=current_user_input(
                            content=content,
                            current_time=datetime.fromisoformat(timestamp),
                            session_id=session.session_id,
                        )
                    )
                )
                continue
            model_message = model_message_from_session(message)
            if model_message is not None:
                messages.append(model_message)
        return ModelRequest(
            request_id=self._new_uuid(),
            route=route,
            system_prompt=system_prompt,
            messages=tuple(messages),
            tools=tools,
            stream=stream,
            model=settings.model,
            max_output=settings.max_output,
            temperature=settings.temperature,
            reasoning_effort=settings.reasoning_effort,
            timeout_seconds=settings.timeout_seconds,
        )

    async def _prepare_summary(
        self,
        session: Session,
        route: AgentRunRoute,
        system_prompt: str,
        tools: tuple[OpenAIToolSchema, ...],
    ) -> Session:
        if self._summary_preparer is None:
            return session
        return await self._summary_preparer(session, route, system_prompt, tools)

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

    async def _complete(
        self,
        provider: ModelProvider,
        request: ModelRequest,
    ) -> AsyncGenerator[ModelStreamEvent, None]:
        yield ModelCompleted(response=await provider.complete(request))

    async def _stream(
        self,
        provider: ModelProvider,
        request: ModelRequest,
    ) -> AsyncGenerator[ModelStreamEvent, None]:
        stream: AsyncIterator[ModelStreamEvent] | None = None
        try:
            stream = provider.stream(request)
            async for event in stream:
                yield event
        except ModelCallError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            raise _model_failure() from None
        finally:
            if stream is not None:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except Exception:
                        pass

    def _request_persist(self, session: Session, *, persisted: bool) -> bool:
        if persisted:
            return True
        try:
            session.persist()
        except Exception:
            return True
        return True

    def _record_tool_result(
        self,
        session: Session,
        result: ToolResult,
        externalize_result: Callable[[ToolResult], ToolResult],
    ) -> ToolResult:
        try:
            result = externalize_result(result)
        except Exception as failure:
            if self._on_artifact_failure is not None:
                self._on_artifact_failure(failure, result.name)
            result = ToolResult(
                tool_call_id=result.tool_call_id,
                name=result.name,
                status="error",
                content=f"{result.name} result could not be stored.",
                artifact=None,
                confirmation=result.confirmation,
            )
        self._add_tool_message(session, result)
        return result

    def _repair_cancelled(
        self,
        session: Session,
        partial_content: list[str],
        pending_tool_calls: list[ModelToolCall],
        *,
        persisted: bool,
    ) -> bool:
        if partial_content:
            try:
                session.add_message(
                    "assistant",
                    "".join(partial_content),
                    tool_calls=[],
                    status="interrupted",
                    error={
                        "code": "turn_cancelled",
                        "message": "Turn interrupted by user.",
                    },
                    token_usage={
                        "model_calls": 1,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                )
            except Exception:
                pass
        for tool_call in pending_tool_calls:
            try:
                self._add_tool_message(
                    session,
                    ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        status="error",
                        content="Tool call interrupted because the turn was cancelled.",
                        artifact=None,
                    ),
                )
            except Exception:
                pass
        pending_tool_calls.clear()
        partial_content.clear()
        return self._request_persist(session, persisted=persisted)

    def _repair_failed(
        self,
        session: Session,
        pending_tool_calls: list[ModelToolCall],
    ) -> None:
        for tool_call in pending_tool_calls:
            try:
                self._add_tool_message(
                    session,
                    ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        status="error",
                        content="Tool call interrupted because the Agent Run failed.",
                        artifact=None,
                    ),
                )
            except Exception:
                pass
        pending_tool_calls.clear()

    @staticmethod
    def _safe_add_failed_assistant(
        session: Session,
        partial_content: list[str],
        stream: bool,
        failure: ModelCallError,
    ) -> None:
        try:
            session.add_message(
                "assistant",
                "".join(partial_content) if stream else "",
                tool_calls=[],
                status="error",
                error={"code": failure.error.code, "message": failure.error.message},
                token_usage={
                    "model_calls": 1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )
        except Exception:
            pass

    @staticmethod
    def _add_tool_message(session: Session, result: ToolResult) -> None:
        fields: dict[str, object] = {
            "tool_call_id": result.tool_call_id,
            "name": result.name,
            "status": result.status,
            "artifact": None if result.artifact is None else result.artifact.to_dict(),
        }
        if result.confirmation is not None:
            fields["confirmation"] = result.confirmation.to_dict()
        session.add_message("tool", result.content, **fields)


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


def _assistant_repair_message(
    *,
    content: str,
    status: Literal["interrupted", "error"],
    error: dict[str, str],
) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [],
        "status": status,
        "error": error,
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


def model_message_from_session(
    message: dict[str, Any],
) -> UserModelMessage | AssistantModelMessage | ToolModelMessage | None:
    """Project persisted conversation history into the next provider request."""
    role = message.get("role")
    if role == "user":
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError("Session user content must be a string")
        return UserModelMessage(content=content)
    if role == "assistant":
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        status = message.get("status")
        if not isinstance(content, str) or not isinstance(tool_calls, list):
            raise TypeError("Session assistant message is malformed")
        projected_tool_calls = tuple(
            ModelToolCall(
                id=tool_call["id"],
                name=tool_call["name"],
                arguments=tool_call["arguments"],
            )
            for tool_call in tool_calls
        )
        if status == "error" and not content and not projected_tool_calls:
            return None
        if status == "interrupted":
            return AssistantModelMessage(
                content=interrupted_assistant_content(content),
                tool_calls=projected_tool_calls,
            )
        return AssistantModelMessage(content=content, tool_calls=projected_tool_calls)
    if role == "tool":
        tool_call_id = message.get("tool_call_id")
        name = message.get("name")
        content = message.get("content")
        if not all(isinstance(value, str) for value in (tool_call_id, name, content)):
            raise TypeError("Session tool message is malformed")
        return ToolModelMessage(
            tool_call_id=cast(str, tool_call_id),
            name=cast(str, name),
            content=cast(str, content),
        )
    raise TypeError("Unsupported Session message role")


__all__ = [
    "AgentRun",
    "AgentRunCancelledPayload",
    "AgentRunCompletedPayload",
    "AgentRunConfirmationRequestedPayload",
    "AgentRunContinuationPreparer",
    "AgentRunEmitter",
    "AgentRunFailedPayload",
    "AgentRunInterface",
    "AgentRunModelCallCompletedPayload",
    "AgentRunModelSettings",
    "AgentRunPayload",
    "AgentRunProvider",
    "AgentRunRoute",
    "AgentRunRouter",
    "AgentRunStartedPayload",
    "AgentRunTextDeltaPayload",
    "AgentRunToolCompletedPayload",
    "AgentRunToolStartedPayload",
]
