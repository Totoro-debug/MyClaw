"""Lane-neutral Agent Run contract and Runtime Core execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, ClassVar, Literal, Protocol
from uuid import UUID

from myclaw.agent.prompts import current_user_input
from myclaw.agent.turn import model_message_from_session
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelCompleted,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
    UserModelMessage,
)
from myclaw.session.session import Session
from myclaw.tools.confirmation import (
    ConfirmationRequest,
    ConfirmationRequester,
    ToolConfirmationChannel,
)
from myclaw.tools.models import ModelToolCall, ToolResult, ToolResultStatus
from myclaw.tools.schema import OpenAIToolSchema
from myclaw.tools.tool_gateway import ToolGateway

type AgentRunRoute = Literal["chat", "schedule"]
type ConfirmationChannel = ToolConfirmationChannel | ConfirmationRequester


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
    | AgentRunCompletedPayload
    | AgentRunFailedPayload
    | AgentRunCancelledPayload
)


@dataclass(frozen=True, slots=True)
class AgentRunModelSettings:
    """Provider-neutral budget and model settings for one Agent Run route."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int
    context_window: int = 0


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
        provider: ModelProvider | Mapping[AgentRunRoute, ModelProvider],
        settings: AgentRunModelSettings | Mapping[AgentRunRoute, AgentRunModelSettings],
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str = "",
        tool_gateway: ToolGateway | None = None,
        tool_gateway_for: Callable[[Session], ToolGateway] | None = None,
        externalize_result: Callable[[ToolResult], ToolResult] | None = None,
        externalize_result_for: Callable[[Session], Callable[[ToolResult], ToolResult]]
        | None = None,
        memory_snapshot: Callable[[], str] | None = None,
        system_prompt_for_memory: Callable[[str], str] | None = None,
        summary_preparer_for_route: (
            Callable[
                [
                    Session,
                    AgentRunRoute,
                    int,
                    int,
                    str,
                    tuple[OpenAIToolSchema, ...],
                ],
                Awaitable[Session],
            ]
            | None
        ) = None,
        summary_preparer: Callable[[Session, AgentRunRoute], Awaitable[Session]] | None = None,
        history_preparer: Callable[[Session], Awaitable[Session]] | None = None,
        history_preparer_for_route: (
            Callable[[AgentRunRoute], Callable[[Session], Awaitable[Session]] | None] | None
        ) = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        if tool_gateway is not None and tool_gateway_for is not None:
            raise ValueError("Provide only one Tool Gateway source")
        if summary_preparer_for_route is not None and (
            summary_preparer is not None
            or history_preparer is not None
            or history_preparer_for_route is not None
        ):
            raise ValueError("Provide only one Summary preparer source")
        if summary_preparer is not None and (
            history_preparer is not None or history_preparer_for_route is not None
        ):
            raise ValueError("Provide only one Summary preparer source")
        self._provider = provider
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._system_prompt = system_prompt
        self._tool_gateway = tool_gateway
        self._tool_gateway_for = tool_gateway_for
        self._externalize_result = externalize_result
        self._externalize_result_for = externalize_result_for
        self._memory_snapshot = memory_snapshot
        self._system_prompt_for_memory = system_prompt_for_memory
        self._summary_preparer_for_route = summary_preparer_for_route
        self._summary_preparer = summary_preparer
        self._history_preparer = history_preparer
        self._history_preparer_for_route = history_preparer_for_route
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

        confirmation_requests: asyncio.Queue[ConfirmationRequest] = asyncio.Queue()
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
            if self._memory_snapshot is not None:
                memory = self._memory_snapshot()
                if self._system_prompt_for_memory is None:
                    raise RuntimeError("Memory snapshot requires a System Prompt factory")
                system_prompt = self._system_prompt_for_memory(memory)
            base_gateway = (
                self._tool_gateway_for(session)
                if self._tool_gateway_for is not None
                else self._tool_gateway
            )
            gateway = (
                None
                if base_gateway is None
                else base_gateway.for_run(
                    confirmation=confirmation,
                    on_confirmation_requested=confirmation_requests.put_nowait,
                )
            )
            frozen_tools = () if gateway is None else gateway.schemas
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
                    settings,
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
                    partial_content.clear()
                    await _close_iterator(events)
                    events = None
                    if response.message.tool_calls and gateway is not None:
                        pending_tool_calls = list(response.message.tool_calls)
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
                                        confirmation_requests,
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
                                summary=_tool_summary("Finished", result.name),
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

    def _route_provider(self, route: AgentRunRoute) -> ModelProvider:
        if isinstance(self._provider, Mapping):
            return self._provider[route]
        return self._provider

    def _route_settings(self, route: AgentRunRoute) -> AgentRunModelSettings:
        if isinstance(self._settings, Mapping):
            return self._settings[route]
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
        settings: AgentRunModelSettings,
        system_prompt: str,
        tools: tuple[OpenAIToolSchema, ...],
    ) -> Session:
        if self._summary_preparer_for_route is not None:
            return await self._summary_preparer_for_route(
                session,
                route,
                settings.context_window,
                settings.max_output,
                system_prompt,
                tools,
            )
        if self._summary_preparer is not None:
            return await self._summary_preparer(session, route)
        preparer = self._history_preparer
        if self._history_preparer_for_route is not None:
            preparer = self._history_preparer_for_route(route)
        if preparer is None:
            return session
        return await preparer(session)

    @staticmethod
    async def _call_tool(
        gateway: ToolGateway,
        tool_call: ModelToolCall,
        confirmation_requests: asyncio.Queue[ConfirmationRequest],
        state: _ToolCallState,
    ) -> AsyncGenerator[ConfirmationRequest | ToolResult, None]:
        operation = asyncio.create_task(gateway.call(tool_call))
        notification = asyncio.create_task(confirmation_requests.get())
        try:
            while True:
                done, _ = await asyncio.wait(
                    {operation, notification},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if notification in done:
                    request = notification.result()
                    notification = asyncio.create_task(confirmation_requests.get())
                    yield request
                    continue
                result = operation.result()
                state.result = result
                yield result
                return
        finally:
            if not operation.done():
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
        except Exception:
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


def _identity_tool_result(result: ToolResult) -> ToolResult:
    return result


def _require_summary(value: str) -> None:
    if len(value) > 240:
        raise ValueError("summary must not exceed 240 characters")


def _tool_summary(action: str, tool_name: str) -> str:
    return " ".join(f"{action} {tool_name}".split())[:240]


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


__all__ = [
    "AgentRun",
    "AgentRunCancelledPayload",
    "AgentRunCompletedPayload",
    "AgentRunConfirmationRequestedPayload",
    "AgentRunFailedPayload",
    "AgentRunInterface",
    "AgentRunModelSettings",
    "AgentRunPayload",
    "AgentRunRoute",
    "AgentRunStartedPayload",
    "AgentRunTextDeltaPayload",
    "AgentRunToolCompletedPayload",
    "AgentRunToolStartedPayload",
]
