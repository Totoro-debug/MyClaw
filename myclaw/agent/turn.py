"""Runtime Core orchestration for one foreground or Scheduled Work Agent turn."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Any, Literal, Protocol, cast
from uuid import UUID

from loguru import logger

from myclaw.agent.events import (
    AgentEventType,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.agent.prompts import current_user_input, interrupted_assistant_content
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
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
)
from myclaw.session.session import Session
from myclaw.tools.models import ModelToolCall, ToolResult
from myclaw.tools.tool_artifacts import ArtifactWriteError
from myclaw.tools.tool_gateway import ToolGateway

type AgentTurnLane = Literal["foreground", "scheduled_work"]
type ToolResultExternalizer = Callable[[ToolResult], ToolResult]
type AgentTurnPayload = (
    TurnStartedPayload
    | TextDeltaPayload
    | ToolStartedPayload
    | ToolCompletedPayload
    | TurnCompletedPayload
    | TurnFailedPayload
    | TurnCancelledPayload
)

_PERSISTENCE_EXCEPTIONS = (OSError, UnicodeError, ValueError)


class AgentTurnModelSettings(Protocol):
    @property
    def model(self) -> str: ...

    @property
    def max_output(self) -> int: ...

    @property
    def temperature(self) -> float: ...

    @property
    def reasoning_effort(self) -> ReasoningEffort | None: ...

    @property
    def timeout_seconds(self) -> int: ...


def agent_turn_event_type(payload: AgentTurnPayload) -> AgentEventType:
    """Return the existing Agent Event type for one Runtime Core payload."""
    if isinstance(payload, TurnStartedPayload):
        return "turn_started"
    if isinstance(payload, TextDeltaPayload):
        return "text_delta"
    if isinstance(payload, ToolStartedPayload):
        return "tool_started"
    if isinstance(payload, ToolCompletedPayload):
        return "tool_completed"
    if isinstance(payload, TurnCompletedPayload):
        return "turn_completed"
    if isinstance(payload, TurnFailedPayload):
        return "turn_failed"
    if isinstance(payload, TurnCancelledPayload):
        return "turn_cancelled"
    raise TypeError("Unsupported Agent turn payload")


class AgentTurn:
    """Coordinate one Agent turn behind a lane-neutral payload interface."""

    def __init__(
        self,
        *,
        lane: AgentTurnLane,
        provider: ModelProvider,
        session: Session,
        settings: AgentTurnModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str,
        tool_gateway: ToolGateway | None,
        history_preparer: Callable[[Any], Awaitable[Any]] | None = None,
        after_user_published: Callable[[Session], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        externalize_result: ToolResultExternalizer | None = None,
        on_terminal_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._lane = lane
        self._provider = provider
        self._session = session
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._system_prompt = system_prompt
        self._tool_gateway = tool_gateway
        self._history_preparer = history_preparer
        self._after_user_published = after_user_published
        self._cancel_requested = cancel_requested or _never_cancelled
        self._externalize_result = externalize_result or _inline_tool_result
        self._on_terminal_failure = on_terminal_failure
        self._session_persist_requested = False

    async def run(self, text: str) -> AsyncGenerator[AgentTurnPayload, None]:
        self._session_persist_requested = False
        yield TurnStartedPayload()

        self._session.add_message("user", text)
        current_user = self._session.messages[-1]
        if self._lane == "foreground" and self._cancel_requested():
            yield await self._cancelled_payload([], [])
            return
        if self._after_user_published is not None:
            self._after_user_published(self._session)

        partial_content: list[str] = []
        pending_tool_calls: list[ModelToolCall] = []
        pending_repair_error: dict[str, str] | None = None
        provider_stream: AsyncIterator[ModelStreamEvent] | None = None
        try:
            while True:
                partial_content = []
                if self._history_preparer is not None:
                    prepared = await self._history_preparer(self._session)
                    if not isinstance(prepared, Session):
                        raise TypeError("Session history preparer must return a Session")
                    self._session = prepared
                request = self._session_model_request(current_user)
                if self._lane == "foreground":
                    cleanup_failure = await _close_provider_stream(provider_stream)
                    if cleanup_failure is not None:
                        _log_cleanup_failure(cleanup_failure)
                    try:
                        provider_stream = self._provider.stream(request)
                    except ModelCallError:
                        raise
                    except Exception:
                        raise _unexpected_provider_failure() from None
                    model_events = _provider_events(provider_stream)
                else:
                    model_events = self._completion_events(request)

                async for model_event in model_events:
                    if isinstance(model_event, TextDelta):
                        partial_content.append(model_event.delta)
                        yield TextDeltaPayload(delta=model_event.delta)
                        if self._lane == "foreground" and self._cancel_requested():
                            yield await self._cancelled_payload(
                                partial_content,
                                pending_tool_calls,
                            )
                            return
                        continue
                    if not isinstance(model_event, ModelCompleted):
                        raise _unexpected_provider_failure()

                    response = model_event.response
                    try:
                        self._add_session_assistant(response)
                    except ValueError:
                        if self._lane == "foreground":
                            raise _unexpected_provider_failure() from None
                        raise

                    partial_content = []
                    if response.message.tool_calls and self._tool_gateway is not None:
                        pending_tool_calls = list(response.message.tool_calls)
                        for index, tool_call in enumerate(response.message.tool_calls):
                            yield ToolStartedPayload(
                                tool_call_id=tool_call.id,
                                tool_name=tool_call.name,
                                summary=_tool_activity_summary("Running", tool_call.name),
                            )
                            if self._lane == "foreground" and self._cancel_requested():
                                yield await self._cancelled_payload(
                                    partial_content,
                                    pending_tool_calls,
                                )
                                return
                            try:
                                raw_result = await self._tool_gateway.call(tool_call)
                            except asyncio.CancelledError:
                                if self._lane == "scheduled_work":
                                    await self._repair_scheduled_cancellation(
                                        response.message.tool_calls[index:]
                                    )
                                raise
                            try:
                                result = self._externalize_result(raw_result)
                            except ArtifactWriteError as failure:
                                if self._lane == "foreground":
                                    _log_artifact_failure(failure, tool_name=tool_call.name)
                                message = f"{tool_call.name} result could not be stored."
                                result = ToolResult(
                                    tool_call_id=tool_call.id,
                                    name=tool_call.name,
                                    status="error",
                                    content=message,
                                    artifact=None,
                                )

                            self._add_session_tool(result)
                            pending_tool_calls.pop(0)
                            yield ToolCompletedPayload(
                                tool_call_id=result.tool_call_id,
                                tool_name=result.name,
                                status=result.status,
                                summary=_tool_activity_summary("Finished", result.name),
                            )
                            if self._lane == "foreground" and self._cancel_requested():
                                yield await self._cancelled_payload(
                                    partial_content,
                                    pending_tool_calls,
                                )
                                return
                        break
                    self._persist_session_once()
                    yield TurnCompletedPayload(
                        content=response.message.content,
                        usage=response.usage,
                    )
                    return
                else:
                    raise ModelCallError(
                        ErrorInfo(
                            code="model_failed",
                            message="The model stream ended without a complete response.",
                        )
                    )
        except ModelCallError as failure:
            if self._lane == "foreground" and failure.error.code == "turn_cancelled":
                yield await self._cancelled_payload(
                    partial_content,
                    pending_tool_calls,
                )
                return
            terminal_error = failure.error
            self._add_session_assistant(
                content="".join(partial_content) if self._lane == "foreground" else "",
                status="error",
                error={"code": failure.error.code, "message": failure.error.message},
                usage=_zero_assistant_usage(),
            )
            if self._lane == "foreground":
                _log_model_failure(failure)
            else:
                self._capture_terminal_failure(failure)
            self._persist_session_once()
            yield TurnFailedPayload(error=terminal_error)
            return
        except GeneratorExit:
            if self._lane == "foreground":
                await self._persist_cancelled_state(partial_content, pending_tool_calls)
            raise
        except asyncio.CancelledError:
            if self._lane == "scheduled_work":
                raise
            yield await self._cancelled_payload(
                partial_content,
                pending_tool_calls,
            )
            return
        finally:
            if self._lane == "foreground":
                try:
                    await self._repair_unfinished_tool_calls(
                        pending_tool_calls,
                        error=pending_repair_error,
                    )
                except _PERSISTENCE_EXCEPTIONS as failure:
                    pending_tool_calls.clear()
                    _log_persistence_failure(
                        failure,
                        operation="interrupted_state_repair",
                    )
                cleanup_failure = await _close_provider_stream(provider_stream)
                if cleanup_failure is not None:
                    _log_cleanup_failure(cleanup_failure)
            self._persist_session_once()

    def _session_model_request(self, current_user: dict[str, Any]) -> ModelRequest:
        session = self._session
        messages: list[ModelMessage] = []
        for message in session.messages[session.last_consolidated :]:
            if message is current_user:
                timestamp = message["timestamp"]
                if not isinstance(timestamp, str):
                    raise TypeError("Session user timestamp must be a string")
                messages.append(
                    UserModelMessage(
                        content=current_user_input(
                            content=message["content"],
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
            route="chat" if self._lane == "foreground" else "cron",
            system_prompt=self._system_prompt,
            messages=tuple(messages),
            tools=() if self._tool_gateway is None else self._tool_gateway.schemas,
            stream=self._lane == "foreground",
            model=self._settings.model,
            max_output=self._settings.max_output,
            temperature=self._settings.temperature,
            reasoning_effort=self._settings.reasoning_effort,
            timeout_seconds=self._settings.timeout_seconds,
        )

    def _add_session_assistant(
        self,
        response: ModelResponse | None = None,
        *,
        content: str | None = None,
        status: str = "completed",
        error: dict[str, str] | None = None,
        usage: dict[str, int] | None = None,
    ) -> None:
        session = self._session
        assert session is not None
        if response is not None:
            message = response.message
            session.add_message(
                "assistant",
                message.content,
                tool_calls=[tool_call.to_dict() for tool_call in message.tool_calls],
                status="completed",
                error=None,
                token_usage=_assistant_token_usage(response.usage),
            )
            return
        session.add_message(
            "assistant",
            content or "",
            tool_calls=[],
            status=status,
            error=error,
            token_usage=usage or _zero_assistant_usage(),
        )

    def _add_session_tool(self, result: ToolResult) -> None:
        session = self._session
        assert session is not None
        session.add_message(
            "tool",
            result.content,
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
            artifact=None if result.artifact is None else result.artifact.to_dict(),
        )

    def _persist_session_once(self) -> None:
        if self._session_persist_requested:
            return
        self._session_persist_requested = True
        try:
            self._session.persist()
        except Exception:
            pass

    def _capture_terminal_failure(self, error: BaseException) -> None:
        if self._on_terminal_failure is not None:
            self._on_terminal_failure(error)

    async def _completion_events(
        self,
        request: ModelRequest,
    ) -> AsyncGenerator[ModelStreamEvent, None]:
        response = await self._provider.complete(request)
        yield ModelCompleted(response=response)

    async def _cancelled_payload(
        self,
        partial_chunks: list[str],
        pending_tool_calls: list[ModelToolCall],
    ) -> TurnCancelledPayload:
        await self._persist_cancelled_state(partial_chunks, pending_tool_calls)
        return TurnCancelledPayload(partial_content="".join(partial_chunks))

    async def _persist_cancelled_state(
        self,
        partial_chunks: list[str],
        pending_tool_calls: list[ModelToolCall],
    ) -> None:
        if partial_chunks:
            self._add_session_assistant(
                content="".join(partial_chunks),
                status="interrupted",
                error={
                    "code": "turn_cancelled",
                    "message": "Turn interrupted by user.",
                },
                usage=_zero_assistant_usage(),
            )
        await self._repair_unfinished_tool_calls(pending_tool_calls)
        self._persist_session_once()

    async def _repair_unfinished_tool_calls(
        self,
        pending_tool_calls: list[ModelToolCall],
        *,
        error: dict[str, str] | None = None,
    ) -> None:
        while pending_tool_calls:
            tool_call = pending_tool_calls[0]
            failure = error or {
                "code": "turn_cancelled",
                "message": "Tool call interrupted because the turn was cancelled.",
            }
            self._add_session_tool(
                ToolResult(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=failure["message"],
                    status="error",
                    artifact=None,
                )
            )
            pending_tool_calls.pop(0)

    async def _repair_scheduled_cancellation(
        self,
        unfinished_tool_calls: tuple[ModelToolCall, ...],
    ) -> None:
        for unfinished in unfinished_tool_calls:
            self._add_session_tool(
                ToolResult(
                    tool_call_id=unfinished.id,
                    name=unfinished.name,
                    content="Scheduled Work tool call cancelled.",
                    status="error",
                    artifact=None,
                )
            )


async def _provider_events(
    stream: AsyncIterator[ModelStreamEvent],
) -> AsyncGenerator[ModelStreamEvent, None]:
    while True:
        try:
            event = await anext(stream)
        except StopAsyncIteration:
            return
        except ModelCallError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception:
            raise _unexpected_provider_failure() from None
        yield event


async def _close_provider_stream(
    stream: AsyncIterator[ModelStreamEvent] | None,
) -> Exception | None:
    if stream is None:
        return None
    close = getattr(stream, "aclose", None)
    if close is None:
        return None
    try:
        await close()
    except Exception as failure:
        return failure
    return None


def _tool_activity_summary(action: str, tool_name: str) -> str:
    return " ".join(f"{action} {tool_name}".split())[:240]


def _unexpected_provider_failure() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="model_failed",
            message="The model request failed.",
        )
    )


def _log_model_failure(failure: ModelCallError) -> None:
    logger.opt(exception=failure).error(
        "Agent Turn failed code={} type={}",
        failure.error.code,
        type(failure).__name__,
    )


def _log_persistence_failure(failure: Exception, *, operation: str) -> None:
    logger.opt(exception=failure).error(
        "Agent Turn failed code=persistence_error operation={} type={}",
        operation,
        type(failure).__name__,
    )


def _log_artifact_failure(failure: ArtifactWriteError, *, tool_name: str) -> None:
    logger.opt(exception=failure).error(
        "Tool Artifact persistence failed code=persistence_error tool={} type={}",
        tool_name,
        type(failure).__name__,
    )


def _log_cleanup_failure(failure: Exception) -> None:
    logger.opt(exception=failure).error(
        "Agent Turn cleanup failed code=model_failed operation=provider_stream_close type={}",
        type(failure).__name__,
    )


def _never_cancelled() -> bool:
    return False


def _inline_tool_result(result: ToolResult) -> ToolResult:
    return result


def _assistant_token_usage(usage: ModelUsage) -> dict[str, int]:
    return {"model_calls": 1, **usage.to_dict()}


def _zero_assistant_usage() -> dict[str, int]:
    return {
        "model_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def model_message_from_session(
    message: dict[str, Any],
) -> UserModelMessage | AssistantModelMessage | ToolModelMessage | None:
    """Project persisted conversation history into the next provider request."""
    return _model_message_from_json(message)


def _model_message_from_json(
    message: dict[str, Any],
) -> UserModelMessage | AssistantModelMessage | ToolModelMessage | None:
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
