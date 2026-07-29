"""Runtime Core orchestration for one foreground or Scheduled Work Agent turn."""

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

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
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
    ToolModelMessage,
    UserModelMessage,
)
from myclaw.session.records import (
    AssistantSessionMessage,
    ConversationSession,
    SessionError,
    SessionMessage,
    ToolSessionMessage,
    UserSessionMessage,
)
from myclaw.session.session_store import SessionStore
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
_FOREGROUND_SESSION_UPDATE_FAILURE = ErrorInfo(
    code="persistence_error",
    message="Conversation Session could not be updated.",
)
_FOREGROUND_SESSION_READ_FAILURE = ErrorInfo(
    code="persistence_error",
    message="Conversation Session could not be read.",
)
_SCHEDULED_SESSION_FAILURE = ErrorInfo(
    code="persistence_error",
    message="Scheduled Work Session could not be updated.",
)

logger = logging.getLogger(__name__)


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
        sessions: SessionStore,
        session_id: str,
        settings: AgentTurnModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str,
        tool_gateway: ToolGateway | None,
        history_preparer: (
            Callable[[ConversationSession], Awaitable[ConversationSession]] | None
        ) = None,
        after_user_published: Callable[[UserSessionMessage], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
        externalize_result: ToolResultExternalizer | None = None,
        on_terminal_failure: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._lane = lane
        self._provider = provider
        self._sessions = sessions
        self._session_id = session_id
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

    async def run(self, text: str) -> AsyncGenerator[AgentTurnPayload, None]:
        yield TurnStartedPayload()

        user_message = UserSessionMessage(
            id=str(self._new_uuid()),
            created_at=self._persisted_now(),
            content=text,
        )
        try:
            await self._sessions.append_message(self._session_id, user_message)
        except asyncio.CancelledError:
            if self._lane == "scheduled_work":
                raise
            yield await self._cancelled_payload([], [])
            return
        except _PERSISTENCE_EXCEPTIONS as failure:
            if self._lane == "foreground":
                _log_persistence_failure(failure, operation="session_append")
            else:
                self._capture_terminal_failure(failure)
            yield TurnFailedPayload(error=self._session_update_failure)
            return
        if self._lane == "foreground" and self._cancel_requested():
            yield await self._cancelled_payload([], [])
            return
        if self._after_user_published is not None:
            self._after_user_published(user_message)

        partial_content: list[str] = []
        pending_tool_calls: list[ModelToolCall] = []
        pending_repair_error: SessionError | None = None
        provider_stream: AsyncIterator[ModelStreamEvent] | None = None
        try:
            while True:
                partial_content = []
                try:
                    session = await self._sessions.load(self._session_id)
                except _PERSISTENCE_EXCEPTIONS as failure:
                    if self._lane == "foreground":
                        _log_persistence_failure(failure, operation="session_read")
                    else:
                        self._capture_terminal_failure(failure)
                    yield TurnFailedPayload(error=self._session_read_failure)
                    return
                if self._history_preparer is not None:
                    session = await self._history_preparer(session)
                request = self._model_request(session, current_user=user_message)
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
                        assistant_message = AssistantSessionMessage(
                            id=str(self._new_uuid()),
                            created_at=self._persisted_now(),
                            content=response.message.content,
                            tool_calls=response.message.tool_calls,
                            status="completed",
                            error=None,
                            usage=response.usage,
                        )
                    except ValueError:
                        if self._lane == "foreground":
                            raise _unexpected_provider_failure() from None
                        raise
                    try:
                        await self._sessions.append_message(
                            self._session_id,
                            assistant_message,
                        )
                    except asyncio.CancelledError:
                        if self._lane == "scheduled_work":
                            raise
                        publication = await self._session_message_publication(assistant_message)
                        if publication is True:
                            partial_content.clear()
                            pending_tool_calls = list(assistant_message.tool_calls)
                        raise
                    except _PERSISTENCE_EXCEPTIONS as failure:
                        if self._lane == "foreground":
                            _log_persistence_failure(failure, operation="session_append")
                        else:
                            self._capture_terminal_failure(failure)
                        if self._lane == "foreground":
                            publication = await self._session_message_publication(assistant_message)
                            if publication is True:
                                pending_tool_calls = list(assistant_message.tool_calls)
                                pending_repair_error = SessionError(
                                    code="persistence_error",
                                    message="Assistant response could not be persisted.",
                                )
                                try:
                                    await self._repair_unfinished_tool_calls(
                                        pending_tool_calls,
                                        error=pending_repair_error,
                                    )
                                except _PERSISTENCE_EXCEPTIONS:
                                    pass
                        yield TurnFailedPayload(error=self._session_update_failure)
                        return

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

                            tool_message = _tool_session_message(
                                result,
                                message_id=str(self._new_uuid()),
                                created_at=self._persisted_now(),
                            )
                            try:
                                await self._sessions.append_message(
                                    self._session_id,
                                    tool_message,
                                )
                            except _PERSISTENCE_EXCEPTIONS as failure:
                                if self._lane == "foreground":
                                    _log_persistence_failure(
                                        failure,
                                        operation="session_append",
                                    )
                                    publication = await self._session_message_publication(
                                        tool_message
                                    )
                                    if publication is True:
                                        pending_tool_calls.pop(0)
                                    pending_repair_error = SessionError(
                                        code="persistence_error",
                                        message="Tool result could not be persisted.",
                                    )
                                    try:
                                        await self._repair_unfinished_tool_calls(
                                            pending_tool_calls,
                                            error=pending_repair_error,
                                        )
                                    except _PERSISTENCE_EXCEPTIONS:
                                        pass
                                else:
                                    self._capture_terminal_failure(failure)
                                yield TurnFailedPayload(error=self._session_update_failure)
                                return
                            except asyncio.CancelledError:
                                if self._lane == "foreground":
                                    publication = await self._session_message_publication(
                                        tool_message
                                    )
                                    if publication is True:
                                        pending_tool_calls.pop(0)
                                raise
                            except BaseException:
                                raise
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
            error_message = AssistantSessionMessage(
                id=str(self._new_uuid()),
                created_at=self._persisted_now(),
                content="".join(partial_content) if self._lane == "foreground" else "",
                tool_calls=(),
                status="error",
                error=SessionError(
                    code=failure.error.code,
                    message=failure.error.message,
                ),
                usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )
            terminal_persistence_failure: Exception | None = None
            try:
                await self._sessions.append_message(
                    self._session_id,
                    error_message,
                )
            except asyncio.CancelledError:
                if self._lane == "scheduled_work":
                    raise
                if await self._session_message_publication(error_message) is True:
                    partial_content.clear()
                yield await self._cancelled_payload(
                    partial_content,
                    pending_tool_calls,
                )
                return
            except _PERSISTENCE_EXCEPTIONS as persistence_failure:
                terminal_error = self._session_update_failure
                terminal_persistence_failure = persistence_failure
            if self._lane == "foreground":
                _log_model_failure(failure)
                if terminal_persistence_failure is not None:
                    _log_persistence_failure(
                        terminal_persistence_failure,
                        operation="terminal_state_append",
                    )
            else:
                diagnostic: BaseException = failure
                if terminal_persistence_failure is not None:
                    diagnostic = ExceptionGroup(
                        "Scheduled Work terminal failure",
                        [failure, terminal_persistence_failure],
                    )
                self._capture_terminal_failure(diagnostic)
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

    def _capture_terminal_failure(self, error: BaseException) -> None:
        if self._on_terminal_failure is not None:
            self._on_terminal_failure(error)

    def _model_request(
        self,
        session: ConversationSession,
        *,
        current_user: UserSessionMessage,
    ) -> ModelRequest:
        messages: list[ModelMessage] = []
        for message in session.short_term_messages:
            if isinstance(message, UserSessionMessage) and message.id == current_user.id:
                messages.append(
                    UserModelMessage(
                        content=current_user_input(
                            content=current_user.content,
                            current_time=current_user.created_at,
                            session_id=self._session_id,
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
        partial_content = "".join(partial_chunks)
        if partial_content:
            try:
                await self._sessions.append_message(
                    self._session_id,
                    AssistantSessionMessage(
                        id=str(self._new_uuid()),
                        created_at=self._persisted_now(),
                        content=partial_content,
                        tool_calls=(),
                        status="interrupted",
                        error=SessionError(
                            code="turn_cancelled",
                            message="Turn interrupted by user.",
                        ),
                        usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                    ),
                )
            except _PERSISTENCE_EXCEPTIONS:
                pass
        try:
            await self._repair_unfinished_tool_calls(pending_tool_calls)
        except _PERSISTENCE_EXCEPTIONS:
            pass

    async def _repair_unfinished_tool_calls(
        self,
        pending_tool_calls: list[ModelToolCall],
        *,
        error: SessionError | None = None,
    ) -> None:
        while pending_tool_calls:
            tool_call = pending_tool_calls[0]
            failure = error or SessionError(
                code="turn_cancelled",
                message="Tool call interrupted because the turn was cancelled.",
            )
            repair_message = ToolSessionMessage(
                id=str(self._new_uuid()),
                created_at=self._persisted_now(),
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=failure.message,
                status="error",
                artifact=None,
            )
            try:
                await self._sessions.append_message(
                    self._session_id,
                    repair_message,
                )
            except asyncio.CancelledError:
                if await self._session_message_publication(repair_message) is True:
                    pending_tool_calls.pop(0)
                raise
            except _PERSISTENCE_EXCEPTIONS:
                if await self._session_message_publication(repair_message) is True:
                    pending_tool_calls.pop(0)
                    continue
                raise
            pending_tool_calls.pop(0)

    async def _repair_scheduled_cancellation(
        self,
        unfinished_tool_calls: tuple[ModelToolCall, ...],
    ) -> None:
        for unfinished in unfinished_tool_calls:
            try:
                await self._sessions.append_message(
                    self._session_id,
                    ToolSessionMessage(
                        id=str(self._new_uuid()),
                        created_at=self._persisted_now(),
                        tool_call_id=unfinished.id,
                        name=unfinished.name,
                        content="Scheduled Work tool call cancelled.",
                        status="error",
                        artifact=None,
                    ),
                )
            except _PERSISTENCE_EXCEPTIONS:
                pass

    async def _session_message_publication(
        self,
        expected: SessionMessage,
    ) -> bool | None:
        try:
            session = await self._sessions.load(self._session_id)
        except _PERSISTENCE_EXCEPTIONS:
            return None
        same_id = tuple(message for message in session.messages if message.id == expected.id)
        if any(message.to_dict() == expected.to_dict() for message in same_id):
            return True
        if same_id:
            return None
        return False

    @property
    def _session_update_failure(self) -> ErrorInfo:
        if self._lane == "foreground":
            return _FOREGROUND_SESSION_UPDATE_FAILURE
        return _SCHEDULED_SESSION_FAILURE

    @property
    def _session_read_failure(self) -> ErrorInfo:
        if self._lane == "foreground":
            return _FOREGROUND_SESSION_READ_FAILURE
        return _SCHEDULED_SESSION_FAILURE

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


def _tool_session_message(
    result: ToolResult,
    *,
    message_id: str,
    created_at: datetime,
) -> ToolSessionMessage:
    return ToolSessionMessage(
        id=message_id,
        created_at=created_at,
        tool_call_id=result.tool_call_id,
        name=result.name,
        content=result.content,
        status=result.status,
        artifact=result.artifact,
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
    safe_failure = ModelCallError(
        ErrorInfo(code=failure.error.code, message="The model request failed.")
    )
    if failure.__cause__ is not None:
        safe_failure.__cause__ = _safe_exception(
            failure.__cause__,
            "Diagnostic detail redacted.",
        )
    elif failure.__context__ is not None and not failure.__suppress_context__:
        safe_failure.__context__ = _safe_exception(
            failure.__context__,
            "Diagnostic detail redacted.",
        )
        safe_failure.__suppress_context__ = False
    safe_failure = safe_failure.with_traceback(failure.__traceback__)
    logger.error(
        "Agent Turn failed code=%s type=%s",
        failure.error.code,
        type(failure).__name__,
        exc_info=(type(safe_failure), safe_failure, safe_failure.__traceback__),
    )


def _log_persistence_failure(failure: Exception, *, operation: str) -> None:
    safe_failure = _safe_exception(
        failure,
        "Conversation Session persistence failed.",
    )
    logger.error(
        "Agent Turn failed code=persistence_error operation=%s type=%s",
        operation,
        type(failure).__name__,
        exc_info=(type(safe_failure), safe_failure, safe_failure.__traceback__),
    )


def _log_artifact_failure(failure: ArtifactWriteError, *, tool_name: str) -> None:
    safe_failure = _safe_exception(
        failure,
        "Tool Artifact persistence failed.",
    )
    logger.error(
        "Tool Artifact persistence failed code=persistence_error tool=%s type=%s",
        tool_name,
        type(failure).__name__,
        exc_info=(type(safe_failure), safe_failure, safe_failure.__traceback__),
    )


def _log_cleanup_failure(failure: Exception) -> None:
    safe_failure = _safe_exception(
        failure,
        "Provider stream cleanup failed.",
    )
    logger.error(
        "Agent Turn cleanup failed code=model_failed operation=provider_stream_close type=%s",
        type(failure).__name__,
        exc_info=(type(safe_failure), safe_failure, safe_failure.__traceback__),
    )


def _safe_exception(failure: BaseException, message: str) -> BaseException:
    try:
        safe_failure = type(failure)(message)
    except Exception:
        safe_failure = RuntimeError(f"{type(failure).__name__}: {message}")
    if failure.__cause__ is not None:
        safe_failure.__cause__ = _safe_exception(
            failure.__cause__,
            "Diagnostic detail redacted.",
        )
    elif failure.__context__ is not None and not failure.__suppress_context__:
        safe_failure.__context__ = _safe_exception(
            failure.__context__,
            "Diagnostic detail redacted.",
        )
        safe_failure.__suppress_context__ = False
    return safe_failure.with_traceback(failure.__traceback__)


def _never_cancelled() -> bool:
    return False


def _inline_tool_result(result: ToolResult) -> ToolResult:
    return result


def model_message_from_session(
    message: SessionMessage,
) -> UserModelMessage | AssistantModelMessage | ToolModelMessage | None:
    """Project persisted conversation history into the next provider request."""
    if isinstance(message, UserSessionMessage):
        return UserModelMessage(content=message.content)
    if isinstance(message, AssistantSessionMessage):
        if message.status == "error" and not message.content and not message.tool_calls:
            return None
        if message.status == "interrupted":
            return AssistantModelMessage(
                content=interrupted_assistant_content(message.content),
                tool_calls=message.tool_calls,
            )
        return AssistantModelMessage(content=message.content, tool_calls=message.tool_calls)
    if isinstance(message, ToolSessionMessage):
        return ToolModelMessage(
            tool_call_id=message.tool_call_id,
            name=message.name,
            content=message.content,
        )
    raise TypeError("Unsupported Short-term Memory message")
