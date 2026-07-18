"""Runtime Core orchestration for one foreground or Scheduled Work Agent turn."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from myclaw.agent.events import (
    AgentEventType,
    PermissionRequestedPayload,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.agent.prompts import current_user_input
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelMessage,
    ModelRequest,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    TextDelta,
    ToolModelMessage,
    UserModelMessage,
)
from myclaw.provider.ports import ModelProvider
from myclaw.session.ports import SessionStore
from myclaw.session.records import (
    AssistantSessionMessage,
    ConversationSession,
    SessionError,
    SessionMessage,
    ToolSessionMessage,
    UserSessionMessage,
)
from myclaw.tools.models import ModelToolCall, ToolResult
from myclaw.tools.tool_artifacts import ArtifactDiscardError
from myclaw.tools.tool_gateway import ToolGateway

type AgentTurnLane = Literal["foreground", "scheduled_work"]
type AgentTurnPayload = (
    TurnStartedPayload
    | TextDeltaPayload
    | ToolStartedPayload
    | PermissionRequestedPayload
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
    if isinstance(payload, PermissionRequestedPayload):
        return "permission_requested"
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
        self._permission_waits: dict[UUID, asyncio.Future[bool]] = {}

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
        except _PERSISTENCE_EXCEPTIONS:
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
                except _PERSISTENCE_EXCEPTIONS:
                    yield TurnFailedPayload(error=self._session_read_failure)
                    return
                if self._history_preparer is not None:
                    session = await self._history_preparer(session)
                request = self._model_request(session, current_user=user_message)
                if self._lane == "foreground":
                    await _close_provider_stream(provider_stream)
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
                    except _PERSISTENCE_EXCEPTIONS:
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
                            approved: bool | None = None
                            permission = self._tool_gateway.permission_request(tool_call)
                            if permission is not None:
                                request_id = self._new_uuid()
                                wait = asyncio.get_running_loop().create_future()
                                self._permission_waits[request_id] = wait
                                try:
                                    yield PermissionRequestedPayload(
                                        request_id=request_id,
                                        tool_call_id=tool_call.id,
                                        tool_name=tool_call.name,
                                        action=permission.action,
                                        resource=permission.resource,
                                        risk_summary=permission.risk_summary,
                                    )
                                    approved = await wait
                                finally:
                                    self._permission_waits.pop(request_id, None)
                                    if not wait.done():
                                        wait.cancel()
                            try:
                                result = await self._tool_gateway.execute(
                                    tool_call,
                                    approved=approved,
                                )
                            except asyncio.CancelledError:
                                if self._lane == "scheduled_work":
                                    await self._repair_scheduled_cancellation(
                                        response.message.tool_calls[index:]
                                    )
                                raise

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
                            except _PERSISTENCE_EXCEPTIONS:
                                if self._lane == "foreground":
                                    publication = await self._session_message_publication(
                                        tool_message
                                    )
                                    self._settle_foreground_artifact(result, publication)
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
                                    await self._reconcile_scheduled_tool_artifact(
                                        result,
                                        tool_message,
                                    )
                                yield TurnFailedPayload(error=self._session_update_failure)
                                return
                            except asyncio.CancelledError:
                                if self._lane == "foreground":
                                    publication = await self._session_message_publication(
                                        tool_message
                                    )
                                    self._settle_foreground_artifact(result, publication)
                                    if publication is True:
                                        pending_tool_calls.pop(0)
                                else:
                                    await self._reconcile_scheduled_tool_artifact(
                                        result,
                                        tool_message,
                                    )
                                raise
                            except BaseException:
                                if self._lane == "foreground":
                                    self._discard_artifact(result)
                                else:
                                    await self._reconcile_scheduled_tool_artifact(
                                        result,
                                        tool_message,
                                    )
                                raise
                            self._tool_gateway.commit_artifact(result)
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
            except _PERSISTENCE_EXCEPTIONS:
                terminal_error = self._session_update_failure
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
                except _PERSISTENCE_EXCEPTIONS:
                    pending_tool_calls.clear()
                await _close_provider_stream(provider_stream)

    async def resolve_permission(self, request_id: UUID, approved: bool) -> None:
        if not isinstance(approved, bool):
            raise TypeError("approved must be a boolean")
        wait = self._permission_waits.get(request_id)
        if wait is None or wait.done():
            raise RuntimeError("Permission request is not pending")
        wait.set_result(approved)

    def cancel_pending_permissions(self) -> None:
        for wait in tuple(self._permission_waits.values()):
            if not wait.done():
                wait.cancel()

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
            tools=() if self._tool_gateway is None else self._tool_gateway.definitions,
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
                error=failure,
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
                        error=SessionError(
                            code="turn_cancelled",
                            message="Scheduled Work tool call cancelled.",
                        ),
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
        if any(message == expected for message in same_id):
            return True
        if same_id:
            return None
        return False

    async def _reconcile_scheduled_tool_artifact(
        self,
        result: ToolResult,
        tool_message: ToolSessionMessage,
    ) -> None:
        try:
            reloaded = await self._sessions.load(self._session_id)
        except BaseException:
            if self._tool_gateway is not None:
                self._tool_gateway.commit_artifact(result)
            return
        if tool_message in reloaded.messages or any(
            message.id == tool_message.id for message in reloaded.messages
        ):
            if self._tool_gateway is not None:
                self._tool_gateway.commit_artifact(result)
            return
        self._discard_artifact(result)

    def _settle_foreground_artifact(
        self,
        result: ToolResult,
        publication: bool | None,
    ) -> None:
        if self._tool_gateway is None:
            return
        if publication is False:
            self._discard_artifact(result)
        else:
            self._tool_gateway.commit_artifact(result)

    def _discard_artifact(self, result: ToolResult) -> None:
        if self._tool_gateway is None:
            return
        try:
            self._tool_gateway.discard_artifact(result)
        except ArtifactDiscardError:
            pass

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
        error=(
            None
            if result.error is None
            else SessionError(
                code=result.error.code,
                message=result.error.message,
            )
        ),
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


async def _close_provider_stream(stream: AsyncIterator[ModelStreamEvent] | None) -> None:
    if stream is None:
        return
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:
        pass


def _tool_activity_summary(action: str, tool_name: str) -> str:
    return " ".join(f"{action} {tool_name}".split())[:240]


def _unexpected_provider_failure() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="model_failed",
            message="The model request failed.",
        )
    )


def _never_cancelled() -> bool:
    return False


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
                content=f"{message.content}\n\n[Turn interrupted by user.]",
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
