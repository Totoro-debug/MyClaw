"""Conversation Port implementation for the first successful streaming turn."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from myclaw.agent.prompts import current_user_input
from myclaw.contracts import (
    AgentEvent,
    AgentEventPayload,
    AgentEventType,
    AssistantModelMessage,
    AssistantSessionMessage,
    ConversationSession,
    ErrorInfo,
    MetadataUpdate,
    ModelCallError,
    ModelCompleted,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelToolCall,
    ModelUsage,
    PermissionRequestedPayload,
    ReasoningEffort,
    SessionError,
    SessionMessage,
    SessionStore,
    TextDelta,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolModelMessage,
    ToolSessionMessage,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
    UserModelMessage,
    UserSessionMessage,
)
from myclaw.session.session_titles import normalize_session_title
from myclaw.tools.tool_artifacts import ArtifactDiscardError
from myclaw.tools.tool_gateway import ToolGateway


@dataclass(frozen=True, slots=True)
class ChatModelSettings:
    """Resolved provider-neutral fields needed for one chat request."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


class StreamingConversationPort:
    """Translate one successful provider stream into Agent Events and Session records."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        sessions: SessionStore,
        session_id: str,
        settings: ChatModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str = "",
        title_prompt: str | None = None,
        title_new_uuid: Callable[[], UUID] = uuid4,
        tool_gateway: ToolGateway | None = None,
        history_preparer: (
            Callable[[ConversationSession], Awaitable[ConversationSession]] | None
        ) = None,
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._session_id = session_id
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._system_prompt = system_prompt
        self._title_prompt = title_prompt
        self._title_new_uuid = title_new_uuid
        self._tool_gateway = tool_gateway
        self._history_preparer = history_preparer
        self._title_task: asyncio.Task[None] | None = None
        self._next_event_id = 0
        self._foreground_active = False
        self._active_task: asyncio.Task[object] | None = None
        self._active_turn_done: asyncio.Event | None = None
        self._cancel_requested = False
        self._permission_waits: dict[UUID, asyncio.Future[bool]] = {}
        self._close_task: asyncio.Task[None] | None = None

    async def submit(self, text: str) -> AsyncGenerator[AgentEvent, None]:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if not text.strip():
            return
        if self._foreground_active:
            raise RuntimeError("A foreground turn is already active")
        self._foreground_active = True
        self._active_task = asyncio.current_task()
        turn_done = asyncio.Event()
        self._active_turn_done = turn_done
        self._cancel_requested = False
        turn = self._submit_turn(text)
        try:
            async for event in turn:
                yield event
        finally:
            try:
                await turn.aclose()
            finally:
                self._active_task = None
                turn_done.set()
                if self._active_turn_done is turn_done:
                    self._active_turn_done = None
                self._cancel_requested = False
                self._foreground_active = False

    async def _submit_turn(self, text: str) -> AsyncGenerator[AgentEvent, None]:
        turn_id = self._new_uuid()
        yield self._event(turn_id, "turn_started", TurnStartedPayload())

        user_created_at = self._persisted_now()
        user_message = UserSessionMessage(
            id=str(self._new_uuid()),
            created_at=user_created_at,
            content=text,
        )
        try:
            await self._sessions.append_message(self._session_id, user_message)
        except asyncio.CancelledError:
            yield await self._cancelled_event(turn_id, [], [])
            return
        except (OSError, UnicodeError, ValueError):
            yield self._event(
                turn_id,
                "turn_failed",
                TurnFailedPayload(
                    error=ErrorInfo(
                        code="persistence_error",
                        message="Conversation Session could not be updated.",
                    )
                ),
            )
            return
        if self._cancel_requested:
            yield await self._cancelled_event(turn_id, [], [])
            return
        if self._title_prompt is not None and self._title_task is None:
            title_task = asyncio.create_task(
                self._generate_title_for_first_user(
                    session_id=self._session_id,
                    first_user_id=user_message.id,
                    first_user_content=text,
                    request_id=self._title_new_uuid(),
                )
            )
            title_task.add_done_callback(_consume_task_exception)
            self._title_task = title_task
        partial_content: list[str] = []
        pending_tool_calls: list[ModelToolCall] = []
        pending_repair_error: SessionError | None = None
        provider_stream: AsyncIterator[ModelStreamEvent] | None = None
        try:
            while True:
                partial_content = []
                try:
                    session = await self._sessions.load(self._session_id)
                except (OSError, UnicodeError, ValueError):
                    yield self._event(
                        turn_id,
                        "turn_failed",
                        TurnFailedPayload(
                            error=ErrorInfo(
                                code="persistence_error",
                                message="Conversation Session could not be read.",
                            )
                        ),
                    )
                    return
                if self._history_preparer is not None:
                    session = await self._history_preparer(session)
                request = self._model_request(session, current_user=user_message)
                await self._close_provider_stream(provider_stream)
                try:
                    provider_stream = self._provider.stream(request)
                except ModelCallError:
                    raise
                except Exception:
                    raise _unexpected_provider_failure() from None
                async for model_event in self._provider_events(provider_stream):
                    if isinstance(model_event, TextDelta):
                        partial_content.append(model_event.delta)
                        yield self._event(
                            turn_id,
                            "text_delta",
                            TextDeltaPayload(delta=model_event.delta),
                        )
                        if self._cancel_requested:
                            yield await self._cancelled_event(
                                turn_id,
                                partial_content,
                                pending_tool_calls,
                            )
                            return
                        continue
                    if isinstance(model_event, ModelCompleted):
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
                            raise _unexpected_provider_failure() from None
                        try:
                            await self._sessions.append_message(
                                self._session_id,
                                assistant_message,
                            )
                        except asyncio.CancelledError:
                            publication = await self._session_message_publication(assistant_message)
                            if publication is True:
                                partial_content.clear()
                                pending_tool_calls = list(assistant_message.tool_calls)
                            raise
                        except (OSError, UnicodeError, ValueError):
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
                                except (OSError, UnicodeError, ValueError):
                                    pass
                            yield self._event(
                                turn_id,
                                "turn_failed",
                                TurnFailedPayload(
                                    error=ErrorInfo(
                                        code="persistence_error",
                                        message="Conversation Session could not be updated.",
                                    )
                                ),
                            )
                            return
                        partial_content = []
                        if response.message.tool_calls and self._tool_gateway is not None:
                            pending_tool_calls = list(response.message.tool_calls)
                            for tool_call in response.message.tool_calls:
                                yield self._event(
                                    turn_id,
                                    "tool_started",
                                    ToolStartedPayload(
                                        tool_call_id=tool_call.id,
                                        tool_name=tool_call.name,
                                        summary=_tool_activity_summary("Running", tool_call.name),
                                    ),
                                )
                                if self._cancel_requested:
                                    yield await self._cancelled_event(
                                        turn_id,
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
                                        yield self._event(
                                            turn_id,
                                            "permission_requested",
                                            PermissionRequestedPayload(
                                                request_id=request_id,
                                                tool_call_id=tool_call.id,
                                                tool_name=tool_call.name,
                                                action=permission.action,
                                                resource=permission.resource,
                                                risk_summary=permission.risk_summary,
                                            ),
                                        )
                                        approved = await wait
                                    finally:
                                        self._permission_waits.pop(request_id, None)
                                        if not wait.done():
                                            wait.cancel()
                                result = await self._tool_gateway.execute(
                                    tool_call,
                                    approved=approved,
                                )
                                tool_message = ToolSessionMessage(
                                    id=str(self._new_uuid()),
                                    created_at=self._persisted_now(),
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
                                try:
                                    await self._sessions.append_message(
                                        self._session_id,
                                        tool_message,
                                    )
                                except (OSError, UnicodeError, ValueError):
                                    publication = await self._session_message_publication(
                                        tool_message
                                    )
                                    if publication is False:
                                        try:
                                            self._tool_gateway.discard_artifact(result)
                                        except ArtifactDiscardError:
                                            pass
                                    elif publication is True:
                                        self._tool_gateway.commit_artifact(result)
                                        pending_tool_calls.pop(0)
                                    else:
                                        self._tool_gateway.commit_artifact(result)
                                    pending_repair_error = SessionError(
                                        code="persistence_error",
                                        message="Tool result could not be persisted.",
                                    )
                                    try:
                                        await self._repair_unfinished_tool_calls(
                                            pending_tool_calls,
                                            error=pending_repair_error,
                                        )
                                    except (OSError, UnicodeError, ValueError):
                                        pass
                                    yield self._event(
                                        turn_id,
                                        "turn_failed",
                                        TurnFailedPayload(
                                            error=ErrorInfo(
                                                code="persistence_error",
                                                message=(
                                                    "Conversation Session could not be updated."
                                                ),
                                            )
                                        ),
                                    )
                                    return
                                except asyncio.CancelledError:
                                    publication = await self._session_message_publication(
                                        tool_message
                                    )
                                    if publication is False:
                                        try:
                                            self._tool_gateway.discard_artifact(result)
                                        except ArtifactDiscardError:
                                            pass
                                    elif publication is True:
                                        self._tool_gateway.commit_artifact(result)
                                        pending_tool_calls.pop(0)
                                    else:
                                        self._tool_gateway.commit_artifact(result)
                                    raise
                                except BaseException:
                                    try:
                                        self._tool_gateway.discard_artifact(result)
                                    except ArtifactDiscardError:
                                        pass
                                    raise
                                self._tool_gateway.commit_artifact(result)
                                pending_tool_calls.pop(0)
                                yield self._event(
                                    turn_id,
                                    "tool_completed",
                                    ToolCompletedPayload(
                                        tool_call_id=result.tool_call_id,
                                        tool_name=result.name,
                                        status=result.status,
                                        summary=_tool_activity_summary("Finished", result.name),
                                    ),
                                )
                                if self._cancel_requested:
                                    yield await self._cancelled_event(
                                        turn_id,
                                        partial_content,
                                        pending_tool_calls,
                                    )
                                    return
                            break
                        yield self._event(
                            turn_id,
                            "turn_completed",
                            TurnCompletedPayload(
                                content=response.message.content,
                                usage=response.usage,
                            ),
                        )
                        return
                    raise _unexpected_provider_failure()
                else:
                    raise ModelCallError(
                        ErrorInfo(
                            code="model_failed",
                            message="The model stream ended without a complete response.",
                        )
                    )
        except ModelCallError as failure:
            if failure.error.code == "turn_cancelled":
                yield await self._cancelled_event(
                    turn_id,
                    partial_content,
                    pending_tool_calls,
                )
                return
            terminal_error = failure.error
            error_message = AssistantSessionMessage(
                id=str(self._new_uuid()),
                created_at=self._persisted_now(),
                content="".join(partial_content),
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
                if await self._session_message_publication(error_message) is True:
                    partial_content.clear()
                yield await self._cancelled_event(
                    turn_id,
                    partial_content,
                    pending_tool_calls,
                )
                return
            except (OSError, UnicodeError, ValueError):
                terminal_error = ErrorInfo(
                    code="persistence_error",
                    message="Conversation Session could not be updated.",
                )
            yield self._event(
                turn_id,
                "turn_failed",
                TurnFailedPayload(error=terminal_error),
            )
            return
        except GeneratorExit:
            await self._persist_cancelled_state(partial_content, pending_tool_calls)
            raise
        except asyncio.CancelledError:
            yield await self._cancelled_event(
                turn_id,
                partial_content,
                pending_tool_calls,
            )
            return
        finally:
            try:
                await self._repair_unfinished_tool_calls(
                    pending_tool_calls,
                    error=pending_repair_error,
                )
            except (OSError, UnicodeError, ValueError):
                pending_tool_calls.clear()
            await self._close_provider_stream(provider_stream)

    async def _generate_title_for_first_user(
        self,
        *,
        session_id: str,
        first_user_id: str,
        first_user_content: str,
        request_id: UUID,
    ) -> None:
        session = await self._sessions.load(session_id)
        if not session.messages or session.messages[0].id != first_user_id:
            return
        await self._generate_title(
            session_id=session_id,
            first_user_content=first_user_content,
            request_id=request_id,
        )

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
            route="chat",
            system_prompt=self._system_prompt,
            messages=tuple(messages),
            tools=() if self._tool_gateway is None else self._tool_gateway.definitions,
            stream=True,
            model=self._settings.model,
            max_output=self._settings.max_output,
            temperature=self._settings.temperature,
            reasoning_effort=self._settings.reasoning_effort,
            timeout_seconds=self._settings.timeout_seconds,
        )

    async def _generate_title(
        self,
        *,
        session_id: str,
        first_user_content: str,
        request_id: UUID,
    ) -> None:
        title = normalize_session_title(first_user_content) or "Untitled session"
        usage_delta: ModelUsage | None = None
        request = ModelRequest(
            request_id=request_id,
            route="chat",
            system_prompt=self._title_prompt or "",
            messages=(UserModelMessage(content=normalize_session_title(first_user_content)),),
            tools=(),
            stream=True,
            model=self._settings.model,
            max_output=self._settings.max_output,
            temperature=self._settings.temperature,
            reasoning_effort=self._settings.reasoning_effort,
            timeout_seconds=self._settings.timeout_seconds,
        )
        provider_stream: AsyncIterator[ModelStreamEvent] | None = None
        try:
            provider_stream = self._provider.stream(request)
            async for model_event in provider_stream:
                if not isinstance(model_event, ModelCompleted):
                    continue
                if not model_event.response.message.tool_calls:
                    generated = normalize_session_title(model_event.response.message.content)
                    if generated:
                        title = generated
                usage_delta = model_event.response.usage
                break
        except Exception:
            pass
        finally:
            await self._close_provider_stream(provider_stream)
        await self._sessions.update_metadata(
            session_id,
            MetadataUpdate(
                title=title,
                updated_at=self._persisted_now(),
                usage_delta=usage_delta,
            ),
        )

    async def resolve_permission(self, request_id: UUID, approved: bool) -> None:
        if not isinstance(approved, bool):
            raise TypeError("approved must be a boolean")
        wait = self._permission_waits.get(request_id)
        if wait is None or wait.done():
            raise RuntimeError("Permission request is not pending")
        wait.set_result(approved)

    async def cancel_active_turn(self) -> None:
        if self._cancel_requested:
            return
        for wait in tuple(self._permission_waits.values()):
            if not wait.done():
                wait.cancel()
        task = self._active_task
        if task is None or task.done():
            return
        self._cancel_requested = True
        if task is not asyncio.current_task():
            task.cancel()

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_active_turn())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_active_turn(self) -> None:
        turn_done = self._active_turn_done
        await self.cancel_active_turn()
        try:
            if turn_done is not None:
                await turn_done.wait()
        finally:
            title = self._title_task
            if title is not None and not title.done():
                title.cancel()
            if title is not None:
                await asyncio.gather(title, return_exceptions=True)

    async def _cancelled_event(
        self,
        turn_id: UUID,
        partial_chunks: list[str],
        pending_tool_calls: list[ModelToolCall],
    ) -> AgentEvent:
        await self._persist_cancelled_state(partial_chunks, pending_tool_calls)
        return self._event(
            turn_id,
            "turn_cancelled",
            TurnCancelledPayload(partial_content="".join(partial_chunks)),
        )

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
            except (OSError, UnicodeError, ValueError):
                pass
        try:
            await self._repair_unfinished_tool_calls(pending_tool_calls)
        except (OSError, UnicodeError, ValueError):
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
            except (OSError, UnicodeError, ValueError):
                if await self._session_message_publication(repair_message) is True:
                    pending_tool_calls.pop(0)
                    continue
                raise
            pending_tool_calls.pop(0)

    async def _session_message_publication(
        self,
        expected: SessionMessage,
    ) -> bool | None:
        """Return whether an exact Session message is durable, or None if unknowable."""
        try:
            session = await self._sessions.load(self._session_id)
        except (OSError, UnicodeError, ValueError):
            return None
        same_id = tuple(message for message in session.messages if message.id == expected.id)
        if any(message == expected for message in same_id):
            return True
        if same_id:
            return None
        return False

    @staticmethod
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

    @staticmethod
    async def _close_provider_stream(
        stream: AsyncIterator[ModelStreamEvent] | None,
    ) -> None:
        if stream is None:
            return
        close = getattr(stream, "aclose", None)
        if close is None:
            return
        try:
            await close()
        except Exception:
            pass

    def _event(
        self,
        turn_id: UUID,
        event_type: AgentEventType,
        payload: AgentEventPayload,
    ) -> AgentEvent:
        event = AgentEvent(
            type=event_type,
            event_id=self._next_event_id,
            turn_id=turn_id,
            created_at=self._now(),
            payload=payload,
        )
        self._next_event_id += 1
        return event

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


def _consume_task_exception(task: asyncio.Future[None]) -> None:
    if not task.cancelled():
        task.exception()


def _tool_activity_summary(action: str, tool_name: str) -> str:
    return " ".join(f"{action} {tool_name}".split())[:240]


def _unexpected_provider_failure() -> ModelCallError:
    return ModelCallError(
        ErrorInfo(
            code="model_failed",
            message="The model request failed.",
        )
    )


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
