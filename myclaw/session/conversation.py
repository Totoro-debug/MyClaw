"""Command-line Conversation adapter over the Runtime Core Agent turn."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from loguru import logger

from myclaw.agent.events import AgentEvent
from myclaw.agent.turn import (
    AgentTurn,
    ToolResultExternalizer,
    agent_turn_event_type,
    model_message_from_session,
)
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.logging.session import session_log
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    ModelCompleted,
    ModelProvider,
    ModelRequest,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    UserModelMessage,
)
from myclaw.session.records import ConversationSession, MetadataUpdate, UserSessionMessage
from myclaw.session.session_store import SessionStore
from myclaw.session.session_titles import normalize_session_title
from myclaw.tools.tool_gateway import ToolGateway

__all__ = ["ChatModelSettings", "StreamingConversationPort", "model_message_from_session"]




@dataclass(frozen=True, slots=True)
class ChatModelSettings:
    """Resolved provider-neutral fields needed for one chat request."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


class StreamingConversationPort:
    """Expose one foreground Agent turn as ordered Agent Events."""

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
        externalize_result: ToolResultExternalizer | None = None,
        workspace_state: WorkspaceState | None = None,
        title_log_ready: Callable[[], Awaitable[object]] | None = None,
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._session_id = session_id
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._title_prompt = title_prompt
        self._title_new_uuid = title_new_uuid
        self._workspace_state = workspace_state
        self._title_log_ready = title_log_ready
        self._title_task: asyncio.Task[None] | None = None
        self._next_event_id = 0
        self._foreground_active = False
        self._active_task: asyncio.Task[object] | None = None
        self._active_turn_done: asyncio.Event | None = None
        self._cancel_requested = False
        self._close_task: asyncio.Task[None] | None = None
        self._turn = AgentTurn(
            lane="foreground",
            provider=provider,
            sessions=sessions,
            session_id=session_id,
            settings=settings,
            now=now,
            new_uuid=new_uuid,
            system_prompt=system_prompt,
            tool_gateway=tool_gateway,
            history_preparer=history_preparer,
            after_user_published=self._start_title_for_first_user,
            cancel_requested=lambda: self._cancel_requested,
            externalize_result=externalize_result,
        )

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
        payloads = self._turn.run(text)
        try:
            async for payload in payloads:
                event = AgentEvent(
                    type=agent_turn_event_type(payload),
                    event_id=self._next_event_id,
                    turn_id=turn_id,
                    created_at=self._now(),
                    payload=payload,
                )
                self._next_event_id += 1
                yield event
        finally:
            await payloads.aclose()

    def _start_title_for_first_user(self, user_message: UserSessionMessage) -> None:
        if self._title_prompt is None or self._title_task is not None:
            return
        title_task = asyncio.create_task(
            self._run_title_task(
                session_id=self._session_id,
                first_user_id=user_message.id,
                first_user_content=user_message.content,
                request_id=self._title_new_uuid(),
            )
        )
        title_task.add_done_callback(_consume_task_exception)
        self._title_task = title_task

    async def _run_title_task(
        self,
        *,
        session_id: str,
        first_user_id: str,
        first_user_content: str,
        request_id: UUID,
    ) -> None:
        if self._title_log_ready is not None:
            await self._title_log_ready()
        correlation = (
            session_log(self._workspace_state, session_id)
            if self._workspace_state is not None
            else logger.contextualize(session_id=session_id)
        )
        with correlation:
            await self._run_title_task_inner(
                session_id=session_id,
                first_user_id=first_user_id,
                first_user_content=first_user_content,
                request_id=request_id,
            )

    async def _run_title_task_inner(
        self,
        *,
        session_id: str,
        first_user_id: str,
        first_user_content: str,
        request_id: UUID,
    ) -> None:
        try:
            await self._generate_title_for_first_user(
                session_id=session_id,
                first_user_id=first_user_id,
                first_user_content=first_user_content,
                request_id=request_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            logger.opt(exception=error).error(
                "Session title task failed type={}", type(error).__name__
            )
            raise

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
        fallback_reason: str | None = "incomplete_stream"
        fallback_type = "ModelStream"
        try:
            provider_stream = self._provider.stream(request)
            async for model_event in provider_stream:
                if not isinstance(model_event, ModelCompleted):
                    continue
                fallback_type = "ModelResponse"
                if model_event.response.message.tool_calls:
                    fallback_reason = "tool_calls"
                else:
                    generated = normalize_session_title(model_event.response.message.content)
                    if generated:
                        title = generated
                        fallback_reason = None
                    else:
                        fallback_reason = "empty_title"
                usage_delta = model_event.response.usage
                break
        except Exception as failure:
            fallback_reason = None
            code = failure.error.code if isinstance(failure, ModelCallError) else "model_failed"
            logger.opt(exception=failure).warning(
                "Session title fallback selected code={} type={}",
                code,
                type(failure).__name__,
            )
        finally:
            await _close_provider_stream(provider_stream)
        if fallback_reason is not None:
            logger.warning(
                "Session title fallback selected code=model_failed type={} reason={}",
                fallback_type,
                fallback_reason,
            )
        try:
            await self._sessions.update_metadata(
                session_id,
                MetadataUpdate(
                    title=title,
                    updated_at=self._persisted_now(),
                    usage_delta=usage_delta,
                ),
            )
        except (OSError, UnicodeError, ValueError) as failure:
            _log_title_persistence_failure(failure, operation="metadata_update")

    async def cancel_active_turn(self) -> None:
        if self._cancel_requested:
            return
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

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)


async def _close_provider_stream(stream: AsyncIterator[ModelStreamEvent] | None) -> None:
    if stream is None:
        return
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception as error:
        logger.opt(exception=error).warning(
            "Session title stream cleanup failed type={}", type(error).__name__
        )


def _consume_task_exception(task: asyncio.Future[None]) -> None:
    if not task.cancelled():
        task.exception()


def _log_title_persistence_failure(failure: Exception, *, operation: str) -> None:
    logger.opt(exception=failure).error(
        "Session title failed code=persistence_error operation={} type={}",
        operation,
        type(failure).__name__,
    )
