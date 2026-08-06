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
from myclaw.session.session import Session
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
        session: Session,
        settings: ChatModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str = "",
        title_prompt: str | None = None,
        title_new_uuid: Callable[[], UUID] = uuid4,
        tool_gateway: ToolGateway | None = None,
        history_preparer: Callable[[Session], Awaitable[Session]] | None = None,
        memory_snapshot: Callable[[], str] | None = None,
        system_prompt_for_memory: Callable[[str], str] | None = None,
        history_preparer_for_memory: (
            Callable[[str], Callable[[Session], Awaitable[Session]]] | None
        ) = None,
        externalize_result: ToolResultExternalizer | None = None,
        workspace_state: WorkspaceState | None = None,
        title_log_ready: Callable[[], Awaitable[object]] | None = None,
    ) -> None:
        self._provider = provider
        self._session = session
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._title_prompt = title_prompt
        self._title_new_uuid = title_new_uuid
        self._tool_gateway = tool_gateway
        self._externalize_result = externalize_result
        self._workspace_state = workspace_state
        self._title_log_ready = title_log_ready
        self._memory_snapshot = memory_snapshot
        self._system_prompt_for_memory = system_prompt_for_memory
        self._history_preparer_for_memory = history_preparer_for_memory
        self._title_task: asyncio.Task[None] | None = None
        self._next_event_id = 0
        self._foreground_active = False
        self._active_task: asyncio.Task[object] | None = None
        self._active_turn_done: asyncio.Event | None = None
        self._cancel_requested = False
        self._close_task: asyncio.Task[None] | None = None
        self._system_prompt = system_prompt
        self._history_preparer = history_preparer

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
        memory_snapshot = (
            None if self._memory_snapshot is None else self._memory_snapshot()
        )
        system_prompt = self._system_prompt
        history_preparer = self._history_preparer
        if memory_snapshot is not None:
            if self._system_prompt_for_memory is None:
                raise RuntimeError("Memory snapshot requires a System Prompt factory")
            system_prompt = self._system_prompt_for_memory(memory_snapshot)
            if self._history_preparer_for_memory is not None:
                history_preparer = self._history_preparer_for_memory(memory_snapshot)
        turn = AgentTurn(
            lane="foreground",
            provider=self._provider,
            session=self._session,
            settings=self._settings,
            now=self._now,
            new_uuid=self._new_uuid,
            system_prompt=system_prompt,
            tool_gateway=self._tool_gateway,
            history_preparer=history_preparer,
            after_user_published=self._start_title_for_first_user,
            cancel_requested=lambda: self._cancel_requested,
            externalize_result=self._externalize_result,
        )
        turn_id = self._new_uuid()
        payloads = turn.run(text)
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

    def _start_title_for_first_user(self, published: Session) -> None:
        if self._title_prompt is None or self._title_task is not None:
            return
        if len(published.messages) != 1:
            return
        first_message = published.messages[0]
        content = first_message.get("content")
        if first_message.get("role") != "user" or not isinstance(content, str):
            return
        title_task = asyncio.create_task(
            self._run_active_title_task(
                session=published,
                first_user_content=content,
                request_id=self._title_new_uuid(),
            )
        )
        title_task.add_done_callback(_consume_task_exception)
        self._title_task = title_task

    async def _run_active_title_task(
        self,
        *,
        session: Session,
        first_user_content: str,
        request_id: UUID,
    ) -> None:
        if self._title_log_ready is not None:
            await self._title_log_ready()
        correlation = (
            session_log(session)
            if self._workspace_state is not None
            else logger.contextualize(session_id=session.session_id)
        )
        with correlation:
            try:
                await self._generate_title(
                    session=session,
                    first_user_content=first_user_content,
                    request_id=request_id,
                )
            except asyncio.CancelledError:
                if session.metadata.get("title") == "Untitled session":
                    session.update_metadata(title=first_user_content)
                raise
            except Exception as error:
                logger.opt(exception=error).error(
                    "Session title task failed type={}", type(error).__name__
                )

    async def _generate_title(
        self,
        *,
        session: Session,
        first_user_content: str,
        request_id: UUID,
    ) -> None:
        title_candidate = first_user_content
        usage_delta: ModelUsage | None = None
        request = ModelRequest(
            request_id=request_id,
            route="chat",
            system_prompt=self._title_prompt or "",
            messages=(UserModelMessage(content=Session._normalize_title(first_user_content)),),
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
                    generated = Session._normalize_title_candidate(
                        model_event.response.message.content
                    )
                    if generated:
                        title_candidate = generated
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
        token_delta = None if usage_delta is None else {"model_calls": 1, **usage_delta.to_dict()}
        session.update_metadata(title=title_candidate, usage_delta=token_delta)

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
