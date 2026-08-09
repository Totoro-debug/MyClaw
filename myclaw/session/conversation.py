"""Command-line Conversation adapter over the Runtime Core Agent Run."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

from loguru import logger

from myclaw.agent.events import (
    AgentEvent,
    AgentEventPayload,
    AgentEventType,
    ConfirmationDecision,
    ConfirmationRequestedPayload,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.agent.run import (
    AgentRun,
    AgentRunCancelledPayload,
    AgentRunCompletedPayload,
    AgentRunConfirmationRequestedPayload,
    AgentRunFailedPayload,
    AgentRunInterface,
    AgentRunModelSettings,
    AgentRunPayload,
    AgentRunStartedPayload,
    AgentRunTextDeltaPayload,
    AgentRunToolCompletedPayload,
    AgentRunToolStartedPayload,
    ToolResultExternalizer,
    _log_artifact_failure,
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
from myclaw.tools.tool_gateway import ConfirmationChannel, ToolGateway

__all__ = ["ChatModelSettings", "StreamingConversationPort", "model_message_from_session"]


@dataclass(frozen=True, slots=True)
class ChatModelSettings:
    """Resolved provider-neutral fields needed for one chat request."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int
    context_window: int = 0


class StreamingConversationPort:
    """Expose one foreground Agent Run as ordered Agent Events."""

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
        agent_run: AgentRunInterface | None = None,
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
        self._agent_run = agent_run
        self._title_task: asyncio.Task[None] | None = None
        self._next_event_id = 0
        self._foreground_active = False
        self._active_task: asyncio.Task[object] | None = None
        self._active_turn_done: asyncio.Event | None = None
        self._cancel_requested = False
        self._close_task: asyncio.Task[None] | None = None
        self._system_prompt = system_prompt
        self._history_preparer = history_preparer
        self._confirmation: ConfirmationChannel | None = None

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
        turn = self._submit_turn(text, self._new_uuid())
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

    async def _submit_turn(self, text: str, turn_id: UUID) -> AsyncGenerator[AgentEvent, None]:
        confirmation = ConfirmationChannel()
        self._confirmation = confirmation
        agent_run = self._agent_run
        if agent_run is None:
            agent_run = AgentRun(
                provider=self._provider,
                settings=AgentRunModelSettings(
                    model=self._settings.model,
                    max_output=self._settings.max_output,
                    temperature=self._settings.temperature,
                    reasoning_effort=self._settings.reasoning_effort,
                    timeout_seconds=self._settings.timeout_seconds,
                    context_window=self._settings.context_window,
                ),
                now=self._now,
                new_uuid=self._new_uuid,
                system_prompt=self._system_prompt,
                tool_gateway=self._tool_gateway,
                externalize_result=self._externalize_result,
                memory_snapshot=self._memory_snapshot,
                system_prompt_for_memory=self._system_prompt_for_memory,
                history_preparer=self._history_preparer,
                history_preparer_for_memory=self._history_preparer_for_memory,
                after_user_published=self._start_title_for_first_user,
                on_terminal_failure=_log_terminal_failure,
                on_artifact_failure=lambda error, tool_name: _log_artifact_failure(
                    error,
                    tool_name=tool_name,
                ),
                cancel_requested=lambda: self._cancel_requested,
            )
            self._agent_run = agent_run
        payloads = agent_run.run_agent(
            self._session,
            text,
            route="chat",
            stream=True,
            confirmation=confirmation,
        )
        try:
            async for payload in payloads:
                event_type, event_payload = _map_agent_run_payload(payload, turn_id=turn_id)
                event = AgentEvent(
                    type=event_type,
                    event_id=self._next_event_id,
                    turn_id=turn_id,
                    created_at=self._now(),
                    payload=event_payload,
                )
                self._next_event_id += 1
                yield event
        finally:
            close = getattr(payloads, "aclose", None)
            if close is not None:
                await close()
            confirmation.close()
            if self._confirmation is confirmation:
                self._confirmation = None

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        confirmation = self._confirmation
        if confirmation is None:
            raise ValueError("No foreground confirmation request is pending")
        confirmation.respond_to_confirmation(confirmation_id, decision)

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
        confirmation = self._confirmation
        if confirmation is not None:
            confirmation.close()
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


def _map_agent_run_payload(
    payload: AgentRunPayload,
    *,
    turn_id: UUID | None = None,
) -> tuple[AgentEventType, AgentEventPayload]:
    if isinstance(payload, AgentRunStartedPayload):
        return "turn_started", TurnStartedPayload()
    if isinstance(payload, AgentRunTextDeltaPayload):
        return "text_delta", TextDeltaPayload(delta=payload.delta)
    if isinstance(payload, AgentRunToolStartedPayload):
        return "tool_started", ToolStartedPayload(
            tool_call_id=payload.tool_call_id,
            tool_name=payload.tool_name,
            summary=payload.summary,
        )
    if isinstance(payload, AgentRunConfirmationRequestedPayload):
        request = payload.request
        if turn_id is None:
            raise RuntimeError("confirmation event is missing its Agent Run turn identity")
        return "confirmation_requested", ConfirmationRequestedPayload(
            confirmation_id=request.confirmation_id,
            turn_id=turn_id,
            tool_call_id=request.tool_call_id,
            tool_name=request.tool_name,
            summary=request.summary,
            details=request.details,
            warnings=request.warnings,
        )
    if isinstance(payload, AgentRunToolCompletedPayload):
        return "tool_completed", ToolCompletedPayload(
            tool_call_id=payload.tool_call_id,
            tool_name=payload.tool_name,
            status=payload.status,
            summary=payload.summary,
        )
    if isinstance(payload, AgentRunCompletedPayload):
        return "turn_completed", TurnCompletedPayload(content=payload.content, usage=payload.usage)
    if isinstance(payload, AgentRunFailedPayload):
        return "turn_failed", TurnFailedPayload(error=payload.error)
    if isinstance(payload, AgentRunCancelledPayload):
        return "turn_cancelled", TurnCancelledPayload(partial_content=payload.partial_content)
    raise TypeError("Unsupported Agent Run payload")


def _log_terminal_failure(error: BaseException) -> None:
    if isinstance(error, ModelCallError):
        logger.opt(exception=error).error(
            "Agent Run failed code={} type={}",
            error.error.code,
            type(error).__name__,
        )
        return
    logger.opt(exception=error).error(
        "Agent Run failed code=model_failed type={}",
        type(error).__name__,
    )
