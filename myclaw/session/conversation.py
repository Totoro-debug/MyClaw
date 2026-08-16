"""Command-line Conversation adapter over the Runtime Core Agent Run."""

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from uuid import UUID

from loguru import logger

from myclaw.agent.context import ContextBuilder
from myclaw.agent.events import (
    AgentEvent,
    AgentEventPayload,
    AgentEventType,
    ConfirmationDecision,
    ConfirmationRequestedPayload,
    ModelCallCompletedPayload,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.agent.prompts import current_user_input
from myclaw.agent.run import (
    AgentRun,
    AgentRunCancelledPayload,
    AgentRunCompletedPayload,
    AgentRunConfirmationRequestedPayload,
    AgentRunFailedPayload,
    AgentRunInterface,
    AgentRunModelCallCompletedPayload,
    AgentRunModelSettings,
    AgentRunPayload,
    AgentRunStartedPayload,
    AgentRunTextDeltaPayload,
    AgentRunToolCompletedPayload,
    AgentRunToolStartedPayload,
    SummaryPreparer,
    ToolResultExternalizer,
    _log_artifact_failure,
    model_message_from_session,
)
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log
from myclaw.provider.errors import ModelCallError
from myclaw.provider.models import (
    DirectModelProvider,
    ModelCompleted,
    ModelMessages,
    ModelProvider,
    ModelStreamEvent,
    ModelUsage,
    ReasoningEffort,
    accepts_direct_provider_call,
    legacy_request_from_direct,
)
from myclaw.session.session import Session
from myclaw.tools.base import OpenAIToolSchema
from myclaw.tools.tool_gateway import ConfirmationChannel, ToolGateway, ToolResultStatus

__all__ = ["ChatModelSettings", "StreamingConversationPort", "model_message_from_session"]


@dataclass(frozen=True, slots=True)
class ChatModelSettings:
    """Resolved provider-neutral fields needed for one chat request."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


type ForegroundSummaryPreparer = Callable[
    [Session, dict[str, Any]],
    Awaitable[Session],
]


_CONVERSATION_STREAM_DONE = object()


class _ConversationEventBridge:
    """Project one awaitable Agent Run while delaying its terminal payload."""

    def __init__(
        self,
        conversation: "StreamingConversationPort",
        *,
        turn_id: UUID,
        queue: asyncio.Queue[object],
    ) -> None:
        self._conversation = conversation
        self._turn_id = turn_id
        self._queue = queue
        self._terminal: AgentRunPayload | None = None
        self._completed_tool_call_ids: set[str] = set()
        self._confirmation_tool_call_ids: set[str] = set()
        self._start_consumed = asyncio.Event()

    @property
    def terminal(self) -> AgentRunPayload | None:
        return self._terminal

    def start(self) -> None:
        self._queue.put_nowait(
            self._conversation._event_from_payload(
                AgentRunStartedPayload(),
                turn_id=self._turn_id,
            )
        )

    async def wait_for_start_consumed(self) -> None:
        await self._start_consumed.wait()

    def release_after_start(self) -> None:
        self._start_consumed.set()

    async def emit(self, payload: AgentRunPayload) -> None:
        if isinstance(payload, AgentRunStartedPayload):
            return
        if isinstance(
            payload,
            (AgentRunCompletedPayload, AgentRunFailedPayload, AgentRunCancelledPayload),
        ):
            if self._terminal is not None:
                raise RuntimeError("Agent Run emitted more than one terminal payload")
            self._terminal = payload
            return
        if isinstance(payload, AgentRunToolCompletedPayload):
            self._completed_tool_call_ids.add(payload.tool_call_id)
        if isinstance(payload, AgentRunConfirmationRequestedPayload):
            self._confirmation_tool_call_ids.add(payload.request.tool_call_id)
        await self._queue.put(
            self._conversation._event_from_payload(payload, turn_id=self._turn_id)
        )
        await asyncio.sleep(0)

    def set_terminal(self, payload: AgentRunPayload) -> None:
        if not isinstance(
            payload,
            (AgentRunCompletedPayload, AgentRunFailedPayload, AgentRunCancelledPayload),
        ):
            raise TypeError("Conversation terminal payload has an unsupported type")
        if self._terminal is not None:
            raise RuntimeError("Agent Run terminal payload is already set")
        self._terminal = payload

    async def publish_terminal(self) -> None:
        payload = self._terminal
        if payload is None:
            raise RuntimeError("Agent Run completed without a terminal payload")
        await self._queue.put(
            self._conversation._event_from_payload(payload, turn_id=self._turn_id)
        )

    async def publish_repaired_tools(self, increment: list[dict[str, Any]]) -> None:
        for message in increment:
            if message.get("role") != "tool":
                continue
            tool_call_id = message.get("tool_call_id")
            tool_name = message.get("name")
            status = message.get("status")
            content = message.get("content")
            if not all(isinstance(value, str) for value in (tool_call_id, tool_name, status)):
                continue
            if cast(str, tool_call_id) in self._completed_tool_call_ids:
                continue
            if cast(str, tool_call_id) not in self._confirmation_tool_call_ids:
                continue
            if status not in {"success", "error", "refused"}:
                continue
            summary = (
                " ".join(content.split())[:240]
                if status == "error" and isinstance(content, str)
                else f"Finished {tool_name}"[:240]
            )
            await self.emit(
                AgentRunToolCompletedPayload(
                    tool_call_id=cast(str, tool_call_id),
                    tool_name=cast(str, tool_name),
                    status=cast(ToolResultStatus, status),
                    summary=summary,
                )
            )


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
        tool_gateway: ToolGateway | None = None,
        agent_run: AgentRunInterface | None = None,
        summary_preparer: SummaryPreparer | None = None,
        foreground_summary_preparer: ForegroundSummaryPreparer | None = None,
        context_builder: ContextBuilder | None = None,
        memory_snapshot: Callable[[], str] | None = None,
        system_prompt_for_memory: Callable[[str], str] | None = None,
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
        self._tool_gateway = tool_gateway
        self._externalize_result = externalize_result
        self._workspace_state = workspace_state
        self._title_log_ready = title_log_ready
        self._memory_snapshot = memory_snapshot
        self._system_prompt_for_memory = system_prompt_for_memory
        self._summary_preparer = summary_preparer
        self._foreground_summary_preparer = foreground_summary_preparer
        self._context_builder = context_builder
        self._agent_run = agent_run
        self._title_task: asyncio.Task[None] | None = None
        self._next_event_id = 0
        self._foreground_active = False
        self._active_task: asyncio.Task[object] | None = None
        self._active_turn_done: asyncio.Event | None = None
        self._cancel_requested = False
        self._close_task: asyncio.Task[None] | None = None
        self._system_prompt = system_prompt
        self._confirmation: ConfirmationChannel | None = None
        self._execution_task: asyncio.Task[None] | None = None

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

    def has_active_turn(self) -> bool:
        return self._foreground_active

    async def _submit_turn(self, text: str, turn_id: UUID) -> AsyncGenerator[AgentEvent, None]:
        confirmation = ConfirmationChannel()
        self._confirmation = confirmation
        queue: asyncio.Queue[object] = asyncio.Queue()
        bridge = _ConversationEventBridge(self, turn_id=turn_id, queue=queue)
        bridge.start()
        execution = asyncio.create_task(
            self._execute_turn(
                text=text,
                current_user={"role": "user", "content": text},
                confirmation=confirmation,
                bridge=bridge,
                queue=queue,
            )
        )
        self._execution_task = execution
        try:
            await asyncio.sleep(0)
            while True:
                item = await queue.get()
                if item is _CONVERSATION_STREAM_DONE:
                    break
                if isinstance(item, BaseException):
                    raise item
                event = cast(AgentEvent, item)
                yield event
                if event.type == "turn_started":
                    bridge.release_after_start()
        finally:
            if not execution.done():
                self._cancel_requested = True
                confirmation.close()
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            if self._execution_task is execution:
                self._execution_task = None
            confirmation.close()
            if self._confirmation is confirmation:
                self._confirmation = None

    async def _execute_turn(
        self,
        *,
        text: str,
        current_user: dict[str, Any],
        confirmation: ConfirmationChannel,
        bridge: _ConversationEventBridge,
        queue: asyncio.Queue[object],
    ) -> None:
        try:
            await bridge.wait_for_start_consumed()
            self._start_title_for_first_user_content(text)
            messages = await self._prepare_foreground_messages(current_user)
            agent_run = self._get_agent_run()
            run = getattr(agent_run, "run", None)
            if callable(run):
                increment = await run(
                    messages,
                    current_user,
                    route="chat",
                    emitter=bridge,
                    confirmation=confirmation,
                )
                if not isinstance(increment, list):
                    raise TypeError("Agent Run increment must be a list")
                if bridge.terminal is None:
                    raise RuntimeError("Agent Run completed without a terminal payload")
                try:
                    self._commit_increment(increment)
                except Exception as failure:
                    logger.opt(exception=failure).error(
                        "Agent Run Session increment failed code=persistence_error type={}",
                        type(failure).__name__,
                    )
                    await queue.put(
                        ModelCallError(
                            ErrorInfo(
                                code="persistence_error",
                                message="The Conversation Session could not be updated.",
                            )
                        )
                    )
                    return
                await bridge.publish_repaired_tools(increment)
                await bridge.publish_terminal()
            else:
                run_agent = getattr(agent_run, "run_agent", None)
                if not callable(run_agent):
                    raise TypeError("Agent Run does not expose a supported execution method")
                payloads = run_agent(
                    self._session,
                    text,
                    route="chat",
                    stream=True,
                    confirmation=confirmation,
                )
                try:
                    async for payload in payloads:
                        await bridge.emit(payload)
                finally:
                    close = getattr(payloads, "aclose", None)
                    if close is not None:
                        await close()
                await bridge.publish_terminal()
        except ModelCallError as failure:
            if bridge.terminal is not None:
                await queue.put(failure)
                return
            await self._finish_preparation_failure(
                current_user=current_user,
                failure=failure,
                bridge=bridge,
            )
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            if bridge.terminal is not None:
                await queue.put(asyncio.CancelledError())
                return
            self._commit_increment([deepcopy(current_user)])
            bridge.set_terminal(AgentRunCancelledPayload(partial_content=""))
            await bridge.publish_terminal()
        except Exception as failure:
            if bridge.terminal is not None:
                await queue.put(failure)
                return
            await self._finish_preparation_failure(
                current_user=current_user,
                failure=ModelCallError(
                    ErrorInfo(
                        code="model_failed",
                        message="The model request failed.",
                    )
                ),
                bridge=bridge,
            )
        finally:
            try:
                queue.put_nowait(_CONVERSATION_STREAM_DONE)
            except RuntimeError:
                pass

    async def _prepare_foreground_messages(
        self,
        current_user: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if self._foreground_summary_preparer is not None:
            prepared = await self._foreground_summary_preparer(
                self._session,
                deepcopy(current_user),
            )
        elif self._summary_preparer is not None:
            prepared = await self._summary_preparer(
                self._session,
                "chat",
                self._system_prompt,
                self._tool_schemas(),
            )
        else:
            prepared = self._session
        if prepared is not self._session:
            raise RuntimeError("Conversation Summary replaced the active Session")

        history = self._session.messages[self._session.last_consolidated :]
        context_builder = self._context_builder
        if context_builder is not None:
            memory = "" if self._memory_snapshot is None else self._memory_snapshot()
            return context_builder.build_messages(
                history=history,
                current_user=current_user,
                session_id=self._session.session_id,
                long_term_memory=memory,
            )
        return [
            {"role": "system", "content": self._system_prompt},
            *[
                model_message.to_dict()
                for message in history
                if (model_message := model_message_from_session(message)) is not None
            ],
            {
                "role": "user",
                "content": current_user_input(
                    content=cast(str, current_user["content"]),
                    current_time=self._now(),
                    session_id=self._session.session_id,
                ),
            },
        ]

    def _get_agent_run(self) -> AgentRunInterface | object:
        agent_run = self._agent_run
        if agent_run is not None:
            return agent_run
        agent_run = AgentRun(
            provider=self._provider,
            settings=AgentRunModelSettings(
                model=self._settings.model,
                max_output=self._settings.max_output,
                temperature=self._settings.temperature,
                reasoning_effort=self._settings.reasoning_effort,
                timeout_seconds=self._settings.timeout_seconds,
            ),
            now=self._now,
            new_uuid=self._new_uuid,
            tool_gateway=self._tool_gateway,
            externalize_result=self._externalize_result,
            on_terminal_failure=_log_terminal_failure,
            on_artifact_failure=lambda error, tool_name: _log_artifact_failure(
                error,
                tool_name=tool_name,
            ),
            cancel_requested=lambda: self._cancel_requested,
        )
        self._agent_run = agent_run
        return agent_run

    def _commit_increment(self, increment: list[dict[str, Any]]) -> None:
        self._session.append_messages(increment)
        self._start_title_for_first_user(self._session)
        try:
            self._session.persist()
        except Exception:
            pass

    def _tool_schemas(self) -> tuple[OpenAIToolSchema, ...]:
        if self._tool_gateway is None:
            return ()
        return tuple(self._tool_gateway.schemas)

    def _event_from_payload(
        self,
        payload: AgentRunPayload,
        *,
        turn_id: UUID,
    ) -> AgentEvent:
        event_type, event_payload = _map_agent_run_payload(payload, turn_id=turn_id)
        event = AgentEvent(
            type=event_type,
            event_id=self._next_event_id,
            turn_id=turn_id,
            created_at=self._now(),
            payload=event_payload,
        )
        self._next_event_id += 1
        return event

    async def _finish_preparation_failure(
        self,
        *,
        current_user: dict[str, Any],
        failure: ModelCallError,
        bridge: _ConversationEventBridge,
    ) -> None:
        _log_terminal_failure(failure)
        self._commit_increment(
            [
                deepcopy(current_user),
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [],
                    "status": "error",
                    "error": {
                        "code": failure.error.code,
                        "message": failure.error.message,
                    },
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                    },
                },
            ]
        )
        bridge.set_terminal(AgentRunFailedPayload(error=failure.error))
        await bridge.publish_terminal()

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
        self._start_title_task(published, content)

    def _start_title_for_first_user_content(self, content: str) -> None:
        if self._title_prompt is None or self._title_task is not None:
            return
        if self._session.messages:
            return
        self._start_title_task(self._session, content)

    def _start_title_task(self, session: Session, content: str) -> None:
        title_task = asyncio.create_task(
            self._run_active_title_task(
                session=session,
                first_user_content=content,
            )
        )
        title_task.add_done_callback(_consume_task_exception)
        self._title_task = title_task

    async def _run_active_title_task(
        self,
        *,
        session: Session,
        first_user_content: str,
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
    ) -> None:
        title_candidate = first_user_content
        usage_delta: ModelUsage | None = None
        provider_stream: AsyncIterator[ModelStreamEvent] | None = None
        fallback_reason: str | None = "incomplete_stream"
        fallback_type = "ModelStream"
        try:
            provider_stream = self._title_stream(
                Session._normalize_title(first_user_content),
            )
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

    def _title_stream(self, normalized_user_content: str) -> AsyncIterator[ModelStreamEvent]:
        messages: ModelMessages = [
            {"role": "system", "content": self._title_prompt or ""},
            {"role": "user", "content": normalized_user_content},
        ]
        route_status = getattr(self._provider, "route_status", None)
        if callable(route_status):
            return cast(
                AsyncIterator[ModelStreamEvent],
                cast(Any, self._provider).stream("chat", messages=messages, tools=()),
            )
        method = cast(Any, self._provider).stream
        if accepts_direct_provider_call(method):
            return cast(DirectModelProvider, self._provider).stream(
                messages=messages,
                tools=(),
                model=self._settings.model,
                max_output=self._settings.max_output,
                temperature=self._settings.temperature,
                reasoning_effort=self._settings.reasoning_effort,
                timeout=self._settings.timeout_seconds,
            )
        return self._provider.stream(
            legacy_request_from_direct(
                route="chat",
                messages=messages,
                tools=(),
                model=self._settings.model,
                max_output=self._settings.max_output,
                temperature=self._settings.temperature,
                reasoning_effort=self._settings.reasoning_effort,
                timeout=self._settings.timeout_seconds,
                stream=True,
            )
        )

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
        execution = self._execution_task
        if execution is not None and not execution.done():
            if execution is not asyncio.current_task():
                execution.cancel()

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
    if isinstance(payload, AgentRunModelCallCompletedPayload):
        return "model_call_completed", ModelCallCompletedPayload(
            content=payload.content,
            continues_with_tools=payload.continues_with_tools,
        )
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
            reason=request.reason,
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
