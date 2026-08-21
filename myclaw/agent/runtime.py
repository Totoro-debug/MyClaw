"""Composition for one prepared command-line Conversation Session."""

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, Protocol, cast
from uuid import UUID

from loguru import logger
from tzlocal import get_localzone_name

from myclaw.agent.context import ContextBuilder
from myclaw.agent.events import (
    AgentEvent,
    ConfirmationDecision,
    ConfirmationRequestedPayload,
    ConversationPort,
    ModelCallCompletedPayload,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.agent.loop import AgentLoop
from myclaw.agent.message_bus import InboundMessage, OutboundMessage
from myclaw.agent.prompts import (
    chat_system_prompt,
    current_user_input,
    runtime_context,
    session_title_prompt,
)
from myclaw.agent.run import AgentRunRouter, ToolResultExternalizer
from myclaw.agent.runner import AgentRunnerResult, AgentRunnerRoute
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState, WorkspaceStateError
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ProviderConfiguration, UserConfiguration
from myclaw.errors import ErrorCode, ErrorInfo
from myclaw.logging.session import session_log, without_session_log
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import (
    ManagementViewService,
    ResolvedChatStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
)
from myclaw.memory.conversation_summary import (
    ConversationSummaryManager,
    WorkspaceJsonlSummaryStore,
    _last_user_index,
)
from myclaw.memory.memory_scheduler import MemoryTaskScheduler
from myclaw.memory.memory_task import (
    MemoryManager,
    RuntimeMemory,
    WorkspaceFileMemoryStore,
)
from myclaw.provider.model_router import Jitter, ModelRouter, RetryClock
from myclaw.provider.models import ModelProvider, ModelUsage
from myclaw.schedule.service import ScheduleClock, ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.conversation import (
    ForegroundSummaryPreparer,
    StreamingConversationPort,
)
from myclaw.session.projection import project_session_message
from myclaw.session.session import Session
from myclaw.terminal.repl import ManagementDispatcher, ProgressiveWriter, ReplInput, run_repl
from myclaw.tools.base import BaseTool, OpenAIToolSchema
from myclaw.tools.tool_gateway import (
    ConfirmationRequest,
    ToolGateway,
    ToolResult,
    ToolResultStatus,
)
from myclaw.utils.scheduler import AsyncioSchedulerClock, SchedulerClock


class _ClosableConversationPort(ConversationPort, Protocol):
    async def close(self) -> None: ...


class _RuntimeSchedulerOwner:
    """Own one terminal scheduler instance per Runtime run."""

    def __init__(
        self,
        factory: Callable[[], MemoryTaskScheduler | ScheduleService],
    ) -> None:
        self._factory = factory
        self._active: MemoryTaskScheduler | ScheduleService | None = None

    def start(self) -> None:
        scheduler = self._active
        if scheduler is None:
            scheduler = self._factory()
            self._active = scheduler
        scheduler.start()

    async def close(self) -> None:
        scheduler = self._active
        if scheduler is None:
            return
        await scheduler.close()
        if self._active is scheduler:
            self._active = None


@dataclass(slots=True)
class _RuntimeLifetime:
    started: bool = False
    close_task: asyncio.Task[None] | None = None
    shutdown_requested: asyncio.Event = field(default_factory=asyncio.Event)
    run_task: asyncio.Task[object] | None = None
    run_done: asyncio.Event | None = None

    def begin(self) -> None:
        if self.started:
            raise RuntimeError("Prepared Runtime is closed")
        self.started = True


@dataclass(frozen=True, slots=True)
class PreparedRuntime:
    """An in-memory Session identity and its injectable REPL composition."""

    conversation: _ClosableConversationPort
    agent_loop: AgentLoop
    management_dispatcher: ManagementDispatcher
    schedule_service: ScheduleService
    _memory_scheduler: _RuntimeSchedulerOwner
    _router: ModelRouter
    _lifetime: _RuntimeLifetime

    @property
    def session_id(self) -> str:
        return self.agent_loop.session.session_id

    @property
    def session(self) -> Session:
        return self.agent_loop.session

    async def start(self) -> None:
        self._lifetime.begin()
        await self._start_schedulers()

    async def _start_schedulers(self) -> None:
        with without_session_log():
            try:
                await self.agent_loop.start()
                self._memory_scheduler.start()
                self.schedule_service.start()
            except BaseException as error:
                logger.opt(exception=error).error(
                    "Runtime startup failed type={}", type(error).__name__
                )
                raise

    async def run(
        self,
        *,
        input_reader: ReplInput,
        writer: ProgressiveWriter,
        management_dispatcher: ManagementDispatcher | None = None,
    ) -> None:
        dispatcher = (
            self.management_dispatcher if management_dispatcher is None else management_dispatcher
        )
        self._lifetime.begin()
        running = asyncio.current_task()
        if running is None:
            raise RuntimeError("Prepared Runtime requires an asyncio Task")
        run_done = asyncio.Event()
        self._lifetime.run_task = running
        self._lifetime.run_done = run_done
        try:
            try:
                await self._start_schedulers()
                await run_repl(
                    conversation=self.conversation,
                    input_reader=input_reader,
                    writer=writer,
                    management_dispatcher=dispatcher,
                    shutdown_requested=self._lifetime.shutdown_requested,
                )
            except BaseException as primary_error:
                if self._lifetime.close_task is None:
                    try:
                        await self.close()
                    except BaseException as cleanup_error:
                        raise primary_error from cleanup_error
                raise
            else:
                if self._lifetime.close_task is None:
                    await self.close()
        finally:
            run_done.set()

    async def close(self) -> None:
        task = self._lifetime.close_task
        if task is None:
            self._lifetime.started = True
            self._lifetime.shutdown_requested.set()
            running = self._lifetime.run_task
            current = asyncio.current_task()
            wait_for_run = (
                self._lifetime.run_done
                if running is not None and running is not current and not running.done()
                else None
            )
            task = asyncio.create_task(self._close_owned_resources(wait_for_run))
            self._lifetime.close_task = task
            if running is not None and running is not current and not running.done():
                running.cancel()
        await _await_shared_shutdown(task)

    async def _close_owned_resources(self, run_done: asyncio.Event | None) -> None:
        with without_session_log():
            failures: list[BaseException] = []
            try:
                await self.schedule_service.close()
            except BaseException as error:
                failures.append(error)

            conversation_close = getattr(self.conversation, "close", None)
            if not callable(conversation_close):
                raise RuntimeError("Prepared Runtime conversation does not support close")
            shutdowns: list[Awaitable[object]] = [
                self._memory_scheduler.close(),
                cast(Callable[[], Awaitable[None]], conversation_close)(),
            ]
            if run_done is not None:
                shutdowns.append(run_done.wait())
            results = await asyncio.gather(*shutdowns, return_exceptions=True)
            failures.extend(result for result in results if isinstance(result, BaseException))

            try:
                await self._router.close()
            except BaseException as error:
                failures.append(error)

            self.agent_loop._close_sessions()
            if not failures:
                return
            failure = (
                failures[0]
                if len(failures) == 1
                else BaseExceptionGroup("Runtime shutdown failed", failures)
            )
            logger.opt(exception=failure).error(
                "Runtime shutdown failed type={}", type(failure).__name__
            )
            raise failure


PreparedReplRuntime = PreparedRuntime


class _DeferredConversationPort:
    def __init__(
        self,
        *,
        model: AgentRunRouter,
        session: Session,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        title_prompt: str,
        tool_gateway: ToolGateway,
        on_foreground_terminal: Callable[[], None],
        foreground_summary_preparer: ForegroundSummaryPreparer | None = None,
        context_builder: ContextBuilder | None = None,
        memory_snapshot: Callable[[], str] | None = None,
        before_submit: Callable[[], Awaitable[None]] | None = None,
        externalize_result: ToolResultExternalizer | None = None,
    ) -> None:
        self._model = model
        self._session = session
        self._now = now
        self._new_uuid = new_uuid
        self._title_prompt = title_prompt
        self._tool_gateway = tool_gateway
        self._foreground_summary_preparer = foreground_summary_preparer
        self._context_builder = context_builder
        self._memory_snapshot = memory_snapshot
        self._before_submit = before_submit
        self._on_foreground_terminal = on_foreground_terminal
        self._externalize_result = externalize_result
        self._delegate: ConversationPort | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._active_task: asyncio.Task[object] | None = None
        self._active_done: asyncio.Event | None = None

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if not text.strip():
            return
        if self._active_task is not None:
            raise RuntimeError("A foreground turn is already active")
        active = asyncio.current_task()
        if active is None:
            raise RuntimeError("Conversation Port requires an asyncio Task")
        active_done = asyncio.Event()
        self._active_task = active
        self._active_done = active_done
        title_log_ready = asyncio.Event()
        correlation: AbstractContextManager[None] = session_log(self._session)
        with correlation:
            try:
                if self._before_submit is not None:
                    await self._before_submit()
                if self._close_task is not None:
                    raise RuntimeError("Conversation Port is closed")
                delegate = self._delegate
                if delegate is None:
                    delegate = StreamingConversationPort(
                        model=self._model,
                        session=self._session,
                        now=self._now,
                        new_uuid=self._new_uuid,
                        title_prompt=self._title_prompt,
                        tool_gateway=self._tool_gateway,
                        foreground_summary_preparer=self._foreground_summary_preparer,
                        context_builder=self._context_builder,
                        memory_snapshot=self._memory_snapshot,
                        externalize_result=self._externalize_result,
                        workspace_state=self._session.workspace_state,
                        title_log_ready=title_log_ready.wait,
                    )
                    self._delegate = delegate
                async for event in delegate.submit(text):
                    if event.type in {"turn_completed", "turn_failed", "turn_cancelled"}:
                        self._on_foreground_terminal()
                    yield event
            finally:
                if self._active_task is active:
                    self._active_task = None
                active_done.set()
                if self._active_done is active_done:
                    self._active_done = None
                title_log_ready.set()

    async def cancel_active_turn(self) -> None:
        delegate = self._delegate
        delegate_active = self._delegate_has_active_turn(delegate)
        if delegate is not None:
            await delegate.cancel_active_turn()
        active = self._active_task
        if (
            active is None
            or active is asyncio.current_task()
            or active.done()
            or active.cancelling()
        ):
            return
        if delegate is None or not delegate_active:
            active.cancel()

    def respond_to_confirmation(
        self, confirmation_id: UUID, decision: ConfirmationDecision
    ) -> None:
        delegate = self._delegate
        if delegate is None:
            raise ValueError("No foreground confirmation request is pending")
        delegate.respond_to_confirmation(confirmation_id, decision)

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_delegate())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_delegate(self) -> None:
        active = self._active_task
        active_done = self._active_done
        delegate = self._delegate
        if active is not None and active is not asyncio.current_task() and not active.done():
            delegate_active = self._delegate_has_active_turn(delegate)
            if delegate is None or not delegate_active:
                active.cancel()
            else:
                await delegate.cancel_active_turn()
        if active_done is not None:
            await active_done.wait()
        if delegate is None:
            return
        close = getattr(delegate, "close", None)
        if close is None:
            await delegate.cancel_active_turn()
        else:
            await close()

    @staticmethod
    def _delegate_has_active_turn(delegate: ConversationPort | None) -> bool:
        if delegate is None:
            return False
        has_active_turn = getattr(delegate, "has_active_turn", None)
        return bool(has_active_turn()) if callable(has_active_turn) else False


class _AgentLoopConversationAdapter:
    """Temporary Terminal bridge for the sparse foreground Message Bus protocol."""

    def __init__(
        self,
        *,
        loop: AgentLoop,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        self._loop = loop
        self._now = now
        self._new_uuid = new_uuid
        self._close_task: asyncio.Task[None] | None = None
        self._active = False
        self._active_done: asyncio.Event | None = None
        self._events: asyncio.Queue[object] | None = None
        self._turn_id: UUID | None = None
        self._next_event_id = 0
        self._model_calls_emitted = 0
        self._response_content: list[str] = []
        self._pending_tool: tuple[str, str] | None = None
        self._confirmation_ids: dict[str, UUID] = {}
        self._confirmation_decisions: dict[UUID, ConfirmationDecision] = {}
        self._terminal_seen = False
        self._tool_results: dict[str, ToolResult] = {}
        self._ready_events: deque[AgentEvent] = deque()
        self._terminal_result: AgentRunnerResult | None = None

    def has_active_turn(self) -> bool:
        return self._active

    def on_confirmation_requested(self, request: object) -> None:
        if not isinstance(request, ConfirmationRequest):
            raise TypeError("Agent Loop confirmation callback requires ConfirmationRequest")
        event_queue = self._events
        turn_id = self._turn_id
        if event_queue is None or turn_id is None:
            raise RuntimeError("No foreground Terminal turn is active")
        confirmation_id = request.confirmation_id
        self._confirmation_ids[request.tool_call_id] = confirmation_id
        event_queue.put_nowait(
            self._event(
                "confirmation_requested",
                ConfirmationRequestedPayload(
                    confirmation_id=confirmation_id,
                    turn_id=turn_id,
                    tool_call_id=request.tool_call_id,
                    tool_name=request.tool_name,
                    reason=request.reason,
                    summary=request.summary,
                    details=request.details,
                    warnings=request.warnings,
                ),
            )
        )

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if not text.strip():
            return
        if self._active:
            raise RuntimeError("A foreground turn is already active")
        await self._loop.start()
        active_done = asyncio.Event()
        self._active = True
        self._active_done = active_done
        self._events = asyncio.Queue()
        self._turn_id = self._new_uuid()
        self._model_calls_emitted = 0
        self._response_content = []
        self._pending_tool = None
        self._confirmation_ids.clear()
        self._confirmation_decisions.clear()
        self._terminal_seen = False
        self._tool_results.clear()
        self._ready_events.clear()
        self._terminal_result = None
        turn_id = self._turn_id
        assert turn_id is not None
        await self._loop.bus.put_inbound(InboundMessage(text))
        await asyncio.sleep(0)
        try:
            yield self._event("turn_started", TurnStartedPayload())
            while True:
                if self._ready_events:
                    yield self._ready_events.popleft()
                    continue
                outbound_task = asyncio.create_task(self._loop.bus.get_outbound())
                legacy_task = asyncio.create_task(self._events.get())
                done, pending = await asyncio.wait(
                    (outbound_task, legacy_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
                if outbound_task in done:
                    if legacy_task in done:
                        self._remember_ready_event(legacy_task.result())
                    outbound = outbound_task.result()
                    outbound_events = self._events_from_outbound(outbound)
                    if outbound.metadata.get("_streamed") is True:
                        while self._ready_events:
                            yield self._ready_events.popleft()
                    for event in outbound_events:
                        yield event
                    if outbound.metadata.get("_streamed") is True:
                        self._terminal_seen = True
                        return
                    continue
                self._remember_ready_event(legacy_task.result())
        finally:
            if self._active:
                if not self._terminal_seen:
                    await self._loop.cancel_active_run()
                self._active = False
            active_done.set()
            if self._active_done is active_done:
                self._active_done = None
            self._events = None
            self._turn_id = None
            self._ready_events.clear()

    def _remember_ready_event(self, event: object) -> None:
        if isinstance(event, AgentEvent):
            self._ready_events.append(event)

    async def cancel_active_turn(self) -> None:
        await self._loop.cancel_active_run()

    def on_tool_completed(self, result: ToolResult) -> None:
        self._tool_results[result.tool_call_id] = result
        pending = self._pending_tool
        if pending is None or pending[0] != result.tool_call_id:
            return
        self._pending_tool = None
        event_queue = self._events
        if event_queue is not None:
            event_queue.put_nowait(self._tool_event(result))

    def on_terminal(self, result: AgentRunnerResult | None) -> None:
        self._terminal_result = result

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        if confirmation_id not in self._confirmation_ids.values():
            raise ValueError("No foreground confirmation request is pending")
        self._confirmation_decisions[confirmation_id] = decision
        self._loop.respond_to_confirmation(confirmation_id, decision)

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_owned())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_owned(self) -> None:
        if self._active:
            await self._loop.cancel_active_run()
        active_done = self._active_done
        if active_done is not None:
            await active_done.wait()
        await self._loop.close()

    def _events_from_outbound(self, outbound: OutboundMessage) -> tuple[AgentEvent, ...]:
        if outbound.type == "model_response" and outbound.metadata.get("_stream_delta") is True:
            self._response_content.append(outbound.content)
            return (
                *self._flush_pending_tool(),
                self._event("text_delta", TextDeltaPayload(outbound.content)),
            )
        if outbound.type == "model_reasoning":
            return self._flush_pending_tool()
        if outbound.type == "tool_call":
            completed_tool = self._flush_pending_tool()
            model_call = self._emit_model_call(continues_with_tools=True)
            tool_call_id = outbound.metadata.get("tool_call_id")
            if not isinstance(tool_call_id, str):
                raise ValueError("Outbound tool call is missing tool_call_id")
            tool_name = outbound.content
            completed = self._tool_results.pop(tool_call_id, None)
            if completed is None:
                self._pending_tool = (tool_call_id, tool_name)
            else:
                self._pending_tool = None
            return (
                *completed_tool,
                *((model_call,) if model_call is not None else ()),
                self._event(
                    "tool_started",
                    ToolStartedPayload(
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                        summary=f"Running {tool_name}"[:240],
                    ),
                ),
                *((self._tool_event(completed),) if completed is not None else ()),
            )
        if outbound.type == "model_response" and outbound.metadata.get("_streamed") is True:
            result = self._terminal_result
            if result is None:
                raise RuntimeError("Completed foreground run is missing its terminal result")
            completed_tool = self._flush_pending_tool()
            completed_calls = self._emit_missing_completed_model_calls(result)
            usage = self._turn_usage(result)
            return (
                *completed_tool,
                *completed_calls,
                self._event(
                    "turn_completed",
                    TurnCompletedPayload(
                        content=result.final_content,
                        usage=usage,
                    ),
                ),
            )
        if outbound.type == "system_control" and outbound.metadata.get("_streamed") is True:
            finish_reason = outbound.metadata.get("finish_reason")
            error_code = outbound.metadata.get("error_code")
            if not isinstance(finish_reason, str) or not isinstance(error_code, str):
                raise ValueError("Outbound system control terminal metadata is malformed")
            completed_calls = (
                self._emit_missing_completed_model_calls(self._terminal_result)
                if error_code == "persistence_error" and self._terminal_result is not None
                else ()
            )
            if finish_reason == "cancelled":
                return (
                    *completed_calls,
                    self._event(
                        "turn_cancelled",
                        TurnCancelledPayload(partial_content="".join(self._response_content)),
                    ),
                )
            return (
                *completed_calls,
                self._event(
                    "turn_failed",
                    TurnFailedPayload(ErrorInfo(cast(ErrorCode, error_code), outbound.content)),
                ),
            )
        return ()

    def _emit_model_call(self, *, continues_with_tools: bool) -> AgentEvent | None:
        if self._events is None:
            return None
        event = self._event(
            "model_call_completed",
            ModelCallCompletedPayload(
                content="".join(self._response_content),
                continues_with_tools=continues_with_tools,
            ),
        )
        self._model_calls_emitted += 1
        self._response_content = []
        return event

    def _emit_missing_completed_model_calls(
        self,
        result: AgentRunnerResult,
    ) -> tuple[AgentEvent, ...]:
        assistants = [message for message in result.messages if message.get("role") == "assistant"]
        events: list[AgentEvent] = []
        while self._model_calls_emitted < len(assistants):
            message = assistants[self._model_calls_emitted]
            content = message.get("content")
            self._response_content = [content] if isinstance(content, str) else []
            event = self._emit_model_call(continues_with_tools=False)
            if event is not None:
                events.append(event)
        return tuple(events)

    def _flush_pending_tool(self) -> tuple[AgentEvent, ...]:
        pending = self._pending_tool
        if pending is None:
            return ()
        tool_call_id, tool_name = pending
        confirmation_id = self._confirmation_ids.get(tool_call_id)
        decision = (
            None if confirmation_id is None else self._confirmation_decisions.get(confirmation_id)
        )
        status: ToolResultStatus = "refused" if decision == "declined" else "success"
        event = (
            self._event(
                "tool_completed",
                ToolCompletedPayload(
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    status=status,
                    summary=(
                        "Tool confirmation was declined."
                        if status == "refused"
                        else f"Finished {tool_name}"
                    )[:240],
                ),
            ),
        )
        self._pending_tool = None
        return event

    def _tool_event(self, result: ToolResult) -> AgentEvent:
        if result.status == "error":
            summary = " ".join(result.content.split())[:240]
        elif result.status == "refused":
            summary = " ".join(result.content.split())[:240]
        else:
            summary = f"Finished {result.name}"[:240]
        return self._event(
            "tool_completed",
            ToolCompletedPayload(
                tool_call_id=result.tool_call_id,
                tool_name=result.name,
                status=result.status,
                summary=summary,
            ),
        )

    @staticmethod
    def _turn_usage(result: AgentRunnerResult) -> ModelUsage:
        return ModelUsage(
            input_tokens=result.usage["input_tokens"],
            output_tokens=result.usage["output_tokens"],
            total_tokens=result.usage["total_tokens"],
        )

    def _event(self, event_type: str, payload: object) -> AgentEvent:
        turn_id = self._turn_id
        if turn_id is None:
            raise RuntimeError("No foreground Terminal turn is active")
        event = AgentEvent(
            type=event_type,  # type: ignore[arg-type]
            event_id=self._next_event_id,
            turn_id=turn_id,
            created_at=self._now(),
            payload=payload,  # type: ignore[arg-type]
        )
        self._next_event_id += 1
        return event


def prepare_runtime(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
    provider_factory: Callable[[ProviderConfiguration], ModelProvider],
    now: Callable[[], datetime],
    new_uuid: Callable[[], UUID],
    retry_clock: RetryClock | None = None,
    retry_jitter: Jitter | None = None,
    memory_scheduler_clock: SchedulerClock | None = None,
    schedule_scheduler_clock: ScheduleClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    timezone_name: str | None = None,
) -> PreparedRuntime:
    """Prepare one Runtime and record terminal composition failures once."""
    try:
        return _prepare_runtime(
            agent_home=agent_home,
            workspace=workspace,
            configuration=configuration,
            provider_factory=provider_factory,
            now=now,
            new_uuid=new_uuid,
            retry_clock=retry_clock,
            retry_jitter=retry_jitter,
            memory_scheduler_clock=memory_scheduler_clock,
            schedule_scheduler_clock=schedule_scheduler_clock,
            monotonic_now=monotonic_now,
            timezone_name=timezone_name,
        )
    except WorkspaceStateError:
        raise
    except Exception as error:
        logger.opt(exception=error).error(
            "Runtime composition failed type={}", type(error).__name__
        )
        raise


def prepare_repl_runtime(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
    provider_factory: Callable[[ProviderConfiguration], ModelProvider],
    now: Callable[[], datetime],
    new_uuid: Callable[[], UUID],
    retry_clock: RetryClock | None = None,
    retry_jitter: Jitter | None = None,
    memory_scheduler_clock: SchedulerClock | None = None,
    schedule_scheduler_clock: ScheduleClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    timezone_name: str | None = None,
) -> PreparedReplRuntime:
    """Compatibility entry point for the prepared REPL runtime."""
    return prepare_runtime(
        agent_home=agent_home,
        workspace=workspace,
        configuration=configuration,
        provider_factory=provider_factory,
        now=now,
        new_uuid=new_uuid,
        retry_clock=retry_clock,
        retry_jitter=retry_jitter,
        memory_scheduler_clock=memory_scheduler_clock,
        schedule_scheduler_clock=schedule_scheduler_clock,
        monotonic_now=monotonic_now,
        timezone_name=timezone_name,
    )


def _prepare_runtime(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
    provider_factory: Callable[[ProviderConfiguration], ModelProvider],
    now: Callable[[], datetime],
    new_uuid: Callable[[], UUID],
    retry_clock: RetryClock | None = None,
    retry_jitter: Jitter | None = None,
    memory_scheduler_clock: SchedulerClock | None = None,
    schedule_scheduler_clock: ScheduleClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    timezone_name: str | None = None,
) -> PreparedRuntime:
    """Prepare a Session and defer provider construction until conversational input."""
    workspace_identity = Workspace.from_path(workspace)
    workspace_state = WorkspaceState(workspace_identity)
    workspace_state.initialize(agent_home_root=agent_home.path)
    schedule_store = WorkspaceScheduleStore(workspace_state)
    schedule_clock = (
        schedule_scheduler_clock
        if schedule_scheduler_clock is not None
        else AsyncioSchedulerClock(now=now)
    )
    schedule_service = ScheduleService(store=schedule_store, clock=schedule_clock)
    long_term_memory = workspace_state.long_term_memory_path.read_text(encoding="utf-8")
    runtime_memory = RuntimeMemory(long_term_memory)
    foreground_context = ContextBuilder(
        workspace_identity,
        get_localzone_name() if timezone_name is None else timezone_name,
    )
    set_context_clock = getattr(foreground_context, "set_clock", None)
    if callable(set_context_clock):
        set_context_clock(now)
    memory_store = WorkspaceFileMemoryStore(workspace_state)
    session = Session.create(
        workspace_state,
        now=now,
        new_uuid=new_uuid,
    )
    router = ModelRouter(
        configuration=configuration,
        provider_factory=provider_factory,
        clock=retry_clock,
        jitter=retry_jitter,
    )

    def system_prompt_for(memory_snapshot: str) -> str:
        return chat_system_prompt(
            workspace=workspace_identity.path,
            long_term_memory=memory_snapshot,
        )

    summaries = WorkspaceJsonlSummaryStore(workspace_state)
    memory_manager = MemoryManager(
        router=router,
        summaries=summaries,
        memory=memory_store,
        long_term_path=workspace_state.long_term_memory_path,
        batch_size=configuration.memory.batch_size,
        runtime_memory=runtime_memory,
    )
    scheduler_clock = (
        memory_scheduler_clock
        if memory_scheduler_clock is not None
        else AsyncioSchedulerClock(now=now)
    )
    memory_scheduler = _RuntimeSchedulerOwner(
        lambda: MemoryTaskScheduler(
            manager=memory_manager,
            schedule=configuration.memory.schedule,
            clock=scheduler_clock,
        )
    )

    def externalize_result_for(active_session: Session) -> ToolResultExternalizer:
        return _build_tool_result_externalizer(
            session=active_session,
            max_tool_result_chars=configuration.runtime.max_tool_result_chars,
        )

    async def prepare_summary(
        active_session: Session,
        route: AgentRunnerRoute,
        current_system_prompt: str,
        tools: tuple[OpenAIToolSchema, ...],
        current_user: dict[str, Any] | None = None,
    ) -> Session:
        effective_route = configuration.resolve_route(route).route
        if route == "chat":

            def project_messages(
                messages: Sequence[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                if _last_user_index(messages) == len(messages):
                    return _project_without_current_user(
                        messages,
                        system_prompt=current_system_prompt,
                    )
                return _project_foreground_messages(
                    foreground_context,
                    messages,
                    session_id=active_session.session_id,
                    long_term_memory=runtime_memory.snapshot(),
                )
        else:

            def project_messages(
                messages: Sequence[dict[str, Any]],
            ) -> list[dict[str, Any]]:
                if _last_user_index(messages) == len(messages):
                    return _project_without_current_user(
                        messages,
                        system_prompt=current_system_prompt,
                    )
                return _project_schedule_messages(
                    messages,
                    system_prompt=current_system_prompt,
                    session_id=active_session.session_id,
                    current_time=now(),
                )

        manager = ConversationSummaryManager(
            provider=router,
            summaries=summaries,
            route_context_window=effective_route.context_window,
            route_max_output=effective_route.max_output,
            consolidation_message_threshold=configuration.memory.consolidation_message_threshold,
            tools=tools,
            now=now,
            project_messages=project_messages,
        )
        return await manager.prepare(
            active_session,
            current_user=current_user,
        )

    async def prepare_schedule_context(
        active_session: Session,
        current_user: dict[str, Any],
    ) -> list[dict[str, Any]]:
        memory_snapshot = runtime_memory.snapshot()
        current_system_prompt = system_prompt_for(memory_snapshot)
        await prepare_summary(
            active_session,
            "schedule",
            current_system_prompt,
            tuple(agent_loop.tool_schemas),
            current_user,
        )
        history = active_session.messages[active_session.last_consolidated :]
        return _project_schedule_messages(
            [*history, current_user],
            system_prompt=current_system_prompt,
            session_id=active_session.session_id,
            current_time=now(),
        )

    async def prepare_foreground_context(
        active_session: Session,
        current_user: dict[str, Any],
    ) -> list[dict[str, Any]]:
        memory_snapshot = runtime_memory.snapshot()
        current_system_prompt = system_prompt_for(memory_snapshot)
        await prepare_summary(
            active_session,
            "chat",
            current_system_prompt,
            tuple(agent_loop.tool_schemas),
            current_user,
        )
        history = active_session.messages[active_session.last_consolidated :]
        return _project_foreground_messages(
            foreground_context,
            [*history, current_user],
            session_id=active_session.session_id,
            long_term_memory=memory_snapshot,
        )

    agent_loop = AgentLoop(
        workspace=workspace_identity,
        session=session,
        schedule_service=schedule_service,
        model_router=router,
        context_preparer=prepare_foreground_context,
        now=now,
        max_iterations=configuration.runtime.max_iterations,
        schedule_context_preparer=prepare_schedule_context,
        schedule_now=schedule_clock.now,
        title_prompt=session_title_prompt(),
        externalize_result_for=externalize_result_for,
    )

    schedule_service.on_schedule_job = agent_loop.run_schedule_job

    conversation = _AgentLoopConversationAdapter(
        loop=agent_loop,
        now=now,
        new_uuid=new_uuid,
    )
    foreground_chat_status: ResolvedChatStatus | None = None

    def capture_foreground_terminal(result: AgentRunnerResult | None) -> None:
        nonlocal foreground_chat_status
        foreground_chat_status = _resolved_chat_status(router)
        conversation.on_terminal(result)

    def current_foreground_chat_status() -> ResolvedChatStatus:
        return (
            _resolved_chat_status(router)
            if foreground_chat_status is None
            else foreground_chat_status
        )

    agent_loop.bind_confirmation_callback(conversation.on_confirmation_requested)
    agent_loop._bind_tool_completion_callback(conversation.on_tool_completed)
    agent_loop._bind_terminal_callback(capture_foreground_terminal)

    status_service = RuntimeStatusService(
        session=agent_loop.session,
        resolved_chat=current_foreground_chat_status,
        next_input=lambda active_session: _runtime_status_input(
            active_session,
            context_builder=foreground_context,
            long_term_memory=runtime_memory.snapshot(),
            session_id=active_session.session_id,
            tool_schemas=agent_loop.tool_schemas,
        ),
        monotonic=monotonic_now,
        schedule_status=lambda: schedule_service.status_snapshot().to_dict(),
    )

    def switch_session(selected_session: Session) -> None:
        agent_loop.switch_session(selected_session)
        status_service.use_session(agent_loop.session)

    management_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            agent_home,
            status_service=status_service,
            workspace_state=workspace_state,
            switch_session=switch_session,
            now=now,
            memory_manager=memory_manager,
            memory_store=memory_store,
        )
    )
    return PreparedRuntime(
        conversation=conversation,
        agent_loop=agent_loop,
        management_dispatcher=management_dispatcher,
        schedule_service=schedule_service,
        _memory_scheduler=memory_scheduler,
        _router=router,
        _lifetime=_RuntimeLifetime(),
    )


def _project_foreground_messages(
    context: ContextBuilder,
    messages: Sequence[dict[str, Any]],
    *,
    session_id: str,
    long_term_memory: str,
) -> list[dict[str, Any]]:
    history, current_user, current_user_index = _current_turn(messages, lane="Foreground")
    projected = context.build_messages(
        history=history,
        current_user=current_user,
        session_id=session_id,
        long_term_memory=long_term_memory,
    )
    projected.extend(_project_continuation(messages, current_user_index))
    return projected


def _project_schedule_messages(
    messages: Sequence[dict[str, Any]],
    *,
    system_prompt: str,
    session_id: str,
    current_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Project Schedule context without using the foreground ContextBuilder."""
    history, current_user, current_user_index = _current_turn(messages, lane="Schedule")
    projected = _project_without_current_user(history, system_prompt=system_prompt)
    content = current_user.get("content")
    timestamp = current_user.get("timestamp")
    if not isinstance(content, str):
        raise TypeError("Session user message is malformed")
    if isinstance(timestamp, str):
        effective_current_time = datetime.fromisoformat(timestamp)
    elif current_time is not None:
        effective_current_time = current_time
    else:
        raise TypeError("Schedule user message is missing a timestamp")
    projected.append(
        {
            "role": "user",
            "content": current_user_input(
                content=content,
                current_time=effective_current_time,
                session_id=session_id,
            ),
        }
    )
    projected.extend(_project_continuation(messages, current_user_index))
    return projected


def _project_continuation(
    messages: Sequence[dict[str, Any]],
    current_user_index: int,
) -> list[dict[str, Any]]:
    return _project_history_messages(messages[current_user_index + 1 :])


def _project_without_current_user(
    messages: Sequence[dict[str, Any]],
    *,
    system_prompt: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        *_project_history_messages(messages),
    ]


def _project_history_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        projected
        for message in messages
        if (projected := project_session_message(message)) is not None
    ]


def _current_turn(
    messages: Sequence[dict[str, Any]],
    *,
    lane: str,
) -> tuple[Sequence[dict[str, Any]], dict[str, Any], int]:
    current_user_index = _last_user_index(messages)
    if current_user_index == len(messages):
        raise ValueError(f"{lane} context requires a current user message")
    return messages[:current_user_index], messages[current_user_index], current_user_index


def _build_tool_result_externalizer(
    *,
    session: Session,
    max_tool_result_chars: int,
) -> ToolResultExternalizer:
    def externalize(result: ToolResult) -> ToolResult:
        if result.status != "success" or len(result.content) <= max_tool_result_chars:
            return result
        output = BaseTool.handle_result(
            result.content,
            workspace=session.workspace_state.workspace,
            session_id=session.session_id,
            tool_call_id=result.tool_call_id,
            limit=max_tool_result_chars,
        )
        return replace(result, content=output.content, artifact=output.artifact)

    return externalize


def _resolved_chat_status(router: ModelRouter) -> ResolvedChatStatus:
    status = router.route_status("chat")
    return ResolvedChatStatus(
        provider_id=status.provider_id,
        model=status.model,
        context_window=status.context_window,
    )


def _runtime_status_input(
    session: Session,
    *,
    context_builder: ContextBuilder | None = None,
    long_term_memory: str = "",
    system_prompt: str = "",
    current_time: datetime | None = None,
    session_id: str,
    tool_schemas: tuple[OpenAIToolSchema, ...],
) -> RuntimeStatusInput:
    if context_builder is not None:
        projected = context_builder.build_messages(
            history=session.messages[session.last_consolidated :],
            current_user={"role": "user", "content": ""},
            session_id=session_id,
            long_term_memory=long_term_memory,
        )
        projected_system = projected[0].get("content")
        if not isinstance(projected_system, str):
            raise TypeError("Context Builder status system message is malformed")
        system_prompt = projected_system
        retained = projected[1:]
        runtime_context_value = ""
    else:
        retained = [
            projected_message
            for message in session.messages[session.last_consolidated :]
            if (projected_message := project_session_message(message)) is not None
        ]
        runtime_context_value = (
            ""
            if current_time is None
            else runtime_context(
                current_time=current_time,
                session_id=session_id,
            )
        )
    retained_messages = tuple(
        json.dumps(message, ensure_ascii=False, separators=(",", ":")) for message in retained
    )
    return RuntimeStatusInput(
        system_prompt=system_prompt,
        retained_messages=retained_messages,
        tool_definitions=tuple(
            json.dumps(
                definition,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for definition in tool_schemas
        ),
        runtime_context=runtime_context_value,
    )


async def _await_shared_shutdown(task: asyncio.Task[None]) -> None:
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            if task.cancelled():
                break
            if cancellation is None:
                cancellation = error
        except BaseException:
            break
    try:
        task.result()
    except BaseException as cleanup_error:
        if cancellation is not None:
            raise cancellation from cleanup_error
        raise
    if cancellation is not None:
        raise cancellation
