"""Serial foreground Agent Runner orchestration over the Runtime Message Bus."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, NoReturn, Protocol, cast
from uuid import UUID

from loguru import logger
from tzlocal import get_localzone_name

from myclaw.agent.blackboard import (
    Blackboard,
    FramingResult,
    TaskFramer,
    TaskFramingEvaluator,
    decode_blackboard,
    encode_blackboard,
)
from myclaw.agent.context import ContextBuilder
from myclaw.agent.message_bus import (
    InboundMessage,
    MessageBus,
    OutboundMessage,
    OutboundMessageType,
)
from myclaw.agent.prompts import (
    chat_system_prompt,
    current_user_input,
    foreground_chat_system_prompt,
    session_title_prompt,
)
from myclaw.agent.runner import (
    AgentRunner,
    AgentRunnerResponseSegmentEnd,
    AgentRunnerResult,
    AgentRunnerRouter,
    AgentRunnerToolCallStarted,
    _build_assistant_repair_message,
)
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import UserConfiguration
from myclaw.errors import TURN_CANCELLED_MESSAGE, ErrorInfo
from myclaw.logging.session import session_log
from myclaw.management.commands import MANAGEMENT_COMMANDS
from myclaw.management.service import RuntimeStatusInput, estimate_input_tokens
from myclaw.memory.conversation_summary import (
    ConversationSummaryManager,
    SummaryModelRouter,
    _last_user_index,
)
from myclaw.memory.manager import MemoryManager
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouteStatus
from myclaw.provider.models import ModelCompleted, ReasoningDelta, TextDelta
from myclaw.schedule.model import ScheduleJob
from myclaw.schedule.service import ScheduleJobExecutionError, ScheduleService
from myclaw.session.projection import project_session_message
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.skills.catalog import ManualSkillInvocation, SkillLoader, SkillMetadata
from myclaw.tools.base import BaseTool, OpenAIToolSchema
from myclaw.tools.core.schedule import ScheduleTool
from myclaw.tools.tool_gateway import (
    ConfirmationDecision,
    ConfirmationRequest,
    ToolGateway,
    ToolResult,
)
from myclaw.utils.async_tasks import await_task_preserving_cancellation


class ForegroundContextPreparer(Protocol):
    """Prepare foreground context with an optional staged Blackboard."""

    def __call__(
        self,
        session: Session,
        current_user: dict[str, Any],
        /,
        blackboard: Blackboard | None = None,
        *,
        manual_invocation: ManualSkillInvocation | None = None,
    ) -> Awaitable[list[dict[str, Any]]]: ...


type ScheduleContextPreparer = Callable[
    [Session, dict[str, Any]],
    Awaitable[list[dict[str, Any]]],
]
type ResultExternalizerFactory = Callable[[Session], Callable[[ToolResult], ToolResult]]


class SkillContextTooLargeError(Exception):
    """The frozen always-loaded Skill snapshot exceeds the chat input budget."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class ConfirmationRequestView(Protocol):
    """Stable confirmation data exposed to a foreground control consumer."""

    @property
    def confirmation_id(self) -> UUID: ...

    @property
    def tool_call_id(self) -> str: ...

    @property
    def tool_name(self) -> str: ...

    @property
    def reason(self) -> str: ...

    @property
    def summary(self) -> str: ...

    @property
    def details(self) -> dict[str, Any]: ...

    @property
    def warnings(self) -> tuple[str, ...]: ...


@dataclass(frozen=True, slots=True)
class ForegroundConversationProjection:
    """Presentation-safe snapshot of the active foreground conversation."""

    session_id: str
    messages: tuple[dict[str, Any], ...]


class AgentLoopControl(Protocol):
    """The independent foreground control surface owned by AgentLoop."""

    @property
    def has_active_run(self) -> bool: ...

    @property
    def has_pending_confirmation(self) -> bool: ...

    async def cancel_active_run(self) -> None: ...

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None: ...

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None: ...


class TerminalAgentLoopControl(AgentLoopControl, Protocol):
    """Foreground control surface including Terminal history projection."""

    def project_foreground_conversation(self) -> ForegroundConversationProjection: ...


type ConfirmationCallback = Callable[[ConfirmationRequestView], None]


@dataclass(slots=True)
class _PendingConfirmation:
    request: ConfirmationRequest
    future: asyncio.Future[ConfirmationDecision]


@dataclass(slots=True)
class _TitleCoordination:
    preparation_started: asyncio.Event
    prepared: asyncio.Future[bool]
    log_ready: asyncio.Event
    foreground_idle: asyncio.Event
    active_foregrounds: int = 0

    def attach_foreground(self) -> None:
        self.active_foregrounds += 1
        self.foreground_idle.clear()

    def release_foreground(self) -> None:
        self.active_foregrounds -= 1
        if self.active_foregrounds == 0:
            self.foreground_idle.set()

    async def wait_until_foreground_idle(self) -> None:
        while True:
            await self.foreground_idle.wait()
            await asyncio.sleep(0)
            if self.active_foregrounds == 0:
                return


@dataclass(frozen=True, slots=True)
class _TitleWork:
    task: asyncio.Task[None]
    coordination: _TitleCoordination


class AgentLoop:
    """Own the complete serial foreground execution path."""

    def __init__(
        self,
        *,
        workspace_path: Path,
        workspace_state: WorkspaceState,
        agent_home: AgentHome,
        configuration: UserConfiguration,
        bus: MessageBus,
        schedule_service: ScheduleService,
        model_router: AgentRunnerRouter,
        memory_manager: MemoryManager,
        session_id: str | None,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        monotonic_now: Callable[[], float],
    ) -> None:
        if not isinstance(workspace_path, Path):
            raise TypeError("Agent Loop requires a Workspace Path")
        if not isinstance(workspace_state, WorkspaceState):
            raise TypeError("Agent Loop requires a Workspace State")
        if workspace_state.workspace_path != workspace_path:
            raise ValueError("Agent Loop Workspace State must belong to the Workspace")
        if not isinstance(agent_home, AgentHome):
            raise TypeError("Agent Loop requires an Agent Home")
        if not isinstance(configuration, UserConfiguration):
            raise TypeError("Agent Loop requires User Configuration")
        if not isinstance(bus, MessageBus):
            raise TypeError("Agent Loop requires a Message Bus")
        if not isinstance(schedule_service, ScheduleService):
            raise TypeError("Agent Loop requires a Schedule Service")
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("Agent Loop requires a Memory Manager")
        if memory_manager.workspace_state is not workspace_state:
            raise ValueError("Agent Loop Memory Manager must belong to the Workspace State")
        if not callable(now):
            raise TypeError("Agent Loop requires a clock")
        if not callable(new_uuid):
            raise TypeError("Agent Loop requires a UUID allocator")
        if not callable(monotonic_now):
            raise TypeError("Agent Loop requires a monotonic clock")
        if session_id is not None and not isinstance(session_id, str):
            raise TypeError("Agent Loop Session ID must be a string or None")

        # Build every generation-local collaborator before publishing any Loop field.
        resolved_chat_route = configuration.resolve_route("chat")
        chat_route = resolved_chat_route.route
        configured_chat_model = f"{resolved_chat_route.provider.provider_id}/{chat_route.model}"
        configured_chat_context_window = chat_route.context_window
        skill_loader = SkillLoader(
            root=agent_home.skills_directory,
            reserved_names=tuple(command.token for command in MANAGEMENT_COMMANDS),
            enable_always_load=configuration.runtime.enable_skill_always_load,
        )
        skill_snapshot = skill_loader.load()
        context_builder = ContextBuilder(
            workspace_path,
            schedule_service.context_timezone_name() or get_localzone_name(),
            clock=now,
            skill_snapshot=skill_snapshot,
        )
        task_framer: TaskFramingEvaluator = TaskFramer(model_router)
        tool_gateway = ToolGateway(
            workspace=workspace_path,
            schedule_service=schedule_service,
            skill_root=skill_snapshot.root,
        )
        runner = AgentRunner(model_router)
        summary_manager = ConversationSummaryManager(
            provider=cast(SummaryModelRouter, model_router),
            memory_manager=memory_manager,
            route_context_window=chat_route.context_window,
            route_max_output=chat_route.max_output,
            consolidation_message_threshold=configuration.memory.consolidation_message_threshold,
            tools=tuple(tool_gateway.schemas),
            now=now,
            project_messages=self._project_foreground_summary_messages,
        )
        title_prompt: str | None = session_title_prompt()
        active_session = (
            Session.create(workspace_state, now=now, new_uuid=new_uuid)
            if session_id is None
            else Session.load(
                workspace_state,
                session_id,
                partition=SessionStoragePartition.FOREGROUND,
                now=now,
            )
        )

        self._workspace_path = workspace_path
        self._workspace_state = workspace_state
        self._agent_home = agent_home
        self._configuration = configuration
        self._configured_chat_model = configured_chat_model
        self._configured_chat_context_window = configured_chat_context_window
        self._session = active_session
        self._skill_loader = skill_loader
        self._skill_snapshot = skill_snapshot
        self._schedule_service = schedule_service
        self._context_builder = context_builder
        self._summary_manager = summary_manager
        self._task_framer = task_framer
        self._now = now
        self._monotonic_now = monotonic_now
        self._schedule_now = schedule_service.current_time
        self._title_prompt = title_prompt
        self._tool_gateway = tool_gateway
        self._model_router = model_router
        self._memory_manager = memory_manager
        self._runner = runner
        self._max_iterations = configuration.runtime.max_iterations
        self._bus = bus
        self._generation_started_at: float | None = None
        self._consumer_task: asyncio.Task[None] | None = None
        self._execution_task: asyncio.Task[None] | None = None
        self._foreground_commit_gate = asyncio.Lock()
        self._replacement_barrier_held = False
        self._schedule_tasks: set[asyncio.Task[None]] = set()
        self._aborted_tasks: set[asyncio.Task[Any]] = set()
        self._abort_task: asyncio.Task[None] | None = None
        self._close_task: asyncio.Task[None] | None = None
        self._execution_ready: asyncio.Event | None = None
        self._title_work: dict[str, _TitleWork] = {}
        self._pending_confirmation: _PendingConfirmation | None = None
        self._confirmation_callback: ConfirmationCallback | None = None
        self._cancel_requested = False
        self._closing = False
        self._closed = False
        self._aborted = False
        self._started = False
        self._preflighted = False
        self._preflight_error: Exception | None = None
        self._session_closed = False
        self._session_abandoned = False
        self._last_foreground_route_status: ModelRouteStatus | None = None

    @property
    def control(self) -> TerminalAgentLoopControl:
        return self

    @property
    def session(self) -> Session:
        return self._session

    @property
    def skill_metadata(self) -> tuple[SkillMetadata, ...]:
        return self._skill_snapshot.metadata

    @property
    def tool_schemas(self) -> tuple[OpenAIToolSchema, ...]:
        return tuple(self._tool_gateway.schemas)

    @property
    def has_active_run(self) -> bool:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        task = self._execution_task
        return task is not None and not task.done()

    @property
    def has_pending_confirmation(self) -> bool:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        pending = self._pending_confirmation
        return pending is not None and not pending.future.done()

    def project_foreground_conversation(self) -> ForegroundConversationProjection:
        """Return presentation data without exposing the owned Session."""
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        return ForegroundConversationProjection(
            session_id=self._session.session_id,
            messages=tuple(deepcopy(message) for message in self._session.messages),
        )

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        """Bind the synchronous foreground confirmation callback exactly once."""
        if self._confirmation_callback is not None:
            raise RuntimeError("Agent Loop confirmation callback is already bound")
        if self._closed or self._aborted:
            raise RuntimeError("Agent Loop is closed")
        if not callable(callback):
            raise TypeError("confirmation callback must be callable")
        self._confirmation_callback = callback

    def unbind_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        """Clear a callback only when it is still bound to this control surface."""
        if self._confirmation_callback is callback:
            self._confirmation_callback = None

    async def start(self) -> None:
        if self._closed or self._aborted or self._closing or self._close_task is not None:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            return
        self.preflight()
        self._activate_prepared()

    async def _pause_for_replacement(self) -> None:
        """Freeze new foreground admission and the final Session commit point."""
        if self._replacement_barrier_held:
            raise RuntimeError("Agent Loop replacement barrier is already held")
        await self._bus.pause_inbound_delivery()
        try:
            await self._foreground_commit_gate.acquire()
        except BaseException as error:
            resume = asyncio.create_task(self._bus.resume_inbound_delivery())
            try:
                await await_task_preserving_cancellation(resume)
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise
        self._replacement_barrier_held = True

    async def _release_replacement_barrier(self, *, resume_inbound: bool) -> None:
        """Release a barrier after rejection or after the target is published."""
        if self._replacement_barrier_held:
            self._replacement_barrier_held = False
            self._foreground_commit_gate.release()
        if resume_inbound:
            resume = asyncio.create_task(self._bus.resume_inbound_delivery())
            await await_task_preserving_cancellation(resume)

    def preflight(self) -> None:
        """Validate this generation synchronously without external side effects."""
        if self._closed or self._aborted or self._closing or self._close_task is not None:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            return
        if self._preflighted:
            return
        if self._preflight_error is not None:
            raise self._preflight_error
        try:
            chat_route = self._configuration.resolve_route("chat").route
            status_input = _foreground_runtime_status_input(
                context_builder=self._context_builder,
                history=(),
                session_id=self._session.session_id,
                long_term_memory=self._memory_manager.memory_snapshot(),
                tool_schemas=self.tool_schemas,
            )
            available_input = chat_route.context_window - chat_route.max_output
            if any(skill.always for skill in self._skill_snapshot.skills):
                estimated = estimate_input_tokens(status_input)
                if estimated > available_input:
                    raise SkillContextTooLargeError(
                        ErrorInfo(
                            "skill_context_too_large",
                            "Always-loaded Skill content exceeds the foreground chat input budget.",
                        )
                    )
        except Exception as error:
            self._preflight_error = error
            raise
        self._preflighted = True

    def _activate_prepared(self) -> None:
        """Sample uptime and atomically publish the preflighted Loop activation."""
        if self._closed or self._aborted or self._closing or self._close_task is not None:
            raise RuntimeError("Agent Loop is closed")
        if self._started:
            return
        if not self._preflighted:
            raise RuntimeError("Agent Loop was not preflighted")
        started_at = self._monotonic_now()
        consumer = self._consume_foreground()
        try:
            consumer_task = asyncio.create_task(consumer)
        except BaseException:
            consumer.close()
            raise
        self._consumer_task = consumer_task
        self._generation_started_at = started_at
        self._started = True

    async def close(self) -> None:
        if self._aborted:
            if self._abort_task is not None:
                await await_task_preserving_cancellation(self._abort_task)
            return
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._finish_close())
            self._close_task = task
        try:
            try:
                await await_task_preserving_cancellation(task)
            except asyncio.CancelledError:
                if not self._aborted:
                    raise
                abort_task = self._abort_task
                if abort_task is None:
                    abort_task = asyncio.create_task(self._finish_abort())
                    self._abort_task = abort_task
                await await_task_preserving_cancellation(abort_task)
        finally:
            if not self._aborted:
                self._close_session()
                await self._session.wait_for_pending_persist()

    async def abort(self) -> None:
        """Cancel and await every Session-scoped task before abandoning the Session."""
        if self._aborted:
            task = self._abort_task
            if task is None:
                task = asyncio.create_task(self._finish_abort())
                self._abort_task = task
            await await_task_preserving_cancellation(task)
            return
        self._request_abort()
        task = self._abort_task
        if task is None:
            task = asyncio.create_task(self._finish_abort())
            self._abort_task = task
        await await_task_preserving_cancellation(task)

    def _request_abort(self) -> None:
        """Synchronously stop new work before the awaited abort barrier runs."""
        if self._aborted:
            return
        self._aborted = True
        self._closing = True
        self._cancel_pending_confirmation()
        self._confirmation_callback = None
        if not self._started:
            self._abandon_session()
            self._closed = True
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        closing = self._close_task
        if closing is not None and closing is not current and not closing.done():
            closing.cancel()
        for task in self._owned_tasks():
            if task is current or task.done():
                continue
            self._retain_aborted_task(task)

    async def _finish_abort(self) -> None:
        try:
            await self._drain_owned_tasks()
            self._abandon_session()
            await self._session.wait_for_pending_persist()
        finally:
            self._clear_owned_task_references()
            self._closed = True

    async def _finish_close(self) -> None:
        if self._aborted:
            return
        self._closing = True
        self._cancel_pending_confirmation()
        if self._execution_task is not None and not self._execution_task.done():
            await self.cancel_active_run()
        current = asyncio.current_task()
        for task in self._owned_tasks():
            if task is not current and not task.done():
                task.cancel()
        await self._drain_owned_tasks()
        self._clear_owned_task_references()
        self._closed = True

    def _owned_tasks(self) -> tuple[asyncio.Task[Any], ...]:
        tasks: list[asyncio.Task[Any]] = []
        for task in (self._consumer_task, self._execution_task):
            if task is not None:
                tasks.append(task)
        tasks.extend(work.task for work in self._title_work.values())
        tasks.extend(self._schedule_tasks)
        return tuple(dict.fromkeys(tasks))

    async def _drain_owned_tasks(self) -> None:
        tasks = self._owned_tasks()
        current = asyncio.current_task()
        awaitable_tasks = tuple(task for task in tasks if task is not current)
        if awaitable_tasks:
            await asyncio.gather(*awaitable_tasks, return_exceptions=True)
        for task in awaitable_tasks:
            self._aborted_tasks.discard(task)
            if task.done() and not task.cancelled():
                try:
                    task.result()
                except BaseException as error:
                    logger.warning(
                        "Drained Agent Loop task failed type={}",
                        type(error).__name__,
                    )

    def _clear_owned_task_references(self) -> None:
        self._consumer_task = None
        self._execution_task = None
        self._execution_ready = None
        self._title_work.clear()
        self._schedule_tasks.clear()
        self._aborted_tasks.clear()

    def _close_session(self) -> None:
        if self._session_closed or self._session_abandoned:
            return
        try:
            self._session.close()
        except BaseException as error:
            logger.warning("Agent Loop Session close failed type={}", type(error).__name__)
        finally:
            self._session_closed = True

    def _abandon_session(self) -> None:
        if self._session_abandoned or self._session_closed:
            return
        self._session.abandon()
        self._session_abandoned = True

    def _retain_aborted_task(self, task: asyncio.Task[Any] | None) -> None:
        if task is None or task.done():
            return
        self._aborted_tasks.add(task)
        task.add_done_callback(self._aborted_task_finished)
        task.cancel()

    def _aborted_task_finished(self, task: asyncio.Task[Any]) -> None:
        self._aborted_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except BaseException as error:
            logger.warning(
                "Aborted Agent Loop task failed type={}",
                type(error).__name__,
            )

    async def cancel_active_run(self) -> None:
        if self._aborted:
            raise RuntimeError("Agent Loop is no longer active")
        active = self._execution_task
        if active is None or active.done():
            return
        self._cancel_requested = True
        self._cancel_pending_confirmation()
        ready = self._execution_ready
        if ready is not None and not ready.is_set():
            await ready.wait()
        await asyncio.sleep(0)
        if active.done():
            return
        active.cancel()
        await asyncio.gather(active, return_exceptions=True)

    async def run_schedule_job(self, job: ScheduleJob) -> None:
        """Execute one Schedule Job without using foreground state or output."""
        if self._aborted or self._closing or self._closed:
            raise RuntimeError("Agent Loop is no longer active")
        if job.source != "user":
            raise ScheduleJobExecutionError(
                ErrorInfo(
                    "schedule_state_error",
                    "Only User Schedule Jobs may run through Agent Loop.",
                )
            )
        current_task = asyncio.current_task()
        if current_task is not None:
            self._schedule_tasks.add(current_task)
        token = ScheduleTool._in_schedule_job.set(True)
        try:
            await self._execute_schedule_job(job)
        finally:
            ScheduleTool._in_schedule_job.reset(token)
            if current_task is not None:
                self._schedule_tasks.discard(current_task)

    async def _execute_schedule_job(self, job: ScheduleJob) -> None:
        schedule_session: Session | None = None
        workspace_state = self._session.workspace_state
        with session_log(workspace_state, job.session_id):
            try:
                try:
                    schedule_session = Session.load(
                        workspace_state,
                        job.session_id,
                        partition=SessionStoragePartition.SCHEDULE,
                        now=self._schedule_now,
                    )
                except FileNotFoundError:
                    schedule_session = Session.create_schedule(
                        workspace_state,
                        job.job_id,
                        now=self._schedule_now,
                    )
                try:
                    await self._run_schedule_agent(schedule_session, job)
                except ScheduleJobExecutionError as failure:
                    logger.warning(
                        "Schedule Job failed job_id={} kind={} code={}",
                        job.job_id,
                        job.schedule.kind,
                        failure.error.code,
                    )
                    raise
            finally:
                if schedule_session is not None:
                    try:
                        if self._aborted:
                            schedule_session.abandon()
                        else:
                            schedule_session.close()
                        persist_drain = asyncio.create_task(
                            schedule_session.wait_for_pending_persist()
                        )
                        await await_task_preserving_cancellation(persist_drain)
                    except Exception as error:
                        logger.error(
                            "Schedule Session close failed job_id={} type={}",
                            job.job_id,
                            type(error).__name__,
                        )

    async def _run_schedule_agent(self, session: Session, job: ScheduleJob) -> None:
        current_user = {"role": "user", "content": job.message}
        try:
            initial_messages = await self._prepare_schedule_context(
                session,
                deepcopy(current_user),
            )
        except asyncio.CancelledError:
            if not self._aborted:
                self._persist_schedule_cancelled_user(session, current_user, job)
            raise
        except ModelCallError as failure:
            if self._aborted:
                raise asyncio.CancelledError() from None
            self._persist_schedule_failure(session, current_user, failure.error)
        except Exception:
            if self._aborted:
                raise asyncio.CancelledError() from None
            self._persist_schedule_failure(
                session,
                current_user,
                ErrorInfo("model_failed", "The model request failed."),
            )

        result = await self._runner.run(
            initial_messages,
            model="schedule",
            tool_gateway=self._tool_gateway,
            on_output=_discard_runner_output,
            confirmation=None,
            externalize_result=self._result_externalizer_for(session),
            cancel_requested=self._schedule_service.cancellation_requested,
            max_iterations=self._max_iterations,
        )
        if self._aborted:
            raise asyncio.CancelledError()
        session.append_messages([deepcopy(current_user), *deepcopy(result.messages)])
        session.persist()

        if result.finish_reason == "cancelled":
            raise asyncio.CancelledError()
        if result.finish_reason != "completed":
            error = result.error or ErrorInfo("model_failed", "The model request failed.")
            raise ScheduleJobExecutionError(error)

    @staticmethod
    def _persist_schedule_cancelled_user(
        session: Session,
        current_user: dict[str, Any],
        job: ScheduleJob,
    ) -> None:
        try:
            session.append_messages([current_user])
            session.persist()
        except Exception as error:
            logger.error(
                "Schedule cancellation persistence failed job_id={} type={}",
                job.job_id,
                type(error).__name__,
            )

    @staticmethod
    def _persist_schedule_failure(
        session: Session,
        current_user: dict[str, Any],
        error: ErrorInfo,
    ) -> NoReturn:
        session.append_messages(
            [
                deepcopy(current_user),
                _build_assistant_repair_message(
                    content="",
                    status="error",
                    error=error,
                ),
            ]
        )
        session.persist()
        raise ScheduleJobExecutionError(error)

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> None:
        if self._aborted:
            raise ValueError("Confirmation response is late or unknown")
        if decision not in {"approved", "declined"}:
            raise ValueError("confirmation decision must be approved or declined")
        pending = self._pending_confirmation
        if pending is None or pending.request.confirmation_id != confirmation_id:
            raise ValueError("Confirmation response is late or unknown")
        if pending.future.done():
            raise ValueError("Confirmation response is late or unknown")
        pending.future.set_result(decision)

    async def _consume_foreground(self) -> None:
        try:
            while not self._closing:
                inbound = await self._bus.get_inbound()
                if self._closing:
                    break
                execution_ready = asyncio.Event()
                execution = asyncio.create_task(
                    self._execute_foreground(inbound, execution_ready=execution_ready)
                )
                self._execution_task = execution
                self._execution_ready = execution_ready
                try:
                    await execution
                except asyncio.CancelledError:
                    if not self._closing:
                        raise
                finally:
                    if self._execution_task is execution:
                        self._execution_task = None
                    if self._execution_ready is execution_ready:
                        self._execution_ready = None
                    self._cancel_requested = False
        except RuntimeError:
            if not self._closing:
                raise
        except asyncio.CancelledError:
            if not self._closing:
                raise

    async def _execute_foreground(
        self,
        inbound: InboundMessage,
        *,
        execution_ready: asyncio.Event,
    ) -> None:
        active_session = self._session
        manual_invocation = None
        if self._skill_snapshot is not None:
            manual_invocation = self._skill_snapshot.resolve_manual(inbound.content)
        start_title = not active_session.messages
        created_title_work = (
            self._start_title_if_needed(active_session, inbound.content) if start_title else None
        )
        title_work = created_title_work or self._title_work.get(active_session.session_id)
        if title_work is not None and title_work.task.done():
            title_work = None
        title_coordination = None if title_work is None else title_work.coordination
        if title_coordination is not None:
            title_coordination.attach_foreground()
        committed = False
        try:
            if title_work is None:
                with session_log(active_session):
                    committed = await self._execute_foreground_logged(
                        active_session,
                        inbound,
                        title_work=None,
                        manual_invocation=manual_invocation,
                        execution_ready=execution_ready,
                    )
            else:
                assert title_coordination is not None
                await title_coordination.log_ready.wait()
                with logger.contextualize(session_id=active_session.session_id):
                    committed = await self._execute_foreground_logged(
                        active_session,
                        inbound,
                        title_work=title_work,
                        manual_invocation=manual_invocation,
                        execution_ready=execution_ready,
                    )
        finally:
            execution_ready.set()
            if title_coordination is not None:
                title_coordination.release_foreground()
            if created_title_work is not None:
                created_coordination = created_title_work.coordination
                if not created_coordination.prepared.done():
                    created_coordination.prepared.set_result(False)
                if not committed:
                    if not created_title_work.task.done():
                        created_title_work.task.cancel()
                    await asyncio.gather(created_title_work.task, return_exceptions=True)
                    if self._title_work.get(active_session.session_id) is created_title_work:
                        self._title_work.pop(active_session.session_id)

    async def _execute_foreground_logged(
        self,
        active_session: Session,
        inbound: InboundMessage,
        *,
        title_work: _TitleWork | None,
        manual_invocation: ManualSkillInvocation | None = None,
        execution_ready: asyncio.Event,
    ) -> bool:
        current_user = {"role": "user", "content": inbound.content}
        if title_work is not None:
            title_work.coordination.preparation_started.set()
        execution_ready.set()
        if not inbound.content.strip():
            return False

        previous_blackboard = decode_blackboard(active_session.metadata.get("blackboard"))
        last_assistant_content = _latest_assistant_content(active_session)
        try:
            framing_result = await self._task_framer.frame(
                previous=previous_blackboard,
                last_assistant_content=last_assistant_content,
                current_user_input=inbound.content,
            )
            if not isinstance(framing_result, FramingResult):
                raise TypeError("Task Framing evaluator returned an invalid result")
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            await self._publish_preparation_failure(
                ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE)
            )
            return False

        if framing_result.status != "resolved":
            _runtime_logger().warning(
                "Task Framing degraded status={}",
                framing_result.status,
            )
        staged_blackboard = framing_result.blackboard
        try:
            initial_messages = await self._prepare_foreground_context(
                active_session,
                deepcopy(current_user),
                blackboard=staged_blackboard,
                manual_invocation=manual_invocation,
            )
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            await self._publish_preparation_failure(
                ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE)
            )
            return False
        except ModelCallError as failure:
            await self._publish_preparation_failure(failure.error)
            return False
        except Exception:
            await self._publish_preparation_failure(
                ErrorInfo("model_failed", "The model request failed.")
            )
            return False

        if title_work is not None and not title_work.coordination.prepared.done():
            title_work.coordination.prepared.set_result(True)
        try:
            result = await self._runner.run(
                initial_messages,
                model="chat",
                tool_gateway=self._tool_gateway,
                on_output=self._publish_runner_output,
                confirmation=self._request_confirmation,
                externalize_result=self._result_externalizer_for(active_session),
                cancel_requested=lambda: self._cancel_requested,
                max_iterations=self._max_iterations,
            )
            self._remember_foreground_route_status()
        except asyncio.CancelledError:
            if not self._cancel_requested:
                raise
            await self._publish_preparation_failure(
                ErrorInfo("turn_cancelled", TURN_CANCELLED_MESSAGE)
            )
            return False

        if self._aborted:
            return False

        if staged_blackboard is None:
            metadata_updates = None
            metadata_removals: tuple[str, ...] = ("blackboard",)
        else:
            encoded_blackboard = encode_blackboard(staged_blackboard)
            assert encoded_blackboard is not None
            metadata_updates = {"blackboard": encoded_blackboard}
            metadata_removals = ()

        async with self._foreground_commit_gate:
            try:
                if self._aborted:
                    return False
                active_session.append_messages(
                    [deepcopy(current_user), *deepcopy(result.messages)],
                    metadata_updates=metadata_updates,
                    metadata_removals=metadata_removals,
                    usage_delta=framing_result.usage_delta,
                )
            except Exception as failure:
                _runtime_logger().opt(exception=failure).error(
                    "Agent Run Session increment failed code=persistence_error type={}",
                    type(failure).__name__,
                )
                await self._publish_commit_failure()
                return False
            try:
                if self._aborted:
                    return False
                active_session.persist()
            except Exception:
                pass
        await self._publish_terminal(result)
        return True

    async def _prepare_foreground_context(
        self,
        active_session: Session,
        current_user: dict[str, Any],
        blackboard: Blackboard | None = None,
        *,
        manual_invocation: ManualSkillInvocation | None = None,
    ) -> list[dict[str, Any]]:
        memory_snapshot = self._memory_manager.memory_snapshot()
        current_system_prompt = foreground_chat_system_prompt(
            workspace=self._workspace_path,
            long_term_memory=memory_snapshot,
            skill_snapshot=self._skill_snapshot,
        )
        route = self._configuration.resolve_route("chat").route

        def project_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            if _last_user_index(messages) == len(messages):
                return _project_without_current_user(
                    messages,
                    system_prompt=current_system_prompt,
                )
            return _project_foreground_messages(
                self._context_builder,
                messages,
                session_id=active_session.session_id,
                long_term_memory=self._memory_manager.memory_snapshot(),
                blackboard=blackboard,
                manual_invocation=manual_invocation,
            )

        await self._summary_manager.prepare(
            active_session,
            current_user=current_user,
            project_messages=project_messages,
            route_context_window=route.context_window,
            route_max_output=route.max_output,
            tools=self.tool_schemas,
        )
        history = active_session.messages[active_session.last_consolidated :]
        return _project_foreground_messages(
            self._context_builder,
            [*history, current_user],
            session_id=active_session.session_id,
            long_term_memory=memory_snapshot,
            blackboard=blackboard,
            manual_invocation=manual_invocation,
        )

    async def _prepare_schedule_context(
        self,
        active_session: Session,
        current_user: dict[str, Any],
    ) -> list[dict[str, Any]]:
        memory_snapshot = self._memory_manager.memory_snapshot()
        current_system_prompt = chat_system_prompt(
            workspace=self._workspace_path,
            long_term_memory=memory_snapshot,
        )
        route = self._configuration.resolve_route("schedule").route

        def project_messages(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
            if _last_user_index(messages) == len(messages):
                return _project_without_current_user(
                    messages,
                    system_prompt=current_system_prompt,
                )
            return _project_schedule_messages(
                messages,
                system_prompt=current_system_prompt,
                session_id=active_session.session_id,
                current_time=self._schedule_now(),
            )

        await self._summary_manager.prepare(
            active_session,
            current_user=current_user,
            project_messages=project_messages,
            route_context_window=route.context_window,
            route_max_output=route.max_output,
            tools=self.tool_schemas,
        )
        history = active_session.messages[active_session.last_consolidated :]
        return _project_schedule_messages(
            [*history, current_user],
            system_prompt=current_system_prompt,
            session_id=active_session.session_id,
            current_time=self._schedule_now(),
        )

    def _project_foreground_summary_messages(
        self,
        messages: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        memory_snapshot = self._memory_manager.memory_snapshot()
        system_prompt = foreground_chat_system_prompt(
            workspace=self._workspace_path,
            long_term_memory=memory_snapshot,
            skill_snapshot=self._skill_snapshot,
        )
        if _last_user_index(messages) == len(messages):
            return _project_without_current_user(messages, system_prompt=system_prompt)
        return _project_foreground_messages(
            self._context_builder,
            messages,
            session_id=self._session.session_id,
            long_term_memory=memory_snapshot,
        )

    def runtime_status_input(self) -> RuntimeStatusInput:
        """Return the status token input projected by this generation's Context Builder."""
        session = self._session
        route_status = self._last_foreground_route_status
        session_id = session.session_id
        messages = session.messages
        metadata = session.metadata
        last_consolidated = session.last_consolidated
        title = metadata.get("title")
        if not isinstance(title, str):
            raise ValueError("Active Session title is malformed")
        usage_value = metadata.get("token_usage")
        if not isinstance(usage_value, dict):
            raise ValueError("Active Session token usage is malformed")
        usage_fields = ("model_calls", "input_tokens", "output_tokens", "total_tokens")
        usage = tuple((field, usage_value.get(field)) for field in usage_fields)
        if any(isinstance(value, bool) or not isinstance(value, int) for _, value in usage):
            raise ValueError("Active Session token usage is malformed")
        return _foreground_runtime_status_input(
            context_builder=self._context_builder,
            history=messages[last_consolidated:],
            session_id=session_id,
            long_term_memory=self._memory_manager.memory_snapshot(),
            tool_schemas=self.tool_schemas,
            session_title=title,
            session_message_count=len(messages),
            last_consolidated=last_consolidated,
            cumulative_usage=tuple((field, cast(int, value)) for field, value in usage),
            chat_model=(
                self._configured_chat_model
                if route_status is None
                else f"{route_status.provider_id}/{route_status.model}"
            ),
            context_window=(
                self._configured_chat_context_window
                if route_status is None
                else route_status.context_window
            ),
            generation_started_at=self._generation_started_at,
        )

    def _remember_foreground_route_status(self) -> None:
        current_call_status = getattr(self._model_router, "current_call_status", None)
        if callable(current_call_status):
            status = current_call_status("chat")
            if isinstance(status, ModelRouteStatus):
                self._last_foreground_route_status = status
                return
        route_status = getattr(self._model_router, "route_status", None)
        if callable(route_status):
            status = route_status("chat")
            if isinstance(status, ModelRouteStatus):
                self._last_foreground_route_status = status

    def _result_externalizer_for(
        self,
        active_session: Session,
    ) -> Callable[[ToolResult], ToolResult] | None:
        max_tool_result_chars = self._configuration.runtime.max_tool_result_chars

        def externalize(result: ToolResult) -> ToolResult:
            if result.status != "success" or len(result.content) <= max_tool_result_chars:
                return result
            output = BaseTool.handle_result(
                result.content,
                workspace=active_session.workspace_state.workspace_path,
                session_id=active_session.session_id,
                tool_call_id=result.tool_call_id,
                limit=max_tool_result_chars,
            )
            return replace(result, content=output.content, artifact=output.artifact)

        return externalize

    async def _publish_runner_output(self, event: object) -> None:
        if self._aborted:
            return
        if isinstance(event, ReasoningDelta):
            await self._bus.put_outbound(
                OutboundMessage(
                    "model_reasoning",
                    event.delta,
                    {"_stream_delta": True},
                )
            )
            return
        if isinstance(event, TextDelta):
            await self._bus.put_outbound(
                OutboundMessage(
                    "model_response",
                    event.delta,
                    {"_stream_delta": True},
                )
            )
            return
        if isinstance(event, AgentRunnerResponseSegmentEnd):
            outbound_type: OutboundMessageType = (
                "model_reasoning" if event.segment == "reasoning" else "model_response"
            )
            await self._bus.put_outbound(OutboundMessage(outbound_type, "", {"_stream_end": True}))
            return
        if isinstance(event, AgentRunnerToolCallStarted):
            await self._bus.put_outbound(
                OutboundMessage(
                    "tool_call",
                    event.tool_name,
                    {
                        "tool_call_id": event.tool_call_id,
                        "arguments": event.arguments,
                    },
                )
            )
            return
        raise TypeError(f"Unsupported Agent Runner output: {type(event).__name__}")

    async def _publish_terminal(self, result: AgentRunnerResult) -> None:
        if self._aborted:
            return
        if result.finish_reason == "completed":
            await self._bus.put_outbound(OutboundMessage("model_response", "", {"_streamed": True}))
            return
        error = result.error
        if error is None:
            error = ErrorInfo("model_failed", "The model request failed.")
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": result.finish_reason,
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _publish_preparation_failure(self, error: ErrorInfo) -> None:
        if self._aborted:
            return
        if error.code != "turn_cancelled":
            _log_agent_failure(error)
        finish_reason = "cancelled" if error.code == "turn_cancelled" else "failed"
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": finish_reason,
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _publish_commit_failure(self) -> None:
        if self._aborted:
            return
        error = ErrorInfo(
            "persistence_error",
            "The Conversation Session could not be updated.",
        )
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                error.message,
                {
                    "finish_reason": "failed",
                    "error_code": error.code,
                    "_streamed": True,
                },
            )
        )

    async def _request_confirmation(
        self,
        request: ConfirmationRequest,
    ) -> ConfirmationDecision:
        if self._aborted or self._closing:
            raise asyncio.CancelledError()
        if self._pending_confirmation is not None:
            raise RuntimeError("A foreground confirmation request is already pending")
        callback = self._confirmation_callback
        if callback is None:
            raise RuntimeError("Agent Loop confirmation callback is not bound")
        future: asyncio.Future[ConfirmationDecision] = asyncio.get_running_loop().create_future()
        pending = _PendingConfirmation(request=request, future=future)
        self._pending_confirmation = pending
        try:
            callback(request)
            return await future
        finally:
            if self._pending_confirmation is pending:
                self._pending_confirmation = None

    def _cancel_pending_confirmation(self) -> None:
        pending = self._pending_confirmation
        if pending is not None and not pending.future.done():
            pending.future.cancel()

    def _start_title_if_needed(
        self,
        session: Session,
        content: str,
    ) -> _TitleWork | None:
        if (
            self._closing
            or self._aborted
            or self._title_prompt is None
            or not content.strip()
            or session.metadata.get("title") != "Untitled session"
            or session.session_id in self._title_work
        ):
            return None
        preparation_started = asyncio.Event()
        prepared: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        log_ready = asyncio.Event()
        foreground_idle = asyncio.Event()
        foreground_idle.set()
        coordination = _TitleCoordination(
            preparation_started=preparation_started,
            prepared=prepared,
            log_ready=log_ready,
            foreground_idle=foreground_idle,
        )
        task = asyncio.create_task(
            self._generate_title(
                session,
                content,
                coordination=coordination,
            )
        )
        work = _TitleWork(
            task=task,
            coordination=coordination,
        )
        self._title_work[session.session_id] = work
        task.add_done_callback(self._title_done)
        return work

    def _title_done(self, task: asyncio.Task[None]) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            return
        except Exception as error:
            logger.opt(exception=error).error(
                "Session title task failed type={}", type(error).__name__
            )

    async def _generate_title(
        self,
        session: Session,
        content: str,
        *,
        coordination: _TitleCoordination,
    ) -> None:
        with session_log(session):
            coordination.log_ready.set()
            try:
                await coordination.preparation_started.wait()
                title, usage_delta = await self._resolve_title(content)
                if (
                    await coordination.prepared
                    and session.metadata.get("title") == "Untitled session"
                ):
                    session.update_metadata(title=title, usage_delta=usage_delta)
            except asyncio.CancelledError:
                if (
                    not self._aborted
                    and coordination.prepared.done()
                    and coordination.prepared.result()
                    and session.metadata.get("title") == "Untitled session"
                ):
                    session.update_metadata(title=Session._normalize_title(content))
                raise
            finally:
                await coordination.wait_until_foreground_idle()

    async def _resolve_title(self, content: str) -> tuple[str, dict[str, int] | None]:
        title = Session._normalize_title(content)
        usage_delta: dict[str, int] | None = None
        events: Any = None
        try:
            events = self._router_stream_title(content)
            async for event in events:
                if not isinstance(event, ModelCompleted):
                    continue
                response = event.response
                usage_delta = {"model_calls": 1, **response.usage.to_dict()}
                if response.message.tool_calls:
                    continue
                candidate = Session._normalize_title_candidate(response.message.content)
                if candidate:
                    title = candidate
                break
        except Exception as error:
            _runtime_logger().opt(exception=error).warning(
                "Session title fallback selected type={}", type(error).__name__
            )
        finally:
            if events is not None:
                close = getattr(events, "aclose", None)
                if close is not None:
                    try:
                        await close()
                    except RuntimeError:
                        pass
        return title, usage_delta

    def _router_stream_title(self, content: str) -> Any:
        return self._model_router.stream(
            "chat",
            messages=[
                {"role": "system", "content": self._title_prompt or ""},
                {"role": "user", "content": Session._normalize_title(content)},
            ],
            tools=(),
            continuation=None,
        )


__all__ = [
    "AgentLoop",
    "AgentLoopControl",
    "ConfirmationCallback",
    "ConfirmationRequestView",
    "ForegroundContextPreparer",
    "ScheduleContextPreparer",
    "SkillContextTooLargeError",
]


def _runtime_logger() -> Any:
    def set_runtime_name(record: Any) -> None:
        record["name"] = "myclaw.agent.loop"

    return logger.patch(set_runtime_name)


def _log_agent_failure(error: ErrorInfo) -> None:
    failure = ModelCallError(error)
    _runtime_logger().opt(exception=failure).error(
        "Agent Run failed code={} type={}",
        error.code,
        type(failure).__name__,
    )


async def _discard_runner_output(event: object) -> None:
    del event


def _latest_assistant_content(session: Session) -> str:
    for message in reversed(session.messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _foreground_runtime_status_input(
    *,
    context_builder: ContextBuilder,
    history: Sequence[dict[str, Any]],
    session_id: str,
    long_term_memory: str,
    tool_schemas: tuple[OpenAIToolSchema, ...],
    session_title: str = "",
    session_message_count: int = 0,
    last_consolidated: int = 0,
    cumulative_usage: tuple[tuple[str, int], ...] = (),
    chat_model: str = "",
    context_window: int = 0,
    generation_started_at: float | None = None,
) -> RuntimeStatusInput:
    """Project and serialize a minimum foreground request for status and preflight."""
    projected = context_builder.build_messages(
        history=history,
        current_user={"role": "user", "content": ""},
        session_id=session_id,
        long_term_memory=long_term_memory,
    )
    projected_system = projected[0].get("content")
    if not isinstance(projected_system, str):
        raise TypeError("Context Builder status system message is malformed")
    return RuntimeStatusInput(
        system_prompt=projected_system,
        retained_messages=tuple(
            json.dumps(message, ensure_ascii=False, separators=(",", ":"))
            for message in projected[1:]
        ),
        tool_definitions=tuple(
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")) for schema in tool_schemas
        ),
        runtime_context="",
        session_id=session_id,
        session_title=session_title,
        session_message_count=session_message_count,
        last_consolidated=last_consolidated,
        cumulative_usage=cumulative_usage,
        chat_model=chat_model,
        context_window=context_window,
        generation_started_at=generation_started_at,
    )


def _project_foreground_messages(
    context: ContextBuilder,
    messages: Sequence[dict[str, Any]],
    *,
    session_id: str,
    long_term_memory: str,
    blackboard: Blackboard | None = None,
    manual_invocation: ManualSkillInvocation | None = None,
) -> list[dict[str, Any]]:
    history, current_user, current_user_index = _current_turn(messages, lane="Foreground")
    projected = context.build_messages(
        history=history,
        current_user=current_user,
        session_id=session_id,
        long_term_memory=long_term_memory,
        blackboard=blackboard,
        manual_invocation=manual_invocation,
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
    """Project Schedule context without exposing foreground Skill content."""
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
