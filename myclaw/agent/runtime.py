"""Composition for one prepared command-line Conversation Session."""

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any, cast
from uuid import UUID

from loguru import logger
from tzlocal import get_localzone_name

from myclaw.agent.blackboard import Blackboard, TaskFramingEvaluator
from myclaw.agent.context import ContextBuilder
from myclaw.agent.loop import AgentLoop, TerminalAgentLoopControl
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.prompts import (
    chat_system_prompt,
    current_user_input,
    foreground_chat_system_prompt,
    runtime_context,
    session_title_prompt,
)
from myclaw.agent.runner import AgentRunnerRoute
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState, WorkspaceStateError
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ProviderConfiguration, UserConfiguration
from myclaw.errors import ErrorInfo
from myclaw.logging.session import without_session_log
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import (
    ManagementError,
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
from myclaw.provider.models import ModelProvider
from myclaw.schedule.service import ScheduleClock, ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.projection import project_session_message
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.terminal.repl import ManagementDispatcher, ProgressiveWriter, ReplInput, run_repl
from myclaw.tools.base import BaseTool, OpenAIToolSchema
from myclaw.tools.tool_gateway import ToolResult
from myclaw.utils.async_tasks import await_task_preserving_cancellation
from myclaw.utils.scheduler import AsyncioSchedulerClock, SchedulerClock

type SessionReplacement = Callable[[str, bool], Awaitable[None]]


class _RuntimeSchedulerOwner:
    """Own one terminal scheduler instance per Runtime run."""

    def __init__(
        self,
        factory: Callable[[], MemoryTaskScheduler | ScheduleService],
    ) -> None:
        self._factory = factory
        self._active: MemoryTaskScheduler | ScheduleService | None = None
        self._aborted = False

    def prepare(self) -> None:
        """Construct and validate the scheduler without starting owned tasks."""
        if self._aborted:
            raise RuntimeError("Runtime scheduler is closed")
        scheduler = self._active
        if scheduler is None:
            scheduler = self._factory()
            self._active = scheduler
        scheduler._prepare_start()

    def activate_prepared(self) -> None:
        """Activate the already validated scheduler."""
        scheduler = self._active
        if scheduler is None:
            raise RuntimeError("Runtime scheduler was not prepared")
        scheduler._activate_prepared()

    async def close(self) -> None:
        if self._aborted:
            return
        scheduler = self._active
        if scheduler is None:
            return
        await scheduler.close()
        if self._active is scheduler:
            self._active = None

    def abort(self) -> None:
        """Synchronously cancel the active scheduler, if it was started."""
        if self._aborted:
            return
        self._aborted = True
        scheduler = self._active
        if scheduler is not None:
            scheduler.abort()


@dataclass(slots=True)
class _RuntimeLifetime:
    started: bool = False
    aborted: bool = False
    validated: bool = False
    close_task: asyncio.Task[None] | None = None
    shutdown_requested: asyncio.Event = field(default_factory=asyncio.Event)
    run_task: asyncio.Task[object] | None = None
    run_done: asyncio.Event | None = None

    def begin(self) -> None:
        if self.started or self.aborted:
            raise RuntimeError("Prepared Runtime is closed")
        self.started = True


@dataclass(frozen=True, slots=True)
class RuntimeBindings:
    """The generation-bound seams consumed by Terminal Conversation."""

    bus: MessageBus
    control: TerminalAgentLoopControl
    management_dispatcher: ManagementDispatcher
    start: Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class PreparedRuntime:
    """An in-memory Runtime exposing the MessageBus foreground seams."""

    agent_loop: AgentLoop
    management_dispatcher: ManagementDispatcher
    schedule_service: ScheduleService
    _memory_scheduler: _RuntimeSchedulerOwner
    _router: ModelRouter
    _lifetime: _RuntimeLifetime
    _schedule_store: WorkspaceScheduleStore
    _runtime_memory: RuntimeMemory
    _memory_manager: MemoryManager
    _context_builder: ContextBuilder
    _management_service: ManagementViewService

    @property
    def session_id(self) -> str:
        return self.agent_loop.session.session_id

    @property
    def bus(self) -> MessageBus:
        return self.agent_loop.bus

    @property
    def control(self) -> TerminalAgentLoopControl:
        return self.agent_loop.control

    @property
    def session(self) -> Session:
        return self.agent_loop.session

    @property
    def bindings(self) -> RuntimeBindings:
        return RuntimeBindings(
            bus=self.bus,
            control=self.control,
            management_dispatcher=self.management_dispatcher,
            start=self.start,
        )

    def validate_unstarted(self) -> None:
        """Validate injected scheduler seams without creating consumer tasks."""
        with without_session_log():
            try:
                if self._lifetime.started or self._lifetime.aborted:
                    raise RuntimeError("Runtime Generation is no longer preparable")
                self.agent_loop._prepare_start()
                self._memory_scheduler.prepare()
                self.schedule_service._prepare_start()
                self._lifetime.validated = True
            except BaseException as error:
                logger.opt(exception=error).error(
                    "Runtime validation failed type={}", type(error).__name__
                )
                raise

    async def start(self) -> None:
        if not self._lifetime.validated:
            self.validate_unstarted()
        self._lifetime.begin()
        await self._start_schedulers()

    def abort(self) -> None:
        """Synchronously abandon this generation without final persistence."""
        if self._lifetime.aborted:
            return
        self._lifetime.aborted = True
        self._lifetime.started = True
        self._lifetime.shutdown_requested.set()
        self._management_service.deactivate()
        self.agent_loop.abort()
        self.schedule_service.abort()
        self._memory_manager.abort()
        self._memory_scheduler.abort()
        self._router.abort()
        closing = self._lifetime.close_task
        if closing is not None and not closing.done():
            closing.cancel()
        running = self._lifetime.run_task
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if running is not None and running is not current and not running.done():
            running.cancel()

    async def _start_schedulers(self) -> None:
        if not self._lifetime.validated:
            raise RuntimeError("Runtime Generation was not validated")
        with without_session_log():
            try:
                self.agent_loop._activate_prepared()
                self._memory_scheduler.activate_prepared()
                self.schedule_service._activate_prepared()
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
        if not self._lifetime.validated:
            self.validate_unstarted()
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
                    bus=self.bus,
                    control=self.control,
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
        if self._lifetime.aborted:
            return
        task = self._lifetime.close_task
        if task is None:
            self._management_service.deactivate()
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
        await await_task_preserving_cancellation(task)

    async def _close_owned_resources(self, run_done: asyncio.Event | None) -> None:
        with without_session_log():
            failures: list[BaseException] = []
            try:
                await self.schedule_service.close()
            except BaseException as error:
                failures.append(error)

            shutdowns: list[Awaitable[object]] = [
                self._memory_scheduler.close(),
                self.agent_loop.close(),
            ]
            if run_done is not None:
                shutdowns.append(run_done.wait())
            results = await asyncio.gather(*shutdowns, return_exceptions=True)
            failures.extend(result for result in results if isinstance(result, BaseException))

            try:
                await self._memory_manager.wait_until_idle()
            except BaseException as error:
                failures.append(error)

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


class RuntimeHost:
    """Coordinate process-level inputs and replace one prepared generation."""

    def __init__(
        self,
        *,
        agent_home: AgentHome,
        workspace: Path | Workspace,
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
    ) -> None:
        self._agent_home = agent_home
        self._workspace = (
            workspace if isinstance(workspace, Workspace) else Workspace.from_path(workspace)
        )
        self._workspace_state = WorkspaceState(self._workspace)
        self._configuration = configuration
        self._provider_factory = provider_factory
        self._now = now
        self._new_uuid = new_uuid
        self._retry_clock = retry_clock
        self._retry_jitter = retry_jitter
        self._memory_scheduler_clock = (
            memory_scheduler_clock
            if memory_scheduler_clock is not None
            else AsyncioSchedulerClock(now=now)
        )
        self._schedule_scheduler_clock = (
            schedule_scheduler_clock
            if schedule_scheduler_clock is not None
            else AsyncioSchedulerClock(now=now)
        )
        self._monotonic_now = monotonic_now
        self._timezone_name = get_localzone_name() if timezone_name is None else timezone_name
        self._terminal_rebind: Callable[[RuntimeBindings], Awaitable[None]] | None = None
        self._replacement_lock = asyncio.Lock()
        self._closed = False
        initial_token = object()
        self._runtime_token = initial_token
        self._runtime = prepare_runtime(
            agent_home=agent_home,
            workspace=self._workspace,
            workspace_state=self._workspace_state,
            configuration=configuration,
            provider_factory=provider_factory,
            now=now,
            new_uuid=new_uuid,
            retry_clock=retry_clock,
            retry_jitter=retry_jitter,
            memory_scheduler_clock=self._memory_scheduler_clock,
            schedule_scheduler_clock=self._schedule_scheduler_clock,
            monotonic_now=monotonic_now,
            timezone_name=self._timezone_name,
            replace_session=self._replacement_callback(initial_token),
        )
        try:
            self._runtime.validate_unstarted()
        except BaseException:
            self._runtime.abort()
            raise

    @property
    def generation(self) -> PreparedRuntime:
        return self._runtime

    @property
    def bindings(self) -> RuntimeBindings:
        return self._runtime.bindings

    @property
    def bus(self) -> MessageBus:
        return self._runtime.bus

    @property
    def control(self) -> TerminalAgentLoopControl:
        return self._runtime.control

    @property
    def management_dispatcher(self) -> ManagementCommandDispatcher:
        return cast(ManagementCommandDispatcher, self._runtime.management_dispatcher)

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime Host is closed")
        await self._runtime.start()

    async def close(self) -> None:
        async with self._replacement_lock:
            if self._closed:
                return
            self._closed = True
            await self._runtime.close()

    def bind_terminal(self, rebind: Callable[[RuntimeBindings], Awaitable[None]]) -> None:
        """Register the sole Terminal binding callback for generation replacement."""
        if self._terminal_rebind is not None:
            raise RuntimeError("Runtime Host Terminal binding is already registered")
        if not callable(rebind):
            raise TypeError("Runtime Host Terminal binding must be callable")
        self._terminal_rebind = rebind

    def unbind_terminal(self, rebind: Callable[[RuntimeBindings], Awaitable[None]]) -> None:
        if self._terminal_rebind is rebind:
            self._terminal_rebind = None

    def _replacement_callback(self, token: object) -> SessionReplacement:
        async def replace_session(session_id: str, force: bool) -> None:
            await self._replace_session(token, session_id, force)

        return replace_session

    def _prepare_target(self, session_id: str, token: object) -> PreparedRuntime:
        target_session: Session | None = None
        target_runtime: PreparedRuntime | None = None
        try:
            target_session = Session.load(
                self._workspace_state,
                session_id,
                partition=SessionStoragePartition.FOREGROUND,
                now=self._now,
            )
            target_runtime = prepare_runtime(
                agent_home=self._agent_home,
                workspace=self._workspace,
                workspace_state=self._workspace_state,
                configuration=self._configuration,
                provider_factory=self._provider_factory,
                now=self._now,
                new_uuid=self._new_uuid,
                retry_clock=self._retry_clock,
                retry_jitter=self._retry_jitter,
                memory_scheduler_clock=self._memory_scheduler_clock,
                schedule_scheduler_clock=self._schedule_scheduler_clock,
                monotonic_now=self._monotonic_now,
                timezone_name=self._timezone_name,
                session=target_session,
                replace_session=self._replacement_callback(token),
            )
            target_runtime.validate_unstarted()
            return target_runtime
        except ManagementError:
            if target_runtime is not None:
                target_runtime.abort()
            if target_session is not None:
                target_session.abandon()
            raise
        except (OSError, UnicodeError, ValueError, WorkspaceStateError) as error:
            if target_runtime is not None:
                target_runtime.abort()
            if target_session is not None:
                target_session.abandon()
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "Conversation Session could not be prepared.",
                )
            ) from error
        except Exception as error:
            if target_runtime is not None:
                target_runtime.abort()
            if target_session is not None:
                target_session.abandon()
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "Conversation Session could not be prepared.",
                )
            ) from error
        except BaseException:
            if target_runtime is not None:
                target_runtime.abort()
            if target_session is not None:
                target_session.abandon()
            raise

    async def _replace_session(
        self,
        source_token: object,
        session_id: str,
        force: bool,
    ) -> None:
        async with self._replacement_lock:
            if self._closed:
                raise ManagementError(ErrorInfo("route_unavailable", "Runtime Host is closed."))
            if source_token is not self._runtime_token:
                raise ManagementError(
                    ErrorInfo(
                        "route_unavailable",
                        "Runtime Generation is no longer active.",
                    )
                )
            old = self._runtime
            if session_id == old.session_id:
                return
            target_token = object()
            target = self._prepare_target(session_id, target_token)
            if old.control.has_active_run and not force:
                target.abort()
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "An active foreground run must be confirmed before switching Sessions.",
                    )
                )

            old.abort()
            self._runtime = target
            self._runtime_token = target_token
            rebind = self._terminal_rebind
            if rebind is None:
                try:
                    await target.start()
                except BaseException:
                    target.abort()
                    raise
                return
            try:
                await rebind(target.bindings)
            except BaseException:
                target.abort()
                raise


def prepare_runtime(
    *,
    agent_home: AgentHome,
    workspace: Path | Workspace,
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
    session: Session | None = None,
    workspace_state: WorkspaceState | None = None,
    replace_session: SessionReplacement | None = None,
    task_framer: TaskFramingEvaluator | None = None,
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
            session=session,
            workspace_state=workspace_state,
            replace_session=replace_session,
            task_framer=task_framer,
        )
    except WorkspaceStateError:
        raise
    except Exception as error:
        logger.opt(exception=error).error(
            "Runtime composition failed type={}", type(error).__name__
        )
        raise


def _prepare_runtime(
    *,
    agent_home: AgentHome,
    workspace: Path | Workspace,
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
    session: Session | None = None,
    workspace_state: WorkspaceState | None = None,
    replace_session: SessionReplacement | None = None,
    task_framer: TaskFramingEvaluator | None = None,
) -> PreparedRuntime:
    """Prepare one unstarted Runtime Generation."""
    workspace_identity = (
        workspace if isinstance(workspace, Workspace) else Workspace.from_path(workspace)
    )
    active_workspace_state = (
        WorkspaceState(workspace_identity) if workspace_state is None else workspace_state
    )
    if active_workspace_state.workspace is not workspace_identity:
        raise ValueError("Runtime Workspace State must belong to the Runtime Workspace")
    active_workspace_state.initialize(agent_home_root=agent_home.path)
    schedule_store = WorkspaceScheduleStore(active_workspace_state)
    schedule_clock = (
        schedule_scheduler_clock
        if schedule_scheduler_clock is not None
        else AsyncioSchedulerClock(now=now)
    )
    schedule_service = ScheduleService(store=schedule_store, clock=schedule_clock)
    long_term_memory = active_workspace_state.long_term_memory_path.read_text(encoding="utf-8")
    runtime_memory = RuntimeMemory(long_term_memory)
    resolved_timezone_name = get_localzone_name() if timezone_name is None else timezone_name
    foreground_context = ContextBuilder(workspace_identity, resolved_timezone_name)
    set_context_clock = getattr(foreground_context, "set_clock", None)
    if callable(set_context_clock):
        set_context_clock(now)
    memory_store = WorkspaceFileMemoryStore(active_workspace_state)
    active_session = session
    if active_session is None:
        active_session = Session.create(
            active_workspace_state,
            now=now,
            new_uuid=new_uuid,
        )
    elif (
        active_session.workspace_state is not active_workspace_state
        or active_session.storage_partition is not SessionStoragePartition.FOREGROUND
    ):
        raise ValueError("Runtime Session must belong to the foreground Runtime Workspace State")
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

    def foreground_system_prompt_for(memory_snapshot: str) -> str:
        return foreground_chat_system_prompt(
            workspace=workspace_identity.path,
            long_term_memory=memory_snapshot,
        )

    summaries = WorkspaceJsonlSummaryStore(active_workspace_state)
    memory_manager = MemoryManager(
        router=router,
        summaries=summaries,
        memory=memory_store,
        long_term_path=active_workspace_state.long_term_memory_path,
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

    def externalize_result_for(active_session: Session) -> Callable[[ToolResult], ToolResult]:
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
        blackboard: Blackboard | None = None,
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
                    blackboard=blackboard,
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
        blackboard: Blackboard | None = None,
    ) -> list[dict[str, Any]]:
        memory_snapshot = runtime_memory.snapshot()
        current_system_prompt = foreground_system_prompt_for(memory_snapshot)
        await prepare_summary(
            active_session,
            "chat",
            current_system_prompt,
            tuple(agent_loop.tool_schemas),
            current_user,
            blackboard=blackboard,
        )
        history = active_session.messages[active_session.last_consolidated :]
        return _project_foreground_messages(
            foreground_context,
            [*history, current_user],
            session_id=active_session.session_id,
            long_term_memory=memory_snapshot,
            blackboard=blackboard,
        )

    agent_loop = AgentLoop(
        workspace=workspace_identity,
        skill_root=agent_home.skills_directory,
        session=active_session,
        schedule_service=schedule_service,
        model_router=router,
        context_preparer=prepare_foreground_context,
        now=now,
        max_iterations=configuration.runtime.max_iterations,
        schedule_context_preparer=prepare_schedule_context,
        schedule_now=schedule_clock.now,
        title_prompt=session_title_prompt(),
        externalize_result_for=externalize_result_for,
        task_framer=task_framer,
    )

    schedule_service.on_schedule_job = agent_loop.run_schedule_job

    def current_foreground_chat_status() -> ResolvedChatStatus:
        foreground_status = agent_loop.last_foreground_route_status
        if foreground_status is not None:
            return ResolvedChatStatus(
                provider_id=foreground_status.provider_id,
                model=foreground_status.model,
                context_window=foreground_status.context_window,
            )
        return _resolved_chat_status(router)

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

    management_service = ManagementViewService(
        agent_home,
        status_service=status_service,
        workspace_state=active_workspace_state,
        replace_session=replace_session,
        now=now,
        memory_manager=memory_manager,
        memory_store=memory_store,
    )
    management_dispatcher = ManagementCommandDispatcher(management_service)
    return PreparedRuntime(
        agent_loop=agent_loop,
        management_dispatcher=management_dispatcher,
        schedule_service=schedule_service,
        _memory_scheduler=memory_scheduler,
        _router=router,
        _lifetime=_RuntimeLifetime(),
        _schedule_store=schedule_store,
        _runtime_memory=runtime_memory,
        _memory_manager=memory_manager,
        _context_builder=foreground_context,
        _management_service=management_service,
    )


def _project_foreground_messages(
    context: ContextBuilder,
    messages: Sequence[dict[str, Any]],
    *,
    session_id: str,
    long_term_memory: str,
    blackboard: Blackboard | None = None,
) -> list[dict[str, Any]]:
    history, current_user, current_user_index = _current_turn(messages, lane="Foreground")
    projected = context.build_messages(
        history=history,
        current_user=current_user,
        session_id=session_id,
        long_term_memory=long_term_memory,
        blackboard=blackboard,
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
) -> Callable[[ToolResult], ToolResult]:
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
