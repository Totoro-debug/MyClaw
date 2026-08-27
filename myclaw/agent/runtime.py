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
from myclaw.agent.workspace_state import (
    WorkspaceState,
    WorkspaceStateError,
    normalize_workspace_path,
)
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ProviderConfiguration, UserConfiguration
from myclaw.errors import ErrorInfo
from myclaw.logging.session import without_session_log
from myclaw.management.commands import (
    MANAGEMENT_COMMANDS,
    ManagementCommandDispatcher,
)
from myclaw.management.service import (
    ManagementError,
    ManagementViewService,
    ResolvedChatStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
    estimate_input_tokens,
)
from myclaw.memory.conversation_summary import (
    ConversationSummaryManager,
    _last_user_index,
)
from myclaw.memory.dream import Dream
from myclaw.memory.manager import MemoryManager
from myclaw.provider.model_router import Jitter, ModelRouter, RetryClock
from myclaw.provider.models import ModelProvider
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleClock, ScheduleService
from myclaw.session.projection import project_session_message
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.skills.catalog import ManualSkillInvocation, SkillLoader, SkillMetadata, SkillSnapshot
from myclaw.terminal.repl import ManagementDispatcher, ProgressiveWriter, ReplInput, run_repl
from myclaw.tools.base import BaseTool, OpenAIToolSchema
from myclaw.tools.tool_gateway import ToolResult
from myclaw.utils.async_tasks import await_task_preserving_cancellation
from myclaw.utils.scheduler import AsyncioSchedulerClock

type SessionReplacement = Callable[[str, bool], Awaitable[None]]


class SkillContextTooLargeError(Exception):
    """The frozen always-loaded Skill snapshot exceeds the chat input budget."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


@dataclass(slots=True)
class _RuntimeLifetime:
    started: bool = False
    aborted: bool = False
    validated: bool = False
    close_task: asyncio.Task[None] | None = None
    abort_task: asyncio.Task[None] | None = None
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
    skill_metadata: tuple[SkillMetadata, ...]


@dataclass(frozen=True, slots=True)
class PreparedRuntime:
    """An in-memory Runtime exposing the MessageBus foreground seams."""

    agent_loop: AgentLoop
    management_dispatcher: ManagementDispatcher
    schedule_service: ScheduleService
    _router: ModelRouter
    _lifetime: _RuntimeLifetime
    _memory_manager: MemoryManager
    _dream: Dream
    _context_builder: ContextBuilder
    _management_service: ManagementViewService
    _skill_snapshot: SkillSnapshot

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
            skill_metadata=self._skill_snapshot.metadata,
        )

    def validate_unstarted(self) -> None:
        """Validate injected scheduler seams without creating consumer tasks."""
        with without_session_log():
            try:
                if self._lifetime.started or self._lifetime.aborted:
                    raise RuntimeError("Runtime Generation is no longer preparable")
                self.agent_loop._prepare_start()
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

    async def abort(self) -> None:
        """Abandon this generation and drain its Memory-owned tasks."""
        self._request_abort()
        task = self._lifetime.abort_task
        if task is None:
            task = asyncio.create_task(self._drain_aborted_memory())
            self._lifetime.abort_task = task
        await await_task_preserving_cancellation(task)

    def _request_abort(self) -> None:
        """Synchronously request abandonment for unstarted or async-drained runtimes."""
        if self._lifetime.aborted:
            return
        self._lifetime.aborted = True
        self._lifetime.started = True
        self._lifetime.shutdown_requested.set()
        self._management_service.deactivate()
        self.agent_loop.abort()
        self.schedule_service.abort()
        self._dream.abort()
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

    async def _drain_aborted_memory(self) -> None:
        await asyncio.gather(
            self._dream.abort_and_wait(),
            self.schedule_service.abort_and_wait(),
        )

    async def _start_schedulers(self) -> None:
        if not self._lifetime.validated:
            raise RuntimeError("Runtime Generation was not validated")
        with without_session_log():
            try:
                self.agent_loop._activate_prepared()
                self.schedule_service._activate_prepared()
            except BaseException as error:
                logger.opt(exception=error).error(
                    "Runtime startup failed type={}", type(error).__name__
                )
                self._request_abort()
                try:
                    await self._drain_aborted_memory()
                    await asyncio.sleep(0)
                except BaseException as cleanup_error:
                    raise error from cleanup_error
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
                self.agent_loop.close(),
            ]
            if run_done is not None:
                shutdowns.append(run_done.wait())
            results = await asyncio.gather(*shutdowns, return_exceptions=True)
            failures.extend(result for result in results if isinstance(result, BaseException))

            try:
                await self._dream.close()
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
        workspace: Path,
        configuration: UserConfiguration,
        provider_factory: Callable[[ProviderConfiguration], ModelProvider],
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        retry_clock: RetryClock | None = None,
        retry_jitter: Jitter | None = None,
        schedule_scheduler_clock: ScheduleClock | None = None,
        monotonic_now: Callable[[], float] = monotonic,
        timezone_name: str | None = None,
    ) -> None:
        self._agent_home = agent_home
        self._workspace = normalize_workspace_path(workspace)
        self._workspace_state = WorkspaceState(self._workspace)
        self._configuration = configuration
        self._provider_factory = provider_factory
        self._now = now
        self._new_uuid = new_uuid
        self._retry_clock = retry_clock
        self._retry_jitter = retry_jitter
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
            schedule_scheduler_clock=self._schedule_scheduler_clock,
            monotonic_now=monotonic_now,
            timezone_name=self._timezone_name,
            replace_session=self._replacement_callback(initial_token),
        )
        try:
            self._runtime.validate_unstarted()
        except BaseException:
            self._runtime._request_abort()
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
                target_runtime._request_abort()
            if target_session is not None:
                target_session.abandon()
            raise
        except (OSError, UnicodeError, ValueError, WorkspaceStateError) as error:
            if target_runtime is not None:
                target_runtime._request_abort()
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
                target_runtime._request_abort()
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
                target_runtime._request_abort()
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
                await old.session.wait_for_pending_persist()
            target_token = object()
            target = self._prepare_target(session_id, target_token)
            if old.control.has_active_run and not force:
                await target.abort()
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "An active foreground run must be confirmed before switching Sessions.",
                    )
                )

            replacement = asyncio.create_task(
                self._commit_replacement(old, target, target_token)
            )
            await await_task_preserving_cancellation(replacement)

    async def _commit_replacement(
        self,
        old: PreparedRuntime,
        target: PreparedRuntime,
        target_token: object,
    ) -> None:
        """Drain and publish a replacement as one cancellation-safe operation."""
        try:
            await old.abort()
            self._runtime = target
            self._runtime_token = target_token
            rebind = self._terminal_rebind
            if rebind is None:
                await target.start()
                return
            await rebind(target.bindings)
        except BaseException:
            await target.abort()
            raise


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
    schedule_scheduler_clock: ScheduleClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    timezone_name: str | None = None,
    skill_snapshot: SkillSnapshot | None = None,
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
            schedule_scheduler_clock=schedule_scheduler_clock,
            monotonic_now=monotonic_now,
            timezone_name=timezone_name,
            skill_snapshot=skill_snapshot,
            session=session,
            workspace_state=workspace_state,
            replace_session=replace_session,
            task_framer=task_framer,
        )
    except (WorkspaceStateError, SkillContextTooLargeError):
        raise
    except Exception as error:
        logger.opt(exception=error).error(
            "Runtime composition failed type={}", type(error).__name__
        )
        raise


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
    schedule_scheduler_clock: ScheduleClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    timezone_name: str | None = None,
    skill_snapshot: SkillSnapshot | None = None,
    session: Session | None = None,
    workspace_state: WorkspaceState | None = None,
    replace_session: SessionReplacement | None = None,
    task_framer: TaskFramingEvaluator | None = None,
) -> PreparedRuntime:
    """Prepare one unstarted Runtime Generation."""
    active_skill_snapshot = skill_snapshot
    if active_skill_snapshot is None:
        active_skill_snapshot = SkillLoader(
            root=agent_home.skills_directory,
            reserved_names=tuple(command.token for command in MANAGEMENT_COMMANDS),
            enable_always_load=configuration.runtime.enable_skill_always_load,
        ).load()
    workspace_path = normalize_workspace_path(workspace)
    active_workspace_state = (
        WorkspaceState(workspace_path) if workspace_state is None else workspace_state
    )
    if active_workspace_state.workspace_path != workspace_path:
        raise ValueError("Runtime Workspace State must belong to the Runtime Workspace")
    active_workspace_state.initialize(agent_home_root=agent_home.path)
    memory_manager = MemoryManager(active_workspace_state)
    long_term_memory = memory_manager.memory_snapshot()
    schedule_clock = (
        schedule_scheduler_clock
        if schedule_scheduler_clock is not None
        else AsyncioSchedulerClock(now=now)
    )
    resolved_timezone_name = get_localzone_name() if timezone_name is None else timezone_name
    foreground_context = ContextBuilder(
        workspace_path,
        resolved_timezone_name,
        skill_snapshot=active_skill_snapshot,
    )
    set_context_clock = getattr(foreground_context, "set_clock", None)
    if callable(set_context_clock):
        set_context_clock(now)
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
            workspace=workspace_path,
            long_term_memory=memory_snapshot,
        )

    def foreground_system_prompt_for(memory_snapshot: str) -> str:
        return foreground_chat_system_prompt(
            workspace=workspace_path,
            long_term_memory=memory_snapshot,
            skill_snapshot=active_skill_snapshot,
        )

    try:
        dream = Dream(
            memory_manager=memory_manager,
            model_router=router,
            batch_size=configuration.memory.batch_size,
            max_iterations=configuration.runtime.max_iterations,
        )
    except BaseException:
        router.abort()
        raise

    agent_loop: AgentLoop | None = None

    async def execute_user_job(job: ScheduleJob) -> None:
        active_loop = agent_loop
        if active_loop is None:
            raise RuntimeError("Schedule Service user executor is not bound")
        await active_loop.run_schedule_job(job)

    schedule_service: ScheduleService | None = None
    try:
        schedule_service = ScheduleService(
            workspace_state=active_workspace_state,
            clock=schedule_clock,
            execute_user_job=execute_user_job,
            execute_dream=dream.run,
        )
        schedule_service._register_dream_job_sync(
            schedule=JobSchedule.from_cron_input(
                configuration.memory.schedule,
                resolved_timezone_name,
            )
        )
    except BaseException:
        if schedule_service is not None:
            schedule_service.abort()
        dream.abort()
        router.abort()
        raise

    assert schedule_service is not None

    def abort_memory_composition() -> None:
        dream.abort()
        schedule_service.abort()
        router.abort()

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
        *,
        manual_invocation: ManualSkillInvocation | None = None,
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
                    long_term_memory=memory_manager.memory_snapshot(),
                    blackboard=blackboard,
                    manual_invocation=manual_invocation,
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
            summaries=memory_manager,
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
        active_loop = agent_loop
        if active_loop is None:
            raise RuntimeError("Schedule context requested before Agent Loop construction")
        memory_snapshot = memory_manager.memory_snapshot()
        current_system_prompt = system_prompt_for(memory_snapshot)
        await prepare_summary(
            active_session,
            "schedule",
            current_system_prompt,
            tuple(active_loop.tool_schemas),
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
        *,
        manual_invocation: ManualSkillInvocation | None = None,
    ) -> list[dict[str, Any]]:
        active_loop = agent_loop
        if active_loop is None:
            raise RuntimeError("Foreground context requested before Agent Loop construction")
        memory_snapshot = memory_manager.memory_snapshot()
        current_system_prompt = foreground_system_prompt_for(memory_snapshot)
        await prepare_summary(
            active_session,
            "chat",
            current_system_prompt,
            tuple(active_loop.tool_schemas),
            current_user,
            blackboard=blackboard,
            manual_invocation=manual_invocation,
        )
        history = active_session.messages[active_session.last_consolidated :]
        return _project_foreground_messages(
            foreground_context,
            [*history, current_user],
            session_id=active_session.session_id,
            long_term_memory=memory_snapshot,
            blackboard=blackboard,
            manual_invocation=manual_invocation,
        )

    try:
        agent_loop = AgentLoop(
            workspace=workspace_path,
            skill_snapshot=active_skill_snapshot,
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
    except BaseException:
        abort_memory_composition()
        raise

    try:
        _preflight_skill_context_budget(
            configuration=configuration,
            context_builder=foreground_context,
            long_term_memory=long_term_memory,
            session_id=active_session.session_id,
            skill_snapshot=active_skill_snapshot,
            tool_schemas=agent_loop.tool_schemas,
        )
    except BaseException:
        agent_loop.abort()
        abort_memory_composition()
        raise

    def current_foreground_chat_status() -> ResolvedChatStatus:
        foreground_status = agent_loop.last_foreground_route_status
        if foreground_status is not None:
            return ResolvedChatStatus(
                provider_id=foreground_status.provider_id,
                model=foreground_status.model,
                context_window=foreground_status.context_window,
            )
        return _resolved_chat_status(router)

    try:
        status_service = RuntimeStatusService(
            session=agent_loop.session,
            resolved_chat=current_foreground_chat_status,
            next_input=lambda active_session: _runtime_status_input(
                active_session,
                context_builder=foreground_context,
                long_term_memory=memory_manager.memory_snapshot(),
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
            dream=dream,
        )
        management_dispatcher = ManagementCommandDispatcher(management_service)
    except BaseException:
        agent_loop.abort()
        abort_memory_composition()
        raise
    return PreparedRuntime(
        agent_loop=agent_loop,
        management_dispatcher=management_dispatcher,
        schedule_service=schedule_service,
        _router=router,
        _lifetime=_RuntimeLifetime(),
        _memory_manager=memory_manager,
        _dream=dream,
        _context_builder=foreground_context,
        _management_service=management_service,
        _skill_snapshot=active_skill_snapshot,
    )


def _preflight_skill_context_budget(
    *,
    configuration: UserConfiguration,
    context_builder: ContextBuilder,
    long_term_memory: str,
    session_id: str,
    skill_snapshot: SkillSnapshot,
    tool_schemas: tuple[OpenAIToolSchema, ...],
) -> None:
    """Reject an always-loaded snapshot when the minimum real foreground request overflows."""
    if not any(skill.always for skill in skill_snapshot.skills):
        return

    resolved_chat = configuration.resolve_route("chat").route
    estimated = estimate_input_tokens(
        _foreground_runtime_status_input(
            context_builder=context_builder,
            history=(),
            session_id=session_id,
            long_term_memory=long_term_memory,
            tool_schemas=tool_schemas,
        )
    )
    available_input = resolved_chat.context_window - resolved_chat.max_output
    if estimated > available_input:
        raise SkillContextTooLargeError(
            ErrorInfo(
                "skill_context_too_large",
                "Always-loaded Skill content exceeds the foreground chat input budget.",
            )
        )


def _foreground_runtime_status_input(
    *,
    context_builder: ContextBuilder,
    history: Sequence[dict[str, Any]],
    session_id: str,
    long_term_memory: str,
    tool_schemas: tuple[OpenAIToolSchema, ...],
) -> RuntimeStatusInput:
    """Project and serialize one real foreground input using the status token seam."""
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
            json.dumps(
                definition,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for definition in tool_schemas
        ),
        runtime_context="",
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
            workspace=session.workspace_state.workspace_path,
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
        return _foreground_runtime_status_input(
            context_builder=context_builder,
            history=session.messages[session.last_consolidated :],
            session_id=session_id,
            long_term_memory=long_term_memory,
            tool_schemas=tool_schemas,
        )
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
