"""Composition for one prepared command-line Conversation Session."""

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic
from uuid import UUID

from loguru import logger
from tzlocal import get_localzone_name

from myclaw.agent.loop import (
    AgentLoop,
    ForegroundConversationProjection,
    TerminalAgentLoopControl,
)
from myclaw.agent.loop import SkillContextTooLargeError as SkillContextTooLargeError
from myclaw.agent.message_bus import MessageBus
from myclaw.agent.repl import ManagementDispatcher, ProgressiveWriter, ReplInput, run_repl
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
    ManagementCommandDispatcher,
)
from myclaw.management.service import (
    ManagementError,
    ManagementViewService,
    RuntimeStatusService,
)
from myclaw.memory.dream import Dream
from myclaw.memory.manager import MemoryManager
from myclaw.provider.model_router import Jitter, ModelRouter, RetryClock
from myclaw.provider.models import ModelProvider
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleClock, ScheduleService
from myclaw.session.session import Session
from myclaw.skills.catalog import SkillMetadata
from myclaw.utils.async_tasks import await_task_preserving_cancellation
from myclaw.utils.scheduler import AsyncioSchedulerClock

type SessionReplacement = Callable[[str, bool], Awaitable[None]]


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
class RuntimeGenerationPresentation:
    """Immutable presentation data handed to an external Terminal adapter."""

    control: TerminalAgentLoopControl
    skill_metadata: tuple[SkillMetadata, ...]
    session_projection: ForegroundConversationProjection


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
    _management_service: ManagementViewService
    _management_service_owned: bool

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
            skill_metadata=self.agent_loop.skill_metadata,
        )

    @property
    def presentation(self) -> RuntimeGenerationPresentation:
        return RuntimeGenerationPresentation(
            control=self.control,
            skill_metadata=self.agent_loop.skill_metadata,
            session_projection=self.control.project_foreground_conversation(),
        )

    def validate_unstarted(self) -> None:
        """Validate injected scheduler seams without creating consumer tasks."""
        with without_session_log():
            try:
                if self._lifetime.started or self._lifetime.aborted:
                    raise RuntimeError("Runtime Generation is no longer preparable")
                self.agent_loop.preflight()
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

    async def abort(self, *, clear_inbound: bool = True) -> None:
        """Abandon this generation and drain its Memory-owned tasks."""
        self._request_abort()
        task = self._lifetime.abort_task
        if task is None:
            drain = (
                self._drain_aborted_memory()
                if clear_inbound
                else self._drain_aborted_memory(clear_inbound=False)
            )
            task = asyncio.create_task(drain)
            self._lifetime.abort_task = task
        await await_task_preserving_cancellation(task)

    def _request_abort(self) -> None:
        """Synchronously request abandonment for unstarted or async-drained runtimes."""
        if self._lifetime.aborted:
            return
        self._lifetime.aborted = True
        self._lifetime.started = True
        self._lifetime.shutdown_requested.set()
        if self._management_service_owned:
            self._management_service.deactivate()
        self.schedule_service.abort()
        self.agent_loop._request_abort()
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

    async def _drain_aborted_memory(self, *, clear_inbound: bool = True) -> None:
        drains: list[Awaitable[object]] = [
            self.agent_loop.abort(),
            self._dream.abort_and_wait(),
            self.schedule_service.abort_and_wait(),
        ]
        if clear_inbound:
            drains.insert(0, self.agent_loop._bus.drain_inbound())
        await asyncio.gather(*drains)

    async def _start_schedulers(self) -> None:
        if not self._lifetime.validated:
            raise RuntimeError("Runtime Generation was not validated")
        with without_session_log():
            try:
                await self.agent_loop.start()
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
            if self._management_service_owned:
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
                # Let a dispatcher that has already handed work to its run task
                # reach the callback before the lifetime owner closes the service.
                await asyncio.sleep(0)
                await self.schedule_service.close()
            except BaseException as error:
                failures.append(error)

            shutdowns: list[Awaitable[object]] = [self.agent_loop.close()]
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
        self._bus = MessageBus()
        self._terminal_rebind: Callable[[RuntimeGenerationPresentation], Awaitable[None]] | None = None
        self._terminal_quiesce: Callable[[], Awaitable[None]] | None = None
        self._replacement_lock = asyncio.Lock()
        self._closed = False
        self._runtime: PreparedRuntime | None = None
        self._management_available = False
        self._management_service = ManagementViewService(
            agent_home,
            current_agent_loop=lambda: self._current_management_agent_loop(),
            workspace_state=self._workspace_state,
            replace_agent_loop=self._replacement_callback(),
            now=now,
            monotonic=monotonic_now,
            current_memory_manager=lambda: self._current_management_memory_manager(),
            current_dream=lambda: self._current_management_dream(),
            schedule_status=lambda: self._current_management_schedule_status(),
        )
        self._management_dispatcher = ManagementCommandDispatcher(self._management_service)
        initial_runtime: PreparedRuntime | None = None
        try:
            initial_runtime = prepare_runtime(
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
                bus=self._bus,
                management_dispatcher=self._management_dispatcher,
                management_service=self._management_service,
            )
            initial_runtime.validate_unstarted()
            self._runtime = initial_runtime
            self._management_available = True
        except BaseException:
            self._management_available = False
            self._management_service.deactivate()
            if initial_runtime is not None:
                initial_runtime._request_abort()
            raise

    def _current_runtime(self) -> PreparedRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Runtime Host has no active Runtime Generation")
        return runtime

    def _current_management_runtime(self) -> PreparedRuntime:
        runtime = self._runtime
        if not self._management_available or runtime is None:
            raise ManagementError(
                ErrorInfo("route_unavailable", "Runtime Generation is unavailable.")
            )
        return runtime

    def _current_management_agent_loop(self) -> AgentLoop:
        return self._current_management_runtime().agent_loop

    def _current_management_memory_manager(self) -> MemoryManager:
        return self._current_management_runtime()._memory_manager

    def _current_management_dream(self) -> Dream:
        return self._current_management_runtime()._dream

    def _current_management_schedule_status(self) -> dict[str, object]:
        return self._current_management_runtime().schedule_service.status_snapshot().to_dict()

    @property
    def generation(self) -> PreparedRuntime:
        return self._current_runtime()

    @property
    def bindings(self) -> RuntimeBindings:
        return self._current_runtime().bindings

    @property
    def bus(self) -> MessageBus:
        return self._bus

    @property
    def control(self) -> TerminalAgentLoopControl:
        return self._current_runtime().control

    @property
    def presentation(self) -> RuntimeGenerationPresentation:
        return self._current_runtime().presentation

    @property
    def management_dispatcher(self) -> ManagementCommandDispatcher:
        return self._management_dispatcher

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Runtime Host is closed")
        try:
            await self._current_runtime().start()
        except BaseException:
            self._management_available = False
            self._management_service.deactivate()
            self._runtime = None
            self._closed = True
            raise
        self._management_available = True

    async def close(self) -> None:
        async with self._replacement_lock:
            if self._closed:
                return
            self._closed = True
            self._management_available = False
            self._management_service.deactivate()
            runtime = self._runtime
            if runtime is not None:
                await runtime.close()

    def bind_terminal(
        self,
        rebind: Callable[[RuntimeGenerationPresentation], Awaitable[None]],
        *,
        quiesce: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Register the sole Terminal binding callback for generation replacement."""
        if self._terminal_rebind is not None:
            raise RuntimeError("Runtime Host Terminal binding is already registered")
        if not callable(rebind):
            raise TypeError("Runtime Host Terminal binding must be callable")
        self._terminal_rebind = rebind
        self._terminal_quiesce = quiesce

    def unbind_terminal(
        self,
        rebind: Callable[[RuntimeGenerationPresentation], Awaitable[None]],
    ) -> None:
        if self._terminal_rebind is rebind:
            self._terminal_rebind = None
            self._terminal_quiesce = None

    def _replacement_callback(self) -> SessionReplacement:
        async def replace_session(session_id: str, force: bool) -> None:
            await self._replace_session(session_id, force)

        return replace_session

    def _prepare_target(self, session_id: str) -> PreparedRuntime:
        target_runtime: PreparedRuntime | None = None
        try:
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
                session_id=session_id,
                bus=self._bus,
                management_dispatcher=self.management_dispatcher,
                management_service=self._management_service,
            )
            target_runtime.validate_unstarted()
            return target_runtime
        except ManagementError:
            if target_runtime is not None:
                target_runtime._request_abort()
            raise
        except (OSError, UnicodeError, ValueError, WorkspaceStateError) as error:
            if target_runtime is not None:
                target_runtime._request_abort()
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "Conversation Session could not be prepared.",
                )
            ) from error
        except Exception as error:
            if target_runtime is not None:
                target_runtime._request_abort()
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "Conversation Session could not be prepared.",
                )
            ) from error
        except BaseException:
            if target_runtime is not None:
                target_runtime._request_abort()
            raise

    async def _replace_session(
        self,
        session_id: str,
        force: bool,
    ) -> None:
        async with self._replacement_lock:
            if self._closed:
                raise ManagementError(ErrorInfo("route_unavailable", "Runtime Host is closed."))
            old = self._current_runtime()
            if session_id == old.session_id:
                await old.session.wait_for_pending_persist()
            target = self._prepare_target(session_id)
            if old.control.has_active_run and not force:
                await target.abort(clear_inbound=False)
                raise ManagementError(
                    ErrorInfo(
                        "model_invalid_request",
                        "An active foreground run must be confirmed before switching Sessions.",
                    )
                )

            replacement = asyncio.create_task(
                self._commit_replacement(old, target)
            )
            await await_task_preserving_cancellation(replacement)

    async def _commit_replacement(
        self,
        old: PreparedRuntime,
        target: PreparedRuntime,
    ) -> None:
        """Drain and publish a replacement as one cancellation-safe operation."""
        rebind = self._terminal_rebind
        quiesce: Callable[[], Awaitable[None]] | None = None
        bus_reset = False
        old_detached = False
        try:
            old_detached = True
            self._management_available = False
            quiesce = self._terminal_quiesce
            if quiesce is not None:
                await quiesce()
            await old.schedule_service.pause_and_drain()
            await old.abort()
            await self._bus.reset()
            bus_reset = True
            self._runtime = target
            if rebind is None:
                await target.start()
                self._management_available = True
                return
            await rebind(target.presentation)
            await target.start()
            self._management_available = True
        except BaseException as error:
            if old_detached:
                self._management_available = False
                self._runtime = None
                self._closed = True
                self._management_service.deactivate()
                self._management_dispatcher._unbind_management(self._management_service)
            if quiesce is not None:
                with suppress(BaseException):
                    await quiesce()
            if rebind is not None:
                self.unbind_terminal(rebind)
            old_cleanup_error: BaseException | None = None
            try:
                if not old._lifetime.aborted:
                    await old.abort()
            except BaseException as caught:
                old_cleanup_error = caught
            try:
                await target.abort(clear_inbound=bus_reset)
            except BaseException as cleanup_error:
                if old_cleanup_error is not None:
                    raise error from BaseExceptionGroup(
                        "Runtime replacement cleanup failed",
                        [old_cleanup_error, cleanup_error],
                    )
                raise error from cleanup_error
            if old_cleanup_error is not None:
                raise error from old_cleanup_error
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
    session_id: str | None = None,
    workspace_state: WorkspaceState | None = None,
    replace_session: SessionReplacement | None = None,
    bus: MessageBus | None = None,
    management_dispatcher: ManagementCommandDispatcher | None = None,
    management_service: ManagementViewService | None = None,
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
            session_id=session_id,
            workspace_state=workspace_state,
            replace_session=replace_session,
            bus=bus,
            management_dispatcher=management_dispatcher,
            management_service=management_service,
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
    session_id: str | None = None,
    workspace_state: WorkspaceState | None = None,
    replace_session: SessionReplacement | None = None,
    bus: MessageBus | None = None,
    management_dispatcher: ManagementCommandDispatcher | None = None,
    management_service: ManagementViewService | None = None,
) -> PreparedRuntime:
    """Prepare one unstarted Runtime Generation."""
    workspace_path = normalize_workspace_path(workspace)
    active_workspace_state = (
        WorkspaceState(workspace_path) if workspace_state is None else workspace_state
    )
    if active_workspace_state.workspace_path != workspace_path:
        raise ValueError("Runtime Workspace State must belong to the Runtime Workspace")
    active_workspace_state.initialize(agent_home_root=agent_home.path)
    memory_manager = MemoryManager(active_workspace_state)
    schedule_clock = (
        schedule_scheduler_clock
        if schedule_scheduler_clock is not None
        else AsyncioSchedulerClock(now=now)
    )
    resolved_timezone_name = get_localzone_name() if timezone_name is None else timezone_name
    router = ModelRouter(
        configuration=configuration,
        provider_factory=provider_factory,
        clock=retry_clock,
        jitter=retry_jitter,
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
            timezone_name=resolved_timezone_name,
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

    try:
        agent_loop = AgentLoop(
            workspace_path=workspace_path,
            workspace_state=active_workspace_state,
            agent_home=agent_home,
            configuration=configuration,
            bus=MessageBus() if bus is None else bus,
            schedule_service=schedule_service,
            model_router=router,
            memory_manager=memory_manager,
            session_id=session_id,
            now=now,
            new_uuid=new_uuid,
            monotonic_now=monotonic_now,
        )
    except BaseException:
        abort_memory_composition()
        raise

    try:
        agent_loop.preflight()
    except BaseException:
        agent_loop._abandon_unstarted()
        abort_memory_composition()
        raise

    management_service_owned = management_service is None
    try:
        assert agent_loop is not None
        if management_service is None:
            status_service = RuntimeStatusService(
                current_agent_loop=lambda: agent_loop,
                monotonic=monotonic_now,
                schedule_status=lambda: schedule_service.status_snapshot().to_dict(),
            )
            management_service = ManagementViewService(
                agent_home,
                status_service=status_service,
                workspace_state=active_workspace_state,
                replace_agent_loop=replace_session,
                now=now,
                memory_manager=memory_manager,
                dream=dream,
            )
        if management_dispatcher is None:
            management_dispatcher = ManagementCommandDispatcher(management_service)
    except BaseException:
        agent_loop._abandon_unstarted()
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
        _management_service=management_service,
        _management_service_owned=management_service_owned,
    )
