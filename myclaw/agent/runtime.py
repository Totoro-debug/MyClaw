"""Composition for one prepared command-line Conversation Session."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import UUID

from loguru import logger

from myclaw.agent.events import AgentEvent, ConversationPort
from myclaw.agent.prompts import (
    chat_system_prompt,
    render_tool_guidance,
    runtime_context,
    session_title_prompt,
)
from myclaw.agent.turn import ToolResultExternalizer, model_message_from_session
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState, WorkspaceStateError
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ProviderConfiguration, UserConfiguration
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
    SummaryModelSettings,
    WorkspaceJsonlSummaryStore,
)
from myclaw.memory.memory_scheduler import MemoryTaskScheduler
from myclaw.memory.memory_task import (
    MemoryManager,
    MemoryTaskModelSettings,
    WorkspaceFileMemoryStore,
)
from myclaw.provider.model_router import Jitter, ModelRouter, RetryClock
from myclaw.provider.models import ModelProvider
from myclaw.schedule.background_coordination import (
    RuntimeEventBroker,
    ScheduledWorkCoordinator,
    ScheduledWorkScheduler,
)
from myclaw.schedule.scheduled_work import (
    CreateScheduledWorkTool,
    WorkspaceJsonScheduledWorkStore,
)
from myclaw.schedule.scheduled_work_execution import (
    ScheduledWorkModelSettings,
    ScheduledWorkRunner,
)
from myclaw.session.conversation import (
    ChatModelSettings,
    StreamingConversationPort,
)
from myclaw.session.records import ConversationSession
from myclaw.session.session import Session
from myclaw.session.session_resume import SwitchableConversationPort
from myclaw.session.session_store import JsonlSessionStore
from myclaw.terminal.repl import ManagementDispatcher, ProgressiveWriter, ReplInput, run_repl
from myclaw.tools.base import BaseTool
from myclaw.tools.files.file_tools import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from myclaw.tools.models import ToolResult
from myclaw.tools.schema import OpenAIToolSchema
from myclaw.tools.security import Security
from myclaw.tools.shell.shell_process import SubprocessShellBoundary
from myclaw.tools.shell.shell_tool import ShellBoundary, ShellTool
from myclaw.tools.tool_artifacts import externalize_tool_result
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.tools.web.web_fetch import (
    AioHttpWebFetchClient,
    PublicWebFetchBoundary,
    SocketDNSResolver,
    WebFetchBoundary,
    WebFetchTool,
)
from myclaw.tools.web.web_search import (
    DuckDuckGoSearchBoundary,
    WebSearchBoundary,
    WebSearchTool,
)
from myclaw.utils.scheduler import AsyncioSchedulerClock, SchedulerClock


class _RuntimeSchedulerOwner:
    """Own one terminal scheduler instance per Runtime run."""

    def __init__(
        self,
        factory: Callable[[], MemoryTaskScheduler | ScheduledWorkScheduler],
    ) -> None:
        self._factory = factory
        self._active: MemoryTaskScheduler | ScheduledWorkScheduler | None = None

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
class PreparedReplRuntime:
    """An in-memory Session identity and its injectable REPL composition."""

    conversation: SwitchableConversationPort
    session: Session
    sessions: JsonlSessionStore
    management_dispatcher: ManagementDispatcher
    scheduled_work_coordinator: ScheduledWorkCoordinator
    _shell: SubprocessShellBoundary | None
    _memory_scheduler: _RuntimeSchedulerOwner
    _scheduled_work_scheduler: _RuntimeSchedulerOwner
    _background_events: RuntimeEventBroker
    _router: ModelRouter
    _lifetime: _RuntimeLifetime

    @property
    def session_id(self) -> str:
        return self.conversation.session_id

    async def start(self) -> None:
        self._lifetime.begin()
        try:
            self._start_schedulers()
        except BaseException as primary_error:
            try:
                await self.close()
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise

    def _start_schedulers(self) -> None:
        with without_session_log():
            try:
                self._memory_scheduler.start()
                self._scheduled_work_scheduler.start()
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
                self._start_schedulers()
                await run_repl(
                    conversation=self.conversation,
                    input_reader=input_reader,
                    writer=writer,
                    management_dispatcher=dispatcher,
                    background_events=self._background_events,
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
                self._background_events.close()
            except BaseException as error:
                failures.append(error)

            shutdowns: list[Awaitable[object]] = [
                self._scheduled_work_scheduler.close(),
                self._memory_scheduler.close(),
                self.conversation.close(),
            ]
            if self._shell is not None:
                shutdowns.append(self._shell.close())
            if run_done is not None:
                shutdowns.append(run_done.wait())
            results = await asyncio.gather(*shutdowns, return_exceptions=True)
            failures.extend(result for result in results if isinstance(result, BaseException))

            try:
                await self._router.close()
            except BaseException as error:
                failures.append(error)

            if not failures:
                self._close_session()
                return
            self._close_session()
            failure = (
                failures[0]
                if len(failures) == 1
                else BaseExceptionGroup("Runtime shutdown failed", failures)
            )
            logger.opt(exception=failure).error(
                "Runtime shutdown failed type={}", type(failure).__name__
            )
            raise failure

    def _close_session(self) -> None:
        try:
            self.session.close()
        except BaseException:
            pass


class _DeferredConversationPort:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        session: Session | None = None,
        sessions: JsonlSessionStore | None = None,
        session_id: str | None = None,
        settings: ChatModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str,
        title_prompt: str,
        tool_gateway: ToolGateway,
        on_foreground_terminal: Callable[[], None],
        history_preparer: Callable[[Any], Awaitable[Any]] | None = None,
        before_submit: Callable[[], Awaitable[None]] | None = None,
        externalize_result: ToolResultExternalizer | None = None,
        workspace_state: WorkspaceState | None = None,
    ) -> None:
        if session is None:
            if sessions is None or session_id is None:
                raise TypeError("Legacy Conversation Port requires sessions and session_id")
        elif sessions is not None or session_id is not None:
            raise TypeError("Active Conversation Port cannot receive a Session Store or ID")
        self._provider = provider
        self._session = session
        self._sessions = sessions
        self._session_id = session.session_id if session is not None else session_id
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._system_prompt = system_prompt
        self._title_prompt = title_prompt
        self._tool_gateway = tool_gateway
        self._history_preparer = history_preparer
        self._before_submit = before_submit
        self._on_foreground_terminal = on_foreground_terminal
        self._workspace_state = workspace_state
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
        correlation: AbstractContextManager[None] = logger.contextualize(
            session_id=self._session_id
        )
        if self._session is not None:
            correlation = session_log(self._session)
        elif self._workspace_state is not None:
            assert self._session_id is not None
            correlation = session_log(self._workspace_state, self._session_id)
        with correlation:
            try:
                if self._before_submit is not None:
                    await self._before_submit()
                if self._close_task is not None:
                    raise RuntimeError("Conversation Port is closed")
                delegate = self._delegate
                if delegate is None:
                    if self._session is not None:
                        delegate = StreamingConversationPort(
                            provider=self._provider,
                            session=self._session,
                            settings=self._settings,
                            now=self._now,
                            new_uuid=self._new_uuid,
                            system_prompt=self._system_prompt,
                            title_prompt=self._title_prompt,
                            tool_gateway=self._tool_gateway,
                            history_preparer=self._history_preparer,
                            externalize_result=self._externalize_result,
                            workspace_state=self._workspace_state,
                            title_log_ready=title_log_ready.wait,
                        )
                    else:
                        assert self._sessions is not None
                        assert self._session_id is not None
                        delegate = StreamingConversationPort(
                            provider=self._provider,
                            sessions=self._sessions,
                            session_id=self._session_id,
                            settings=self._settings,
                            now=self._now,
                            new_uuid=self._new_uuid,
                            system_prompt=self._system_prompt,
                            title_prompt=self._title_prompt,
                            tool_gateway=self._tool_gateway,
                            history_preparer=self._history_preparer,
                            externalize_result=self._externalize_result,
                            workspace_state=self._workspace_state,
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
        active.cancel()

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_delegate())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_delegate(self) -> None:
        active = self._active_task
        active_done = self._active_done
        if active is not None and active is not asyncio.current_task() and not active.done():
            active.cancel()
        if active_done is not None:
            await active_done.wait()
        delegate = self._delegate
        if delegate is None:
            return
        close = getattr(delegate, "close", None)
        if close is None:
            await delegate.cancel_active_turn()
        else:
            await close()


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
    scheduled_work_scheduler_clock: SchedulerClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    web_search: WebSearchBoundary | None = None,
    web_fetch: WebFetchBoundary | None = None,
    shell: ShellBoundary | None = None,
) -> PreparedReplRuntime:
    """Prepare one Runtime and record terminal composition failures once."""
    try:
        return _prepare_repl_runtime(
            agent_home=agent_home,
            workspace=workspace,
            configuration=configuration,
            provider_factory=provider_factory,
            now=now,
            new_uuid=new_uuid,
            retry_clock=retry_clock,
            retry_jitter=retry_jitter,
            memory_scheduler_clock=memory_scheduler_clock,
            scheduled_work_scheduler_clock=scheduled_work_scheduler_clock,
            monotonic_now=monotonic_now,
            web_search=web_search,
            web_fetch=web_fetch,
            shell=shell,
        )
    except WorkspaceStateError:
        raise
    except Exception as error:
        logger.opt(exception=error).error(
            "Runtime composition failed type={}", type(error).__name__
        )
        raise


def _prepare_repl_runtime(
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
    scheduled_work_scheduler_clock: SchedulerClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    web_search: WebSearchBoundary | None = None,
    web_fetch: WebFetchBoundary | None = None,
    shell: ShellBoundary | None = None,
) -> PreparedReplRuntime:
    """Prepare a Session and defer provider construction until conversational input."""
    configuration.resolve_route("default")
    workspace_identity = Workspace.from_path(workspace)
    workspace_state = WorkspaceState(workspace_identity)
    workspace_state.initialize(agent_home_root=agent_home.path)
    long_term_memory = workspace_state.long_term_memory_path.read_text(encoding="utf-8")
    memory_store = WorkspaceFileMemoryStore(workspace_state)
    sessions = JsonlSessionStore(
        workspace_state=workspace_state,
        now=now,
        new_uuid=new_uuid,
    )
    session = Session.create(
        workspace_state,
        now=now,
        new_uuid=new_uuid,
    )
    resolved = configuration.resolve_route("chat")
    settings = ChatModelSettings(
        model=resolved.route.model,
        max_output=resolved.route.max_output,
        temperature=resolved.route.temperature,
        reasoning_effort=resolved.route.reasoning_effort,
        timeout_seconds=resolved.route.timeout,
    )
    router = ModelRouter(
        configuration=configuration,
        provider_factory=provider_factory,
        clock=retry_clock,
        jitter=retry_jitter,
    )
    configured_web_search = (
        (web_search if web_search is not None else DuckDuckGoSearchBoundary())
        if configuration.tools.web.enabled
        else None
    )
    configured_web_fetch = (
        (
            web_fetch
            if web_fetch is not None
            else PublicWebFetchBoundary(
                resolver=SocketDNSResolver(),
                http_client=AioHttpWebFetchClient(),
            )
        )
        if configuration.tools.web.enabled
        else None
    )
    configured_shell = (
        (shell if shell is not None else SubprocessShellBoundary())
        if configuration.tools.shell.enabled
        else None
    )
    owned_shell = (
        configured_shell if isinstance(configured_shell, SubprocessShellBoundary) else None
    )
    scheduled_work_store = WorkspaceJsonScheduledWorkStore(workspace_state)
    scheduled_work_tool = CreateScheduledWorkTool()
    tool_gateway = _build_tool_gateway(
        agent_home=agent_home,
        session=session,
        web_search=configured_web_search,
        web_fetch=configured_web_fetch,
        shell=configured_shell,
        scheduled_work=scheduled_work_tool,
    )
    system_prompt = chat_system_prompt(
        workspace=workspace_identity.path,
        long_term_memory=long_term_memory,
        tool_guidance=render_tool_guidance(tool_gateway.schemas),
    )
    resolved_memory = configuration.resolve_route("memory")
    summaries = WorkspaceJsonlSummaryStore(workspace_state)
    summary_manager = ConversationSummaryManager(
        provider=router,
        summaries=summaries,
        settings=SummaryModelSettings(
            model=resolved_memory.route.model,
            max_output=resolved_memory.route.max_output,
            temperature=resolved_memory.route.temperature,
            reasoning_effort=resolved_memory.route.reasoning_effort,
            timeout_seconds=resolved_memory.route.timeout,
        ),
        chat_context_window=resolved.route.context_window,
        chat_max_output=resolved.route.max_output,
        consolidation_message_threshold=configuration.memory.consolidation_message_threshold,
        chat_system_prompt=system_prompt,
        tools=tool_gateway.schemas,
        now=now,
        new_uuid=new_uuid,
    )
    memory_manager = MemoryManager(
        provider=router,
        summaries=summaries,
        memory=memory_store,
        long_term_path=workspace_state.long_term_memory_path,
        settings=MemoryTaskModelSettings(
            model=resolved_memory.route.model,
            max_output=resolved_memory.route.max_output,
            temperature=resolved_memory.route.temperature,
            reasoning_effort=resolved_memory.route.reasoning_effort,
            timeout_seconds=resolved_memory.route.timeout,
        ),
        batch_size=configuration.memory.batch_size,
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

    def scheduled_work_gateway_for(session_id: str) -> ToolGateway:
        return _build_tool_gateway(
            workspace_state=workspace_state,
            agent_home=agent_home,
            session_id=session_id,
            web_search=configured_web_search,
            web_fetch=configured_web_fetch,
            shell=configured_shell,
            scheduled_work=scheduled_work_tool,
        )

    def externalize_result_for(session_id: str) -> ToolResultExternalizer:
        return _build_tool_result_externalizer(
            workspace_state=workspace_state,
            session_id=session_id,
            max_tool_result_chars=configuration.runtime.max_tool_result_chars,
        )

    active_externalize_result = _build_tool_result_externalizer(
        session=session,
        max_tool_result_chars=configuration.runtime.max_tool_result_chars,
    )

    resolved_cron = configuration.resolve_route("cron")
    scheduled_work_runner = ScheduledWorkRunner(
        provider=router,
        sessions=sessions,
        workspace=Path(workspace_identity.path),
        long_term_memory=long_term_memory,
        settings=ScheduledWorkModelSettings(
            model=resolved_cron.route.model,
            max_output=resolved_cron.route.max_output,
            temperature=resolved_cron.route.temperature,
            reasoning_effort=resolved_cron.route.reasoning_effort,
            timeout_seconds=resolved_cron.route.timeout,
        ),
        now=now,
        new_uuid=new_uuid,
        tool_gateway_for=scheduled_work_gateway_for,
        externalize_result_for=externalize_result_for,
    )
    background_events = RuntimeEventBroker()
    scheduled_work_coordinator = ScheduledWorkCoordinator(
        runner=scheduled_work_runner,
        events=background_events,
        now=now,
        new_uuid=new_uuid,
        workspace_state=workspace_state,
    )
    scheduled_scheduler_clock = (
        scheduled_work_scheduler_clock
        if scheduled_work_scheduler_clock is not None
        else AsyncioSchedulerClock(now=now)
    )
    scheduled_work_scheduler = _RuntimeSchedulerOwner(
        lambda: ScheduledWorkScheduler(
            store=scheduled_work_store,
            coordinator=scheduled_work_coordinator,
            clock=scheduled_scheduler_clock,
        )
    )
    foreground_chat_status: ResolvedChatStatus | None = None

    def capture_foreground_chat_status() -> None:
        nonlocal foreground_chat_status
        foreground_chat_status = _resolved_chat_status(router)

    def current_foreground_chat_status() -> ResolvedChatStatus:
        return (
            _resolved_chat_status(router)
            if foreground_chat_status is None
            else foreground_chat_status
        )

    def conversation_for(session_id: str) -> ConversationPort:
        if session_id == session.session_id:
            return _DeferredConversationPort(
                provider=router,
                session=session,
                settings=settings,
                now=now,
                new_uuid=new_uuid,
                system_prompt=system_prompt,
                title_prompt=session_title_prompt(),
                tool_gateway=tool_gateway,
                history_preparer=summary_manager.prepare,
                on_foreground_terminal=capture_foreground_chat_status,
                externalize_result=active_externalize_result,
                workspace_state=workspace_state,
            )
        session_tool_gateway = _build_tool_gateway(
            workspace_state=workspace_state,
            agent_home=agent_home,
            session_id=session_id,
            web_search=configured_web_search,
            web_fetch=configured_web_fetch,
            shell=configured_shell,
            scheduled_work=scheduled_work_tool,
        )
        return _DeferredConversationPort(
            provider=router,
            sessions=sessions,
            session_id=session_id,
            settings=settings,
            now=now,
            new_uuid=new_uuid,
            system_prompt=system_prompt,
            title_prompt=session_title_prompt(),
            tool_gateway=session_tool_gateway,
            on_foreground_terminal=capture_foreground_chat_status,
            externalize_result=externalize_result_for(session_id),
            workspace_state=workspace_state,
        )

    conversation = SwitchableConversationPort(
        session_id=session.session_id,
        build_conversation=conversation_for,
        event_sequencer=background_events,
    )
    status_service = RuntimeStatusService(
        session=session,
        sessions=sessions,
        resolved_chat=current_foreground_chat_status,
        next_input=lambda current_session: _runtime_status_input(
            current_session,
            system_prompt=system_prompt,
            current_time=now(),
            session_id=_runtime_status_session_id(current_session),
            tool_schemas=tool_gateway.schemas,
        ),
        monotonic=monotonic_now,
    )

    def switch_session(session_id: str) -> None:
        conversation.switch_session(session_id)
        status_service.use_session(session_id)

    management_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            agent_home,
            status_service=status_service,
            sessions=sessions,
            workspace=Path(workspace_identity.path),
            switch_session=switch_session,
            memory_manager=memory_manager,
            memory_store=memory_store,
        )
    )
    return PreparedReplRuntime(
        conversation=conversation,
        session=session,
        sessions=sessions,
        management_dispatcher=management_dispatcher,
        scheduled_work_coordinator=scheduled_work_coordinator,
        _shell=owned_shell,
        _memory_scheduler=memory_scheduler,
        _scheduled_work_scheduler=scheduled_work_scheduler,
        _background_events=background_events,
        _router=router,
        _lifetime=_RuntimeLifetime(),
    )


def _build_tool_gateway(
    *,
    agent_home: AgentHome,
    session: Session | None = None,
    workspace_state: WorkspaceState | None = None,
    session_id: str | None = None,
    web_search: WebSearchBoundary | None,
    web_fetch: WebFetchBoundary | None,
    shell: ShellBoundary | None,
    scheduled_work: CreateScheduledWorkTool,
) -> ToolGateway:
    if session is not None:
        if workspace_state is not None or session_id is not None:
            raise TypeError("Active Tool Gateway requires only a Session")
        resolved_workspace_state = session.workspace_state
        resolved_session_id = session.session_id
    else:
        if workspace_state is None or session_id is None:
            raise TypeError("Legacy Tool Gateway requires Workspace State and Session ID")
        resolved_workspace_state = workspace_state
        resolved_session_id = session_id
    workspace = resolved_workspace_state.workspace
    gateway = ToolGateway()
    security = Security(
        workspace=workspace,
        agent_home=agent_home.path,
        artifact_directory=(
            resolved_workspace_state.sessions_directory / "artifacts" / resolved_session_id
        ),
    )
    tools: list[BaseTool] = [
        ReadFileTool(security=security),
        ListFilesTool(security=security),
        SearchFilesTool(security=security),
        WriteFileTool(),
        EditFileTool(),
    ]
    if web_search is not None:
        tools.append(WebSearchTool(search=web_search))
    if web_fetch is not None:
        tools.append(WebFetchTool(fetcher=web_fetch))
    if shell is not None:
        tools.append(ShellTool(workspace=Path(workspace.path), boundary=shell))
    tools.append(scheduled_work)
    gateway.register_tools(tuple(tools))
    return gateway


def _build_tool_result_externalizer(
    *,
    session: Session | None = None,
    workspace_state: WorkspaceState | None = None,
    session_id: str | None = None,
    max_tool_result_chars: int,
) -> ToolResultExternalizer:
    if session is not None:
        if workspace_state is not None or session_id is not None:
            raise TypeError("Active Tool Artifact externalizer requires only a Session")
    elif workspace_state is None or session_id is None:
        raise TypeError("Legacy Tool Artifact externalizer requires Workspace State and Session ID")

    def externalize(result: ToolResult) -> ToolResult:
        return externalize_tool_result(
            result,
            workspace_state=workspace_state,
            session=session,
            session_id=session_id,
            max_tool_result_chars=max_tool_result_chars,
        )

    return externalize


def _resolved_chat_status(router: ModelRouter) -> ResolvedChatStatus:
    status = router.route_status("chat")
    return ResolvedChatStatus(
        provider_id=status.provider_id,
        model=status.model,
        context_window=status.context_window,
    )


def _runtime_status_input(
    session: Session | ConversationSession,
    *,
    system_prompt: str,
    current_time: datetime,
    session_id: str,
    tool_schemas: tuple[OpenAIToolSchema, ...],
) -> RuntimeStatusInput:
    if isinstance(session, Session):
        retained_messages = tuple(
            json.dumps(
                model_message.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for message in session.messages[session.last_consolidated :]
            if (model_message := model_message_from_session(message)) is not None
        )
    else:
        retained_messages = tuple(
            json.dumps(
                model_message.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for message in session.short_term_messages
            if (model_message := model_message_from_session(message)) is not None
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
        runtime_context=runtime_context(
            current_time=current_time,
            session_id=session_id,
        ),
    )


def _runtime_status_session_id(session: Session | ConversationSession) -> str:
    if isinstance(session, Session):
        return session.session_id
    return session.metadata.id


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
