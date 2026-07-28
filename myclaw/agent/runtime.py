"""Composition for one prepared command-line Conversation Session."""

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import UUID

from myclaw.agent.events import AgentEvent
from myclaw.agent.ports import ConversationPort
from myclaw.agent.prompts import (
    chat_system_prompt,
    render_tool_guidance,
    runtime_context,
    session_title_prompt,
)
from myclaw.agent.turn import ToolResultExternalizer, model_message_from_session
from myclaw.agent.workspace import Workspace
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ProviderConfiguration, UserConfiguration
from myclaw.management.commands import ManagementCommandDispatcher
from myclaw.management.service import (
    ManagementViewService,
    ResolvedChatStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
)
from myclaw.memory.conversation_summary import (
    ConversationSummaryManager,
    JsonlSummaryStore,
    SummaryModelSettings,
)
from myclaw.memory.memory_scheduler import (
    AsyncioMemorySchedulerClock,
    MemorySchedulerClock,
    MemoryTaskScheduler,
)
from myclaw.memory.memory_task import FileMemoryStore, MemoryManager, MemoryTaskModelSettings
from myclaw.provider.model_router import AsyncioRetryClock, Jitter, ModelRouter, RetryClock
from myclaw.provider.ports import ModelProvider
from myclaw.runtime_log import RuntimeLogLifetime, log_sanitized_exception
from myclaw.schedule.background_coordination import (
    AsyncioScheduledWorkSchedulerClock,
    RuntimeEventBroker,
    ScheduledWorkCoordinator,
    ScheduledWorkScheduler,
    ScheduledWorkSchedulerClock,
)
from myclaw.schedule.scheduled_work import CreateScheduledWorkTool, JsonScheduledWorkStore
from myclaw.schedule.scheduled_work_execution import (
    ScheduledWorkModelSettings,
    ScheduledWorkRunner,
)
from myclaw.session.conversation import (
    ChatModelSettings,
    StreamingConversationPort,
)
from myclaw.session.records import ConversationSession
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

logger = logging.getLogger(__name__)


class ProviderFactory(Protocol):
    def __call__(self, configuration: ProviderConfiguration) -> ModelProvider: ...


class ProviderAdapterUnavailable(RuntimeError):
    """Raised when production composition has no installed adapter for a provider."""


def unavailable_provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
    """Fail closed until a production Provider Adapter owns this boundary."""
    raise ProviderAdapterUnavailable(
        f"Provider Adapter for protocol '{configuration.protocol}' is not available."
    )


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
    sessions: JsonlSessionStore
    management_dispatcher: ManagementDispatcher
    scheduled_work_runner: ScheduledWorkRunner
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
        try:
            self._memory_scheduler.start()
            self._scheduled_work_scheduler.start()
        except BaseException as error:
            log_sanitized_exception(
                logger,
                logging.ERROR,
                f"Runtime startup failed type={type(error).__name__}",
                error,
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
            return
        failure = (
            failures[0]
            if len(failures) == 1
            else BaseExceptionGroup("Runtime shutdown failed", failures)
        )
        log_sanitized_exception(
            logger,
            logging.ERROR,
            f"Runtime shutdown failed type={type(failure).__name__}",
            failure,
        )
        raise failure


class _DeferredConversationPort:
    def __init__(
        self,
        *,
        provider: ModelProvider,
        sessions: JsonlSessionStore,
        session_id: str,
        settings: ChatModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str,
        title_prompt: str,
        tool_gateway: ToolGateway,
        history_preparer: Callable[[ConversationSession], Awaitable[ConversationSession]],
        before_submit: Callable[[], Awaitable[None]],
        on_foreground_terminal: Callable[[], None],
        runtime_log: RuntimeLogLifetime | None = None,
        externalize_result: ToolResultExternalizer | None = None,
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._session_id = session_id
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._system_prompt = system_prompt
        self._title_prompt = title_prompt
        self._tool_gateway = tool_gateway
        self._history_preparer = history_preparer
        self._before_submit = before_submit
        self._on_foreground_terminal = on_foreground_terminal
        self._runtime_log = runtime_log
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
        correlation = (
            nullcontext()
            if self._runtime_log is None
            else self._runtime_log.session(self._session_id)
        )
        with correlation:
            try:
                await self._before_submit()
                if self._close_task is not None:
                    raise RuntimeError("Conversation Port is closed")
                delegate = self._delegate
                if delegate is None:
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
    provider_factory: ProviderFactory,
    now: Callable[[], datetime],
    new_uuid: Callable[[], UUID],
    retry_clock: RetryClock | None = None,
    retry_jitter: Jitter | None = None,
    memory_scheduler_clock: MemorySchedulerClock | None = None,
    scheduled_work_scheduler_clock: ScheduledWorkSchedulerClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    web_search: WebSearchBoundary | None = None,
    web_fetch: WebFetchBoundary | None = None,
    shell: ShellBoundary | None = None,
    runtime_log: RuntimeLogLifetime | None = None,
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
            runtime_log=runtime_log,
        )
    except Exception as error:
        log_sanitized_exception(
            logger,
            logging.ERROR,
            f"Runtime composition failed type={type(error).__name__}",
            error,
        )
        raise


def _prepare_repl_runtime(
    *,
    agent_home: AgentHome,
    workspace: Path,
    configuration: UserConfiguration,
    provider_factory: ProviderFactory,
    now: Callable[[], datetime],
    new_uuid: Callable[[], UUID],
    retry_clock: RetryClock | None = None,
    retry_jitter: Jitter | None = None,
    memory_scheduler_clock: MemorySchedulerClock | None = None,
    scheduled_work_scheduler_clock: ScheduledWorkSchedulerClock | None = None,
    monotonic_now: Callable[[], float] = monotonic,
    web_search: WebSearchBoundary | None = None,
    web_fetch: WebFetchBoundary | None = None,
    shell: ShellBoundary | None = None,
    runtime_log: RuntimeLogLifetime | None = None,
) -> PreparedReplRuntime:
    """Prepare a Session and defer provider construction until conversational input."""
    configuration.resolve_route("default")
    workspace_identity = Workspace.from_path(workspace)
    long_term_memory = (agent_home.path / "memory" / "memory.md").read_text(encoding="utf-8")
    sessions = JsonlSessionStore(
        agent_home=agent_home,
        workspace=workspace_identity,
        now=now,
        new_uuid=new_uuid,
    )
    metadata = sessions.prepare()
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
        clock=retry_clock if retry_clock is not None else AsyncioRetryClock(),
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
    scheduled_work_store = JsonScheduledWorkStore(agent_home)
    scheduled_work_tool = CreateScheduledWorkTool()
    tool_gateway = _build_tool_gateway(
        workspace=workspace_identity,
        agent_home=agent_home,
        session_id=metadata.id,
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
    summaries = JsonlSummaryStore(agent_home)
    summary_manager = ConversationSummaryManager(
        provider=router,
        sessions=sessions,
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
        consolidation_message_threshold=(configuration.memory.consolidation_message_threshold),
        chat_system_prompt=system_prompt,
        tools=tool_gateway.schemas,
        now=now,
        new_uuid=new_uuid,
    )
    memory_manager = MemoryManager(
        provider=router,
        summaries=summaries,
        memory=FileMemoryStore(agent_home),
        long_term_path=agent_home.path / "memory" / "memory.md",
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
        else AsyncioMemorySchedulerClock(now=now)
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
            workspace=workspace_identity,
            agent_home=agent_home,
            session_id=session_id,
            web_search=configured_web_search,
            web_fetch=configured_web_fetch,
            shell=configured_shell,
            scheduled_work=scheduled_work_tool,
        )

    def externalize_result_for(session_id: str) -> ToolResultExternalizer:
        return _build_tool_result_externalizer(
            workspace=workspace_identity,
            agent_home=agent_home,
            session_id=session_id,
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
    )
    scheduled_scheduler_clock = (
        scheduled_work_scheduler_clock
        if scheduled_work_scheduler_clock is not None
        else AsyncioScheduledWorkSchedulerClock(now=now)
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
        session_tool_gateway = _build_tool_gateway(
            workspace=workspace_identity,
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
            history_preparer=summary_manager.prepare,
            before_submit=summary_manager.recover_pending,
            on_foreground_terminal=capture_foreground_chat_status,
            runtime_log=runtime_log,
            externalize_result=externalize_result_for(session_id),
        )

    conversation = SwitchableConversationPort(
        session_id=metadata.id,
        build_conversation=conversation_for,
        event_sequencer=background_events,
    )
    status_service = RuntimeStatusService(
        sessions=sessions,
        session_id=lambda: conversation.session_id,
        resolved_chat=current_foreground_chat_status,
        next_input=lambda session: _runtime_status_input(
            session,
            system_prompt=system_prompt,
            current_time=now(),
            session_id=conversation.session_id,
            tool_schemas=tool_gateway.schemas,
        ),
        monotonic=monotonic_now,
    )
    management_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(
            agent_home,
            status_service=status_service,
            sessions=sessions,
            workspace=Path(workspace_identity.path),
            switch_session=conversation.switch_session,
            memory_manager=memory_manager,
        )
    )
    return PreparedReplRuntime(
        conversation=conversation,
        sessions=sessions,
        management_dispatcher=management_dispatcher,
        scheduled_work_runner=scheduled_work_runner,
        _shell=owned_shell,
        _memory_scheduler=memory_scheduler,
        _scheduled_work_scheduler=scheduled_work_scheduler,
        _background_events=background_events,
        _router=router,
        _lifetime=_RuntimeLifetime(),
    )


def _build_tool_gateway(
    *,
    workspace: Workspace,
    agent_home: AgentHome,
    session_id: str,
    web_search: WebSearchBoundary | None,
    web_fetch: WebFetchBoundary | None,
    shell: ShellBoundary | None,
    scheduled_work: CreateScheduledWorkTool,
) -> ToolGateway:
    gateway = ToolGateway()
    security = Security(
        workspace=workspace,
        agent_home=agent_home.path,
        artifact_directory=(
            agent_home.path / "sessions" / workspace.slug / "artifacts" / session_id
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
    workspace: Workspace,
    agent_home: AgentHome,
    session_id: str,
    max_tool_result_chars: int,
) -> ToolResultExternalizer:
    def externalize(result: ToolResult) -> ToolResult:
        return externalize_tool_result(
            result,
            agent_home=agent_home.path,
            workspace=workspace,
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
    session: ConversationSession,
    *,
    system_prompt: str,
    current_time: datetime,
    session_id: str,
    tool_schemas: tuple[OpenAIToolSchema, ...],
) -> RuntimeStatusInput:
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
