"""Composition for one prepared command-line Conversation Session."""

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import UUID

from myclaw.agent_home import AgentHome
from myclaw.config import ProviderConfiguration, UserConfiguration
from myclaw.contracts import (
    AgentEvent,
    ConversationPort,
    ConversationSession,
    ModelProvider,
    ToolDefinition,
    ToolExecutionContext,
)
from myclaw.conversation import (
    ChatModelSettings,
    StreamingConversationPort,
    model_message_from_session,
)
from myclaw.conversation_summary import (
    ConversationSummaryManager,
    JsonlSummaryStore,
    SummaryModelSettings,
)
from myclaw.management import (
    ManagementViewService,
    ResolvedChatStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
)
from myclaw.management_commands import ManagementCommandDispatcher
from myclaw.model_router import AsyncioRetryClock, Jitter, ModelRouter, RetryClock
from myclaw.prompts import chat_system_prompt, runtime_context, session_title_prompt
from myclaw.repl import ManagementDispatcher, ProgressiveWriter, ReplInput, run_repl
from myclaw.scheduled_work import CreateScheduledWorkTool, JsonScheduledWorkStore
from myclaw.session_resume import SwitchableConversationPort
from myclaw.session_store import JsonlSessionStore
from myclaw.shell_tool import ShellBoundary, UnavailableShellBoundary
from myclaw.tool_gateway import ToolGateway
from myclaw.web_fetch import (
    AioHttpWebFetchClient,
    PublicWebFetchBoundary,
    SocketDNSResolver,
    WebFetchBoundary,
)
from myclaw.web_search import DuckDuckGoSearchBoundary, WebSearchBoundary
from myclaw.workspace import Workspace


class ProviderFactory(Protocol):
    def __call__(self, configuration: ProviderConfiguration) -> ModelProvider: ...


class ProviderAdapterUnavailable(RuntimeError):
    """Raised when production composition has no installed adapter for a provider."""


def unavailable_provider_factory(configuration: ProviderConfiguration) -> ModelProvider:
    """Fail closed until a production Provider Adapter owns this boundary."""
    raise ProviderAdapterUnavailable(
        f"Provider Adapter for protocol '{configuration.protocol}' is not available."
    )


@dataclass(frozen=True, slots=True)
class PreparedReplRuntime:
    """An in-memory Session identity and its injectable REPL composition."""

    conversation: SwitchableConversationPort
    sessions: JsonlSessionStore
    management_dispatcher: ManagementDispatcher

    @property
    def session_id(self) -> str:
        return self.conversation.session_id

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
        await run_repl(
            conversation=self.conversation,
            input_reader=input_reader,
            writer=writer,
            management_dispatcher=dispatcher,
        )


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
        self._delegate: ConversationPort | None = None

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if not text.strip():
            return
        await self._before_submit()
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
            )
            self._delegate = delegate
        async for event in delegate.submit(text):
            yield event

    async def resolve_permission(self, request_id: UUID, approved: bool) -> None:
        if self._delegate is None:
            raise RuntimeError("No foreground turn is active")
        await self._delegate.resolve_permission(request_id, approved)

    async def cancel_active_turn(self) -> None:
        if self._delegate is not None:
            await self._delegate.cancel_active_turn()


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
    monotonic_now: Callable[[], float] = monotonic,
    web_search: WebSearchBoundary | None = None,
    web_fetch: WebFetchBoundary | None = None,
    shell: ShellBoundary | None = None,
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
        (shell if shell is not None else UnavailableShellBoundary())
        if configuration.tools.shell.enabled
        else None
    )
    scheduled_work_tool = CreateScheduledWorkTool(
        store=JsonScheduledWorkStore(agent_home),
        now=now,
        new_uuid=new_uuid,
    )
    tool_gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=Path(workspace_identity.path),
            agent_home=agent_home.path,
            session_id=metadata.id,
        ),
        max_tool_result_chars=configuration.runtime.max_tool_result_chars,
        web_search=configured_web_search,
        web_fetch=configured_web_fetch,
        shell=configured_shell,
        scheduled_work=scheduled_work_tool,
    )
    system_prompt = chat_system_prompt(
        workspace=workspace_identity.path,
        long_term_memory=long_term_memory,
        tool_guidance="\n".join(
            f"- {definition.name}: {definition.description}"
            for definition in tool_gateway.definitions
        ),
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
        tools=tool_gateway.definitions,
        now=now,
        new_uuid=new_uuid,
    )

    def conversation_for(session_id: str) -> ConversationPort:
        session_tool_gateway = ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=Path(workspace_identity.path),
                agent_home=agent_home.path,
                session_id=session_id,
            ),
            max_tool_result_chars=configuration.runtime.max_tool_result_chars,
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
        )

    conversation = SwitchableConversationPort(
        session_id=metadata.id,
        build_conversation=conversation_for,
    )
    status_service = RuntimeStatusService(
        sessions=sessions,
        session_id=lambda: conversation.session_id,
        resolved_chat=lambda: _resolved_chat_status(router),
        next_input=lambda session: _runtime_status_input(
            session,
            system_prompt=system_prompt,
            current_time=now(),
            session_id=conversation.session_id,
            tool_definitions=tool_gateway.definitions,
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
        )
    )
    return PreparedReplRuntime(
        conversation=conversation,
        sessions=sessions,
        management_dispatcher=management_dispatcher,
    )


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
    tool_definitions: tuple[ToolDefinition, ...],
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
                definition.to_dict(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            for definition in tool_definitions
        ),
        runtime_context=runtime_context(
            current_time=current_time,
            session_id=session_id,
        ),
    )
