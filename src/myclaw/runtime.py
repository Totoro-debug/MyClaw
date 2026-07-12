"""Composition for one prepared command-line Conversation Session."""

import json
from collections.abc import AsyncIterator, Callable
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
    AssistantModelMessage,
    AssistantSessionMessage,
    ConversationPort,
    ConversationSession,
    ModelProvider,
    ToolModelMessage,
    ToolSessionMessage,
    UserModelMessage,
    UserSessionMessage,
)
from myclaw.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.management import (
    ManagementViewService,
    ResolvedChatStatus,
    RuntimeStatusInput,
    RuntimeStatusService,
)
from myclaw.management_commands import ManagementCommandDispatcher
from myclaw.model_router import AsyncioRetryClock, Jitter, ModelRouter, RetryClock
from myclaw.prompts import chat_system_prompt, runtime_context
from myclaw.repl import ManagementDispatcher, ProgressiveWriter, ReplInput, run_repl
from myclaw.session_store import JsonlSessionStore
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

    conversation: ConversationPort
    sessions: JsonlSessionStore
    session_id: str
    management_dispatcher: ManagementDispatcher

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
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._session_id = session_id
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._system_prompt = system_prompt
        self._delegate: ConversationPort | None = None

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if not text.strip():
            return
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
) -> PreparedReplRuntime:
    """Prepare a Session and defer provider construction until conversational input."""
    configuration.resolve_route("default")
    workspace_identity = Workspace.from_path(workspace)
    system_prompt = chat_system_prompt(
        workspace=workspace_identity.path,
        long_term_memory=(agent_home.path / "memory" / "memory.md").read_text(encoding="utf-8"),
    )
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
    status_service = RuntimeStatusService(
        sessions=sessions,
        session_id=metadata.id,
        resolved_chat=lambda: _resolved_chat_status(router),
        next_input=lambda session: _runtime_status_input(
            session,
            system_prompt=system_prompt,
            current_time=now(),
            session_id=metadata.id,
        ),
        monotonic=monotonic_now,
    )
    management_dispatcher = ManagementCommandDispatcher(
        ManagementViewService(agent_home, status_service=status_service)
    )
    conversation = _DeferredConversationPort(
        provider=router,
        sessions=sessions,
        session_id=metadata.id,
        settings=settings,
        now=now,
        new_uuid=new_uuid,
        system_prompt=system_prompt,
    )
    return PreparedReplRuntime(
        conversation=conversation,
        sessions=sessions,
        session_id=metadata.id,
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
) -> RuntimeStatusInput:
    retained_messages = tuple(
        json.dumps(
            _model_message_for_status(message).to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        for message in session.short_term_messages
    )
    return RuntimeStatusInput(
        system_prompt=system_prompt,
        retained_messages=retained_messages,
        tool_definitions=(),
        runtime_context=runtime_context(
            current_time=current_time,
            session_id=session_id,
        ),
    )


def _model_message_for_status(
    message: UserSessionMessage | AssistantSessionMessage | ToolSessionMessage,
) -> UserModelMessage | AssistantModelMessage | ToolModelMessage:
    if isinstance(message, UserSessionMessage):
        return UserModelMessage(content=message.content)
    if isinstance(message, AssistantSessionMessage):
        return AssistantModelMessage(content=message.content, tool_calls=message.tool_calls)
    return ToolModelMessage(
        tool_call_id=message.tool_call_id,
        name=message.name,
        content=message.content,
    )
