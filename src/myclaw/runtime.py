"""Composition for one prepared command-line Conversation Session."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol
from uuid import UUID

from myclaw.agent_home import AgentHome
from myclaw.config import ProviderConfiguration, UserConfiguration
from myclaw.contracts import AgentEvent, ConversationPort, ModelProvider
from myclaw.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.prompts import chat_system_prompt
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

    async def run(
        self,
        *,
        input_reader: ReplInput,
        writer: ProgressiveWriter,
        management_dispatcher: ManagementDispatcher | None = None,
    ) -> None:
        await run_repl(
            conversation=self.conversation,
            input_reader=input_reader,
            writer=writer,
            management_dispatcher=management_dispatcher,
        )


class _DeferredConversationPort:
    def __init__(
        self,
        *,
        provider_configuration: ProviderConfiguration,
        provider_factory: ProviderFactory,
        sessions: JsonlSessionStore,
        session_id: str,
        settings: ChatModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        system_prompt: str,
    ) -> None:
        self._provider_configuration = provider_configuration
        self._provider_factory = provider_factory
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
                provider=self._provider_factory(self._provider_configuration),
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
    conversation = _DeferredConversationPort(
        provider_configuration=resolved.provider,
        provider_factory=provider_factory,
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
    )
