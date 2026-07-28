"""Concrete read-only views exposed through the Management Port."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from myclaw import __version__
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.config.models import ConfigView
from myclaw.errors import ErrorInfo
from myclaw.management.models import RuntimeStatus
from myclaw.memory.models import MemoryTaskResult
from myclaw.session.models import ResumeResult
from myclaw.session.records import ConversationSession, SessionSummary
from myclaw.session.session_store import SessionListingReport


class _CurrentSessionStore(Protocol):
    async def current_session(self, session_id: str) -> ConversationSession: ...


class _ManualMemoryManager(Protocol):
    async def run_manual(self) -> MemoryTaskResult: ...


class _ResumableSessionStore(_CurrentSessionStore, Protocol):
    async def load(self, session_id: str) -> ConversationSession: ...

    async def scan_for_workspace(self, workspace: Path) -> SessionListingReport: ...


@dataclass(frozen=True, slots=True)
class ResolvedChatStatus:
    """The actual provider/model identity and context window used for chat."""

    provider_id: str
    model: str
    context_window: int

    def __post_init__(self) -> None:
        if not self.provider_id or not self.model:
            msg = "provider_id and model must not be empty"
            raise ValueError(msg)
        if self.context_window <= 0:
            msg = "context_window must be positive"
            raise ValueError(msg)

    @property
    def chat_model(self) -> str:
        return f"{self.provider_id}/{self.model}"


@dataclass(frozen=True, slots=True)
class RuntimeStatusInput:
    """Exact next-request text fragments included in the token estimate."""

    system_prompt: str
    retained_messages: tuple[str, ...]
    tool_definitions: tuple[str, ...]
    runtime_context: str


def estimate_input_tokens(status_input: RuntimeStatusInput) -> int:
    """Estimate tokens as ceil(total UTF-8 bytes / 4)."""
    components = (
        status_input.system_prompt,
        *status_input.retained_messages,
        *status_input.tool_definitions,
        status_input.runtime_context,
    )
    byte_count = sum(len(component.encode("utf-8")) for component in components)
    return (byte_count + 3) // 4


class RuntimeStatusService:
    """Build one Management Port status snapshot from injectable runtime state."""

    def __init__(
        self,
        *,
        sessions: _CurrentSessionStore,
        session_id: str | Callable[[], str],
        resolved_chat: Callable[[], ResolvedChatStatus],
        next_input: Callable[[ConversationSession], RuntimeStatusInput],
        monotonic: Callable[[], float],
        version: str = __version__,
    ) -> None:
        self._sessions = sessions
        self._session_id: Callable[[], str]
        if isinstance(session_id, str):
            self._session_id = lambda: session_id
        else:
            self._session_id = session_id
        self._resolved_chat = resolved_chat
        self._next_input = next_input
        self._monotonic = monotonic
        self._started_at = monotonic()
        self._version = version

    async def status(self) -> RuntimeStatus:
        """Return all required runtime and current-session status fields."""
        session = await self._sessions.current_session(self._session_id())
        resolved = self._resolved_chat()
        estimated = estimate_input_tokens(self._next_input(session))
        uptime = max(0, int(self._monotonic() - self._started_at))
        return RuntimeStatus(
            version=self._version,
            chat_model=resolved.chat_model,
            uptime_seconds=uptime,
            estimated_input_tokens=estimated,
            context_window=resolved.context_window,
            context_used_percent=estimated / resolved.context_window * 100,
            session_message_count=len(session.messages),
            consolidation_cursor=session.metadata.consolidation_cursor,
            cumulative_usage=session.metadata.cumulative_usage,
        )


class ManagementError(Exception):
    """A safe persistence error suitable for a Management Command."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class ManagementViewService:
    """Read configuration and Long-term Memory from Agent Home."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        status_service: RuntimeStatusService | None = None,
        sessions: _ResumableSessionStore | None = None,
        workspace: Path | None = None,
        switch_session: Callable[[str], None] | None = None,
        memory_manager: _ManualMemoryManager | None = None,
    ) -> None:
        self._config = ConfigLoader(agent_home)
        self._long_term_memory = agent_home.path / "memory" / "memory.md"
        self._status_service = status_service
        self._sessions = sessions
        self._workspace = workspace
        self._switch_session = switch_session
        self._memory_manager = memory_manager

    async def config_view(self) -> ConfigView:
        """Return complete redacted User Configuration content."""
        try:
            self._config.ensure_default()
            return self._config.view()
        except (OSError, UnicodeError) as error:
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "User Configuration could not be read or written.",
                )
            ) from error

    async def memory_view(self) -> str:
        """Return the complete current Long-term Memory file."""
        try:
            return self._long_term_memory.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Long-term Memory could not be read.")
            ) from error

    async def dream(self) -> MemoryTaskResult:
        """Run one foreground Memory Task and return its safe summary."""
        if self._memory_manager is None:
            raise ManagementError(ErrorInfo("route_unavailable", "Memory Task is unavailable."))
        return await self._memory_manager.run_manual()

    async def status(self) -> RuntimeStatus:
        """Return the injected Runtime status snapshot."""
        if self._status_service is None:
            raise ManagementError(ErrorInfo("route_unavailable", "Runtime status is unavailable."))
        try:
            return await self._status_service.status()
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Runtime status could not be read.")
            ) from error

    async def resumable_sessions(self) -> tuple[SessionSummary, ...]:
        """Return only valid Sessions belonging to this runtime's Workspace."""
        return (await self.resumable_listing()).sessions

    async def resumable_listing(self) -> SessionListingReport:
        """Return one atomic Session picker result including skipped diagnostics."""
        if self._sessions is None or self._workspace is None:
            raise ManagementError(ErrorInfo("route_unavailable", "Session resume is unavailable."))
        try:
            return await self._sessions.scan_for_workspace(self._workspace)
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Conversation Sessions could not be listed.")
            ) from error

    async def resume(self, session_id: str) -> ResumeResult:
        """Revalidate and select one Session from the current Workspace."""
        if self._sessions is None or self._switch_session is None:
            raise ManagementError(ErrorInfo("route_unavailable", "Session resume is unavailable."))
        sessions = await self.resumable_sessions()
        if session_id not in {summary.id for summary in sessions}:
            raise ManagementError(
                ErrorInfo(
                    "model_invalid_request",
                    "The selected Conversation Session is not resumable.",
                )
            )
        try:
            session = await self._sessions.load(session_id)
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "The selected Conversation Session could not be loaded.",
                )
            ) from error
        self._switch_session(session.metadata.id)
        return ResumeResult(session_id=session.metadata.id)
