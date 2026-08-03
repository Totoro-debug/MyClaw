"""Concrete read-only views exposed through the Management Port."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from myclaw import __version__
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ConfigView
from myclaw.errors import ErrorInfo
from myclaw.memory.memory_task import MemoryStore, MemoryTaskResult
from myclaw.session.identifiers import require_session_id
from myclaw.session.records import ConversationSession, CumulativeUsage, SessionSummary
from myclaw.session.session import Session
from myclaw.session.session_store import SessionListingReport
from myclaw.utils.validation import require_nonnegative_int, require_nonnegative_number


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


@dataclass(frozen=True, slots=True)
class RuntimeStatus:
    """The required observable fields for the `/status` view."""

    version: str
    chat_model: str
    uptime_seconds: int
    estimated_input_tokens: int
    context_window: int
    context_used_percent: float
    session_message_count: int
    consolidation_cursor: int
    cumulative_usage: CumulativeUsage

    def __post_init__(self) -> None:
        require_nonnegative_int(self.uptime_seconds, field="uptime_seconds")
        require_nonnegative_int(self.estimated_input_tokens, field="estimated_input_tokens")
        require_nonnegative_int(self.context_window, field="context_window")
        require_nonnegative_number(self.context_used_percent, field="context_used_percent")
        require_nonnegative_int(self.session_message_count, field="session_message_count")
        require_nonnegative_int(self.consolidation_cursor, field="consolidation_cursor")

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "chat_model": self.chat_model,
            "uptime_seconds": self.uptime_seconds,
            "estimated_input_tokens": self.estimated_input_tokens,
            "context_window": self.context_window,
            "context_used_percent": self.context_used_percent,
            "session_message_count": self.session_message_count,
            "consolidation_cursor": self.consolidation_cursor,
            "cumulative_usage": self.cumulative_usage.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ResumeResult:
    """Identity of the Conversation Session selected by a successful resume."""

    session_id: str

    def __post_init__(self) -> None:
        require_session_id(self.session_id)


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
        session: Session | Callable[[], Session] | None = None,
        sessions: _CurrentSessionStore | None = None,
        session_id: str | Callable[[], str] | None = None,
        resolved_chat: Callable[[], ResolvedChatStatus],
        next_input: Callable[[Session | ConversationSession], RuntimeStatusInput],
        monotonic: Callable[[], float],
        version: str = __version__,
    ) -> None:
        self._session: Callable[[], Session] | None
        self._initial_session: Callable[[], Session] | None
        self._sessions: _CurrentSessionStore | None
        self._session_id: Callable[[], str] | None
        if session is not None:
            if isinstance(session, Session):
                self._session = lambda: session
            else:
                self._session = session
            self._initial_session = self._session
            self._sessions = sessions
            if session_id is None:
                self._session_id = None
            elif isinstance(session_id, str):
                self._session_id = lambda: session_id
            else:
                self._session_id = session_id
        else:
            if sessions is None or session_id is None:
                raise TypeError("Legacy status requires a Session Store and Session ID")
            self._session = None
            self._initial_session = None
            self._sessions = sessions
            if isinstance(session_id, str):
                self._session_id = lambda: session_id
            else:
                self._session_id = session_id
        self._resolved_chat = resolved_chat
        self._next_input = next_input
        self._monotonic = monotonic
        self._started_at = monotonic()
        self._version = version

    def use_session(self, session_id: str) -> None:
        """Select legacy Store-backed status only after the existing resume flow switches."""
        require_session_id(session_id)
        initial_session = self._initial_session
        if initial_session is not None and initial_session().session_id == session_id:
            self._session = initial_session
            return
        if self._sessions is None:
            raise RuntimeError("Legacy Session status is unavailable")
        self._session = None
        self._session_id = lambda: session_id

    async def status(self) -> RuntimeStatus:
        """Return all required runtime and current-session status fields."""
        active_session = self._session
        if active_session is not None:
            session = active_session()
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
                consolidation_cursor=session.last_consolidated,
                cumulative_usage=_active_session_usage(session),
            )
        sessions = self._sessions
        session_id = self._session_id
        assert sessions is not None
        assert session_id is not None
        persisted_session = await sessions.current_session(session_id())
        resolved = self._resolved_chat()
        estimated = estimate_input_tokens(self._next_input(persisted_session))
        uptime = max(0, int(self._monotonic() - self._started_at))
        return RuntimeStatus(
            version=self._version,
            chat_model=resolved.chat_model,
            uptime_seconds=uptime,
            estimated_input_tokens=estimated,
            context_window=resolved.context_window,
            context_used_percent=estimated / resolved.context_window * 100,
            session_message_count=len(persisted_session.messages),
            consolidation_cursor=persisted_session.metadata.consolidation_cursor,
            cumulative_usage=persisted_session.metadata.cumulative_usage,
        )


def _active_session_usage(session: Session) -> CumulativeUsage:
    value = session.metadata.get("token_usage")
    if not isinstance(value, dict):
        raise ValueError("Active Session token usage is malformed")
    fields = ("model_calls", "input_tokens", "output_tokens", "total_tokens")
    usage = {field: value.get(field) for field in fields}
    if any(isinstance(item, bool) or not isinstance(item, int) for item in usage.values()):
        raise ValueError("Active Session token usage is malformed")
    return CumulativeUsage(
        model_calls=cast(int, usage["model_calls"]),
        input_tokens=cast(int, usage["input_tokens"]),
        output_tokens=cast(int, usage["output_tokens"]),
        total_tokens=cast(int, usage["total_tokens"]),
    )


class ManagementError(Exception):
    """A safe persistence error suitable for a Management Command."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class ManagementViewService:
    """Read global configuration and injected runtime-owned views."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        status_service: RuntimeStatusService | None = None,
        sessions: _ResumableSessionStore | None = None,
        workspace: Path | None = None,
        switch_session: Callable[[str], None] | None = None,
        memory_manager: _ManualMemoryManager | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._config = ConfigLoader(agent_home)
        self._status_service = status_service
        self._sessions = sessions
        self._workspace = workspace
        self._switch_session = switch_session
        self._memory_manager = memory_manager
        self._memory_store = memory_store

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
        if self._memory_store is None:
            raise ManagementError(
                ErrorInfo("route_unavailable", "Long-term Memory is unavailable.")
            )
        try:
            return await self._memory_store.read_long_term()
        except (OSError, UnicodeError, ValueError) as error:
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
