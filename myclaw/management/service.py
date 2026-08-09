"""Concrete read-only views exposed through the Management Port."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

from loguru import logger

from myclaw import __version__
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ConfigView
from myclaw.errors import ErrorInfo
from myclaw.memory.memory_task import MemoryStore, MemoryTaskResult
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_nonnegative_int, require_nonnegative_number


class _ManualMemoryManager(Protocol):
    async def run_manual(self) -> MemoryTaskResult: ...


@dataclass(frozen=True, slots=True)
class SessionListingEntry:
    """The fields needed to render one resumable Conversation Session."""

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

    def __post_init__(self) -> None:
        Session._require_id(self.id, field="id", partition=SessionStoragePartition.FOREGROUND)
        if not self.title or " ".join(self.title.split()) != self.title or len(self.title) > 60:
            raise ValueError("title is not normalized")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        if self.updated_at.tzinfo is None or self.updated_at.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        require_nonnegative_int(self.message_count, field="message_count")

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "created_at": format_rfc3339_milliseconds(self.created_at),
            "updated_at": format_rfc3339_milliseconds(self.updated_at),
            "message_count": self.message_count,
        }


@dataclass(frozen=True, slots=True)
class SessionListingReport:
    """Valid current-format Sessions and the entries skipped during listing."""

    sessions: tuple[SessionListingEntry, ...]
    skipped_count: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.skipped_count, bool)
            or not isinstance(self.skipped_count, int)
            or self.skipped_count < 0
        ):
            raise ValueError("skipped_count must be a nonnegative integer")


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
    last_consolidated: int
    cumulative_usage: dict[str, int]
    schedule: dict[str, object] | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.uptime_seconds, field="uptime_seconds")
        require_nonnegative_int(self.estimated_input_tokens, field="estimated_input_tokens")
        require_nonnegative_int(self.context_window, field="context_window")
        require_nonnegative_number(self.context_used_percent, field="context_used_percent")
        require_nonnegative_int(self.session_message_count, field="session_message_count")
        require_nonnegative_int(self.last_consolidated, field="last_consolidated")

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "version": self.version,
            "chat_model": self.chat_model,
            "uptime_seconds": self.uptime_seconds,
            "estimated_input_tokens": self.estimated_input_tokens,
            "context_window": self.context_window,
            "context_used_percent": self.context_used_percent,
            "session_message_count": self.session_message_count,
            "last_consolidated": self.last_consolidated,
            "cumulative_usage": dict(self.cumulative_usage),
        }
        if self.schedule is not None:
            result["schedule"] = dict(self.schedule)
        return result


@dataclass(frozen=True, slots=True)
class ResumeResult:
    """Identity of the Conversation Session selected by a successful resume."""

    session_id: str

    def __post_init__(self) -> None:
        Session._require_id(self.session_id, partition=SessionStoragePartition.FOREGROUND)


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
        session: Session | Callable[[], Session],
        resolved_chat: Callable[[], ResolvedChatStatus],
        next_input: Callable[[Session], RuntimeStatusInput],
        monotonic: Callable[[], float],
        schedule_status: Callable[[], dict[str, object]] | None = None,
        version: str = __version__,
    ) -> None:
        self._session: Callable[[], Session]
        if isinstance(session, Session):
            self._session = lambda: session
        else:
            self._session = session
        self._resolved_chat = resolved_chat
        self._next_input = next_input
        self._monotonic = monotonic
        self._schedule_status = schedule_status
        self._started_at = monotonic()
        self._version = version

    def use_session(self, session: Session) -> None:
        """Make the selected Session the status authority."""
        self._session = lambda: session

    async def status(self) -> RuntimeStatus:
        """Return all required runtime and current-session status fields."""
        session = self._session()
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
            last_consolidated=session.last_consolidated,
            cumulative_usage=_active_session_usage(session),
            schedule=(None if self._schedule_status is None else dict(self._schedule_status())),
        )


def _active_session_usage(session: Session) -> dict[str, int]:
    value = session.metadata.get("token_usage")
    if not isinstance(value, dict):
        raise ValueError("Active Session token usage is malformed")
    fields = ("model_calls", "input_tokens", "output_tokens", "total_tokens")
    usage = {field: value.get(field) for field in fields}
    if any(isinstance(item, bool) or not isinstance(item, int) for item in usage.values()):
        raise ValueError("Active Session token usage is malformed")
    return {field: cast(int, usage[field]) for field in fields}


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
        workspace_state: WorkspaceState | None = None,
        switch_session: Callable[[Session], None] | None = None,
        now: Callable[[], datetime] | None = None,
        memory_manager: _ManualMemoryManager | None = None,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self._config = ConfigLoader(agent_home)
        self._status_service = status_service
        self._workspace_state = workspace_state
        self._switch_session = switch_session
        self._now = now
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

    async def resumable_sessions(self) -> tuple[SessionListingEntry, ...]:
        """Return only valid Sessions belonging to this runtime's Workspace."""
        return (await self.resumable_listing()).sessions

    async def resumable_listing(self) -> SessionListingReport:
        """Return one atomic Session picker result including skipped diagnostics."""
        workspace_state = self._workspace_state
        if workspace_state is None:
            raise ManagementError(ErrorInfo("route_unavailable", "Session resume is unavailable."))
        summaries: list[SessionListingEntry] = []
        skipped_count = 0
        try:
            sessions_directory = workspace_state.existing_sessions_directory()
            if sessions_directory is None:
                return SessionListingReport(sessions=(), skipped_count=0)
            paths = tuple(HOST_FILESYSTEM.path_for_io(sessions_directory).glob("*.jsonl"))
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Conversation Sessions could not be listed.")
            ) from error
        for path in paths:
            try:
                session = Session.load(
                    workspace_state,
                    path.stem,
                    partition=SessionStoragePartition.FOREGROUND,
                    now=self._now,
                )
            except (OSError, UnicodeError, ValueError) as error:
                logger.opt(exception=error).warning(
                    "Skipped corrupt or unreadable Conversation Session entry path={} type={}",
                    path,
                    type(error).__name__,
                )
                skipped_count += 1
                continue
            summaries.append(
                SessionListingEntry(
                    id=session.session_id,
                    title=_session_title(session),
                    created_at=session.created_at,
                    updated_at=session.updated_at,
                    message_count=len(session.messages),
                )
            )
        return SessionListingReport(
            sessions=tuple(
                sorted(
                    summaries,
                    key=lambda summary: (
                        summary.updated_at,
                        summary.created_at,
                        summary.id,
                    ),
                    reverse=True,
                )
            ),
            skipped_count=skipped_count,
        )

    async def resume(self, session_id: str) -> ResumeResult:
        """Revalidate and select one Session from the current Workspace."""
        workspace_state = self._workspace_state
        switch_session = self._switch_session
        if workspace_state is None or switch_session is None:
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
            session = Session.load(
                workspace_state,
                session_id,
                partition=SessionStoragePartition.FOREGROUND,
                now=self._now,
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo(
                    "persistence_error",
                    "The selected Conversation Session could not be loaded.",
                )
            ) from error
        switch_session(session)
        return ResumeResult(session_id=session.session_id)


def _session_title(session: Session) -> str:
    title = session.metadata.get("title")
    if not isinstance(title, str):
        raise ValueError("Session title is malformed")
    return title
