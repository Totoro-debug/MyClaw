"""Concrete read-only views exposed through the Management Port."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from loguru import logger

from myclaw import __version__
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader, ConfigView
from myclaw.errors import ErrorInfo
from myclaw.memory.dream import DreamResult
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.skills.catalog import SkillMetadata
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from myclaw.utils.time import format_rfc3339_milliseconds
from myclaw.utils.validation import require_nonnegative_int, require_nonnegative_number


class _MemoryReader(Protocol):
    async def read_long_term(self) -> str: ...


class _DreamRunner(Protocol):
    async def run(self) -> DreamResult: ...


class _StatusProjectionLoop(Protocol):
    def runtime_status_input(self) -> "RuntimeStatusInput": ...


class _ManagementAgentLoop(_StatusProjectionLoop, Protocol):
    def reload_skill(self) -> tuple[SkillMetadata, ...]: ...


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
class RuntimeStatusInput:
    """Immutable status projection and exact next-request text fragments."""

    system_prompt: str
    retained_messages: tuple[str, ...]
    tool_definitions: tuple[str, ...]
    runtime_context: str
    session_id: str = ""
    session_title: str = ""
    session_message_count: int = 0
    last_consolidated: int = 0
    cumulative_usage: tuple[tuple[str, int], ...] = ()
    chat_model: str = ""
    context_window: int = 0
    generation_started_at: float | None = None


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


class ManagementError(Exception):
    """A safe persistence error suitable for a Management Command."""

    def __init__(self, error: ErrorInfo) -> None:
        self.error = error
        super().__init__(error.message)


class FatalManagementError(ManagementError):
    """A safe Management error that must terminate the owning application."""


class ManagementViewService:
    """Read global configuration and dynamically selected runtime-owned views."""

    def __init__(
        self,
        agent_home: AgentHome,
        *,
        current_agent_loop: Callable[[], _ManagementAgentLoop],
        workspace_state: WorkspaceState,
        replace_agent_loop: Callable[[str, bool], Awaitable[None]],
        prepare_session_resume: Callable[[str], Awaitable[None]],
        memory_manager: _MemoryReader,
        dream: _DreamRunner,
        schedule_status: Callable[[], dict[str, object]],
        now: Callable[[], datetime],
        monotonic: Callable[[], float],
    ) -> None:
        self._config = ConfigLoader(agent_home)
        self._current_agent_loop = current_agent_loop
        self._workspace_state = workspace_state
        self._replace_agent_loop = replace_agent_loop
        self._prepare_session_resume = prepare_session_resume
        self._now = now
        self._monotonic = monotonic
        self._schedule_status = schedule_status
        self._memory_reader = memory_manager
        self._dream = dream
        self._aborted = False

    async def reload_skill(self) -> tuple[SkillMetadata, ...]:
        """Reload the current Agent Loop Skill state and return published metadata."""
        try:
            self._ensure_active()
            current_agent_loop = self._current_agent_loop()
            metadata = current_agent_loop.reload_skill()
            if not isinstance(metadata, tuple) or not all(
                isinstance(item, SkillMetadata) for item in metadata
            ):
                raise TypeError("Agent Loop Skill metadata is malformed")
            return metadata
        except Exception as error:
            logger.warning(
                "Skill reload failed type={}",
                type(error).__name__,
            )
            raise ManagementError(
                ErrorInfo("skill_reload_failed", "Skill reload failed.")
            ) from error

    def deactivate(self) -> None:
        """Reject new Management work after its Runtime Generation is detached."""
        self._aborted = True

    def _ensure_active(self) -> None:
        if self._aborted:
            raise ManagementError(
                ErrorInfo("route_unavailable", "Runtime Generation is no longer active.")
            )

    def _ensure_current_generation(self) -> None:
        self._current_agent_loop()

    async def config_view(self) -> ConfigView:
        """Return complete redacted User Configuration content."""
        self._ensure_active()
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
        self._ensure_active()
        self._ensure_current_generation()
        try:
            return await self._memory_reader.read_long_term()
        except ManagementError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Long-term Memory could not be read.")
            ) from error

    async def dream(self) -> DreamResult:
        """Run one foreground Memory Task and return its safe summary."""
        self._ensure_active()
        self._ensure_current_generation()
        return await self._dream.run()

    async def status(self) -> RuntimeStatus:
        """Return all required runtime and current-session status fields."""
        self._ensure_active()
        try:
            projection = self._current_agent_loop().runtime_status_input()
            estimated = estimate_input_tokens(projection)
            if projection.context_window <= 0:
                raise ValueError("Runtime status context window must be positive")
            started_at = projection.generation_started_at
            uptime = 0 if started_at is None else max(0, int(self._monotonic() - started_at))
            return RuntimeStatus(
                version=__version__,
                chat_model=projection.chat_model,
                uptime_seconds=uptime,
                estimated_input_tokens=estimated,
                context_window=projection.context_window,
                context_used_percent=estimated / projection.context_window * 100,
                session_message_count=projection.session_message_count,
                last_consolidated=projection.last_consolidated,
                cumulative_usage=dict(projection.cumulative_usage),
                schedule=dict(self._schedule_status()),
            )
        except ManagementError:
            raise
        except (OSError, UnicodeError, ValueError) as error:
            raise ManagementError(
                ErrorInfo("persistence_error", "Runtime status could not be read.")
            ) from error

    async def resumable_listing(self) -> SessionListingReport:
        """Return one atomic Session picker result including skipped diagnostics."""
        self._ensure_active()
        self._ensure_current_generation()
        return await self._resumable_listing()

    async def _resumable_listing(self) -> SessionListingReport:
        workspace_state = self._workspace_state
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

    async def resume(self, session_id: str, *, force: bool = False) -> ResumeResult:
        """Revalidate and select one Session from the current Workspace."""
        self._ensure_active()
        self._ensure_current_generation()
        await self._prepare_session_resume(session_id)
        listing = await self._resumable_listing()
        sessions = listing.sessions
        if session_id not in {summary.id for summary in sessions}:
            raise ManagementError(
                ErrorInfo(
                    "model_invalid_request",
                    "The selected Conversation Session is not resumable.",
                )
            )
        await self._replace_agent_loop(session_id, force)
        return ResumeResult(session_id=session_id)


def _session_title(session: Session) -> str:
    title = session.metadata.get("title")
    if not isinstance(title, str):
        raise ValueError("Session title is malformed")
    return title
