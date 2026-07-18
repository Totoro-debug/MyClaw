"""Non-persisted values returned through the Management Port."""

from dataclasses import dataclass
from pathlib import Path

from myclaw.contracts.common import (
    require_nonnegative_int,
    require_nonnegative_number,
    require_session_id,
)
from myclaw.contracts.errors import ErrorInfo
from myclaw.contracts.sessions import CumulativeUsage


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
class ConfigView:
    """A configuration path, redacted content, and optional safe parse error."""

    path: Path
    redacted_content: str
    error: ErrorInfo | None


@dataclass(frozen=True, slots=True)
class ResumeResult:
    """Identity of the Conversation Session selected by a successful resume."""

    session_id: str

    def __post_init__(self) -> None:
        require_session_id(self.session_id)


@dataclass(frozen=True, slots=True)
class MemoryTaskResult:
    """Observable summary returned by a manual Memory Task run."""

    status: str
    processed_count: int
    memory_updated: bool
    cursor: int
    error: ErrorInfo | None = None

    def __post_init__(self) -> None:
        require_nonnegative_int(self.processed_count, field="processed_count")
        require_nonnegative_int(self.cursor, field="cursor")
        if not isinstance(self.memory_updated, bool):
            msg = "memory_updated must be a boolean"
            raise ValueError(msg)
