"""Non-persisted Management Port values."""

from dataclasses import dataclass

from myclaw.session.records import CumulativeUsage
from myclaw.utils.validation import require_nonnegative_int, require_nonnegative_number


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
