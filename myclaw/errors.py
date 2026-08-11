"""Stable user-visible error values."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal

type ErrorCode = Literal[
    "config_missing",
    "config_parse_error",
    "config_invalid",
    "persistence_error",
    "schedule_state_error",
    "route_unavailable",
    "provider_auth_error",
    "provider_rate_limited",
    "provider_timeout",
    "provider_unavailable",
    "model_invalid_request",
    "model_context_overflow",
    "memory_context_too_large",
    "interactive_terminal_required",
    "model_failed",
    "turn_cancelled",
    "tool_not_found",
    "tool_invalid_arguments",
    "tool_denied",
    "tool_refused",
    "tool_failed",
    "memory_task_running",
]

STABLE_ERROR_CODES: frozenset[str] = frozenset(
    {
        "config_missing",
        "config_parse_error",
        "config_invalid",
        "persistence_error",
        "schedule_state_error",
        "route_unavailable",
        "provider_auth_error",
        "provider_rate_limited",
        "provider_timeout",
        "provider_unavailable",
        "model_invalid_request",
        "model_context_overflow",
        "memory_context_too_large",
        "interactive_terminal_required",
        "model_failed",
        "turn_cancelled",
        "tool_not_found",
        "tool_invalid_arguments",
        "tool_denied",
        "tool_refused",
        "tool_failed",
        "memory_task_running",
    }
)


@dataclass(frozen=True, slots=True)
class ErrorInfo:
    """A safe runtime error that may cross a Port boundary."""

    code: ErrorCode
    message: str
    retryable: bool = False
    retry_after_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.code not in STABLE_ERROR_CODES:
            msg = "code must be a stable error code"
            raise ValueError(msg)
        if not self.message:
            msg = "message must not be empty"
            raise ValueError(msg)
        if not isinstance(self.retryable, bool):
            msg = "retryable must be a boolean"
            raise ValueError(msg)
        if self.retry_after_seconds is not None and (
            isinstance(self.retry_after_seconds, bool)
            or not isinstance(self.retry_after_seconds, (int, float))
            or self.retry_after_seconds < 0
            or not isfinite(self.retry_after_seconds)
        ):
            msg = "retry_after_seconds must be a finite nonnegative number"
            raise ValueError(msg)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "retry_after_seconds": self.retry_after_seconds,
        }
