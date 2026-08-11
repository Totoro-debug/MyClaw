from typing import cast

import pytest

from myclaw.errors import (
    STABLE_ERROR_CODES,
    ErrorCode,
    ErrorInfo,
)


def test_error_info_uses_the_frozen_structure_and_code_vocabulary() -> None:
    assert STABLE_ERROR_CODES == frozenset(
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
    error = ErrorInfo(
        code="provider_rate_limited",
        message="Provider rate limit reached.",
        retryable=True,
        retry_after_seconds=1.5,
    )

    assert error.to_dict() == {
        "code": "provider_rate_limited",
        "message": "Provider rate limit reached.",
        "retryable": True,
        "retry_after_seconds": 1.5,
    }
    with pytest.raises(ValueError, match="stable error code"):
        ErrorInfo(code=cast(ErrorCode, "new_unaccepted_code"), message="Not accepted.")
