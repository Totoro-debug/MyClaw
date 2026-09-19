from typing import cast

import pytest

from myclaw.errors import (
    MODEL_CONTEXT_OVERFLOW_MESSAGE,
    STABLE_ERROR_CODES,
    TURN_CANCELLED_MESSAGE,
    ErrorCode,
    ErrorInfo,
)
from myclaw.provider.errors import ModelCallError, model_context_overflow_error


def test_turn_cancelled_message_is_the_stable_user_visible_contract() -> None:
    assert TURN_CANCELLED_MESSAGE == "MyClaw 已取消本轮对话。"


def test_model_context_overflow_error_returns_fresh_normalized_failures() -> None:
    first = model_context_overflow_error()
    second = model_context_overflow_error()

    assert isinstance(first, ModelCallError)
    assert isinstance(second, ModelCallError)
    assert first is not second
    assert first.error is not second.error
    assert first.error.code == second.error.code == "model_context_overflow"
    assert first.error.message == second.error.message == MODEL_CONTEXT_OVERFLOW_MESSAGE


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
            "interactive_terminal_required",
            "model_failed",
            "agent_iteration_limit",
            "turn_cancelled",
            "tool_not_found",
            "tool_invalid_arguments",
            "tool_denied",
            "tool_refused",
            "tool_failed",
            "memory_task_running",
            "skill_reload_failed",
        }
    )
    assert MODEL_CONTEXT_OVERFLOW_MESSAGE == (
        "Model request context exceeds the available input budget."
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
