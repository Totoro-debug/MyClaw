import pytest

from myclaw.utils.validation import (
    TokenUsageValidationIssue,
    token_usage_validation_issue,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (
            {"model_calls": 1, "input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            None,
        ),
        ({"model_calls": 1, "input_tokens": 7, "output_tokens": 3}, "fields"),
        (
            {
                "model_calls": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
                "extra": 0,
            },
            "fields",
        ),
        (
            {
                "model_calls": True,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            "values",
        ),
        (
            {
                "model_calls": 1,
                "input_tokens": 7.0,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            "values",
        ),
        (
            {
                "model_calls": -1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            "values",
        ),
        (
            {"model_calls": 1, "input_tokens": 7, "output_tokens": 3, "total_tokens": 11},
            "total",
        ),
    ],
)
def test_token_usage_validation_issue_classifies_shared_contract(
    value: dict[str, object],
    expected: TokenUsageValidationIssue | None,
) -> None:
    assert token_usage_validation_issue(value) == expected
