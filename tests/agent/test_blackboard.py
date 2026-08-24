from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence

import pytest

from myclaw.agent.blackboard import (
    Blackboard,
    FramingResult,
    TaskFramer,
    TaskFramingModelRouter,
    decode_blackboard,
    encode_blackboard,
)
from myclaw.agent.prompts import blackboard_prompt
from myclaw.errors import ErrorInfo
from myclaw.provider.errors import ModelCallError
from myclaw.provider.model_router import ModelRouter
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelMessages,
    ModelResponse,
    ModelRoute,
    ModelUsage,
)
from myclaw.tools.base import OpenAIToolSchema


def _model_router_satisfies_task_framing_protocol(
    router: ModelRouter,
) -> TaskFramingModelRouter:
    return router


class _FakeRouter:
    def __init__(
        self,
        response: ModelResponse | None = None,
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.response = response
        self.failure = failure
        self.calls: list[dict[str, object]] = []

    async def complete(
        self,
        route: ModelRoute,
        *,
        messages: ModelMessages,
        tools: Sequence[OpenAIToolSchema],
    ) -> ModelResponse:
        self.calls.append(
            {
                "route": route,
                "messages": messages,
                "tools": tools,
            }
        )
        if self.failure is not None:
            raise self.failure
        assert self.response is not None
        return self.response


def _response(content: str, *, input_tokens: int = 7, output_tokens: int = 3) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
        ),
        finish_reason="stop",
    )


def _decision(
    action: str,
    goal: str | None,
    completion_boundary: str | None,
) -> str:
    return json.dumps(
        {
            "action": action,
            "goal": goal,
            "completion_boundary": completion_boundary,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _usage() -> dict[str, int]:
    return {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.parametrize(
    ("status", "blackboard", "usage_delta"),
    [
        ("resolved", None, _usage()),
        (
            "resolved",
            Blackboard(goal="Goal", completion_boundary="Boundary"),
            _usage(),
        ),
        ("invalid_response", None, _usage()),
        ("model_failed", None, None),
    ],
)
def test_framing_result_accepts_the_complete_legal_state_table(
    status: str,
    blackboard: Blackboard | None,
    usage_delta: dict[str, int] | None,
) -> None:
    result = FramingResult(
        blackboard=blackboard,
        usage_delta=usage_delta,
        status=status,  # type: ignore[arg-type]
    )

    assert result.status == status
    assert result.blackboard is blackboard
    assert result.usage_delta == usage_delta


@pytest.mark.parametrize(
    ("blackboard", "usage_delta", "status", "error_type", "match"),
    [
        (None, _usage(), 1, TypeError, "status must be a string"),
        (None, _usage(), "unknown", ValueError, "status is invalid"),
        (object(), _usage(), "resolved", TypeError, "Blackboard or None"),
        (None, [], "resolved", TypeError, "dictionary or None"),
        (None, None, "resolved", ValueError, "must include usage"),
        (
            Blackboard(goal="Goal", completion_boundary="Boundary"),
            _usage(),
            "invalid_response",
            ValueError,
            "cannot include a Blackboard",
        ),
        (None, None, "invalid_response", ValueError, "must include usage"),
        (
            Blackboard(goal="Goal", completion_boundary="Boundary"),
            None,
            "model_failed",
            ValueError,
            "cannot include a Blackboard or usage",
        ),
        (None, _usage(), "model_failed", ValueError, "cannot include a Blackboard or usage"),
        (
            None,
            {"model_calls": 1},
            "resolved",
            ValueError,
            "exactly four fields",
        ),
        (
            None,
            {
                "model_calls": True,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            },
            "resolved",
            ValueError,
            "nonnegative integers",
        ),
        (
            None,
            {
                "model_calls": 1,
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 11,
            },
            "resolved",
            ValueError,
            "must equal input plus output",
        ),
    ],
)
def test_framing_result_rejects_every_invalid_state_combination(
    blackboard: object,
    usage_delta: object,
    status: object,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        FramingResult(
            blackboard=blackboard,  # type: ignore[arg-type]
            usage_delta=usage_delta,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
        )


def test_blackboard_canonicalizes_and_round_trips_without_truncation() -> None:
    goal = "  保留全部目标 " + ("x" * 4096) + "  "
    boundary = '  完成边界 {with braces} and \\"quotes\\"  '
    value = Blackboard(goal=goal, completion_boundary=boundary)

    assert value.goal == goal.strip()
    assert value.completion_boundary == boundary.strip()
    encoded = encode_blackboard(value)
    assert encoded == {
        "goal": goal.strip(),
        "completion_boundary": boundary.strip(),
    }
    assert decode_blackboard(encoded) == value


@pytest.mark.parametrize(
    "value",
    [
        None,
        "not an object",
        [],
        {},
        {"goal": "goal"},
        {"completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": "boundary", "extra": "reject"},
        {"goal": 1, "completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": False},
        {"goal": "", "completion_boundary": "boundary"},
        {"goal": "   ", "completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": ""},
        {"goal": "goal", "completion_boundary": "\t\n"},
    ],
)
def test_decode_blackboard_treats_every_malformed_optional_shape_as_empty(
    value: object,
) -> None:
    assert decode_blackboard(value) is None


def test_encode_blackboard_has_one_strict_public_shape() -> None:
    assert encode_blackboard(None) is None
    with pytest.raises(TypeError):
        encode_blackboard({"goal": "goal", "completion_boundary": "boundary"})  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_frame_sends_only_the_three_complete_json_inputs_and_no_tools() -> None:
    previous = Blackboard(goal="Old goal", completion_boundary="Old boundary")
    last_assistant_content = 'Assistant {answer} with "quotes" and C:\\path'
    current_user_input = "继续: 保留换行\n以及非 ASCII 内容。"
    router = _FakeRouter(_response(_decision("replace", "New goal", "New boundary")))

    result = await TaskFramer(router).frame(
        previous=previous,
        last_assistant_content=last_assistant_content,
        current_user_input=current_user_input,
    )

    assert result.blackboard == Blackboard(goal="New goal", completion_boundary="New boundary")
    assert result.status == "resolved"
    assert result.usage_delta == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert len(router.calls) == 1
    call = router.calls[0]
    assert call["route"] == "chat"
    assert call["tools"] == ()
    assert call["messages"] == [
        {"role": "system", "content": blackboard_prompt()},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "previous_blackboard": encode_blackboard(previous),
                    "last_assistant_content": last_assistant_content,
                    "current_user_input": current_user_input,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


@pytest.mark.asyncio
async def test_frame_preserves_empty_last_assistant_content_as_json_string() -> None:
    router = _FakeRouter(_response(_decision("clear", None, None)))

    result = await TaskFramer(router).frame(
        previous=None,
        last_assistant_content="",
        current_user_input="cancel",
    )

    assert result.blackboard is None
    assert result.status == "resolved"
    call = router.calls[0]
    messages = call["messages"]
    assert isinstance(messages, Sequence)
    user_message = messages[1]
    assert user_message["content"] == (
        '{"previous_blackboard":null,"last_assistant_content":"","current_user_input":"cancel"}'
    )


@pytest.mark.parametrize(
    ("action", "previous", "expected"),
    [
        (
            "keep",
            Blackboard(goal="Goal", completion_boundary="Boundary"),
            Blackboard(goal="Goal", completion_boundary="Boundary"),
        ),
        ("keep", None, None),
        (
            "replace",
            Blackboard(goal="Old", completion_boundary="Old boundary"),
            Blackboard(goal="New", completion_boundary="New boundary"),
        ),
        ("replace", None, Blackboard(goal="New", completion_boundary="New boundary")),
        ("clear", Blackboard(goal="Goal", completion_boundary="Boundary"), None),
        ("clear", None, None),
    ],
)
@pytest.mark.asyncio
async def test_frame_reduces_keep_replace_and_clear_with_or_without_previous(
    action: str,
    previous: Blackboard | None,
    expected: Blackboard | None,
) -> None:
    goal = None if action != "replace" else "New"
    boundary = None if action != "replace" else "New boundary"
    router = _FakeRouter(_response(_decision(action, goal, boundary)))

    result = await TaskFramer(router).frame(
        previous=previous,
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "resolved"
    assert result.blackboard == expected


@pytest.mark.parametrize(
    "response_content",
    [
        _decision("replace", "Goal", "Boundary"),
        "```json\n" + _decision("replace", "Goal", "Boundary") + "\n```",
        "The model decided: " + _decision("replace", "Goal", "Boundary") + " Done.",
    ],
)
@pytest.mark.asyncio
async def test_frame_accepts_raw_fenced_and_prose_surrounded_json(
    response_content: str,
) -> None:
    router = _FakeRouter(_response(response_content))

    result = await TaskFramer(router).frame(
        previous=None,
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "resolved"
    assert result.blackboard == Blackboard(goal="Goal", completion_boundary="Boundary")


@pytest.mark.asyncio
async def test_balanced_scan_handles_quoted_braces_escaped_quotes_and_backslashes() -> None:
    response_content = (
        "Provider note: "
        + json.dumps(
            {
                "action": "replace",
                "goal": 'Keep {braces} and "quotes"',
                "completion_boundary": r"Use C:\path and finish {it}",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + " end note."
    )
    router = _FakeRouter(_response(response_content))

    result = await TaskFramer(router).frame(
        previous=None,
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "resolved"
    assert result.blackboard == Blackboard(
        goal='Keep {braces} and "quotes"',
        completion_boundary=r"Use C:\path and finish {it}",
    )


@pytest.mark.asyncio
async def test_balanced_scan_ignores_an_unmatched_quote_before_the_first_object() -> None:
    router = _FakeRouter(
        _response(
            'Provider says "decision: {"action":"clear","goal":null,"completion_boundary":null}'
        )
    )

    result = await TaskFramer(router).frame(
        previous=Blackboard(goal="Old", completion_boundary="Old boundary"),
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "resolved"
    assert result.blackboard is None
    assert result.usage_delta == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.asyncio
async def test_balanced_scan_does_not_skip_an_earlier_braced_prose_fragment() -> None:
    router = _FakeRouter(
        _response(
            "Provider note {not JSON} then "
            '{"action":"clear","goal":null,"completion_boundary":null}'
        )
    )

    result = await TaskFramer(router).frame(
        previous=Blackboard(goal="Old", completion_boundary="Old boundary"),
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "invalid_response"
    assert result.blackboard is None
    assert result.usage_delta == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.parametrize(
    "response_content",
    [
        "not JSON",
        "[]",
        "{'action':'clear','goal':null,'completion_boundary':null}",
        '{"action":"clear","goal":null,"completion_boundary":null,}',
        '{"action":null,"goal":null,"completion_boundary":null}',
        '{"action":1,"goal":null,"completion_boundary":null}',
        '{"action":"rename","goal":null,"completion_boundary":null}',
        '{"action":"keep","goal":"goal","completion_boundary":null}',
        '{"action":"keep","goal":null,"completion_boundary":"boundary"}',
        '{"action":"clear","goal":"goal","completion_boundary":null}',
        '{"action":"clear","goal":null,"completion_boundary":"boundary"}',
        '{"action":"replace","goal":"","completion_boundary":"boundary"}',
        '{"action":"replace","goal":"   ","completion_boundary":"boundary"}',
        '{"action":"replace","goal":null,"completion_boundary":"boundary"}',
        '{"action":"replace","goal":1,"completion_boundary":"boundary"}',
        '{"action":"replace","goal":"goal","completion_boundary":null}',
        '{"action":"replace","goal":"goal","completion_boundary":1}',
        '{"action":"replace","goal":"goal","completion_boundary":""}',
        '{"action":"replace","goal":"goal","completion_boundary":"   "}',
        '{"action":"replace","goal":"goal","completion_boundary":NaN}',
        '{"action":"replace","goal":"goal","completion_boundary":Infinity}',
        '{"action":"replace","goal":"goal","completion_boundary":-Infinity}',
        '{"action":"replace","goal":"first","goal":"second","completion_boundary":"boundary"}',
        '{"action":"replace","goal":"goal"}',
        '{"action":"replace","goal":"goal","completion_boundary":"boundary","extra":true}',
        '{"task":"replace","goal":"goal","completion_boundary":"boundary"}',
        '{"action":"replace","goal":"goal"} prose {"action":"clear","goal":null,"completion_boundary":null}',
        "```json\n" + _decision("replace", "Goal", "Boundary") + " trailing prose\n```",
    ],
)
@pytest.mark.asyncio
async def test_frame_rejects_repairs_guesses_and_ambiguous_or_invalid_decisions(
    response_content: str,
) -> None:
    router = _FakeRouter(_response(response_content))

    result = await TaskFramer(router).frame(
        previous=Blackboard(goal="Old", completion_boundary="Old boundary"),
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "invalid_response"
    assert result.blackboard is None
    assert result.usage_delta == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.parametrize(
    ("first", "trailing", "expected"),
    [
        (
            _decision("clear", None, None),
            " prose " + _decision("replace", "Ignored", "Ignored boundary"),
            None,
        ),
        (
            _decision("replace", "New goal", "New boundary"),
            ' prose {"ignored":true}',
            Blackboard(goal="New goal", completion_boundary="New boundary"),
        ),
        (
            _decision("clear", None, None),
            " prose {unclosed",
            None,
        ),
    ],
)
@pytest.mark.asyncio
async def test_frame_uses_first_balanced_object_and_preserves_usage(
    first: str,
    trailing: str,
    expected: Blackboard | None,
) -> None:
    router = _FakeRouter(_response(f"prefix {first}{trailing}"))

    result = await TaskFramer(router).frame(
        previous=Blackboard(goal="Old", completion_boundary="Old boundary"),
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "resolved"
    assert result.blackboard == expected
    assert result.usage_delta == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.asyncio
async def test_frame_rejects_a_later_valid_object_when_the_first_is_invalid() -> None:
    response_content = (
        '{"action":"replace","goal":"","completion_boundary":"invalid"} prose '
        '{"action":"clear","goal":null,"completion_boundary":null}'
    )
    router = _FakeRouter(_response(response_content))

    result = await TaskFramer(router).frame(
        previous=Blackboard(goal="Old", completion_boundary="Old boundary"),
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "invalid_response"
    assert result.blackboard is None
    assert result.usage_delta == {
        "model_calls": 1,
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }


@pytest.mark.asyncio
async def test_model_call_error_is_fail_open_without_usage() -> None:
    failure = ModelCallError(ErrorInfo("model_failed", "The framing model failed."))
    router = _FakeRouter(failure=failure)

    result = await TaskFramer(router).frame(
        previous=Blackboard(goal="Old", completion_boundary="Old boundary"),
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "model_failed"
    assert result.blackboard is None
    assert result.usage_delta is None


@pytest.mark.asyncio
async def test_cancelled_model_call_propagates_unchanged() -> None:
    router = _FakeRouter(failure=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await TaskFramer(router).frame(
            previous=None,
            last_assistant_content="Last answer",
            current_user_input="Current input",
        )


@pytest.mark.asyncio
async def test_ordinary_model_exception_is_fail_open_without_usage() -> None:
    router = _FakeRouter(failure=RuntimeError("provider implementation failed"))

    result = await TaskFramer(router).frame(
        previous=Blackboard(goal="Old", completion_boundary="Old boundary"),
        last_assistant_content="Last answer",
        current_user_input="Current input",
    )

    assert result.status == "model_failed"
    assert result.blackboard is None
    assert result.usage_delta is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("previous", "last_assistant_content", "current_user_input"),
    [
        (object(), "Last answer", "Current input"),
        (None, None, "Current input"),
        (None, 1, "Current input"),
        (None, "Last answer", 1),
    ],
)
async def test_frame_rejects_wrong_public_input_types_without_coercion(
    previous: object,
    last_assistant_content: object,
    current_user_input: object,
) -> None:
    router = _FakeRouter(_response(_decision("clear", None, None)))

    with pytest.raises(TypeError):
        await TaskFramer(router).frame(
            previous=previous,  # type: ignore[arg-type]
            last_assistant_content=last_assistant_content,  # type: ignore[arg-type]
            current_user_input=current_user_input,  # type: ignore[arg-type]
        )

    assert router.calls == []


def test_blackboard_prompt_is_versioned_and_restricts_the_domain() -> None:
    prompt = blackboard_prompt()

    assert "keep" in prompt
    assert "replace" in prompt
    assert "clear" in prompt
    assert "JSON" in prompt
    assert "completion_boundary" in prompt
    assert "Do not answer the user's task" in prompt
