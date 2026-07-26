from uuid import UUID

import pytest

from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextDelta,
    ToolModelMessage,
    UserModelMessage,
    validate_model_stream_events,
)
from myclaw.tools.models import ModelToolCall
from myclaw.tools.schema import OpenAIToolSchema


def test_model_usage_serializes_to_the_exact_contract_shape() -> None:
    usage = ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144)

    assert usage.to_dict() == {
        "input_tokens": 120,
        "output_tokens": 24,
        "total_tokens": 144,
    }


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens", "total_tokens"),
    [(-1, 0, -1), (1, 2, 4), (True, 0, 1)],
)
def test_model_usage_rejects_values_outside_the_counter_contract(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
) -> None:
    with pytest.raises(ValueError):
        ModelUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )


def test_model_provider_transcript_uses_the_frozen_runtime_shapes() -> None:
    definition: OpenAIToolSchema = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    tool_call = ModelToolCall(
        id="call_123",
        name="read_file",
        arguments='{"path":"CONTEXT.md"}',
    )
    user = UserModelMessage(content="Inspect the project.")
    assistant = AssistantModelMessage(content="I will inspect it.", tool_calls=(tool_call,))
    tool = ToolModelMessage(
        tool_call_id="call_123",
        name="read_file",
        content="project context",
    )
    request = ModelRequest(
        request_id=UUID("550e8400-e29b-41d4-a716-446655440000"),
        route="chat",
        system_prompt="You are MyClaw.",
        messages=(user, assistant, tool),
        tools=(definition,),
        stream=True,
        model="model-id",
        max_output=8192,
        temperature=0.2,
        reasoning_effort=None,
        timeout_seconds=120,
    )
    response = ModelResponse(
        message=assistant,
        usage=ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144),
        finish_reason="tool_calls",
    )

    assert request.to_dict() == {
        "request_id": "550e8400-e29b-41d4-a716-446655440000",
        "route": "chat",
        "system_prompt": "You are MyClaw.",
        "messages": [
            {"role": "user", "content": "Inspect the project."},
            {
                "role": "assistant",
                "content": "I will inspect it.",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "name": "read_file",
                        "arguments": '{"path":"CONTEXT.md"}',
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "name": "read_file",
                "content": "project context",
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a UTF-8 file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            }
        ],
        "stream": True,
        "model": "model-id",
        "max_output": 8192,
        "temperature": 0.2,
        "reasoning_effort": None,
        "timeout_seconds": 120,
    }
    assert response.to_dict() == {
        "message": {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "id": "call_123",
                    "name": "read_file",
                    "arguments": '{"path":"CONTEXT.md"}',
                }
            ],
        },
        "usage": {"input_tokens": 120, "output_tokens": 24, "total_tokens": 144},
        "finish_reason": "tool_calls",
    }
    assert TextDelta(delta="I will").to_dict() == {"type": "text_delta", "delta": "I will"}
    assert ModelCompleted(response=response).to_dict() == {
        "type": "completed",
        "response": response.to_dict(),
    }


def test_model_boundary_rejects_non_uuid4_nonstreaming_chat_and_empty_deltas() -> None:
    def request(request_id: UUID, *, stream: bool) -> ModelRequest:
        return ModelRequest(
            request_id=request_id,
            route="chat",
            system_prompt="You are MyClaw.",
            messages=(),
            tools=(),
            stream=stream,
            model="model-id",
            max_output=8192,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=120,
        )

    with pytest.raises(ValueError, match="UUID4"):
        request(UUID("123e4567-e89b-12d3-a456-426614174000"), stream=True)
    with pytest.raises(ValueError, match="chat requests must stream"):
        request(UUID("550e8400-e29b-41d4-a716-446655440000"), stream=False)
    with pytest.raises(ValueError, match="delta must not be empty"):
        TextDelta(delta="")


def test_model_stream_accepts_ordered_deltas_followed_by_one_completed_event() -> None:
    response = ModelResponse(
        message=AssistantModelMessage(content="Hello", tool_calls=()),
        usage=ModelUsage(input_tokens=10, output_tokens=1, total_tokens=11),
        finish_reason="stop",
    )

    validate_model_stream_events(
        (TextDelta(delta="Hel"), TextDelta(delta="lo"), ModelCompleted(response=response))
    )


def test_model_stream_rejects_missing_duplicate_or_nonfinal_completed_events() -> None:
    response = ModelResponse(
        message=AssistantModelMessage(content="Hello", tool_calls=()),
        usage=ModelUsage(input_tokens=10, output_tokens=1, total_tokens=11),
        finish_reason="stop",
    )
    completed = ModelCompleted(response=response)

    with pytest.raises(ValueError, match="exactly one completed"):
        validate_model_stream_events((TextDelta(delta="Hello"),))
    with pytest.raises(ValueError, match="exactly one completed"):
        validate_model_stream_events((completed, completed))
    with pytest.raises(ValueError, match="completed event must be last"):
        validate_model_stream_events((completed, TextDelta(delta="late")))
