import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.events import (
    ToolCompletedPayload,
    ToolStartedPayload,
)
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    ToolModelMessage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.records import (
    AssistantSessionMessage,
    ToolSessionMessage,
)
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.base import BaseTool
from myclaw.tools.models import ModelToolCall
from myclaw.tools.tool_gateway import ToolGateway
from tests.fixtures import (
    FakeClock,
    FakeTool,
    ScriptedFakeProvider,
    StreamScript,
    validate_agent_event_sequence,
)

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)

SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
USER_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
REQUEST_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
ASSISTANT_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
TOOL_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
TOOL_TWO_UUID = UUID("11111111-1111-4111-8111-111111111111")
FINAL_REQUEST_UUID = UUID("a8098c1a-f86e-4f33-8a28-25f602f8e603")
FINAL_ASSISTANT_UUID = UUID("67e55044-10b1-426f-9247-bb680e5fe0c8")


def _gateway(*tools: BaseTool) -> ToolGateway:
    gateway = ToolGateway()
    gateway.register_tools(tuple(tools))
    return gateway


@pytest.mark.asyncio
async def test_unknown_long_tool_is_normalized_and_the_turn_continues_without_event_leaks(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    long_unknown_name = "unknown_" + "x" * 400
    secret_argument = "complete-private-argument"
    tool_call = ModelToolCall(
        id="call_unknown",
        name=long_unknown_name,
        arguments='{"query":"complete-private-argument"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will try the requested capability.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=10, output_tokens=4, total_tokens=14),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="That capability is unavailable."
                            ),
                            usage=ModelUsage(input_tokens=18, output_tokens=5, total_tokens=23),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=ToolGateway(),
    )

    events = [event async for event in conversation.submit("Use the requested capability.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    validate_agent_event_sequence(events)
    started = events[1].payload
    completed = events[2].payload
    assert isinstance(started, ToolStartedPayload)
    assert isinstance(completed, ToolCompletedPayload)
    assert len(started.summary) <= 240
    assert len(completed.summary) <= 240
    assert completed.status == "error"
    assert all(secret_argument not in repr(event) for event in events[1:3])

    reloaded = await store.load(session.id)
    assert [message.role for message in reloaded.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assistant = reloaded.messages[1]
    assert isinstance(assistant, AssistantSessionMessage)
    assert assistant.content == "I will try the requested capability."
    assert assistant.tool_calls == (tool_call,)
    tool_message = reloaded.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.status == "error"
    assert tool_message.content == "The requested tool is not available."
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    normalized = second_request.messages[-1]
    assert isinstance(normalized, ToolModelMessage)
    assert normalized.content == "The requested tool is not available."
    assert reloaded.metadata.cumulative_usage.to_dict() == {
        "model_calls": 2,
        "input_tokens": 28,
        "output_tokens": 9,
        "total_tokens": 37,
    }


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected_before_the_tool_boundary_and_returned_to_model(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool = FakeTool(
        name="send_notice",
        description="Send one notice through an external boundary.",
        required=("message",),
        outcomes=("must not execute",),
    )
    secret_argument = "complete-private-invalid-argument"
    tool_call = ModelToolCall(
        id="call_invalid",
        name="send_notice",
        arguments=json.dumps({"message": 42, "secret": secret_argument}),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(tool_calls=(tool_call,), content=""),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="The notice was not sent."),
                            usage=ModelUsage(input_tokens=12, output_tokens=4, total_tokens=16),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )

    events = [event async for event in conversation.submit("Send my notice.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    validate_agent_event_sequence(events)
    assert tool.calls == []
    assert all(secret_argument not in repr(event) for event in events[1:3])
    reloaded = await store.load(session.id)
    tool_message = reloaded.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.status == "error"
    assert tool_message.content == "Invalid arguments for send_notice."
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    normalized = second_request.messages[-1]
    assert isinstance(normalized, ToolModelMessage)
    assert normalized.content == "Invalid arguments for send_notice."


@pytest.mark.asyncio
async def test_tool_exception_executes_once_and_becomes_a_safe_result_before_model_continues(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    raw_exception_detail = "private-service-response-body"
    tool = FakeTool(
        name="send_notice",
        description="Send one notice through an external boundary.",
        required=("message",),
        outcomes=(RuntimeError(raw_exception_detail),),
    )
    tool_call = ModelToolCall(
        id="call_failed",
        name="send_notice",
        arguments='{"message":"hello"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will send the notice.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=9, output_tokens=3, total_tokens=12),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="The notice could not be sent."),
                            usage=ModelUsage(input_tokens=15, output_tokens=5, total_tokens=20),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )

    events = [event async for event in conversation.submit("Send the notice.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    validate_agent_event_sequence(events)
    assert len(tool.calls) == 1
    assert all(raw_exception_detail not in repr(event) for event in events)
    reloaded = await store.load(session.id)
    assert raw_exception_detail not in str([message.to_dict() for message in reloaded.messages])
    tool_message = reloaded.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.status == "error"
    assert tool_message.content == "send_notice could not complete the request."
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    normalized = second_request.messages[-1]
    assert isinstance(normalized, ToolModelMessage)
    assert normalized.content == "send_notice could not complete the request."


@pytest.mark.asyncio
async def test_multiple_tool_calls_keep_mixed_results_ordered_and_recoverable(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    success_result = "private raw success result"
    raw_failure = "private raw failure detail"
    success_tool = FakeTool(
        name="lookup_notice",
        description="Look up a notice through an external boundary.",
        required=("notice_id",),
        outcomes=(success_result,),
    )
    failed_tool = FakeTool(
        name="send_notice",
        description="Send one notice through an external boundary.",
        required=("message",),
        outcomes=(RuntimeError(raw_failure),),
    )
    lookup_call = ModelToolCall(
        id="call_lookup",
        name="lookup_notice",
        arguments='{"notice_id":"private-argument-one"}',
    )
    send_call = ModelToolCall(
        id="call_send",
        name="send_notice",
        arguments='{"message":"private-argument-two"}',
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will look up and send the notice.",
                                tool_calls=(lookup_call, send_call),
                            ),
                            usage=ModelUsage(input_tokens=11, output_tokens=5, total_tokens=16),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="The lookup succeeded, but sending failed."
                            ),
                            usage=ModelUsage(input_tokens=22, output_tokens=7, total_tokens=29),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                TOOL_TWO_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(success_tool, failed_tool),
    )

    events = [event async for event in conversation.submit("Process both notices.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    validate_agent_event_sequence(events)
    completed_payloads = (events[2].payload, events[4].payload)
    assert all(isinstance(payload, ToolCompletedPayload) for payload in completed_payloads)
    assert [
        payload.status
        for payload in completed_payloads
        if isinstance(payload, ToolCompletedPayload)
    ] == [
        "success",
        "error",
    ]
    activity = repr(events[1:5])
    for private_value in (
        "private-argument-one",
        "private-argument-two",
        success_result,
        raw_failure,
    ):
        assert private_value not in activity
    assert len(success_tool.calls) == 1
    assert len(failed_tool.calls) == 1

    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    assert [message.to_dict() for message in second_request.messages[-3:]] == [
        {
            "role": "assistant",
            "content": "I will look up and send the notice.",
            "tool_calls": [lookup_call.to_dict(), send_call.to_dict()],
        },
        {
            "role": "tool",
            "tool_call_id": "call_lookup",
            "name": "lookup_notice",
            "content": success_result,
        },
        {
            "role": "tool",
            "tool_call_id": "call_send",
            "name": "send_notice",
            "content": "send_notice could not complete the request.",
        },
    ]
    reloaded = await store.load(session.id)
    assert [message.role for message in reloaded.messages] == [
        "user",
        "assistant",
        "tool",
        "tool",
        "assistant",
    ]
    assistant = reloaded.messages[1]
    assert isinstance(assistant, AssistantSessionMessage)
    assert assistant.content == "I will look up and send the notice."
    assert assistant.tool_calls == (lookup_call, send_call)
    first_result = reloaded.messages[2]
    second_result = reloaded.messages[3]
    assert isinstance(first_result, ToolSessionMessage)
    assert isinstance(second_result, ToolSessionMessage)
    assert first_result.status == "success"
    assert second_result.status == "error"
    assert second_result.content == "send_notice could not complete the request."
    assert reloaded.metadata.cumulative_usage.to_dict() == {
        "model_calls": 2,
        "input_tokens": 33,
        "output_tokens": 12,
        "total_tokens": 45,
    }


@pytest.mark.asyncio
async def test_json_schema_format_is_enforced_before_tool_boundary_and_turn_continues(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        workspace_state=WorkspaceState(Workspace.from_path(workspace)),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    invalid_email = "not-an-email-private-value"
    tool = FakeTool(
        name="send_email",
        description="Send an email through an external boundary.",
        required=("recipient",),
        outcomes=("must not execute",),
    )
    tool_call = ModelToolCall(
        id="call_invalid_format",
        name="send_email",
        arguments=json.dumps({"recipient": invalid_email}),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(tool_calls=(tool_call,), content=""),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="The recipient address is invalid."
                            ),
                            usage=ModelUsage(input_tokens=12, output_tokens=5, total_tokens=17),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=clock.now,
        new_uuid=iter(
            (
                TURN_UUID,
                USER_UUID,
                REQUEST_UUID,
                ASSISTANT_UUID,
                TOOL_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=_gateway(tool),
    )

    events = [event async for event in conversation.submit("Send the email.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    validate_agent_event_sequence(events)
    assert tool.calls == []
    assert invalid_email not in repr(events[1:3])
    reloaded = await store.load(session.id)
    tool_message = reloaded.messages[2]
    assert isinstance(tool_message, ToolSessionMessage)
    assert tool_message.status == "error"
    assert tool_message.content == "Invalid arguments for send_email."
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    normalized = second_request.messages[-1]
    assert isinstance(normalized, ToolModelMessage)
    assert normalized.content == "Invalid arguments for send_email."
