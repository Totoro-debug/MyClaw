import asyncio
from collections import deque
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.contracts import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    PermissionRequestedPayload,
    ToolDefinition,
    ToolExecutionContext,
    ToolModelMessage,
)
from myclaw.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.repl import run_repl
from myclaw.session_store import JsonlSessionStore
from myclaw.tool_gateway import ToolGateway
from myclaw.workspace import Workspace
from tests.fixtures import FakeClock, FakeTool, ScriptedFakeProvider, StreamScript

SESSION_ID = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"
LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
TURN_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
USER_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
REQUEST_UUID = UUID("16fd2706-8baf-433b-82eb-8c7fada847da")
ASSISTANT_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")
PERMISSION_UUID = UUID("a8098c1a-f86e-4f33-8a28-25f602f8e603")
TOOL_UUID = UUID("9b2c3a42-1d2e-4a1e-a827-61f36dc54713")
FINAL_REQUEST_UUID = UUID("67e55044-10b1-426f-9247-bb680e5fe0c8")
FINAL_ASSISTANT_UUID = UUID("11111111-1111-4111-8111-111111111111")


class ScriptedPermissionInput:
    def __init__(self, values: Iterable[str | None]) -> None:
        self._values = deque(values)

    async def read(self) -> str | None:
        return self._values.popleft()


class RecordingPermissionWriter:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.finished_turns = 0

    async def write_delta(self, delta: str) -> None:
        raise AssertionError(f"Unexpected text delta: {delta}")

    async def finish_turn(self) -> None:
        self.finished_turns += 1

    async def write_line(self, content: str) -> None:
        self.lines.append(content)


@pytest.mark.asyncio
async def test_background_permission_ask_is_refused_without_executing(
    agent_home: Path,
    workspace: Path,
) -> None:
    tool = FakeTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write one Workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        outcomes=("must not execute",),
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="scheduled_work",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        tools=(tool,),
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_write",
            name="write_file",
            arguments={"path": "notes.txt", "content": "private content"},
        )
    )

    assert result.status == "refused"
    assert result.error is not None
    assert result.error.code == "tool_refused"
    assert result.content == "Permission confirmation is unavailable in background work."
    assert tool.calls == []


@pytest.mark.asyncio
async def test_foreground_permission_accept_resumes_the_same_tool_call_without_extra_history(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    private_content = "private content that must not enter the permission event"
    tool_call = ModelToolCall(
        id="call_write",
        name="write_file",
        arguments={"path": "notes.txt", "content": private_content},
    )
    tool = FakeTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write one Workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        outcomes=("Wrote notes.txt",),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(
                                content="I will update the notes.",
                                tool_calls=(tool_call,),
                            ),
                            usage=ModelUsage(input_tokens=10, output_tokens=3, total_tokens=13),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="The notes are updated."),
                            usage=ModelUsage(input_tokens=16, output_tokens=5, total_tokens=21),
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
                PERMISSION_UUID,
                TOOL_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            ),
            tools=(tool,),
        ),
    )

    events = conversation.submit("Update my notes.")
    observed = [await anext(events), await anext(events), await anext(events)]

    permission = observed[-1]
    assert permission.type == "permission_requested"
    payload = permission.payload
    assert isinstance(payload, PermissionRequestedPayload)
    assert payload.to_dict() == {
        "request_id": str(PERMISSION_UUID),
        "tool_call_id": "call_write",
        "tool_name": "write_file",
        "action": "write",
        "resource": "notes.txt",
        "risk_summary": "This changes a Workspace file.",
    }
    assert private_content not in str(permission.to_dict())

    await conversation.resolve_permission(payload.request_id, approved=True)
    observed.extend([event async for event in events])

    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "permission_requested",
        "tool_completed",
        "turn_completed",
    ]
    assert len(tool.calls) == 1
    assert tool.calls[0].arguments == tool_call.arguments
    reloaded = await store.load(session.id)
    assert [message.role for message in reloaded.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    second_request_result = second_request.messages[-1]
    assert isinstance(second_request_result, ToolModelMessage)
    assert second_request_result.tool_call_id == tool_call.id
    assert second_request_result.content == "Wrote notes.txt"


@pytest.mark.asyncio
async def test_foreground_permission_refusal_persists_a_refused_tool_result(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool_call = ModelToolCall(
        id="call_edit",
        name="edit_file",
        arguments={
            "path": "notes.txt",
            "old_text": "before",
            "new_text": "private replacement",
        },
    )
    tool = FakeTool(
        definition=ToolDefinition(
            name="edit_file",
            description="Edit one Workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_text": {"type": "string"},
                    "new_text": {"type": "string"},
                },
                "required": ["path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        ),
        outcomes=("must not execute",),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(tool_calls=(tool_call,), content=""),
                            usage=ModelUsage(input_tokens=9, output_tokens=2, total_tokens=11),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="The edit was refused."),
                            usage=ModelUsage(input_tokens=14, output_tokens=5, total_tokens=19),
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
                PERMISSION_UUID,
                TOOL_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            ),
            tools=(tool,),
        ),
    )

    events = conversation.submit("Edit my notes.")
    observed = [await anext(events), await anext(events), await anext(events)]

    permission = observed[-1]
    assert permission.type == "permission_requested"
    payload = permission.payload
    assert isinstance(payload, PermissionRequestedPayload)
    assert payload.action == "edit"
    assert payload.resource == "notes.txt"
    assert "private replacement" not in str(permission.to_dict())

    await conversation.resolve_permission(payload.request_id, approved=False)
    observed.extend([event async for event in events])

    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "permission_requested",
        "tool_completed",
        "turn_completed",
    ]
    assert tool.calls == []
    reloaded = await store.load(session.id)
    assert [message.role for message in reloaded.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    refused = reloaded.messages[2]
    assert refused.role == "tool"
    assert refused.tool_call_id == tool_call.id
    assert refused.status == "refused"
    assert refused.content == "Permission denied by user."
    assert refused.error is not None
    assert refused.error.code == "tool_refused"
    second_request = provider.stream_requests[1]
    assert isinstance(second_request, ModelRequest)
    second_request_result = second_request.messages[-1]
    assert isinstance(second_request_result, ToolModelMessage)
    assert second_request_result.tool_call_id == tool_call.id
    assert second_request_result.content == "Permission denied by user."


@pytest.mark.asyncio
async def test_cancelling_the_active_turn_releases_its_permission_wait(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool_call = ModelToolCall(
        id="call_write",
        name="write_file",
        arguments={"path": "notes.txt", "content": "private content"},
    )
    tool = FakeTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write one Workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        outcomes=("must not execute",),
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
            (TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, PERMISSION_UUID)
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            ),
            tools=(tool,),
        ),
    )

    events = conversation.submit("Update my notes.")
    observed = [await anext(events), await anext(events), await anext(events)]
    permission = observed[-1]
    assert permission.type == "permission_requested"
    payload = permission.payload
    assert isinstance(payload, PermissionRequestedPayload)

    await conversation.cancel_active_turn()

    with pytest.raises(RuntimeError, match="not pending"):
        await conversation.resolve_permission(payload.request_id, approved=True)
    observed.extend([event async for event in events])
    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "permission_requested",
        "turn_cancelled",
    ]
    assert tool.calls == []


@pytest.mark.asyncio
async def test_closing_turn_iterator_releases_its_permission_wait(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    tool_call = ModelToolCall(
        id="call_write",
        name="write_file",
        arguments={"path": "notes.txt", "content": "private content"},
    )
    tool = FakeTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write one Workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        outcomes=("must not execute",),
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
            (TURN_UUID, USER_UUID, REQUEST_UUID, ASSISTANT_UUID, PERMISSION_UUID)
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            ),
            tools=(tool,),
        ),
    )

    events = conversation.submit("Update my notes.")
    permission = [await anext(events), await anext(events), await anext(events)][-1]
    assert permission.type == "permission_requested"
    payload = permission.payload
    assert isinstance(payload, PermissionRequestedPayload)

    await events.aclose()

    with pytest.raises(RuntimeError, match="not pending"):
        await conversation.resolve_permission(payload.request_id, approved=True)
    assert tool.calls == []


@pytest.mark.asyncio
async def test_repl_answers_a_permission_request_before_continuing_the_turn(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(NOW)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    session = store.prepare()
    private_content = "private content"
    tool_call = ModelToolCall(
        id="call_write",
        name="write_file",
        arguments={"path": "notes.txt", "content": private_content},
    )
    tool = FakeTool(
        definition=ToolDefinition(
            name="write_file",
            description="Write one Workspace file.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        outcomes=("Wrote notes.txt",),
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
                            message=AssistantModelMessage(content="The notes are updated."),
                            usage=ModelUsage(input_tokens=14, output_tokens=5, total_tokens=19),
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
                PERMISSION_UUID,
                TOOL_UUID,
                FINAL_REQUEST_UUID,
                FINAL_ASSISTANT_UUID,
            )
        ).__next__,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session.id,
            ),
            tools=(tool,),
        ),
    )
    writer = RecordingPermissionWriter()

    await asyncio.wait_for(
        run_repl(
            conversation=conversation,
            input_reader=ScriptedPermissionInput(("Update my notes.", "yes", None)),
            writer=writer,
        ),
        timeout=0.5,
    )

    assert len(tool.calls) == 1
    assert writer.finished_turns == 1
    assert len(writer.lines) == 1
    assert "notes.txt" in writer.lines[0]
    assert private_content not in writer.lines[0]
    assert len(provider.stream_requests) == 2
