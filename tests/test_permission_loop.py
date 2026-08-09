from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

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
from myclaw.session.session import Session
from myclaw.tools.core.edit_file import EditFileTool
from myclaw.tools.core.write_file import WriteFileTool
from myclaw.tools.tool_gateway import ModelToolCall
from tests.fixtures import ScriptedFakeProvider, SingleToolGateway, StreamScript

NOW = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=timezone(timedelta(hours=8)))


@pytest.mark.asyncio
async def test_foreground_mutations_execute_without_a_permission_pause(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    session = Session.create(state, now=lambda: NOW, new_uuid=uuid4)
    target = workspace / "notes.txt"
    target.write_text("before", encoding="utf-8")
    calls = (
        ModelToolCall(
            id="call_write",
            name="write_file",
            arguments='{"path":"created.txt","content":"must not be written"}',
        ),
        ModelToolCall(
            id="call_edit",
            name="edit_file",
            arguments='{"path":"notes.txt","old_text":"before","new_text":"after"}',
        ),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=ModelUsage(
                                input_tokens=8,
                                output_tokens=2,
                                total_tokens=10,
                            ),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Mutations were refused."),
                            usage=ModelUsage(
                                input_tokens=12,
                                output_tokens=3,
                                total_tokens=15,
                            ),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    identity = Workspace.from_path(workspace)
    gateway = SingleToolGateway(
        (WriteFileTool(workspace=identity), EditFileTool(workspace=identity))
    )
    conversation = StreamingConversationPort(
        provider=provider,
        session=session,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway=gateway,
    )

    events = [event async for event in conversation.submit("Change the files.")]

    assert [event.type for event in events] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "must not be written"
    assert target.read_text(encoding="utf-8") == "after"
    tool_messages = [message for message in session.messages if message["role"] == "tool"]
    assert [message["status"] for message in tool_messages] == ["success", "success"]
    follow_up = provider.stream_requests[1]
    assert isinstance(follow_up, ModelRequest)
    model_results = [
        message for message in follow_up.messages if isinstance(message, ToolModelMessage)
    ]
    assert [(message.name, message.content) for message in model_results] == [
        ("write_file", "File written successfully."),
        ("edit_file", "File edited successfully."),
    ]
