import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.events import AgentEvent
from myclaw.agent.workspace import Workspace
from myclaw.config.agent_home import AgentHome
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.records import ToolSessionMessage
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.models import (
    ModelToolCall,
    ToolDefinition,
    ToolExecutionContext,
)
from myclaw.tools.tool_gateway import ToolGateway
from myclaw.utils.json_types import JsonObject
from tests.fixtures import FakeTool, ScriptedFakeProvider, StreamScript

NOW = datetime(2026, 7, 12, 19, 30, tzinfo=timezone(timedelta(hours=8)))


def _usage() -> ModelUsage:
    return ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10)


class BlockingAfterFirstTool:
    _definition = ToolDefinition(
        name="inspect",
        description="Inspect one named item.",
        input_schema={
            "type": "object",
            "properties": {"item": {"type": "string"}},
            "required": ["item"],
            "additionalProperties": False,
        },
    )

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.blocked = asyncio.Event()

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(
        self,
        arguments: JsonObject,
        context: ToolExecutionContext,
    ) -> str:
        del context
        item = arguments["item"]
        assert isinstance(item, str)
        self.calls.append(item)
        if len(self.calls) == 1:
            return "first complete"
        self.blocked.set()
        await asyncio.Event().wait()
        raise AssertionError("blocked Tool should have been cancelled")


@pytest.mark.asyncio
async def test_same_task_cancellation_after_tool_started_prevents_execution(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    session_id = sessions.prepare().id
    calls = tuple(
        ModelToolCall(id=f"call_{item}", name="inspect", arguments={"item": item})
        for item in ("one", "two")
    )
    tool = FakeTool(
        definition=BlockingAfterFirstTool._definition,
        outcomes=("must not execute", "must not execute"),
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=_usage(),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session_id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session_id,
            ),
            tools=(tool,),
        ),
    )

    events = conversation.submit("Inspect both items.")
    observed = [await anext(events), await anext(events)]
    assert [event.type for event in observed] == ["turn_started", "tool_started"]

    await conversation.cancel_active_turn()
    observed.extend([event async for event in events])

    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "turn_cancelled",
    ]
    assert tool.calls == []
    reloaded = await sessions.load(session_id)
    repaired = [message for message in reloaded.messages if isinstance(message, ToolSessionMessage)]
    assert [(message.tool_call_id, message.status) for message in repaired] == [
        ("call_one", "error"),
        ("call_two", "error"),
    ]
    assert all(
        message.content == "Tool call interrupted because the turn was cancelled."
        for message in repaired
    )


@pytest.mark.asyncio
async def test_same_task_cancellation_after_tool_completed_stops_model_continuation(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    session_id = sessions.prepare().id
    tool_call = ModelToolCall(id="call_done", name="inspect", arguments={"item": "done"})
    tool = FakeTool(definition=BlockingAfterFirstTool._definition, outcomes=("complete",))
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=(tool_call,)),
                            usage=_usage(),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="must not complete"),
                            usage=_usage(),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session_id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session_id,
            ),
            tools=(tool,),
        ),
    )

    events = conversation.submit("Inspect one item.")
    observed = [await anext(events), await anext(events), await anext(events)]
    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "tool_completed",
    ]

    await conversation.cancel_active_turn()
    observed.extend([event async for event in events])

    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_cancelled",
    ]
    assert len(provider.stream_requests) == 1
    reloaded = await sessions.load(session_id)
    tool_messages = [
        message for message in reloaded.messages if isinstance(message, ToolSessionMessage)
    ]
    assert [(message.tool_call_id, message.status) for message in tool_messages] == [
        ("call_done", "success")
    ]


@pytest.mark.asyncio
async def test_tool_execution_cancellation_keeps_completed_and_repairs_remaining_calls(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    sessions = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    session_id = sessions.prepare().id
    calls = tuple(
        ModelToolCall(id=f"call_{item}", name="inspect", arguments={"item": item})
        for item in ("first", "second", "third")
    )
    tool = BlockingAfterFirstTool()
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=calls),
                            usage=_usage(),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=sessions,
        session_id=session_id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway=ToolGateway(
            context=ToolExecutionContext(
                lane="foreground",
                workspace=workspace,
                agent_home=agent_home,
                session_id=session_id,
            ),
            tools=(tool,),
        ),
    )

    async def collect_events() -> list[AgentEvent]:
        return [event async for event in conversation.submit("Inspect all three.")]

    collector = asyncio.create_task(collect_events())
    await asyncio.wait_for(tool.blocked.wait(), timeout=0.5)

    await conversation.cancel_active_turn()
    observed = await asyncio.wait_for(collector, timeout=0.5)

    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "tool_started",
        "turn_cancelled",
    ]
    assert tool.calls == ["first", "second"]
    reloaded = await sessions.load(session_id)
    tool_messages = [
        message for message in reloaded.messages if isinstance(message, ToolSessionMessage)
    ]
    assert [(message.tool_call_id, message.status) for message in tool_messages] == [
        ("call_first", "success"),
        ("call_second", "error"),
        ("call_third", "error"),
    ]
    assert tool_messages[0].content == "first complete"
    assert all(
        message.content == "Tool call interrupted because the turn was cancelled."
        for message in tool_messages[1:]
    )
