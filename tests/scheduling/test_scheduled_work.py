import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from myclaw.schedule.scheduled_work import (
    CreateScheduledWorkTool,
    JsonScheduledWorkStore,
    ScheduledWorkPersistenceError,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.records import ToolSessionMessage
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.models import ModelToolCall
from myclaw.tools.tool_gateway import ToolGateway
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript

NOW = datetime(2026, 7, 12, 20, 0, 0, 123456, tzinfo=timezone(timedelta(hours=8)))


def _usage() -> ModelUsage:
    return ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10)


def test_create_scheduled_work_exports_exact_schema_and_zero_retries() -> None:
    tool = CreateScheduledWorkTool()

    assert tool.to_schema() == {
        "type": "function",
        "function": {
            "name": "create_scheduled_work",
            "description": "Create recurring work with a five-field cron schedule.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short task title.",
                        "minLength": 1,
                        "maxLength": 120,
                    },
                    "cron": {
                        "type": "string",
                        "description": "Five-field cron schedule.",
                        "minLength": 1,
                    },
                    "prompt": {
                        "type": "string",
                        "description": "Task prompt.",
                        "minLength": 1,
                        "maxLength": 20000,
                    },
                },
                "required": ["title", "cron", "prompt"],
            },
        },
    }
    assert tool.max_retries == 0


@pytest.mark.asyncio
async def test_foreground_scheduled_work_creation_is_refused_without_confirmation(
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
    prompt = "Use disabled shell and web tools later if the task needs them."
    tool_call = ModelToolCall(
        id="call_schedule",
        name="create_scheduled_work",
        arguments=json.dumps(
            {
                "title": "Weekly project review",
                "cron": "0 9 * * 1",
                "prompt": prompt,
            }
        ),
    )
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
                            message=AssistantModelMessage(content="Scheduled."),
                            usage=_usage(),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    store = JsonScheduledWorkStore(home)
    gateway = ToolGateway()
    gateway.register_tools((CreateScheduledWorkTool(),))
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
        tool_gateway=gateway,
    )

    observed = [event async for event in conversation.submit("Schedule the review.")]
    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    assert store.load() == ()
    session = await sessions.load(session_id)
    refused = session.messages[2]
    assert isinstance(refused, ToolSessionMessage)
    assert refused.status == "refused"
    assert refused.content == (
        "Scheduled Work creation is unavailable because confirmation is not implemented."
    )


@pytest.mark.asyncio
async def test_runtime_chat_catalog_exposes_scheduled_work_creation(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Ready."),
                            usage=_usage(),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    events = [event async for event in runtime.conversation.submit("What can you schedule?")]

    assert events[-1].type == "turn_completed"
    request = provider.stream_requests[0]
    assert isinstance(request, ModelRequest)
    assert "create_scheduled_work" in {
        schema["function"]["name"] for schema in request.tools
    }


def test_invalid_existing_record_returns_persistence_error_without_rewrite(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonScheduledWorkStore(home)
    invalid_content = json.dumps(
        [
            {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "title": "Invalid persisted task",
                "cron": "0 9 * * 1",
                "prompt": "This existing record is disabled.",
                "created_at": "2026-07-12T20:00:00.123+08:00",
                "enabled": "false",
                "session_id": ("20260712-200000-123456_0f8fad5b-d9cb-469f-a165-70867728950e"),
            }
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    store.path.write_text(invalid_content, encoding="utf-8")

    with pytest.raises(ScheduledWorkPersistenceError):
        store.load()

    assert store.path.read_text(encoding="utf-8") == invalid_content
