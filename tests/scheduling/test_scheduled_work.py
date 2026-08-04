import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from myclaw.schedule.records import ScheduledWork
from myclaw.schedule.scheduled_work import (
    CreateScheduledWorkTool,
    ScheduledWorkPersistenceError,
    WorkspaceJsonScheduledWorkStore,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session import Session
from myclaw.tools.models import ModelToolCall
from myclaw.tools.tool_gateway import ToolGateway
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript

NOW = datetime(2026, 7, 12, 20, 0, 0, 123456, tzinfo=timezone(timedelta(hours=8)))


def _state(path: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(path))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _scheduled_record(*, title: str, task_id: str, session_id: str) -> ScheduledWork:
    return ScheduledWork(
        id=task_id,
        title=title,
        cron="0 9 * * 1",
        prompt=f"Run {title}.",
        created_at=NOW.replace(microsecond=123000),
        enabled=True,
        session_id=session_id,
    )


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
    state = _state(workspace)
    session = Session.create(state, now=lambda: NOW, new_uuid=uuid4)
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
    store = WorkspaceJsonScheduledWorkStore(state)
    gateway = ToolGateway()
    gateway.register_tools((CreateScheduledWorkTool(),))
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

    observed = [event async for event in conversation.submit("Schedule the review.")]
    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "tool_completed",
        "turn_completed",
    ]
    assert store.load() == ()
    refused = session.messages[2]
    assert refused["status"] == "refused"
    assert refused["content"] == (
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
    assert "create_scheduled_work" in {schema["function"]["name"] for schema in request.tools}


def test_invalid_existing_record_returns_persistence_error_without_rewrite(
    workspace: Path,
) -> None:
    store = WorkspaceJsonScheduledWorkStore(_state(workspace))
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


def test_workspace_scheduled_work_is_lazy_and_isolated(workspace: Path) -> None:
    first_workspace = workspace / "first"
    second_workspace = workspace / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()
    first = WorkspaceJsonScheduledWorkStore(_state(first_workspace))
    second = WorkspaceJsonScheduledWorkStore(_state(second_workspace))

    assert first.load() == ()
    assert second.load() == ()
    assert not first.path.exists()
    assert not second.path.exists()

    first_record = _scheduled_record(
        title="First Workspace task",
        task_id="11111111-1111-4111-8111-111111111111",
        session_id="20260712-200000-123000_22222222-2222-4222-8222-222222222222",
    )
    second_record = _scheduled_record(
        title="Second Workspace task",
        task_id="33333333-3333-4333-8333-333333333333",
        session_id="20260712-200000-123000_44444444-4444-4444-8444-444444444444",
    )
    for store, record in ((first, first_record), (second, second_record)):
        store.path.write_text(
            json.dumps([record.to_dict()], ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )

    assert first.load() == (first_record,)
    assert second.load() == (second_record,)


@pytest.mark.parametrize("unsafe_kind", ("directory", "hardlink"))
def test_workspace_scheduled_work_rejects_unsafe_files_without_modifying_targets(
    workspace: Path,
    unsafe_kind: str,
) -> None:
    store = WorkspaceJsonScheduledWorkStore(_state(workspace))
    outside = workspace.parent / f"outside-{unsafe_kind}.json"
    outside_bytes = b"outside Scheduled Work must not be parsed or modified"

    if unsafe_kind == "directory":
        store.path.mkdir()
    else:
        outside.write_bytes(outside_bytes)
        store.path.hardlink_to(outside)

    with pytest.raises(ScheduledWorkPersistenceError):
        store.load()

    if unsafe_kind == "hardlink":
        assert outside.read_bytes() == outside_bytes


def test_invalid_workspace_scheduled_work_is_not_rewritten(workspace: Path) -> None:
    store = WorkspaceJsonScheduledWorkStore(_state(workspace))
    invalid_bytes = b"invalid Workspace Scheduled Work\xff"
    store.path.write_bytes(invalid_bytes)

    with pytest.raises(ScheduledWorkPersistenceError):
        store.load()

    assert store.path.read_bytes() == invalid_bytes
