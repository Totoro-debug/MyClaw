import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import pytest

from myclaw.agent_home import AgentHome
from myclaw.config import ConfigLoader
from myclaw.contracts import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelToolCall,
    ModelUsage,
    PermissionRequestedPayload,
    ToolExecutionContext,
)
from myclaw.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.runtime import prepare_repl_runtime
from myclaw.scheduled_work import CreateScheduledWorkTool, JsonScheduledWorkStore
from myclaw.session_store import JsonlSessionStore
from myclaw.tool_gateway import ToolGateway
from myclaw.workspace import Workspace
from tests.fixtures import ScriptedFakeProvider, StreamScript
from tests.test_config import VALID_CONFIG

NOW = datetime(2026, 7, 12, 20, 0, 0, 123456, tzinfo=timezone(timedelta(hours=8)))


def _usage() -> ModelUsage:
    return ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10)


@pytest.mark.asyncio
async def test_foreground_approval_persists_one_complete_scheduled_work_record(
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
        arguments={
            "title": "Weekly project review",
            "cron": "0 9 * * 1",
            "prompt": prompt,
        },
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
            tools=(CreateScheduledWorkTool(store=store, now=lambda: NOW, new_uuid=uuid4),),
        ),
    )

    events = conversation.submit("Schedule the review.")
    observed = [await anext(events), await anext(events), await anext(events)]
    permission = observed[-1]
    assert permission.type == "permission_requested"
    payload = permission.payload
    assert isinstance(payload, PermissionRequestedPayload)
    assert payload.tool_call_id == tool_call.id
    assert payload.action == "schedule"
    assert payload.resource == "Weekly project review | 0 9 * * 1"
    assert prompt not in str(permission.to_dict())

    await conversation.resolve_permission(payload.request_id, approved=True)
    observed.extend([event async for event in events])

    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "permission_requested",
        "tool_completed",
        "turn_completed",
    ]
    persisted = json.loads((agent_home / "scheduled-work.json").read_text(encoding="utf-8"))
    assert len(persisted) == 1
    record = persisted[0]
    assert set(record) == {
        "id",
        "title",
        "cron",
        "prompt",
        "created_at",
        "enabled",
        "session_id",
    }
    assert UUID(record["id"]).version == 4
    assert record["title"] == "Weekly project review"
    assert record["cron"] == "0 9 * * 1"
    assert record["prompt"] == prompt
    assert record["created_at"] == "2026-07-12T20:00:00.123+08:00"
    assert record["enabled"] is True
    assert record["session_id"] != session_id
    assert len(await store.load()) == 1


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
    assert "create_scheduled_work" in {definition.name for definition in request.tools}


@pytest.mark.asyncio
async def test_invalid_cron_returns_scheduled_work_error_without_writing(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonScheduledWorkStore(home)
    tool_call = ModelToolCall(
        id="call_invalid_schedule",
        name="create_scheduled_work",
        arguments={
            "title": "Invalid schedule",
            "cron": "0 9 * * * *",
            "prompt": "This must never be persisted.",
        },
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-200000-123456_550e8400-e29b-41d4-a716-446655440000",
        ),
        tools=(CreateScheduledWorkTool(store=store, now=lambda: NOW, new_uuid=uuid4),),
    )

    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "scheduled_work_invalid"
    assert not store.path.exists()
    assert await store.load() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lane", "approved", "expected_message"),
    [
        ("foreground", False, "Permission denied by user."),
        (
            "scheduled_work",
            None,
            "Permission confirmation is unavailable in background work.",
        ),
    ],
)
async def test_refused_scheduled_work_creation_does_not_allocate_a_record(
    lane: Literal["foreground", "scheduled_work"],
    approved: bool | None,
    expected_message: str,
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonScheduledWorkStore(home)
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane=lane,
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-200000-123456_550e8400-e29b-41d4-a716-446655440000",
        ),
        tools=(CreateScheduledWorkTool(store=store, now=lambda: NOW, new_uuid=uuid4),),
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_refused_schedule",
            name="create_scheduled_work",
            arguments={
                "title": "Do not persist",
                "cron": "0 9 * * 1",
                "prompt": "This request was not approved.",
            },
        ),
        approved=approved,
    )

    assert result.status == "refused"
    assert result.error is not None
    assert result.error.code == "tool_refused"
    assert result.content == expected_message
    assert not store.path.exists()
    assert await store.load() == ()


@pytest.mark.asyncio
async def test_invalid_existing_record_returns_persistence_error_without_rewrite(
    agent_home: Path,
    workspace: Path,
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
                "enabled": False,
                "session_id": ("20260712-200000-123456_0f8fad5b-d9cb-469f-a165-70867728950e"),
            }
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    store.path.write_text(invalid_content, encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-200000-123456_7c9e6679-7425-40de-944b-e07fc1f90ae7",
        ),
        tools=(CreateScheduledWorkTool(store=store, now=lambda: NOW, new_uuid=uuid4),),
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_after_corrupt_store",
            name="create_scheduled_work",
            arguments={
                "title": "New valid task",
                "cron": "0 10 * * 2",
                "prompt": "Do not overwrite the invalid existing store.",
            },
        ),
        approved=True,
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "persistence_error"
    assert store.path.read_text(encoding="utf-8") == invalid_content


@pytest.mark.asyncio
async def test_atomic_publication_failure_preserves_existing_complete_array(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    session_id = "20260712-200000-123456_550e8400-e29b-41d4-a716-446655440000"
    context = ToolExecutionContext(
        lane="foreground",
        workspace=workspace,
        agent_home=agent_home,
        session_id=session_id,
    )
    store = JsonScheduledWorkStore(home)
    first = await ToolGateway(
        context=context,
        tools=(CreateScheduledWorkTool(store=store, now=lambda: NOW, new_uuid=uuid4),),
    ).execute(
        ModelToolCall(
            id="call_first_schedule",
            name="create_scheduled_work",
            arguments={
                "title": "Existing task",
                "cron": "0 9 * * 1",
                "prompt": "Preserve this task.",
            },
        ),
        approved=True,
    )
    assert first.status == "success"
    official_bytes = store.path.read_bytes()
    attempted_documents: list[str] = []

    def fail_atomic_replace(path: Path, content: str) -> None:
        assert path == store.path
        attempted_documents.append(content)
        raise OSError("private atomic replacement detail")

    failing_store = JsonScheduledWorkStore(home, replace_text=fail_atomic_replace)
    result = await ToolGateway(
        context=context,
        tools=(CreateScheduledWorkTool(store=failing_store, now=lambda: NOW, new_uuid=uuid4),),
    ).execute(
        ModelToolCall(
            id="call_failed_schedule",
            name="create_scheduled_work",
            arguments={
                "title": "Unpublished task",
                "cron": "0 10 * * 2",
                "prompt": "This task must not appear in the official file.",
            },
        ),
        approved=True,
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "persistence_error"
    assert "private atomic replacement detail" not in result.content
    assert len(attempted_documents) == 1
    assert len(json.loads(attempted_documents[0])) == 2
    assert store.path.read_bytes() == official_bytes
    assert len(await store.load()) == 1
