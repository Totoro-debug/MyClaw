from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.tools.core.schedule import ScheduleTool
from myclaw.tools.tool_gateway import ModelToolCall
from tests.fixtures import SingleToolGateway, write_schedule_state

JOB_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def _store(
    workspace: Path,
    agent_home: Path,
    *persisted_jobs: ScheduleJob,
) -> WorkspaceScheduleStore:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    if persisted_jobs:
        write_schedule_state(state, *persisted_jobs)
    return WorkspaceScheduleStore(state)


def _gateway(tool: ScheduleTool) -> SingleToolGateway:
    return SingleToolGateway((tool,))


def test_schema_exposes_the_three_actions_and_optional_schedule_branches(
    workspace: Path,
    agent_home: Path,
) -> None:
    schema = ScheduleTool(store=_store(workspace, agent_home)).to_schema()["function"]["parameters"]

    assert schema["required"] == ["action"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {
        "action",
        "message",
        "every_seconds",
        "cron_expr",
        "timezone",
        "at_time",
        "job_id",
    }


@pytest.mark.asyncio
async def test_add_uses_the_common_gateway_without_confirmation_and_ignores_lower_priority_fields(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(
        ScheduleTool(
            store=store,
            now=lambda: NOW,
            new_uuid=lambda: JOB_UUID,
        )
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_add",
            name="schedule",
            arguments=json.dumps(
                {
                    "action": "add",
                    "message": "  Run it  ",
                    "every_seconds": "60",
                    "cron_expr": {"invalid": True},
                    "timezone": ["invalid"],
                    "at_time": {"invalid": True},
                    "job_id": "ignored",
                }
            ),
        )
    )

    assert result.status == "success"
    assert result.confirmation is None
    assert json.loads(result.content) == {
        "action": "add",
        "job": {
            "job_id": str(JOB_UUID),
            "message": "Run it",
            "schedule": {"type": "every", "every_seconds": 60},
        },
    }
    jobs = await store.snapshot()
    assert len(jobs) == 1
    assert jobs[0].message == "Run it"


@pytest.mark.asyncio
async def test_add_normalizes_cron_timezone_and_at_time_without_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    uuids = iter((JOB_UUID, UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")))
    gateway = _gateway(
        ScheduleTool(
            store=store,
            now=lambda: NOW,
            new_uuid=lambda: next(uuids),
        )
    )

    cron = await gateway.call(
        ModelToolCall(
            id="call_cron",
            name="schedule",
            arguments='{"action":"add","message":"Cron","cron_expr":" 0 9 * * 1 ","timezone":"Asia/Shanghai"}',
        )
    )
    at = await gateway.call(
        ModelToolCall(
            id="call_at",
            name="schedule",
            arguments='{"action":"add","message":"Past","at_time":"2020-01-02T03:04:05.123456Z"}',
        )
    )

    assert cron.status == "success"
    assert json.loads(cron.content)["job"]["schedule"] == {
        "type": "cron",
        "cron_expr": "0 9 * * 1",
        "timezone": "Asia/Shanghai",
    }
    assert at.status == "success"
    assert json.loads(at.content)["job"]["schedule"] == {
        "type": "at",
        "at_time": "2020-01-02T03:04:05.123+00:00",
    }


@pytest.mark.asyncio
async def test_cron_accepts_valid_iana_timezone_alias(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID))

    result = await gateway.call(
        ModelToolCall(
            id="call_alias_cron",
            name="schedule",
            arguments=(
                '{"action":"add","message":"Alias","cron_expr":"0 9 * * *","timezone":"US/Eastern"}'
            ),
        )
    )

    assert result.status == "success"
    assert json.loads(result.content)["job"]["schedule"]["timezone"] == "US/Eastern"


@pytest.mark.asyncio
async def test_cron_defaults_to_utc_and_invalid_at_time_is_rejected(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID))

    cron = await gateway.call(
        ModelToolCall(
            id="call_utc_cron",
            name="schedule",
            arguments='{"action":"add","message":"UTC","cron_expr":"0 9 * * 1"}',
        )
    )
    naive = await gateway.call(
        ModelToolCall(
            id="call_naive_at",
            name="schedule",
            arguments='{"action":"add","message":"Naive","at_time":"2026-08-07T13:00:00"}',
        )
    )

    assert cron.status == "success"
    assert json.loads(cron.content)["job"]["schedule"]["timezone"] == "UTC"
    assert naive.status == "error"
    assert naive.content == "Invalid arguments for schedule."


@pytest.mark.parametrize(
    "schedule_arguments",
    [
        {"cron_expr": "@daily"},
        {"cron_expr": "0 9 * * *", "timezone": "Not/A_Timezone"},
    ],
)
@pytest.mark.asyncio
async def test_add_rejects_invalid_cron_inputs_without_mutating_the_store(
    workspace: Path,
    agent_home: Path,
    schedule_arguments: dict[str, object],
) -> None:
    store = _store(workspace, agent_home)
    gateway = _gateway(ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID))

    result = await gateway.call(
        ModelToolCall(
            id="call_invalid_cron",
            name="schedule",
            arguments=json.dumps(
                {"action": "add", "message": "Invalid", **schedule_arguments},
                separators=(",", ":"),
            ),
        )
    )

    assert result.status == "error"
    assert result.content == "Invalid arguments for schedule."
    assert await store.snapshot() == ()


@pytest.mark.asyncio
async def test_list_returns_only_public_jobs_in_creation_then_id_order(
    workspace: Path,
    agent_home: Path,
) -> None:
    first_id = JOB_UUID
    second_id = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
    earlier_id = UUID("9ba7b810-9dad-41d1-80b4-00c04fd430c8")
    first = ScheduleJob(
        job_id=str(first_id),
        message="First",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=20,
        updated_at_ms=20,
    )
    second = ScheduleJob(
        job_id=str(second_id),
        message="Second",
        schedule=JobSchedule.at("2026-08-07T14:00:00.000+00:00"),
        created_at_ms=20,
        updated_at_ms=20,
    )
    earlier = ScheduleJob(
        job_id=str(earlier_id),
        message="Earlier",
        schedule=JobSchedule.at("2026-08-07T12:30:00.000+00:00"),
        created_at_ms=10,
        updated_at_ms=10,
    )
    hidden = ScheduleJob(
        job_id="8ba7b810-9dad-41d1-80b4-00c04fd430c8",
        source="system",
        message="Hidden",
        schedule=JobSchedule.at("2026-08-07T15:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    store = _store(workspace, agent_home, hidden)
    await store.add_user_job(second)
    await store.add_user_job(first)
    await store.add_user_job(earlier)

    result = await _gateway(
        ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    ).call(ModelToolCall(id="call_list", name="schedule", arguments='{"action":"list"}'))

    assert result.status == "success"
    assert result.confirmation is None
    assert json.loads(result.content) == {
        "jobs": [
            {
                "job_id": str(earlier_id),
                "message": "Earlier",
                "schedule": {"type": "at", "at_time": "2026-08-07T12:30:00.000+00:00"},
            },
            {
                "job_id": str(first_id),
                "message": "First",
                "schedule": {"type": "at", "at_time": "2026-08-07T13:00:00.000+00:00"},
            },
            {
                "job_id": str(second_id),
                "message": "Second",
                "schedule": {"type": "at", "at_time": "2026-08-07T14:00:00.000+00:00"},
            },
        ]
    }


@pytest.mark.asyncio
async def test_remove_requires_canonical_uuid_and_hides_unknown_or_system_jobs(
    workspace: Path,
    agent_home: Path,
) -> None:
    public = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Remove me",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    hidden = ScheduleJob(
        job_id="9ba7b810-9dad-41d1-80b4-00c04fd430c8",
        source="system",
        message="Internal",
        schedule=JobSchedule.at("2026-08-07T14:00:00.000+00:00"),
        created_at_ms=2,
        updated_at_ms=2,
    )
    store = _store(workspace, agent_home, hidden)
    await store.add_user_job(public)
    gateway = _gateway(ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID))

    invalid = await gateway.call(
        ModelToolCall(
            id="call_invalid_remove",
            name="schedule",
            arguments='{"action":"remove","job_id":"550E8400-E29B-41D4-A716-446655440000"}',
        )
    )
    unknown = await gateway.call(
        ModelToolCall(
            id="call_unknown_remove",
            name="schedule",
            arguments='{"action":"remove","job_id":"8ba7b810-9dad-41d1-80b4-00c04fd430c8"}',
        )
    )
    system = await gateway.call(
        ModelToolCall(
            id="call_system_remove",
            name="schedule",
            arguments='{"action":"remove","job_id":"9ba7b810-9dad-41d1-80b4-00c04fd430c8"}',
        )
    )
    removed = await gateway.call(
        ModelToolCall(
            id="call_remove",
            name="schedule",
            arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert invalid.status == "error"
    assert invalid.content == "Invalid arguments for schedule."
    assert unknown.status == "error"
    assert unknown.content == "Schedule Job was not found."
    assert system.status == "error"
    assert system.content == "Schedule Job was not found."
    assert removed.status == "success"
    assert removed.confirmation is None
    assert json.loads(removed.content)["job"]["job_id"] == str(JOB_UUID)
    assert await store.snapshot() == (hidden,)


@pytest.mark.asyncio
async def test_scheduled_agent_rejects_add_but_permits_list_and_remove(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = _store(workspace, agent_home)
    existing = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Existing",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(existing)
    gateway = _gateway(
        ScheduleTool(
            store=store,
            scheduled_agent=True,
            now=lambda: NOW,
            new_uuid=lambda: UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e"),
        )
    )

    invalid_add = await gateway.call(
        ModelToolCall(
            id="call_invalid_scheduled_add",
            name="schedule",
            arguments='{"action":"add","every_seconds":60}',
        )
    )
    add = await gateway.call(
        ModelToolCall(
            id="call_scheduled_add",
            name="schedule",
            arguments='{"action":"add","message":"Recursive","every_seconds":60}',
        )
    )
    listed = await gateway.call(
        ModelToolCall(id="call_scheduled_list", name="schedule", arguments='{"action":"list"}')
    )
    removed = await gateway.call(
        ModelToolCall(
            id="call_scheduled_remove",
            name="schedule",
            arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert invalid_add.status == "error"
    assert invalid_add.confirmation is None
    assert add.status == "refused"
    assert add.confirmation is None
    assert "scheduled Agent context" in add.content
    assert listed.status == "success"
    assert removed.status == "success"
    assert await store.snapshot() == ()
