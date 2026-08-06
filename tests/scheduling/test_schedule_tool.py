from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

import myclaw.schedule.tool as schedule_tool_module
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.schedule.tool import ScheduleTool
from myclaw.tools.confirmation import ConfirmationChannel
from myclaw.tools.models import ModelToolCall
from myclaw.tools.tool_gateway import ToolGateway

JOB_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
OTHER_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
TURN_UUID = UUID("7ba7b810-9dad-41d1-80b4-00c04fd430c8")
CONFIRMATION_UUID = UUID("8ba7b810-9dad-41d1-80b4-00c04fd430c8")
NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    return state


def _gateway(
    tool: ScheduleTool,
    *,
    channel: ConfirmationChannel | None = None,
) -> ToolGateway:
    gateway = ToolGateway(
        turn_id=TURN_UUID,
        confirmation=channel,
        new_uuid=lambda: CONFIRMATION_UUID,
    )
    gateway.register_tools((tool,))
    return gateway


def test_schedule_schema_keeps_branch_fields_optional_and_normalized_message_unbounded(
    workspace: Path,
    agent_home: Path,
) -> None:
    schema = ScheduleTool(
        store=WorkspaceScheduleStore(_state(workspace, agent_home)),
        now=lambda: NOW,
        new_uuid=lambda: JOB_UUID,
    ).to_schema()["function"]["parameters"]

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
    message_schema = properties["message"]
    assert isinstance(message_schema, dict)
    assert "maxLength" not in message_schema


@pytest.mark.asyncio
async def test_add_priority_ignores_invalid_lower_priority_fields_before_coercion(
    workspace: Path,
    agent_home: Path,
) -> None:
    channel = ConfirmationChannel(TURN_UUID)
    task = asyncio.create_task(
        _gateway(
            ScheduleTool(
                store=WorkspaceScheduleStore(_state(workspace, agent_home)),
                now=lambda: NOW,
                new_uuid=lambda: JOB_UUID,
            ),
            channel=channel,
        ).call(
            ModelToolCall(
                id="call_priority",
                name="schedule",
                arguments=json.dumps(
                    {
                        "action": "add",
                        "message": "Run it",
                        "every_seconds": "60",
                        "cron_expr": {"invalid": True},
                        "timezone": ["invalid"],
                        "at_time": {"invalid": True},
                    }
                ),
            )
        )
    )

    request = await channel.next_request()
    assert request.details == {
        "action": "add",
        "message": "Run it",
        "schedule": {"type": "every", "every_seconds": 60},
    }
    channel.respond_to_confirmation(request.confirmation_id, "declined")
    assert (await task).status == "refused"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        ('{"action":"LIST"}', "Invalid arguments for schedule."),
        (
            '{"action":"add","message":"Run it","at_time":"2026-08-07T13:00:00"}',
            "Invalid arguments for schedule.",
        ),
    ],
)
async def test_action_and_at_time_require_exact_normalized_inputs(
    workspace: Path,
    agent_home: Path,
    arguments: str,
    expected: str,
) -> None:
    result = await _gateway(
        ScheduleTool(
            store=WorkspaceScheduleStore(_state(workspace, agent_home)),
            now=lambda: NOW,
            new_uuid=lambda: JOB_UUID,
        )
    ).call(ModelToolCall(id="call_invalid", name="schedule", arguments=arguments))

    assert result.status == "error"
    assert result.content == expected


@pytest.mark.asyncio
async def test_add_at_normalizes_time_and_allocates_job_only_after_approval(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    allocated: list[UUID] = []

    def new_job_uuid() -> UUID:
        allocated.append(JOB_UUID)
        return JOB_UUID

    tool = ScheduleTool(store=store, now=lambda: NOW, new_uuid=new_job_uuid)
    channel = ConfirmationChannel(TURN_UUID)
    task = asyncio.create_task(
        _gateway(tool, channel=channel).call(
            ModelToolCall(
                id="call_add",
                name="schedule",
                arguments=json.dumps(
                    {
                        "action": "add",
                        "message": "  Check the inbox.  ",
                        "at_time": "2026-08-07T13:00:00Z",
                        "job_id": "not-used",
                    }
                ),
            )
        )
    )

    request = await channel.next_request()
    assert allocated == []
    assert request.summary == "Add Schedule Job"
    assert request.details == {
        "action": "add",
        "message": "Check the inbox.",
        "schedule": {"type": "at", "at_time": "2026-08-07T13:00:00.000+00:00"},
    }
    channel.respond_to_confirmation(request.confirmation_id, "approved")

    result = await task

    assert allocated == [JOB_UUID]
    assert result.status == "success"
    assert result.content == (
        '{"action":"add","job":{"job_id":"550e8400-e29b-41d4-a716-446655440000",'
        '"message":"Check the inbox.","schedule":{"type":"at",'
        '"at_time":"2026-08-07T13:00:00.000+00:00"}}}'
    )
    assert (await store.snapshot())[0].session_id == f"schedule_{JOB_UUID}"


@pytest.mark.asyncio
async def test_add_accepts_boundary_whitespace_when_normalized_message_fits(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    message = f" {'x' * 20_000} "
    tool = ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    channel = ConfirmationChannel(TURN_UUID)
    task = asyncio.create_task(
        _gateway(tool, channel=channel).call(
            ModelToolCall(
                id="call_boundary_message",
                name="schedule",
                arguments=json.dumps(
                    {
                        "action": "add",
                        "message": message,
                        "at_time": "2026-08-07T13:00:00+00:00",
                    }
                ),
            )
        )
    )

    request = await channel.next_request()
    assert request.details["message"] == "x" * 20_000
    channel.respond_to_confirmation(request.confirmation_id, "declined")
    assert (await task).status == "refused"


@pytest.mark.asyncio
async def test_add_executes_the_cron_timezone_shown_for_confirmation(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resolved_timezones = iter(("UTC", "Asia/Shanghai"))
    monkeypatch.setattr(
        schedule_tool_module,
        "get_localzone_name",
        lambda: next(resolved_timezones),
    )
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    channel = ConfirmationChannel(TURN_UUID)
    task = asyncio.create_task(
        _gateway(
            ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID),
            channel=channel,
        ).call(
            ModelToolCall(
                id="call_default_timezone",
                name="schedule",
                arguments='{"action":"add","message":"Run it","cron_expr":"0 9 * * 1"}',
            )
        )
    )

    request = await channel.next_request()
    assert request.details["schedule"] == {
        "type": "cron",
        "cron_expr": "0 9 * * 1",
        "timezone": "UTC",
    }
    channel.respond_to_confirmation(request.confirmation_id, "approved")

    result = await task

    assert result.status == "success"
    assert '"timezone":"UTC"' in result.content
    assert (await store.snapshot())[0].schedule.timezone == "UTC"
    assert next(resolved_timezones) == "Asia/Shanghai"


@pytest.mark.asyncio
async def test_declined_add_does_not_allocate_or_write(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    allocated: list[UUID] = []

    def allocate() -> UUID:
        allocated.append(JOB_UUID)
        return JOB_UUID

    tool = ScheduleTool(
        store=store,
        now=lambda: NOW,
        new_uuid=allocate,
    )
    channel = ConfirmationChannel(TURN_UUID)
    task = asyncio.create_task(
        _gateway(tool, channel=channel).call(
            ModelToolCall(
                id="call_decline",
                name="schedule",
                arguments='{"action":"add","message":"Run it","at_time":"2026-08-07T13:00:00+00:00"}',
            )
        )
    )

    request = await channel.next_request()
    channel.respond_to_confirmation(request.confirmation_id, "declined")

    result = await task

    assert result.status == "refused"
    assert allocated == []
    assert await store.snapshot() == ()
    assert not store.path.exists()


@pytest.mark.asyncio
async def test_list_returns_sorted_public_snapshots_without_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    await store.add_user_job(
        ScheduleJob(
            job_id=str(OTHER_UUID),
            message="Later",
            schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
            created_at_ms=1,
            updated_at_ms=1,
        )
    )
    await store.add_user_job(
        ScheduleJob(
            job_id=str(JOB_UUID),
            message="Earlier",
            schedule=JobSchedule.at("2026-08-07T14:00:00.000+00:00"),
            created_at_ms=2,
            updated_at_ms=2,
        )
    )
    await store.add_system_job(
        ScheduleJob(
            job_id="9ba7b810-9dad-41d1-80b4-00c04fd430c8",
            source="system",
            message="Hidden",
            schedule=JobSchedule.at("2026-08-07T15:00:00.000+00:00"),
            created_at_ms=3,
            updated_at_ms=3,
        )
    )

    result = await _gateway(
        ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: JOB_UUID)
    ).call(ModelToolCall(id="call_list", name="schedule", arguments='{"action":"list"}'))

    assert result.status == "success"
    assert result.content == (
        '{"jobs":[{"job_id":"550e8400-e29b-41d4-a716-446655440000",'
        '"message":"Earlier","schedule":{"type":"at",'
        '"at_time":"2026-08-07T14:00:00.000+00:00"}},'
        '{"job_id":"6fa459ea-ee8a-4ca4-894e-db77e160355e",'
        '"message":"Later","schedule":{"type":"at",'
        '"at_time":"2026-08-07T13:00:00.000+00:00"}}]}'
    )


@pytest.mark.asyncio
async def test_remove_missing_job_returns_safe_error_without_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    channel = ConfirmationChannel(TURN_UUID)
    result = await _gateway(
        ScheduleTool(
            store=WorkspaceScheduleStore(_state(workspace, agent_home)),
            now=lambda: NOW,
            new_uuid=lambda: JOB_UUID,
        ),
        channel=channel,
    ).call(
        ModelToolCall(
            id="call_missing",
            name="schedule",
            arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert result.status == "error"
    assert result.content == "Schedule Job was not found."
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(channel.next_request(), timeout=0.01)


@pytest.mark.asyncio
async def test_remove_rejects_a_noncanonical_job_id_as_invalid_arguments(
    workspace: Path,
    agent_home: Path,
) -> None:
    result = await _gateway(
        ScheduleTool(
            store=WorkspaceScheduleStore(_state(workspace, agent_home)),
            now=lambda: NOW,
            new_uuid=lambda: JOB_UUID,
        )
    ).call(
        ModelToolCall(
            id="call_invalid_job_id",
            name="schedule",
            arguments='{"action":"remove","job_id":"not-a-uuid"}',
        )
    )

    assert result.status == "error"
    assert result.content == "Invalid arguments for schedule."


@pytest.mark.asyncio
async def test_remove_hides_system_jobs_as_not_found_without_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    system_job = ScheduleJob(
        job_id=str(JOB_UUID),
        source="system",
        message="Internal",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_system_job(system_job)
    channel = ConfirmationChannel(TURN_UUID)

    result = await _gateway(
        ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: OTHER_UUID),
        channel=channel,
    ).call(
        ModelToolCall(
            id="call_remove_system",
            name="schedule",
            arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
        )
    )

    assert result.status == "error"
    assert result.content == "Schedule Job was not found."
    assert await store.snapshot() == (system_job,)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(channel.next_request(), timeout=0.01)


@pytest.mark.asyncio
async def test_approved_remove_returns_the_confirmed_public_snapshot(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Remove me",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    channel = ConfirmationChannel(TURN_UUID)
    task = asyncio.create_task(
        _gateway(
            ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: OTHER_UUID),
            channel=channel,
        ).call(
            ModelToolCall(
                id="call_remove",
                name="schedule",
                arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
            )
        )
    )

    request = await channel.next_request()
    channel.respond_to_confirmation(request.confirmation_id, "approved")
    result = await task

    assert result.status == "success"
    assert result.content == (
        '{"action":"remove","job":{"job_id":"550e8400-e29b-41d4-a716-446655440000",'
        '"message":"Remove me","schedule":{"type":"at",'
        '"at_time":"2026-08-07T13:00:00.000+00:00"}}}'
    )
    assert await store.snapshot() == ()


@pytest.mark.asyncio
async def test_remove_rejects_a_changed_public_snapshot_after_confirmation(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    original = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Original",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    changed = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Changed",
        schedule=original.schedule,
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(original)
    channel = ConfirmationChannel(TURN_UUID)
    tool = ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: OTHER_UUID)
    task = asyncio.create_task(
        _gateway(tool, channel=channel).call(
            ModelToolCall(
                id="call_stale_remove",
                name="schedule",
                arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
            )
        )
    )
    request = await channel.next_request()
    await store.remove_user_job(str(JOB_UUID), expected=original)
    await store.add_user_job(changed)
    channel.respond_to_confirmation(request.confirmation_id, "approved")

    result = await task

    assert result.status == "error"
    assert result.content == "Schedule Job changed before removal. Request removal again."
    assert await store.snapshot() == (changed,)


@pytest.mark.asyncio
async def test_remove_reports_stale_when_confirmed_job_disappears(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Remove me",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    channel = ConfirmationChannel(TURN_UUID)
    task = asyncio.create_task(
        _gateway(
            ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: OTHER_UUID),
            channel=channel,
        ).call(
            ModelToolCall(
                id="call_disappeared_remove",
                name="schedule",
                arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
            )
        )
    )

    request = await channel.next_request()
    await store.remove_user_job(str(JOB_UUID), expected=job)
    channel.respond_to_confirmation(request.confirmation_id, "approved")

    result = await task

    assert result.status == "error"
    assert result.content == "Schedule Job changed before removal. Request removal again."


@pytest.mark.asyncio
async def test_concurrent_remove_confirmations_keep_their_own_public_snapshot(
    workspace: Path,
    agent_home: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace, agent_home))
    original = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Original",
        schedule=JobSchedule.at("2026-08-07T13:00:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    changed = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Changed",
        schedule=original.schedule,
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(original)
    tool = ScheduleTool(store=store, now=lambda: NOW, new_uuid=lambda: OTHER_UUID)
    first_channel = ConfirmationChannel(TURN_UUID)
    first_task = asyncio.create_task(
        _gateway(tool, channel=first_channel).call(
            ModelToolCall(
                id="call_first_remove",
                name="schedule",
                arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
            )
        )
    )
    first_request = await first_channel.next_request()

    await store.remove_user_job(str(JOB_UUID), expected=original)
    await store.add_user_job(changed)
    second_channel = ConfirmationChannel(TURN_UUID)
    second_task = asyncio.create_task(
        _gateway(tool, channel=second_channel).call(
            ModelToolCall(
                id="call_second_remove",
                name="schedule",
                arguments=json.dumps({"action": "remove", "job_id": str(JOB_UUID)}),
            )
        )
    )
    second_request = await second_channel.next_request()

    first_channel.respond_to_confirmation(first_request.confirmation_id, "approved")
    first_result = await first_task

    assert first_result.status == "error"
    assert first_result.content == "Schedule Job changed before removal. Request removal again."
    assert await store.snapshot() == (changed,)

    second_channel.respond_to_confirmation(second_request.confirmation_id, "approved")
    second_result = await second_task
    assert second_result.status == "success"
    assert '"message":"Changed"' in second_result.content
    assert await store.snapshot() == ()
