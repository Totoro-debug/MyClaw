from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleJobExecutionError, ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.tools.core.schedule import ScheduleTool
from myclaw.tools.tool_gateway import ModelToolCall, ToolGateway, ToolResult

NOW = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


class _Clock:
    def now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    return state


async def _wait_until(predicate: object) -> None:
    if not callable(predicate):
        raise TypeError("predicate must be callable")
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_schedule_service_facade_preserves_user_job_management(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    service = ScheduleService(store=store, clock=_Clock())
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Review the project.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )

    assert await service.add_user_job(job) == job
    assert await service.public_snapshot() == (job,)
    assert await service.remove_user_job(job.job_id, expected=job) is True
    assert await service.public_snapshot() == ()


@pytest.mark.asyncio
async def test_schedule_tool_guard_is_task_local_and_list_remove_stay_available(
    workspace: Path,
    agent_home: Path,
) -> None:
    identity = Workspace.from_path(workspace)
    state = WorkspaceState(identity)
    state.initialize(agent_home_root=agent_home)
    service = ScheduleService(store=WorkspaceScheduleStore(state), clock=_Clock())
    foreground = ToolGateway(workspace=identity, schedule_service=service)
    scheduled = ToolGateway(workspace=identity, schedule_service=service)
    barrier = asyncio.Event()
    foreground_done = asyncio.Event()

    async def scheduled_task() -> tuple[ToolResult, ToolResult, ToolResult]:
        token = ScheduleTool._in_schedule_job.set(True)
        try:
            await barrier.wait()
            refused = await scheduled.call(
                ModelToolCall(
                    id="scheduled_add",
                    name="schedule",
                    arguments=json.dumps(
                        {
                            "action": "add",
                            "message": "recursive",
                            "every_seconds": 60,
                        }
                    ),
                )
            )
            await foreground_done.wait()
            listed = await scheduled.call(
                ModelToolCall(
                    id="scheduled_list",
                    name="schedule",
                    arguments='{"action":"list"}',
                )
            )
            job_id = (await service.public_snapshot())[0].job_id
            removed = await scheduled.call(
                ModelToolCall(
                    id="scheduled_remove",
                    name="schedule",
                    arguments=json.dumps({"action": "remove", "job_id": job_id}),
                )
            )
            return refused, listed, removed
        finally:
            ScheduleTool._in_schedule_job.reset(token)

    async def foreground_task() -> ToolResult:
        await barrier.wait()
        try:
            return await foreground.call(
                ModelToolCall(
                    id="foreground_add",
                    name="schedule",
                    arguments=json.dumps(
                        {
                            "action": "add",
                            "message": "foreground",
                            "every_seconds": 60,
                        }
                    ),
                )
            )
        finally:
            foreground_done.set()

    barrier.set()
    scheduled_result, foreground_result = await asyncio.gather(
        scheduled_task(),
        foreground_task(),
    )
    refused, listed, removed = scheduled_result
    assert refused.status == "refused"
    assert refused.content == "Schedule add is unavailable in scheduled Agent context."
    assert foreground_result.status == "success"
    assert listed.status == "success"
    assert removed.status == "success"
    assert await service.public_snapshot() == ()


@pytest.mark.asyncio
async def test_schedule_service_start_rejects_unbound_callback_before_reservation(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run this.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    service = ScheduleService(store=store, clock=_Clock())

    with pytest.raises(RuntimeError, match="on_schedule_job"):
        service.start()

    assert await store.snapshot() == (job,)
    assert service.status_snapshot().active_job_count == 0


@pytest.mark.asyncio
async def test_schedule_service_maps_structured_callback_failure_without_leaking_details(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="6fa459ea-ee8a-4ca4-894e-db77e160355e",
        message="Run this.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    started = asyncio.Event()

    async def callback(active_job: ScheduleJob) -> None:
        assert active_job == job
        started.set()
        raise ScheduleJobExecutionError(
            ErrorInfo(code="model_failed", message="safe model failure")
        )

    service = ScheduleService(store=store, clock=_Clock(), on_schedule_job=callback)
    service.start()
    await started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    saved = (await store.snapshot())[0]
    assert saved.state.last_status == "error"
    assert saved.state.last_error == "safe model failure"


@pytest.mark.asyncio
async def test_schedule_service_callback_cancellation_leaves_job_pending(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id="9ba7b810-9dad-41d1-80b4-00c04fd430c8",
        message="Run this.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    started = asyncio.Event()

    async def callback(active_job: ScheduleJob) -> None:
        assert active_job == job
        started.set()
        raise asyncio.CancelledError()

    service = ScheduleService(store=store, clock=_Clock(), on_schedule_job=callback)
    service.start()
    await started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert await store.snapshot() == (job,)
