from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import UUID

import pytest

from myclaw.agent.run import (
    AgentRunCancelledPayload,
    AgentRunCompletedPayload,
    AgentRunFailedPayload,
    AgentRunInterface,
    AgentRunPayload,
    AgentRunStartedPayload,
)
from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.errors import ErrorInfo
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from myclaw.schedule.model import JobSchedule, ScheduleJob
from myclaw.schedule.service import ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.session import Session, SessionStoragePartition
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider

JOB_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
OTHER_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
START = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


class ControlledClock:
    def __init__(self, start: datetime) -> None:
        self._now = start
        self._waiters: list[tuple[datetime, asyncio.Future[None]]] = []

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        future = asyncio.get_running_loop().create_future()
        self._waiters.append((self._now + timedelta(seconds=seconds), future))
        await future

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        for deadline, future in tuple(self._waiters):
            if deadline <= self._now and not future.done():
                future.set_result(None)
                self._waiters.remove((deadline, future))


class RecordingAgentRun(AgentRunInterface):
    def __init__(self, *, block: bool = False) -> None:
        self.calls: list[tuple[Session, str, str, bool]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = block

    def run_agent(
        self,
        session: Session,
        input: str,
        route: str,
        stream: bool,
        confirmation: object | None = None,
    ) -> AsyncIterator[AgentRunPayload]:
        del confirmation

        async def run() -> AsyncIterator[AgentRunPayload]:
            self.calls.append((session, input, route, stream))
            self.started.set()
            session.add_message("user", input)
            if self.block:
                await self.release.wait()
            session.add_message(
                "assistant",
                "Done.",
                tool_calls=[],
                status="completed",
                error=None,
                token_usage={
                    "model_calls": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            )
            yield AgentRunStartedPayload()
            yield AgentRunCompletedPayload(
                content="Done.",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return run()


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    return state


def _job(*, job_id: UUID = JOB_UUID, at_time: str = "2026-08-07T11:59:00.000+00:00") -> ScheduleJob:
    return ScheduleJob(
        job_id=str(job_id),
        message="Run this.",
        schedule=JobSchedule.at(at_time),
        created_at_ms=1,
        updated_at_ms=1,
    )


async def _wait_until(predicate: object) -> None:
    check = predicate
    if not callable(check):
        raise TypeError("predicate must be callable")
    for _ in range(100):
        if check():
            return
        await asyncio.sleep(0)
    raise AssertionError("condition did not become true")


@pytest.mark.asyncio
async def test_overdue_at_runs_once_through_shared_agent_run_and_deletes_definition(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_job())
    agent_run = RecordingAgentRun()
    service = ScheduleService(store=store, agent_run=agent_run, workspace_state=state, clock=ControlledClock(START))

    service.start()
    await agent_run.started.wait()
    await service.close()

    assert len(agent_run.calls) == 1
    session, message, route, stream = agent_run.calls[0]
    assert message == "Run this."
    assert route == "schedule"
    assert stream is False
    assert session.session_id == f"schedule_{JOB_UUID}"
    assert session.storage_directory == state.schedule_sessions_directory
    assert (state.schedule_sessions_directory / f"schedule_{JOB_UUID}.jsonl").exists()
    assert not (state.sessions_directory / f"schedule_{JOB_UUID}.jsonl").exists()
    assert await store.snapshot() == ()
    assert service.status_snapshot().to_dict() == {
        "status": "available",
        "active_job_count": 0,
    }


@pytest.mark.asyncio
async def test_revision_wakeup_dispatches_newly_added_at_job(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(store=store, agent_run=agent_run, workspace_state=state, clock=clock)
    service.start()
    await asyncio.sleep(0)

    await store.add_user_job(_job(at_time="2026-08-07T13:00:00.000+00:00"))
    await asyncio.sleep(0)
    assert agent_run.calls == []
    clock.advance(60 * 60)
    await agent_run.started.wait()
    await service.close()

    assert len(agent_run.calls) == 1


@pytest.mark.asyncio
async def test_remove_before_reservation_prevents_the_at_run(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _job(at_time="2026-08-07T13:00:00.000+00:00")
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(store=store, agent_run=agent_run, workspace_state=state, clock=clock)
    service.start()
    await asyncio.sleep(0)

    await store.remove_user_job(job.job_id, expected=job)
    clock.advance(60 * 60)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    await service.close()

    assert agent_run.calls == []
    assert await store.snapshot() == ()


@pytest.mark.asyncio
async def test_remove_after_reservation_allows_the_current_run_without_resurrection(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _job()
    await store.add_user_job(job)
    agent_run = RecordingAgentRun(block=True)
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )
    service.start()
    await agent_run.started.wait()

    await store.remove_user_job(job.job_id, expected=job)
    agent_run.release.set()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert len(agent_run.calls) == 1
    assert await store.snapshot() == ()
    assert service.status_snapshot().status == "available"


@pytest.mark.asyncio
async def test_store_fault_stops_dispatch_and_is_visible_in_status(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)

    def fail_replace(path: Path, content: str) -> None:
        del path, content
        raise OSError("injected replacement failure")

    store = WorkspaceScheduleStore(state, replace_text=fail_replace)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )
    service.start()

    with pytest.raises(OSError, match="injected replacement failure"):
        await store.add_user_job(_job())
    await asyncio.sleep(0)
    await service.close()

    assert agent_run.calls == []
    assert service.status_snapshot().to_dict() == {
        "status": "faulted",
        "active_job_count": 0,
    }


@pytest.mark.asyncio
async def test_shutdown_cancellation_keeps_at_job_pending(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_job())
    agent_run = RecordingAgentRun(block=True)
    service = ScheduleService(store=store, agent_run=agent_run, workspace_state=state, clock=ControlledClock(START))

    service.start()
    await agent_run.started.wait()
    await service.close()

    assert await store.snapshot() == (_job(),)


@pytest.mark.asyncio
async def test_agent_run_cancelled_payload_keeps_at_job_pending(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_job())

    class CancelledRun(RecordingAgentRun):
        def run_agent(
            self,
            session: Session,
            input: str,
            route: str,
            stream: bool,
            confirmation: object | None = None,
        ) -> AsyncIterator[AgentRunPayload]:
            del confirmation

            async def run() -> AsyncIterator[AgentRunPayload]:
                self.calls.append((session, input, route, stream))
                yield AgentRunStartedPayload()
                yield AgentRunCancelledPayload(partial_content="")

            return run()

    agent_run = CancelledRun()
    service = ScheduleService(store=store, agent_run=agent_run, workspace_state=state, clock=ControlledClock(START))
    service.start()
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await service.close()

    assert await store.snapshot() == (_job(),)


@pytest.mark.asyncio
async def test_prepared_runtime_executes_at_job_with_schedule_route_and_partition(
    workspace: Path,
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_job())
    provider = ScriptedFakeProvider(
        completions=(
            ModelResponse(
                message=AssistantModelMessage(content="Scheduled result."),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            ),
        )
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: START,
        new_uuid=lambda: OTHER_UUID,
    )

    runtime.schedule_service.start()
    await _wait_until(lambda: len(provider.complete_requests) == 1)
    await runtime.close()

    request = cast(ModelRequest, provider.complete_requests[0])
    assert request.route == "schedule"
    assert request.stream is False
    assert await WorkspaceScheduleStore(state).snapshot() == ()
    session = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in session.messages] == ["user", "assistant"]
    assert session.messages[-1]["content"] == "Scheduled result."
    assert (state.schedule_sessions_directory / f"schedule_{JOB_UUID}.jsonl").exists()
    assert not (state.sessions_directory / f"schedule_{JOB_UUID}.jsonl").exists()


@pytest.mark.asyncio
async def test_failed_at_is_deleted_and_a_new_service_does_not_replay_it(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_job())

    class FailedRun(RecordingAgentRun):
        def run_agent(
            self,
            session: Session,
            input: str,
            route: str,
            stream: bool,
            confirmation: object | None = None,
        ) -> AsyncIterator[AgentRunPayload]:
            del confirmation

            async def run() -> AsyncIterator[AgentRunPayload]:
                self.calls.append((session, input, route, stream))
                yield AgentRunStartedPayload()
                yield AgentRunFailedPayload(
                    error=ErrorInfo(code="model_failed", message="The model request failed.")
                )

            return run()

    first = FailedRun()
    service = ScheduleService(
        store=store,
        agent_run=first,
        workspace_state=state,
        clock=ControlledClock(START),
    )
    service.start()
    await _wait_until(lambda: len(first.calls) == 1)
    await service.close()

    assert await store.snapshot() == ()

    restarted_store = WorkspaceScheduleStore(state)
    second = RecordingAgentRun()
    restarted = ScheduleService(
        store=restarted_store,
        agent_run=second,
        workspace_state=state,
        clock=ControlledClock(START),
    )
    restarted.start()
    await asyncio.sleep(0)
    await restarted.close()

    assert second.calls == []
