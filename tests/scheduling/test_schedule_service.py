from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast
from uuid import UUID, uuid4

import pytest

from myclaw.agent.context import ContextBuilder
from myclaw.agent.run import (
    AgentRun,
    AgentRunCancelledPayload,
    AgentRunCompletedPayload,
    AgentRunEmitter,
    AgentRunFailedPayload,
    AgentRunInterface,
    AgentRunModelCallCompletedPayload,
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
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelStreamEvent,
    ModelUsage,
)
from myclaw.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState
from myclaw.schedule.service import ScheduleService
from myclaw.schedule.store import WorkspaceScheduleStore
from myclaw.session.session import Session, SessionStoragePartition
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, write_schedule_state
from tests.fixtures.diagnostic_capture import capture_diagnostics

JOB_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
OTHER_UUID = UUID("6fa459ea-ee8a-4ca4-894e-db77e160355e")
START = datetime(2026, 8, 7, 12, 0, tzinfo=UTC)


class ControlledClock:
    def __init__(self, start: datetime) -> None:
        self._now = start
        self._monotonic = 0.0
        self.wait_started = asyncio.Event()
        self._waiters: list[tuple[float, asyncio.Future[None]]] = []

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    async def sleep(self, seconds: float) -> None:
        self.wait_started.set()
        future = asyncio.get_running_loop().create_future()
        self._waiters.append((self._monotonic + seconds, future))
        await future

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds
        for deadline, future in tuple(self._waiters):
            if deadline <= self._monotonic and not future.done():
                future.set_result(None)
                self._waiters.remove((deadline, future))

    def jump_wall(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


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
            yield AgentRunModelCallCompletedPayload(
                content="Done.",
                continues_with_tools=False,
            )
            yield AgentRunCompletedPayload(
                content="Done.",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )

        return run()


class AwaitableRecordingAgentRun:
    def __init__(self, timeline: list[str]) -> None:
        self.calls: list[tuple[list[dict[str, object]], dict[str, object], str]] = []
        self.started = asyncio.Event()
        self._timeline = timeline

    async def run(
        self,
        messages: list[dict[str, object]],
        current_user: dict[str, object],
        *,
        route: str,
        emitter: AgentRunEmitter,
        continuation_preparer: object | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, object]]:
        del continuation_preparer, cancel_requested
        self._timeline.append("run")
        self.calls.append((messages, current_user, route))
        self.started.set()
        await emitter.emit(AgentRunStartedPayload())
        await emitter.emit(
            AgentRunCompletedPayload(
                content="Done.",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        )
        return [
            {"role": "user", "content": "Run this."},
            {
                "role": "assistant",
                "content": "Done.",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
        ]


class OutcomeAwaitableAgentRun:
    def __init__(
        self,
        terminal: AgentRunCompletedPayload | AgentRunFailedPayload | AgentRunCancelledPayload,
        increment: list[dict[str, object]],
    ) -> None:
        self.terminal = terminal
        self.increment = increment
        self.started = asyncio.Event()

    async def run(
        self,
        messages: list[dict[str, object]],
        current_user: dict[str, object],
        *,
        route: str,
        emitter: AgentRunEmitter,
        continuation_preparer: object | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, object]]:
        del messages, current_user, route, continuation_preparer, cancel_requested
        self.started.set()
        await emitter.emit(AgentRunStartedPayload())
        await emitter.emit(self.terminal)
        return self.increment


class RaisingAwaitableAgentRun:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def run(
        self,
        messages: list[dict[str, object]],
        current_user: dict[str, object],
        *,
        route: str,
        emitter: AgentRunEmitter,
        continuation_preparer: object | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> list[dict[str, object]]:
        del (
            messages,
            current_user,
            route,
            emitter,
            continuation_preparer,
            cancel_requested,
        )
        self.started.set()
        raise RuntimeError("unexpected Agent Run failure")


class _BlockingScheduleRouter:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self._never = asyncio.Event()

    def route_status(self, route: str) -> None:
        del route

    def stream(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[object],
    ) -> AsyncIterator[ModelStreamEvent]:
        del route, messages, tools
        raise AssertionError("Schedule Agent Run must not stream")

    async def complete(
        self,
        route: str,
        *,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[object],
    ) -> ModelResponse:
        del route, messages, tools
        self.started.set()
        await self._never.wait()
        raise AssertionError("Blocking Schedule Router was unexpectedly released")


class ConcurrentScheduleAndForegroundProvider:
    def __init__(self) -> None:
        self.schedule_started = asyncio.Event()
        self.release_schedule = asyncio.Event()
        self.complete_requests: list[ModelRequest] = []
        self.stream_requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.complete_requests.append(request)
        self.schedule_started.set()
        await self.release_schedule.wait()
        return ModelResponse(
            message=AssistantModelMessage(content="Scheduled result."),
            usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
            finish_reason="stop",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]:
        self.stream_requests.append(request)
        yield ModelCompleted(
            response=ModelResponse(
                message=AssistantModelMessage(content="Foreground result."),
                usage=ModelUsage(input_tokens=2, output_tokens=1, total_tokens=3),
                finish_reason="stop",
            )
        )

    async def close(self) -> None:
        return None


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


def _every_job(
    *,
    job_id: UUID = JOB_UUID,
    every_seconds: int = 10,
    created_at_ms: int | None = None,
) -> ScheduleJob:
    timestamp = int(START.timestamp() * 1000) if created_at_ms is None else created_at_ms
    return ScheduleJob(
        job_id=str(job_id),
        message="Run this.",
        schedule=JobSchedule.every(every_seconds),
        created_at_ms=timestamp,
        updated_at_ms=timestamp,
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
    service = ScheduleService(
        store=store, agent_run=agent_run, workspace_state=state, clock=ControlledClock(START)
    )

    service.start()
    await agent_run.started.wait()
    await service.close()

    assert len(agent_run.calls) == 1
    session, message, route, stream = agent_run.calls[0]
    assert message == "Run this."
    assert route == "schedule"
    assert stream is False
    assert session.session_id == f"schedule_{JOB_UUID}"
    assert (state.schedule_sessions_directory / f"schedule_{JOB_UUID}.jsonl").exists()
    assert not (state.sessions_directory / f"schedule_{JOB_UUID}.jsonl").exists()
    assert await store.snapshot() == ()
    assert service.status_snapshot().to_dict() == {
        "status": "available",
        "active_job_count": 0,
    }


@pytest.mark.asyncio
async def test_awaitable_schedule_run_prepares_context_and_persists_before_terminal_commit(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(
        _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    )
    timeline: list[str] = []

    async def prepare_context(
        session: Session,
        current_user: dict[str, object],
    ) -> list[dict[str, object]]:
        del session
        timeline.append("context")
        return [
            {"role": "system", "content": "Schedule system"},
            {"role": "user", "content": current_user["content"]},
        ]

    original_append = Session.append_messages
    original_persist = Session.persist
    original_commit = store.commit_terminal

    def fail_context_builder(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("Schedule Service must not construct ContextBuilder")

    monkeypatch.setattr(ContextBuilder, "__init__", fail_context_builder)

    def append_messages(session: Session, messages: list[dict[str, object]]) -> None:
        timeline.append("append")
        original_append(session, messages)

    def persist(session: Session) -> None:
        timeline.append("persist")
        original_persist(session)

    async def commit_terminal(
        job_id: str,
        *,
        expected: ScheduleJob | None = None,
        finished_at_ms: int,
        status: Literal["ok", "error"],
        error: str | None = None,
        now_ms: int | None = None,
    ) -> ScheduleJob | None:
        timeline.append("terminal")
        return await original_commit(
            job_id,
            expected=expected,
            finished_at_ms=finished_at_ms,
            status=status,
            error=error,
            now_ms=now_ms,
        )

    monkeypatch.setattr(Session, "append_messages", append_messages)
    monkeypatch.setattr(Session, "persist", persist)
    monkeypatch.setattr(store, "commit_terminal", commit_terminal)
    agent_run = AwaitableRecordingAgentRun(timeline)
    service = ScheduleService(
        store=store,
        agent_run=agent_run,  # type: ignore[arg-type]
        workspace_state=state,
        clock=ControlledClock(START),
        context_preparer=prepare_context,
    )

    service.start()
    await agent_run.started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert timeline == ["context", "run", "append", "persist", "terminal"]
    messages, current_user, route = agent_run.calls[0]
    assert messages == [
        {"role": "system", "content": "Schedule system"},
        {"role": "user", "content": "Run this."},
    ]
    assert current_user == {"role": "user", "content": "Run this."}
    assert route == "schedule"
    session = Session.load(
        state,
        f"schedule_{JOB_UUID}",
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [message["role"] for message in session.messages] == ["user", "assistant"]


@pytest.mark.asyncio
async def test_failed_awaitable_schedule_run_persists_repair_before_error_state(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await store.add_user_job(job)
    failure = AgentRunFailedPayload(
        error=ErrorInfo(code="model_failed", message="The model request failed.")
    )
    run = OutcomeAwaitableAgentRun(
        failure,
        [
            {"role": "user", "content": job.message},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [],
                "status": "error",
                "error": {"code": "model_failed", "message": "The model request failed."},
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            },
        ],
    )

    async def prepare_context(
        session: Session,
        current_user: dict[str, object],
    ) -> list[dict[str, object]]:
        del session
        return [
            {"role": "system", "content": "Schedule system"},
            {"role": "user", "content": current_user["content"]},
        ]

    service = ScheduleService(
        store=store,
        agent_run=run,  # type: ignore[arg-type]
        workspace_state=state,
        clock=ControlledClock(START),
        context_preparer=prepare_context,
    )
    service.start()
    await run.started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    saved = (await store.snapshot())[0]
    assert saved.state.last_status == "error"
    assert saved.state.last_error == "The model request failed."
    session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert session.messages[-1]["status"] == "error"


@pytest.mark.asyncio
async def test_unexpected_awaitable_failure_persists_repair_before_error_state(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await store.add_user_job(job)
    run = RaisingAwaitableAgentRun()

    async def prepare_context(
        session: Session,
        current_user: dict[str, object],
    ) -> list[dict[str, object]]:
        del session
        return [
            {"role": "system", "content": "Schedule system"},
            {"role": "user", "content": current_user["content"]},
        ]

    service = ScheduleService(
        store=store,
        agent_run=run,  # type: ignore[arg-type]
        workspace_state=state,
        clock=ControlledClock(START),
        context_preparer=prepare_context,
    )
    service.start()
    await run.started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    saved = (await store.snapshot())[0]
    assert saved.state.last_status == "error"
    assert saved.state.last_error == "The model request failed."
    session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert session.messages[-1]["status"] == "error"


@pytest.mark.asyncio
async def test_cancelled_awaitable_schedule_run_persists_increment_but_keeps_job_pending(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _job()
    await store.add_user_job(job)
    run = OutcomeAwaitableAgentRun(
        AgentRunCancelledPayload(partial_content="Partial."),
        [
            {"role": "user", "content": job.message},
            {
                "role": "assistant",
                "content": "Partial.",
                "tool_calls": [],
                "status": "interrupted",
                "error": {
                    "code": "turn_cancelled",
                    "message": "Turn interrupted by user.",
                },
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            },
        ],
    )

    async def prepare_context(
        session: Session,
        current_user: dict[str, object],
    ) -> list[dict[str, object]]:
        del session
        return [
            {"role": "system", "content": "Schedule system"},
            {"role": "user", "content": current_user["content"]},
        ]

    service = ScheduleService(
        store=store,
        agent_run=run,  # type: ignore[arg-type]
        workspace_state=state,
        clock=ControlledClock(START),
        context_preparer=prepare_context,
    )
    service.start()
    await run.started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert await store.snapshot() == (job,)
    session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert session.messages[-1]["status"] == "interrupted"


@pytest.mark.asyncio
async def test_production_awaitable_shutdown_persists_user_before_leaving_job_pending(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _job()
    await store.add_user_job(job)
    router = _BlockingScheduleRouter()

    async def prepare_context(
        session: Session,
        current_user: dict[str, object],
    ) -> list[dict[str, object]]:
        del session
        return [
            {"role": "system", "content": "Schedule system"},
            {"role": "user", "content": current_user["content"]},
        ]

    service = ScheduleService(
        store=store,
        agent_run=AgentRun(provider=router),
        workspace_state=state,
        clock=ControlledClock(START),
        context_preparer=prepare_context,
    )
    service.start()
    await router.started.wait()
    await service.close()

    assert await store.snapshot() == (job,)
    session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [(message["role"], message["content"]) for message in session.messages] == [
        ("user", job.message)
    ]


@pytest.mark.asyncio
async def test_awaitable_shutdown_during_context_preparation_persists_user(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _job()
    await store.add_user_job(job)
    run = OutcomeAwaitableAgentRun(
        AgentRunCompletedPayload(
            content="Unused.",
            usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        ),
        [],
    )
    context_started = asyncio.Event()
    context_never_completes = asyncio.Event()

    async def prepare_context(
        session: Session,
        current_user: dict[str, object],
    ) -> list[dict[str, object]]:
        del session, current_user
        context_started.set()
        await context_never_completes.wait()
        raise AssertionError("Schedule context preparation was unexpectedly released")

    service = ScheduleService(
        store=store,
        agent_run=run,  # type: ignore[arg-type]
        workspace_state=state,
        clock=ControlledClock(START),
        context_preparer=prepare_context,
    )
    service.start()
    await context_started.wait()
    await service.close()

    assert not run.started.is_set()
    assert await store.snapshot() == (job,)
    session = Session.load(
        state,
        job.session_id,
        partition=SessionStoragePartition.SCHEDULE,
    )
    assert [(message["role"], message["content"]) for message in session.messages] == [
        ("user", job.message)
    ]


@pytest.mark.asyncio
async def test_cron_startup_ignores_match_at_startup_boundary(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("0 8 * * *", "America/New_York"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    assert agent_run.calls == []

    clock.advance(24 * 60 * 60 - 60)
    await asyncio.sleep(0)
    assert agent_run.calls == []
    clock.advance(60)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()
    saved = (await store.snapshot())[0]
    assert saved.state.last_status == "ok"
    assert saved.state.last_finished_at_ms == int(clock.now().timestamp() * 1000)


@pytest.mark.asyncio
async def test_cron_startup_skips_an_overdue_match_until_the_next_one(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("0 * * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    clock = ControlledClock(START + timedelta(minutes=30))
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    assert agent_run.calls == []
    clock.advance(30 * 60)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await service.close()


@pytest.mark.asyncio
async def test_cron_spring_gap_skips_the_nonexistent_local_time(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("30 2 * * *", "America/New_York"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    start = datetime(2026, 3, 8, 6, 0, tzinfo=UTC)
    clock = ControlledClock(start)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    clock.advance(2 * 60 * 60)
    await asyncio.sleep(0)
    assert agent_run.calls == []

    clock.advance(22 * 60 * 60 + 30 * 60)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await service.close()


@pytest.mark.asyncio
async def test_cron_fall_overlap_runs_both_absolute_instants(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("30 1 * * *", "America/New_York"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    clock = ControlledClock(datetime(2026, 11, 1, 4, 0, tzinfo=UTC))
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    clock.advance(90 * 60)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)

    clock.advance(60 * 60)
    await _wait_until(lambda: len(agent_run.calls) == 2)
    await service.close()


@pytest.mark.asyncio
async def test_cron_forward_wall_jump_skips_missed_occurrences(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("0 * * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    clock.jump_wall(2 * 60 * 60)
    clock.advance(60)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert agent_run.calls == []

    clock.advance(59 * 60)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await service.close()


@pytest.mark.asyncio
async def test_cron_backward_wall_jump_does_not_repeat_an_absolute_occurrence(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("0 * * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    clock.advance(60 * 60)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)

    clock.jump_wall(-30 * 60)
    clock.advance(30 * 60)
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert len(agent_run.calls) == 1

    clock.advance(60 * 60)
    await _wait_until(lambda: len(agent_run.calls) == 2)
    await service.close()


@pytest.mark.asyncio
async def test_cron_active_overlap_consumes_the_skipped_occurrence(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("* * * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun(block=True)
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )
    capture = capture_diagnostics()

    try:
        service.start()
        await clock.wait_started.wait()
        clock.wait_started.clear()
        clock.advance(60)
        await agent_run.started.wait()
        await clock.wait_started.wait()
        clock.wait_started.clear()

        clock.advance(60)
        await _wait_until(clock.wait_started.is_set)
        assert len(agent_run.calls) == 1

        agent_run.block = False
        agent_run.release.set()
        await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
        assert len(agent_run.calls) == 1

        clock.advance(60)
        await _wait_until(lambda: len(agent_run.calls) == 2)
        await service.close()
    finally:
        capture.close()

    assert capture.event_text.count("Schedule Job occurrence skipped while active") == 1
    session_log_text = (state.logs_directory / f"{job.session_id}.log").read_text(encoding="utf-8")
    assert session_log_text.count("Schedule Job occurrence skipped while active") == 1


@pytest.mark.asyncio
async def test_cron_failure_waits_for_the_next_match_without_retry(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.cron("* * * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await store.add_user_job(job)

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
                self.started.set()
                yield AgentRunStartedPayload()
                yield AgentRunFailedPayload(
                    error=ErrorInfo(code="model_failed", message="The model request failed.")
                )

            return run()

    clock = ControlledClock(START)
    agent_run = FailedRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    saved = (await store.snapshot())[0]
    assert saved.state.last_status == "error"
    assert saved.state.last_error == "The model request failed."

    clock.advance(30)
    await asyncio.sleep(0)
    assert len(agent_run.calls) == 1
    clock.advance(30)
    await _wait_until(lambda: len(agent_run.calls) == 2)
    await service.close()


@pytest.mark.asyncio
async def test_overdue_every_commits_completion_and_restarts_interval_from_finish(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await agent_run.started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)

    saved = (await store.snapshot())[0]
    assert len(agent_run.calls) == 1
    assert saved.state.last_status == "ok"
    assert saved.state.last_finished_at_ms == int(START.timestamp() * 1000)

    clock.advance(9)
    await asyncio.sleep(0)
    assert len(agent_run.calls) == 1
    clock.advance(1)
    await _wait_until(lambda: len(agent_run.calls) == 2)
    await service.close()

    assert len(agent_run.calls) == 2


@pytest.mark.asyncio
async def test_future_every_uses_created_at_for_its_first_deadline(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    created_at_ms = int((START - timedelta(seconds=5)).timestamp() * 1000)
    await store.add_user_job(_every_job(created_at_ms=created_at_ms))
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    assert agent_run.calls == []
    clock.advance(4)
    await asyncio.sleep(0)
    assert agent_run.calls == []
    clock.advance(1)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await service.close()


@pytest.mark.asyncio
async def test_every_live_deadline_uses_monotonic_time_across_wall_clock_jump(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_every_job())
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await clock.wait_started.wait()
    clock.jump_wall(60)
    await asyncio.sleep(0)
    assert agent_run.calls == []
    clock.advance(9)
    await asyncio.sleep(0)
    assert agent_run.calls == []
    clock.advance(1)
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("wall_jump_seconds", [-60 * 60, 60 * 60])
async def test_every_completion_starts_a_monotonic_interval_across_wall_clock_jump(
    workspace: Path,
    agent_home: Path,
    wall_jump_seconds: int,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    async def jump_when_terminal_state_is_published() -> None:
        revision = store.revision
        await store.wait_for_change(revision)
        clock.jump_wall(wall_jump_seconds)

    jump_task = asyncio.create_task(jump_when_terminal_state_is_published())
    await asyncio.sleep(0)
    service.start()
    try:
        await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
        await jump_task

        assert len(agent_run.calls) == 1
        clock.advance(9)
        await asyncio.sleep(0)
        assert len(agent_run.calls) == 1
        clock.advance(1)
        await _wait_until(lambda: len(agent_run.calls) == 2)
    finally:
        await service.close()


@pytest.mark.asyncio
async def test_every_execution_time_is_included_before_the_next_start(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun(block=True)
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await agent_run.started.wait()
    clock.advance(3)
    agent_run.block = False
    agent_run.release.set()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)

    clock.advance(9)
    await asyncio.sleep(0)
    assert len(agent_run.calls) == 1
    clock.advance(1)
    await _wait_until(lambda: len(agent_run.calls) == 2)
    await service.close()


@pytest.mark.asyncio
async def test_every_active_job_skips_due_occurrence_without_store_mutation(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(
        every_seconds=5,
        created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000),
    )
    await store.add_user_job(job)
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun(block=True)
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await agent_run.started.wait()
    initial_revision = store.revision
    clock.advance(5)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert len(agent_run.calls) == 1
    assert store.revision == initial_revision
    assert (await store.snapshot())[0].state == ScheduleJobState()

    agent_run.block = False
    agent_run.release.set()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()


@pytest.mark.asyncio
async def test_different_every_jobs_run_concurrently(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    created_at_ms = int((START - timedelta(seconds=20)).timestamp() * 1000)
    await store.add_user_job(_every_job(created_at_ms=created_at_ms))
    await store.add_user_job(
        _every_job(
            job_id=OTHER_UUID,
            created_at_ms=created_at_ms,
        )
    )
    clock = ControlledClock(START)
    agent_run = RecordingAgentRun(block=True)
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await _wait_until(lambda: len(agent_run.calls) == 2)
    assert service.status_snapshot().active_job_count == 2

    agent_run.block = False
    agent_run.release.set()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()


@pytest.mark.asyncio
async def test_every_safe_failure_commits_terminal_error_without_early_retry(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await store.add_user_job(job)

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
                self.started.set()
                yield AgentRunStartedPayload()
                yield AgentRunFailedPayload(
                    error=ErrorInfo(code="model_failed", message="The model request failed.")
                )

            return run()

    clock = ControlledClock(START)
    agent_run = FailedRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=clock,
    )

    service.start()
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    saved = (await store.snapshot())[0]
    assert saved.state.last_status == "error"
    assert saved.state.last_error == "The model request failed."

    clock.advance(9)
    await asyncio.sleep(0)
    assert len(agent_run.calls) == 1
    clock.advance(1)
    await _wait_until(lambda: len(agent_run.calls) == 2)
    await service.close()


@pytest.mark.asyncio
async def test_overdue_every_restarts_once_from_persisted_last_completion(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    first_store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=100)).timestamp() * 1000))
    await first_store.add_user_job(job)
    finished_at_ms = int((START - timedelta(seconds=20)).timestamp() * 1000)
    await first_store.commit_terminal(
        job.job_id,
        finished_at_ms=finished_at_ms,
        status="ok",
    )

    store = WorkspaceScheduleStore(state)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )

    service.start()
    await _wait_until(lambda: len(agent_run.calls) == 1)
    await service.close()

    assert (await store.snapshot())[0].state.last_status == "ok"


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
async def test_due_candidates_are_revalidated_before_reservation(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _job()
    await store.add_user_job(job)
    original_reserve_due = store.reserve_due

    async def remove_before_reservation(
        candidates: tuple[ScheduleJob, ...],
    ) -> tuple[ScheduleJob, ...]:
        await store.remove_user_job(job.job_id, expected=job)
        return await original_reserve_due(candidates)

    monkeypatch.setattr(store, "reserve_due", remove_before_reservation)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )

    service.start()
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
async def test_terminal_at_removal_does_not_delete_a_replacement_job(
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
    replacement = ScheduleJob(
        job_id=job.job_id,
        message="Replacement run.",
        schedule=job.schedule,
        created_at_ms=2,
        updated_at_ms=2,
    )
    await store.add_user_job(replacement)
    agent_run.release.set()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

    assert await store.snapshot() == (replacement,)
    assert service.status_snapshot().status == "available"


@pytest.mark.asyncio
async def test_system_at_job_is_deleted_after_terminal_completion(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    job = ScheduleJob(
        job_id=str(JOB_UUID),
        message="Run this.",
        schedule=JobSchedule.at("2026-08-07T11:59:00.000+00:00"),
        created_at_ms=1,
        updated_at_ms=1,
        source="system",
    )
    write_schedule_state(state, job)
    store = WorkspaceScheduleStore(state)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )

    service.start()
    await agent_run.started.wait()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()

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
async def test_every_terminal_store_fault_preserves_previous_state_and_faults_service(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    initial_store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await initial_store.add_user_job(job)

    def fail_replace(path: Path, content: str) -> None:
        del path, content
        raise OSError("PRIVATE_TERMINAL_STORE_BODY")

    store = WorkspaceScheduleStore(state, replace_text=fail_replace)
    agent_run = RecordingAgentRun()
    capture = capture_diagnostics()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )

    try:
        service.start()
        await agent_run.started.wait()
        await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
        await service.close()
    finally:
        capture.close()

    assert await store.snapshot() == (job,)
    assert store.health == "faulted"
    assert service.status_snapshot().to_dict() == {
        "status": "faulted",
        "active_job_count": 0,
    }
    assert capture.event_text.count("Schedule terminal update failed") == 1
    assert "PRIVATE_TERMINAL_STORE_BODY" not in capture.text
    session_log_text = (state.logs_directory / f"{job.session_id}.log").read_text(encoding="utf-8")
    assert session_log_text.count("Schedule terminal update failed") == 1
    assert "PRIVATE_TERMINAL_STORE_BODY" not in session_log_text


@pytest.mark.asyncio
async def test_safe_agent_failure_logs_one_warning_without_failure_message(
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
                self.started.set()
                yield AgentRunStartedPayload()
                yield AgentRunFailedPayload(
                    error=ErrorInfo(
                        code="model_failed",
                        message="PRIVATE_AGENT_FAILURE_MESSAGE",
                    )
                )

            return run()

    agent_run = FailedRun()
    capture = capture_diagnostics()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )

    try:
        service.start()
        await agent_run.started.wait()
        await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
        await service.close()
    finally:
        capture.close()

    assert capture.event_text.count("Schedule Job failed") == 1
    assert "code=model_failed" in capture.event_text
    assert "PRIVATE_AGENT_FAILURE_MESSAGE" not in capture.text
    session_log_text = (state.logs_directory / f"{_job().session_id}.log").read_text(
        encoding="utf-8"
    )
    assert session_log_text.count("Schedule Job failed") == 1
    assert "code=model_failed" in session_log_text
    assert "PRIVATE_AGENT_FAILURE_MESSAGE" not in session_log_text


@pytest.mark.asyncio
async def test_one_job_exception_does_not_stop_other_due_jobs(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_job())
    await store.add_user_job(_job(job_id=OTHER_UUID))

    class FirstRunRaises(RecordingAgentRun):
        def __init__(self) -> None:
            super().__init__()
            self.second_started = asyncio.Event()

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
                if len(self.calls) == 1:
                    raise RuntimeError("PRIVATE_JOB_EXCEPTION_BODY")
                self.second_started.set()
                yield AgentRunStartedPayload()
                yield AgentRunCompletedPayload(
                    content="Done.",
                    usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                )

            return run()

    agent_run = FirstRunRaises()
    capture = capture_diagnostics()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )

    try:
        service.start()
        await agent_run.second_started.wait()
        await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
        await service.close()
    finally:
        capture.close()

    assert len(agent_run.calls) == 2
    assert await store.snapshot() == ()
    assert service.status_snapshot().status == "available"
    assert "PRIVATE_JOB_EXCEPTION_BODY" not in capture.text


@pytest.mark.asyncio
async def test_terminal_commit_keeps_job_active_until_store_operation_finishes(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    await store.add_user_job(job)
    original_commit_terminal = store.commit_terminal
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()

    async def blocked_commit_terminal(*args: object, **kwargs: object) -> ScheduleJob | None:
        commit_started.set()
        await release_commit.wait()
        return await original_commit_terminal(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(store, "commit_terminal", blocked_commit_terminal)
    agent_run = RecordingAgentRun()
    service = ScheduleService(
        store=store,
        agent_run=agent_run,
        workspace_state=state,
        clock=ControlledClock(START),
    )

    service.start()
    await agent_run.started.wait()
    await commit_started.wait()
    assert service.status_snapshot().active_job_count == 1

    release_commit.set()
    await _wait_until(lambda: service.status_snapshot().active_job_count == 0)
    await service.close()


@pytest.mark.asyncio
async def test_shutdown_cancellation_keeps_at_job_pending(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(_job())
    agent_run = RecordingAgentRun(block=True)
    service = ScheduleService(
        store=store, agent_run=agent_run, workspace_state=state, clock=ControlledClock(START)
    )

    service.start()
    await agent_run.started.wait()
    await service.close()

    assert await store.snapshot() == (_job(),)


@pytest.mark.asyncio
async def test_shutdown_cancellation_keeps_every_job_without_terminal_state(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    job = _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
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
    await service.close()

    assert await store.snapshot() == (job,)
    assert service.status_snapshot().to_dict() == {
        "status": "available",
        "active_job_count": 0,
    }


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
    service = ScheduleService(
        store=store, agent_run=agent_run, workspace_state=state, clock=ControlledClock(START)
    )
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
async def test_prepared_runtime_runs_foreground_while_every_job_is_active(
    workspace: Path,
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    (agent_home / "config.toml").write_text(VALID_CONFIG, encoding="utf-8")
    state = _state(workspace, agent_home)
    store = WorkspaceScheduleStore(state)
    await store.add_user_job(
        _every_job(created_at_ms=int((START - timedelta(seconds=20)).timestamp() * 1000))
    )
    provider = ConcurrentScheduleAndForegroundProvider()
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _configuration: provider,
        now=lambda: START,
        new_uuid=uuid4,
        schedule_scheduler_clock=ControlledClock(START),
    )

    await runtime.start()
    try:
        await provider.schedule_started.wait()
        assert runtime.schedule_service.status_snapshot().active_job_count == 1

        events = [
            event async for event in runtime.conversation.submit("Run the foreground request.")
        ]

        assert events[-1].type == "turn_completed"
        assert runtime.schedule_service.status_snapshot().active_job_count == 1
        assert provider.complete_requests[0].route == "schedule"
        assert any(request.route == "chat" for request in provider.stream_requests)
    finally:
        provider.release_schedule.set()
        await runtime.close()


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
