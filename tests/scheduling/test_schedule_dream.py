from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.errors import ErrorInfo
from myclaw.logging.session import session_log
from myclaw.memory.dream import Dream, DreamResult
from myclaw.memory.manager import MemoryManager
from myclaw.provider.models import AssistantModelMessage, ModelResponse, ModelUsage
from myclaw.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState
from myclaw.schedule.service import ScheduleService
from myclaw.schedule.store import ScheduleStateError, WorkspaceScheduleStore
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.fixtures import ScriptedFakeProvider, ScriptedFakeRouter


class _Clock:
    def now(self) -> datetime:
        return datetime(2026, 8, 7, 12, 0, tzinfo=UTC)

    def monotonic(self) -> float:
        return 0.0

    async def sleep(self, seconds: float) -> None:
        del seconds


class _AdvancingClock:
    def __init__(self, start: datetime | None = None) -> None:
        self.current = (
            datetime(2026, 8, 7, 12, 0, tzinfo=UTC) if start is None else start
        )
        self.elapsed = 0.0
        self.wait_started = asyncio.Event()
        self._waiters: list[tuple[float, asyncio.Future[None]]] = []

    def now(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.elapsed

    async def sleep(self, seconds: float) -> None:
        self.wait_started.set()
        future = asyncio.get_running_loop().create_future()
        self._waiters.append((self.elapsed + seconds, future))
        await future

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)
        self.elapsed += seconds
        for deadline, future in tuple(self._waiters):
            if deadline <= self.elapsed and not future.done():
                future.set_result(None)
                self._waiters.remove((deadline, future))


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=agent_home)
    return state


def _dream(state: WorkspaceState, provider: ScriptedFakeProvider) -> Dream:
    return Dream(
        memory_manager=MemoryManager(state),
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )


def _memory_response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
        finish_reason="stop",
    )


def _service(state: WorkspaceState) -> ScheduleService:
    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> object:
        raise AssertionError("Dream must not run during registration")

    return ScheduleService(
        workspace_state=state,
        clock=_Clock(),
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )


@pytest.mark.asyncio
async def test_dream_registration_persists_a_hidden_recurring_system_job(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)

    service = _service(state)

    registered = await service.register_dream_job(
        schedule=JobSchedule.cron("0 * * * *", "Asia/Shanghai")
    )

    assert registered.job_id == "dream"
    assert registered.source == "system"
    assert registered.message
    assert registered.schedule == JobSchedule.cron("0 * * * *", "Asia/Shanghai")
    assert await service.public_snapshot() == ()
    assert await WorkspaceScheduleStore(state).snapshot() == (registered,)


@pytest.mark.asyncio
async def test_store_public_mutations_reject_the_dream_system_identity(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    service = _service(state)
    store = service._store
    await service.register_dream_job(schedule=JobSchedule.every(60))

    with pytest.raises(ValueError, match="canonical UUID4"):
        await store.commit_terminal(
            "dream",
            finished_at_ms=1,
            status="ok",
        )
    with pytest.raises(ValueError, match="canonical UUID4"):
        await store.remove_job("dream")


@pytest.mark.asyncio
async def test_exact_dream_registration_performs_zero_store_writes(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    schedule = JobSchedule.cron("0 * * * *", "Asia/Shanghai")
    existing = ScheduleJob(
        job_id="dream",
        source="system",
        message="Preserve this internal message.",
        schedule=schedule,
        created_at_ms=1,
        updated_at_ms=1,
    )
    state.schedule_path.write_text(
        json.dumps([existing.to_dict()], ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    writes: list[tuple[Path, str]] = []
    original = HOST_FILESYSTEM.atomic_replace_text

    def record_write(path: Path, content: str) -> None:
        writes.append((path, content))
        original(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_write)
    service = _service(state)

    registered = await service.register_dream_job(schedule=schedule)

    assert registered == existing
    assert writes == []
    assert (await WorkspaceScheduleStore(state).snapshot()) == (existing,)


@pytest.mark.asyncio
async def test_dream_registration_reconciles_cron_without_losing_job_history(
    workspace: Path,
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    existing = ScheduleJob(
        job_id="dream",
        source="system",
        message="Keep the persisted placeholder.",
        schedule=JobSchedule.cron("0 * * * *", "UTC"),
        state=ScheduleJobState(
            last_finished_at_ms=10,
            last_status="ok",
        ),
        created_at_ms=1,
        updated_at_ms=10,
    )
    state.schedule_path.write_text(
        json.dumps([existing.to_dict()], ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    writes: list[tuple[Path, str]] = []
    original = HOST_FILESYSTEM.atomic_replace_text

    def record_write(path: Path, content: str) -> None:
        writes.append((path, content))
        original(path, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_text", record_write)
    service = _service(state)
    replacement_schedule = JobSchedule.cron("30 * * * *", "UTC")

    registered = await service.register_dream_job(schedule=replacement_schedule)

    assert registered.job_id == existing.job_id
    assert registered.source == existing.source
    assert registered.message == existing.message
    assert registered.created_at_ms == existing.created_at_ms
    assert registered.state == existing.state
    assert registered.updated_at_ms >= existing.updated_at_ms
    assert registered.schedule == replacement_schedule
    assert len(writes) == 1
    assert await WorkspaceScheduleStore(state).snapshot() == (registered,)


@pytest.mark.asyncio
async def test_dream_reconcile_recomputes_next_cron_occurrence_for_timezone_and_dst(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    existing = ScheduleJob(
        job_id="dream",
        source="system",
        message="Keep this placeholder.",
        schedule=JobSchedule.cron("30 2 * * *", "UTC"),
        created_at_ms=1,
        updated_at_ms=1,
    )
    state.schedule_path.write_text(
        json.dumps([existing.to_dict()], ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    clock = _AdvancingClock(datetime(2026, 3, 8, 6, 0, tzinfo=UTC))
    calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal calls
        calls += 1
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    replacement_schedule = JobSchedule.cron("30 2 * * *", "America/New_York")
    await service.register_dream_job(schedule=replacement_schedule)

    service.start()
    await clock.wait_started.wait()
    clock.advance(24 * 60 * 60 + 30 * 60)
    for _ in range(100):
        if calls == 1:
            break
        await asyncio.sleep(0)

    assert calls == 1
    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.schedule == replacement_schedule
    assert persisted.message == existing.message
    await service.close()


@pytest.mark.parametrize(
    ("job_id", "source", "schedule"),
    [
        ("dream", "user", {"kind": "cron", "at_time": None, "every_seconds": None, "cron_expr": "0 * * * *", "timezone": "UTC"}),
        (
            "550e8400-e29b-41d4-a716-446655440000",
            "system",
            {"kind": "cron", "at_time": None, "every_seconds": None, "cron_expr": "0 * * * *", "timezone": "UTC"},
        ),
        ("unknown", "system", {"kind": "cron", "at_time": None, "every_seconds": None, "cron_expr": "0 * * * *", "timezone": "UTC"}),
        ("dream", "system", {"kind": "cron", "at_time": None, "every_seconds": None, "cron_expr": "0 * * * *", "timezone": "Not/A_Timezone"}),
        ("dream", "system", {"kind": "at", "at_time": "2026-08-07T12:00:00.000+00:00", "every_seconds": None, "cron_expr": None, "timezone": None}),
    ],
)
def test_schedule_store_rejects_corrupt_or_conflicting_system_state(
    workspace: Path,
    agent_home: Path,
    job_id: str,
    source: str,
    schedule: dict[str, object],
) -> None:
    state = _state(workspace, agent_home)
    state.schedule_path.write_text(
        json.dumps(
            [
                {
                    "job_id": job_id,
                    "source": source,
                    "message": "Internal Dream schedule.",
                    "schedule": schedule,
                    "state": {
                        "last_finished_at_ms": None,
                        "last_status": None,
                        "last_error": None,
                    },
                    "created_at_ms": 1,
                    "updated_at_ms": 1,
                }
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ScheduleStateError):
        WorkspaceScheduleStore(state)


def test_schedule_service_rejects_an_unsupported_internal_dispatch_identity(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    service = _service(state)
    invalid = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Must not dispatch.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    object.__setattr__(invalid, "source", "system")

    with pytest.raises(ScheduleStateError):
        service._reserve(invalid, current_monotonic=0.0)

    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0


@pytest.mark.asyncio
async def test_due_dream_job_dispatches_directly_without_user_or_session_execution(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0
    user_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        nonlocal user_calls
        del job
        user_calls += 1

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.cron("* * * * *", "UTC"))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    assert dream_calls == 1
    assert user_calls == 0
    assert await service.public_snapshot() == ()
    persisted = await WorkspaceScheduleStore(state).snapshot()
    assert len(persisted) == 1
    assert persisted[0].source == "system"
    assert persisted[0].state.last_status == "ok"
    assert not tuple(state.schedule_sessions_directory.glob("*.jsonl"))
    await service.close()


@pytest.mark.asyncio
async def test_memory_task_running_advances_one_dream_occurrence_without_terminal_write(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        return DreamResult(
            status="Memory Task is already running.",
            processed_count=0,
            memory_updated=False,
            cursor=0,
            error=ErrorInfo(
                code="memory_task_running",
                message="A Memory Task is already running.",
            ),
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(600)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    assert dream_calls == 1
    assert (await WorkspaceScheduleStore(state).snapshot())[0].state.last_status is None
    for _ in range(20):
        await asyncio.sleep(0)
    assert dream_calls == 1

    clock.advance(59)
    await asyncio.sleep(0)
    assert dream_calls == 1
    clock.advance(1)
    for _ in range(100):
        if dream_calls == 2:
            break
        await asyncio.sleep(0)
    assert dream_calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_dream_periodic_overlap_skips_one_occurrence_without_a_second_dream_call(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        started.set()
        await release.wait()
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await started.wait()
    clock.advance(60)
    for _ in range(100):
        await asyncio.sleep(0)

    assert dream_calls == 1
    assert service.status_snapshot().active_job_count == 1
    release.set()
    for _ in range(100):
        if service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)
    assert service.status_snapshot().active_job_count == 0

    clock.advance(60)
    for _ in range(100):
        if dream_calls == 2:
            break
        await asyncio.sleep(0)
    assert dream_calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_pause_and_drain_cancels_user_and_dream_then_resume_keeps_progress(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    user_calls = 0
    dream_calls = 0
    user_started = asyncio.Event()
    dream_started = asyncio.Event()
    user_cancelled = asyncio.Event()
    dream_cancelled = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        nonlocal user_calls
        del job
        user_calls += 1
        if user_calls == 1:
            user_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                user_cancelled.set()
                raise

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        if dream_calls == 1:
            dream_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                dream_cancelled.set()
                raise
        return DreamResult(
            status="No pending summaries",
            processed_count=0,
            memory_updated=False,
            cursor=0,
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    user_job = ScheduleJob(
        job_id="550e8400-e29b-41d4-a716-446655440000",
        message="Run a user occurrence.",
        schedule=JobSchedule.every(60),
        created_at_ms=1,
        updated_at_ms=1,
    )
    await service.add_user_job(user_job)
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await user_started.wait()
    await dream_started.wait()

    await service.pause_and_drain()

    assert service._paused is True
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    assert service.status_snapshot().active_job_count == 0
    assert user_cancelled.is_set()
    assert dream_cancelled.is_set()
    assert user_calls == 1
    assert dream_calls == 1
    assert all(
        job.state == ScheduleJobState()
        for job in await WorkspaceScheduleStore(state).snapshot()
    )

    clock.advance(600)
    for _ in range(20):
        await asyncio.sleep(0)
    assert user_calls == 1
    assert dream_calls == 1

    service.resume()
    for _ in range(100):
        if (
            user_calls == 2
            and dream_calls == 2
            and service.status_snapshot().active_job_count == 0
        ):
            break
        await asyncio.sleep(0)
    assert user_calls == 2
    assert dream_calls == 2
    assert all(
        job.state.last_status == "ok"
        for job in await WorkspaceScheduleStore(state).snapshot()
    )
    await service.close()


@pytest.mark.asyncio
async def test_cancelled_pause_and_drain_still_finishes_the_barrier(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    started = asyncio.Event()
    cancelled = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await cleanup_release.wait()
            raise
        raise AssertionError("blocked Dream executor returned without cancellation")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await started.wait()

    paused = asyncio.create_task(service.pause_and_drain())
    await asyncio.sleep(0)
    paused.cancel()
    await cancelled.wait()
    assert not paused.done()
    cleanup_release.set()
    with pytest.raises(asyncio.CancelledError):
        await paused

    assert service._paused is True
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    service.resume()
    await service.close()


@pytest.mark.asyncio
async def test_dream_error_persists_terminal_error_and_keeps_recurrence(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        return DreamResult(
            status="Memory failed",
            processed_count=0,
            memory_updated=False,
            cursor=0,
            error=ErrorInfo(code="model_failed", message="safe Dream failure"),
        )

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.state.last_status == "error"
    assert persisted.state.last_error == "safe Dream failure"
    assert dream_calls == 1
    clock.advance(59)
    await asyncio.sleep(0)
    assert dream_calls == 1
    clock.advance(1)
    for _ in range(100):
        if dream_calls == 2:
            break
        await asyncio.sleep(0)
    assert dream_calls == 2
    await service.close()


@pytest.mark.asyncio
async def test_dream_executor_exception_persists_safe_terminal_error(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        raise RuntimeError("PRIVATE_DREAM_EXCEPTION")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.state.last_status == "error"
    assert persisted.state.last_error == "Schedule Job execution failed."
    assert "PRIVATE_DREAM_EXCEPTION" not in persisted.state.last_error
    await service.close()


@pytest.mark.asyncio
async def test_dream_executor_cancellation_leaves_occurrence_pending_without_terminal_state(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    dream_calls = 0

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        nonlocal dream_calls
        dream_calls += 1
        raise asyncio.CancelledError()

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    for _ in range(100):
        if dream_calls == 1 and service.status_snapshot().active_job_count == 0:
            break
        await asyncio.sleep(0)

    persisted = (await WorkspaceScheduleStore(state).snapshot())[0]
    assert persisted.state == ScheduleJobState()
    assert dream_calls == 1
    await service.close()


@pytest.mark.asyncio
async def test_schedule_service_runs_dream_silently_without_a_foreground_session_log(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    manager = MemoryManager(state)
    await manager.append_summary("A pending summary.", datetime(2026, 8, 7, 12, 0, tzinfo=UTC))
    provider = ScriptedFakeProvider(completions=(_memory_response("No update needed."),))
    dream = Dream(
        memory_manager=manager,
        model_router=ScriptedFakeRouter(provider),
        batch_size=10,
        max_iterations=50,
    )
    clock = _AdvancingClock()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=dream.run,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))

    foreground_session_id = (
        "20260827-120000-000000_550e8400-e29b-41d4-a716-446655440000"
    )
    with session_log(state, foreground_session_id):
        service.start()
        await clock.wait_started.wait()
        clock.advance(60)
        for _ in range(100):
            if len(provider.complete_requests) == 1 and service.status_snapshot().active_job_count == 0:
                break
            await asyncio.sleep(0)

    await service.close()
    await dream.close()
    assert len(provider.complete_requests) == 1
    assert not (state.logs_directory / f"{foreground_session_id}.log").exists()
    assert not tuple(state.schedule_sessions_directory.glob("*.jsonl"))


class _BlockingDreamRouter:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.cancelled = asyncio.Event()

    async def complete(self, *args: object, **kwargs: object) -> ModelResponse:
        del args, kwargs
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        raise AssertionError("blocking Dream router completed unexpectedly")


@pytest.mark.asyncio
async def test_schedule_service_close_cancels_and_drains_an_active_dream_run(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    manager = MemoryManager(state)
    await manager.append_summary("A pending summary.", datetime(2026, 8, 7, 12, 0, tzinfo=UTC))
    router = _BlockingDreamRouter()
    dream = Dream(memory_manager=manager, model_router=router, batch_size=10, max_iterations=50)
    clock = _AdvancingClock()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=dream.run,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await router.started.wait()

    await service.close()
    assert router.cancelled.is_set()
    assert dream._task is None
    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
    await dream.close()


@pytest.mark.asyncio
async def test_schedule_service_abort_then_close_still_drains_owned_tasks(
    workspace: Path,
    agent_home: Path,
) -> None:
    state = _state(workspace, agent_home)
    clock = _AdvancingClock()
    started = asyncio.Event()

    async def execute_user_job(job: ScheduleJob) -> None:
        raise AssertionError(f"unexpected user Job: {job.job_id}")

    async def execute_dream() -> DreamResult:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("aborted Dream unexpectedly completed")

    service = ScheduleService(
        workspace_state=state,
        clock=clock,
        execute_user_job=execute_user_job,
        execute_dream=execute_dream,
    )
    await service.register_dream_job(schedule=JobSchedule.every(60))
    service.start()
    await clock.wait_started.wait()
    clock.advance(60)
    await started.wait()

    service.abort()
    await service.close()
    await service.abort_and_wait()

    assert service._loop_task is None
    assert service._run_tasks == set()
    assert service._terminal_commit_tasks == set()
