import asyncio
import json
from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.schedule.model import JobSchedule, ScheduleJob, ScheduleJobState
from myclaw.schedule.store import (
    ScheduleStateError,
    ScheduleStoreFaultedError,
    WorkspaceScheduleStore,
)

JOB_ID = "550e8400-e29b-41d4-a716-446655440000"
SYSTEM_ID = "6fa459ea-ee8a-4ca4-894e-db77e160355e"


def _state(path: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(path))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _job(
    job_id: str = JOB_ID,
    *,
    source: str = "user",
    message: str = "Run this.",
    state: ScheduleJobState | None = None,
) -> ScheduleJob:
    return ScheduleJob(
        job_id=job_id,
        source=source,  # type: ignore[arg-type]
        message=message,
        schedule=JobSchedule(kind="every", every_seconds=60),
        state=ScheduleJobState() if state is None else state,
        created_at_ms=10,
        updated_at_ms=10,
    )


@pytest.mark.asyncio
async def test_missing_schedule_state_is_empty_until_first_mutation(workspace: Path) -> None:
    state = _state(workspace)
    store = WorkspaceScheduleStore(state)

    assert await store.snapshot() == ()
    assert not store.path.exists()

    await store.add_user_job(_job())

    assert await store.snapshot() == (_job(),)
    assert store.path.read_text(encoding="utf-8") == json.dumps(
        [_job().to_dict()], ensure_ascii=False, separators=(",", ":")
    )


@pytest.mark.asyncio
async def test_snapshot_is_immutable_and_public_snapshot_hides_system_jobs(
    workspace: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace))
    await store.add_user_job(_job())
    await store.add_system_job(_job(SYSTEM_ID, source="system", message="Internal run."))

    snapshot = await store.snapshot()
    public = await store.public_snapshot()
    assert [job.job_id for job in snapshot] == [JOB_ID, SYSTEM_ID]
    assert [job.job_id for job in public] == [JOB_ID]

    snapshot_dict = snapshot[0].to_dict()
    snapshot_dict["message"] = "mutated"
    snapshot_dict["schedule"]["every_seconds"] = 999  # type: ignore[index]
    assert (await store.snapshot())[0].message == "Run this."
    assert (await store.snapshot())[0].schedule.every_seconds == 60
    assert [job.job_id for job in public] == sorted([JOB_ID])


@pytest.mark.asyncio
async def test_final_user_removal_persists_an_empty_array(workspace: Path) -> None:
    store = WorkspaceScheduleStore(_state(workspace))
    await store.add_user_job(_job())

    assert await store.remove_user_job(JOB_ID, expected=_job()) is True
    assert await store.snapshot() == ()
    assert store.path.read_text(encoding="utf-8") == "[]"


@pytest.mark.asyncio
async def test_successful_mutations_increment_revision_and_wake_waiters(workspace: Path) -> None:
    store = WorkspaceScheduleStore(_state(workspace))
    revision = store.revision
    waiter = asyncio.create_task(store.wait_for_change(revision))
    await asyncio.sleep(0)

    await store.add_user_job(_job())

    assert await asyncio.wait_for(waiter, timeout=1) == revision + 1
    assert store.revision == revision + 1


@pytest.mark.asyncio
async def test_terminal_commit_updates_state_without_exposing_mutable_authority(
    workspace: Path,
) -> None:
    store = WorkspaceScheduleStore(_state(workspace))
    await store.add_user_job(_job())

    committed = await store.commit_terminal(
        JOB_ID,
        finished_at_ms=20,
        status="error",
        error="The run failed.",
        now_ms=25,
    )

    assert committed is not None
    assert committed.state == ScheduleJobState(
        last_finished_at_ms=20,
        last_status="error",
        last_error="The run failed.",
    )
    assert committed.updated_at_ms == 25
    assert (await store.snapshot())[0] == committed


@pytest.mark.asyncio
async def test_write_failure_keeps_old_snapshot_and_latches_fault(
    workspace: Path,
) -> None:
    state = _state(workspace)

    def fail_replace(path: Path, content: str) -> None:
        del path, content
        raise OSError("injected replacement failure")

    store = WorkspaceScheduleStore(state, replace_text=fail_replace)
    revision = store.revision
    waiter = asyncio.create_task(store.wait_for_change(revision))
    await asyncio.sleep(0)

    with pytest.raises(OSError, match="injected replacement failure"):
        await store.add_user_job(_job())

    assert await store.snapshot() == ()
    assert await asyncio.wait_for(waiter, timeout=1) == revision
    assert store.revision == revision
    assert store.health == "faulted"
    with pytest.raises(ScheduleStoreFaultedError):
        await store.add_system_job(_job(SYSTEM_ID, source="system"))


@pytest.mark.asyncio
async def test_write_failure_leaves_the_last_complete_document_for_restart(
    workspace: Path,
) -> None:
    state = _state(workspace)
    initial = _job()
    healthy = WorkspaceScheduleStore(state)
    await healthy.add_user_job(initial)
    document_before_failure = state.schedule_path.read_bytes()

    def fail_replace(path: Path, content: str) -> None:
        del path, content
        raise OSError("injected replacement failure")

    failing = WorkspaceScheduleStore(state, replace_text=fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        await failing.commit_terminal(
            initial.job_id,
            finished_at_ms=20,
            status="ok",
            now_ms=20,
        )

    assert state.schedule_path.read_bytes() == document_before_failure
    restarted = WorkspaceScheduleStore(state)
    assert await restarted.snapshot() == (initial,)


@pytest.mark.asyncio
async def test_public_removal_treats_a_system_job_as_missing(workspace: Path) -> None:
    store = WorkspaceScheduleStore(_state(workspace))
    system_job = _job(SYSTEM_ID, source="system", message="Internal run.")
    await store.add_system_job(system_job)
    revision = store.revision

    assert await store.remove_user_job(SYSTEM_ID) is False
    assert await store.snapshot() == (system_job,)
    assert await store.public_snapshot() == ()
    assert store.revision == revision


def test_strict_load_rejects_duplicate_keys_and_duplicate_job_ids(workspace: Path) -> None:
    state = _state(workspace)
    duplicate_key = (
        '[{"job_id":"550e8400-e29b-41d4-a716-446655440000",'
        '"job_id":"6fa459ea-ee8a-4ca4-894e-db77e160355e"}]'
    )
    state.schedule_path.write_text(duplicate_key, encoding="utf-8")

    with pytest.raises(ScheduleStateError) as raised:
        WorkspaceScheduleStore(state)
    assert raised.value.path == state.schedule_path
    assert raised.value.error.to_dict() == {
        "code": "schedule_state_error",
        "message": (
            "Schedule state could not be loaded. Repair or move the file, then start MyClaw again."
        ),
        "retryable": False,
        "retry_after_seconds": None,
    }

    state.schedule_path.write_text(
        json.dumps([_job().to_dict(), _job().to_dict()], separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(Exception, match="Schedule state"):
        WorkspaceScheduleStore(state)


def test_strict_load_rejects_duplicate_nested_keys(workspace: Path) -> None:
    state = _state(workspace)
    state.schedule_path.write_text(
        '[{"job_id":"550e8400-e29b-41d4-a716-446655440000",'
        '"source":"user","message":"Run this.",'
        '"schedule":{"kind":"every","at_time":null,"at_time":null,'
        '"every_seconds":60,"cron_expr":null,"timezone":null},'
        '"state":{"last_finished_at_ms":null,"last_status":null,"last_error":null},'
        '"created_at_ms":10,"updated_at_ms":10}]',
        encoding="utf-8",
    )

    with pytest.raises(Exception, match="Schedule state"):
        WorkspaceScheduleStore(state)


@pytest.mark.parametrize(
    "content",
    [
        "",
        "{}",
        "null",
        "[[]]",
        "[" * 5_000 + "]" * 5_000,
    ],
)
def test_strict_load_maps_invalid_json_and_root_shapes_to_state_error(
    workspace: Path,
    content: str,
) -> None:
    state = _state(workspace)
    state.schedule_path.write_text(content, encoding="utf-8")

    with pytest.raises(ScheduleStateError) as raised:
        WorkspaceScheduleStore(state)

    assert raised.value.path == state.schedule_path
    assert "Schedule state could not be loaded" in str(raised.value)


def test_strict_load_rejects_invalid_utf8(workspace: Path) -> None:
    state = _state(workspace)
    state.schedule_path.write_bytes(b"\xff")

    with pytest.raises(ScheduleStateError):
        WorkspaceScheduleStore(state)


def test_strict_load_rejects_never_run_job_with_changed_timestamp(workspace: Path) -> None:
    state = _state(workspace)
    document = _job().to_dict()
    document["updated_at_ms"] = 11
    state.schedule_path.write_text(json.dumps([document], separators=(",", ":")), encoding="utf-8")

    with pytest.raises(ScheduleStateError):
        WorkspaceScheduleStore(state)


@pytest.mark.asyncio
async def test_strict_load_accepts_the_canonical_empty_array(workspace: Path) -> None:
    state = _state(workspace)
    state.schedule_path.write_text("[]", encoding="utf-8")

    store = WorkspaceScheduleStore(state)

    assert await store.snapshot() == ()
    assert store.revision == 0


@pytest.mark.asyncio
async def test_strict_load_is_one_time_and_disk_changes_do_not_replace_authority(
    workspace: Path,
) -> None:
    state = _state(workspace)
    state.schedule_path.write_text(
        json.dumps([_job().to_dict()], separators=(",", ":")),
        encoding="utf-8",
    )
    store = WorkspaceScheduleStore(state)
    state.schedule_path.write_text(
        json.dumps([_job(SYSTEM_ID, source="system").to_dict()], separators=(",", ":")),
        encoding="utf-8",
    )

    assert [job.job_id for job in await store.snapshot()] == [JOB_ID]


@pytest.mark.parametrize("kind", ["directory", "hardlink"])
def test_strict_load_rejects_unsafe_schedule_paths(workspace: Path, kind: str) -> None:
    state = _state(workspace)
    outside = workspace.parent / f"outside-schedule-{kind}.json"
    if kind == "directory":
        state.schedule_path.mkdir()
    else:
        outside.write_text("[]", encoding="utf-8")
        state.schedule_path.hardlink_to(outside)

    with pytest.raises(Exception, match="Schedule state"):
        WorkspaceScheduleStore(state)
