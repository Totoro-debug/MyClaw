import asyncio
import copy
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.session.session import Session, SessionStoragePartition
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

LOCAL_OFFSET = timezone(timedelta(hours=8))
CREATED_AT = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
UPDATED_AT = CREATED_AT + timedelta(seconds=5)
SESSION_ID = "20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION_ID = "20260711-153012-123000_6fa459ea-ee8a-4ca4-894e-db77e160355e"
SCHEDULE_JOB_ID = "6fa459ea-ee8a-4ca4-894e-db77e160355e"
ZERO_USAGE = {
    "model_calls": 0,
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
}


def _state(workspace: Path, agent_home: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=agent_home)
    return state


def _header(**updates: Any) -> dict[str, Any]:
    header: dict[str, Any] = {
        "session_id": SESSION_ID,
        "created_at": CREATED_AT.isoformat(timespec="milliseconds"),
        "updated_at": UPDATED_AT.isoformat(timespec="milliseconds"),
        "last_consolidated": 0,
        "metadata": {
            "title": "Project review",
            "token_usage": dict(ZERO_USAGE),
        },
    }
    header.update(updates)
    return header


def _write_jsonl(
    state: WorkspaceState,
    records: list[dict[str, Any]],
    *,
    trailing_newline: bool = True,
) -> Path:
    content = "\n".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) for record in records
    )
    if trailing_newline:
        content += "\n"
    path = state.sessions_directory / f"{SESSION_ID}.jsonl"
    path.write_text(content, encoding="utf-8")
    return path


def _assert_local_timestamp(value: Any) -> None:
    assert isinstance(value, str)
    parsed = datetime.fromisoformat(value)
    assert parsed.utcoffset() is not None
    assert parsed.utcoffset() == datetime.now().astimezone().utcoffset()


def test_create_starts_a_memory_only_session_with_private_identity_generation(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)

    session = Session.create(state)

    timestamp, uuid_text = session.session_id.split("_", maxsplit=1)
    assert timestamp == session.created_at.strftime("%Y%m%d-%H%M%S-%f")
    assert UUID(uuid_text).version == 4
    assert str(UUID(uuid_text)) == uuid_text
    assert session.created_at.utcoffset() is not None
    assert session.updated_at == session.created_at
    assert session.messages == []
    assert session.metadata == {
        "title": "Untitled session",
        "token_usage": ZERO_USAGE,
    }
    assert session.last_consolidated == 0
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()

    with pytest.raises(TypeError, match=r"Session\.create\(\) or Session\.load\(\)"):
        Session()


def test_create_schedule_session_uses_a_lazy_isolated_storage_partition(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)

    session = Session.create(
        state,
        partition=SessionStoragePartition.SCHEDULE,
        job_id=SCHEDULE_JOB_ID,
        now=lambda: CREATED_AT,
    )

    assert session.session_id == f"schedule_{SCHEDULE_JOB_ID}"
    assert session.storage_partition is SessionStoragePartition.SCHEDULE
    assert not state.schedule_sessions_directory.exists()

    session.add_message("user", "Run the scheduled task.")
    session.close()

    assert (state.schedule_sessions_directory / f"{session.session_id}.jsonl").exists()
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()
    loaded = Session.load(state, session.session_id)
    assert loaded.storage_partition is SessionStoragePartition.SCHEDULE
    assert loaded.messages == session.messages


@pytest.mark.parametrize(
    "job_id",
    [
        "6FA459EA-EE8A-4CA4-894E-DB77E160355E",
        "550e8400-e29b-11d4-a716-446655440000",
        "not-a-uuid",
    ],
)
def test_schedule_session_requires_a_canonical_uuid4_job_id(
    agent_home: Path,
    workspace: Path,
    job_id: str,
) -> None:
    state = _state(workspace, agent_home)

    with pytest.raises(ValueError, match="canonical UUID4"):
        Session.create(
            state,
            partition=SessionStoragePartition.SCHEDULE,
            job_id=job_id,
            now=lambda: CREATED_AT,
        )


@pytest.mark.asyncio
async def test_persist_writes_one_complete_compact_utf8_snapshot_atomically(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    timestamps = iter((CREATED_AT, CREATED_AT + timedelta(seconds=1), UPDATED_AT))
    session = Session.create(
        state,
        now=timestamps.__next__,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    session.add_message("user", "请读取 README。", extension={"nested": ["value"]})
    replacements: list[tuple[Path, bytes]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(target: Path, content: bytes) -> None:
        replacements.append((target, content))
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)

    session.persist()
    expected = (
        '{"session_id":"20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000",'
        '"created_at":"2026-07-11T15:30:12.123+08:00",'
        '"updated_at":"2026-07-11T15:30:17.123+08:00",'
        '"last_consolidated":0,'
        '"metadata":{"title":"Untitled session",'
        '"token_usage":{"model_calls":0,"input_tokens":0,'
        '"output_tokens":0,"total_tokens":0}}}\n'
        '{"role":"user","content":"请读取 README。",'
        '"timestamp":"2026-07-11T15:30:13.123+08:00",'
        '"extension":{"nested":["value"]}}\n'
    ).encode()

    assert replacements == []
    await asyncio.sleep(0)

    path = state.sessions_directory / f"{SESSION_ID}.jsonl"
    raw = HOST_FILESYSTEM.path_for_io(path).read_bytes()

    assert replacements == [(path, expected)]
    assert raw == expected
    assert b"\xe8\xaf\xb7\xe8\xaf\xbb" in raw
    assert Session.load(state, SESSION_ID).messages == session.messages


@pytest.mark.asyncio
async def test_persist_freezes_each_call_and_finishes_snapshots_in_call_order(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "Before mutation")
    snapshots: list[list[dict[str, Any]]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(_target: Path, content: bytes) -> None:
        records = [json.loads(line) for line in content.splitlines()]
        snapshots.append(records[1:])
        replace(_target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)

    session.persist()
    session.messages[0]["content"] = "After mutation"
    session.persist()
    await asyncio.sleep(0)

    assert [snapshot[0]["content"] for snapshot in snapshots] == [
        "Before mutation",
        "After mutation",
    ]
    assert Session.load(state, session.session_id).messages[0]["content"] == "After mutation"


@pytest.mark.asyncio
async def test_persist_retries_a_transient_write_with_async_backoff(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "Retry this snapshot")
    attempts: list[bytes] = []
    delays: list[float] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_twice_then_replace(target: Path, content: bytes) -> None:
        attempts.append(content)
        if len(attempts) < 3:
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def immediate_backoff(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_twice_then_replace)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", immediate_backoff)

    session.persist()
    await yield_once(0)

    assert len(attempts) == 3
    assert delays == [0.1, 0.2]
    assert Session.load(state, session.session_id).messages == session.messages


@pytest.mark.asyncio
async def test_persist_retries_each_snapshot_before_starting_the_next_snapshot(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "First snapshot")
    snapshots: list[tuple[str, ...]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_first_snapshot_twice(target: Path, content: bytes) -> None:
        records = [json.loads(line) for line in content.splitlines()]
        snapshots.append(tuple(record["content"] for record in records[1:]))
        if len(snapshots) <= 2:
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def immediate_backoff(_delay: float) -> None:
        return

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_first_snapshot_twice)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", immediate_backoff)

    session.persist()
    session.add_message("user", "Second snapshot")
    session.persist()
    await yield_once(0)

    assert snapshots == [
        ("First snapshot",),
        ("First snapshot",),
        ("First snapshot",),
        ("First snapshot", "Second snapshot"),
    ]
    assert Session.load(state, session.session_id).messages == session.messages


@pytest.mark.asyncio
async def test_abandon_cancels_every_pending_snapshot_when_latest_has_not_started(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "First snapshot")
    writes: list[bytes] = []
    backoff_started = asyncio.Event()
    backoff_cancelled = asyncio.Event()
    release_backoff = asyncio.Event()
    yield_once = asyncio.sleep

    def fail_first_write(_target: Path, content: bytes) -> None:
        writes.append(content)
        raise OSError("transient snapshot failure")

    async def blocked_backoff(_delay: float) -> None:
        backoff_started.set()
        try:
            await release_backoff.wait()
        except asyncio.CancelledError:
            backoff_cancelled.set()
            raise

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_first_write)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", blocked_backoff)

    session.persist()
    await yield_once(0)
    assert backoff_started.is_set()

    session.add_message("user", "Second snapshot")
    session.persist()
    session.abandon()
    await yield_once(0)

    try:
        assert not release_backoff.is_set()
        assert backoff_cancelled.is_set()
        assert len(writes) == 1
        assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()
    finally:
        release_backoff.set()
        await yield_once(0)

    assert len(writes) == 1
    session.abandon()


@pytest.mark.asyncio
async def test_close_wins_against_an_old_async_snapshot_in_backoff(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "Old snapshot")
    snapshots: list[tuple[str, ...]] = []
    backoff_started = asyncio.Event()
    release_backoff = asyncio.Event()
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_async_then_save_sync(target: Path, content: bytes) -> None:
        records = [json.loads(line) for line in content.splitlines()]
        snapshots.append(tuple(record["content"] for record in records[1:]))
        if len(snapshots) == 1:
            raise OSError("transient snapshot failure")
        replace(target, content)

    async def blocked_backoff(_delay: float) -> None:
        backoff_started.set()
        await release_backoff.wait()

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_async_then_save_sync)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", blocked_backoff)

    session.persist()
    await yield_once(0)
    assert backoff_started.is_set()

    session.add_message("user", "Final state")
    session.close()
    assert snapshots == [("Old snapshot",), ("Old snapshot", "Final state")]

    release_backoff.set()
    await yield_once(0)

    assert snapshots == [("Old snapshot",), ("Old snapshot", "Final state")]
    assert Session.load(state, session.session_id).messages == session.messages


@pytest.mark.asyncio
async def test_abandon_rejects_mutators_and_close_does_not_save(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "Before abandonment")
    original_messages = copy.deepcopy(session.messages)
    original_metadata = copy.deepcopy(session.metadata)

    session.abandon()

    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.add_message("user", "Rejected")
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.append_messages([{"role": "user", "content": "Rejected"}])
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.update_metadata(title="Rejected")
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.add_message(123, None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.append_messages(None)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.update_metadata("Rejected")  # type: ignore[arg-type]

    session.persist()
    session.close()

    assert session.messages == original_messages
    assert session.metadata == original_metadata
    assert not (state.sessions_directory / f"{session.session_id}.jsonl").exists()


def test_close_then_abandon_rejects_later_mutation_without_another_save(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "Saved before abandonment")
    session.close()
    path = state.sessions_directory / f"{session.session_id}.jsonl"
    saved = path.read_bytes()

    session.abandon()

    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.add_message("user", "Rejected")
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.append_messages([{"role": "user", "content": "Rejected"}])
    with pytest.raises(RuntimeError, match="Session has been abandoned"):
        session.update_metadata(title="Rejected")
    session.persist()
    session.close()

    assert path.read_bytes() == saved


@pytest.mark.asyncio
async def test_ordinary_persist_failure_is_silent_and_a_later_persist_is_independent(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "First attempt")
    attempts: list[bytes] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes
    yield_once = asyncio.sleep

    def fail_replace(_target: Path, content: bytes) -> None:
        attempts.append(content)
        raise OSError("simulated snapshot failure")

    async def immediate_backoff(_delay: float) -> None:
        return

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_replace)
    monkeypatch.setattr("myclaw.session.session.asyncio.sleep", immediate_backoff)
    session.persist()
    await yield_once(0)
    assert len(attempts) == 3

    replacements: list[bytes] = []

    def record_later_replace(target: Path, content: bytes) -> None:
        replacements.append(content)
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_later_replace)
    session.add_message("user", "Second attempt")
    session.persist()
    await yield_once(0)

    assert len(attempts) == 3
    assert len(replacements) == 1
    assert [message["content"] for message in Session.load(state, session.session_id).messages] == [
        "First attempt",
        "Second attempt",
    ]


@pytest.mark.asyncio
async def test_empty_session_persist_and_close_remain_unmaterialized(
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    session = Session.create(state)

    session.persist()
    session.close()
    await asyncio.sleep(0)

    assert not state.sessions_directory.exists()


def test_close_retries_latest_snapshot_with_bounded_delays(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "Save during shutdown")
    attempts: list[bytes] = []
    sleeps: list[float] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def fail_twice_then_replace(target: Path, content: bytes) -> None:
        attempts.append(content)
        if len(attempts) < 3:
            raise OSError("transient snapshot failure")
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_twice_then_replace)
    monkeypatch.setattr("myclaw.session.session.time.sleep", sleeps.append)

    session.close()

    assert len(attempts) == 3
    assert sleeps == [0.1, 0.2]
    assert Session.load(state, session.session_id).messages == session.messages


def test_close_swallows_failure_after_three_attempts(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "Best effort shutdown")
    attempts: list[bytes] = []
    sleeps: list[float] = []

    def fail_replace(_target: Path, content: bytes) -> None:
        attempts.append(content)
        raise OSError("permanent snapshot failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_replace)
    monkeypatch.setattr("myclaw.session.session.time.sleep", sleeps.append)

    session.close()

    assert len(attempts) == 3
    assert sleeps == [0.1, 0.2]


@pytest.mark.asyncio
async def test_close_supersedes_queued_persist_and_refreshes_each_attempt_timestamp(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    timestamps = iter(
        (
            CREATED_AT,
            datetime(2026, 7, 11, 15, 30, 13, 50000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 100000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 200000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 300000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 400000, tzinfo=LOCAL_OFFSET),
        )
    )
    session = Session.create(
        state,
        now=timestamps.__next__,
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    session.add_message("user", "Final state")
    replacements: list[dict[str, Any]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(target: Path, content: bytes) -> None:
        replacements.append(json.loads(content.splitlines()[0]))
        if len(replacements) < 3:
            raise OSError("transient snapshot failure")
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)
    monkeypatch.setattr("myclaw.session.session.time.sleep", lambda _delay: None)

    session.persist()
    session.close()
    await asyncio.sleep(0)

    assert [header["updated_at"] for header in replacements] == [
        "2026-07-11T15:30:13.200+08:00",
        "2026-07-11T15:30:13.300+08:00",
        "2026-07-11T15:30:13.400+08:00",
    ]
    assert Session.load(state, session.session_id).messages == session.messages


def test_session_identity_fields_are_read_only(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises(AttributeError):
        session.session_id = "another-session"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        session.created_at = CREATED_AT  # type: ignore[misc]
    with pytest.raises(AttributeError):
        session.updated_at = UPDATED_AT  # type: ignore[misc]


def test_public_state_is_directly_mutable_and_message_inputs_are_deep_copied(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    extension = {"nested": ["before"]}
    tool_calls = [{"id": "call-1", "name": "read_file", "arguments": '{"path":"README.md"}'}]
    usage = {
        "model_calls": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }

    session.add_message("user", "Inspect this project.", extension=extension)
    session.add_message(
        "assistant",
        "I will inspect it.",
        tool_calls=tool_calls,
        status="completed",
        error=None,
        token_usage=usage,
    )

    extension["nested"].append("after")
    tool_calls[0]["arguments"] = "changed"
    usage["input_tokens"] = 99
    session.metadata["future_key"] = {"enabled": True}
    session.last_consolidated = -1

    assert session.messages[0]["extension"] == {"nested": ["before"]}
    assert session.messages[1]["tool_calls"] == [
        {
            "id": "call-1",
            "name": "read_file",
            "arguments": '{"path":"README.md"}',
        }
    ]
    assert "id" not in session.messages[0]
    assert "id" not in session.messages[1]
    _assert_local_timestamp(session.messages[0]["timestamp"])
    _assert_local_timestamp(session.messages[1]["timestamp"])
    assert session.metadata["token_usage"] == {
        "model_calls": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }
    assert session.metadata["future_key"] == {"enabled": True}
    assert session.last_consolidated == -1


def test_append_messages_commits_a_valid_increment_in_order_with_timestamps_and_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home), now=lambda: CREATED_AT)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect this project."},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
        },
        {
            "role": "tool",
            "content": "README.md",
            "tool_call_id": "call-1",
            "name": "read_file",
            "status": "success",
            "artifact": {
                "path": ".myclaw/artifacts/session-1/call-1.txt",
                "total_chars": 123,
                "preview_chars": 80,
            },
            "confirmation": {"approved": True},
            "provider_extension": {"trace": [1, 2]},
        },
    ]

    session.append_messages(messages)

    assert [message["role"] for message in session.messages] == ["user", "assistant", "tool"]
    assert [message["content"] for message in session.messages] == [
        "Inspect this project.",
        "I will inspect it.",
        "README.md",
    ]
    assert all("timestamp" in message for message in session.messages)
    assert all(
        message["timestamp"] == CREATED_AT.isoformat(timespec="milliseconds")
        for message in session.messages
    )
    assert session.messages[2]["artifact"] == messages[2]["artifact"]
    assert session.messages[2]["confirmation"] == {"approved": True}
    assert session.metadata["token_usage"] == {
        "model_calls": 1,
        "input_tokens": 12,
        "output_tokens": 3,
        "total_tokens": 15,
    }


def test_append_messages_commits_blackboard_and_combined_usage_atomically(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": "Inspect this project."},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
        },
        {
            "role": "tool",
            "content": "README.md",
            "tool_call_id": "call-1",
            "name": "read_file",
            "status": "success",
        },
    ]
    metadata_updates: dict[str, Any] = {
        "blackboard": {
            "goal": "  Review the project  ",
            "completion_boundary": "  Review is complete  ",
        },
        "future": {"nested": ["value"]},
    }
    usage_delta = {
        "model_calls": 1,
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    original_messages = copy.deepcopy(messages)
    original_updates = copy.deepcopy(metadata_updates)

    session.append_messages(
        messages,
        metadata_updates=metadata_updates,
        usage_delta=usage_delta,
    )

    assert [message["role"] for message in session.messages] == ["user", "assistant", "tool"]
    assert session.metadata["blackboard"] == {
        "goal": "Review the project",
        "completion_boundary": "Review is complete",
    }
    assert session.metadata["future"] == {"nested": ["value"]}
    assert session.metadata["token_usage"] == {
        "model_calls": 2,
        "input_tokens": 17,
        "output_tokens": 5,
        "total_tokens": 22,
    }
    assert messages == original_messages
    assert metadata_updates == original_updates

    session.close()
    path = state.sessions_directory / f"{session.session_id}.jsonl"
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["metadata"]["blackboard"] == {
        "goal": "Review the project",
        "completion_boundary": "Review is complete",
    }


def test_append_messages_removes_blackboard_and_omits_it_from_round_trip(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state, now=lambda: CREATED_AT)
    session.append_messages(
        [{"role": "user", "content": "Start the review."}],
        metadata_updates={
            "blackboard": {
                "goal": "Review the project",
                "completion_boundary": "Review is complete",
            }
        },
    )
    session.append_messages(
        [{"role": "user", "content": "Cancel the review."}],
        metadata_removals=("blackboard",),
    )

    session.close()
    path = state.sessions_directory / f"{session.session_id}.jsonl"
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])

    assert "blackboard" not in header["metadata"]
    assert "blackboard" not in Session.load(state, session.session_id).metadata


def test_append_messages_preserves_latest_title_and_usage_before_accumulating(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.update_metadata(
        title="Generated title",
        usage_delta={
            "model_calls": 2,
            "input_tokens": 8,
            "output_tokens": 4,
            "total_tokens": 12,
        },
    )

    session.append_messages(
        [{"role": "user", "content": "Continue the review."}],
        metadata_updates={
            "blackboard": {
                "goal": "Review the project",
                "completion_boundary": "Review is complete",
            }
        },
        usage_delta={
            "model_calls": 1,
            "input_tokens": 5,
            "output_tokens": 2,
            "total_tokens": 7,
        },
    )

    assert session.metadata["title"] == "Generated title"
    assert session.metadata["token_usage"] == {
        "model_calls": 3,
        "input_tokens": 13,
        "output_tokens": 6,
        "total_tokens": 19,
    }


@pytest.mark.parametrize(
    ("metadata_updates", "metadata_removals", "match"),
    [
        (
            {"blackboard": {"goal": "Goal", "completion_boundary": "Boundary"}},
            ("blackboard",),
            "same key",
        ),
        ({}, ("title",), "required"),
        ({}, ("token_usage",), "required"),
        (
            {
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                }
            },
            (),
            "token usage",
        ),
        # update_metadata already reserves these names as token usage aliases.
        ({"token_usage_delta": {}}, (), "token usage"),
        ({"usage_delta": {}}, (), "token usage"),
    ],
)
def test_append_messages_rejects_conflicting_required_and_usage_metadata_patches(
    agent_home: Path,
    workspace: Path,
    metadata_updates: dict[str, Any],
    metadata_removals: tuple[str, ...],
    match: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    original_updates = copy.deepcopy(metadata_updates)

    with pytest.raises(ValueError, match=match):
        session.append_messages(
            [{"role": "user", "content": "Rejected patch."}],
            metadata_updates=metadata_updates,
            metadata_removals=metadata_removals,
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert metadata_updates == original_updates


@pytest.mark.parametrize(
    ("messages", "metadata_updates", "usage_delta", "match"),
    [
        (
            [
                {"role": "user", "content": "Before the invalid record."},
                {"role": "assistant", "content": "Missing durable fields."},
            ],
            {"future": {"nested": ["value"]}},
            {"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            "assistant message is missing",
        ),
        (
            [{"role": "user", "content": "Invalid Blackboard."}],
            {"blackboard": {"goal": "", "completion_boundary": "Boundary"}},
            None,
            "metadata.blackboard",
        ),
        (
            [{"role": "user", "content": "Invalid usage."}],
            {"future": {"nested": ["value"]}},
            {"model_calls": True, "input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
            "usage_delta",
        ),
        (
            [{"role": "user", "content": "Invalid usage total."}],
            {"future": {"nested": ["value"]}},
            {"model_calls": 1, "input_tokens": 2, "output_tokens": 1, "total_tokens": 99},
            "usage_delta",
        ),
        (
            [{"role": "user", "content": "Invalid extension."}],
            {"future": {"invalid": {1, 2}}},
            None,
            "metadata_updates",
        ),
    ],
)
def test_append_messages_leaves_session_and_inputs_unchanged_on_candidate_failure(
    agent_home: Path,
    workspace: Path,
    messages: list[dict[str, Any]],
    metadata_updates: dict[str, Any],
    usage_delta: dict[str, Any] | None,
    match: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.add_message("user", "Existing history")
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    original_messages = copy.deepcopy(messages)
    original_updates = copy.deepcopy(metadata_updates)
    original_usage = copy.deepcopy(usage_delta)

    with pytest.raises((TypeError, ValueError), match=match):
        session.append_messages(
            messages,
            metadata_updates=metadata_updates,
            usage_delta=usage_delta,
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert messages == original_messages
    assert metadata_updates == original_updates
    assert usage_delta == original_usage


def test_append_messages_does_not_commit_before_metadata_comparison_finishes(
    agent_home: Path,
    workspace: Path,
) -> None:
    class EqualityFailure(str):
        def __eq__(self, other: object) -> bool:
            del other
            raise RuntimeError("metadata comparison failed")

        __hash__ = str.__hash__

    session = Session.create(_state(workspace, agent_home))
    session.update_metadata(unstable=EqualityFailure("stored"))
    stored_value = session.metadata["unstable"]
    metadata_container = session.metadata
    messages_container = session.messages

    # Keep the hostile extension first so equality is exercised before changed usage.
    title = session.metadata["title"]
    token_usage = session.metadata["token_usage"]
    session.metadata.clear()
    session.metadata.update(
        unstable=stored_value,
        title=title,
        token_usage=token_usage,
    )
    before_keys = tuple(session.metadata)
    before_title = session.metadata["title"]
    before_usage = copy.deepcopy(session.metadata["token_usage"])

    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Prepared but not committed.",
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 2,
                "output_tokens": 1,
                "total_tokens": 3,
            },
        }
    ]
    metadata_updates = {"unstable": EqualityFailure("candidate")}
    usage_delta = {
        "model_calls": 1,
        "input_tokens": 5,
        "output_tokens": 2,
        "total_tokens": 7,
    }
    original_messages = copy.deepcopy(messages)
    update_value = metadata_updates["unstable"]
    original_usage_delta = copy.deepcopy(usage_delta)

    with pytest.raises(RuntimeError, match="metadata comparison failed"):
        session.append_messages(
            messages,
            metadata_updates=metadata_updates,
            usage_delta=usage_delta,
        )

    assert session.messages is messages_container
    assert len(session.messages) == 0
    assert session.metadata is metadata_container
    assert tuple(session.metadata) == before_keys
    assert session.metadata["title"] == before_title
    assert session.metadata["token_usage"] == before_usage
    assert session.metadata["unstable"] is stored_value
    assert messages == original_messages
    assert metadata_updates["unstable"] is update_value
    assert usage_delta == original_usage_delta


@pytest.mark.parametrize(
    ("metadata_updates", "metadata_removals", "usage_delta", "match"),
    [
        ([], (), None, "metadata_updates"),
        ({1: "invalid"}, (), None, "metadata_updates"),
        ({"future": {"enabled": True}}, [], None, "metadata_removals"),
        ({"future": {"enabled": True}}, (1,), None, "metadata_removals"),
        ({"future": {"enabled": True}}, (), [], "usage_delta"),
    ],
)
def test_append_messages_validates_optional_argument_containers_before_use(
    agent_home: Path,
    workspace: Path,
    metadata_updates: object,
    metadata_removals: object,
    usage_delta: object,
    match: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)

    with pytest.raises((TypeError, ValueError), match=match):
        session.append_messages(
            [{"role": "user", "content": "Invalid optional arguments."}],
            metadata_updates=metadata_updates,  # type: ignore[arg-type]
            metadata_removals=metadata_removals,  # type: ignore[arg-type]
            usage_delta=usage_delta,  # type: ignore[arg-type]
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata


def test_update_metadata_rejects_malformed_blackboard_without_mutating_state_or_input(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    patch = {
        "blackboard": {"goal": "", "completion_boundary": "Boundary"},
        "future": {"nested": ["value"]},
    }
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)
    original_patch = copy.deepcopy(patch)

    with pytest.raises(ValueError, match=r"metadata\.blackboard"):
        session.update_metadata(patch)

    assert session.messages == before_messages
    assert session.metadata == before_metadata
    assert patch == original_patch


def test_append_messages_leaves_state_unchanged_when_a_middle_message_is_invalid(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.add_message("user", "Existing history")
    before_messages = copy.deepcopy(session.messages)
    before_usage = copy.deepcopy(session.metadata["token_usage"])

    with pytest.raises(ValueError, match="assistant message is missing"):
        session.append_messages(
            [
                {"role": "user", "content": "Before the invalid record."},
                {"role": "assistant", "content": "Missing durable fields."},
                {"role": "user", "content": "After the invalid record."},
            ]
        )

    assert session.messages == before_messages
    assert session.metadata["token_usage"] == before_usage


def test_append_messages_leaves_state_unchanged_when_the_final_message_is_invalid(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.add_message("user", "Existing history")
    before_messages = copy.deepcopy(session.messages)
    before_usage = copy.deepcopy(session.metadata["token_usage"])

    with pytest.raises(ValueError, match="tool_call_id"):
        session.append_messages(
            [
                {
                    "role": "assistant",
                    "content": "A valid assistant record.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                },
                {
                    "role": "tool",
                    "content": "Invalid result.",
                    "name": "read_file",
                    "status": "error",
                },
            ]
        )

    assert session.messages == before_messages
    assert session.metadata["token_usage"] == before_usage


def test_append_messages_isolated_from_nested_caller_mutations(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    messages: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
            "provider_extension": {"trace": ["before"]},
        },
        {
            "role": "tool",
            "content": "README.md",
            "tool_call_id": "call-1",
            "name": "read_file",
            "status": "success",
            "artifact": {
                "path": ".myclaw/artifacts/session-1/call-1.txt",
                "total_chars": 123,
                "preview_chars": 80,
            },
            "confirmation": {"approved": True, "details": {"source": "user"}},
        },
    ]

    session.append_messages(messages)

    messages[0]["tool_calls"][0]["arguments"] = '{"path":"changed"}'
    messages[0]["provider_extension"]["trace"].append("after")
    messages[0]["token_usage"]["input_tokens"] = 99
    messages[1]["artifact"]["total_chars"] = 999
    messages[1]["confirmation"]["details"]["source"] = "changed"

    assert session.messages[0]["tool_calls"] == [
        {"id": "call-1", "name": "read_file", "arguments": "{}"}
    ]
    assert session.messages[0]["provider_extension"] == {"trace": ["before"]}
    assert session.messages[0]["token_usage"]["input_tokens"] == 12
    assert session.messages[1]["artifact"]["total_chars"] == 123
    assert session.messages[1]["confirmation"] == {
        "approved": True,
        "details": {"source": "user"},
    }


@pytest.mark.parametrize(
    "usage",
    [
        {
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 99,
        },
        {
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 1,
        },
        {
            "model_calls": 1,
            "input_tokens": 2,
            "output_tokens": 1,
            "total_tokens": 3,
            "cached_tokens": 0,
        },
    ],
)
def test_append_messages_rejects_usage_shape_errors_without_state_changes(
    agent_home: Path,
    workspace: Path,
    usage: dict[str, int],
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.add_message("user", "Existing history")
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)

    with pytest.raises(ValueError, match="token"):
        session.append_messages(
            [
                {
                    "role": "assistant",
                    "content": "Invalid usage.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": usage,
                }
            ]
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata


def test_append_messages_accumulates_multiple_assistants_once_beyond_64_bit_range(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    signed_64_max = 2**63 - 1
    session.update_metadata(
        token_usage_delta={
            "model_calls": 7,
            "input_tokens": signed_64_max - 2,
            "output_tokens": 1,
            "total_tokens": signed_64_max - 1,
        }
    )

    session.append_messages(
        [
            {
                "role": "assistant",
                "content": "First response.",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 2,
                    "output_tokens": 1,
                    "total_tokens": 3,
                },
            },
            {
                "role": "assistant",
                "content": "Second response.",
                "tool_calls": [],
                "status": "completed",
                "error": None,
                "token_usage": {
                    "model_calls": 1,
                    "input_tokens": 4,
                    "output_tokens": 2,
                    "total_tokens": 6,
                },
            },
        ]
    )

    assert session.metadata["token_usage"] == {
        "model_calls": 9,
        "input_tokens": signed_64_max + 4,
        "output_tokens": 4,
        "total_tokens": signed_64_max + 8,
    }


def test_append_messages_rejects_invalid_existing_usage_without_state_changes(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    session.metadata["token_usage"] = {
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": "invalid",
    }
    before_messages = copy.deepcopy(session.messages)
    before_metadata = copy.deepcopy(session.metadata)

    with pytest.raises(ValueError, match=r"metadata\.token_usage"):
        session.append_messages(
            [
                {
                    "role": "assistant",
                    "content": "A valid message.",
                    "tool_calls": [],
                    "status": "completed",
                    "error": None,
                    "token_usage": {
                        "model_calls": 1,
                        "input_tokens": 2,
                        "output_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            ]
        )

    assert session.messages == before_messages
    assert session.metadata == before_metadata


def test_tool_message_preserves_provider_fields_and_unknown_extensions(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    session.add_message(
        "tool",
        "README.md",
        tool_call_id="call-1",
        name="read_file",
        status="success",
        artifact={
            "path": ".myclaw/artifacts/session-1/call-1.txt",
            "total_chars": 123,
            "preview_chars": 80,
        },
        provider_extension={"trace": [1, 2]},
    )

    message = session.messages[0]
    assert {key: value for key, value in message.items() if key != "timestamp"} == {
        "role": "tool",
        "content": "README.md",
        "tool_call_id": "call-1",
        "name": "read_file",
        "status": "success",
        "artifact": {
            "path": ".myclaw/artifacts/session-1/call-1.txt",
            "total_chars": 123,
            "preview_chars": 80,
        },
        "provider_extension": {"trace": [1, 2]},
    }
    _assert_local_timestamp(message["timestamp"])


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        ("\n  \u300c  Project\t review  \u300d\nIgnore this line", "Project review"),
        ("\n\t\t", "Untitled session"),
        ('"  Quoted title  "', "Quoted title"),
        ("\u754c" * 70, "\u754c" * 60),
    ],
)
def test_update_metadata_normalizes_title_shallow_merges_and_accumulates_usage(
    agent_home: Path,
    workspace: Path,
    candidate: str,
    expected: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    extra = {"nested": ["before"]}

    session.update_metadata(
        title=candidate,
        future=extra,
        token_usage_delta={
            "model_calls": 2,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
    )
    extra["nested"].append("after")

    assert session.metadata == {
        "title": expected,
        "token_usage": {
            "model_calls": 2,
            "input_tokens": 20,
            "output_tokens": 5,
            "total_tokens": 25,
        },
        "future": {"nested": ["before"]},
    }


@pytest.mark.parametrize("value", [object(), {"bad": object()}])
def test_mutation_helpers_reject_non_json_values(
    agent_home: Path,
    workspace: Path,
    value: object,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises((TypeError, ValueError), match="JSON"):
        session.add_message("user", "Hello", extension=value)
    with pytest.raises((TypeError, ValueError), match="JSON"):
        session.update_metadata(future=value)


def test_known_message_contracts_and_unsupported_legacy_fields_are_validated(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))

    with pytest.raises(ValueError, match="role"):
        session.add_message("system", "Unsupported")
    with pytest.raises(ValueError, match="reserved"):
        session.add_message("user", "Hello", timestamp="override")
    with pytest.raises(ValueError, match="unsupported"):
        session.add_message("user", "Hello", id="legacy-message-id")
    with pytest.raises(ValueError, match="status"):
        session.add_message(
            "assistant",
            "Answer",
            tool_calls=[],
            status="unknown",
            error=None,
            token_usage={
                "model_calls": 1,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            },
        )
    with pytest.raises(ValueError, match="tool_call_id"):
        session.add_message("tool", "Result", name="read_file", status="success")


@pytest.mark.parametrize("missing", ["tool_calls", "status", "error", "token_usage"])
def test_assistant_message_requires_every_provider_relevant_field(
    agent_home: Path,
    workspace: Path,
    missing: str,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    fields: dict[str, Any] = {
        "tool_calls": [],
        "status": "completed",
        "error": None,
        "token_usage": {
            "model_calls": 1,
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
        },
    }
    del fields[missing]

    with pytest.raises(ValueError, match=missing):
        session.add_message("assistant", "Answer", **fields)


def test_assistant_status_error_content_and_model_call_contract_remains_coherent(
    agent_home: Path,
    workspace: Path,
) -> None:
    session = Session.create(_state(workspace, agent_home))
    usage = {"model_calls": 1, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    with pytest.raises(ValueError, match="completed assistant"):
        session.add_message(
            "assistant",
            "Answer",
            tool_calls=[],
            status="completed",
            error={"code": "model_failed", "message": "failed"},
            token_usage=usage,
        )
    with pytest.raises(ValueError, match="non-completed assistant"):
        session.add_message(
            "assistant",
            "Partial",
            tool_calls=[],
            status="interrupted",
            error=None,
            token_usage=usage,
        )
    with pytest.raises(ValueError, match="content or tool_calls"):
        session.add_message(
            "assistant",
            "",
            tool_calls=[],
            status="interrupted",
            error={"code": "turn_cancelled", "message": "interrupted"},
            token_usage=usage,
        )
    with pytest.raises(ValueError, match="model_calls"):
        session.add_message(
            "assistant",
            "Answer",
            tool_calls=[],
            status="completed",
            error=None,
            token_usage=dict(ZERO_USAGE),
        )
    with pytest.raises(ValueError, match="model_calls"):
        session.add_message(
            "assistant",
            "Failed",
            tool_calls=[],
            status="error",
            error={"code": "model_failed", "message": "failed"},
            token_usage=dict(ZERO_USAGE),
        )

    session.add_message(
        "assistant",
        "Iteration limit reached",
        tool_calls=[],
        status="error",
        error={"code": "agent_iteration_limit", "message": "limit reached"},
        token_usage=dict(ZERO_USAGE),
    )

    assert session.messages[-1]["error"]["code"] == "agent_iteration_limit"


def test_load_current_five_field_jsonl_preserves_json_native_extensions(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    header = _header(
        last_consolidated=2,
        metadata={
            "title": "Project review",
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
            "future": {"enabled": True},
        },
    )
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Inspect this project.",
            "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            "provider_extension": {"trace": [1, 2]},
        },
        {
            "role": "assistant",
            "content": "Done.",
            "timestamp": UPDATED_AT.isoformat(timespec="milliseconds"),
            "tool_calls": [],
            "status": "completed",
            "error": None,
            "token_usage": {
                "model_calls": 1,
                "input_tokens": 12,
                "output_tokens": 3,
                "total_tokens": 15,
            },
        },
    ]
    _write_jsonl(state, [header, *messages])

    loaded = Session.load(state, SESSION_ID)

    assert loaded.session_id == SESSION_ID
    assert loaded.created_at == CREATED_AT
    assert loaded.updated_at == UPDATED_AT
    assert loaded.last_consolidated == 2
    assert loaded.metadata == header["metadata"]
    assert loaded.messages == messages


def test_load_canonicalizes_a_valid_blackboard_metadata_value(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    header = _header(
        metadata={
            "title": "Project review",
            "token_usage": dict(ZERO_USAGE),
            "blackboard": {
                "goal": "  Review the project  ",
                "completion_boundary": "  Review is complete  ",
            },
        }
    )
    _write_jsonl(state, [header])

    loaded = Session.load(state, SESSION_ID)

    assert loaded.metadata["blackboard"] == {
        "goal": "Review the project",
        "completion_boundary": "Review is complete",
    }


@pytest.mark.parametrize(
    "blackboard",
    [
        None,
        "not an object",
        {"goal": "goal"},
        {"completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": "boundary", "extra": True},
        {"goal": 1, "completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": False},
        {"goal": "", "completion_boundary": "boundary"},
        {"goal": "   ", "completion_boundary": "boundary"},
        {"goal": "goal", "completion_boundary": ""},
        {"goal": "goal", "completion_boundary": "\t\n"},
    ],
)
def test_load_treats_malformed_blackboard_metadata_as_absent(
    agent_home: Path,
    workspace: Path,
    blackboard: object,
) -> None:
    state = _state(workspace, agent_home)
    header = _header(
        metadata={
            "title": "Project review",
            "token_usage": dict(ZERO_USAGE),
            "blackboard": blackboard,
        }
    )
    _write_jsonl(state, [header])

    loaded = Session.load(state, SESSION_ID)

    assert "blackboard" not in loaded.metadata


@pytest.mark.parametrize(
    "records",
    [
        [_header(extra=True)],
        [_header(session_id=OTHER_SESSION_ID)],
        [_header(created_at="2026-07-11T15:30:12")],
        [_header(last_consolidated=-1)],
        [_header(metadata={"title": "Project review", "token_usage": {}})],
        [
            _header(),
            {
                "role": "assistant",
                "content": "Missing known fields",
                "timestamp": CREATED_AT.isoformat(timespec="milliseconds"),
            },
        ],
    ],
)
def test_load_rejects_malformed_core_or_unsupported_message_shapes(
    agent_home: Path,
    workspace: Path,
    records: list[dict[str, Any]],
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(state, records)

    with pytest.raises(ValueError):
        Session.load(state, SESSION_ID)


def test_load_rejects_jsonl_without_a_trailing_newline(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(state, [_header()], trailing_newline=False)

    with pytest.raises(ValueError, match="newline"):
        Session.load(state, SESSION_ID)
