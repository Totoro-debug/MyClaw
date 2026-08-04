import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.session.session import Session
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

LOCAL_OFFSET = timezone(timedelta(hours=8))
CREATED_AT = datetime(2026, 7, 11, 15, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
UPDATED_AT = CREATED_AT + timedelta(seconds=5)
SESSION_ID = "20260711-153012-123000_550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION_ID = "20260711-153012-123000_6fa459ea-ee8a-4ca4-894e-db77e160355e"
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


@pytest.mark.asyncio
async def test_persist_writes_one_complete_compact_utf8_snapshot_atomically(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "请读取 README。", extension={"nested": ["value"]})
    replacements: list[tuple[Path, bytes]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(target: Path, content: bytes) -> None:
        replacements.append((target, content))
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_replace)

    message_timestamp = session.messages[0]["timestamp"]
    assert isinstance(message_timestamp, str)

    session.persist()
    expected = (
        f'{{"session_id":"{session.session_id}",'
        f'"created_at":"{session.created_at.isoformat(timespec="milliseconds")}",'
        f'"updated_at":"{session.updated_at.isoformat(timespec="milliseconds")}",'
        '"last_consolidated":0,'
        '"metadata":{"title":"Untitled session",'
        '"token_usage":{"model_calls":0,"input_tokens":0,'
        '"output_tokens":0,"total_tokens":0}}}\n'
        f'{{"role":"user","content":"请读取 README。",'
        f'"timestamp":"{message_timestamp}",'
        '"extension":{"nested":["value"]}}\n'
    ).encode()

    assert replacements == []
    await asyncio.sleep(0)

    path = state.sessions_directory / f"{session.session_id}.jsonl"
    raw = HOST_FILESYSTEM.path_for_io(path).read_bytes()

    assert replacements == [(path, expected)]
    assert raw == expected
    assert b"\xe8\xaf\xb7\xe8\xaf\xbb" in raw
    assert Session.load(state, session.session_id).messages == session.messages


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
async def test_ordinary_persist_failure_is_silent_and_a_later_persist_is_independent(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _state(workspace, agent_home)
    session = Session.create(state)
    session.add_message("user", "First attempt")
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def fail_replace(_target: Path, _content: bytes) -> None:
        raise OSError("simulated snapshot failure")

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", fail_replace)
    session.persist()
    await asyncio.sleep(0)

    replacements: list[bytes] = []

    def record_later_replace(target: Path, content: bytes) -> None:
        replacements.append(content)
        replace(target, content)

    monkeypatch.setattr(HOST_FILESYSTEM, "atomic_replace_bytes", record_later_replace)
    session.add_message("user", "Second attempt")
    session.persist()
    await asyncio.sleep(0)

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
    session = Session.create(state)
    session.add_message("user", "Final state")
    timestamps = iter(
        (
            datetime(2026, 7, 11, 15, 30, 13, 100000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 200000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 300000, tzinfo=LOCAL_OFFSET),
            datetime(2026, 7, 11, 15, 30, 13, 400000, tzinfo=LOCAL_OFFSET),
        )
    )
    replacements: list[dict[str, Any]] = []
    replace = HOST_FILESYSTEM.atomic_replace_bytes

    def record_replace(target: Path, content: bytes) -> None:
        replacements.append(json.loads(content.splitlines()[0]))
        if len(replacements) < 3:
            raise OSError("transient snapshot failure")
        replace(target, content)

    monkeypatch.setattr("myclaw.session.session._local_now", timestamps.__next__)
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
        artifact={"path": "artifacts/call-1.txt", "total_chars": 123},
        provider_extension={"trace": [1, 2]},
    )

    message = session.messages[0]
    assert {key: value for key, value in message.items() if key != "timestamp"} == {
        "role": "tool",
        "content": "README.md",
        "tool_call_id": "call-1",
        "name": "read_file",
        "status": "success",
        "artifact": {"path": "artifacts/call-1.txt", "total_chars": 123},
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
def test_load_rejects_malformed_core_legacy_or_partial_message_records(
    agent_home: Path,
    workspace: Path,
    records: list[dict[str, Any]],
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(state, records)

    with pytest.raises(ValueError):
        Session.load(state, SESSION_ID)


def test_load_rejects_a_jsonl_file_without_a_complete_trailing_line(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _state(workspace, agent_home)
    _write_jsonl(state, [_header()], trailing_newline=False)

    with pytest.raises(ValueError, match="newline"):
        Session.load(state, SESSION_ID)
