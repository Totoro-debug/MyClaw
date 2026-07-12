import asyncio
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from myclaw.agent_home import AgentHome
from myclaw.contracts import (
    AssistantSessionMessage,
    ConversationSession,
    CumulativeUsage,
    MetadataUpdate,
    ModelUsage,
    SessionError,
    SessionMetadata,
    SessionStore,
    UserSessionMessage,
)
from myclaw.session_store import JsonlSessionStore
from myclaw.workspace import Workspace
from tests.fixtures import FakeClock

LOCAL_OFFSET = timezone(timedelta(hours=8))
CREATED_AT = datetime(2026, 7, 11, 15, 30, 12, 123456, tzinfo=LOCAL_OFFSET)
SESSION_UUID = UUID("550e8400-e29b-41d4-a716-446655440000")
SESSION_ID = "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000"
USER_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
ASSISTANT_UUID = UUID("7c9e6679-7425-40de-944b-e07fc1f90ae7")


def read_bytes(path: Path) -> bytes:
    if os.name == "nt":
        path = Path(f"\\\\?\\{path.absolute()}")
    return path.read_bytes()


@pytest.mark.asyncio
async def test_prepared_session_materializes_exact_jsonl_on_first_user_message_and_reloads(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(CREATED_AT)
    session_uuids = iter((SESSION_UUID,))
    workspace_identity = Workspace.from_path(workspace)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=workspace_identity,
        now=clock.now,
        new_uuid=session_uuids.__next__,
    )

    metadata = store.prepare()
    session_path = (
        agent_home
        / "sessions"
        / workspace_identity.slug
        / "20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000.jsonl"
    )

    assert metadata.id == SESSION_ID
    assert store.path_for(SESSION_ID) == session_path
    assert not session_path.parent.exists()

    user_message = UserSessionMessage(
        id=str(USER_UUID),
        created_at=metadata.created_at,
        content="Help me inspect this project.",
    )
    await store.append_message(SESSION_ID, user_message)

    assert read_bytes(session_path) == (
        b'{"record_type":"metadata","schema_version":1,'
        b'"id":"20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000",'
        b'"title":"Untitled session",'
        b'"created_at":"2026-07-11T15:30:12.123+08:00",'
        b'"updated_at":"2026-07-11T15:30:12.123+08:00",'
        b'"consolidation_cursor":0,'
        b'"cumulative_usage":{"model_calls":0,"input_tokens":0,'
        b'"output_tokens":0,"total_tokens":0}}\n'
        b'{"record_type":"message","id":"0f8fad5b-d9cb-469f-a165-70867728950e",'
        b'"created_at":"2026-07-11T15:30:12.123+08:00","role":"user",'
        b'"content":"Help me inspect this project."}\n'
    )
    assert await store.load(SESSION_ID) == ConversationSession(
        metadata=metadata,
        messages=(user_message,),
    )
    assert isinstance(store, SessionStore)


@pytest.mark.asyncio
async def test_completed_assistant_is_one_complete_record_and_reloads(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(CREATED_AT)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    metadata = store.prepare()
    user_message = UserSessionMessage(
        id=str(USER_UUID),
        created_at=metadata.created_at,
        content="Help me inspect this project.",
    )
    assistant_message = AssistantSessionMessage(
        id=str(ASSISTANT_UUID),
        created_at=datetime(2026, 7, 11, 15, 30, 13, tzinfo=LOCAL_OFFSET),
        content="I will inspect the files.",
        tool_calls=(),
        status="completed",
        error=None,
        usage=ModelUsage(input_tokens=120, output_tokens=24, total_tokens=144),
    )

    await store.append_message(SESSION_ID, user_message)
    await store.append_message(SESSION_ID, assistant_message)

    persisted = read_bytes(store.path_for(SESSION_ID))
    assert persisted.count(b'"role":"assistant"') == 1
    assert persisted.endswith(
        b'{"record_type":"message","id":"7c9e6679-7425-40de-944b-e07fc1f90ae7",'
        b'"created_at":"2026-07-11T15:30:13.000+08:00","role":"assistant",'
        b'"content":"I will inspect the files.","tool_calls":[],'
        b'"status":"completed","error":null,'
        b'"usage":{"input_tokens":120,"output_tokens":24,"total_tokens":144}}\n'
    )
    assert await store.load(SESSION_ID) == ConversationSession(
        metadata=SessionMetadata(
            id=metadata.id,
            title=metadata.title,
            created_at=metadata.created_at,
            updated_at=assistant_message.created_at,
            consolidation_cursor=0,
            cumulative_usage=CumulativeUsage(
                model_calls=1,
                input_tokens=120,
                output_tokens=24,
                total_tokens=144,
            ),
        ),
        messages=(user_message, assistant_message),
    )


@pytest.mark.asyncio
async def test_same_runtime_concurrent_session_writes_preserve_every_record_and_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(CREATED_AT)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    metadata = store.prepare()
    user_message = UserSessionMessage(
        id=str(USER_UUID),
        created_at=metadata.created_at,
        content="Run concurrent Session writes.",
    )
    await store.append_message(metadata.id, user_message)
    assistants = tuple(
        AssistantSessionMessage(
            id=str(uuid4()),
            created_at=metadata.created_at + timedelta(seconds=index + 1),
            content=f"Concurrent response {index}",
            tool_calls=(),
            status="completed",
            error=None,
            usage=ModelUsage(
                input_tokens=index + 1,
                output_tokens=1,
                total_tokens=index + 2,
            ),
        )
        for index in range(12)
    )

    await asyncio.gather(*(store.append_message(metadata.id, message) for message in assistants))

    persisted = await store.load(metadata.id)
    assert persisted.messages[0] == user_message
    assert {message.id for message in persisted.messages[1:]} == {
        message.id for message in assistants
    }
    assert len(persisted.messages) == 13
    assert persisted.metadata.cumulative_usage == CumulativeUsage(
        model_calls=12,
        input_tokens=78,
        output_tokens=12,
        total_tokens=90,
    )


@pytest.mark.asyncio
async def test_failed_and_interrupted_assistants_reload_with_safe_error_details(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=FakeClock(CREATED_AT).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    metadata = store.prepare()
    user_message = UserSessionMessage(
        id=str(USER_UUID),
        created_at=metadata.created_at,
        content="Keep the terminal outcome.",
    )
    failed_message = AssistantSessionMessage(
        id=str(ASSISTANT_UUID),
        created_at=metadata.created_at + timedelta(seconds=1),
        content="",
        tool_calls=(),
        status="error",
        error=SessionError(code="provider_timeout", message="The model timed out."),
        usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
    )
    interrupted_message = AssistantSessionMessage(
        id="a3bb189e-8bf9-4c4b-ae4a-c6699f6f7e34",
        created_at=metadata.created_at + timedelta(seconds=2),
        content="Partial answer",
        tool_calls=(),
        status="interrupted",
        error=SessionError(code="turn_cancelled", message="Turn interrupted by user."),
        usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
    )

    await store.append_message(metadata.id, user_message)
    await store.append_message(metadata.id, failed_message)
    await store.append_message(metadata.id, interrupted_message)

    reloaded = await store.load(metadata.id)
    assert reloaded.messages == (user_message, failed_message, interrupted_message)


@pytest.mark.asyncio
async def test_first_materialized_message_updates_time_without_estimated_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=FakeClock(CREATED_AT).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    metadata = store.prepare()
    first_message_at = metadata.created_at + timedelta(seconds=3)
    await store.append_message(
        metadata.id,
        UserSessionMessage(
            id=str(USER_UUID),
            created_at=first_message_at,
            content="First message.",
        ),
    )

    materialized = await store.load(metadata.id)

    assert (
        materialized.metadata.updated_at,
        materialized.metadata.cumulative_usage,
    ) == (
        first_message_at,
        CumulativeUsage(model_calls=0, input_tokens=0, output_tokens=0, total_tokens=0),
    )


@pytest.mark.asyncio
async def test_appended_messages_update_session_time_and_actual_cumulative_usage(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    clock = FakeClock(CREATED_AT)
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=clock.now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    metadata = store.prepare()
    user_message = UserSessionMessage(
        id=str(USER_UUID),
        created_at=metadata.created_at,
        content="Track this session.",
    )
    await store.append_message(SESSION_ID, user_message)
    clock.advance(5)
    updated_at = clock.now().replace(microsecond=123_000)
    assistant_message = AssistantSessionMessage(
        id=str(ASSISTANT_UUID),
        created_at=updated_at,
        content="Tracked.",
        tool_calls=(),
        status="completed",
        error=None,
        usage=ModelUsage(input_tokens=7, output_tokens=2, total_tokens=9),
    )

    await store.append_message(SESSION_ID, assistant_message)

    session = await store.load(SESSION_ID)
    assert (
        len(session.messages),
        session.metadata.updated_at,
        session.metadata.cumulative_usage,
    ) == (
        2,
        updated_at,
        CumulativeUsage(model_calls=1, input_tokens=7, output_tokens=2, total_tokens=9),
    )


@pytest.mark.asyncio
async def test_metadata_update_preserves_message_bytes_and_sets_exact_session_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=FakeClock(CREATED_AT).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    metadata = store.prepare()
    user_message = UserSessionMessage(
        id=str(USER_UUID),
        created_at=metadata.created_at,
        content="Keep my exact UTF-8 history: \u4f60\u597d.",
    )
    await store.append_message(SESSION_ID, user_message)
    session_path = store.path_for(SESSION_ID)
    original_messages = read_bytes(session_path).partition(b"\n")[2]
    updated_at = datetime(2026, 7, 11, 16, 0, 0, 456000, tzinfo=LOCAL_OFFSET)
    expected_usage = CumulativeUsage(
        model_calls=2,
        input_tokens=21,
        output_tokens=5,
        total_tokens=26,
    )

    await store.update_metadata(
        SESSION_ID,
        MetadataUpdate(
            title="Tracked session",
            updated_at=updated_at,
            consolidation_cursor=1,
            cumulative_usage=expected_usage,
        ),
    )

    reloaded = await store.load(SESSION_ID)
    assert (
        reloaded.metadata.title,
        reloaded.metadata.updated_at,
        reloaded.metadata.consolidation_cursor,
        reloaded.metadata.cumulative_usage,
        read_bytes(session_path).partition(b"\n")[2],
    ) == (
        "Tracked session",
        updated_at,
        1,
        expected_usage,
        original_messages,
    )


@pytest.mark.asyncio
async def test_ordinary_append_remains_complete_when_metadata_rewrite_fails(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=FakeClock(CREATED_AT).now,
        new_uuid=iter((SESSION_UUID,)).__next__,
    )
    metadata = store.prepare()
    user_message = UserSessionMessage(
        id=str(USER_UUID),
        created_at=metadata.created_at,
        content="First record.",
    )
    await store.append_message(SESSION_ID, user_message)
    assistant_message = AssistantSessionMessage(
        id=str(ASSISTANT_UUID),
        created_at=datetime(2026, 7, 11, 15, 31, tzinfo=LOCAL_OFFSET),
        content="Second record.",
        tool_calls=(),
        status="completed",
        error=None,
        usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
    )

    def fail_metadata_replace(_source: str | bytes, _target: str | bytes) -> None:
        raise OSError("simulated metadata replacement failure")

    monkeypatch.setattr(os, "replace", fail_metadata_replace)

    with pytest.raises(OSError, match="simulated metadata replacement failure"):
        await store.append_message(SESSION_ID, assistant_message)

    reloaded = await store.load(SESSION_ID)
    assert (reloaded.messages, reloaded.metadata) == (
        (user_message, assistant_message),
        metadata,
    )
