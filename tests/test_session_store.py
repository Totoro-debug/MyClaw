import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent_home import AgentHome
from myclaw.contracts import (
    AssistantSessionMessage,
    ConversationSession,
    ModelUsage,
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
        metadata=metadata,
        messages=(user_message, assistant_message),
    )
