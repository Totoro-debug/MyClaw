import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from myclaw.agent.events import PermissionRequestedPayload
from myclaw.agent.workspace import Workspace
from myclaw.config.agent_home import AgentHome
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.conversation import ChatModelSettings, StreamingConversationPort
from myclaw.session.session_store import JsonlSessionStore
from myclaw.tools.models import (
    ModelToolCall,
    ToolExecutionContext,
)
from myclaw.tools.tool_gateway import ToolGateway
from tests.fixtures import ScriptedFakeProvider, StreamScript

NOW = datetime(2026, 7, 12, 19, 0, tzinfo=timezone(timedelta(hours=8)))


def _outside_directory_aliases(workspace: Path, outside: Path) -> tuple[Path, ...]:
    aliases: list[Path] = []
    symlink = workspace / "symlink-escape"
    try:
        symlink.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable on this host: {error}")
    else:
        aliases.append(symlink)

    if os.name == "nt":
        junction = workspace / "junction-escape"
        try:
            subprocess.run(
                ("cmd", "/c", "mklink", "/J", str(junction), str(outside)),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            if not aliases:
                pytest.skip(f"directory links are unavailable on this host: {error}")
        else:
            aliases.append(junction)
    return tuple(aliases)


@pytest.mark.asyncio
async def test_conversation_approved_write_file_executes_the_same_call_once(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    store = JsonlSessionStore(
        agent_home=home,
        workspace=Workspace.from_path(workspace),
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    session = store.prepare()
    target = workspace / "new" / "notes.txt"
    tool_call = ModelToolCall(
        id="call_write_once",
        name="write_file",
        arguments={"path": "new/notes.txt", "content": "approved\ncontent\n"},
    )
    provider = ScriptedFakeProvider(
        streams=(
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="", tool_calls=(tool_call,)),
                            usage=ModelUsage(input_tokens=8, output_tokens=2, total_tokens=10),
                            finish_reason="tool_calls",
                        )
                    ),
                )
            ),
            StreamScript(
                events=(
                    ModelCompleted(
                        response=ModelResponse(
                            message=AssistantModelMessage(content="Done."),
                            usage=ModelUsage(input_tokens=12, output_tokens=2, total_tokens=14),
                            finish_reason="stop",
                        )
                    ),
                )
            ),
        )
    )
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=session.id,
        )
    )
    conversation = StreamingConversationPort(
        provider=provider,
        sessions=store,
        session_id=session.id,
        settings=ChatModelSettings(
            model="test-model",
            max_output=1024,
            temperature=0.2,
            reasoning_effort=None,
            timeout_seconds=30,
        ),
        now=lambda: NOW,
        new_uuid=uuid4,
        tool_gateway=gateway,
    )
    real_write_text = Path.write_text
    written_paths: list[Path] = []

    def recording_write_text(
        path: Path,
        content: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        written_paths.append(path)
        return real_write_text(
            path,
            content,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", recording_write_text)

    events = conversation.submit("Write my notes.")
    observed = [await anext(events), await anext(events), await anext(events)]

    permission_event = observed[-1]
    assert permission_event.type == "permission_requested"
    permission = permission_event.payload
    assert isinstance(permission, PermissionRequestedPayload)
    assert permission.tool_call_id == tool_call.id
    assert permission.resource == "new/notes.txt"
    assert not target.exists()

    await conversation.resolve_permission(permission.request_id, approved=True)
    observed.extend([event async for event in events])

    assert [event.type for event in observed] == [
        "turn_started",
        "tool_started",
        "permission_requested",
        "tool_completed",
        "turn_completed",
    ]
    assert written_paths == [target.resolve()]
    assert target.read_bytes() == b"approved\ncontent\n"
    assert len(provider.stream_requests) == 2


@pytest.mark.asyncio
async def test_gateway_approved_edit_file_replaces_exact_text_once(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = workspace / "notes.txt"
    target.write_text("alpha\nbefore\nomega\n", encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
    )
    tool_call = ModelToolCall(
        id="call_edit_once",
        name="edit_file",
        arguments={
            "path": "notes.txt",
            "old_text": "before",
            "new_text": "after",
        },
    )
    real_write_text = Path.write_text
    written_paths: list[Path] = []

    def recording_write_text(
        path: Path,
        content: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        written_paths.append(path)
        return real_write_text(
            path,
            content,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", recording_write_text)

    permission = gateway.permission_request(tool_call)
    assert permission is not None
    assert permission.action == "edit"
    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "success"
    assert written_paths == [target.resolve()]
    assert target.read_text(encoding="utf-8") == "alpha\nafter\nomega\n"


@pytest.mark.asyncio
async def test_edit_file_preserves_unedited_line_endings(
    agent_home: Path,
    workspace: Path,
) -> None:
    target = workspace / "mixed-newlines.txt"
    target.write_bytes(b"before\r\nmiddle\nomega\r\n")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_edit_newlines",
            name="edit_file",
            arguments={
                "path": "mixed-newlines.txt",
                "old_text": "middle",
                "new_text": "after",
            },
        ),
        approved=True,
    )

    assert result.status == "success"
    assert target.read_bytes() == b"before\r\nafter\nomega\r\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial", "old_text"),
    (("alpha\nomega\n", "missing"), ("repeat\nrepeat\n", "repeat")),
)
async def test_edit_file_requires_one_exact_old_text_match(
    agent_home: Path,
    workspace: Path,
    initial: str,
    old_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = workspace / "notes.txt"
    target.write_text(initial, encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
    )
    tool_call = ModelToolCall(
        id="call_edit_invalid_match",
        name="edit_file",
        arguments={"path": "notes.txt", "old_text": old_text, "new_text": "after"},
    )
    write_calls = 0
    real_write_text = Path.write_text

    def recording_write_text(
        path: Path,
        content: str,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        nonlocal write_calls
        write_calls += 1
        return real_write_text(
            path,
            content,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "write_text", recording_write_text)

    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_invalid_arguments"
    assert write_calls == 0
    assert target.read_text(encoding="utf-8") == initial


@pytest.mark.asyncio
async def test_edit_file_replace_all_replaces_every_exact_match(
    agent_home: Path,
    workspace: Path,
) -> None:
    target = workspace / "repeated.txt"
    target.write_bytes(b"before\r\nbefore\nend\r\n")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
    )
    tool_call = ModelToolCall(
        id="call_edit_all",
        name="edit_file",
        arguments={
            "path": "repeated.txt",
            "old_text": "before",
            "new_text": "after",
            "replace_all": True,
        },
    )

    assert gateway.permission_request(tool_call) is not None
    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "success"
    assert target.read_bytes() == b"after\r\nafter\nend\r\n"


@pytest.mark.asyncio
async def test_workspace_write_tools_deny_escape_paths_before_confirmation(
    agent_home: Path,
    workspace: Path,
) -> None:
    outside = workspace.parent / "outside-write-target"
    outside.mkdir()
    secret = outside / "secret.txt"
    secret.write_text("outside secret", encoding="utf-8")
    aliases = _outside_directory_aliases(workspace, outside)
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
    )
    calls = [
        ModelToolCall(
            id="call_write_traversal",
            name="write_file",
            arguments={"path": "../outside-write-target/traversal.txt", "content": "escape"},
        ),
        ModelToolCall(
            id="call_write_absolute",
            name="write_file",
            arguments={"path": str(outside / "absolute.txt"), "content": "escape"},
        ),
        ModelToolCall(
            id="call_edit_traversal",
            name="edit_file",
            arguments={
                "path": "../outside-write-target/secret.txt",
                "old_text": "outside secret",
                "new_text": "escape",
            },
        ),
    ]
    for index, alias in enumerate(aliases):
        calls.extend(
            (
                ModelToolCall(
                    id=f"call_write_alias_{index}",
                    name="write_file",
                    arguments={
                        "path": str(alias / "missing" / "notes.txt"),
                        "content": "escape",
                    },
                ),
                ModelToolCall(
                    id=f"call_edit_alias_{index}",
                    name="edit_file",
                    arguments={
                        "path": str(alias / "secret.txt"),
                        "old_text": "outside secret",
                        "new_text": "escape",
                    },
                ),
            )
        )

    for tool_call in calls:
        assert gateway.permission_request(tool_call) is None
        result = await gateway.execute(tool_call, approved=True)
        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "tool_denied"

    assert secret.read_text(encoding="utf-8") == "outside secret"
    assert not (outside / "traversal.txt").exists()
    assert not (outside / "absolute.txt").exists()
    assert not (outside / "missing").exists()


@pytest.mark.asyncio
async def test_workspace_write_tools_always_deny_agent_home_internal_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "containing-workspace"
    workspace.mkdir()
    agent_home = workspace / ".myclaw"
    protected_paths = (
        "config.toml",
        "sessions/workspace/session.jsonl",
        "sessions/workspace/artifacts/session/call.txt",
        "memory/summary.jsonl",
        "memory/.cursor",
        "memory/memory.md",
        "memory/pending-consolidations/session.json",
        "scheduled-work.json",
    )
    for relative in protected_paths:
        target = agent_home / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("protected", encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
    )

    for index, relative in enumerate(protected_paths):
        workspace_relative = (Path(".myclaw") / relative).as_posix()
        if index % 2 == 0:
            tool_call = ModelToolCall(
                id=f"call_write_agent_home_{index}",
                name="write_file",
                arguments={"path": workspace_relative, "content": "overwritten"},
            )
        else:
            tool_call = ModelToolCall(
                id=f"call_edit_agent_home_{index}",
                name="edit_file",
                arguments={
                    "path": workspace_relative,
                    "old_text": "protected",
                    "new_text": "overwritten",
                },
            )

        assert gateway.permission_request(tool_call) is None
        result = await gateway.execute(tool_call, approved=True)
        assert result.status == "error"
        assert result.error is not None
        assert result.error.code == "tool_denied"

    for relative in protected_paths:
        assert (agent_home / relative).read_text(encoding="utf-8") == "protected"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="NUL is a device name only on Windows")
async def test_write_file_denies_windows_device_name_before_confirmation(
    agent_home: Path,
    workspace: Path,
) -> None:
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id="20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000",
        )
    )
    tool_call = ModelToolCall(
        id="call_write_device",
        name="write_file",
        arguments={"path": "NUL", "content": "must not reach the device"},
    )

    assert gateway.permission_request(tool_call) is None
    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
