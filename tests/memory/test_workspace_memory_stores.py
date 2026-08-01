import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from myclaw.agent.runtime import prepare_repl_runtime
from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.config.config import ConfigLoader
from myclaw.memory.conversation_summary import WorkspaceJsonlSummaryStore
from myclaw.memory.memory_task import MemoryPathDeniedError, WorkspaceFileMemoryStore
from myclaw.provider.models import (
    AssistantModelMessage,
    ModelCompleted,
    ModelRequest,
    ModelResponse,
    ModelUsage,
)
from myclaw.session.records import UserSessionMessage
from myclaw.session.session_store import JsonlSessionStore
from myclaw.templates import load_template
from myclaw.utils.host_filesystem import HOST_FILESYSTEM
from tests.configuration.test_config import VALID_CONFIG
from tests.fixtures import ScriptedFakeProvider, StreamScript

LOCAL_OFFSET = timezone(timedelta(hours=8))
NOW = datetime(2026, 7, 31, 16, 30, 12, 123000, tzinfo=LOCAL_OFFSET)
SESSION_ID = "20260731-163012-123000_550e8400-e29b-41d4-a716-446655440000"
USER_ID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")


def _state(path: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(path))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


async def _sessions(state: WorkspaceState) -> JsonlSessionStore:
    sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=uuid4,
    )
    sessions.prepare_with_id(session_id=SESSION_ID, title="Memory test", created_at=NOW)
    await sessions.append_message(
        SESSION_ID,
        UserSessionMessage(id=str(USER_ID), created_at=NOW, content="Workspace question."),
    )
    return sessions


def _response(content: str) -> ModelResponse:
    return ModelResponse(
        message=AssistantModelMessage(content=content),
        usage=ModelUsage(input_tokens=4, output_tokens=2, total_tokens=6),
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_workspace_memory_stores_preserve_formats_and_isolate_roots(
    workspace: Path,
) -> None:
    other_workspace = workspace.parent / "other-workspace"
    other_workspace.mkdir()
    first_state = _state(workspace)
    second_state = _state(other_workspace)
    first_unknown = first_state.path / "future-state.bin"
    second_unknown = second_state.memory_directory / "future-memory.bin"
    first_unknown.write_bytes(b"first future schema")
    second_unknown.write_bytes(b"second future schema")
    first_memory = WorkspaceFileMemoryStore(first_state)
    second_memory = WorkspaceFileMemoryStore(second_state)
    first_summaries = WorkspaceJsonlSummaryStore(first_state)
    second_summaries = WorkspaceJsonlSummaryStore(second_state)

    assert await first_memory.read_long_term() == load_template("long-term-memory.md")
    await first_memory.replace_long_term("# First Workspace\n\nProject: 界\n")
    await second_memory.replace_long_term("# Second Workspace\n")
    assert not (first_state.memory_directory / ".cursor").exists()
    assert not first_summaries.path.exists()
    assert not first_summaries.pending_directory.exists()
    await first_memory.write_summary_cursor(7)
    await second_memory.write_summary_cursor(3)
    first_entry = await first_summaries.append("First summary 界", NOW)
    second_entry = await second_summaries.append("Second summary", NOW)

    assert await first_memory.read_long_term() == "# First Workspace\n\nProject: 界\n"
    assert await second_memory.read_long_term() == "# Second Workspace\n"
    assert await first_memory.read_summary_cursor() == 7
    assert await second_memory.read_summary_cursor() == 3
    assert (first_state.memory_directory / ".cursor").read_bytes() == b"7\n"
    assert (second_state.memory_directory / ".cursor").read_bytes() == b"3\n"
    assert (
        first_summaries.path.read_bytes()
        == (
            '{"index":1,"timestamp":"2026-07-31T16:30:12.123+08:00","content":"First summary 界"}\n'
        ).encode()
    )
    assert second_summaries.path.read_bytes() == (
        b'{"index":1,"timestamp":"2026-07-31T16:30:12.123+08:00","content":"Second summary"}\n'
    )
    assert first_entry.index == second_entry.index == 1
    assert first_unknown.read_bytes() == b"first future schema"
    assert second_unknown.read_bytes() == b"second future schema"


@pytest.mark.parametrize("failed_step", ("journal", "summary", "cursor", "delete"))
@pytest.mark.asyncio
async def test_workspace_consolidation_crash_windows_recover_exactly_once(
    workspace: Path,
    failed_step: str,
) -> None:
    state = _state(workspace)
    sessions = await _sessions(state)
    operations: list[str] = []

    def replace_summary_state(target: Path, content: bytes) -> None:
        step = "journal" if target.suffix == ".json" else "summary"
        operations.append(step)
        if step == failed_step:
            raise OSError(f"injected {step} failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    def replace_session_cursor(target: Path, content: bytes) -> None:
        operations.append("cursor")
        if failed_step == "cursor":
            raise OSError("injected cursor failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    def unlink_journal(target: Path) -> None:
        operations.append("delete")
        if failed_step == "delete":
            raise OSError("injected journal deletion failure")
        target.unlink()

    transaction_sessions = JsonlSessionStore(
        workspace_state=state,
        now=lambda: NOW,
        new_uuid=uuid4,
        replace_bytes=replace_session_cursor,
    )
    transaction_summaries = WorkspaceJsonlSummaryStore(
        state,
        replace_bytes=replace_summary_state,
        unlink_file=unlink_journal,
    )

    with pytest.raises(OSError):
        await transaction_summaries.commit_consolidation(
            sessions=transaction_sessions,
            session_id=SESSION_ID,
            old_cursor=0,
            new_cursor=1,
            content="Recovered workspace summary.",
            timestamp=NOW,
        )

    expected = ("journal", "summary", "cursor", "delete")
    assert operations == list(expected[: expected.index(failed_step) + 1])
    healthy = WorkspaceJsonlSummaryStore(state)
    if failed_step == "journal":
        await healthy.commit_consolidation(
            sessions=sessions,
            session_id=SESSION_ID,
            old_cursor=0,
            new_cursor=1,
            content="Recovered workspace summary.",
            timestamp=NOW,
        )
    else:
        assert await healthy.recover_pending(sessions) == 1
    assert await healthy.recover_pending(sessions) == 0

    assert (await sessions.load(SESSION_ID)).metadata.consolidation_cursor == 1
    assert [json.loads(line) for line in healthy.path.read_text(encoding="utf-8").splitlines()] == [
        {
            "index": 1,
            "timestamp": "2026-07-31T16:30:12.123+08:00",
            "content": "Recovered workspace summary.",
        }
    ]
    assert list(healthy.pending_directory.glob("*.json")) == []


@pytest.mark.asyncio
async def test_workspace_recovery_rejects_a_session_store_from_another_workspace(
    workspace: Path,
) -> None:
    other_workspace = workspace.parent / "other-workspace"
    other_workspace.mkdir()
    first_state = _state(workspace)
    second_state = _state(other_workspace)
    first_sessions = await _sessions(first_state)
    second_sessions = await _sessions(second_state)
    replace_calls = 0

    def fail_summary(target: Path, content: bytes) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("injected summary failure")
        HOST_FILESYSTEM.atomic_replace_bytes(target, content)

    failing = WorkspaceJsonlSummaryStore(first_state, replace_bytes=fail_summary)
    with pytest.raises(OSError):
        await failing.commit_consolidation(
            sessions=first_sessions,
            session_id=SESSION_ID,
            old_cursor=0,
            new_cursor=1,
            content="First workspace pending summary.",
            timestamp=NOW,
        )
    journal_bytes = next(failing.pending_directory.glob("*.json")).read_bytes()

    with pytest.raises(ValueError, match="another Workspace State"):
        await WorkspaceJsonlSummaryStore(first_state).recover_pending(second_sessions)
    with pytest.raises(ValueError, match="another Workspace State"):
        await WorkspaceJsonlSummaryStore(first_state).commit_consolidation(
            sessions=second_sessions,
            session_id=SESSION_ID,
            old_cursor=0,
            new_cursor=1,
            content="Must not cross Workspace ownership.",
            timestamp=NOW,
        )

    assert (await second_sessions.load(SESSION_ID)).metadata.consolidation_cursor == 0
    assert not (second_state.memory_directory / "summary.jsonl").exists()
    assert next(failing.pending_directory.glob("*.json")).read_bytes() == journal_bytes
    assert await WorkspaceJsonlSummaryStore(first_state).recover_pending(first_sessions) == 1
    assert (await first_sessions.load(SESSION_ID)).metadata.consolidation_cursor == 1


@pytest.mark.parametrize("path_name", ("memory.md", ".cursor", "summary.jsonl"))
@pytest.mark.asyncio
async def test_workspace_memory_files_reject_directories_and_hard_links(
    workspace: Path,
    path_name: str,
) -> None:
    state = _state(workspace)
    path = (
        state.long_term_memory_path
        if path_name == "memory.md"
        else state.memory_directory / path_name
    )
    if path.exists():
        path.unlink()
    path.mkdir()
    memory = WorkspaceFileMemoryStore(state)
    summaries = WorkspaceJsonlSummaryStore(state)

    with pytest.raises((MemoryPathDeniedError, PermissionError)):
        if path_name == "memory.md":
            await memory.read_long_term()
        elif path_name == ".cursor":
            await memory.read_summary_cursor()
        else:
            await summaries.after(0, 1)

    path.rmdir()
    outside = workspace.parent / f"outside-{path_name.lstrip('.')}"
    outside.write_bytes(b"outside memory state must remain unchanged")
    path.hardlink_to(outside)
    with pytest.raises((MemoryPathDeniedError, PermissionError)):
        if path_name == "memory.md":
            await memory.read_long_term()
        elif path_name == ".cursor":
            await memory.read_summary_cursor()
        else:
            await summaries.after(0, 1)
    assert outside.read_bytes() == b"outside memory state must remain unchanged"


@pytest.mark.asyncio
async def test_workspace_pending_directory_and_journals_reject_aliases(
    workspace: Path,
) -> None:
    state = _state(workspace)
    sessions = await _sessions(state)
    summaries = WorkspaceJsonlSummaryStore(state)
    summaries.pending_directory.write_bytes(b"not a directory")
    with pytest.raises(PermissionError):
        await summaries.recover_pending(sessions)
    summaries.pending_directory.unlink()

    outside_directory = workspace.parent / "outside-pending"
    outside_directory.mkdir()
    outside_secret = outside_directory / f"{SESSION_ID}.json"
    outside_secret.write_bytes(b"outside journal secret")
    subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(summaries.pending_directory), str(outside_directory)),
        capture_output=True,
        text=True,
        check=True,
    )
    with pytest.raises(PermissionError):
        await summaries.recover_pending(sessions)
    assert outside_secret.read_bytes() == b"outside journal secret"


@pytest.mark.asyncio
async def test_workspace_pending_journal_rejects_a_hard_link(
    workspace: Path,
) -> None:
    state = _state(workspace)
    sessions = await _sessions(state)
    summaries = WorkspaceJsonlSummaryStore(state)
    summaries.pending_directory.mkdir()
    outside = workspace.parent / "outside-journal.json"
    outside.write_bytes(b"outside journal must not be parsed")
    journal = summaries.pending_directory / f"{SESSION_ID}.json"
    journal.hardlink_to(outside)

    with pytest.raises(PermissionError):
        await summaries.recover_pending(sessions)

    assert outside.read_bytes() == b"outside journal must not be parsed"


@pytest.mark.asyncio
async def test_runtime_routes_memory_and_summaries_to_the_current_workspace(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(workspace)
    legacy_files = {
        agent_home / "memory" / "memory.md": b"# Agent Home Memory\n",
        agent_home / "memory" / "summary.jsonl": b"legacy summary must not be parsed\xff\n",
        agent_home / "memory" / ".cursor": b"not-an-index\n",
        agent_home / "memory" / "pending-consolidations" / "legacy.json": b"not-json\n",
    }
    for path, content in legacy_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    state.long_term_memory_path.write_text("# Workspace Memory\n", encoding="utf-8")
    (agent_home / "config.toml").write_text(
        VALID_CONFIG.replace(
            "consolidation_message_threshold = 50",
            "consolidation_message_threshold = 4",
        ),
        encoding="utf-8",
    )
    provider = ScriptedFakeProvider(
        streams=tuple(
            StreamScript(events=(ModelCompleted(response=_response(f"Answer {index}.")),))
            for index in range(1, 4)
        ),
        completions=(_response("Workspace summary."),),
    )
    runtime = prepare_repl_runtime(
        agent_home=home,
        workspace=workspace,
        configuration=ConfigLoader(home).load(),
        provider_factory=lambda _: provider,
        now=lambda: NOW,
        new_uuid=uuid4,
    )

    try:
        for index in range(1, 4):
            _ = [event async for event in runtime.conversation.submit(f"Question {index}.")]
        memory_view = await runtime.management_dispatcher.dispatch("/memory")
    finally:
        await runtime.close()

    first_request = provider.stream_requests[0]
    assert isinstance(first_request, ModelRequest)
    assert "# Workspace Memory" in first_request.system_prompt
    assert "# Agent Home Memory" not in first_request.system_prompt
    assert memory_view.output == "# Workspace Memory\n"
    assert "Workspace summary." in (state.memory_directory / "summary.jsonl").read_text(
        encoding="utf-8"
    )
    assert not (state.memory_directory / ".cursor").exists()
    assert state.long_term_memory_path.read_text(encoding="utf-8") == "# Workspace Memory\n"
    assert {path: path.read_bytes() for path in legacy_files} == legacy_files
