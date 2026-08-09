import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState, WorkspaceStateError
from myclaw.config.agent_home import AgentHome
from myclaw.provider.models import AssistantModelMessage
from myclaw.session.session import Session
from myclaw.tools.files.file_tools import ListFilesTool, SearchFilesTool
from myclaw.tools.security import Security
from myclaw.tools.tool_artifacts import externalize_tool_result
from myclaw.tools.tool_gateway import ModelToolCall, ToolGateway, ToolResult
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

SESSION_ID = "20260713-040000-000000_550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION_ID = "20260713-050000-000000_550e8400-e29b-41d4-a716-446655440000"


def _read_file_gateway(*, agent_home: Path, workspace: Path) -> ToolGateway:
    workspace_identity = Workspace.from_path(workspace)
    workspace_state = WorkspaceState(workspace_identity)
    security = Security(
        workspace=workspace_identity,
        agent_home=agent_home,
        artifact_directory=workspace_state.artifacts_directory / SESSION_ID,
    )
    gateway = ToolGateway()
    gateway.register_tools(
        (
            ListFilesTool(security=security),
            SearchFilesTool(security=security),
        )
    )
    return gateway


def _workspace_state(workspace: Path) -> WorkspaceState:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


def _artifact_directory(workspace: Path, session_id: str = SESSION_ID) -> Path:
    state = WorkspaceState(Workspace.from_path(workspace))
    return state.artifacts_directory / session_id


def _artifact_session(state: WorkspaceState) -> Session:
    session = Session.create(
        state,
        now=lambda: datetime(2026, 7, 13, 4, tzinfo=UTC),
        new_uuid=lambda: UUID("550e8400-e29b-41d4-a716-446655440000"),
    )
    session.update_metadata(title="Artifact test")
    return session


def _create_directory_alias(alias: Path, target: Path) -> None:
    subprocess.run(
        ("cmd", "/c", "mklink", "/J", str(alias), str(target)),
        check=True,
        capture_output=True,
        text=True,
    )


def _long_path(path: Path) -> Path:
    return HOST_FILESYSTEM.path_for_io(path)


def test_assistant_message_rejects_duplicate_tool_call_ids() -> None:
    first = ModelToolCall(
        id="duplicate-call",
        name="read_file",
        arguments='{"path":"a.txt"}',
    )
    second = ModelToolCall(
        id="duplicate-call",
        name="read_file",
        arguments='{"path":"b.txt"}',
    )

    with pytest.raises(ValueError, match="tool call IDs must be unique"):
        AssistantModelMessage(content="", tool_calls=(first, second))


def test_workspace_state_initialization_denies_an_external_memory_directory_alias(
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.path.mkdir()
    outside = workspace.parent / "outside-memory"
    outside.mkdir()
    _create_directory_alias(state.memory_directory, outside)

    with pytest.raises(WorkspaceStateError) as captured:
        state.initialize(agent_home_root=Path.home() / ".myclaw")

    assert captured.value.path == state.memory_directory
    assert not (outside / "memory.md").exists()


def test_workspace_state_initialization_denies_an_external_sessions_directory_alias(
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.path.mkdir()
    outside = workspace.parent / "outside-sessions"
    outside.mkdir()
    _create_directory_alias(state.sessions_directory, outside)

    with pytest.raises(WorkspaceStateError) as captured:
        state.initialize(agent_home_root=Path.home() / ".myclaw")

    assert captured.value.path == state.sessions_directory


def test_workspace_state_initialization_denies_a_hard_linked_memory_file(
    workspace: Path,
) -> None:
    state = WorkspaceState(Workspace.from_path(workspace))
    state.memory_directory.mkdir(parents=True)
    outside = workspace.parent / "outside-memory.md"
    protected_content = b"outside memory must remain unchanged\n"
    outside.write_bytes(protected_content)
    state.long_term_memory_path.hardlink_to(outside)

    with pytest.raises(WorkspaceStateError) as captured:
        state.initialize(agent_home_root=Path.home() / ".myclaw")

    assert captured.value.path == state.long_term_memory_path
    assert outside.read_bytes() == protected_content


@pytest.mark.asyncio
async def test_search_files_skips_hard_links_to_external_files(
    agent_home: Path,
    workspace: Path,
) -> None:
    outside = workspace.parent / "outside-search-secret.txt"
    secret = "SEARCH SECRET MUST NOT BE READ"
    outside.write_text(secret, encoding="utf-8")
    alias = workspace / "search-alias.txt"
    alias.hardlink_to(outside)
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_search_external_hard_link",
            name="search_files",
            arguments=json.dumps({"query": secret}),
        )
    )

    assert result.status == "success"
    assert result.content == ""


@pytest.mark.asyncio
async def test_list_files_omits_hard_links_to_external_files(
    agent_home: Path,
    workspace: Path,
) -> None:
    (workspace / "local.txt").write_text("local", encoding="utf-8")
    outside = workspace.parent / "outside-list-secret.txt"
    outside.write_text("outside", encoding="utf-8")
    alias = workspace / "list-alias.txt"
    alias.hardlink_to(outside)
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_list_external_hard_link",
            name="list_files",
            arguments="{}",
        )
    )

    assert result.status == "success"
    assert result.content == "local.txt"


def test_tool_artifact_publication_uses_the_shared_workspace_state_directory(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    workspace_state = _workspace_state(workspace)
    raw_result = "PERSISTED ARTIFACT CONTENT"
    result = ToolResult(
        tool_call_id="call_workspace_artifact",
        name="read_file",
        status="success",
        content=raw_result,
        artifact=None,
    )

    projected = externalize_tool_result(
        result,
        session=_artifact_session(workspace_state),
        max_tool_result_chars=1,
    )

    assert projected.artifact is not None
    assert projected.artifact.path.startswith(".myclaw/artifacts/")
    assert (workspace / projected.artifact.path).read_text(encoding="utf-8") == raw_result


def test_tool_artifact_publication_overwrites_a_reused_tool_call_id(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    workspace_state = _workspace_state(workspace)
    first_content = "FIRST PRIVATE ARTIFACT"
    second_content = "SECOND PRIVATE ARTIFACT"
    first = ToolResult(
        tool_call_id="reused-call",
        name="read_file",
        status="success",
        content=first_content,
        artifact=None,
    )
    second = ToolResult(
        tool_call_id="reused-call",
        name="read_file",
        status="success",
        content=second_content,
        artifact=None,
    )

    externalize_tool_result(
        first,
        session=_artifact_session(workspace_state),
        max_tool_result_chars=1,
    )
    externalize_tool_result(
        second,
        session=_artifact_session(workspace_state),
        max_tool_result_chars=1,
    )

    artifact_path = _long_path(_artifact_directory(workspace) / "reused-call.txt")
    assert artifact_path.read_text(encoding="utf-8") == second_content


def test_tool_artifact_externalization_returns_a_new_immutable_result(
    agent_home: Path,
    workspace: Path,
) -> None:
    AgentHome(agent_home).initialize()
    workspace_state = _workspace_state(workspace)
    raw_result = "PERSISTED RAW ARTIFACT"
    original = ToolResult(
        tool_call_id="call_immutable",
        name="read_file",
        status="success",
        content=raw_result,
        artifact=None,
    )

    projected = externalize_tool_result(
        original,
        session=_artifact_session(workspace_state),
        max_tool_result_chars=1,
    )

    artifact_path = _long_path(_artifact_directory(workspace) / "call_immutable.txt")
    assert projected is not original
    assert original.content == raw_result
    assert original.artifact is None
    assert projected.artifact is not None
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_list_files_filters_nested_agent_home_state_by_read_scope(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "local.txt").write_text("local", encoding="utf-8")
    agent_home = workspace / ".myclaw"
    protected_paths = (
        "config.toml",
        "memory/summary.jsonl",
        "memory/.cursor",
        "sessions/workspace/session.jsonl",
        f"sessions/legacy-workspace-slug/artifacts/{OTHER_SESSION_ID}/other.txt",
        "schedule.json",
    )
    for relative in protected_paths:
        target = _long_path(agent_home / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("protected", encoding="utf-8")
    memory = agent_home / "memory" / "memory.md"
    memory.write_text("allowed memory", encoding="utf-8")
    artifact = _long_path(agent_home / "sessions" / "artifacts" / SESSION_ID / "current.txt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("allowed artifact", encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_list_nested_agent_home",
            name="list_files",
            arguments='{"path":".","recursive":true}',
        )
    )

    assert result.status == "success"
    assert result.content == "local.txt"


@pytest.mark.asyncio
async def test_search_files_filters_nested_agent_home_state_by_read_scope(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "local.txt").write_text("scope needle local", encoding="utf-8")
    agent_home = workspace / ".myclaw"
    protected_paths = (
        "config.toml",
        "memory/summary.jsonl",
        "sessions/workspace/session.jsonl",
        f"sessions/legacy-workspace-slug/artifacts/{OTHER_SESSION_ID}/other.txt",
        "schedule.json",
    )
    for relative in protected_paths:
        target = _long_path(agent_home / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("scope needle protected", encoding="utf-8")
    memory = agent_home / "memory" / "memory.md"
    memory.write_text("scope needle memory", encoding="utf-8")
    artifact = _long_path(agent_home / "sessions" / "artifacts" / SESSION_ID / "current.txt")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("scope needle current", encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_search_nested_agent_home",
            name="search_files",
            arguments='{"query":"scope needle","path":"."}',
        )
    )

    assert result.status == "success"
    assert result.content == "local.txt:1:scope needle local"
