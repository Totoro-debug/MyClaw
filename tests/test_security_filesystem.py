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
from myclaw.tools.files.file_tools import ListFilesTool, ReadFileTool, SearchFilesTool
from myclaw.tools.security import Security
from myclaw.tools.tool_artifacts import ArtifactWriteError, externalize_tool_result
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
        artifact_directory=workspace_state.sessions_directory / "artifacts" / SESSION_ID,
    )
    gateway = ToolGateway()
    gateway.register_tools(
        (
            ReadFileTool(security=security),
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
    return state.sessions_directory / "artifacts" / session_id


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
async def test_read_file_denies_a_hard_link_to_an_external_file(
    agent_home: Path,
    workspace: Path,
) -> None:
    outside = workspace.parent / "outside-secret.txt"
    secret = "EXTERNAL SECRET MUST NOT BE READ"
    outside.write_text(secret, encoding="utf-8")
    alias = workspace / "external-alias.txt"
    alias.hardlink_to(outside)
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_external_hard_link",
            name="read_file",
            arguments=json.dumps({"path": alias.name}),
        )
    )

    assert result.status == "error"
    assert result.content == "The requested path must identify an unaliased regular file."
    assert secret not in result.content


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


def test_tool_artifact_publication_denies_an_external_directory_alias(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    workspace_state = _workspace_state(workspace)
    raw_result = "PRIVATE ARTIFACT CONTENT MUST STAY IN WORKSPACE STATE"
    outside = workspace.parent / "outside-artifacts"
    outside.mkdir()
    _create_directory_alias(workspace_state.sessions_directory / "artifacts", outside)
    result = ToolResult(
        tool_call_id="call_external_artifact_alias",
        name="read_file",
        status="success",
        content=raw_result,
        artifact=None,
    )

    with pytest.raises(ArtifactWriteError):
        externalize_tool_result(
            result,
            session=_artifact_session(workspace_state),
            max_tool_result_chars=1,
        )

    assert list(outside.iterdir()) == []


def test_tool_artifact_publication_never_overwrites_a_reused_tool_call_id(
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
    with pytest.raises(ArtifactWriteError):
        externalize_tool_result(
            second,
            session=_artifact_session(workspace_state),
            max_tool_result_chars=1,
        )

    artifact_path = _long_path(_artifact_directory(workspace) / "reused-call.txt")
    assert artifact_path.read_text(encoding="utf-8") == first_content


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
async def test_read_file_denies_a_windows_alternate_data_stream(
    agent_home: Path,
    workspace: Path,
) -> None:
    base_file = workspace / "notes.txt"
    base_file.write_text("public", encoding="utf-8")
    secret = "ALTERNATE STREAM SECRET MUST NOT BE READ"
    (workspace / "notes.txt:private").write_text(secret, encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_alternate_stream",
            name="read_file",
            arguments='{"path":"notes.txt:private"}',
        )
    )

    assert result.status == "error"
    assert result.content == "The requested path identifies a Windows alternate data stream."
    assert secret not in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("device_path", ("NUL", "CON.txt", "LPT1.", "COM9 "))
async def test_read_file_denies_a_windows_device_path(
    agent_home: Path,
    workspace: Path,
    device_path: str,
) -> None:
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_windows_device",
            name="read_file",
            arguments=json.dumps({"path": device_path}),
        )
    )

    assert result.status == "error"
    assert result.content == "The requested path identifies a Windows device."


@pytest.mark.asyncio
async def test_read_file_prioritizes_agent_home_protection_inside_the_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_home = workspace / ".myclaw"
    agent_home.mkdir()
    secret = "API_KEY=must-not-leak"
    (agent_home / "config.toml").write_text(secret, encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_nested_agent_home_config",
            name="read_file",
            arguments='{"path":".myclaw/config.toml"}',
        )
    )

    assert result.status == "error"
    assert result.content == "Agent Home internal state is not readable by file Tools."
    assert secret not in result.content


@pytest.mark.asyncio
async def test_read_file_allows_exact_long_term_memory_inside_the_workspace(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = _workspace_state(workspace)
    state.long_term_memory_path.write_text("# Long-term Memory\n", encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_long_term_memory",
            name="read_file",
            arguments='{"path":"memory/memory.md"}',
        )
    )

    direct_relative = await gateway.call(
        ModelToolCall(
            id="call_read_direct_long_term_memory",
            name="read_file",
            arguments='{"path":".myclaw/memory/memory.md"}',
        )
    )
    direct_absolute = await gateway.call(
        ModelToolCall(
            id="call_read_absolute_long_term_memory",
            name="read_file",
            arguments=json.dumps({"path": str(state.long_term_memory_path)}),
        )
    )

    assert result.status == "success"
    assert result.content == "# Long-term Memory"
    assert (direct_relative.status, direct_absolute.status) == ("error", "error")
    assert all(
        denied.content == "Workspace State internal files are not readable by file Tools."
        for denied in (direct_relative, direct_absolute)
    )


@pytest.mark.asyncio
async def test_legacy_agent_home_memory_remains_untouched_and_unread(
    agent_home: Path,
    workspace: Path,
) -> None:
    legacy = agent_home / "memory" / "memory.md"
    legacy.parent.mkdir(parents=True)
    legacy_bytes = b"legacy Agent Home memory secret"
    legacy.write_bytes(legacy_bytes)
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    direct = await gateway.call(
        ModelToolCall(
            id="call_read_legacy_memory",
            name="read_file",
            arguments=json.dumps({"path": str(legacy)}),
        )
    )

    assert direct.status == "error"
    assert "legacy Agent Home memory secret" not in direct.content
    assert legacy.read_bytes() == legacy_bytes


@pytest.mark.asyncio
async def test_read_file_denies_an_aliased_long_term_memory_location(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    protected_directory = agent_home / "protected-memory"
    protected_directory.mkdir()
    secret = "AGENT HOME INTERNAL SECRET"
    (protected_directory / "memory.md").write_text(secret, encoding="utf-8")
    _create_directory_alias(agent_home / "memory", protected_directory)
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_aliased_long_term_memory",
            name="read_file",
            arguments=json.dumps({"path": str(agent_home / "memory" / "memory.md")}),
        )
    )

    assert result.status == "error"
    assert result.content == "Agent Home internal state is not readable by file Tools."
    assert secret not in result.content


@pytest.mark.asyncio
async def test_read_file_allows_the_active_session_artifact_reference(
    agent_home: Path,
    workspace: Path,
) -> None:
    _workspace_state(workspace)
    artifact = _long_path(_artifact_directory(workspace) / "call_result.txt")
    artifact.parent.mkdir(parents=True)
    artifact.write_text("current session artifact", encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_current_artifact",
            name="read_file",
            arguments=json.dumps({"path": f"artifacts/{SESSION_ID}/call_result.txt"}),
        )
    )

    assert result.status == "success"
    assert result.content == "current session artifact"


@pytest.mark.asyncio
async def test_read_file_denies_an_aliased_active_session_artifact_directory(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    _workspace_state(workspace)
    protected_directory = agent_home / "protected-artifacts"
    protected_directory.mkdir()
    secret = "OTHER AGENT HOME ARTIFACT SECRET"
    (protected_directory / "call_result.txt").write_text(secret, encoding="utf-8")
    artifact_directory = _long_path(_artifact_directory(workspace))
    artifact_directory.parent.mkdir(parents=True)
    _create_directory_alias(artifact_directory, _long_path(protected_directory))
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_aliased_current_artifact",
            name="read_file",
            arguments=json.dumps({"path": f"artifacts/{SESSION_ID}/call_result.txt"}),
        )
    )

    assert result.status == "error"
    assert result.content == "Agent Home internal state is not readable by file Tools."
    assert secret not in result.content


@pytest.mark.asyncio
async def test_read_file_denies_other_session_and_direct_workspace_state_artifacts(
    agent_home: Path,
    workspace: Path,
) -> None:
    _workspace_state(workspace)
    current = _long_path(_artifact_directory(workspace) / "current.txt")
    other = _long_path(_artifact_directory(workspace, OTHER_SESSION_ID) / "other.txt")
    for path, content in ((current, "current artifact"), (other, "other artifact secret")):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    direct_relative = await gateway.call(
        ModelToolCall(
            id="call_direct_state_artifact",
            name="read_file",
            arguments=json.dumps({"path": f".myclaw/sessions/artifacts/{SESSION_ID}/current.txt"}),
        )
    )
    direct_absolute = await gateway.call(
        ModelToolCall(
            id="call_absolute_state_artifact",
            name="read_file",
            arguments=json.dumps({"path": str(current)}),
        )
    )
    other_alias = await gateway.call(
        ModelToolCall(
            id="call_other_session_artifact",
            name="read_file",
            arguments=json.dumps({"path": f"artifacts/{OTHER_SESSION_ID}/other.txt"}),
        )
    )

    assert {direct_relative.status, direct_absolute.status, other_alias.status} == {"error"}
    assert all(
        result.content == "Workspace State internal files are not readable by file Tools."
        for result in (direct_relative, direct_absolute, other_alias)
    )
    assert "other artifact secret" not in str((direct_relative, direct_absolute, other_alias))


@pytest.mark.asyncio
async def test_legacy_agent_home_artifact_remains_untouched_and_unread(
    agent_home: Path,
    workspace: Path,
) -> None:
    legacy = _long_path(
        agent_home / "sessions" / "legacy-workspace-slug" / "artifacts" / SESSION_ID / "legacy.txt"
    )
    legacy.parent.mkdir(parents=True)
    legacy_bytes = b"legacy Agent Home artifact secret"
    legacy.write_bytes(legacy_bytes)
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    alias = await gateway.call(
        ModelToolCall(
            id="call_legacy_artifact_alias",
            name="read_file",
            arguments=json.dumps({"path": f"artifacts/{SESSION_ID}/legacy.txt"}),
        )
    )
    direct = await gateway.call(
        ModelToolCall(
            id="call_legacy_artifact_direct",
            name="read_file",
            arguments=json.dumps({"path": str(legacy)}),
        )
    )

    assert (alias.status, direct.status) == ("error", "error")
    assert "legacy Agent Home artifact secret" not in alias.content + direct.content
    assert legacy.read_bytes() == legacy_bytes


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
