import json
import os
import subprocess
from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.config.agent_home import AgentHome
from myclaw.provider.models import AssistantModelMessage
from myclaw.tools.errors import ToolError
from myclaw.tools.files.file_tools import ListFilesTool, ReadFileTool, SearchFilesTool
from myclaw.tools.files.workspace_write_tools import WriteFileTool
from myclaw.tools.models import ModelToolCall, ToolExecutionContext, ToolResult
from myclaw.tools.security import Security
from myclaw.tools.tool_artifacts import ArtifactWriteError, externalize_tool_result
from myclaw.tools.tool_gateway import ToolGateway

SESSION_ID = "20260713-040000-000000_550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION_ID = "20260713-050000-000000_550e8400-e29b-41d4-a716-446655440000"


def _read_file_gateway(*, agent_home: Path, workspace: Path) -> ToolGateway:
    workspace_identity = Workspace.from_path(workspace)
    security = Security(
        workspace=workspace_identity,
        agent_home=agent_home,
        artifact_directory=(
            agent_home
            / "sessions"
            / workspace_identity.slug
            / "artifacts"
            / SESSION_ID
        ),
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


def _create_directory_alias(alias: Path, target: Path) -> None:
    try:
        alias.symlink_to(target, target_is_directory=True)
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable on this host: {symlink_error}")
        try:
            subprocess.run(
                ("cmd", "/c", "mklink", "/J", str(alias), str(target)),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as junction_error:
            pytest.skip(f"directory aliases are unavailable on this host: {junction_error}")


def _long_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    return Path(f"\\\\?\\{path.absolute()}")


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


def test_agent_home_initialization_denies_an_external_memory_directory_alias(
    agent_home: Path,
) -> None:
    agent_home.mkdir(parents=True)
    outside = agent_home.parent / "outside-memory"
    outside.mkdir()
    _create_directory_alias(agent_home / "memory", outside)

    with pytest.raises(PermissionError):
        AgentHome(agent_home).initialize()

    assert not (outside / "memory.md").exists()


def test_agent_home_initialization_denies_an_external_sessions_directory_alias(
    agent_home: Path,
) -> None:
    agent_home.mkdir(parents=True)
    outside = agent_home.parent / "outside-sessions"
    outside.mkdir()
    _create_directory_alias(agent_home / "sessions", outside)

    with pytest.raises(PermissionError):
        AgentHome(agent_home).initialize()


def test_agent_home_initialization_denies_a_hard_linked_memory_file(
    agent_home: Path,
) -> None:
    memory_directory = agent_home / "memory"
    memory_directory.mkdir(parents=True)
    outside = agent_home.parent / "outside-memory.md"
    protected_content = b"outside memory must remain unchanged\n"
    outside.write_bytes(protected_content)
    try:
        (memory_directory / "memory.md").hardlink_to(outside)
    except OSError as error:
        pytest.skip(f"file hard links are unavailable on this host: {error}")

    with pytest.raises(PermissionError):
        AgentHome(agent_home).initialize()

    assert outside.read_bytes() == protected_content


@pytest.mark.asyncio
async def test_write_file_denies_a_hard_link_to_agent_home_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    protected = agent_home / "memory" / "memory.md"
    protected_content = b"# Protected memory\n"
    protected.write_bytes(protected_content)
    alias = workspace / "memory-alias.md"
    try:
        alias.hardlink_to(protected)
    except OSError as error:
        pytest.skip(f"file hard links are unavailable on this host: {error}")
    identity = Workspace.from_path(workspace)
    tool = WriteFileTool(
        security=Security(
            workspace=identity,
            agent_home=agent_home,
            artifact_directory=(
                agent_home / "sessions" / identity.slug / "artifacts" / SESSION_ID
            ),
        )
    )

    with pytest.raises(ToolError, match="unalias"):
        await tool.execute(path=alias.name, content="overwritten")

    assert protected.read_bytes() == protected_content


@pytest.mark.asyncio
async def test_read_file_denies_a_hard_link_to_an_external_file(
    agent_home: Path,
    workspace: Path,
) -> None:
    outside = workspace.parent / "outside-secret.txt"
    secret = "EXTERNAL SECRET MUST NOT BE READ"
    outside.write_text(secret, encoding="utf-8")
    alias = workspace / "external-alias.txt"
    try:
        alias.hardlink_to(outside)
    except OSError as error:
        pytest.skip(f"file hard links are unavailable on this host: {error}")
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
    try:
        alias.hardlink_to(outside)
    except OSError as error:
        pytest.skip(f"file hard links are unavailable on this host: {error}")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

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
    try:
        alias.hardlink_to(outside)
    except OSError as error:
        pytest.skip(f"file hard links are unavailable on this host: {error}")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

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
    raw_result = "PRIVATE ARTIFACT CONTENT MUST STAY IN AGENT HOME"
    outside = agent_home.parent / "outside-artifacts"
    outside.mkdir()
    sessions = agent_home / "sessions"
    sessions.rmdir()
    _create_directory_alias(sessions, outside)
    result = ToolResult(
        tool_call_id="call_external_artifact_alias",
        name="read_file",
        status="success",
        content=raw_result,
        error=None,
        artifact=None,
    )

    with pytest.raises(ArtifactWriteError):
        externalize_tool_result(
            result,
            agent_home=agent_home,
            workspace=Workspace.from_path(workspace),
            session_id=SESSION_ID,
            max_tool_result_chars=1,
        )

    assert list(outside.iterdir()) == []


def test_tool_artifact_publication_never_overwrites_a_reused_tool_call_id(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    first_content = "FIRST PRIVATE ARTIFACT"
    second_content = "SECOND PRIVATE ARTIFACT"
    first = ToolResult(
        tool_call_id="reused-call",
        name="read_file",
        status="success",
        content=first_content,
        error=None,
        artifact=None,
    )
    second = ToolResult(
        tool_call_id="reused-call",
        name="read_file",
        status="success",
        content=second_content,
        error=None,
        artifact=None,
    )

    externalize_tool_result(
        first,
        agent_home=agent_home,
        workspace=Workspace.from_path(workspace),
        session_id=SESSION_ID,
        max_tool_result_chars=1,
    )
    with pytest.raises(ArtifactWriteError):
        externalize_tool_result(
            second,
            agent_home=agent_home,
            workspace=Workspace.from_path(workspace),
            session_id=SESSION_ID,
            max_tool_result_chars=1,
        )

    artifact_path = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "reused-call.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == first_content


def test_tool_artifact_externalization_returns_a_new_immutable_result(
    agent_home: Path,
    workspace: Path,
) -> None:
    AgentHome(agent_home).initialize()
    raw_result = "PERSISTED RAW ARTIFACT"
    original = ToolResult(
        tool_call_id="call_immutable",
        name="read_file",
        status="success",
        content=raw_result,
        error=None,
        artifact=None,
    )

    projected = externalize_tool_result(
        original,
        agent_home=agent_home,
        workspace=Workspace.from_path(workspace),
        session_id=SESSION_ID,
        max_tool_result_chars=1,
    )

    artifact_path = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "call_immutable.txt"
    )
    assert projected is not original
    assert original.content == raw_result
    assert original.artifact is None
    assert projected.artifact is not None
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate streams are Windows-only")
async def test_write_file_denies_a_windows_alternate_data_stream(
    agent_home: Path,
    workspace: Path,
) -> None:
    identity = Workspace.from_path(workspace)
    tool = WriteFileTool(
        security=Security(
            workspace=identity,
            agent_home=agent_home,
            artifact_directory=(
                agent_home / "sessions" / identity.slug / "artifacts" / SESSION_ID
            ),
        )
    )

    with pytest.raises(ToolError, match="alternate data stream"):
        await tool.execute(path="notes.txt:private", content="hidden")

    assert not (workspace / "notes.txt").exists()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate streams are Windows-only")
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
@pytest.mark.skipif(os.name != "nt", reason="Windows device paths are Windows-only")
async def test_read_file_denies_a_windows_device_path(
    agent_home: Path,
    workspace: Path,
) -> None:
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_windows_device",
            name="read_file",
            arguments='{"path":"NUL"}',
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
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    agent_home = workspace / ".myclaw"
    memory = agent_home / "memory" / "memory.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("# Long-term Memory\n", encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_long_term_memory",
            name="read_file",
            arguments='{"path":".myclaw/memory/memory.md"}',
        )
    )

    assert result.status == "success"
    assert result.content == "# Long-term Memory"


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
async def test_read_file_allows_the_current_session_artifact_reference(
    agent_home: Path,
    workspace: Path,
) -> None:
    artifact = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "call_result.txt"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text("current session artifact", encoding="utf-8")
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_current_artifact",
            name="read_file",
            arguments=json.dumps(
                {"path": f"artifacts/{SESSION_ID}/call_result.txt"}
            ),
        )
    )

    assert result.status == "success"
    assert result.content == "current session artifact"


@pytest.mark.asyncio
async def test_read_file_denies_an_aliased_current_session_artifact_directory(
    agent_home: Path,
    workspace: Path,
) -> None:
    agent_home.mkdir(parents=True)
    protected_directory = agent_home / "protected-artifacts"
    protected_directory.mkdir()
    secret = "OTHER AGENT HOME ARTIFACT SECRET"
    (protected_directory / "call_result.txt").write_text(secret, encoding="utf-8")
    artifact_directory = _long_path(
        agent_home / "sessions" / Workspace.from_path(workspace).slug / "artifacts" / SESSION_ID
    )
    artifact_directory.parent.mkdir(parents=True)
    _create_directory_alias(artifact_directory, _long_path(protected_directory))
    gateway = _read_file_gateway(agent_home=agent_home, workspace=workspace)

    result = await gateway.call(
        ModelToolCall(
            id="call_read_aliased_current_artifact",
            name="read_file",
            arguments=json.dumps(
                {"path": f"artifacts/{SESSION_ID}/call_result.txt"}
            ),
        )
    )

    assert result.status == "error"
    assert result.content == "Agent Home internal state is not readable by file Tools."
    assert secret not in result.content


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
        f"sessions/{Workspace.from_path(workspace).slug}/artifacts/{OTHER_SESSION_ID}/other.txt",
        "scheduled-work.json",
    )
    for relative in protected_paths:
        target = _long_path(agent_home / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("protected", encoding="utf-8")
    memory = agent_home / "memory" / "memory.md"
    memory.write_text("allowed memory", encoding="utf-8")
    artifact = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "current.txt"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("allowed artifact", encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_list_nested_agent_home",
            name="list_files",
            arguments='{"path":".","recursive":true}',
        )
    )

    assert result.status == "success"
    assert result.content == (
        f"artifacts/{SESSION_ID}/\nartifacts/{SESSION_ID}/current.txt\nlocal.txt\nmemory/memory.md"
    )


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
        f"sessions/{Workspace.from_path(workspace).slug}/artifacts/{OTHER_SESSION_ID}/other.txt",
        "scheduled-work.json",
    )
    for relative in protected_paths:
        target = _long_path(agent_home / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("scope needle protected", encoding="utf-8")
    memory = agent_home / "memory" / "memory.md"
    memory.write_text("scope needle memory", encoding="utf-8")
    artifact = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "current.txt"
    )
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text("scope needle current", encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.call(
        ModelToolCall(
            id="call_search_nested_agent_home",
            name="search_files",
            arguments='{"query":"scope needle","path":"."}',
        )
    )

    assert result.status == "success"
    assert result.content == (
        f"artifacts/{SESSION_ID}/current.txt:1:scope needle current\n"
        "local.txt:1:scope needle local\n"
        "memory/memory.md:1:scope needle memory"
    )
