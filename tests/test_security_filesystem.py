import os
import subprocess
from pathlib import Path

import pytest

from myclaw.agent_home import AgentHome
from myclaw.contracts import AssistantModelMessage, ModelToolCall, ToolExecutionContext
from myclaw.tool_artifacts import ArtifactDiscardError
from myclaw.tool_gateway import ToolGateway
from myclaw.workspace import Workspace

SESSION_ID = "20260713-040000-000000_550e8400-e29b-41d4-a716-446655440000"
OTHER_SESSION_ID = "20260713-050000-000000_550e8400-e29b-41d4-a716-446655440000"


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
    first = ModelToolCall(id="duplicate-call", name="read_file", arguments={"path": "a.txt"})
    second = ModelToolCall(
        id="duplicate-call",
        name="read_file",
        arguments={"path": "b.txt"},
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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )
    tool_call = ModelToolCall(
        id="call_write_agent_home_hard_link",
        name="write_file",
        arguments={"path": alias.name, "content": "overwritten"},
    )

    assert gateway.permission_request(tool_call) is None
    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_external_hard_link",
            name="read_file",
            arguments={"path": alias.name},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
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

    result = await gateway.execute(
        ModelToolCall(
            id="call_search_external_hard_link",
            name="search_files",
            arguments={"query": secret},
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

    result = await gateway.execute(
        ModelToolCall(
            id="call_list_external_hard_link",
            name="list_files",
            arguments={},
        )
    )

    assert result.status == "success"
    assert result.content == "local.txt"


@pytest.mark.asyncio
async def test_tool_artifact_publication_denies_an_external_directory_alias(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    raw_result = "PRIVATE ARTIFACT CONTENT MUST STAY IN AGENT HOME"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    outside = agent_home.parent / "outside-artifacts"
    outside.mkdir()
    sessions = agent_home / "sessions"
    sessions.rmdir()
    _create_directory_alias(sessions, outside)
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        max_tool_result_chars=1,
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_external_artifact_alias",
            name="read_file",
            arguments={"path": "large.txt"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_failed"
    assert raw_result not in result.content
    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_tool_artifact_publication_never_overwrites_a_reused_tool_call_id(
    agent_home: Path,
    workspace: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    first_content = "FIRST PRIVATE ARTIFACT"
    second_content = "SECOND PRIVATE ARTIFACT"
    (workspace / "first.txt").write_text(first_content, encoding="utf-8")
    (workspace / "second.txt").write_text(second_content, encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        max_tool_result_chars=1,
    )
    first_call = ModelToolCall(
        id="reused-call",
        name="read_file",
        arguments={"path": "first.txt"},
    )
    second_call = ModelToolCall(
        id="reused-call",
        name="read_file",
        arguments={"path": "second.txt"},
    )

    first = await gateway.execute(first_call)
    second = await gateway.execute(second_call)

    assert first.status == "success"
    assert second.status == "error"
    assert second.error is not None
    assert second.error.code == "tool_failed"
    assert gateway.discard_artifact(second) is False
    artifact_path = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "reused-call.txt"
    )
    assert artifact_path.read_text(encoding="utf-8") == first_content


@pytest.mark.asyncio
async def test_tool_gateway_discards_an_unpersisted_artifact_it_created(
    agent_home: Path,
    workspace: Path,
) -> None:
    AgentHome(agent_home).initialize()
    raw_result = "UNPERSISTED RAW ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        max_tool_result_chars=1,
    )
    result = await gateway.execute(
        ModelToolCall(
            id="call_discard_unpersisted",
            name="read_file",
            arguments={"path": "large.txt"},
        )
    )
    artifact_path = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "call_discard_unpersisted.txt"
    )
    assert result.artifact is not None
    assert artifact_path.read_text(encoding="utf-8") == raw_result

    discarded = gateway.discard_artifact(result)

    assert discarded is True
    assert not artifact_path.exists()


@pytest.mark.asyncio
async def test_tool_gateway_commits_a_persisted_artifact_without_deleting_it(
    agent_home: Path,
    workspace: Path,
) -> None:
    AgentHome(agent_home).initialize()
    raw_result = "PERSISTED RAW ARTIFACT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        max_tool_result_chars=1,
    )
    result = await gateway.execute(
        ModelToolCall(
            id="call_commit_persisted",
            name="read_file",
            arguments={"path": "large.txt"},
        )
    )
    artifact_path = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "call_commit_persisted.txt"
    )
    assert result.artifact is not None
    assert artifact_path.read_text(encoding="utf-8") == raw_result

    committed = gateway.commit_artifact(result)

    assert committed is True
    assert artifact_path.read_text(encoding="utf-8") == raw_result
    assert gateway.discard_artifact(result) is False
    assert artifact_path.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_tool_gateway_never_discards_a_replacement_at_a_tracked_artifact_path(
    agent_home: Path,
    workspace: Path,
) -> None:
    AgentHome(agent_home).initialize()
    raw_result = "ORIGINAL OWNED ARTIFACT"
    replacement = "UNRELATED REPLACEMENT"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        max_tool_result_chars=1,
    )
    result = await gateway.execute(
        ModelToolCall(
            id="call_replaced_artifact",
            name="read_file",
            arguments={"path": "large.txt"},
        )
    )
    artifact_path = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "call_replaced_artifact.txt"
    )
    moved_original = artifact_path.with_name("moved-original.txt")
    artifact_path.rename(moved_original)
    artifact_path.write_text(replacement, encoding="utf-8")

    with pytest.raises(ArtifactDiscardError) as raised:
        gateway.discard_artifact(result)

    assert raw_result not in str(raised.value)
    assert artifact_path.read_text(encoding="utf-8") == replacement
    assert moved_original.read_text(encoding="utf-8") == raw_result


@pytest.mark.asyncio
async def test_tool_gateway_removes_a_just_created_artifact_when_tracking_validation_fails(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    AgentHome(agent_home).initialize()
    raw_result = "RAW ARTIFACT MUST BE ROLLED BACK"
    (workspace / "large.txt").write_text(raw_result, encoding="utf-8")
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        ),
        max_tool_result_chars=1,
    )
    artifact_path = _long_path(
        agent_home
        / "sessions"
        / Workspace.from_path(workspace).slug
        / "artifacts"
        / SESSION_ID
        / "call_tracking_validation_fault.txt"
    )
    real_resolve = Path.resolve

    def fail_post_write_artifact_resolution(
        path: Path,
        *,
        strict: bool = False,
    ) -> Path:
        if path.name == artifact_path.name and path.exists():
            raise OSError("injected post-write resolution failure")
        return real_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "resolve", fail_post_write_artifact_resolution)

    result = await gateway.execute(
        ModelToolCall(
            id="call_tracking_validation_fault",
            name="read_file",
            arguments={"path": "large.txt"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_failed"
    assert raw_result not in result.content
    assert not artifact_path.exists()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="NTFS alternate streams are Windows-only")
async def test_write_file_denies_a_windows_alternate_data_stream(
    agent_home: Path,
    workspace: Path,
) -> None:
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )
    tool_call = ModelToolCall(
        id="call_write_alternate_stream",
        name="write_file",
        arguments={"path": "notes.txt:private", "content": "hidden"},
    )

    assert gateway.permission_request(tool_call) is None
    result = await gateway.execute(tool_call, approved=True)

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_alternate_stream",
            name="read_file",
            arguments={"path": "notes.txt:private"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
    assert secret not in result.content


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="Windows device paths are Windows-only")
async def test_read_file_denies_a_windows_device_path(
    agent_home: Path,
    workspace: Path,
) -> None:
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_windows_device",
            name="read_file",
            arguments={"path": "NUL"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"


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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_nested_agent_home_config",
            name="read_file",
            arguments={"path": ".myclaw/config.toml"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_long_term_memory",
            name="read_file",
            arguments={"path": ".myclaw/memory/memory.md"},
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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_aliased_long_term_memory",
            name="read_file",
            arguments={"path": str(agent_home / "memory" / "memory.md")},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_current_artifact",
            name="read_file",
            arguments={"path": f"artifacts/{SESSION_ID}/call_result.txt"},
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
    gateway = ToolGateway(
        context=ToolExecutionContext(
            lane="foreground",
            workspace=workspace,
            agent_home=agent_home,
            session_id=SESSION_ID,
        )
    )

    result = await gateway.execute(
        ModelToolCall(
            id="call_read_aliased_current_artifact",
            name="read_file",
            arguments={"path": f"artifacts/{SESSION_ID}/call_result.txt"},
        )
    )

    assert result.status == "error"
    assert result.error is not None
    assert result.error.code == "tool_denied"
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

    result = await gateway.execute(
        ModelToolCall(
            id="call_list_nested_agent_home",
            name="list_files",
            arguments={"path": ".", "recursive": True},
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

    result = await gateway.execute(
        ModelToolCall(
            id="call_search_nested_agent_home",
            name="search_files",
            arguments={"query": "scope needle", "path": "."},
        )
    )

    assert result.status == "success"
    assert result.content == (
        f"artifacts/{SESSION_ID}/current.txt:1:scope needle current\n"
        "local.txt:1:scope needle local\n"
        "memory/memory.md:1:scope needle memory"
    )
