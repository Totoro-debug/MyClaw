import os
import subprocess
from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.tools.errors import ToolError
from myclaw.tools.files.workspace_write_tools import EditFileTool, WriteFileTool
from myclaw.tools.models import ModelToolCall
from myclaw.tools.security import Security
from myclaw.tools.tool_gateway import ToolGateway

SESSION_ID = "20260712-190000-000000_550e8400-e29b-41d4-a716-446655440000"


def _security(*, agent_home: Path, workspace: Path) -> Security:
    identity = Workspace.from_path(workspace)
    return Security(
        workspace=identity,
        agent_home=agent_home,
        artifact_directory=(
            agent_home / "sessions" / identity.slug / "artifacts" / SESSION_ID
        ),
    )


def _tools(*, agent_home: Path, workspace: Path) -> tuple[WriteFileTool, EditFileTool]:
    security = _security(agent_home=agent_home, workspace=workspace)
    return WriteFileTool(security=security), EditFileTool(security=security)


def _outside_directory_alias(workspace: Path, outside: Path) -> Path:
    alias = workspace / "alias-escape"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as symlink_error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks are unavailable on this host: {symlink_error}")
        try:
            subprocess.run(
                ("cmd", "/c", "mklink", "/J", str(alias), str(outside)),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as junction_error:
            pytest.skip(f"directory aliases are unavailable on this host: {junction_error}")
    return alias


def test_workspace_mutation_tools_export_exact_schemas_and_zero_retries(
    agent_home: Path,
    workspace: Path,
) -> None:
    write_file, edit_file = _tools(agent_home=agent_home, workspace=workspace)

    assert write_file.to_schema() == {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write UTF-8 text to a file within the current Workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace file path.",
                        "minLength": 1,
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete UTF-8 text content.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    }
    assert edit_file.to_schema() == {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace exact UTF-8 text in a file within the current Workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Existing Workspace file path.",
                        "minLength": 1,
                    },
                    "old_text": {
                        "type": "string",
                        "description": "Exact text to replace.",
                        "minLength": 1,
                    },
                    "new_text": {
                        "type": "string",
                        "description": "Replacement text.",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace every exact match.",
                        "default": False,
                    },
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    }
    assert write_file.max_retries == 0
    assert edit_file.max_retries == 0


@pytest.mark.asyncio
async def test_registered_catalog_refuses_workspace_mutations_before_execution(
    agent_home: Path,
    workspace: Path,
) -> None:
    target = workspace / "notes.txt"
    target.write_text("before", encoding="utf-8")
    gateway = ToolGateway()
    gateway.register_tools(_tools(agent_home=agent_home, workspace=workspace))

    write_result = await gateway.call(
        ModelToolCall(
            id="call_write",
            name="write_file",
            arguments='{"path":"created.txt","content":"must not be written"}',
        )
    )
    edit_result = await gateway.call(
        ModelToolCall(
            id="call_edit",
            name="edit_file",
            arguments=(
                '{"path":"notes.txt","old_text":"before","new_text":"after"}'
            ),
        )
    )

    assert write_result.status == "refused"
    assert write_result.content == (
        "Writing Workspace files is unavailable because confirmation is not implemented."
    )
    assert write_result.artifact is None
    assert edit_result.status == "refused"
    assert edit_result.content == (
        "Editing Workspace files is unavailable because confirmation is not implemented."
    )
    assert edit_result.artifact is None
    assert not (workspace / "created.txt").exists()
    assert target.read_text(encoding="utf-8") == "before"


@pytest.mark.asyncio
async def test_write_file_direct_execution_writes_exact_utf8_content(
    agent_home: Path,
    workspace: Path,
) -> None:
    write_file, _ = _tools(agent_home=agent_home, workspace=workspace)

    result = await write_file.execute(path="nested/notes.txt", content="alpha\n界\r\n")

    assert result == "Wrote notes.txt."
    assert (workspace / "nested" / "notes.txt").read_bytes() == "alpha\n界\r\n".encode()


@pytest.mark.asyncio
async def test_edit_file_direct_execution_preserves_unedited_line_endings(
    agent_home: Path,
    workspace: Path,
) -> None:
    target = workspace / "mixed-newlines.txt"
    target.write_bytes(b"before\r\nmiddle\nomega\r\n")
    _, edit_file = _tools(agent_home=agent_home, workspace=workspace)

    result = await edit_file.execute(
        path="mixed-newlines.txt",
        old_text="middle",
        new_text="after",
        replace_all=False,
    )

    assert result == "Edited mixed-newlines.txt."
    assert target.read_bytes() == b"before\r\nafter\nomega\r\n"


@pytest.mark.asyncio
async def test_edit_file_direct_execution_requires_exact_match_unless_replace_all(
    agent_home: Path,
    workspace: Path,
) -> None:
    target = workspace / "repeated.txt"
    target.write_text("before\nbefore\n", encoding="utf-8", newline="")
    _, edit_file = _tools(agent_home=agent_home, workspace=workspace)

    with pytest.raises(ToolError, match="old_text must match exactly once"):
        await edit_file.execute(
            path="repeated.txt",
            old_text="before",
            new_text="after",
            replace_all=False,
        )
    assert target.read_text(encoding="utf-8") == "before\nbefore\n"

    await edit_file.execute(
        path="repeated.txt",
        old_text="before",
        new_text="after",
        replace_all=True,
    )
    assert target.read_text(encoding="utf-8") == "after\nafter\n"


@pytest.mark.asyncio
async def test_edit_file_direct_execution_rejects_non_utf8_without_writing(
    agent_home: Path,
    workspace: Path,
) -> None:
    target = workspace / "binary.dat"
    original = b"before\xffafter"
    target.write_bytes(original)
    _, edit_file = _tools(agent_home=agent_home, workspace=workspace)

    with pytest.raises(ToolError, match="not valid UTF-8"):
        await edit_file.execute(
            path="binary.dat",
            old_text="before",
            new_text="changed",
            replace_all=False,
        )

    assert target.read_bytes() == original


@pytest.mark.asyncio
async def test_direct_mutation_execution_denies_escape_alias_and_agent_home_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret", encoding="utf-8")
    alias = _outside_directory_alias(workspace, outside)
    agent_home = workspace / ".myclaw"
    protected = agent_home / "config.toml"
    protected.parent.mkdir()
    protected.write_text("protected", encoding="utf-8")
    write_file, edit_file = _tools(agent_home=agent_home, workspace=workspace)

    with pytest.raises(ToolError, match="outside the Workspace"):
        await write_file.execute(path="../outside/new.txt", content="escape")
    with pytest.raises(ToolError, match="outside the Workspace"):
        await edit_file.execute(
            path=str(alias / "secret.txt"),
            old_text="secret",
            new_text="escape",
            replace_all=False,
        )
    with pytest.raises(ToolError, match="Agent Home internal state"):
        await write_file.execute(path=".myclaw/config.toml", content="escape")

    assert (outside / "secret.txt").read_text(encoding="utf-8") == "secret"
    assert protected.read_text(encoding="utf-8") == "protected"
    assert not (outside / "new.txt").exists()
