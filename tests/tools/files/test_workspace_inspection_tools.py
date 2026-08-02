from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.tools.files.file_tools import (
    EditFileTool,
    ListFilesTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from myclaw.tools.models import ModelToolCall
from myclaw.tools.security import Security
from myclaw.tools.tool_gateway import ToolGateway
from tests.fixtures.log_capture import install_log_capture

SESSION_ID = "20260727-120000-000000_550e8400-e29b-41d4-a716-446655440000"


def _tools(*, agent_home: Path, workspace: Path) -> tuple[ListFilesTool, SearchFilesTool]:
    identity = Workspace.from_path(workspace)
    state = WorkspaceState(identity)
    security = Security(
        workspace=identity,
        agent_home=agent_home,
        artifact_directory=state.sessions_directory / "artifacts" / SESSION_ID,
    )
    return ListFilesTool(security=security), SearchFilesTool(security=security)


def test_workspace_inspection_tools_export_exact_openai_schemas(
    agent_home: Path,
    workspace: Path,
) -> None:
    list_files, search_files = _tools(agent_home=agent_home, workspace=workspace)

    assert list_files.to_schema() == {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and directories within the current Workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory to list.",
                        "default": ".",
                    },
                    "recursive": {
                        "type": "boolean",
                        "description": "Include nested entries.",
                        "default": False,
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": "Maximum entries to return.",
                        "minimum": 1,
                        "maximum": 10000,
                        "default": 1000,
                    },
                },
                "required": [],
            },
        },
    }
    assert search_files.to_schema() == {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search UTF-8 text files within the current Workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Literal text to find.",
                        "minLength": 1,
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search.",
                        "default": ".",
                    },
                    "glob": {
                        "type": ["string", "null"],
                        "description": "Optional path glob used to filter searched files.",
                        "default": None,
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum matches to return.",
                        "minimum": 1,
                        "maximum": 1000,
                        "default": 200,
                    },
                },
                "required": ["query"],
            },
        },
    }
    assert list_files.max_retries == 0
    assert search_files.max_retries == 0


def test_workspace_mutation_tools_export_exact_schemas_and_zero_retries() -> None:
    write_file = WriteFileTool()
    edit_file = EditFileTool()

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
async def test_registered_catalog_refuses_workspace_mutations() -> None:
    gateway = ToolGateway()
    gateway.register_tools((WriteFileTool(), EditFileTool()))

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
            arguments='{"path":"notes.txt","old_text":"before","new_text":"after"}',
        )
    )

    assert (write_result.status, write_result.content) == (
        "refused",
        "Writing Workspace files is unavailable because confirmation is not implemented.",
    )
    assert (edit_result.status, edit_result.content) == (
        "refused",
        "Editing Workspace files is unavailable because confirmation is not implemented.",
    )


@pytest.mark.asyncio
async def test_gateway_prepares_defaults_nullable_glob_and_ignores_unknown_arguments(
    agent_home: Path,
    workspace: Path,
) -> None:
    (workspace / "alpha.txt").write_text("needle one\nneedle two\n", encoding="utf-8")
    list_files, search_files = _tools(agent_home=agent_home, workspace=workspace)
    gateway = ToolGateway()
    gateway.register_tools((list_files, search_files))

    listing = await gateway.call(
        ModelToolCall(
            id="call_list_defaults",
            name="list_files",
            arguments='{"undeclared":"ignored"}',
        )
    )
    search = await gateway.call(
        ModelToolCall(
            id="call_search_nullable",
            name="search_files",
            arguments='{"query":"needle","glob":null,"max_results":"1"}',
        )
    )

    assert listing.status == "success"
    assert listing.content == "alpha.txt"
    assert search.status == "success"
    assert search.content == "alpha.txt:1:needle one"


@pytest.mark.asyncio
async def test_workspace_state_is_omitted_and_rejected_by_listing_and_search(
    agent_home: Path,
    workspace: Path,
) -> None:
    (workspace / "public.txt").write_text("isolation needle public\n", encoding="utf-8")
    state_session = workspace / ".myclaw" / "sessions" / "private.jsonl"
    state_session.parent.mkdir(parents=True)
    state_session.write_text("isolation needle private\n", encoding="utf-8")
    list_files, search_files = _tools(agent_home=agent_home, workspace=workspace)
    gateway = ToolGateway()
    gateway.register_tools((list_files, search_files))

    listing = await gateway.call(
        ModelToolCall(
            id="call_list_workspace_state",
            name="list_files",
            arguments='{"path":".","recursive":true}',
        )
    )
    search = await gateway.call(
        ModelToolCall(
            id="call_search_workspace_state",
            name="search_files",
            arguments='{"query":"isolation needle","path":"."}',
        )
    )
    direct_listing = await gateway.call(
        ModelToolCall(
            id="call_list_workspace_state_direct",
            name="list_files",
            arguments='{"path":".myclaw"}',
        )
    )
    direct_search = await gateway.call(
        ModelToolCall(
            id="call_search_workspace_state_direct",
            name="search_files",
            arguments='{"query":"isolation needle","path":".myclaw"}',
        )
    )

    assert listing.status == "success"
    assert listing.content == "public.txt"
    assert search.status == "success"
    assert search.content == "public.txt:1:isolation needle public"
    assert (direct_listing.status, direct_search.status) == ("error", "error")
    assert "Workspace State" in direct_listing.content
    assert direct_search.content == direct_listing.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "arguments",
    (
        "{}",
        '{"query":""}',
        '{"query":"needle","max_results":0}',
        '{"query":"needle","max_results":1001}',
        '{"query":"needle","glob":false}',
    ),
)
async def test_gateway_rejects_invalid_search_contract_arguments(
    agent_home: Path,
    workspace: Path,
    arguments: str,
) -> None:
    _, search_files = _tools(agent_home=agent_home, workspace=workspace)
    gateway = ToolGateway()
    gateway.register_tools((search_files,))

    result = await gateway.call(
        ModelToolCall(
            id="call_invalid_search",
            name="search_files",
            arguments=arguments,
        )
    )

    assert result.status == "error"
    assert result.content == "Invalid arguments for search_files."


@pytest.mark.asyncio
async def test_read_file_failure_logs_once_without_path_content_or_boundary_detail(
    agent_home: Path,
    workspace: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = Workspace.from_path(workspace)
    state = WorkspaceState(identity)
    security = Security(
        workspace=identity,
        agent_home=agent_home,
        artifact_directory=state.sessions_directory / "artifacts" / SESSION_ID,
    )
    target = workspace / "RAW_TOOL_PATH_51.txt"
    target.write_text("RAW_FILE_CONTENT_51", encoding="utf-8")
    original_read_bytes = Path.read_bytes

    def failing_read_bytes(path: Path) -> bytes:
        if path.name == target.name:
            raise OSError("RAW_FILESYSTEM_BODY_51")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    gateway = ToolGateway()
    gateway.register_tools((ReadFileTool(security=security),))
    lifetime = install_log_capture(AgentHome(agent_home))

    with lifetime.session("foreground-session-51"):
        result = await gateway.call(
            ModelToolCall(
                id="call_read_failure",
                name="read_file",
                arguments='{"path":"RAW_TOOL_PATH_51.txt"}',
            )
        )
    lifetime.close()

    assert (result.status, result.content) == (
        "error",
        "The requested file could not be read.",
    )
    content = (agent_home / "logs" / "run.log.0").read_text(encoding="utf-8")
    assert content.count(" ERROR ") == 1
    assert "name=read_file attempt=1/1 type=ToolError" in content
    assert "session=foreground-session-51" in content
    assert "RAW_TOOL_PATH_51.txt" not in content
    assert "RAW_FILE_CONTENT_51" not in content
    assert "RAW_FILESYSTEM_BODY_51" not in content
