from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState
from myclaw.tools.core.edit_file import EditFileTool
from myclaw.tools.core.write_file import WriteFileTool
from myclaw.tools.files.file_tools import ListFilesTool, SearchFilesTool
from myclaw.tools.security import Security
from myclaw.tools.tool_gateway import ModelToolCall, ToolGateway

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


def test_workspace_mutation_tools_export_exact_schemas_and_zero_retries(
    workspace: Path,
) -> None:
    identity = Workspace.from_path(workspace)
    write_file = WriteFileTool(workspace=identity)
    edit_file = EditFileTool(workspace=identity)

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
                        "description": "Workspace-relative or absolute file path.",
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
                        "description": "Workspace-relative or absolute file path.",
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
async def test_registered_catalog_executes_internal_workspace_mutations(workspace: Path) -> None:
    identity = Workspace.from_path(workspace)
    (workspace / "notes.txt").write_text("before", encoding="utf-8")
    gateway = ToolGateway()
    gateway.register_tools((WriteFileTool(workspace=identity), EditFileTool(workspace=identity)))

    write_result = await gateway.call(
        ModelToolCall(
            id="call_write",
            name="write_file",
            arguments='{"path":"created.txt","content":"written"}',
        )
    )
    edit_result = await gateway.call(
        ModelToolCall(
            id="call_edit",
            name="edit_file",
            arguments='{"path":"notes.txt","old_text":"before","new_text":"after"}',
        )
    )

    assert write_result.status == "success"
    assert edit_result.status == "success"
    assert (workspace / "created.txt").read_text(encoding="utf-8") == "written"
    assert (workspace / "notes.txt").read_text(encoding="utf-8") == "after"


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
