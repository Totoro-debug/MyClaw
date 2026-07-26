from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.tools.files.file_tools import ListFilesTool, SearchFilesTool
from myclaw.tools.models import ModelToolCall
from myclaw.tools.security import Security
from myclaw.tools.tool_gateway import ToolGateway

SESSION_ID = "20260727-120000-000000_550e8400-e29b-41d4-a716-446655440000"


def _tools(*, agent_home: Path, workspace: Path) -> tuple[ListFilesTool, SearchFilesTool]:
    identity = Workspace.from_path(workspace)
    security = Security(
        workspace=identity,
        agent_home=agent_home,
        artifact_directory=(
            agent_home / "sessions" / identity.slug / "artifacts" / SESSION_ID
        ),
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
