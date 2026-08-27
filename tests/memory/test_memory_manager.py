import ast
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.config.agent_home import AgentHome
from myclaw.memory.manager import MemoryManager
from myclaw.memory.memory_task import MemoryManager as FacadeMemoryManager
from myclaw.memory.records import SummaryEntry
from myclaw.memory.store import (
    WorkspaceJsonlSummaryStore,
    WorkspaceLongTermMemoryStore,
    WorkspaceSummaryCursorStore,
)

NOW = datetime(2026, 8, 27, 10, 0, tzinfo=timezone(timedelta(hours=8)))


def _state(agent_home: Path) -> WorkspaceState:
    workspace = agent_home.parent / "memory-manager-workspace"
    workspace.mkdir()
    state = WorkspaceState(workspace)
    state.initialize(agent_home_root=Path.home() / ".myclaw")
    return state


@pytest.mark.asyncio
async def test_manager_owns_persistence_and_caches_the_startup_snapshot(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    expected = state.long_term_memory_path.read_text(encoding="utf-8")

    manager = MemoryManager(state)

    assert manager.memory_snapshot() == expected
    assert await manager.read_long_term() == expected
    assert manager.long_term_path == state.long_term_memory_path
    assert isinstance(manager._summary_store, WorkspaceJsonlSummaryStore)
    assert isinstance(manager._cursor_store, WorkspaceSummaryCursorStore)
    assert isinstance(manager._long_term_store, WorkspaceLongTermMemoryStore)
    assert len(
        {
            id(manager._summary_store),
            id(manager._cursor_store),
            id(manager._long_term_store),
        }
    ) == 3


@pytest.mark.asyncio
async def test_manager_appends_and_claims_summaries_with_cursor_preadvance(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    manager = MemoryManager(state)
    await manager.append_summary("A durable preference.", NOW)

    claim = await manager.claim_summaries(limit=10)

    assert claim.previous_cursor == 0
    assert claim.cursor == 1
    assert claim.entries == (
        SummaryEntry(index=1, timestamp=NOW, content="A durable preference."),
    )
    assert (state.memory_directory / ".cursor").read_text(encoding="ascii") == "1\n"


@pytest.mark.asyncio
async def test_manager_reads_disk_and_refreshes_snapshot_after_an_edit(
    agent_home: Path,
) -> None:
    home = AgentHome(agent_home)
    home.initialize()
    state = _state(agent_home)
    manager = MemoryManager(state)
    replacement = manager.memory_snapshot().replace(
        "## User Preference\n",
        "## User Preference\n\nPrefers concise reports.\n",
    )

    updated = await manager.edit_long_term(
        old="## User Preference\n",
        new="## User Preference\n\nPrefers concise reports.\n",
    )

    assert updated == replacement
    assert await manager.read_long_term() == replacement
    assert manager.memory_snapshot() == replacement


def test_manager_module_has_no_execution_dependencies() -> None:
    tree = ast.parse(Path("myclaw/memory/manager.py").read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_from_names = {
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    imported = imported_names | imported_from_names

    assert not any(
        name.startswith((
            "myclaw.agent.runner",
            "myclaw.agent.prompts",
            "myclaw.provider",
            "myclaw.tools",
            "myclaw.schedule",
        ))
        for name in imported
    )


def test_memory_task_facade_contains_only_reexports() -> None:
    source = Path("myclaw/memory/memory_task.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert FacadeMemoryManager is MemoryManager
    assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) for node in tree.body)
