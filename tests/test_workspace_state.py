import shutil
import subprocess
from pathlib import Path

import pytest

from myclaw.agent.workspace import Workspace
from myclaw.agent.workspace_state import WorkspaceState, WorkspaceStateError
from myclaw.tools.errors import ToolError
from myclaw.tools.security import Security

EXPECTED_MEMORY = (
    "# Long-term Memory\n\n## User Info\n\n## User Preference\n\n## Project Fact\n\n## Lesson\n"
)


def state_for(workspace: Path) -> WorkspaceState:
    return WorkspaceState(Workspace.from_path(workspace))


def test_initialization_rejects_agent_home_as_workspace(
    agent_home: Path,
) -> None:
    user_home = agent_home.parent

    with pytest.raises(WorkspaceStateError) as captured:
        state_for(user_home).initialize(agent_home_root=agent_home)

    assert captured.value.path == agent_home
    assert not (agent_home / "memory").exists()
    assert not (agent_home / "sessions").exists()


def test_initialization_rejects_workspace_beneath_agent_home_without_reading_legacy_state(
    agent_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_files = {
        agent_home / "memory" / "memory.md": b"legacy memory\r\n",
        agent_home / "sessions" / "legacy-session.jsonl": b"legacy session\r\n",
        agent_home / "scheduled-work.json": b"legacy scheduled work\r\n",
        agent_home / "sessions" / "artifacts" / "legacy" / "tool.txt": b"legacy artifact\r\n",
    }
    for path, content in legacy_files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    workspace = agent_home / "nested-workspace"
    workspace.mkdir()
    state = state_for(workspace)
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def reject_legacy_bytes(path: Path) -> bytes:
        if path in legacy_files:
            raise AssertionError(f"legacy state was read: {path}")
        return original_read_bytes(path)

    def reject_legacy_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path in legacy_files:
            raise AssertionError(f"legacy state was read: {path}")
        return original_read_text(path, encoding=encoding, errors=errors)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "read_bytes", reject_legacy_bytes)
        guarded.setattr(Path, "read_text", reject_legacy_text)
        with pytest.raises(WorkspaceStateError) as captured:
            state.initialize(agent_home_root=agent_home)

    assert captured.value.path == state.path
    assert not state.path.exists()
    assert {path: path.read_bytes() for path in legacy_files} == legacy_files


def test_initialization_rejects_case_and_junction_aliases_of_agent_home(
    agent_home: Path,
    tmp_path: Path,
) -> None:
    user_home = agent_home.parent
    case_alias = Path(str(user_home).swapcase())
    junction_alias = tmp_path / "user-home-junction"
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction_alias), str(user_home)],
        capture_output=True,
        check=True,
        text=True,
    )

    for workspace in (case_alias, junction_alias):
        with pytest.raises(WorkspaceStateError):
            state_for(workspace).initialize(agent_home_root=agent_home)

    assert not agent_home.exists()


def test_initialization_accepts_project_beneath_user_home(agent_home: Path) -> None:
    workspace = agent_home.parent / "project"
    workspace.mkdir()
    state = state_for(workspace)

    state.initialize(agent_home_root=agent_home)

    assert state.memory_directory.is_dir()
    assert state.sessions_directory.is_dir()
    assert state.long_term_memory_path.read_text(encoding="utf-8") == EXPECTED_MEMORY


def test_initialization_creates_only_required_base_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = state_for(workspace)

    state.initialize(agent_home_root=agent_home)

    tree = tuple(
        sorted(
            "/".join(path.relative_to(state.path).parts) + ("/" if path.is_dir() else "")
            for path in state.path.rglob("*")
        )
    )
    assert tree == (".gitignore", "memory/", "memory/memory.md", "sessions/")
    assert (state.path / ".gitignore").read_text(encoding="utf-8") == "*\n"
    assert state.long_term_memory_path.read_text(encoding="utf-8") == EXPECTED_MEMORY


def test_repeated_initialization_preserves_policy_memory_and_unknown_entries(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = state_for(workspace)
    state.path.mkdir()
    gitignore = state.path / ".gitignore"
    gitignore.write_bytes(b"!memory/memory.md\r\n")
    unknown = state.path / "future-state.bin"
    unknown.write_bytes(b"future schema")
    state.memory_directory.mkdir()
    state.long_term_memory_path.write_bytes(b"# Existing memory\r\n")

    state.initialize(agent_home_root=agent_home)
    state.initialize(agent_home_root=agent_home)

    assert gitignore.read_bytes() == b"!memory/memory.md\r\n"
    assert unknown.read_bytes() == b"future schema"
    assert state.long_term_memory_path.read_bytes() == b"# Existing memory\r\n"


def test_copying_a_complete_workspace_retains_its_workspace_state(
    agent_home: Path,
    workspace: Path,
    tmp_path: Path,
) -> None:
    state = state_for(workspace)
    state.initialize(agent_home_root=agent_home)
    state.long_term_memory_path.write_bytes(b"# Portable memory\n")
    session = state.sessions_directory / "portable-session.jsonl"
    session.write_bytes(b'{"portable_test":true}\n')
    copied_workspace = tmp_path / "copied-workspace"

    shutil.copytree(workspace, copied_workspace)
    copied_state = state_for(copied_workspace)

    assert copied_state.long_term_memory_path.read_bytes() == b"# Portable memory\n"
    assert (copied_state.sessions_directory / session.name).read_bytes() == session.read_bytes()
    assert copied_state.path != state.path
    assert not (copied_workspace / agent_home.name / "config.toml").exists()


def test_generic_file_security_rejects_workspace_state(
    agent_home: Path,
    workspace: Path,
) -> None:
    state = state_for(workspace)
    state.initialize(agent_home_root=agent_home)
    security = Security(
        workspace=state.workspace,
        agent_home=agent_home,
        artifact_directory=state.sessions_directory / "artifacts" / "session",
    )

    with pytest.raises(ToolError, match="Workspace State"):
        security.resolve_read_path(str(state.long_term_memory_path))


def test_initialization_rejects_non_directory_root(
    agent_home: Path,
    workspace: Path,
) -> None:
    root = workspace / ".myclaw"
    root.write_text("collision", encoding="utf-8")

    with pytest.raises(WorkspaceStateError) as captured:
        state_for(workspace).initialize(agent_home_root=agent_home)

    assert captured.value.path == root
    assert not (root / "memory").exists()
    assert root.read_text(encoding="utf-8") == "collision"


def test_initialization_rejects_junction_root(
    agent_home: Path,
    workspace: Path,
    tmp_path: Path,
) -> None:
    root = workspace / ".myclaw"
    target = tmp_path / "outside-junction"
    target.mkdir()
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(root), str(target)],
        capture_output=True,
        check=True,
        text=True,
    )

    with pytest.raises(WorkspaceStateError) as captured:
        state_for(workspace).initialize(agent_home_root=agent_home)

    assert captured.value.path == root
    assert root.is_junction()
    assert not (target / "memory").exists()
