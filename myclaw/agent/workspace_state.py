"""Workspace-owned persistent state boundary and startup initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

from myclaw.agent.workspace import Workspace
from myclaw.errors import ErrorInfo
from myclaw.templates import load_template
from myclaw.utils.host_filesystem import HOST_FILESYSTEM

_GITIGNORE_CONTENT: Final = "*\n"
_LONG_TERM_MEMORY_TEMPLATE: Final = load_template("long-term-memory.md")


class WorkspaceStateError(Exception):
    """A safe startup failure tied to one Workspace State path."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.error = ErrorInfo(
            "persistence_error",
            "Workspace State could not be initialized at the reserved path.",
        )
        super().__init__(self.error.message)


class _UnsafeStatePath(PermissionError):
    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__("Workspace State path is unavailable or unsafe")


@dataclass(frozen=True, slots=True)
class WorkspaceState:
    """Canonical persistent-state paths owned by one Workspace."""

    workspace: Workspace

    @property
    def path(self) -> Path:
        return Path(self.workspace.path) / ".myclaw"

    @property
    def memory_directory(self) -> Path:
        return self.path / "memory"

    @property
    def sessions_directory(self) -> Path:
        return self.path / "sessions"

    @property
    def scheduled_work_path(self) -> Path:
        return self.path / "scheduled-work.json"

    @property
    def long_term_memory_path(self) -> Path:
        return self.memory_directory / "memory.md"

    def initialize(self, *, agent_home_root: Path) -> None:
        """Create required base state while rejecting redirected known paths."""
        try:
            normalized_agent_home = agent_home_root.resolve(strict=False)
            normalized_state_root = self.path.resolve(strict=False)
            if normalized_state_root.is_relative_to(
                normalized_agent_home
            ) or normalized_agent_home.is_relative_to(normalized_state_root):
                raise _UnsafeStatePath(self.path)

            workspace_root = Path(self.workspace.path).resolve(strict=True)
            if not workspace_root.is_dir():
                raise _UnsafeStatePath(Path(self.workspace.path))

            self.path.mkdir(exist_ok=True)
            state_root = HOST_FILESYSTEM.require_owned_directory(
                self.path, within=workspace_root
            )

            # Publication is create-only, so an existing policy is never read or repaired.
            HOST_FILESYSTEM.atomic_create_text(
                self.path / ".gitignore", _GITIGNORE_CONTENT
            )

            self.memory_directory.mkdir(exist_ok=True)
            HOST_FILESYSTEM.require_owned_directory(
                self.memory_directory, within=state_root
            )
            self.sessions_directory.mkdir(exist_ok=True)
            HOST_FILESYSTEM.require_owned_directory(
                self.sessions_directory, within=state_root
            )

            HOST_FILESYSTEM.atomic_create_text(
                self.long_term_memory_path, _LONG_TERM_MEMORY_TEMPLATE
            )
            HOST_FILESYSTEM.require_owned_regular_file(
                self.long_term_memory_path, within=state_root
            )
        except WorkspaceStateError:
            raise
        except _UnsafeStatePath as error:
            raise WorkspaceStateError(error.path) from error
        except OSError as error:
            affected = Path(error.filename) if isinstance(error.filename, str) else self.path
            raise WorkspaceStateError(affected) from error
        except RuntimeError as error:
            raise WorkspaceStateError(self.path) from error
