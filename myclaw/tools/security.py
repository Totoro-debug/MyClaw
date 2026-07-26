"""Shared filesystem security rules for concrete Tools."""

import os
from pathlib import Path

from myclaw.agent.workspace import Workspace
from myclaw.tools.errors import ToolError

_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


class Security:
    """Resolve model-provided paths within configured readable roots."""

    def __init__(
        self,
        *,
        workspace: Workspace,
        agent_home: Path,
        artifact_directory: Path,
    ) -> None:
        self._workspace_identity = workspace
        self._workspace = _io_path(Path(workspace.path)).resolve(strict=False)
        self._agent_home = _io_path(agent_home).resolve(strict=False)
        self._long_term_memory = self._agent_home / "memory" / "memory.md"
        # Preserve the configured directory's lexical identity. Resolving it here
        # would make a pre-existing symlink or junction target part of the trusted
        # read scope.
        self._artifact_directory = _io_path(artifact_directory).absolute()
        self._session_id = artifact_directory.name

    def resolve_read_path(self, requested: str) -> Path:
        """Resolve one existing readable path or raise a public-safe Tool error."""
        candidate = Path(requested)
        if os.name == "nt" and any(_is_windows_reserved(part) for part in candidate.parts):
            raise ToolError("The requested path identifies a Windows device.")
        if not candidate.is_absolute():
            if candidate.parts and candidate.parts[0] == "artifacts":
                candidate = (
                    self._agent_home
                    / "sessions"
                    / self._workspace_identity.slug
                    / candidate
                )
            else:
                candidate = self._workspace / candidate
        else:
            candidate = _io_path(candidate)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ToolError("The requested path does not exist.") from error

        scope = self._read_scope(resolved)
        if scope is None and (
            resolved == self._agent_home or resolved.is_relative_to(self._agent_home)
        ):
            raise ToolError("Agent Home internal state is not readable by file Tools.")
        if scope is None:
            raise ToolError("The requested path resolves outside the Workspace.")
        relative_parts = (
            resolved.relative_to(self._agent_home).parts
            if scope in {"memory", "artifact"}
            else resolved.relative_to(self._workspace).parts
        )
        if os.name == "nt" and any(":" in part for part in relative_parts):
            raise ToolError("The requested path identifies a Windows alternate data stream.")
        return resolved

    def reported_read_path(self, target: Path) -> str:
        """Return one stable model-visible path for a previously resolved target."""
        scope = self._read_scope(target)
        if scope == "memory":
            return "memory/memory.md"
        if scope == "artifact":
            suffix = target.relative_to(self._artifact_directory)
            return (Path("artifacts") / self._session_id / suffix).as_posix()
        if scope == "workspace":
            return target.relative_to(self._workspace).as_posix()
        raise ToolError("The requested path is outside the readable scope.")

    def _read_scope(self, target: Path) -> str | None:
        if target == self._agent_home or target.is_relative_to(self._agent_home):
            if target == self._long_term_memory:
                return "memory"
            if target == self._artifact_directory or target.is_relative_to(
                self._artifact_directory
            ):
                return "artifact"
            return None
        if target.is_relative_to(self._workspace):
            return "workspace"
        return None


def _is_windows_reserved(component: str) -> bool:
    normalized = component.rstrip(" .")
    basename = normalized.split(".", maxsplit=1)[0].upper()
    return basename in _WINDOWS_RESERVED_BASENAMES


def _io_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    native = str(path.absolute())
    if native.startswith("\\\\?\\"):
        return path
    if native.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{native.lstrip('\\')}")
    return Path(f"\\\\?\\{native}")
