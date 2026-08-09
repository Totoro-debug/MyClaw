"""Shared filesystem security rules for concrete Tools."""

from pathlib import Path

from myclaw.agent.workspace import Workspace
from myclaw.tools.base import ToolError
from myclaw.utils.host_filesystem import HOST_FILESYSTEM


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
        self._workspace = HOST_FILESYSTEM.path_for_io(Path(workspace.path)).resolve(strict=False)
        self._agent_home = HOST_FILESYSTEM.path_for_io(agent_home).resolve(strict=False)
        self._workspace_state = self._workspace / ".myclaw"
        self._long_term_memory = self._workspace_state / "memory" / "memory.md"
        # Preserve the configured directory's lexical identity. Resolving it here
        # would make a pre-existing symlink or junction target part of the trusted
        # read scope.
        self._artifact_directory = HOST_FILESYSTEM.path_for_io(artifact_directory).absolute()
        self._artifact_root = self._artifact_directory.parent
        self._session_id = artifact_directory.name

    def resolve_read_path(self, requested: str) -> Path:
        """Resolve one existing readable path or raise a public-safe Tool error."""
        candidate = Path(requested)
        artifact_alias = False
        memory_alias = False
        if any(HOST_FILESYSTEM.is_reserved_component(part) for part in candidate.parts):
            raise ToolError("The requested path identifies a Windows device.")
        if not candidate.is_absolute():
            if candidate.parts == ("memory", "memory.md"):
                memory_alias = True
                candidate = self._long_term_memory
            elif candidate.parts and candidate.parts[0] == "artifacts":
                artifact_alias = True
                candidate = self._artifact_root.joinpath(*candidate.parts[1:])
            elif candidate.parts[:3] == (".myclaw", "artifacts", self._session_id):
                artifact_alias = True
                candidate = self._workspace / candidate
            else:
                candidate = self._workspace / candidate
        else:
            candidate = HOST_FILESYSTEM.path_for_io(candidate)
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ToolError("The requested path does not exist.") from error

        scope = self._read_scope(resolved)
        if scope == "artifact" and not artifact_alias:
            raise ToolError("Workspace State internal files are not readable by file Tools.")
        if scope == "memory" and not memory_alias:
            raise ToolError("Workspace State internal files are not readable by file Tools.")
        if scope is None and (
            resolved == self._agent_home or resolved.is_relative_to(self._agent_home)
        ):
            raise ToolError("Agent Home internal state is not readable by file Tools.")
        if scope is None and (
            resolved == self._workspace_state or resolved.is_relative_to(self._workspace_state)
        ):
            raise ToolError("Workspace State internal files are not readable by file Tools.")
        if scope is None:
            raise ToolError("The requested path resolves outside the Workspace.")
        if scope == "memory":
            relative_parts = resolved.relative_to(self._long_term_memory.parent).parts
        elif scope == "artifact":
            relative_parts = resolved.relative_to(self._artifact_directory).parts
        else:
            relative_parts = resolved.relative_to(self._workspace).parts
        if any(HOST_FILESYSTEM.has_alternate_data_stream(part) for part in relative_parts):
            raise ToolError("The requested path identifies a Windows alternate data stream.")
        return resolved

    def reported_read_path(self, target: Path) -> str:
        """Return one stable model-visible path for a previously resolved target."""
        scope = self._read_scope(target)
        if scope == "memory":
            return "memory/memory.md"
        if scope == "artifact":
            suffix = target.relative_to(self._artifact_directory)
            return _slash_reference(Path("artifacts") / self._session_id / suffix)
        if scope == "workspace":
            return _slash_reference(target.relative_to(self._workspace))
        raise ToolError("The requested path is outside the readable scope.")

    def _read_scope(self, target: Path) -> str | None:
        if target == self._long_term_memory:
            return "memory"
        if target == self._artifact_directory or target.is_relative_to(self._artifact_directory):
            return "artifact"
        if target == self._agent_home or target.is_relative_to(self._agent_home):
            return None
        if target == self._workspace_state or target.is_relative_to(self._workspace_state):
            return None
        if target.is_relative_to(self._workspace):
            return "workspace"
        return None


def _slash_reference(path: Path) -> str:
    return "/".join(path.parts)
