"""Atomic externalization for oversized Tool Results."""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Final

from myclaw.agent.workspace_state import WorkspaceState
from myclaw.tools.artifacts import ArtifactReference, encode_artifact_tool_call_id
from myclaw.tools.models import ToolResult
from myclaw.utils.host_filesystem import (
    HOST_FILESYSTEM,
    FileIdentity,
)

type ArtifactWriter = Callable[[Path, str], None]

_PREVIEW_CHARS: Final = 2000


class ArtifactWriteError(Exception):
    """Signal that an oversized result could not be externalized safely."""


def externalize_tool_result(
    result: ToolResult,
    *,
    workspace_state: WorkspaceState,
    session_id: str,
    max_tool_result_chars: int,
    write_text: ArtifactWriter | None = None,
) -> ToolResult:
    """Externalize one oversized successful result without lifecycle ownership."""
    if result.status != "success" or len(result.content) <= max_tool_result_chars:
        return result

    try:
        raw_content = result.content
        encoded_tool_call_id = encode_artifact_tool_call_id(result.tool_call_id)
        relative_path = f"artifacts/{session_id}/{encoded_tool_call_id}.txt"
        preview = raw_content[:_PREVIEW_CHARS]
        projected = replace(
            result,
            content=f"{preview}\n\n...[truncated; full result stored at {relative_path}]",
            artifact=ArtifactReference(
                path=relative_path,
                total_chars=len(raw_content),
                preview_chars=len(preview),
            ),
        )
        io_workspace = HOST_FILESYSTEM.path_for_io(Path(workspace_state.workspace.path))
        workspace_root = io_workspace.resolve(strict=True)
        io_state = HOST_FILESYSTEM.path_for_io(workspace_state.path)
        state_root = HOST_FILESYSTEM.require_owned_directory(io_state, within=workspace_root)
        io_sessions = HOST_FILESYSTEM.path_for_io(workspace_state.sessions_directory)
        sessions_root = HOST_FILESYSTEM.require_owned_directory(io_sessions, within=state_root)
        artifacts_directory = io_sessions / "artifacts"
        artifacts_directory.mkdir(exist_ok=True)
        artifacts_root = HOST_FILESYSTEM.require_owned_directory(
            artifacts_directory, within=sessions_root
        )
        io_directory = artifacts_directory / session_id
        io_directory.mkdir(exist_ok=True)
        HOST_FILESYSTEM.require_owned_directory(io_directory, within=artifacts_root)
        artifact_path = io_directory / f"{encoded_tool_call_id}.txt"
        if write_text is None:
            identity = _atomic_create_artifact(artifact_path, raw_content)
            resolved_artifact = HOST_FILESYSTEM.require_owned_regular_file(
                artifact_path, within=sessions_root
            )
            owned_path = HOST_FILESYSTEM.path_for_io(resolved_artifact)
            if HOST_FILESYSTEM.file_identity(owned_path.stat(follow_symlinks=False)) != identity:
                raise PermissionError("Tool Artifact must be an unaliased regular file")
        else:
            write_text(artifact_path, raw_content)
    except Exception as error:
        raise ArtifactWriteError from error
    return projected


def _atomic_create_artifact(path: Path, content: str) -> FileIdentity:
    identity = HOST_FILESYSTEM.atomic_create_text_with_identity(path, content)
    if identity is None:
        raise FileExistsError("Tool Artifact already exists")
    return identity
