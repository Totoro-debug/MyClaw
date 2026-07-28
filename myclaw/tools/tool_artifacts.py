"""Atomic externalization for oversized Tool Results."""

import os
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from stat import S_ISREG
from typing import Final

from myclaw.agent.workspace import Workspace
from myclaw.tools.artifacts import ArtifactReference, encode_artifact_tool_call_id
from myclaw.tools.models import ToolResult
from myclaw.utils.atomic_files import (
    FileIdentity,
    atomic_create_text_with_identity,
    file_identity,
    path_for_io,
)

type ArtifactWriter = Callable[[Path, str], None]

_PREVIEW_CHARS: Final = 2000


class ArtifactWriteError(Exception):
    """Signal that an oversized result could not be externalized safely."""


def externalize_tool_result(
    result: ToolResult,
    *,
    agent_home: Path,
    workspace: Workspace,
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
        io_agent_home = path_for_io(agent_home)
        io_directory = io_agent_home / "sessions" / workspace.slug / "artifacts" / session_id
        io_agent_home.mkdir(parents=True, exist_ok=True)
        agent_home_root = _resolved_for_comparison(io_agent_home)
        existing_parent = io_directory
        while not existing_parent.exists():
            parent = existing_parent.parent
            if parent == existing_parent:
                raise PermissionError("Tool Artifact path has no existing parent")
            existing_parent = parent
        resolved_parent = _resolved_for_comparison(existing_parent)
        if not resolved_parent.is_relative_to(agent_home_root) or not existing_parent.is_dir():
            raise PermissionError("Tool Artifact directory must remain inside Agent Home")
        io_directory.mkdir(parents=True, exist_ok=True)
        resolved_directory = _resolved_for_comparison(io_directory)
        if not resolved_directory.is_relative_to(agent_home_root) or not io_directory.is_dir():
            raise PermissionError("Tool Artifact directory must remain inside Agent Home")
        artifact_path = io_directory / f"{encoded_tool_call_id}.txt"
        if write_text is None:
            identity = _atomic_create_artifact(artifact_path, raw_content)
            resolved_artifact = _resolved_for_comparison(artifact_path)
            if not resolved_artifact.is_relative_to(agent_home_root):
                raise PermissionError("Tool Artifact file must remain inside Agent Home")
            owned_path = path_for_io(resolved_artifact)
            status = owned_path.lstat()
            if (
                not S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or file_identity(status) != identity
            ):
                raise PermissionError("Tool Artifact must be an unaliased regular file")
        else:
            write_text(artifact_path, raw_content)
    except Exception as error:
        raise ArtifactWriteError from error
    return projected


def _atomic_create_artifact(path: Path, content: str) -> FileIdentity:
    identity = atomic_create_text_with_identity(path, content)
    if identity is None:
        raise FileExistsError("Tool Artifact already exists")
    return identity


def _resolved_for_comparison(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if os.name != "nt":
        return resolved
    native = str(resolved)
    if native.startswith("\\\\?\\UNC\\"):
        return Path(f"\\\\{native.removeprefix('\\\\?\\UNC\\')}")
    return Path(native.removeprefix("\\\\?\\"))
