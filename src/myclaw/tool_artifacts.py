"""Atomic externalization for oversized Tool Results."""

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG
from typing import Final

from myclaw.atomic_files import FileIdentity, atomic_create_text_with_identity, file_identity
from myclaw.contracts import ArtifactReference, ToolExecutionContext, ToolResult
from myclaw.contracts.tools import encode_artifact_tool_call_id
from myclaw.workspace import Workspace

type ArtifactWriter = Callable[[Path, str], None]

_PREVIEW_CHARS: Final = 2000


class ArtifactWriteError(Exception):
    """Signal that an oversized result could not be externalized safely."""


class ArtifactDiscardError(Exception):
    """Signal that a tracked Tool Artifact could not be discarded safely."""


@dataclass(frozen=True, slots=True)
class _OwnedArtifact:
    result: ToolResult
    path: Path
    identity: FileIdentity


class ToolArtifactExternalizer:
    """Store oversized results beside their Conversation Session."""

    def __init__(
        self,
        *,
        context: ToolExecutionContext,
        max_tool_result_chars: int,
        write_text: ArtifactWriter | None = None,
    ) -> None:
        self._session_id = context.session_id
        self._max_tool_result_chars = max_tool_result_chars
        self._tracks_default_writes = write_text is None
        self._write_text = _atomic_create_artifact if write_text is None else write_text
        self._agent_home = context.agent_home
        self._owned_artifacts: dict[int, _OwnedArtifact] = {}
        workspace = Workspace.from_path(context.workspace)
        self._directory = (
            context.agent_home / "sessions" / workspace.slug / "artifacts" / context.session_id
        )

    def externalize(self, result: ToolResult) -> ToolResult:
        """Return an inline result or its persisted preview and reference."""
        raw_content = result.content
        if len(raw_content) <= self._max_tool_result_chars:
            return result

        created_path: Path | None = None
        created_identity: FileIdentity | None = None
        try:
            encoded_tool_call_id = encode_artifact_tool_call_id(result.tool_call_id)
            relative_path = f"artifacts/{self._session_id}/{encoded_tool_call_id}.txt"
            preview = raw_content[:_PREVIEW_CHARS]
            projected = ToolResult(
                tool_call_id=result.tool_call_id,
                name=result.name,
                status=result.status,
                content=(f"{preview}\n\n...[truncated; full result stored at {relative_path}]"),
                error=result.error,
                artifact=ArtifactReference(
                    path=relative_path,
                    total_chars=len(raw_content),
                    preview_chars=len(preview),
                ),
            )
            io_directory = _io_path(self._directory)
            io_agent_home = _io_path(self._agent_home)
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
            if self._tracks_default_writes:
                created_path = artifact_path
                created_identity = _atomic_create_artifact(artifact_path, raw_content)
                resolved_artifact = _resolved_for_comparison(artifact_path)
                if not resolved_artifact.is_relative_to(agent_home_root):
                    raise PermissionError("Tool Artifact file must remain inside Agent Home")
                owned_path = _io_path(resolved_artifact)
                status = owned_path.lstat()
                if (
                    not S_ISREG(status.st_mode)
                    or status.st_nlink != 1
                    or file_identity(status) != created_identity
                ):
                    raise PermissionError("Tool Artifact must be an unaliased regular file")
                self._owned_artifacts[id(projected)] = _OwnedArtifact(
                    result=projected,
                    path=owned_path,
                    identity=created_identity,
                )
            else:
                self._write_text(artifact_path, raw_content)
            return projected
        except Exception as error:
            if created_path is not None and created_identity is not None:
                try:
                    _discard_created_artifact(created_path, created_identity)
                except ArtifactDiscardError:
                    pass
            raise ArtifactWriteError from error

    def discard(self, result: ToolResult) -> bool:
        """Delete a still-owned artifact that has not been published to its Session."""
        owned = self._owned_artifacts.get(id(result))
        if owned is None or owned.result is not result:
            return False
        try:
            status = owned.path.lstat()
            agent_home_root = _resolved_for_comparison(_io_path(self._agent_home))
            resolved_artifact = _resolved_for_comparison(owned.path)
            if (
                not resolved_artifact.is_relative_to(agent_home_root)
                or not S_ISREG(status.st_mode)
                or status.st_nlink != 1
                or file_identity(status) != owned.identity
            ):
                raise PermissionError("Tool Artifact ownership could not be verified")
            owned.path.unlink()
        except FileNotFoundError:
            self._owned_artifacts.pop(id(result), None)
            return False
        except Exception as error:
            raise ArtifactDiscardError from error
        self._owned_artifacts.pop(id(result), None)
        return True

    def commit(self, result: ToolResult) -> bool:
        """Release rollback ownership after the artifact reference is persisted."""
        owned = self._owned_artifacts.get(id(result))
        if owned is None or owned.result is not result:
            return False
        self._owned_artifacts.pop(id(result), None)
        return True


def _io_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    native = str(path.absolute())
    if native.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{native.lstrip('\\')}")
    return Path(f"\\\\?\\{native}")


def _atomic_create_artifact(path: Path, content: str) -> FileIdentity:
    identity = atomic_create_text_with_identity(path, content)
    if identity is None:
        raise FileExistsError("Tool Artifact already exists")
    return identity


def _discard_created_artifact(path: Path, identity: FileIdentity) -> None:
    try:
        status = path.lstat()
        if not S_ISREG(status.st_mode) or status.st_nlink != 1 or file_identity(status) != identity:
            raise PermissionError("Tool Artifact ownership could not be verified")
        path.unlink()
    except Exception as error:
        raise ArtifactDiscardError from error


def _resolved_for_comparison(path: Path) -> Path:
    resolved = path.resolve(strict=True)
    if os.name != "nt":
        return resolved
    native = str(resolved)
    if native.startswith("\\\\?\\UNC\\"):
        return Path(f"\\\\{native.removeprefix('\\\\?\\UNC\\')}")
    return Path(native.removeprefix("\\\\?\\"))
