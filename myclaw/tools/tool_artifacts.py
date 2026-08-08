"""Persist and reference oversized Tool Results."""

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Final
from urllib.parse import quote, unquote

from myclaw.session.session import Session
from myclaw.tools.tool_gateway import ToolResult
from myclaw.utils.host_filesystem import (
    HOST_FILESYSTEM,
    FileIdentity,
)
from myclaw.utils.validation import require_nonnegative_int

type ArtifactWriter = Callable[[Path, str], None]

_PREVIEW_CHARS: Final = 2000
_WINDOWS_RESERVED_BASENAMES: Final = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def encode_artifact_tool_call_id(tool_call_id: str) -> str:
    """Return the canonical Windows filename component for a Tool call ID."""
    basename = tool_call_id.split(".", maxsplit=1)[0].upper()
    if basename in _WINDOWS_RESERVED_BASENAMES:
        return "".join(f"%{byte:02X}" for byte in tool_call_id.encode("utf-8"))
    return quote(tool_call_id, safe="-_.", encoding="utf-8", errors="strict")


@dataclass(frozen=True, slots=True)
class ArtifactReference:
    """A persisted relative reference to an externalized Tool result."""

    path: str
    total_chars: int
    preview_chars: int

    def __post_init__(self) -> None:
        parts = self.path.split("/")
        if len(parts) != 3 or parts[0] != "artifacts":
            raise ValueError("path must match the persisted artifact path contract")
        Session._require_id(parts[1])
        filename = parts[2]
        if not filename.endswith(".txt"):
            raise ValueError("artifact filename must end with .txt")
        encoded_tool_call_id = filename.removesuffix(".txt")
        if not encoded_tool_call_id:
            raise ValueError("artifact filename requires a percent-encoded tool call ID")
        try:
            tool_call_id = unquote(encoded_tool_call_id, encoding="utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError("artifact filename must use valid UTF-8 percent-encoding") from error
        if encode_artifact_tool_call_id(tool_call_id) != encoded_tool_call_id:
            raise ValueError("artifact filename must use canonical UTF-8 percent-encoding")
        require_nonnegative_int(self.total_chars, field="total_chars")
        require_nonnegative_int(self.preview_chars, field="preview_chars")
        if self.preview_chars > self.total_chars:
            raise ValueError("preview_chars must not exceed total_chars")

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "total_chars": self.total_chars,
            "preview_chars": self.preview_chars,
        }


class ArtifactWriteError(Exception):
    """Signal that an oversized result could not be externalized safely."""


def externalize_tool_result(
    result: ToolResult,
    *,
    session: Session,
    max_tool_result_chars: int,
    write_text: ArtifactWriter | None = None,
) -> ToolResult:
    """Externalize one oversized successful result without lifecycle ownership."""
    resolved_workspace_state = session.workspace_state
    resolved_session_id = session.session_id
    if result.status != "success" or len(result.content) <= max_tool_result_chars:
        return result

    try:
        raw_content = result.content
        encoded_tool_call_id = encode_artifact_tool_call_id(result.tool_call_id)
        relative_path = f"artifacts/{resolved_session_id}/{encoded_tool_call_id}.txt"
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
        io_workspace = HOST_FILESYSTEM.path_for_io(Path(resolved_workspace_state.workspace.path))
        workspace_root = io_workspace.resolve(strict=True)
        io_state = HOST_FILESYSTEM.path_for_io(resolved_workspace_state.path)
        state_root = HOST_FILESYSTEM.require_owned_directory(io_state, within=workspace_root)
        io_sessions = HOST_FILESYSTEM.path_for_io(session.storage_directory)
        io_sessions.mkdir(exist_ok=True)
        sessions_root = HOST_FILESYSTEM.require_owned_directory(io_sessions, within=state_root)
        artifacts_directory = HOST_FILESYSTEM.path_for_io(session.artifact_directory.parent)
        artifacts_directory.mkdir(exist_ok=True)
        artifacts_root = HOST_FILESYSTEM.require_owned_directory(
            artifacts_directory, within=sessions_root
        )
        io_directory = HOST_FILESYSTEM.path_for_io(session.artifact_directory)
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
