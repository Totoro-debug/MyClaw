"""Atomic externalization for oversized Tool Results."""

import os
from collections.abc import Callable
from pathlib import Path
from typing import Final

from myclaw.atomic_files import atomic_replace_text
from myclaw.contracts import ArtifactReference, ToolExecutionContext, ToolResult
from myclaw.contracts.tools import encode_artifact_tool_call_id
from myclaw.workspace import Workspace

type ArtifactWriter = Callable[[Path, str], None]

_PREVIEW_CHARS: Final = 2000


class ArtifactWriteError(Exception):
    """Signal that an oversized result could not be externalized safely."""


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
        self._write_text = atomic_replace_text if write_text is None else write_text
        workspace = Workspace.from_path(context.workspace)
        self._directory = (
            context.agent_home / "sessions" / workspace.slug / "artifacts" / context.session_id
        )

    def externalize(self, result: ToolResult) -> ToolResult:
        """Return an inline result or its persisted preview and reference."""
        raw_content = result.content
        if len(raw_content) <= self._max_tool_result_chars:
            return result

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
            io_directory.mkdir(parents=True, exist_ok=True)
            self._write_text(io_directory / f"{encoded_tool_call_id}.txt", raw_content)
            return projected
        except Exception as error:
            raise ArtifactWriteError from error


def _io_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    native = str(path.absolute())
    if native.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{native.lstrip('\\')}")
    return Path(f"\\\\?\\{native}")
