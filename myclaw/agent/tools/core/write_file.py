"""Write File Core Catalog Tool."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from myclaw.agent.tools.base import BaseTool, ToolError, ToolParam
from myclaw.agent.tools.permission import FileAccess


class WriteFileTool(BaseTool):
    """Write exact UTF-8 bytes to any host-readable file path."""

    name = "write_file"
    description = "Write UTF-8 text to a file. Paths outside the Workspace require confirmation."
    required = ("path", "content")

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    content: Annotated[str, ToolParam(description="Complete UTF-8 text content.")]

    def __init__(self, *, workspace: Path) -> None:
        self._workspace = workspace

    def build_file_accesses(self, prepared_arguments: dict[str, object]) -> tuple[FileAccess, ...]:
        return (
            self.canonical_file_access(
                workspace=self._workspace,
                base=self._workspace,
                requested=str(prepared_arguments["path"]),
                role="write",
            ),
        )

    async def execute(self, *, path: str, content: str) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
        except (OSError, UnicodeError) as error:
            raise ToolError(f"Write File failed: {error}") from error
        return "File written successfully."
