"""Write File Core Catalog Tool."""

from __future__ import annotations

from typing import Annotated

from myclaw.agent.workspace import Workspace
from myclaw.tools.base import BaseTool, ToolError, ToolParam


class WriteFileTool(BaseTool):
    """Write exact UTF-8 bytes to any host-readable file path."""

    name = "write_file"
    description = "Write UTF-8 text to a file within the current Workspace."
    required = ("path", "content")

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    content: Annotated[str, ToolParam(description="Complete UTF-8 text content.")]

    def __init__(self, *, workspace: Workspace) -> None:
        self._workspace = workspace

    async def check_safety(  # type: ignore[override]
        self,
        *,
        path: str,
        content: str,
    ) -> str | None:
        del content
        return self.workspace_path_safety_reason(workspace=self._workspace, requested=path)

    async def execute(self, *, path: str, content: str) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
        except (OSError, UnicodeError) as error:
            raise ToolError(f"Write File failed: {error}") from error
        return "File written successfully."
