"""Read File Core Catalog Tool."""

from __future__ import annotations

from typing import Annotated

from myclaw.agent.workspace import Workspace
from myclaw.tools.base import BaseTool, ToolError, ToolParam


class ReadFileTool(BaseTool):
    """Read a strict UTF-8 line window from any host-readable file."""

    name = "read_file"
    description = "Read UTF-8 text lines from a file within the current Workspace."
    required = ("path",)

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    offset: Annotated[int, ToolParam(description="One-based first line.", minimum=1)] = 1
    limit: Annotated[
        int,
        ToolParam(description="Maximum lines to return.", minimum=1, maximum=10000),
    ] = 2000

    def __init__(self, *, workspace: Workspace) -> None:
        self._workspace = workspace

    async def check_safety(  # type: ignore[override]
        self,
        *,
        path: str,
        offset: int,
        limit: int,
    ) -> str | None:
        del offset, limit
        return self.workspace_path_safety_reason(workspace=self._workspace, requested=path)

    async def execute(self, *, path: str, offset: int, limit: int) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        try:
            raw_content = target.read_bytes()
        except OSError as error:
            raise ToolError(f"Read File failed: {error}") from error
        try:
            content = raw_content.decode("utf-8")
        except UnicodeError as error:
            raise ToolError("Read File failed: the target is not valid UTF-8 text.") from error
        lines = content.splitlines(keepends=True)
        return "".join(lines[offset - 1 : offset - 1 + limit])
