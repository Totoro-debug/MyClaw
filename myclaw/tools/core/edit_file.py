"""Edit File Core Catalog Tool."""

from __future__ import annotations

from typing import Annotated

from myclaw.agent.workspace import Workspace
from myclaw.tools.base import BaseTool, ToolError, ToolParam


class EditFileTool(BaseTool):
    """Replace exact strict UTF-8 text in any host-readable file path."""

    name = "edit_file"
    description = "Replace exact UTF-8 text in a file within the current Workspace."
    required = ("path", "old_text", "new_text")

    path: Annotated[
        str,
        ToolParam(description="Workspace-relative or absolute file path.", min_length=1),
    ]
    old_text: Annotated[str, ToolParam(description="Exact text to replace.", min_length=1)]
    new_text: Annotated[str, ToolParam(description="Replacement text.")]
    replace_all: Annotated[bool, ToolParam(description="Replace every exact match.")] = False

    def __init__(self, *, workspace: Workspace) -> None:
        self._workspace = workspace

    async def check_safety(  # type: ignore[override]
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str | None:
        del old_text, new_text, replace_all
        return self.workspace_path_safety_reason(workspace=self._workspace, requested=path)

    async def execute(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str:
        target = self.resolve_path_argument(workspace=self._workspace, requested=path)
        try:
            raw_content = target.read_bytes()
        except OSError as error:
            raise ToolError(f"Edit File read failed: {error}") from error
        try:
            content = raw_content.decode("utf-8")
        except UnicodeError as error:
            raise ToolError("Edit File failed: the target is not valid UTF-8 text.") from error

        match_count = content.count(old_text)
        if match_count == 0:
            raise ToolError("Edit File found zero matches for the requested text.")
        if not replace_all and match_count != 1:
            raise ToolError(
                "Edit File found ambiguous text; use replace_all to replace every match."
            )

        replacement = (
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1)
        )
        try:
            target.write_bytes(replacement.encode("utf-8"))
        except (OSError, UnicodeError) as error:
            raise ToolError(f"Edit File write failed: {error}") from error
        return "File edited successfully."
