"""Workspace-scoped file mutation Tools."""

from typing import Annotated

from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.schema import ToolParam
from myclaw.tools.security import Security


class WriteFileTool(BaseTool):
    """Write exact UTF-8 text to one Workspace file."""

    name = "write_file"
    description = "Write UTF-8 text to a file within the current Workspace."
    required = ("path", "content")

    path: Annotated[str, ToolParam(description="Workspace file path.", min_length=1)]
    content: Annotated[str, ToolParam(description="Complete UTF-8 text content.")]

    def __init__(self, *, security: Security) -> None:
        self._security = security

    def refusal_reason(self, *, path: str, content: str) -> str:
        del path, content
        return "Writing Workspace files is unavailable because confirmation is not implemented."

    async def execute(self, *, path: str, content: str) -> str:
        target = self._security.resolve_write_path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8", newline="")
        except OSError as error:
            raise ToolError("The requested file could not be written.") from error
        return f"Wrote {target.name}."


class EditFileTool(BaseTool):
    """Replace exact UTF-8 text in one Workspace file."""

    name = "edit_file"
    description = "Replace exact UTF-8 text in a file within the current Workspace."
    required = ("path", "old_text", "new_text")

    path: Annotated[str, ToolParam(description="Existing Workspace file path.", min_length=1)]
    old_text: Annotated[str, ToolParam(description="Exact text to replace.", min_length=1)]
    new_text: Annotated[str, ToolParam(description="Replacement text.")]
    replace_all: Annotated[bool, ToolParam(description="Replace every exact match.")] = False

    def __init__(self, *, security: Security) -> None:
        self._security = security

    def refusal_reason(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str:
        del path, old_text, new_text, replace_all
        return "Editing Workspace files is unavailable because confirmation is not implemented."

    async def execute(
        self,
        *,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool,
    ) -> str:
        target = self._security.resolve_write_path(path)
        try:
            content = target.read_bytes().decode("utf-8")
        except OSError as error:
            raise ToolError("The requested file could not be read for editing.") from error
        except UnicodeError as error:
            raise ToolError("The requested file is not valid UTF-8 text.") from error
        match_count = content.count(old_text)
        if match_count == 0 or (not replace_all and match_count != 1):
            raise ToolError("old_text must match exactly once unless replace_all is true.")
        replaced = (
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1)
        )
        try:
            target.write_text(replaced, encoding="utf-8", newline="")
        except OSError as error:
            raise ToolError("The requested file could not be edited.") from error
        return f"Edited {target.name}."
