"""Workspace-bounded built-in file tools."""

from pathlib import Path
from typing import Annotated

from myclaw.tools.base import BaseTool
from myclaw.tools.errors import ToolError
from myclaw.tools.schema import ToolParam
from myclaw.tools.security import Security
from myclaw.utils.windows_filesystem import is_windows_regular_file


class ReadFileTool(BaseTool):
    """Read a stable UTF-8 line window from a Workspace file."""

    name = "read_file"
    description = "Read UTF-8 text lines from a file within the current Workspace."
    required = ("path",)

    path: Annotated[str, ToolParam(description="Workspace-relative or allowed readable path.")]
    offset: Annotated[int, ToolParam(description="Zero-based first line.", minimum=0)] = 0
    limit: Annotated[
        int,
        ToolParam(description="Maximum lines to return.", minimum=1, maximum=10000),
    ] = 2000

    def __init__(self, *, security: Security) -> None:
        self._security = security

    async def execute(self, *, path: str, offset: int, limit: int) -> str:
        target = self._security.resolve_read_path(path)
        try:
            status = target.lstat()
        except OSError as error:
            raise ToolError("The requested file could not be inspected.") from error
        if not is_windows_regular_file(status):
            raise ToolError("The requested path must identify a regular file.")
        if status.st_nlink != 1:
            raise ToolError("The requested path must identify an unaliased regular file.")
        try:
            raw_content = target.read_bytes()
        except OSError as error:
            raise ToolError("The requested file could not be read.") from error
        if b"\x00" in raw_content:
            raise ToolError("The requested file contains binary NUL bytes.")
        try:
            lines = raw_content.decode("utf-8").splitlines()
        except UnicodeError as error:
            raise ToolError("The requested file is not valid UTF-8 text.") from error
        return "\n".join(lines[offset : offset + limit])


class ListFilesTool(BaseTool):
    """Return a stable Workspace-relative directory listing."""

    name = "list_files"
    description = "List files and directories within the current Workspace."

    path: Annotated[str, ToolParam(description="Directory to list.")] = "."
    recursive: Annotated[bool, ToolParam(description="Include nested entries.")] = False
    max_entries: Annotated[
        int,
        ToolParam(description="Maximum entries to return.", minimum=1, maximum=10000),
    ] = 1000

    def __init__(self, *, security: Security) -> None:
        self._security = security

    async def execute(self, *, path: str, recursive: bool, max_entries: int) -> str:
        target = self._security.resolve_read_path(path)
        if not target.is_dir():
            raise ToolError("The requested path must identify a directory.")

        candidates = target.rglob("*") if recursive else target.iterdir()
        entries: list[str] = []
        try:
            for candidate in candidates:
                try:
                    resolved = self._security.resolve_read_path(str(candidate))
                    status = resolved.lstat()
                except (OSError, ToolError):
                    continue
                is_file = is_windows_regular_file(status)
                is_directory = resolved.is_dir()
                if not (is_file or is_directory):
                    continue
                if is_file and status.st_nlink != 1:
                    continue
                relative = self._security.reported_read_path(resolved)
                entries.append(f"{relative}/" if is_directory else relative)
        except OSError as error:
            raise ToolError("The requested directory could not be listed.") from error
        return "\n".join(sorted(entries)[:max_entries])


class SearchFilesTool(BaseTool):
    """Search UTF-8 Workspace text in stable path and line order."""

    name = "search_files"
    description = "Search UTF-8 text files within the current Workspace."
    required = ("query",)

    query: Annotated[str, ToolParam(description="Literal text to find.", min_length=1)]
    path: Annotated[str, ToolParam(description="File or directory to search.")] = "."
    glob: Annotated[
        str | None,
        ToolParam(description="Optional path glob used to filter searched files."),
    ] = None
    max_results: Annotated[
        int,
        ToolParam(description="Maximum matches to return.", minimum=1, maximum=1000),
    ] = 200

    def __init__(self, *, security: Security) -> None:
        self._security = security

    async def execute(
        self,
        *,
        query: str,
        path: str,
        glob: str | None,
        max_results: int,
    ) -> str:
        target = self._security.resolve_read_path(path)
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = list(target.rglob("*"))
        else:
            raise ToolError("The requested path must identify a file or directory.")

        files: list[tuple[str, Path]] = []
        for candidate in candidates:
            try:
                resolved = self._security.resolve_read_path(str(candidate))
                status = resolved.lstat()
            except (OSError, ToolError):
                continue
            if not is_windows_regular_file(status) or status.st_nlink != 1:
                continue
            relative = self._security.reported_read_path(resolved)
            if glob is not None:
                try:
                    if not Path(relative).match(glob):
                        continue
                except ValueError as error:
                    raise ToolError("The requested glob pattern is invalid.") from error
            files.append((relative, resolved))

        matches: list[str] = []
        for relative, candidate in sorted(files):
            try:
                content = candidate.read_bytes()
                if b"\x00" in content:
                    continue
                lines = content.decode("utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if query not in line:
                    continue
                matches.append(f"{relative}:{line_number}:{line}")
                if len(matches) == max_results:
                    return "\n".join(matches)
        return "\n".join(matches)


class WriteFileTool(BaseTool):
    """Declare unavailable Workspace file creation."""

    name = "write_file"
    description = "Write UTF-8 text to a file within the current Workspace."
    required = ("path", "content")

    path: Annotated[str, ToolParam(description="Workspace file path.", min_length=1)]
    content: Annotated[str, ToolParam(description="Complete UTF-8 text content.")]

    def refusal_reason(self, *, path: str, content: str) -> str:
        del path, content
        return "Writing Workspace files is unavailable because confirmation is not implemented."

    async def execute(self, *, path: str, content: str) -> str:
        raise AssertionError("Refusal-only Tool reached execution")


class EditFileTool(BaseTool):
    """Declare unavailable Workspace file editing."""

    name = "edit_file"
    description = "Replace exact UTF-8 text in a file within the current Workspace."
    required = ("path", "old_text", "new_text")

    path: Annotated[str, ToolParam(description="Existing Workspace file path.", min_length=1)]
    old_text: Annotated[str, ToolParam(description="Exact text to replace.", min_length=1)]
    new_text: Annotated[str, ToolParam(description="Replacement text.")]
    replace_all: Annotated[bool, ToolParam(description="Replace every exact match.")] = False

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
        raise AssertionError("Refusal-only Tool reached execution")
