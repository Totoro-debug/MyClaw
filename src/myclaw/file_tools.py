"""Workspace-bounded built-in file tools."""

from pathlib import Path

from myclaw.contracts import JsonObject, ToolDefinition, ToolExecutionContext


class FileToolArgumentsError(ValueError):
    """Raised when a file tool receives arguments outside its public schema."""


class FileToolAccessDenied(PermissionError):
    """Raised when a requested path resolves outside the Workspace."""


class ReadFileTool:
    """Read a stable UTF-8 line window from a Workspace file."""

    _definition = ToolDefinition(
        name="read_file",
        description="Read UTF-8 text lines from a file within the current Workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 2000,
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        _reject_unknown(arguments, {"path", "offset", "limit"})
        path = _required_string_argument(arguments, "path")
        offset = _nonnegative_int_argument(arguments, "offset", default=0)
        limit = _bounded_int_argument(
            arguments,
            "limit",
            default=2000,
            minimum=1,
            maximum=10000,
        )
        target = _workspace_path(context.workspace.resolve(strict=True), path)
        if not target.is_file():
            raise FileToolArgumentsError("path must identify a regular file")
        raw_content = target.read_bytes()
        if b"\x00" in raw_content:
            raise UnicodeError("file contains binary NUL bytes")
        lines = raw_content.decode("utf-8").splitlines()
        return "\n".join(lines[offset : offset + limit])


class ListFilesTool:
    """Return a stable Workspace-relative directory listing."""

    _definition = ToolDefinition(
        name="list_files",
        description="List files and directories within the current Workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "default": "."},
                "recursive": {"type": "boolean", "default": False},
                "max_entries": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10000,
                    "default": 1000,
                },
            },
            "additionalProperties": False,
        },
    )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        _reject_unknown(arguments, {"path", "recursive", "max_entries"})
        path = _string_argument(arguments, "path", default=".")
        recursive = _bool_argument(arguments, "recursive", default=False)
        max_entries = _bounded_int_argument(
            arguments,
            "max_entries",
            default=1000,
            minimum=1,
            maximum=10000,
        )
        workspace = context.workspace.resolve(strict=True)
        target = _workspace_path(workspace, path)
        if not target.is_dir():
            raise FileToolArgumentsError("path must identify a directory")

        candidates = target.rglob("*") if recursive else target.iterdir()
        entries: list[str] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_relative_to(workspace):
                continue
            if not (resolved.is_file() or resolved.is_dir()):
                continue
            relative = candidate.relative_to(workspace).as_posix()
            if resolved.is_dir():
                relative += "/"
            entries.append(relative)
        return "\n".join(sorted(entries)[:max_entries])


class SearchFilesTool:
    """Search UTF-8 Workspace text in stable path and line order."""

    _definition = ToolDefinition(
        name="search_files",
        description="Search UTF-8 text files within the current Workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "path": {"type": "string", "default": "."},
                "glob": {"type": ["string", "null"], "default": None},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "default": 200,
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        _reject_unknown(arguments, {"query", "path", "glob", "max_results"})
        query = _required_string_argument(arguments, "query")
        path = _string_argument(arguments, "path", default=".")
        glob = _optional_string_argument(arguments, "glob")
        max_results = _bounded_int_argument(
            arguments,
            "max_results",
            default=200,
            minimum=1,
            maximum=1000,
        )
        workspace = context.workspace.resolve(strict=True)
        target = _workspace_path(workspace, path)
        if target.is_file():
            candidates = [target]
        elif target.is_dir():
            candidates = list(target.rglob("*"))
        else:
            raise FileToolArgumentsError("path must identify a file or directory")

        files: list[tuple[str, Path]] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            if not resolved.is_relative_to(workspace) or not resolved.is_file():
                continue
            relative = candidate.relative_to(workspace).as_posix()
            if glob is not None:
                try:
                    if not Path(relative).match(glob):
                        continue
                except ValueError as exc:
                    raise FileToolArgumentsError("glob is invalid") from exc
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


def _workspace_path(workspace: Path, requested: str) -> Path:
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = workspace / candidate
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileToolArgumentsError("path does not exist") from exc
    if not resolved.is_relative_to(workspace):
        raise FileToolAccessDenied("path resolves outside the Workspace")
    return resolved


def _reject_unknown(arguments: JsonObject, allowed: set[str]) -> None:
    if any(name not in allowed for name in arguments):
        raise FileToolArgumentsError("arguments contain an unknown field")


def _string_argument(arguments: JsonObject, name: str, *, default: str) -> str:
    value = arguments.get(name, default)
    if not isinstance(value, str):
        raise FileToolArgumentsError(f"{name} must be a string")
    return value


def _required_string_argument(arguments: JsonObject, name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise FileToolArgumentsError(f"{name} must be a non-empty string")
    return value


def _optional_string_argument(arguments: JsonObject, name: str) -> str | None:
    value = arguments.get(name)
    if value is not None and not isinstance(value, str):
        raise FileToolArgumentsError(f"{name} must be a string or null")
    return value


def _bool_argument(arguments: JsonObject, name: str, *, default: bool) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise FileToolArgumentsError(f"{name} must be a boolean")
    return value


def _bounded_int_argument(
    arguments: JsonObject,
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise FileToolArgumentsError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise FileToolArgumentsError(f"{name} is outside the allowed range")
    return value


def _nonnegative_int_argument(arguments: JsonObject, name: str, *, default: int) -> int:
    value = arguments.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FileToolArgumentsError(f"{name} must be a nonnegative integer")
    return value
