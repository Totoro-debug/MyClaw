"""Workspace-bounded built-in file tools."""

import os
from dataclasses import dataclass
from pathlib import Path
from stat import S_ISREG

from myclaw.agent.workspace import Workspace
from myclaw.tools.models import ToolDefinition, ToolExecutionContext
from myclaw.utils.json_types import JsonObject

_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


@dataclass(frozen=True, slots=True)
class _ReadRoots:
    workspace: Path
    agent_home: Path
    long_term_memory: Path
    artifact_directory: Path
    session_id: str

    def scope(self, target: Path) -> str | None:
        if target == self.agent_home or target.is_relative_to(self.agent_home):
            if target == self.long_term_memory:
                return "memory"
            if target == self.artifact_directory or target.is_relative_to(self.artifact_directory):
                return "artifact"
            return None
        if target.is_relative_to(self.workspace):
            return "workspace"
        return None

    def reported_path(self, target: Path, scope: str) -> str:
        if scope == "memory":
            return "memory/memory.md"
        if scope == "artifact":
            suffix = target.relative_to(self.artifact_directory)
            return (Path("artifacts") / self.session_id / suffix).as_posix()
        return target.relative_to(self.workspace).as_posix()


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
        target = _workspace_path(context, path)
        status = target.lstat()
        if not S_ISREG(status.st_mode):
            raise FileToolArgumentsError("path must identify a regular file")
        if status.st_nlink != 1:
            raise FileToolAccessDenied("path must identify an unaliased regular file")
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
        roots = _read_roots(context)
        target = _workspace_path(context, path, roots=roots)
        if not target.is_dir():
            raise FileToolArgumentsError("path must identify a directory")

        candidates = target.rglob("*") if recursive else target.iterdir()
        entries: list[str] = []
        for candidate in candidates:
            try:
                resolved = candidate.resolve(strict=True)
            except OSError:
                continue
            scope = roots.scope(resolved)
            if scope is None:
                continue
            is_file = resolved.is_file()
            is_directory = resolved.is_dir()
            if not (is_file or is_directory):
                continue
            if is_file:
                try:
                    if resolved.lstat().st_nlink != 1:
                        continue
                except OSError:
                    continue
            relative = roots.reported_path(resolved, scope)
            if is_directory:
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
        roots = _read_roots(context)
        target = _workspace_path(context, path, roots=roots)
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
            scope = roots.scope(resolved)
            if scope is None or not resolved.is_file():
                continue
            try:
                if resolved.lstat().st_nlink != 1:
                    continue
            except OSError:
                continue
            relative = roots.reported_path(resolved, scope)
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


def _workspace_path(
    context: ToolExecutionContext,
    requested: str,
    *,
    roots: _ReadRoots | None = None,
) -> Path:
    roots = _read_roots(context) if roots is None else roots
    candidate = Path(requested)
    if os.name == "nt" and any(_is_windows_reserved(part) for part in candidate.parts):
        raise FileToolAccessDenied("path identifies a Windows device")
    if not candidate.is_absolute():
        if candidate.parts and candidate.parts[0] == "artifacts":
            workspace_slug = Workspace.from_path(context.workspace).slug
            candidate = roots.agent_home / "sessions" / workspace_slug / candidate
        else:
            candidate = roots.workspace / candidate
    else:
        candidate = _io_path(candidate)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileToolArgumentsError("path does not exist") from exc
    scope = roots.scope(resolved)
    if scope is None and (
        resolved == roots.agent_home or resolved.is_relative_to(roots.agent_home)
    ):
        raise FileToolAccessDenied("Agent Home internal state is not readable by file tools")
    if scope is None:
        raise FileToolAccessDenied("path resolves outside the Workspace")
    relative_parts = (
        resolved.relative_to(roots.agent_home).parts
        if scope in {"memory", "artifact"}
        else resolved.relative_to(roots.workspace).parts
    )
    if os.name == "nt" and any(":" in part for part in relative_parts):
        raise FileToolAccessDenied("path identifies a Windows alternate data stream")
    return resolved


def _read_roots(context: ToolExecutionContext) -> _ReadRoots:
    workspace = _io_path(context.workspace).resolve(strict=True)
    agent_home = _io_path(context.agent_home).resolve(strict=False)
    workspace_slug = Workspace.from_path(context.workspace).slug
    return _ReadRoots(
        workspace=workspace,
        agent_home=agent_home,
        long_term_memory=agent_home / "memory" / "memory.md",
        artifact_directory=(
            agent_home / "sessions" / workspace_slug / "artifacts" / context.session_id
        ),
        session_id=context.session_id,
    )


def _is_windows_reserved(component: str) -> bool:
    normalized = component.rstrip(" .")
    basename = normalized.split(".", maxsplit=1)[0].upper()
    return basename in _WINDOWS_RESERVED_BASENAMES


def _io_path(path: Path) -> Path:
    if os.name != "nt":
        return path
    native = str(path.absolute())
    if native.startswith("\\\\?\\"):
        return path
    if native.startswith("\\\\"):
        return Path(f"\\\\?\\UNC\\{native.lstrip('\\')}")
    return Path(f"\\\\?\\{native}")


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
