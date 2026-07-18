"""Workspace-scoped file mutation Tools."""

import os
from pathlib import Path
from stat import S_ISREG

from myclaw.contracts import JsonObject, ToolDefinition, ToolExecutionContext
from myclaw.tools.files.file_tools import FileToolAccessDenied, FileToolArgumentsError

_WINDOWS_RESERVED_BASENAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{index}" for index in range(1, 10)}
    | {f"LPT{index}" for index in range(1, 10)}
)


def resolve_workspace_write_path(context: ToolExecutionContext, requested: str) -> Path:
    """Resolve an existing target or its nearest existing parent inside the Workspace."""
    candidate = Path(requested)
    try:
        workspace = context.workspace.resolve(strict=True)
        agent_home = context.agent_home.resolve(strict=False)
        if not candidate.is_absolute():
            candidate = workspace / candidate
        target = candidate.resolve(strict=False)
    except (OSError, RuntimeError, ValueError) as error:
        raise FileToolArgumentsError("path cannot be resolved") from error
    if not target.is_relative_to(workspace):
        raise FileToolAccessDenied("path resolves outside the Workspace")
    if target == agent_home or target.is_relative_to(agent_home):
        raise FileToolAccessDenied("Agent Home internal state cannot be changed by file tools")
    relative_parts = target.relative_to(workspace).parts
    if os.name == "nt" and any(":" in part for part in relative_parts):
        raise FileToolAccessDenied("path identifies a Windows alternate data stream")
    if os.name == "nt" and any(_is_windows_reserved(part) for part in relative_parts):
        raise FileToolAccessDenied("path identifies a Windows device")
    if target.exists():
        status = target.lstat()
        if not S_ISREG(status.st_mode) or status.st_nlink != 1:
            raise FileToolAccessDenied("path must identify an unaliased regular file")
    existing_parent = target.parent
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if not existing_parent.is_dir():
        raise FileToolAccessDenied("nearest existing parent must be a directory")
    return target


def _is_windows_reserved(component: str) -> bool:
    normalized = component.rstrip(" .")
    basename = normalized.split(".", maxsplit=1)[0].upper()
    return basename in _WINDOWS_RESERVED_BASENAMES


class WriteFileTool:
    """Write exact UTF-8 text to one Workspace file."""

    _definition = ToolDefinition(
        name="write_file",
        description="Write UTF-8 text to a file within the current Workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        requested = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(requested, str) or not requested or not isinstance(content, str):
            raise FileToolArgumentsError("write_file requires a path and content")

        target = resolve_workspace_write_path(context, requested)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="")
        return f"Wrote {target.name}."


class EditFileTool:
    """Replace exact UTF-8 text in one Workspace file."""

    _definition = ToolDefinition(
        name="edit_file",
        description="Replace exact UTF-8 text in a file within the current Workspace.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean", "default": False},
            },
            "required": ["path", "old_text", "new_text"],
            "additionalProperties": False,
        },
    )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: JsonObject, context: ToolExecutionContext) -> str:
        requested = arguments.get("path")
        old_text = arguments.get("old_text")
        new_text = arguments.get("new_text")
        replace_all = arguments.get("replace_all", False)
        if (
            not isinstance(requested, str)
            or not requested
            or not isinstance(old_text, str)
            or not old_text
            or not isinstance(new_text, str)
            or not isinstance(replace_all, bool)
        ):
            raise FileToolArgumentsError("edit_file requires a path and replacement text")

        target = resolve_workspace_write_path(context, requested)
        content = target.read_bytes().decode("utf-8")
        match_count = content.count(old_text)
        if match_count == 0 or (not replace_all and match_count != 1):
            raise FileToolArgumentsError("old_text must match exactly once")
        replaced = (
            content.replace(old_text, new_text)
            if replace_all
            else content.replace(old_text, new_text, 1)
        )
        target.write_text(
            replaced,
            encoding="utf-8",
            newline="",
        )
        return f"Edited {target.name}."
