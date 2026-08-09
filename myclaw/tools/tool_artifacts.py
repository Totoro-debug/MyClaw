"""Compatibility adapter for the BaseTool result-handling capability."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from myclaw.session.session import Session
from myclaw.tools.base import ArtifactReference, BaseTool
from myclaw.tools.tool_gateway import ToolResult

type ArtifactWriter = Callable[[Path, str], None]

_ARTIFACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class ArtifactWriteError(Exception):
    """Retained for callers that still import the former adapter exception."""


def encode_artifact_tool_call_id(tool_call_id: str) -> str:
    """Return a legal artifact filename component or a UUID fallback."""
    return tool_call_id if _ARTIFACT_ID_PATTERN.fullmatch(tool_call_id) else str(uuid4())


def externalize_tool_result(
    result: ToolResult,
    *,
    session: Session,
    max_tool_result_chars: int,
    write_text: ArtifactWriter | None = None,
) -> ToolResult:
    """Adapt a normalized Tool Result to BaseTool's content-only handler."""
    if result.status != "success" or len(result.content) <= max_tool_result_chars:
        return result
    output = BaseTool.handle_result(
        result.content,
        workspace=session.workspace_state.workspace,
        session_id=session.session_id,
        tool_call_id=result.tool_call_id,
        limit=max_tool_result_chars,
        write_text=write_text,
    )
    return replace(result, content=output.content, artifact=output.artifact)


__all__ = [
    "ArtifactReference",
    "ArtifactWriteError",
    "encode_artifact_tool_call_id",
    "externalize_tool_result",
]
