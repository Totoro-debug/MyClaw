"""Render version-tracked prompts shared by runtime orchestration."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import PurePath
from typing import TYPE_CHECKING

from myclaw.memory.records import SummaryEntry
from myclaw.templates import render_template
from myclaw.utils.time import format_rfc3339_milliseconds

if TYPE_CHECKING:
    from myclaw.tools.base import OpenAIToolSchema


def current_user_input(*, content: str, current_time: datetime, session_id: str) -> str:
    """Wrap only the current raw user input with dynamic Runtime Context."""
    return render_template(
        "current-user-input.md",
        runtime_context=runtime_context(current_time=current_time, session_id=session_id),
        user_input=render_template("user-input.md", content=content),
    )


def runtime_context(*, current_time: datetime, session_id: str) -> str:
    """Render the per-turn metadata included in the next model request."""
    return render_template(
        "runtime-context.md",
        current_time=format_rfc3339_milliseconds(current_time),
        session_id=session_id,
    )


def chat_system_prompt(
    *, workspace: PurePath, long_term_memory: str, tool_guidance: str = ""
) -> str:
    """Compose chat system context in the accepted fixed order."""
    return render_template(
        "chat-system-prompt.md",
        identity=render_template("builtin-identity.md", workspace=workspace),
        long_term_memory=long_term_memory,
        tool_guidance=tool_guidance,
    )


def render_tool_guidance(schemas: Iterable[OpenAIToolSchema]) -> str:
    """Render the model-visible tool catalog in stable definition order."""
    return "\n".join(
        render_template(
            "tool-guidance-entry.md",
            name=schema["function"]["name"],
            description=schema["function"]["description"],
        )
        for schema in schemas
    )


def session_title_prompt() -> str:
    """Return the isolated prompt used for Session title generation."""
    return render_template("session-title-prompt.md")


def conversation_summary_prompt() -> str:
    """Return the isolated prompt used for Conversation Summary generation."""
    return render_template("conversation-summary-system-prompt.md")


def conversation_summary_input(*, messages: str) -> str:
    """Wrap serialized earlier messages for Conversation Summary generation."""
    return render_template("conversation-summary-input.md", messages=messages)


def memory_task_prompt(*, long_term_path: PurePath) -> str:
    """Return the restricted four-section Long-term Memory maintenance prompt."""
    return render_template("memory-task-prompt.md", long_term_path=long_term_path)


def memory_task_input(*, cursor: int, summaries: tuple[SummaryEntry, ...]) -> str:
    """Render only the pending ordered Conversation Summary batch."""
    records = "\n".join(entry.to_json_line().rstrip("\n") for entry in summaries)
    return render_template(
        "memory-task-input.md",
        cursor=cursor,
        records=records,
    )


def interrupted_assistant_content(content: str) -> str:
    """Mark interrupted assistant output when projecting persisted history."""
    return render_template("interrupted-assistant-content.md", content=content)
