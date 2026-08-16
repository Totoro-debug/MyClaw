"""Build provider-neutral context for foreground Conversation Sessions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from myclaw.agent.prompts import (
    current_user_input,
    interrupted_assistant_content,
)
from myclaw.agent.workspace import Workspace
from myclaw.templates import render_template


class ContextBuilder:
    """Build the complete model-visible message list for one foreground turn."""

    def __init__(
        self,
        workspace: Workspace,
        timezone_name: str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("Context Builder requires a Workspace")
        self._workspace = workspace
        self._timezone = ZoneInfo(timezone_name)
        self._clock = clock

    def set_clock(self, clock: Callable[[], datetime] | None) -> None:
        """Override the clock used while composing Runtime Context in tests."""
        self._clock = clock

    def build_messages(
        self,
        history: Sequence[dict[str, Any]],
        current_user: dict[str, Any],
        session_id: str,
        long_term_memory: str,
    ) -> list[dict[str, Any]]:
        """Build system-first context without mutating any caller-owned message."""
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": render_template(
                    "foreground-chat-system-prompt.md",
                    identity=render_template("builtin-identity.md", workspace=self._workspace.path),
                    long_term_memory=long_term_memory,
                ),
            }
        ]
        messages.extend(
            projected
            for message in history
            if (projected := _project_history_message(message)) is not None
        )
        messages.append(
            {
                "role": "user",
                "content": current_user_input(
                    content=deepcopy(current_user["content"]),
                    current_time=self._current_time(),
                    session_id=session_id,
                ),
            }
        )
        return messages

    def _current_time(self) -> datetime:
        if self._clock is None:
            return datetime.now(self._timezone)
        current_time = self._clock()
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("Context Builder clock must return an aware datetime")
        return current_time.astimezone(self._timezone)


def _project_history_message(message: dict[str, Any]) -> dict[str, Any] | None:
    role = message["role"]
    if role == "user":
        return {"role": "user", "content": deepcopy(message["content"])}

    if role == "assistant":
        content = message["content"]
        projected_tool_calls = [
            {
                "id": deepcopy(tool_call["id"]),
                "name": deepcopy(tool_call["name"]),
                "arguments": deepcopy(tool_call["arguments"]),
            }
            for tool_call in message["tool_calls"]
        ]
        if message["status"] == "error" and not content and not projected_tool_calls:
            return None
        if message["status"] == "interrupted":
            content = interrupted_assistant_content(content)
        return {
            "role": "assistant",
            "content": deepcopy(content),
            "tool_calls": projected_tool_calls,
        }

    # Session owns structural validation, so the remaining supported role is Tool.
    return {
        "role": "tool",
        "tool_call_id": deepcopy(message["tool_call_id"]),
        "name": deepcopy(message["name"]),
        "content": deepcopy(message["content"]),
    }


__all__ = ["ContextBuilder"]
