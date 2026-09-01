"""Project validated Conversation Session messages for model input."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any

from myclaw.agent.prompts import interrupted_assistant_content


def _last_user_index(messages: Sequence[dict[str, Any]]) -> int:
    """Return the final user message index, or the sequence length if absent."""
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return len(messages)


def project_session_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Return one model-visible message without durable Session fields."""
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

    return {
        "role": "tool",
        "tool_call_id": deepcopy(message["tool_call_id"]),
        "name": deepcopy(message["name"]),
        "content": deepcopy(message["content"]),
    }


__all__ = ["project_session_message"]
