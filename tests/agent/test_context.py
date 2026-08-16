from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfoNotFoundError

import pytest

import myclaw.agent.context as context
from myclaw.agent.context import ContextBuilder
from myclaw.agent.workspace import Workspace

FIXED_UTC = datetime(2026, 8, 16, 4, 5, 6, 789000, tzinfo=UTC)
FIXED_TOOL_GUIDANCE = "\n".join(
    (
        "- read_file: Read UTF-8 text lines from a file within the current Workspace.",
        "- write_file: Write UTF-8 text to a file within the current Workspace.",
        "- edit_file: Replace exact UTF-8 text in a file within the current Workspace.",
        "- list_dir: List files and directories within a directory root.",
        "- glob: Match files and directories beneath a directory root.",
        "- grep: Search UTF-8 text in a file or directory.",
        "- exec: Run one Bash login-shell command with captured output in the current Workspace.",
        "- web_search: Search the public web and return normalized result summaries.",
        "- web_fetch: Fetch readable content from an HTTP or HTTPS URL.",
        "- schedule: Manage one-time and recurring Schedule Jobs.",
    )
)


class _FrozenDateTime(datetime):
    @classmethod
    def now(cls, tz: object = None) -> _FrozenDateTime:
        if tz is None:
            return cls.fromtimestamp(FIXED_UTC.timestamp(), tz=UTC)
        return cls.fromtimestamp(FIXED_UTC.timestamp(), tz=tz)  # type: ignore[arg-type]


def _builder(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    timezone_name: str,
) -> ContextBuilder:
    monkeypatch.setattr(context, "datetime", _FrozenDateTime)
    return context.ContextBuilder(Workspace.from_path(workspace), timezone_name)


def test_context_builder_rejects_an_invalid_iana_timezone_before_building(
    workspace: Path,
) -> None:
    with pytest.raises(ZoneInfoNotFoundError):
        ContextBuilder(Workspace.from_path(workspace), "Mars/Olympus")


@pytest.mark.parametrize(
    ("timezone_name", "expected_time"),
    [("UTC", "2026-08-16T04:05:06.789+00:00"), ("Asia/Shanghai", "2026-08-16T12:05:06.789+08:00")],
)
def test_context_builder_builds_system_history_and_current_user_in_order(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
    timezone_name: str,
    expected_time: str,
) -> None:
    builder = _builder(monkeypatch, workspace, timezone_name)
    history: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": "Earlier question.",
            "timestamp": "2026-08-15T12:00:00.000+08:00",
        },
        {
            "role": "assistant",
            "content": "Earlier answer.",
            "timestamp": "2026-08-15T12:00:01.000+08:00",
            "tool_calls": [],
            "status": "completed",
        },
    ]
    current_user: dict[str, Any] = {
        "role": "user",
        "content": "Current question.",
        "timestamp": "2026-08-16T12:00:00.000+08:00",
    }

    messages = builder.build_messages(
        history=history,
        current_user=current_user,
        session_id="20260816-120000-000000_550e8400-e29b-41d4-a716-446655440000",
        long_term_memory="# Memory\nRemember this.",
    )

    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == (
        "You are the MyClaw Personal Agent.\n"
        "Act within the user's current Workspace.\n"
        f"Workspace: {Workspace.from_path(workspace).path}\n\n"
        "<long_term_memory>\n"
        "# Memory\n"
        "Remember this.</long_term_memory>\n\n"
        "<tool_guidance>\n"
        f"{FIXED_TOOL_GUIDANCE}</tool_guidance>"
    )
    assert messages[1] == {"role": "user", "content": "Earlier question."}
    assert messages[2] == {
        "role": "assistant",
        "content": "Earlier answer.",
        "tool_calls": [],
    }
    assert messages[3] == {
        "role": "user",
        "content": (
            "<runtime_context>\n"
            f"current_time: {expected_time}\n"
            "session_id: 20260816-120000-000000_550e8400-e29b-41d4-a716-446655440000\n"
            "</runtime_context>\n\n"
            "<user_input>\n"
            "Current question.\n"
            "</user_input>"
        ),
    }
    assert all("timestamp" not in message for message in messages)


def test_context_builder_projects_tool_and_interrupted_history_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    history: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "Calling.",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": '{"path":"a.txt"}'}],
            "status": "completed",
            "timestamp": "old",
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "name": "read_file",
            "content": "File contents.",
            "timestamp": "old",
        },
        {
            "role": "assistant",
            "content": "Partial.",
            "tool_calls": [],
            "status": "interrupted",
            "timestamp": "old",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [],
            "status": "error",
            "timestamp": "old",
        },
    ]
    current_user: dict[str, Any] = {"role": "user", "content": "Now", "timestamp": "old"}
    original_history = deepcopy(history)
    original_current_user = deepcopy(current_user)

    messages = builder.build_messages(
        history=history,
        current_user=current_user,
        session_id="session-id",
        long_term_memory="memory",
    )

    assert messages[1] == {
        "role": "assistant",
        "content": "Calling.",
        "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": '{"path":"a.txt"}'}],
    }
    assert messages[2] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "name": "read_file",
        "content": "File contents.",
    }
    assert messages[3] == {
        "role": "assistant",
        "content": "Partial.\n\n[Turn interrupted by user.]",
        "tool_calls": [],
    }
    assert len(messages) == 5
    assert history == original_history
    assert current_user == original_current_user

    projected_call = messages[1]["tool_calls"]
    assert isinstance(projected_call, list)
    projected_call[0]["arguments"] = "changed"
    assert history[0]["tool_calls"][0]["arguments"] == '{"path":"a.txt"}'


def test_context_builder_does_not_accept_tool_gateway_or_schemas(workspace: Path) -> None:
    with pytest.raises(TypeError):
        ContextBuilder(Workspace.from_path(workspace), "UTC", tool_gateway=object())  # type: ignore[call-arg]
