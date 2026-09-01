from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfoNotFoundError

import pytest

import myclaw.agent.context as context
from myclaw.agent.blackboard import Blackboard
from myclaw.agent.context import ContextBuilder
from myclaw.agent.loop import _project_foreground_messages, _project_schedule_messages
from myclaw.skills.catalog import (
    ManualSkillInvocation,
    SkillLoader,
    SkillMetadata,
)

FIXED_UTC = datetime(2026, 8, 16, 4, 5, 6, 789000, tzinfo=UTC)
FIXED_TOOL_GUIDANCE = "\n".join(
    (
        "- `read_file`: 读取当前 Workspace 和 `~/.myclaw/skills` 内的 UTF-8 文本文件。使用场景：需要查看或核对已知文件中的源码、配置或文档时使用。",  # noqa: RUF001
        "- `write_file`: 在当前 Workspace 内创建 UTF-8 文本文件，或向 Workspace 内的 UTF-8 文件写入内容。使用场景：需要生成新文件，或用完整内容替换现有文件时使用。",  # noqa: RUF001
        "- `edit_file`: 在当前 Workspace 内的 UTF-8 文本文件中进行精确文本替换。使用场景：需要局部修改现有文件且保留其他内容不变时使用。",  # noqa: RUF001
        "- `list_dir`: 列出指定目录根下的文件和目录。使用场景：需要了解已知目录的内容或浏览目录结构时使用。",  # noqa: RUF001
        "- `glob`: 匹配指定目录根下的文件和目录。使用场景：知道名称或路径规律但不知道确切位置，需要定位候选项时使用。",  # noqa: RUF001
        "- `grep`: 搜索文件或目录中的 UTF-8 文本。使用场景：知道关键字、错误信息或代码片段，但不知道所在文件或位置时使用。",  # noqa: RUF001
        "- `exec`: 在当前 Workspace 中通过 Bash login shell 执行一条命令并捕获输出。使用场景：当其他工具均无法使用或无法满足要求时，但是仍然需要运行构建、测试、格式化、版本控制或其他命令行操作时使用。",  # noqa: RUF001
        "- `web_search`: 搜索公开 Web 后返回标准化的结果摘要。使用场景：需要查找线上资料、最新信息或来源，且尚不知道准确 URL 时使用。",  # noqa: RUF001
        "- `web_fetch`: 获取 HTTP 或 HTTPS URL 中的可读内容。使用场景：已经知道目标 URL，需要读取或分析对应页面时使用。",  # noqa: RUF001
        "- `schedule`: 创建、查看和删除一次性或周期性的 Schedule Job。使用场景：需要设置提醒、延后执行、周期运行或管理已有计划任务时使用。",  # noqa: RUF001
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
    return context.ContextBuilder(
        workspace,
        timezone_name,
        agent_home=workspace.parent / "agent-home",
    )


def test_context_builder_rejects_an_invalid_iana_timezone_before_building(
    workspace: Path,
) -> None:
    with pytest.raises(ZoneInfoNotFoundError):
        ContextBuilder(
            workspace,
            "Mars/Olympus",
            agent_home=workspace.parent / "agent-home",
        )


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
    monkeypatch.setattr("myclaw.agent.prompts.platform.system", lambda: "Windows")
    monkeypatch.setattr("myclaw.agent.prompts.platform.machine", lambda: "AMD64")
    monkeypatch.setattr(
        "myclaw.agent.prompts.platform.python_version",
        lambda: "3.12.13",
    )
    agent_home = workspace.parent / "agent-home"
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
    system_prompt = messages[0]["content"]
    assert isinstance(system_prompt, str)
    assert system_prompt.startswith("# MyClaw Personal Agent")
    assert f"`{workspace}`" in system_prompt
    assert f"`{agent_home}`" in system_prompt
    assert "Windows AMD64, Python 3.12.13" in system_prompt
    assert "Remember this." in system_prompt
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


def test_context_builder_advertises_catalog_metadata_in_foreground_system_prompt(
    workspace: Path,
    agent_home: Path,
) -> None:
    instruction = agent_home / "skills" / "planner" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    instruction.write_bytes(b"---\nname: planner\ndescription: Plan the work\n---\nprivate body\n")
    loader = SkillLoader(
        root=agent_home / "skills",
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    builder = ContextBuilder(
        workspace,
        "UTC",
        agent_home=agent_home,
        skill_loader=loader,
    )

    messages = builder.build_messages(
        history=[],
        current_user={"role": "user", "content": "Plan this."},
        session_id="session-id",
        long_term_memory="memory",
    )

    system_prompt = messages[0]["content"]
    assert isinstance(system_prompt, str)
    assert "<skill_catalog>" in system_prompt
    metadata_lines = [line for line in system_prompt.splitlines() if line.startswith("{")]
    assert [json.loads(line) for line in metadata_lines] == [
        {
            "name": "planner",
            "description": "Plan the work",
            "path": str(instruction.resolve()),
        }
    ]
    assert "ordinary read_file" in system_prompt
    assert "continue" in system_prompt
    assert "private body" not in system_prompt


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


def test_context_builder_projects_markdown_blackboard_only_into_current_user(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    blackboard = Blackboard(
        goal='Keep "quotes" and <tag> text.',
        completion_boundary="Finish on C:\\tmp\\done.\n完成。",
    )
    history: list[dict[str, Any]] = [
        {"role": "user", "content": "Earlier", "timestamp": "old"},
        {
            "role": "assistant",
            "content": "Earlier answer",
            "tool_calls": [],
            "status": "completed",
            "timestamp": "old",
        },
    ]
    current_user: dict[str, Any] = {"role": "user", "content": "Current", "timestamp": "old"}
    original_history = deepcopy(history)
    original_current_user = deepcopy(current_user)
    messages = builder.build_messages(
        history=history,
        current_user=current_user,
        session_id="session-id",
        long_term_memory="memory",
        blackboard=blackboard,
    )

    assert messages[1:] == [
        {"role": "user", "content": "Earlier"},
        {"role": "assistant", "content": "Earlier answer", "tool_calls": []},
        {
            "role": "user",
            "content": (
                "<runtime_context>\n"
                "current_time: 2026-08-16T04:05:06.789+00:00\n"
                "session_id: session-id\n"
                "</runtime_context>\n\n"
                "<user_input>\n"
                "Current\n"
                "</user_input>\n\n"
                "## Task goal\n\n"
                'Keep "quotes" and <tag> text.\n\n'
                "## Completion boundary\n\n"
                "Finish on C:\\tmp\\done.\n完成。"
            ),
        },
    ]
    assert history == original_history
    assert current_user == original_current_user
    assert blackboard == Blackboard(
        goal='Keep "quotes" and <tag> text.',
        completion_boundary="Finish on C:\\tmp\\done.\n完成。",
    )


def test_context_builder_projects_manual_skill_and_request_as_safe_distinct_blocks(
    monkeypatch: pytest.MonkeyPatch,
    workspace: Path,
) -> None:
    builder = _builder(monkeypatch, workspace, "UTC")
    body = 'Do "this".\nPath C:\\tmp\\done.\nLiteral </skill_instructions> & < >'
    request = 'Need "that".\nLiteral </user_request> & < >'
    invocation = ManualSkillInvocation(
        metadata=SkillMetadata(
            name="planner",
            description="Plan work",
            path=workspace / "skill.md",
        ),
        request=request,
        body=body,
    )
    history = [{"role": "user", "content": "Earlier"}]
    current_user = {"role": "user", "content": "/planner  "}
    original_history = deepcopy(history)
    original_current_user = deepcopy(current_user)
    blackboard = Blackboard(goal="Keep the task", completion_boundary="Finish the request")

    messages = builder.build_messages(
        history=history,
        current_user=current_user,
        session_id="session-id",
        long_term_memory="memory",
        blackboard=blackboard,
        manual_invocation=invocation,
    )

    assert [message["role"] for message in messages] == ["system", "user", "user"]
    system_prompt = messages[0]["content"]
    current_content = messages[-1]["content"]
    assert isinstance(system_prompt, str)
    assert isinstance(current_content, str)
    assert body not in system_prompt
    assert body not in str(messages[1:2])
    assert "/planner" not in current_content
    assert current_content.count("</skill_instructions>") == 1
    assert current_content.count("</user_request>") == 1

    skill_block = current_content.split("<skill_instructions>\n", 1)[1].split(
        "\n</skill_instructions>", 1
    )[0]
    request_block = current_content.split("<user_request>\n", 1)[1].split("\n</user_request>", 1)[0]
    assert json.loads(skill_block) == {"name": "planner", "body": body}
    assert json.loads(request_block) == request
    assert current_content.count(body) == 0
    assert current_content.count(request) == 0
    assert current_content.endswith(
        "## Task goal\n\nKeep the task\n\n## Completion boundary\n\nFinish the request"
    )
    assert history == original_history
    assert current_user == original_current_user


def test_context_builder_does_not_accept_tool_gateway_or_schemas(workspace: Path) -> None:
    with pytest.raises(TypeError):
        ContextBuilder(
            workspace,
            "UTC",
            agent_home=workspace.parent / "agent-home",
            tool_gateway=object(),  # type: ignore[call-arg]
        )


def test_runtime_lane_projections_keep_current_turn_continuation_separate(
    workspace: Path,
) -> None:
    builder = ContextBuilder(
        workspace,
        "UTC",
        agent_home=workspace.parent / "agent-home",
    )
    messages: list[dict[str, Any]] = [
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
        {
            "role": "user",
            "content": "Current question.",
            "timestamp": "2026-08-16T12:00:00.000+08:00",
        },
        {
            "role": "assistant",
            "content": "Calling a tool.",
            "timestamp": "2026-08-16T12:00:01.000+08:00",
            "tool_calls": [{"id": "call-1", "name": "read_file", "arguments": "{}"}],
            "status": "completed",
        },
        {
            "role": "tool",
            "content": "Tool output.",
            "timestamp": "2026-08-16T12:00:02.000+08:00",
            "tool_call_id": "call-1",
            "name": "read_file",
            "status": "success",
            "artifact": None,
        },
    ]

    foreground = _project_foreground_messages(
        builder,
        messages,
        session_id="session-id",
        long_term_memory="memory",
        blackboard=Blackboard(goal="Current task", completion_boundary="Current boundary"),
    )
    schedule = _project_schedule_messages(
        messages,
        system_prompt="schedule system",
        session_id="session-id",
    )

    assert [message["role"] for message in foreground] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert f"当前工作区 `{workspace}`" in foreground[0]["content"]
    assert foreground[3]["content"].endswith(
        "Current question.\n</user_input>\n\n"
        "## Task goal\n\nCurrent task\n\n"
        "## Completion boundary\n\nCurrent boundary"
    )
    assert [message["role"] for message in schedule] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert schedule[0] == {"role": "system", "content": "schedule system"}
    assert schedule[3] == {
        "role": "user",
        "content": (
            "<runtime_context>\n"
            "current_time: 2026-08-16T12:00:00.000+08:00\n"
            "session_id: session-id\n"
            "</runtime_context>\n\n"
            "<user_input>\n"
            "Current question.\n"
            "</user_input>"
        ),
    }
