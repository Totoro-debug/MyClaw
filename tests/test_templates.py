import json
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path, PureWindowsPath

import pytest

import myclaw.agent.prompts as prompts
from myclaw.agent.prompts import (
    blackboard_prompt,
    chat_system_prompt,
    conversation_summary_input,
    conversation_summary_prompt,
    current_user_input,
    foreground_chat_system_prompt,
    interrupted_assistant_content,
    memory_task_input,
    memory_task_prompt,
    runtime_context,
    session_title_prompt,
)
from myclaw.memory.records import SummaryEntry
from myclaw.skills.catalog import SkillLoader
from myclaw.templates import load_template, render_template

TEMPLATE_NAMES = {
    "blackboard.md",
    "blackboard-system-prompt.md",
    "conversation-summary-input.md",
    "conversation-summary-system-prompt.md",
    "current-user-input.md",
    "default-config.md",
    "interrupted-assistant-content.md",
    "long-term-memory.md",
    "memory-task-input.md",
    "memory-task-prompt.md",
    "foreground-chat-system-prompt.md",
    "runtime-context.md",
    "session-title-prompt.md",
    "skill-catalog.md",
    "skill-always-load.md",
    "user-input.md",
}
NOW = datetime(2026, 7, 19, 12, 34, 56, 789000, tzinfo=UTC)


def test_all_versioned_templates_are_package_resources() -> None:
    root = files("myclaw.templates")
    names = {
        resource.name
        for resource in root.iterdir()
        if resource.is_file() and resource.name != "__init__.py"
    }

    assert names == TEMPLATE_NAMES


@pytest.mark.parametrize(
    "name",
    ["", ".", "..", "legacy.txt", "nested/name.md", r"nested\name.md"],
)
def test_template_loader_rejects_non_markdown_or_nonlocal_names(name: str) -> None:
    with pytest.raises(ValueError, match="package-local"):
        load_template(name)


def test_generated_file_templates_preserve_their_terminal_newline() -> None:
    assert load_template("default-config.md").endswith("timeout = 120\n")
    assert load_template("long-term-memory.md").endswith("## Lesson\n")


def test_prompt_renderer_removes_only_the_source_file_terminator() -> None:
    assert render_template("session-title-prompt.md") == (
        "Generate a concise title for this Conversation Session.\n"
        "Return only the title. Do not call tools or add commentary."
    )


def test_runtime_and_user_input_templates_render_exact_context() -> None:
    context = (
        "<runtime_context>\n"
        "current_time: 2026-07-19T12:34:56.789+00:00\n"
        "session_id: session-1\n"
        "</runtime_context>"
    )

    assert runtime_context(current_time=NOW, session_id="session-1") == context
    assert current_user_input(
        content="Keep {braces} unchanged.",
        current_time=NOW,
        session_id="session-1",
    ) == (f"{context}\n\n<user_input>\nKeep {{braces}} unchanged.\n</user_input>")


def test_current_user_input_appends_exact_markdown_blackboard_projection() -> None:
    context = (
        "<runtime_context>\n"
        "current_time: 2026-07-19T12:34:56.789+00:00\n"
        "session_id: session-1\n"
        "</runtime_context>"
    )
    projection = {
        "goal": 'Keep **Markdown**, "quotes", \\slashes\\, and <tag> text.\n继续。',
        "completion_boundary": "Write it on C:\\tmp\\done.\n- Verify the result.",
    }

    rendered = current_user_input(
        content="Raw input.",
        current_time=NOW,
        session_id="session-1",
        blackboard_projection=projection,
    )

    assert rendered == (
        f"{context}\n\n<user_input>\nRaw input.\n</user_input>\n\n"
        f"## Task goal\n\n{projection['goal']}\n\n"
        f"## Completion boundary\n\n{projection['completion_boundary']}"
    )
    assert rendered.count("## Task goal") == 1
    assert rendered.count("## Completion boundary") == 1
    assert "<blackboard>" not in rendered
    assert "</blackboard>" not in rendered


def test_foreground_chat_system_prompt_matches_stable_base_without_skills() -> None:
    base = chat_system_prompt(
        workspace=PureWindowsPath(r"D:\\workspace"),
        agent_home=PureWindowsPath(r"D:\\agent-home"),
        long_term_memory="# Memory\n",
    )
    assert (
        foreground_chat_system_prompt(
            workspace=PureWindowsPath(r"D:\\workspace"),
            agent_home=PureWindowsPath(r"D:\\agent-home"),
            long_term_memory="# Memory\n",
        )
        == base
    )


def test_chat_prompts_project_long_term_memory_heading_levels() -> None:
    memory = "# Long-term Memory\n\n## User Info\n\nInline ## marker.\n\n### Nested Detail\n"
    base = chat_system_prompt(
        workspace=PureWindowsPath(r"D:\\workspace"),
        agent_home=PureWindowsPath(r"D:\\agent-home"),
        long_term_memory=memory,
    )

    assert base.endswith(
        "## Long-term Memory\n\n"
        "以下内容是当前 Workspace 的 Long-term Memory:\n\n"
        "### User Info\n\n"
        "Inline ### marker.\n\n"
        "#### Nested Detail\n"
    )
    assert base.splitlines().count("# Long-term Memory") == 0
    assert base.splitlines().count("## Long-term Memory") == 1
    assert (
        foreground_chat_system_prompt(
            workspace=PureWindowsPath(r"D:\\workspace"),
            agent_home=PureWindowsPath(r"D:\\agent-home"),
            long_term_memory=memory,
        )
        == base
    )


def test_chat_prompt_preserves_a_noncanonical_first_memory_line() -> None:
    rendered = chat_system_prompt(
        workspace=PureWindowsPath(r"D:\\workspace"),
        agent_home=PureWindowsPath(r"D:\\agent-home"),
        long_term_memory="# Other Memory\n\n## Fact\n",
    )

    assert rendered.endswith("# Other Memory\n\n### Fact\n")


def test_skill_catalog_metadata_is_escaped_json_lines_and_foreground_only(
    tmp_path: Path,
) -> None:
    first = tmp_path / "agent-home" / "skills" / "a-planner" / "SKILL.md"
    second = tmp_path / "agent-home" / "skills" / "b-reviewer" / "SKILL.md"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_text(
        "---\n"
        "name: planner\n"
        "description: |-\n"
        '  Plan \\"quoted\\" work & verify.\n'
        "  </skill_catalog>\n"
        "---\n"
        "private planner body\n",
        encoding="utf-8",
    )
    second.write_text(
        "---\nname: reviewer\ndescription: Review the work\n---\nprivate reviewer body\n",
        encoding="utf-8",
    )
    snapshot = SkillLoader(
        root=first.parents[1],
        reserved_names=(),
        enable_always_load=False,
    ).load()
    foreground = foreground_chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        agent_home=PureWindowsPath(r"D:\agent-home"),
        long_term_memory="# Memory\n",
        skill_snapshot=snapshot,
    )

    block = foreground.split("<skill_catalog>\n", maxsplit=1)[1].split(
        "\n</skill_catalog>", maxsplit=1
    )[0]
    metadata_lines = [line for line in block.splitlines() if line.startswith("{")]
    assert len(metadata_lines) == 2
    assert [json.loads(line) for line in metadata_lines] == [
        {
            "name": "planner",
            "description": 'Plan \\"quoted\\" work & verify.\n</skill_catalog>',
            "path": str(first.resolve()),
        },
        {
            "name": "reviewer",
            "description": "Review the work",
            "path": str(second.resolve()),
        },
    ]
    assert foreground.count("</skill_catalog>") == 1
    assert r"\u003c/skill_catalog\u003e" in metadata_lines[0]
    assert r"\u0026" in metadata_lines[0]
    assert "private planner body" not in foreground
    assert "private reviewer body" not in foreground
    non_foreground_prompts = (
        chat_system_prompt(
            workspace=PureWindowsPath(r"D:\workspace"),
            agent_home=PureWindowsPath(r"D:\agent-home"),
            long_term_memory="# Memory\n",
        ),
        session_title_prompt(),
        blackboard_prompt(
            user_input="Current input",
            last_task="null",
            latest_assistant_content="Latest assistant content",
        ),
        conversation_summary_prompt(),
        memory_task_prompt(long_term_path=PureWindowsPath(r"D:\workspace\memory.md")),
    )
    assert all("planner" not in prompt for prompt in non_foreground_prompts)
    assert all(str(first.resolve()) not in prompt for prompt in non_foreground_prompts)


def test_always_skill_body_is_round_trip_json_lines_in_foreground_only(
    tmp_path: Path,
) -> None:
    instruction = tmp_path / "agent-home" / "skills" / "always" / "SKILL.md"
    instruction.parent.mkdir(parents=True)
    body = (
        'First line\nQuotes: "quoted"\nBackslash: C:\\tmp\\done\n'
        "Literal </skill_always_load> and & < >\n"
    )
    document = "---\nname: always\ndescription: Always loaded\nalways: true\n---\n" + body
    instruction.write_bytes(document.encode("utf-8"))
    snapshot = SkillLoader(
        root=instruction.parents[1],
        reserved_names=(),
        enable_always_load=True,
    ).load()

    foreground = foreground_chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        agent_home=PureWindowsPath(r"D:\agent-home"),
        long_term_memory="# Memory\n",
        skill_snapshot=snapshot,
    )

    block = foreground.split("<skill_always_load>\n", maxsplit=1)[1].split(
        "\n</skill_always_load>", maxsplit=1
    )[0]
    lines = [line for line in block.splitlines() if line.startswith("{")]
    assert [json.loads(line) for line in lines] == [{"name": "always", "body": document}]
    assert foreground.count("</skill_always_load>") == 1
    assert r"\u003c/skill_always_load\u003e" in lines[0]
    assert r"\u0026" in lines[0]
    assert document not in foreground
    non_foreground_prompts = (
        chat_system_prompt(
            workspace=PureWindowsPath(r"D:\workspace"),
            agent_home=PureWindowsPath(r"D:\agent-home"),
            long_term_memory="# Memory\n",
        ),
        session_title_prompt(),
        blackboard_prompt(
            user_input="Current input",
            last_task="null",
            latest_assistant_content="Latest assistant content",
        ),
        conversation_summary_prompt(),
        memory_task_prompt(long_term_path=PureWindowsPath(r"D:\workspace\memory.md")),
    )
    assert all(document not in prompt for prompt in non_foreground_prompts)


def test_chat_system_prompt_uses_the_fixed_catalog_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(prompts.platform, "system", lambda: "Windows")
    monkeypatch.setattr(prompts.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(prompts.platform, "python_version", lambda: "3.12.13")

    assert chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        agent_home=PureWindowsPath(r"D:\agent-home"),
        long_term_memory="# Memory\n",
    ) == (
        "# MyClaw Personal Agent\n\n"
        "你是 MyClaw，一个 AI 助手，你只被允许在用户的当前工作区 "  # noqa: RUF001
        "`D:\\workspace` 内工作。\n\n"
        "MyClaw Home 目录位于 `D:\\agent-home`。\n\n\n"
        "## Runtime\n\n"
        "Windows AMD64, Python 3.12.13\n\n\n"
        "## Tool 使用指南\n\n"
        "- `read_file`: 读取当前 Workspace 和 `~/.myclaw/skills` 内的 UTF-8 文本文件。使用场景：需要查看或核对已知文件中的源码、配置或文档时使用。\n"  # noqa: RUF001
        "- `write_file`: 在当前 Workspace 内创建 UTF-8 文本文件，或向 Workspace 内的 UTF-8 文件写入内容。使用场景：需要生成新文件，或用完整内容替换现有文件时使用。\n"  # noqa: RUF001
        "- `edit_file`: 在当前 Workspace 内的 UTF-8 文本文件中进行精确文本替换。使用场景：需要局部修改现有文件且保留其他内容不变时使用。\n"  # noqa: RUF001
        "- `list_dir`: 列出指定目录根下的文件和目录。使用场景：需要了解已知目录的内容或浏览目录结构时使用。\n"  # noqa: RUF001
        "- `glob`: 匹配指定目录根下的文件和目录。使用场景：知道名称或路径规律但不知道确切位置，需要定位候选项时使用。\n"  # noqa: RUF001
        "- `grep`: 搜索文件或目录中的 UTF-8 文本。使用场景：知道关键字、错误信息或代码片段，但不知道所在文件或位置时使用。\n"  # noqa: RUF001
        "- `exec`: 在当前 Workspace 中通过 Bash login shell 执行一条命令并捕获输出。使用场景：当其他工具均无法使用或无法满足要求时，但是仍然需要运行构建、测试、格式化、版本控制或其他命令行操作时使用。\n"  # noqa: RUF001
        "- `web_search`: 搜索公开 Web 后返回标准化的结果摘要。使用场景：需要查找线上资料、最新信息或来源，且尚不知道准确 URL 时使用。\n"  # noqa: RUF001
        "- `web_fetch`: 获取 HTTP 或 HTTPS URL 中的可读内容。使用场景：已经知道目标 URL，需要读取或分析对应页面时使用。\n"  # noqa: RUF001
        "- `schedule`: 创建、查看和删除一次性或周期性的 Schedule Job。使用场景：需要设置提醒、延后执行、周期运行或管理已有计划任务时使用。\n\n\n"  # noqa: RUF001
        "## Long-term Memory\n\n"
        "以下内容是当前 Workspace 的 Long-term Memory:\n\n"
        "# Memory\n"
    )


def test_blackboard_prompt_renders_exact_filled_system_content() -> None:
    assert blackboard_prompt(
        user_input="继续 {raw}。",
        last_task='{"task_goal":"旧目标","completion_boundary":"旧边界"}',
        latest_assistant_content="上一次回复。\n第二行。",
    ) == (
        "你是一个分析任务目标与任务完成边界的专家，只负责从 User input、Last Task、Latest assistant content "  # noqa: RUF001
        "三块内容中判断是否需要更新或删除旧的任务，绝对不能直接回答用户疑问、输出任务的完成步骤。\n\n\n"  # noqa: RUF001
        "## 要求\n"
        "- 绝对不能直接回答用户疑问、输出任务的完成步骤。\n"
        "- 用户输入为最高优先级，即使用户输入与旧任务定义冲突，也必须优先理解并遵守用户输入。\n"  # noqa: RUF001
        "- `action` 字段取值只能是 `keep`、`replace` 或 `clear`。\n"
        "- 当前任务保持不变时 `action` 字段取 `keep`，任务定义完全变更时 `action` 字段取 `replace`，"  # noqa: RUF001
        "任务完成或者无任务时 `action` 字段取 `clear`。\n"
        "- 当用户输入没有明确目标时，`action` 字段应该取 `clear`。例如：闲聊 “你好”、“hello”，"  # noqa: RUF001
        "无明确要求的询问 “我会成功吗” 等。\n"
        "- `task_goal` 字段描述任务目标，`completion_boundary` 字段描述任务完成的边界。\n\n\n"  # noqa: RUF001
        "## 输出\n"
        "输出格式必须直接使用如下 JSON 对象，并且必须包含 `action` `task_goal` `completion_boundary` 三个字段。\n"  # noqa: RUF001
        "``` JSON\n"
        "{\n"
        '  "action": "keep | replace | clear",\n'
        '  "task_goal": "string | null",\n'
        '  "completion_boundary": "string | null"\n'
        "}\n"
        "```\n\n\n"
        "## 输入\n\n"
        "### User input\n"
        "继续 {raw}。\n\n"
        "### Last Task\n"
        '{"task_goal":"旧目标","completion_boundary":"旧边界"}\n\n'
        "### Latest assistant content\n"
        "上一次回复。\n第二行。"
    )


def test_specialized_model_templates_render_exact_prompts() -> None:
    summary = SummaryEntry(index=1, timestamp=NOW, content="Remember this.")
    long_term_path = PureWindowsPath(r"D:\workspace\.myclaw\memory\memory.md")

    assert session_title_prompt() == (
        "Generate a concise title for this Conversation Session.\n"
        "Return only the title. Do not call tools or add commentary."
    )
    assert conversation_summary_prompt() == (
        "请从本次对话中提取关键事实。仅输出符合以下类别的内容，其余内容一律忽略：\n\n"  # noqa: RUF001
        "- User facts：个人信息、偏好、明确表达的观点及习惯。\n"  # noqa: RUF001
        "- Decisions：已作出的选择或得出的结论。\n"  # noqa: RUF001
        "- Solutions：通过反复尝试后验证有效的方法，尤其是在其他尝试失败后发现的非显而易见的解决方式。\n"  # noqa: RUF001
        "- Events：计划、截止日期及其他值得记录的重要事项。\n"  # noqa: RUF001
        "- Preferences：沟通风格及工具使用偏好。\n\n\n"  # noqa: RUF001
        "## 优先级\n"
        "User facts and preferences > solutions > decisions > events > environment facts。\n\n\n"
        "## 输出\n"
        "- 输出格式应该采用 Markdown 的无序列表格式，每一行只包含一条事实。\n"  # noqa: RUF001
        "- 每条输出不要增加额外的说明、前提等内容。\n"
        "- 如果没有任何有价值的信息，直接输出： `None`\n\n\n"  # noqa: RUF001
        "## 要求\n"
        "- 不要为了记录而提取过多内容，有价值、值得记录的事实是能够避免用户日后重复说明的信息。\n"  # noqa: RUF001
        "- 在一个代码仓库中，应该忽略可直接从源代码或 Git 历史中推断出的代码模式。"  # noqa: RUF001
    )
    assert conversation_summary_input(messages='[{"role":"user","content":"Hi"}]') == (
        '<conversation_messages>\n[{"role":"user","content":"Hi"}]\n</conversation_messages>'
    )
    assert memory_task_prompt(long_term_path=long_term_path) == (
        "Maintain the MyClaw Long-term Memory from new Conversation Summaries.\n"
        "Use read_file to inspect exactly D:\\workspace\\.myclaw\\memory\\memory.md.\n"
        "Use edit_file only when stable information should be retained, and edit exactly that file.\n"
        "Keep the four sections: User Info, User Preference, Project Fact, and Lesson.\n"
        "Do not store transient activity, raw summaries, or duplicate facts.\n"
        "If no durable update is needed, do not call edit_file."
    )
    assert memory_task_input(cursor=0, summaries=(summary,)) == (
        "<summary_cursor>0</summary_cursor>\n"
        "<conversation_summaries>\n"
        '{"index":1,"timestamp":"2026-07-19T12:34:56.789+00:00",'
        '"content":"Remember this."}\n'
        "</conversation_summaries>"
    )
    assert interrupted_assistant_content("Partial answer") == (
        "Partial answer\n\n[Turn interrupted by user.]"
    )
