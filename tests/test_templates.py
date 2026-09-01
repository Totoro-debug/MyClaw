import json
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path, PureWindowsPath

import pytest

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

    assert "## Long-term Memory" in base
    assert "### User Info" in base
    assert "Inline ### marker." in base
    assert "#### Nested Detail" in base
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
    loader = SkillLoader(
        root=first.parents[1],
        reserved_names=(),
        enable_always_load=False,
    )
    loader.load()
    foreground = foreground_chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        agent_home=PureWindowsPath(r"D:\agent-home"),
        long_term_memory="# Memory\n",
        skill_loader=loader,
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
    loader = SkillLoader(
        root=instruction.parents[1],
        reserved_names=(),
        enable_always_load=True,
    )
    loader.load()

    foreground = foreground_chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        agent_home=PureWindowsPath(r"D:\agent-home"),
        long_term_memory="# Memory\n",
        skill_loader=loader,
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


def test_chat_system_prompt_includes_runtime_and_catalog_guidance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("myclaw.agent.prompts.platform.system", lambda: "Windows")
    monkeypatch.setattr("myclaw.agent.prompts.platform.machine", lambda: "AMD64")
    monkeypatch.setattr(
        "myclaw.agent.prompts.platform.python_version",
        lambda: "3.12.13",
    )

    rendered = chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        agent_home=PureWindowsPath(r"D:\agent-home"),
        long_term_memory="# Memory\n",
    )
    assert rendered.startswith("# MyClaw Personal Agent")
    assert "`D:\\workspace`" in rendered
    assert "`D:\\agent-home`" in rendered
    assert "Windows AMD64, Python 3.12.13" in rendered
    assert "## Tool 使用指南" in rendered
    assert "`read_file`" in rendered
    assert "`schedule`" in rendered
    assert "## Long-term Memory" in rendered


def test_blackboard_prompt_renders_dynamic_markdown_context() -> None:
    rendered = blackboard_prompt(
        user_input="继续 {raw}。",
        last_task='{"task_goal":"旧目标","completion_boundary":"旧边界"}',
        latest_assistant_content="上一次回复。\n第二行。",
    )
    assert "## 输入" in rendered
    assert "### User input\n继续 {raw}。" in rendered
    assert (
        "### Last Task\n```json\n"
        '{"task_goal":"旧目标","completion_boundary":"旧边界"}\n```'
    ) in rendered
    assert "### Latest assistant content\n上一次回复。\n第二行。" in rendered
    assert "<user_input>" not in rendered
    assert "<last_task>" not in rendered


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
