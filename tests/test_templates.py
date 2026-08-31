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
    "builtin-identity.md",
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
        long_term_memory="# Memory\n",
    )
    assert (
        foreground_chat_system_prompt(
            workspace=PureWindowsPath(r"D:\\workspace"),
            long_term_memory="# Memory\n",
        )
        == base
    )


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


def test_chat_system_prompt_uses_the_fixed_catalog_guidance() -> None:
    assert chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        long_term_memory="# Memory\n",
    ) == (
        "You are the MyClaw Personal Agent.\n"
        "Act within the user's current Workspace.\n"
        "Workspace: D:\\workspace\n\n"
        "<long_term_memory>\n"
        "# Memory\n"
        "</long_term_memory>\n\n"
        "<tool_guidance>\n"
        "- read_file: Read UTF-8 text lines from a file within the current Workspace.\n"
        "- write_file: Write UTF-8 text to a file within the current Workspace.\n"
        "- edit_file: Replace exact UTF-8 text in a file within the current Workspace.\n"
        "- list_dir: List files and directories within a directory root.\n"
        "- glob: Match files and directories beneath a directory root.\n"
        "- grep: Search UTF-8 text in a file or directory.\n"
        "- exec: Run one Bash login-shell command with captured output in the current Workspace.\n"
        "- web_search: Search the public web and return normalized result summaries.\n"
        "- web_fetch: Fetch readable content from an HTTP or HTTPS URL.\n"
        "- schedule: Manage one-time and recurring Schedule Jobs.</tool_guidance>"
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
        "Summarize the provided earlier conversation messages.\n"
        "Preserve decisions, user intent, important facts, and unresolved work concisely."
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
