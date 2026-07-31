from datetime import UTC, datetime
from importlib.resources import files
from pathlib import PureWindowsPath

import pytest

from myclaw.agent.prompts import (
    chat_system_prompt,
    conversation_summary_input,
    conversation_summary_prompt,
    current_user_input,
    interrupted_assistant_content,
    memory_task_input,
    memory_task_prompt,
    render_tool_guidance,
    runtime_context,
    session_title_prompt,
)
from myclaw.memory.records import SummaryEntry
from myclaw.templates import load_template, render_template
from myclaw.tools.schema import OpenAIToolSchema

TEMPLATE_NAMES = {
    "builtin-identity.md",
    "chat-system-prompt.md",
    "conversation-summary-input.md",
    "conversation-summary-system-prompt.md",
    "current-user-input.md",
    "default-config.md",
    "interrupted-assistant-content.md",
    "long-term-memory.md",
    "memory-task-input.md",
    "memory-task-prompt.md",
    "runtime-context.md",
    "session-title-prompt.md",
    "tool-guidance-entry.md",
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


def test_chat_and_tool_templates_render_exact_system_prompt() -> None:
    schemas: tuple[OpenAIToolSchema, ...] = (
        {
            "type": "function",
            "function": {"name": "read_file", "description": "Read a file.", "parameters": {}},
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write a file.",
                "parameters": {},
            },
        },
    )
    guidance = render_tool_guidance(schemas)

    assert guidance == "- read_file: Read a file.\n- write_file: Write a file."
    assert chat_system_prompt(
        workspace=PureWindowsPath(r"D:\workspace"),
        long_term_memory="# Memory\n",
        tool_guidance=guidance,
    ) == (
        "You are the MyClaw Personal Agent.\n"
        "Act within the user's current Workspace.\n"
        "Workspace: D:\\workspace\n\n"
        "<long_term_memory>\n"
        "# Memory\n"
        "</long_term_memory>\n\n"
        "<tool_guidance>\n"
        "- read_file: Read a file.\n"
        "- write_file: Write a file.</tool_guidance>"
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
