"""Render version-tracked prompts shared by runtime orchestration."""

from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import PurePath

from myclaw.skills.catalog import ManualSkillInvocation, SkillLoader
from myclaw.templates import render_template
from myclaw.utils.time import format_rfc3339_milliseconds

_MARKDOWN_SAFE_JSON_TRANSLATION: dict[int, str] = {
    ord("&"): r"\u0026",
    ord("`"): r"\u0060",
    ord("<"): r"\u003c",
    ord(">"): r"\u003e",
}


def current_user_input(
    *,
    content: str,
    current_time: datetime,
    session_id: str,
    blackboard_projection: dict[str, str] | None = None,
    manual_invocation: ManualSkillInvocation | None = None,
) -> str:
    """Wrap one current user projection and optionally append Runtime-owned blocks."""
    if manual_invocation is None:
        rendered = (
            f"{runtime_context(current_time=current_time, session_id=session_id)}\n\n"
            "## User Input\n\n"
            f"{content}"
        )
    else:
        rendered = (
            f"{runtime_context(current_time=current_time, session_id=session_id)}\n\n"
            "## Skill Instructions\n\n"
            "```json\n"
            f"{_skill_manual_json(manual_invocation)}\n"
            "```\n\n"
            "## User Request\n\n"
            "```json\n"
            f"{_skill_request_json(manual_invocation.request)}\n"
            "```"
        )
    if blackboard_projection is None:
        return rendered
    blackboard = (
        "## Task goal\n\n"
        f"{blackboard_projection['goal']}\n\n"
        "## Completion boundary\n\n"
        f"{blackboard_projection['completion_boundary']}"
    )
    return f"{rendered}\n\n{blackboard}"


def runtime_context(*, current_time: datetime, session_id: str) -> str:
    """Render the per-turn metadata included in the next model request."""
    return (
        "## Runtime Context\n\n"
        f"- Current time: {format_rfc3339_milliseconds(current_time)}\n"
        f"- Session ID: {session_id}"
    )


def _project_long_term_memory(long_term_memory: str) -> str:
    """Nest the persisted Long-term Memory document in its prompt section."""
    heading = "# Long-term Memory"
    if long_term_memory == heading:
        projected = ""
    elif long_term_memory.startswith(f"{heading}\n"):
        projected = long_term_memory.removeprefix(f"{heading}\n").removeprefix("\n")
    else:
        projected = long_term_memory
    return projected.replace("##", "###")


def chat_system_prompt(
    *,
    workspace: PurePath,
    agent_home: PurePath,
    long_term_memory: str,
) -> str:
    """Compose the fixed Tool-guidance System Prompt."""
    runtime = (
        f"{platform.system()} "
        f"{platform.machine()}, Python {platform.python_version()}"
    )
    return render_template(
        "foreground-chat-system-prompt.md",
        workspace=workspace,
        agent_home=agent_home,
        runtime=runtime,
        long_term_memory=_project_long_term_memory(long_term_memory),
    )


def foreground_chat_system_prompt(
    *,
    workspace: PurePath,
    agent_home: PurePath,
    long_term_memory: str,
    skill_loader: SkillLoader | None = None,
) -> str:
    """Compose the foreground prompt with optional Skill metadata."""
    sections = [
        chat_system_prompt(
            workspace=workspace,
            agent_home=agent_home,
            long_term_memory=long_term_memory,
        )
    ]
    if skill_loader is not None and skill_loader.skills:
        entries = "\n".join(
            _skill_metadata_json(
                name=metadata.name,
                description=metadata.description,
                path=str(metadata.path),
            )
            for metadata in skill_loader.metadata
        )
        sections.append(render_template("skill-catalog.md", entries=entries))
        always_entries = "\n".join(
            _skill_always_json(name=skill.metadata.name, body=skill.document)
            for skill in skill_loader.skills
            if skill.always
        )
        if always_entries:
            sections.append(render_template("skill-always-load.md", entries=always_entries))
    return "\n\n".join(sections)


def _skill_metadata_json(*, name: str, description: str, path: str) -> str:
    serialized = json.dumps(
        {"name": name, "description": description, "path": path},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.translate(_MARKDOWN_SAFE_JSON_TRANSLATION)


def _skill_always_json(*, name: str, body: str) -> str:
    serialized = json.dumps(
        {"name": name, "body": body},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.translate(_MARKDOWN_SAFE_JSON_TRANSLATION)


def _skill_manual_json(invocation: ManualSkillInvocation) -> str:
    serialized = json.dumps(
        {"name": invocation.metadata.name, "body": invocation.body},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.translate(_MARKDOWN_SAFE_JSON_TRANSLATION)


def _skill_request_json(request: str) -> str:
    return json.dumps(
        request, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).translate(_MARKDOWN_SAFE_JSON_TRANSLATION)


def session_title_prompt() -> str:
    """Return the isolated prompt used for Session title generation."""
    return render_template("session-title-prompt.md")


def blackboard_prompt(
    *,
    user_input: str,
    last_task: str,
    latest_assistant_content: str,
) -> str:
    """Return the isolated prompt used for Task Framing."""
    return render_template(
        "blackboard-system-prompt.md",
        **{
            "User input": user_input,
            "Last Task": last_task,
            "Latest assistant content": latest_assistant_content,
        },
    )


def interrupted_assistant_content(content: str) -> str:
    """Mark interrupted assistant output when projecting persisted history."""
    return render_template("interrupted-assistant-content.md", content=content)
