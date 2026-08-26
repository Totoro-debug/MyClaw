"""Render version-tracked prompts shared by runtime orchestration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import PurePath

from myclaw.memory.records import SummaryEntry
from myclaw.skills.catalog import ManualSkillInvocation, RuntimeSkillSnapshot
from myclaw.templates import render_template
from myclaw.utils.time import format_rfc3339_milliseconds

_SKILL_METADATA_TRANSLATION: dict[int, str] = {
    ord("&"): r"\u0026",
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
        rendered = render_template(
            "current-user-input.md",
            runtime_context=runtime_context(current_time=current_time, session_id=session_id),
            user_input=render_template("user-input.md", content=content),
        )
    else:
        rendered = (
            f"{runtime_context(current_time=current_time, session_id=session_id)}\n\n"
            "<skill_instructions>\n"
            f"{_skill_manual_json(manual_invocation)}\n"
            "</skill_instructions>\n\n"
            "<user_request>\n"
            f"{_skill_request_json(manual_invocation.request)}\n"
            "</user_request>"
        )
    if blackboard_projection is None:
        return rendered
    serialized = json.dumps(
        blackboard_projection,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"{rendered}\n\n<blackboard>\n{serialized}\n</blackboard>"


def runtime_context(*, current_time: datetime, session_id: str) -> str:
    """Render the per-turn metadata included in the next model request."""
    return render_template(
        "runtime-context.md",
        current_time=format_rfc3339_milliseconds(current_time),
        session_id=session_id,
    )


def chat_system_prompt(*, workspace: PurePath, long_term_memory: str) -> str:
    """Compose the fixed Tool-guidance System Prompt."""
    return render_template(
        "foreground-chat-system-prompt.md",
        identity=render_template("builtin-identity.md", workspace=workspace),
        long_term_memory=long_term_memory,
    )


def foreground_chat_system_prompt(
    *,
    workspace: PurePath,
    long_term_memory: str,
    skill_snapshot: RuntimeSkillSnapshot | None = None,
) -> str:
    """Compose the foreground prompt with optional Skill metadata and Blackboard guidance."""
    sections = [chat_system_prompt(workspace=workspace, long_term_memory=long_term_memory)]
    if skill_snapshot is not None and skill_snapshot.catalog.entries:
        entries = "\n".join(
            _skill_metadata_json(
                name=metadata.name,
                description=metadata.description,
                path=str(metadata.path),
            )
            for metadata in skill_snapshot.catalog.entries
        )
        sections.append(render_template("skill-catalog.md", entries=entries))
        always_entries = "\n".join(
            _skill_always_json(name=skill.metadata.name, body=skill.body)
            for skill in skill_snapshot.always_loaded
        )
        if always_entries:
            sections.append(render_template("skill-always-load.md", entries=always_entries))
    sections.append(render_template("blackboard-guidance.md"))
    return "\n\n".join(sections)


def _skill_metadata_json(*, name: str, description: str, path: str) -> str:
    serialized = json.dumps(
        {"name": name, "description": description, "path": path},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.translate(_SKILL_METADATA_TRANSLATION)


def _skill_always_json(*, name: str, body: str) -> str:
    serialized = json.dumps(
        {"name": name, "body": body},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.translate(_SKILL_METADATA_TRANSLATION)


def _skill_manual_json(invocation: ManualSkillInvocation) -> str:
    serialized = json.dumps(
        {"name": invocation.metadata.name, "body": invocation.body},
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return serialized.translate(_SKILL_METADATA_TRANSLATION)


def _skill_request_json(request: str) -> str:
    return json.dumps(
        request, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).translate(_SKILL_METADATA_TRANSLATION)


def session_title_prompt() -> str:
    """Return the isolated prompt used for Session title generation."""
    return render_template("session-title-prompt.md")


def blackboard_prompt() -> str:
    """Return the isolated prompt used for Task Framing."""
    return render_template("blackboard-system-prompt.md")


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
