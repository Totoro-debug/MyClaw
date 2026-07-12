"""Version-tracked prompt fragments shared by runtime orchestration."""

from datetime import datetime
from pathlib import PurePath

from myclaw.contracts import SummaryEntry, format_rfc3339_milliseconds

_RUNTIME_CONTEXT = """<runtime_context>
current_time: {current_time}
session_id: {session_id}
</runtime_context>"""

_USER_INPUT = """<user_input>
{content}
</user_input>"""

_BUILTIN_IDENTITY = """You are the MyClaw Personal Agent.
Act within the user's current Workspace.
Workspace: {workspace}"""

_CHAT_SYSTEM_PROMPT = """{identity}

<long_term_memory>
{long_term_memory}</long_term_memory>

<tool_guidance>
{tool_guidance}</tool_guidance>"""

_SESSION_TITLE_PROMPT = """Generate a concise title for this Conversation Session.
Return only the title. Do not call tools or add commentary."""

_MEMORY_TASK_PROMPT = """Maintain the MyClaw Long-term Memory from new Conversation Summaries.
Use read_file to inspect exactly {long_term_path}.
Use edit_file only when stable information should be retained, and edit exactly that file.
Keep the four sections: User Info, User Preference, Project Fact, and Lesson.
Do not store transient activity, raw summaries, or duplicate facts.
If no durable update is needed, do not call edit_file."""


def current_user_input(*, content: str, current_time: datetime, session_id: str) -> str:
    """Wrap only the current raw user input with dynamic Runtime Context."""
    return f"{runtime_context(current_time=current_time, session_id=session_id)}\n\n" + (
        _USER_INPUT.format(content=content)
    )


def runtime_context(*, current_time: datetime, session_id: str) -> str:
    """Render the per-turn metadata included in the next model request."""
    return _RUNTIME_CONTEXT.format(
        current_time=format_rfc3339_milliseconds(current_time),
        session_id=session_id,
    )


def chat_system_prompt(
    *, workspace: PurePath, long_term_memory: str, tool_guidance: str = ""
) -> str:
    """Compose chat system context in the accepted fixed order."""
    return _CHAT_SYSTEM_PROMPT.format(
        identity=_BUILTIN_IDENTITY.format(workspace=workspace),
        long_term_memory=long_term_memory,
        tool_guidance=tool_guidance,
    )


def session_title_prompt() -> str:
    """Return the isolated prompt used for Session title generation."""
    return _SESSION_TITLE_PROMPT


def memory_task_prompt(*, long_term_path: PurePath) -> str:
    """Return the restricted four-section Long-term Memory maintenance prompt."""
    return _MEMORY_TASK_PROMPT.format(long_term_path=long_term_path)


def memory_task_input(*, cursor: int, summaries: tuple[SummaryEntry, ...]) -> str:
    """Render only the pending ordered Conversation Summary batch."""
    records = "\n".join(entry.to_json_line().rstrip("\n") for entry in summaries)
    return (
        f"<summary_cursor>{cursor}</summary_cursor>\n"
        f"<conversation_summaries>\n{records}\n</conversation_summaries>"
    )
