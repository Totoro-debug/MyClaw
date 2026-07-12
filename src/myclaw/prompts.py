"""Version-tracked prompt fragments shared by runtime orchestration."""

from datetime import datetime
from pathlib import PurePath

from myclaw.contracts import format_rfc3339_milliseconds

_CURRENT_USER_INPUT = """<runtime_context>
current_time: {current_time}
session_id: {session_id}
</runtime_context>

<user_input>
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


def current_user_input(*, content: str, current_time: datetime, session_id: str) -> str:
    """Wrap only the current raw user input with dynamic Runtime Context."""
    return _CURRENT_USER_INPUT.format(
        current_time=format_rfc3339_milliseconds(current_time),
        session_id=session_id,
        content=content,
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
