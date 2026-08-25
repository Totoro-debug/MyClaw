"""Build provider-neutral context for foreground Conversation Sessions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from myclaw.agent.blackboard import Blackboard, encode_blackboard
from myclaw.agent.prompts import current_user_input, foreground_chat_system_prompt
from myclaw.agent.workspace import Workspace
from myclaw.session.projection import project_session_message
from myclaw.skills.catalog import SkillCatalog


class ContextBuilder:
    """Build the complete model-visible message list for one foreground turn."""

    def __init__(
        self,
        workspace: Workspace,
        timezone_name: str,
        *,
        clock: Callable[[], datetime] | None = None,
        skill_catalog: SkillCatalog | None = None,
    ) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("Context Builder requires a Workspace")
        if skill_catalog is not None and not isinstance(skill_catalog, SkillCatalog):
            raise TypeError("Context Builder requires a Skill Catalog")
        self._workspace = workspace
        self._timezone = ZoneInfo(timezone_name)
        self._clock = clock
        self._skill_catalog = skill_catalog

    def set_clock(self, clock: Callable[[], datetime] | None) -> None:
        """Override the clock used while composing Runtime Context in tests."""
        self._clock = clock

    def build_messages(
        self,
        history: Sequence[dict[str, Any]],
        current_user: dict[str, Any],
        session_id: str,
        long_term_memory: str,
        *,
        blackboard: Blackboard | None = None,
    ) -> list[dict[str, Any]]:
        """Build system-first context without mutating any caller-owned message."""
        blackboard_projection = encode_blackboard(blackboard)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": foreground_chat_system_prompt(
                    workspace=self._workspace.path,
                    long_term_memory=long_term_memory,
                    skill_catalog=self._skill_catalog,
                ),
            }
        ]
        messages.extend(
            projected
            for message in history
            if (projected := project_session_message(message)) is not None
        )
        messages.append(
            {
                "role": "user",
                "content": current_user_input(
                    content=deepcopy(current_user["content"]),
                    current_time=self._current_time(),
                    session_id=session_id,
                    blackboard_projection=blackboard_projection,
                ),
            }
        )
        return messages

    def _current_time(self) -> datetime:
        if self._clock is None:
            return datetime.now(self._timezone)
        current_time = self._clock()
        if current_time.tzinfo is None or current_time.utcoffset() is None:
            raise ValueError("Context Builder clock must return an aware datetime")
        return current_time.astimezone(self._timezone)


__all__ = ["ContextBuilder"]
