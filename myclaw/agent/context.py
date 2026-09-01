"""Build provider-neutral context for foreground Conversation Sessions."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from myclaw.agent.blackboard import Blackboard, encode_blackboard
from myclaw.agent.prompts import current_user_input, foreground_chat_system_prompt
from myclaw.session.projection import project_session_message
from myclaw.skills.catalog import ManualSkillInvocation, SkillSnapshot


class ContextBuilder:
    """Build the complete model-visible message list for one foreground turn."""

    def __init__(
        self,
        workspace: Path,
        timezone_name: str,
        *,
        agent_home: Path,
        clock: Callable[[], datetime] | None = None,
        skill_snapshot: SkillSnapshot | None = None,
    ) -> None:
        if not isinstance(workspace, Path):
            raise TypeError("Context Builder requires a Path")
        if not isinstance(agent_home, Path):
            raise TypeError("Context Builder requires an Agent Home Path")
        if skill_snapshot is not None and not isinstance(skill_snapshot, SkillSnapshot):
            raise TypeError("Context Builder requires a Runtime Skill snapshot")
        self._workspace = workspace
        self._agent_home = agent_home
        self._timezone = ZoneInfo(timezone_name)
        self._clock = clock
        self._skill_snapshot = skill_snapshot

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
        manual_invocation: ManualSkillInvocation | None = None,
    ) -> list[dict[str, Any]]:
        """Build system-first context without mutating any caller-owned message."""
        if manual_invocation is not None and not isinstance(
            manual_invocation, ManualSkillInvocation
        ):
            raise TypeError("Context Builder requires a Manual Skill Invocation")
        blackboard_projection = encode_blackboard(blackboard)
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": foreground_chat_system_prompt(
                    workspace=self._workspace,
                    agent_home=self._agent_home,
                    long_term_memory=long_term_memory,
                    skill_snapshot=self._skill_snapshot,
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
                    manual_invocation=manual_invocation,
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
