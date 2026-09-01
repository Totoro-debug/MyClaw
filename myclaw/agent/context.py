"""Build provider-neutral context for foreground Conversation Sessions."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from myclaw.agent.blackboard import Blackboard
from myclaw.agent.prompts import (
    chat_system_prompt,
    current_user_input,
    foreground_chat_system_prompt,
)
from myclaw.agent.prompts import (
    session_title_prompt as render_session_title_prompt,
)
from myclaw.memory.manager import MemoryManager
from myclaw.session.projection import project_session_message
from myclaw.skills.catalog import ManualSkillInvocation, SkillLoader


class ContextBuilder:
    """Build the complete model-visible message list for one foreground turn."""

    def __init__(
        self,
        workspace: Path,
        timezone_name: str,
        *,
        agent_home: Path,
        memory_manager: MemoryManager,
        skill_loader: SkillLoader,
    ) -> None:
        if not isinstance(workspace, Path):
            raise TypeError("Context Builder requires a Path")
        if not isinstance(agent_home, Path):
            raise TypeError("Context Builder requires an Agent Home Path")
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("Context Builder requires a Memory Manager")
        if not isinstance(skill_loader, SkillLoader):
            raise TypeError("Context Builder requires a Skill Loader")
        self._workspace = workspace
        self._agent_home = agent_home
        self._timezone = ZoneInfo(timezone_name)
        self._memory_manager = memory_manager
        self._skill_loader = skill_loader

    def foreground_system_prompt(self) -> str:
        """Build the current foreground System Prompt from owned runtime state."""
        return foreground_chat_system_prompt(
            workspace=self._workspace,
            agent_home=self._agent_home,
            long_term_memory=self._memory_manager.memory_snapshot(),
            skill_loader=self._skill_loader,
        )

    def schedule_system_prompt(self) -> str:
        """Build the Schedule System Prompt without foreground Skill content."""
        return chat_system_prompt(
            workspace=self._workspace,
            agent_home=self._agent_home,
            long_term_memory=self._memory_manager.memory_snapshot(),
        )

    def session_title_prompt(self) -> str:
        """Return the isolated versioned prompt used for Session titles."""
        return render_session_title_prompt()

    def build_title_messages(self, content: str) -> list[dict[str, Any]]:
        """Build the minimal System/User request used for a Session title."""
        if not isinstance(content, str):
            raise TypeError("Context Builder title content must be a string")
        return [
            {"role": "system", "content": self.session_title_prompt()},
            {"role": "user", "content": deepcopy(content)},
        ]

    def build_status_messages(
        self,
        history: Sequence[dict[str, Any]],
        *,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Build the minimum foreground request used by status and preflight."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.foreground_system_prompt()}
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
                    content="",
                    current_time=datetime.now(self._timezone),
                    session_id=session_id,
                ),
            }
        )
        return messages

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
        if blackboard is not None and not isinstance(blackboard, Blackboard):
            raise TypeError("value must be a Blackboard or None")
        blackboard_projection = None if blackboard is None else blackboard.to_dict()
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": foreground_chat_system_prompt(
                    workspace=self._workspace,
                    agent_home=self._agent_home,
                    long_term_memory=long_term_memory,
                    skill_loader=self._skill_loader,
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
                    current_time=datetime.now(self._timezone),
                    session_id=session_id,
                    blackboard_projection=blackboard_projection,
                    manual_invocation=manual_invocation,
                ),
            }
        )
        return messages


__all__ = ["ContextBuilder"]
