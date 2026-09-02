"""Build provider-neutral context for Agent Loop model requests."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass
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
from myclaw.session.projection import _last_user_index, project_session_message
from myclaw.skills.catalog import ManualSkillInvocation, SkillLoader


@dataclass(frozen=True, slots=True)
class _ScheduleProjectionSnapshot:
    builder_id: int
    system_prompt: str
    current_time: datetime


_SCHEDULE_PROJECTION_SNAPSHOT: ContextVar[_ScheduleProjectionSnapshot | None] = ContextVar(
    "schedule_projection_snapshot",
    default=None,
)


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
        return self.build_foreground_messages(
            [*history, {"role": "user", "content": ""}],
            session_id=session_id,
        )

    def build_foreground_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session_id: str,
        blackboard: Blackboard | None = None,
        manual_invocation: ManualSkillInvocation | None = None,
    ) -> list[dict[str, Any]]:
        """Build the canonical initial Model Request Context for a foreground turn."""
        if manual_invocation is not None and not isinstance(
            manual_invocation, ManualSkillInvocation
        ):
            raise TypeError("Context Builder requires a Manual Skill Invocation")
        if blackboard is not None and not isinstance(blackboard, Blackboard):
            raise TypeError("value must be a Blackboard or None")
        return self._build_messages(
            messages,
            system_prompt=self.foreground_system_prompt(),
            session_id=session_id,
            blackboard=blackboard,
            manual_invocation=manual_invocation,
        )

    def build_schedule_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        session_id: str,
    ) -> list[dict[str, Any]]:
        """Build the canonical initial Model Request Context for a Schedule Job."""
        snapshot = _SCHEDULE_PROJECTION_SNAPSHOT.get()
        if snapshot is None or snapshot.builder_id != id(self):
            system_prompt = self.schedule_system_prompt()
            current_time = None
        else:
            system_prompt = snapshot.system_prompt
            current_time = snapshot.current_time
        return self._build_messages(
            messages,
            system_prompt=system_prompt,
            session_id=session_id,
            current_time=current_time,
        )

    @contextmanager
    def schedule_projection_scope(self) -> Iterator[None]:
        """Keep one Schedule request's dynamic projection stable across budgeting."""
        token = _SCHEDULE_PROJECTION_SNAPSHOT.set(
            _ScheduleProjectionSnapshot(
                builder_id=id(self),
                system_prompt=self.schedule_system_prompt(),
                current_time=datetime.now(self._timezone),
            )
        )
        try:
            yield
        finally:
            _SCHEDULE_PROJECTION_SNAPSHOT.reset(token)

    def _build_messages(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        system_prompt: str,
        session_id: str,
        blackboard: Blackboard | None = None,
        manual_invocation: ManualSkillInvocation | None = None,
        current_time: datetime | None = None,
    ) -> list[dict[str, Any]]:
        current_user_index = _last_user_index(messages)
        projected = [{"role": "system", "content": system_prompt}]
        if current_user_index == len(messages):
            projected.extend(_project_history_messages(messages))
            return projected

        projected.extend(_project_history_messages(messages[:current_user_index]))
        projected.append(
            {
                "role": "user",
                "content": current_user_input(
                    content=deepcopy(messages[current_user_index]["content"]),
                    current_time=(
                        datetime.now(self._timezone) if current_time is None else current_time
                    ),
                    session_id=session_id,
                    blackboard_projection=(
                        None
                        if manual_invocation is not None or blackboard is None
                        else blackboard.to_dict()
                    ),
                    manual_invocation=manual_invocation,
                ),
            }
        )
        projected.extend(_project_history_messages(messages[current_user_index + 1 :]))
        return projected


def _project_history_messages(
    messages: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        projected
        for message in messages
        if (projected := project_session_message(message)) is not None
    ]


__all__ = ["ContextBuilder"]
