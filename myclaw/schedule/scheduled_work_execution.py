"""Scheduled Work adapter over the Runtime Core Agent turn."""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from myclaw.agent.events import TurnCompletedPayload, TurnFailedPayload
from myclaw.agent.prompts import chat_system_prompt
from myclaw.agent.turn import AgentTurn
from myclaw.agent.workspace import Workspace
from myclaw.errors import ErrorInfo
from myclaw.provider.models import ReasoningEffort
from myclaw.provider.ports import ModelProvider
from myclaw.schedule.records import ScheduledWork
from myclaw.session.ports import SessionStore
from myclaw.session.records import SessionMetadata
from myclaw.session.session_titles import normalize_session_title
from myclaw.tools.tool_gateway import ToolGateway

_SESSION_FAILURE = ErrorInfo(
    code="persistence_error",
    message="Scheduled Work Session could not be updated.",
)


class ScheduledWorkSessionStore(SessionStore, Protocol):
    """Session operations needed when a task owns its identity before first trigger."""

    def prepare_with_id(
        self,
        *,
        session_id: str,
        title: str,
        created_at: datetime,
    ) -> SessionMetadata: ...


@dataclass(frozen=True, slots=True)
class ScheduledWorkModelSettings:
    """Resolved provider-neutral settings for the cron Model Route."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


@dataclass(frozen=True, slots=True)
class ScheduledWorkRunResult:
    """Safe terminal outcome consumed by background coordination."""

    status: Literal["completed", "failed"]
    content: str
    error: ErrorInfo | None


class ScheduledWorkRunner:
    """Adapt one Scheduled Work trigger to a non-streaming Runtime Core turn."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        sessions: ScheduledWorkSessionStore,
        workspace: Path,
        long_term_memory: str,
        settings: ScheduledWorkModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        tool_gateway_for: Callable[[str], ToolGateway],
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._workspace = Workspace.from_path(workspace)
        self._long_term_memory = long_term_memory
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._tool_gateway_for = tool_gateway_for

    async def run(self, task: ScheduledWork) -> ScheduledWorkRunResult:
        try:
            self._sessions.prepare_with_id(
                session_id=task.session_id,
                title=normalize_session_title(task.title),
                created_at=task.created_at,
            )
        except (OSError, UnicodeError, ValueError):
            return ScheduledWorkRunResult(
                status="failed",
                content="",
                error=_SESSION_FAILURE,
            )

        gateway = self._tool_gateway_for(task.session_id)
        system_prompt = chat_system_prompt(
            workspace=self._workspace.path,
            long_term_memory=self._long_term_memory,
            tool_guidance="\n".join(
                f"- {definition.name}: {definition.description}"
                for definition in gateway.definitions
            ),
        )
        turn = AgentTurn(
            lane="scheduled_work",
            provider=self._provider,
            sessions=self._sessions,
            session_id=task.session_id,
            settings=self._settings,
            now=self._now,
            new_uuid=self._new_uuid,
            system_prompt=system_prompt,
            tool_gateway=gateway,
        )
        async for payload in turn.run(task.prompt):
            if isinstance(payload, TurnCompletedPayload):
                return ScheduledWorkRunResult(
                    status="completed",
                    content=payload.content,
                    error=None,
                )
            if isinstance(payload, TurnFailedPayload):
                return ScheduledWorkRunResult(
                    status="failed",
                    content="",
                    error=payload.error,
                )
        raise AssertionError("Agent turn ended without a terminal payload")
