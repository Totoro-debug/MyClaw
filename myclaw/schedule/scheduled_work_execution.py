"""Scheduled Work adapter over the Runtime Core Agent turn."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from loguru import logger

from myclaw.agent.events import TurnCompletedPayload, TurnFailedPayload
from myclaw.agent.prompts import chat_system_prompt, render_tool_guidance
from myclaw.agent.turn import AgentTurn, ToolResultExternalizer
from myclaw.agent.workspace import Workspace
from myclaw.errors import ErrorInfo
from myclaw.provider.models import ModelProvider, ReasoningEffort
from myclaw.schedule.records import ScheduledWork
from myclaw.session.records import SessionMetadata
from myclaw.session.session_store import SessionStore
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
    diagnostic: BaseException | None = field(default=None, compare=False, repr=False)


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
        externalize_result_for: Callable[[str], ToolResultExternalizer] | None = None,
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._workspace = Workspace.from_path(workspace)
        self._long_term_memory = long_term_memory
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._tool_gateway_for = tool_gateway_for
        self._externalize_result_for = externalize_result_for

    async def run(self, task: ScheduledWork) -> ScheduledWorkRunResult:
        try:
            result = await self._run_once(task)
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            logger.opt(exception=error).error("Scheduled Work crashed")
            raise
        if result.status == "failed":
            code = result.error.code if result.error is not None else "unknown"
            if result.diagnostic is None:
                logger.error("Scheduled Work failed code={}", code)
            else:
                logger.opt(exception=result.diagnostic).error("Scheduled Work failed code={}", code)
        return result

    async def _run_once(self, task: ScheduledWork) -> ScheduledWorkRunResult:
        try:
            self._sessions.prepare_with_id(
                session_id=task.session_id,
                title=normalize_session_title(task.title),
                created_at=task.created_at,
            )
        except (OSError, UnicodeError, ValueError) as error:
            return ScheduledWorkRunResult(
                status="failed",
                content="",
                error=_SESSION_FAILURE,
                diagnostic=error,
            )

        gateway = self._tool_gateway_for(task.session_id)
        system_prompt = chat_system_prompt(
            workspace=self._workspace.path,
            long_term_memory=self._long_term_memory,
            tool_guidance=render_tool_guidance(gateway.schemas),
        )
        terminal_diagnostic: BaseException | None = None

        def capture_terminal_failure(error: BaseException) -> None:
            nonlocal terminal_diagnostic
            terminal_diagnostic = error

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
            externalize_result=(
                None
                if self._externalize_result_for is None
                else self._externalize_result_for(task.session_id)
            ),
            on_terminal_failure=capture_terminal_failure,
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
                    diagnostic=terminal_diagnostic,
                )
        raise AssertionError("Agent turn ended without a terminal payload")
