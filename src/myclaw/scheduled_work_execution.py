"""Execute one Scheduled Work trigger in its task-specific Session."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

from myclaw.contracts import (
    AssistantSessionMessage,
    ErrorInfo,
    ModelCallError,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelUsage,
    ReasoningEffort,
    ScheduledWork,
    SessionError,
    SessionMetadata,
    SessionStore,
    ToolResult,
    ToolSessionMessage,
    UserModelMessage,
    UserSessionMessage,
)
from myclaw.conversation import model_message_from_session
from myclaw.prompts import chat_system_prompt, current_user_input
from myclaw.session_titles import normalize_session_title
from myclaw.tool_artifacts import ArtifactDiscardError
from myclaw.tool_gateway import ToolGateway
from myclaw.workspace import Workspace

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
    """Run one complete non-streaming cron turn for a Scheduled Work record."""

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
        """Persist one task prompt and its final assistant response."""
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
        created_at = self._persisted_now()
        user = UserSessionMessage(
            id=str(self._new_uuid()),
            created_at=created_at,
            content=task.prompt,
        )
        try:
            await self._sessions.append_message(task.session_id, user)
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
        while True:
            try:
                session = await self._sessions.load(task.session_id)
            except (OSError, UnicodeError, ValueError):
                return ScheduledWorkRunResult(
                    status="failed",
                    content="",
                    error=_SESSION_FAILURE,
                )
            messages: list[ModelMessage] = []
            for message in session.short_term_messages:
                if isinstance(message, UserSessionMessage) and message.id == user.id:
                    messages.append(
                        UserModelMessage(
                            content=current_user_input(
                                content=user.content,
                                current_time=user.created_at,
                                session_id=task.session_id,
                            )
                        )
                    )
                    continue
                projected = model_message_from_session(message)
                if projected is not None:
                    messages.append(projected)
            try:
                response = await self._provider.complete(
                    ModelRequest(
                        request_id=self._new_uuid(),
                        route="cron",
                        system_prompt=system_prompt,
                        messages=tuple(messages),
                        tools=gateway.definitions,
                        stream=False,
                        model=self._settings.model,
                        max_output=self._settings.max_output,
                        temperature=self._settings.temperature,
                        reasoning_effort=self._settings.reasoning_effort,
                        timeout_seconds=self._settings.timeout_seconds,
                    )
                )
            except ModelCallError as failure:
                model_error = AssistantSessionMessage(
                    id=str(self._new_uuid()),
                    created_at=self._persisted_now(),
                    content="",
                    tool_calls=(),
                    status="error",
                    error=SessionError(
                        code=failure.error.code,
                        message=failure.error.message,
                    ),
                    usage=ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0),
                )
                try:
                    await self._sessions.append_message(task.session_id, model_error)
                except (OSError, UnicodeError, ValueError):
                    return ScheduledWorkRunResult(
                        status="failed",
                        content="",
                        error=_SESSION_FAILURE,
                    )
                return ScheduledWorkRunResult(
                    status="failed",
                    content="",
                    error=failure.error,
                )
            assistant = AssistantSessionMessage(
                id=str(self._new_uuid()),
                created_at=self._persisted_now(),
                content=response.message.content,
                tool_calls=response.message.tool_calls,
                status="completed",
                error=None,
                usage=response.usage,
            )
            try:
                await self._sessions.append_message(task.session_id, assistant)
            except (OSError, UnicodeError, ValueError):
                return ScheduledWorkRunResult(
                    status="failed",
                    content="",
                    error=_SESSION_FAILURE,
                )
            if not response.message.tool_calls:
                return ScheduledWorkRunResult(
                    status="completed",
                    content=response.message.content,
                    error=None,
                )
            for index, tool_call in enumerate(response.message.tool_calls):
                try:
                    result = await gateway.execute(tool_call)
                except asyncio.CancelledError:
                    for unfinished in response.message.tool_calls[index:]:
                        try:
                            await self._sessions.append_message(
                                task.session_id,
                                ToolSessionMessage(
                                    id=str(self._new_uuid()),
                                    created_at=self._persisted_now(),
                                    tool_call_id=unfinished.id,
                                    name=unfinished.name,
                                    content="Scheduled Work tool call cancelled.",
                                    status="error",
                                    error=SessionError(
                                        code="turn_cancelled",
                                        message="Scheduled Work tool call cancelled.",
                                    ),
                                    artifact=None,
                                ),
                            )
                        except (OSError, UnicodeError, ValueError):
                            pass
                    raise
                tool_message = ToolSessionMessage(
                    id=str(self._new_uuid()),
                    created_at=self._persisted_now(),
                    tool_call_id=result.tool_call_id,
                    name=result.name,
                    content=result.content,
                    status=result.status,
                    error=(
                        None
                        if result.error is None
                        else SessionError(
                            code=result.error.code,
                            message=result.error.message,
                        )
                    ),
                    artifact=result.artifact,
                )
                try:
                    await self._sessions.append_message(task.session_id, tool_message)
                except (OSError, UnicodeError, ValueError):
                    await self._reconcile_tool_artifact(
                        session_id=task.session_id,
                        gateway=gateway,
                        result=result,
                        tool_message=tool_message,
                    )
                    return ScheduledWorkRunResult(
                        status="failed",
                        content="",
                        error=_SESSION_FAILURE,
                    )
                except BaseException:
                    await self._reconcile_tool_artifact(
                        session_id=task.session_id,
                        gateway=gateway,
                        result=result,
                        tool_message=tool_message,
                    )
                    raise
                gateway.commit_artifact(result)

    async def _reconcile_tool_artifact(
        self,
        *,
        session_id: str,
        gateway: ToolGateway,
        result: ToolResult,
        tool_message: ToolSessionMessage,
    ) -> None:
        try:
            reloaded = await self._sessions.load(session_id)
        except BaseException:
            gateway.commit_artifact(result)
            return
        if tool_message in reloaded.messages:
            gateway.commit_artifact(result)
            return
        if any(message.id == tool_message.id for message in reloaded.messages):
            gateway.commit_artifact(result)
            return
        try:
            gateway.discard_artifact(result)
        except ArtifactDiscardError:
            pass

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)
