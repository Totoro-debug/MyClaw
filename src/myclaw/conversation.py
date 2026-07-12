"""Conversation Port implementation for the first successful streaming turn."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from myclaw.contracts import (
    AgentEvent,
    AgentEventPayload,
    AgentEventType,
    AssistantSessionMessage,
    ModelCompleted,
    ModelProvider,
    ModelRequest,
    ReasoningEffort,
    SessionStore,
    TextDelta,
    TextDeltaPayload,
    TurnCompletedPayload,
    TurnStartedPayload,
    UserModelMessage,
    UserSessionMessage,
)


@dataclass(frozen=True, slots=True)
class ChatModelSettings:
    """Resolved provider-neutral fields needed for one chat request."""

    model: str
    max_output: int
    temperature: float
    reasoning_effort: ReasoningEffort | None
    timeout_seconds: int


class StreamingConversationPort:
    """Translate one successful provider stream into Agent Events and Session records."""

    def __init__(
        self,
        *,
        provider: ModelProvider,
        sessions: SessionStore,
        session_id: str,
        settings: ChatModelSettings,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._session_id = session_id
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._next_event_id = 0

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if not text.strip():
            return

        turn_id = self._new_uuid()
        yield self._event(turn_id, "turn_started", TurnStartedPayload())

        user_message = UserSessionMessage(
            id=str(self._new_uuid()),
            created_at=self._persisted_now(),
            content=text,
        )
        await self._sessions.append_message(self._session_id, user_message)
        request = ModelRequest(
            request_id=self._new_uuid(),
            route="chat",
            system_prompt="",
            messages=(UserModelMessage(content=text),),
            tools=(),
            stream=True,
            model=self._settings.model,
            max_output=self._settings.max_output,
            temperature=self._settings.temperature,
            reasoning_effort=self._settings.reasoning_effort,
            timeout_seconds=self._settings.timeout_seconds,
        )

        async for model_event in self._provider.stream(request):
            if isinstance(model_event, TextDelta):
                yield self._event(
                    turn_id,
                    "text_delta",
                    TextDeltaPayload(delta=model_event.delta),
                )
                continue
            if isinstance(model_event, ModelCompleted):
                response = model_event.response
                assistant_message = AssistantSessionMessage(
                    id=str(self._new_uuid()),
                    created_at=self._persisted_now(),
                    content=response.message.content,
                    tool_calls=response.message.tool_calls,
                    status="completed",
                    error=None,
                    usage=response.usage,
                )
                await self._sessions.append_message(self._session_id, assistant_message)
                yield self._event(
                    turn_id,
                    "turn_completed",
                    TurnCompletedPayload(
                        content=response.message.content,
                        usage=response.usage,
                    ),
                )
                return
            raise TypeError("Unsupported Model Provider stream event")

        raise ValueError("Model Provider stream ended without completion")

    async def resolve_permission(self, request_id: UUID, approved: bool) -> None:
        raise NotImplementedError

    async def cancel_active_turn(self) -> None:
        raise NotImplementedError

    def _event(
        self,
        turn_id: UUID,
        event_type: AgentEventType,
        payload: AgentEventPayload,
    ) -> AgentEvent:
        event = AgentEvent(
            type=event_type,
            event_id=self._next_event_id,
            turn_id=turn_id,
            created_at=self._now(),
            payload=payload,
        )
        self._next_event_id += 1
        return event

    def _persisted_now(self) -> datetime:
        value = self._now()
        return value.replace(microsecond=value.microsecond // 1000 * 1000)
