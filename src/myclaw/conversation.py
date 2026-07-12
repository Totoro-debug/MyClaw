"""Conversation Port implementation for the first successful streaming turn."""

from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from myclaw.contracts import (
    AgentEvent,
    AgentEventPayload,
    AgentEventType,
    AssistantModelMessage,
    AssistantSessionMessage,
    ModelCompleted,
    ModelProvider,
    ModelRequest,
    ReasoningEffort,
    SessionMessage,
    SessionStore,
    TextDelta,
    TextDeltaPayload,
    TurnCompletedPayload,
    TurnStartedPayload,
    UserModelMessage,
    UserSessionMessage,
)
from myclaw.prompts import current_user_input


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
        system_prompt: str = "",
    ) -> None:
        self._provider = provider
        self._sessions = sessions
        self._session_id = session_id
        self._settings = settings
        self._now = now
        self._new_uuid = new_uuid
        self._system_prompt = system_prompt
        self._next_event_id = 0
        self._foreground_active = False

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if not text.strip():
            return
        if self._foreground_active:
            raise RuntimeError("A foreground turn is already active")
        self._foreground_active = True
        try:
            async for event in self._submit_turn(text):
                yield event
        finally:
            self._foreground_active = False

    async def _submit_turn(self, text: str) -> AsyncIterator[AgentEvent]:
        turn_id = self._new_uuid()
        yield self._event(turn_id, "turn_started", TurnStartedPayload())

        user_created_at = self._persisted_now()
        user_message = UserSessionMessage(
            id=str(self._new_uuid()),
            created_at=user_created_at,
            content=text,
        )
        await self._sessions.append_message(self._session_id, user_message)
        session = await self._sessions.load(self._session_id)
        history = tuple(
            _session_message_for_model(message) for message in session.short_term_messages[:-1]
        )
        request = ModelRequest(
            request_id=self._new_uuid(),
            route="chat",
            system_prompt=self._system_prompt,
            messages=(
                *history,
                UserModelMessage(
                    content=current_user_input(
                        content=text,
                        current_time=user_created_at,
                        session_id=self._session_id,
                    )
                ),
            ),
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


def _session_message_for_model(
    message: SessionMessage,
) -> UserModelMessage | AssistantModelMessage:
    if isinstance(message, UserSessionMessage):
        return UserModelMessage(content=message.content)
    if isinstance(message, AssistantSessionMessage):
        return AssistantModelMessage(content=message.content, tool_calls=message.tool_calls)
    raise TypeError("Unsupported Short-term Memory message")
