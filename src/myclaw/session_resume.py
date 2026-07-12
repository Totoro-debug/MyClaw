"""Runtime-owned Conversation Session selection without changing Port callers."""

from collections.abc import AsyncIterator, Callable
from uuid import UUID

from myclaw.contracts import AgentEvent, ConversationPort


class SwitchableConversationPort:
    """Delegate Conversation Port calls to the currently selected Session."""

    def __init__(
        self,
        *,
        session_id: str,
        build_conversation: Callable[[str], ConversationPort],
    ) -> None:
        self._session_id = session_id
        self._build_conversation = build_conversation
        self._delegate: ConversationPort | None = None
        self._active_delegate: ConversationPort | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def switch_session(self, session_id: str) -> None:
        if self._active_delegate is not None:
            raise RuntimeError("Cannot switch Session during an active foreground turn")
        if session_id == self._session_id:
            return
        self._session_id = session_id
        self._delegate = None

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._active_delegate is not None:
            raise RuntimeError("A foreground turn is already active")
        delegate = self._delegate
        if delegate is None:
            delegate = self._build_conversation(self._session_id)
            self._delegate = delegate
        self._active_delegate = delegate
        try:
            async for event in delegate.submit(text):
                yield event
        finally:
            if self._active_delegate is delegate:
                self._active_delegate = None

    async def resolve_permission(self, request_id: UUID, approved: bool) -> None:
        delegate = self._active_delegate
        if delegate is None:
            raise RuntimeError("No foreground turn is active")
        await delegate.resolve_permission(request_id, approved)

    async def cancel_active_turn(self) -> None:
        if self._active_delegate is not None:
            await self._active_delegate.cancel_active_turn()
