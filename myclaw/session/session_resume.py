"""Runtime-owned Conversation Session selection without changing Port callers."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Protocol
from uuid import UUID

from myclaw.contracts import AgentEvent, ConversationPort


class AgentEventSequencer(Protocol):
    def sequence_foreground(self, event: AgentEvent) -> AgentEvent: ...


class SwitchableConversationPort:
    """Delegate Conversation Port calls to the currently selected Session."""

    def __init__(
        self,
        *,
        session_id: str,
        build_conversation: Callable[[str], ConversationPort],
        event_sequencer: AgentEventSequencer | None = None,
    ) -> None:
        self._session_id = session_id
        self._build_conversation = build_conversation
        self._delegate: ConversationPort | None = None
        self._active_delegate: ConversationPort | None = None
        self._owned_delegates: list[ConversationPort] = []
        self._event_sequencer = event_sequencer
        self._close_task: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    def switch_session(self, session_id: str) -> None:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if self._active_delegate is not None:
            raise RuntimeError("Cannot switch Session during an active foreground turn")
        if session_id == self._session_id:
            return
        self._session_id = session_id
        self._delegate = None

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if self._active_delegate is not None:
            raise RuntimeError("A foreground turn is already active")
        delegate = self._delegate
        if delegate is None:
            delegate = self._build_conversation(self._session_id)
            self._delegate = delegate
            self._owned_delegates.append(delegate)
        self._active_delegate = delegate
        try:
            async for event in delegate.submit(text):
                sequencer = self._event_sequencer
                yield event if sequencer is None else sequencer.sequence_foreground(event)
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

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_owned_delegates())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_owned_delegates(self) -> None:
        results = await asyncio.gather(
            *(self._close_delegate(delegate) for delegate in self._owned_delegates),
            return_exceptions=True,
        )
        failures = [result for result in results if isinstance(result, BaseException)]
        if len(failures) == 1:
            raise failures[0]
        if failures:
            raise BaseExceptionGroup("Conversation Port shutdown failed", failures)

    @staticmethod
    async def _close_delegate(delegate: ConversationPort) -> None:
        close = getattr(delegate, "close", None)
        if close is None:
            await delegate.cancel_active_turn()
        else:
            await close()
