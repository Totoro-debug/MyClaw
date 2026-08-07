"""Runtime-owned Conversation Session selection without changing Port callers."""

import asyncio
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from uuid import UUID

from myclaw.agent.events import AgentEvent, ConfirmationDecision, ConversationPort
from myclaw.session.session import Session


class SwitchableConversationPort:
    """Delegate Conversation Port calls to the currently selected Session."""

    def __init__(
        self,
        *,
        session: Session,
        build_conversation: Callable[[Session], ConversationPort],
    ) -> None:
        self._session = session
        self._build_conversation = build_conversation
        self._delegate: ConversationPort | None = None
        self._active_delegate: ConversationPort | None = None
        self._owned_delegates: list[ConversationPort] = []
        self._previous_sessions: list[Session] = []
        self._next_event_id = 0
        self._close_task: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def session(self) -> Session:
        session = self._session
        if session is None:
            raise RuntimeError("This Conversation Port has no Session authority")
        return session

    def switch_session(self, session: Session) -> None:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if self._active_delegate is not None:
            raise RuntimeError("Cannot switch Session during an active foreground turn")
        if session.session_id == self.session_id:
            return
        if self._session is not None:
            self._previous_sessions.append(self._session)
        self._session = session
        self._delegate = None

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if self._active_delegate is not None:
            raise RuntimeError("A foreground turn is already active")
        delegate = self._delegate
        if delegate is None:
            delegate = self._build_conversation(self.session)
            self._delegate = delegate
            self._owned_delegates.append(delegate)
        self._active_delegate = delegate
        try:
            async for event in delegate.submit(text):
                sequenced = replace(event, event_id=self._next_event_id)
                self._next_event_id += 1
                yield sequenced
        finally:
            if self._active_delegate is delegate:
                self._active_delegate = None

    async def cancel_active_turn(self) -> None:
        if self._active_delegate is not None:
            await self._active_delegate.cancel_active_turn()

    def respond_to_confirmation(self, confirmation_id: UUID, decision: ConfirmationDecision) -> None:
        delegate = self._active_delegate
        if delegate is None:
            raise ValueError("No foreground confirmation request is pending")
        delegate.respond_to_confirmation(confirmation_id, decision)

    async def close(self) -> None:
        task = self._close_task
        if task is None:
            task = asyncio.create_task(self._close_owned_delegates())
            self._close_task = task
        await asyncio.shield(task)

    async def _close_owned_delegates(self) -> None:
        try:
            results = await asyncio.gather(
                *(self._close_delegate(delegate) for delegate in self._owned_delegates),
                return_exceptions=True,
            )
            failures = [result for result in results if isinstance(result, BaseException)]
            if len(failures) == 1:
                raise failures[0]
            if failures:
                raise BaseExceptionGroup("Conversation Port shutdown failed", failures)
        finally:
            for session in self._previous_sessions:
                try:
                    session.close()
                except BaseException:
                    pass

    @staticmethod
    async def _close_delegate(delegate: ConversationPort) -> None:
        close = getattr(delegate, "close", None)
        if close is None:
            await delegate.cancel_active_turn()
        else:
            await close()
