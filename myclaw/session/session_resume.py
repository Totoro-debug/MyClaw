"""Runtime-owned Conversation Session selection without changing Port callers."""

import asyncio
from collections.abc import AsyncIterator, Callable
from typing import Any, Protocol

from myclaw.agent.events import AgentEvent, ConversationPort
from myclaw.session.session import Session


class AgentEventSequencer(Protocol):
    def sequence_foreground(self, event: AgentEvent) -> AgentEvent: ...


class SwitchableConversationPort:
    """Delegate Conversation Port calls to the currently selected Session."""

    def __init__(
        self,
        *,
        session: Session | None = None,
        session_id: str | None = None,
        build_conversation: Callable[[Any], ConversationPort],
        event_sequencer: AgentEventSequencer | None = None,
    ) -> None:
        if session is None and session_id is None:
            raise TypeError("Conversation Port requires a Session or Session ID")
        if session is not None and session_id is not None:
            raise TypeError("Conversation Port cannot receive both a Session and Session ID")
        self._session = session
        self._session_id = session.session_id if session is not None else session_id
        self._build_conversation = build_conversation
        self._delegate: ConversationPort | None = None
        self._active_delegate: ConversationPort | None = None
        self._owned_delegates: list[ConversationPort] = []
        self._previous_sessions: list[Session] = []
        self._event_sequencer = event_sequencer
        self._close_task: asyncio.Task[None] | None = None

    @property
    def session_id(self) -> str:
        session_id = self._session_id
        assert session_id is not None
        return session_id

    @property
    def session(self) -> Session:
        session = self._session
        if session is None:
            raise RuntimeError("This Conversation Port has no Session authority")
        return session

    def switch_session(self, session: Session | str) -> None:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if self._active_delegate is not None:
            raise RuntimeError("Cannot switch Session during an active foreground turn")
        session_id = session.session_id if isinstance(session, Session) else session
        if session_id == self.session_id:
            return
        if self._session is not None:
            self._previous_sessions.append(self._session)
        self._session = session if isinstance(session, Session) else None
        self._session_id = session_id
        self._delegate = None

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        if self._close_task is not None:
            raise RuntimeError("Conversation Port is closed")
        if self._active_delegate is not None:
            raise RuntimeError("A foreground turn is already active")
        delegate = self._delegate
        if delegate is None:
            delegate = self._build_conversation(
                self._session if self._session is not None else self.session_id
            )
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
