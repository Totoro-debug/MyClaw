"""User-facing Conversation boundary."""

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from myclaw.agent.events import AgentEvent


@runtime_checkable
class ConversationPort(Protocol):
    def submit(self, text: str) -> AsyncIterator[AgentEvent]: ...

    async def cancel_active_turn(self) -> None: ...
