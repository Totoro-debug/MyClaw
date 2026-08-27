import asyncio
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from loguru import logger

__all__ = [
    "InboundMessage",
    "MessageBus",
    "OutboundMessage",
    "OutboundMessageType",
]


@dataclass(slots=True)
class InboundMessage:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


OutboundMessageType = Literal[
    "model_reasoning",
    "model_response",
    "tool_call",
    "system_control",
]


@dataclass(slots=True)
class OutboundMessage:
    type: OutboundMessageType
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class MessageBus:
    def __init__(self) -> None:
        self._inbound: deque[InboundMessage] = deque()
        self._condition = asyncio.Condition()
        self._inbound_changed_callback: Callable[[tuple[InboundMessage, ...]], None] | None = None
        self._outbound: deque[OutboundMessage] = deque()

    async def inbound_snapshot(self) -> tuple[InboundMessage, ...]:
        async with self._condition:
            return tuple(self._inbound)

    async def put_inbound(self, message: InboundMessage) -> None:
        async with self._condition:
            self._inbound.append(message)
            snapshot = tuple(self._inbound)
            callback = self._inbound_changed_callback
            self._condition.notify_all()
        self._invoke_inbound_changed_callback(callback, snapshot)

    async def get_inbound(self) -> InboundMessage:
        async with self._condition:
            while not self._inbound:
                await self._condition.wait()
            message = self._inbound.popleft()
            snapshot = tuple(self._inbound)
            callback = self._inbound_changed_callback
        self._invoke_inbound_changed_callback(callback, snapshot)
        return message

    async def drain_inbound(self) -> tuple[InboundMessage, ...]:
        async with self._condition:
            messages = tuple(self._inbound)
            self._inbound.clear()
            snapshot = tuple(self._inbound)
            callback = self._inbound_changed_callback
        self._invoke_inbound_changed_callback(callback, snapshot)
        return messages

    async def put_outbound(self, message: OutboundMessage) -> None:
        async with self._condition:
            self._outbound.append(message)
            self._condition.notify_all()

    async def get_outbound(self) -> OutboundMessage:
        async with self._condition:
            while not self._outbound:
                await self._condition.wait()
            return self._outbound.popleft()

    async def reset(self) -> None:
        """Atomically discard pending Inbound and Outbound messages."""
        async with self._condition:
            self._inbound.clear()
            self._outbound.clear()
            callback = self._inbound_changed_callback
            self._condition.notify_all()
        self._invoke_inbound_changed_callback(callback, ())

    def set_inbound_changed_callback(
        self,
        callback: Callable[[tuple[InboundMessage, ...]], None] | None,
    ) -> None:
        self._inbound_changed_callback = callback

    def unbind_inbound_changed_callback(
        self,
        callback: Callable[[tuple[InboundMessage, ...]], None],
    ) -> None:
        """Clear only the callback still owned by the caller."""
        if self._inbound_changed_callback is callback:
            self._inbound_changed_callback = None

    @staticmethod
    def _invoke_inbound_changed_callback(
        callback: Callable[[tuple[InboundMessage, ...]], None] | None,
        snapshot: tuple[InboundMessage, ...],
    ) -> None:
        if callback is None:
            return
        try:
            callback(snapshot)
        except Exception as error:
            logger.opt(exception=error).error("Inbound changed callback failed")
