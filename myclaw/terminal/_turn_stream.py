"""Shared turn-stream cleanup for terminal conversation hosts."""

from asyncio import current_task
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from myclaw.agent.events import AgentEvent


@runtime_checkable
class _ClosableEventStream(Protocol):
    async def aclose(self) -> None: ...


async def close_event_stream(events: AsyncIterator[AgentEvent]) -> None:
    if isinstance(events, _ClosableEventStream):
        await events.aclose()


def clear_current_task_cancellation() -> None:
    task = current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()
