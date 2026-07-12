"""Injectable asynchronous command-line conversation loop."""

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Protocol

from rich.console import Console

from myclaw.contracts import AgentEvent, ConversationPort, TextDeltaPayload, TurnFailedPayload


class ReplInput(Protocol):
    async def read(self) -> str | None: ...


class ConsoleReplInput:
    """Read terminal input asynchronously and treat noninteractive streams as EOF."""

    def __init__(self, console: Console) -> None:
        self._console = console

    async def read(self) -> str | None:
        if not self._console.is_terminal or not sys.stdin.isatty():
            return None
        try:
            return await asyncio.to_thread(self._console.input, "You: ", markup=False)
        except EOFError:
            return None


class ProgressiveWriter(Protocol):
    async def write_delta(self, delta: str) -> None: ...

    async def finish_turn(self) -> None: ...

    async def write_line(self, content: str) -> None: ...


class ConsoleProgressiveWriter:
    """Render streamed text and complete lines through a Rich Console."""

    def __init__(self, console: Console) -> None:
        self._console = console

    async def write_delta(self, delta: str) -> None:
        self._console.print(delta, end="", markup=False, highlight=False, soft_wrap=True)

    async def finish_turn(self) -> None:
        self._console.print()

    async def write_line(self, content: str) -> None:
        self._console.print(
            content,
            end="" if content.endswith("\n") else "\n",
            markup=False,
            highlight=False,
            soft_wrap=True,
        )


class ManagementDispatchResult(Protocol):
    @property
    def handled(self) -> bool: ...

    @property
    def output(self) -> str | None: ...


class ManagementDispatcher(Protocol):
    async def dispatch(self, command: str) -> ManagementDispatchResult: ...


async def run_repl(
    *,
    conversation: ConversationPort,
    input_reader: ReplInput,
    writer: ProgressiveWriter,
    management_dispatcher: ManagementDispatcher | None = None,
) -> None:
    """Read interactive input until EOF while preserving an unmaterialized empty Session."""
    while True:
        text = await input_reader.read()
        if text is None:
            return
        stripped = text.strip()
        if not stripped:
            continue
        if stripped.casefold() in {"exit", "quit"}:
            return
        if management_dispatcher is not None:
            result = await management_dispatcher.dispatch(text)
            if result.handled:
                if result.output is not None:
                    await writer.write_line(result.output)
                continue
        events = conversation.submit(text)
        try:
            await _render_turn(events, writer)
        except asyncio.CancelledError:
            active_task = asyncio.current_task()
            if active_task is not None and active_task.cancelling():
                active_task.uncancel()
            await conversation.cancel_active_turn()
            await _render_turn(events, writer)


async def _render_turn(
    events: AsyncIterator[AgentEvent],
    writer: ProgressiveWriter,
) -> None:
    async for event in events:
        if event.type == "text_delta":
            payload = event.payload
            if not isinstance(payload, TextDeltaPayload):
                raise TypeError("text_delta event has an invalid payload")
            await writer.write_delta(payload.delta)
        elif event.type in {"turn_completed", "turn_cancelled"}:
            await writer.finish_turn()
        elif event.type == "turn_failed":
            payload = event.payload
            if not isinstance(payload, TurnFailedPayload):
                raise TypeError("turn_failed event has an invalid payload")
            await writer.write_line(payload.error.message)
