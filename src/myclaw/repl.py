"""Injectable asynchronous command-line conversation loop."""

import asyncio
import sys
from typing import Protocol

from rich.console import Console

from myclaw.contracts import ConversationPort, TextDeltaPayload


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
        if not text.strip():
            continue
        if management_dispatcher is not None:
            result = await management_dispatcher.dispatch(text)
            if result.handled:
                if result.output is not None:
                    await writer.write_line(result.output)
                continue
        async for event in conversation.submit(text):
            if event.type == "text_delta":
                payload = event.payload
                if not isinstance(payload, TextDeltaPayload):
                    raise TypeError("text_delta event has an invalid payload")
                await writer.write_delta(payload.delta)
            elif event.type == "turn_completed":
                await writer.finish_turn()
