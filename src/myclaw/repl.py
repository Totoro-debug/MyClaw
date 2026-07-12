"""Injectable asynchronous command-line conversation loop."""

import asyncio
import sys
from collections.abc import AsyncIterator
from typing import Protocol

from rich.console import Console

from myclaw.contracts import (
    AgentEvent,
    BackgroundCompletedPayload,
    ConversationPort,
    PermissionRequestedPayload,
    SessionSummary,
    TextDeltaPayload,
    TurnFailedPayload,
)


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


class BackgroundEventSource(Protocol):
    async def next_background_event(self) -> AgentEvent: ...

    def next_background_event_nowait(self) -> AgentEvent | None: ...


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

    @property
    def resume_sessions(self) -> tuple[SessionSummary, ...] | None: ...


class ManagementDispatcher(Protocol):
    async def dispatch(self, command: str) -> ManagementDispatchResult: ...

    async def resume(self, session_id: str) -> ManagementDispatchResult: ...


async def run_repl(
    *,
    conversation: ConversationPort,
    input_reader: ReplInput,
    writer: ProgressiveWriter,
    management_dispatcher: ManagementDispatcher | None = None,
    background_events: BackgroundEventSource | None = None,
) -> None:
    """Read interactive input until EOF while preserving an unmaterialized empty Session."""
    input_task: asyncio.Task[str | None] | None = None
    background_task: asyncio.Task[AgentEvent] | None = None
    try:
        while True:
            if background_events is not None and input_task is None:
                queued_event = background_events.next_background_event_nowait()
                if queued_event is not None:
                    await _render_background_event(queued_event, writer)
                    continue
            if input_task is None:
                input_task = asyncio.create_task(input_reader.read())
            if background_events is None:
                text = await input_task
                input_task = None
            else:
                background_task = asyncio.create_task(background_events.next_background_event())
                completed, _ = await asyncio.wait(
                    (input_task, background_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if background_task in completed:
                    event = background_task.result()
                    background_task = None
                    await _render_background_event(event, writer)
                    continue
                background_task.cancel()
                await asyncio.gather(background_task, return_exceptions=True)
                background_task = None
                text = input_task.result()
                input_task = None
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
                    if result.resume_sessions:
                        await _choose_resume_session(
                            sessions=result.resume_sessions,
                            input_reader=input_reader,
                            writer=writer,
                            management_dispatcher=management_dispatcher,
                        )
                    continue
            events = conversation.submit(text)
            try:
                await _render_turn(
                    events,
                    writer,
                    conversation=conversation,
                    input_reader=input_reader,
                )
            except asyncio.CancelledError:
                active_task = asyncio.current_task()
                if active_task is not None and active_task.cancelling():
                    active_task.uncancel()
                await conversation.cancel_active_turn()
                await _render_turn(
                    events,
                    writer,
                    conversation=conversation,
                    input_reader=input_reader,
                )
    finally:
        pending = tuple(
            task for task in (input_task, background_task) if task is not None and not task.done()
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


async def _render_background_event(event: AgentEvent, writer: ProgressiveWriter) -> None:
    if event.type != "background_completed":
        raise TypeError("background event source yielded a non-background event")
    payload = event.payload
    if not isinstance(payload, BackgroundCompletedPayload):
        raise TypeError("background_completed event has an invalid payload")
    await writer.write_line(
        f"[Scheduled Work] {payload.title} ({payload.status}): {payload.summary}"
    )


async def _choose_resume_session(
    *,
    sessions: tuple[SessionSummary, ...],
    input_reader: ReplInput,
    writer: ProgressiveWriter,
    management_dispatcher: ManagementDispatcher,
) -> None:
    while True:
        selection = await input_reader.read()
        if selection is None:
            return
        normalized = selection.strip()
        if normalized.casefold() == "cancel":
            await writer.write_line("Resume cancelled.")
            return
        try:
            index = int(normalized)
        except ValueError:
            index = 0
        if not 1 <= index <= len(sessions):
            await writer.write_line("Choose a listed session number or enter cancel.")
            continue
        result = await management_dispatcher.resume(sessions[index - 1].id)
        if result.output is not None:
            await writer.write_line(result.output)
        return


async def _render_turn(
    events: AsyncIterator[AgentEvent],
    writer: ProgressiveWriter,
    *,
    conversation: ConversationPort,
    input_reader: ReplInput,
) -> None:
    async for event in events:
        if event.type == "text_delta":
            payload = event.payload
            if not isinstance(payload, TextDeltaPayload):
                raise TypeError("text_delta event has an invalid payload")
            await writer.write_delta(payload.delta)
        elif event.type == "permission_requested":
            payload = event.payload
            if not isinstance(payload, PermissionRequestedPayload):
                raise TypeError("permission_requested event has an invalid payload")
            await writer.write_line(
                f"Permission required: {payload.action} {payload.resource}\n"
                f"Risk: {payload.risk_summary}\n"
                "Approve? [y/N]"
            )
            response = await input_reader.read()
            approved = response is not None and response.strip().casefold() in {"y", "yes"}
            await conversation.resolve_permission(payload.request_id, approved)
        elif event.type in {"turn_completed", "turn_cancelled"}:
            await writer.finish_turn()
        elif event.type == "turn_failed":
            payload = event.payload
            if not isinstance(payload, TurnFailedPayload):
                raise TypeError("turn_failed event has an invalid payload")
            await writer.write_line(payload.error.message)
