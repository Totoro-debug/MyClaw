"""Injectable asynchronous command-line conversation loop."""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from prompt_toolkit import PromptSession
from rich.console import Console

from myclaw.agent.events import (
    AgentEvent,
    BackgroundCompletedPayload,
    ConfirmationDecision,
    ConfirmationRequestedPayload,
    ConfirmationResponsePort,
    ConversationPort,
    TextDeltaPayload,
    TurnFailedPayload,
)
from myclaw.management.service import SessionListingEntry


class ReplInput(Protocol):
    async def read(self) -> str | None: ...


class PromptSessionBoundary(Protocol):
    async def prompt_async(self, message: str, *, handle_sigint: bool) -> str: ...


class ConsoleReplInput:
    """Read terminal input asynchronously and treat noninteractive streams as EOF."""

    def __init__(
        self,
        console: Console,
        *,
        prompt_session: PromptSessionBoundary | None = None,
    ) -> None:
        self._console = console
        self._prompt_session = prompt_session

    async def read(self) -> str | None:
        if not self._console.is_terminal or not sys.stdin.isatty():
            return None
        prompt_session = self._prompt_session
        if prompt_session is None:
            prompt_session = PromptSession()
            self._prompt_session = prompt_session
        try:
            return await prompt_session.prompt_async("You: ", handle_sigint=False)
        except EOFError:
            return None


class ProgressiveWriter(Protocol):
    async def write_delta(self, delta: str) -> None: ...

    async def finish_turn(self) -> None: ...

    async def write_line(self, content: str) -> None: ...


class BackgroundEventSource(Protocol):
    async def next_background_event(self) -> AgentEvent: ...

    def next_background_event_nowait(self) -> AgentEvent | None: ...


@runtime_checkable
class _ClosableEventStream(Protocol):
    async def aclose(self) -> None: ...


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
    def resume_sessions(self) -> tuple[SessionListingEntry, ...] | None: ...


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
    shutdown_requested: asyncio.Event | None = None,
) -> None:
    """Read interactive input until EOF while preserving an unmaterialized empty Session."""
    input_task: asyncio.Task[str | None] | None = None
    background_task: asyncio.Task[AgentEvent] | None = None
    try:
        while True:
            if shutdown_requested is not None and shutdown_requested.is_set():
                return
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
                if input_task in completed and not input_task.cancelled():
                    try:
                        completed_text = input_task.result()
                    except BaseException:
                        pass
                    else:
                        if completed_text is not None and completed_text.strip().casefold() in {
                            "exit",
                            "quit",
                        }:
                            input_task = None
                            if not background_task.done():
                                background_task.cancel()
                            await asyncio.gather(background_task, return_exceptions=True)
                            background_task = None
                            return
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
                try:
                    await _render_turn(
                        events,
                        writer,
                        input_reader=input_reader,
                        confirmation=conversation,
                    )
                except asyncio.CancelledError:
                    if shutdown_requested is not None and shutdown_requested.is_set():
                        raise
                    _clear_current_task_cancellation()
                    await conversation.cancel_active_turn()
                    await _render_turn(
                        events,
                        writer,
                        input_reader=input_reader,
                        confirmation=conversation,
                    )
            except BaseException as primary_error:
                try:
                    await _close_event_stream(events)
                except BaseException as cleanup_error:
                    raise primary_error from cleanup_error
                raise
            else:
                await _close_event_stream(events)
                if shutdown_requested is None or not shutdown_requested.is_set():
                    _clear_current_task_cancellation()
    finally:
        children = tuple(task for task in (input_task, background_task) if task is not None)
        for task in children:
            if not task.done():
                task.cancel()
        if children:
            await asyncio.gather(*children, return_exceptions=True)


async def _render_background_event(event: AgentEvent, writer: ProgressiveWriter) -> None:
    if event.type != "background_completed":
        raise TypeError("background event source yielded a non-background event")
    payload = event.payload
    if not isinstance(payload, BackgroundCompletedPayload):
        raise TypeError("background_completed event has an invalid payload")
    await writer.write_line(
        f"[Scheduled Work] {payload.title} ({payload.status}): {payload.summary}"
    )


async def _close_event_stream(events: AsyncIterator[AgentEvent]) -> None:
    if isinstance(events, _ClosableEventStream):
        await events.aclose()


def _clear_current_task_cancellation() -> None:
    task = asyncio.current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()


async def _choose_resume_session(
    *,
    sessions: tuple[SessionListingEntry, ...],
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
    input_reader: ReplInput,
    confirmation: ConfirmationResponsePort,
) -> None:
    async for event in events:
        if event.type == "text_delta":
            payload = event.payload
            if not isinstance(payload, TextDeltaPayload):
                raise TypeError("text_delta event has an invalid payload")
            await writer.write_delta(payload.delta)
        elif event.type == "confirmation_requested":
            payload = event.payload
            if not isinstance(payload, ConfirmationRequestedPayload):
                raise TypeError("confirmation_requested event has an invalid payload")
            await _respond_to_confirmation(
                request=payload,
                input_reader=input_reader,
                writer=writer,
                confirmation=confirmation,
            )
        elif event.type in {"turn_completed", "turn_cancelled"}:
            await writer.finish_turn()
        elif event.type == "turn_failed":
            payload = event.payload
            if not isinstance(payload, TurnFailedPayload):
                raise TypeError("turn_failed event has an invalid payload")
            await writer.write_line(payload.error.message)


async def _respond_to_confirmation(
    *,
    request: ConfirmationRequestedPayload,
    input_reader: ReplInput,
    writer: ProgressiveWriter,
    confirmation: ConfirmationResponsePort,
) -> None:
    await writer.write_line(_format_confirmation_request(request))
    while True:
        response = await input_reader.read()
        if response is None:
            decision: ConfirmationDecision = "declined"
            break
        normalized = response.strip().casefold()
        if normalized in {"yes", "y"}:
            decision = "approved"
            break
        if normalized in {"", "no", "n"}:
            decision = "declined"
            break
        await writer.write_line("Invalid confirmation response.")
    confirmation.respond_to_confirmation(request.confirmation_id, decision)


def _format_confirmation_request(request: ConfirmationRequestedPayload) -> str:
    details = json.dumps(
        request.details,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    lines = [
        f"Confirmation required: {request.summary}",
        f"Tool: {request.tool_name}",
        f"Details: {details}",
    ]
    if request.warnings:
        lines.append(f"Warnings: {'; '.join(request.warnings)}")
    lines.append("Confirm? [yes/y, no/n]:")
    return "\n".join(lines)
