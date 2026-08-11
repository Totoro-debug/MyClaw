"""Injectable headless Conversation seam for Runtime regression coverage."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Protocol

from myclaw.agent.events import (
    AgentEvent,
    ConfirmationDecision,
    ConfirmationRequestedPayload,
    ConfirmationResponsePort,
    ConversationPort,
    TextDeltaPayload,
    TurnFailedPayload,
)
from myclaw.management.service import SessionListingEntry
from myclaw.terminal._turn_stream import (
    clear_current_task_cancellation,
    close_event_stream,
)


class ReplInput(Protocol):
    async def read(self) -> str | None: ...


class ProgressiveWriter(Protocol):
    async def write_delta(self, delta: str) -> None: ...

    async def finish_turn(self) -> None: ...

    async def write_line(self, content: str) -> None: ...


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
    shutdown_requested: asyncio.Event | None = None,
) -> None:
    """Read interactive input until EOF while preserving an unmaterialized empty Session."""
    while True:
        if shutdown_requested is not None and shutdown_requested.is_set():
            return
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
                clear_current_task_cancellation()
                await conversation.cancel_active_turn()
                await _render_turn(
                    events,
                    writer,
                    input_reader=input_reader,
                    confirmation=conversation,
                )
        except BaseException as primary_error:
            try:
                await close_event_stream(events)
            except BaseException as cleanup_error:
                raise primary_error from cleanup_error
            raise
        else:
            await close_event_stream(events)
            if shutdown_requested is None or not shutdown_requested.is_set():
                clear_current_task_cancellation()


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
