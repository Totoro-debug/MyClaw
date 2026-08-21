"""Injectable headless host for the Runtime Message Bus foreground seams."""

from __future__ import annotations

import asyncio
import json
from typing import Protocol

from myclaw.agent.loop import AgentLoopControl, ConfirmationRequestView
from myclaw.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.management.service import SessionListingEntry


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
    bus: MessageBus,
    control: AgentLoopControl,
    input_reader: ReplInput,
    writer: ProgressiveWriter,
    management_dispatcher: ManagementDispatcher | None = None,
    shutdown_requested: asyncio.Event | None = None,
) -> None:
    """Read input and consume the active Runtime MessageBus foreground run."""
    confirmations: asyncio.Queue[ConfirmationRequestView] = asyncio.Queue()
    control.bind_confirmation_callback(confirmations.put_nowait)

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

        await bus.put_inbound(InboundMessage(content=text))
        try:
            await _render_run(
                bus=bus,
                control=control,
                confirmations=confirmations,
                input_reader=input_reader,
                writer=writer,
            )
        except asyncio.CancelledError:
            if shutdown_requested is not None and shutdown_requested.is_set():
                raise
            await control.cancel_active_run()
            _clear_current_task_cancellation()
            await _render_run(
                bus=bus,
                control=control,
                confirmations=confirmations,
                input_reader=input_reader,
                writer=writer,
            )


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


async def _render_run(
    *,
    bus: MessageBus,
    control: AgentLoopControl,
    confirmations: asyncio.Queue[ConfirmationRequestView],
    input_reader: ReplInput,
    writer: ProgressiveWriter,
) -> None:
    while True:
        outbound_task = asyncio.create_task(bus.get_outbound())
        confirmation_task = asyncio.create_task(confirmations.get())
        try:
            done, pending = await asyncio.wait(
                (outbound_task, confirmation_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

            if confirmation_task in done:
                await _respond_to_confirmation(
                    request=confirmation_task.result(),
                    input_reader=input_reader,
                    writer=writer,
                    control=control,
                )

            if outbound_task in done:
                outbound = outbound_task.result()
                if await _render_outbound(outbound, writer):
                    return
        finally:
            for task in (outbound_task, confirmation_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(outbound_task, confirmation_task, return_exceptions=True)


async def _render_outbound(outbound: OutboundMessage, writer: ProgressiveWriter) -> bool:
    if outbound.type in {"model_reasoning", "model_response"}:
        if outbound.metadata.get("_stream_delta") is True:
            await writer.write_delta(outbound.content)
        if outbound.metadata.get("_streamed") is True:
            await writer.finish_turn()
            return True
        return False
    if outbound.type == "tool_call":
        arguments = outbound.metadata.get("arguments", "")
        await writer.write_line(f"Tool: {outbound.content}\nArguments: {arguments}")
        return False
    if outbound.metadata.get("_streamed") is True:
        if outbound.metadata.get("finish_reason") != "cancelled":
            await writer.write_line(outbound.content)
        await writer.finish_turn()
        return True
    return False


async def _respond_to_confirmation(
    *,
    request: ConfirmationRequestView,
    input_reader: ReplInput,
    writer: ProgressiveWriter,
    control: AgentLoopControl,
) -> None:
    await writer.write_line(_format_confirmation_request(request))
    while True:
        response = await input_reader.read()
        if response is None:
            decision = "declined"
            break
        normalized = response.strip().casefold()
        if normalized in {"yes", "y"}:
            decision = "approved"
            break
        if normalized in {"", "no", "n"}:
            decision = "declined"
            break
        await writer.write_line("Invalid confirmation response.")
    control.respond_to_confirmation(request.confirmation_id, decision)  # type: ignore[arg-type]


def _format_confirmation_request(request: ConfirmationRequestView) -> str:
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


def _clear_current_task_cancellation() -> None:
    current = asyncio.current_task()
    if current is None:
        return
    uncancel = getattr(current, "uncancel", None)
    if callable(uncancel):
        while current.cancelling():
            uncancel()
