from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass

import pytest

from myclaw.agent.message_bus import InboundMessage, MessageBus, OutboundMessage
from myclaw.management.service import SessionListingEntry
from myclaw.terminal.repl import run_repl


class _Input:
    def __init__(self, values: tuple[str | None, ...]) -> None:
        self._values = deque(values)

    async def read(self) -> str | None:
        return self._values.popleft()


class _Writer:
    def __init__(self) -> None:
        self.operations: list[tuple[str, str]] = []

    async def write_delta(self, delta: str) -> None:
        self.operations.append(("delta", delta))

    async def finish_turn(self) -> None:
        self.operations.append(("finish", ""))

    async def write_line(self, content: str) -> None:
        self.operations.append(("line", content))


class _Control:
    has_active_run = False
    has_pending_confirmation = False

    def __init__(self, bus: MessageBus) -> None:
        self._bus = bus
        self.cancel_calls = 0
        self.confirmation_callback: object | None = None

    def bind_confirmation_callback(self, callback: object) -> None:
        self.confirmation_callback = callback

    async def cancel_active_run(self) -> None:
        self.cancel_calls += 1
        await self._bus.put_outbound(
            OutboundMessage(
                "system_control",
                "MyClaw 已取消本轮对话。",
                {"finish_reason": "cancelled", "_streamed": True},
            )
        )

    def respond_to_confirmation(self, confirmation_id: object, decision: object) -> None:
        del confirmation_id, decision


@dataclass(frozen=True)
class _DispatchResult:
    handled: bool
    output: str | None = None
    resume_sessions: tuple[SessionListingEntry, ...] | None = None


class _Dispatcher:
    async def dispatch(self, command: str) -> _DispatchResult:
        if command.strip().casefold() == "/memory":
            return _DispatchResult(handled=True, output="memory output")
        return _DispatchResult(handled=False)

    async def resume(self, session_id: str) -> _DispatchResult:
        del session_id
        return _DispatchResult(handled=True, output="resumed")


async def _produce_for_inputs(
    bus: MessageBus,
    outputs: dict[str, tuple[OutboundMessage, ...]],
) -> list[InboundMessage]:
    observed: list[InboundMessage] = []
    try:
        while True:
            inbound = await bus.get_inbound()
            observed.append(inbound)
            for message in outputs.get(inbound.content, ()):
                await bus.put_outbound(message)
    except asyncio.CancelledError:
        return observed


@pytest.mark.asyncio
async def test_repl_ignores_blank_and_exit_input_without_creating_inbound_messages() -> None:
    bus = MessageBus()
    writer = _Writer()
    control = _Control(bus)

    await run_repl(
        bus=bus,
        control=control,
        input_reader=_Input(("  ", "\t", "quit")),
        writer=writer,
    )

    assert await bus.inbound_snapshot() == ()
    assert writer.operations == []


@pytest.mark.asyncio
async def test_repl_projects_sparse_segments_tool_arguments_and_one_terminal_marker() -> None:
    bus = MessageBus()
    await bus.put_outbound(OutboundMessage("model_response", "queued", {"_stream_delta": True}))
    await bus.put_outbound(OutboundMessage("model_response", "", {"_stream_end": True}))
    await bus.put_outbound(OutboundMessage("model_response", "", {"_streamed": True}))
    writer = _Writer()

    await run_repl(
        bus=bus,
        control=_Control(bus),
        input_reader=_Input(("Hello", "exit")),
        writer=writer,
    )

    assert writer.operations == [("delta", "queued"), ("finish", "")]
    inbound = await bus.inbound_snapshot()
    assert [(message.content, message.metadata) for message in inbound] == [("Hello", {})]


@pytest.mark.asyncio
async def test_repl_keeps_unknown_slash_input_on_the_inbound_bus_and_dispatches_management() -> (
    None
):
    bus = MessageBus()
    producer = asyncio.create_task(
        _produce_for_inputs(
            bus,
            {
                "/unknown": (
                    OutboundMessage("model_response", "ordinary", {"_stream_delta": True}),
                    OutboundMessage("model_response", "", {"_streamed": True}),
                )
            },
        )
    )
    writer = _Writer()

    await run_repl(
        bus=bus,
        control=_Control(bus),
        input_reader=_Input(("/memory", "/unknown", "exit")),
        writer=writer,
        management_dispatcher=_Dispatcher(),
    )
    producer.cancel()
    observed = await asyncio.gather(producer, return_exceptions=True)

    assert isinstance(observed[0], list)
    assert [(message.content, message.metadata) for message in observed[0]] == [("/unknown", {})]
    assert writer.operations == [
        ("line", "memory output"),
        ("delta", "ordinary"),
        ("finish", ""),
    ]


@pytest.mark.asyncio
async def test_repl_task_cancellation_requests_control_cancel_and_repairs_input_loop() -> None:
    class BlockingProducer:
        def __init__(self, bus: MessageBus) -> None:
            self.bus = bus
            self.delta_seen = asyncio.Event()

        async def run(self) -> None:
            inbound = await self.bus.get_inbound()
            assert inbound.content == "Start"
            await self.bus.put_outbound(
                OutboundMessage("model_response", "partial", {"_stream_delta": True})
            )
            self.delta_seen.set()
            await asyncio.Event().wait()

    bus = MessageBus()
    producer = BlockingProducer(bus)
    producer_task = asyncio.create_task(producer.run())
    control = _Control(bus)
    running = asyncio.create_task(
        run_repl(
            bus=bus,
            control=control,
            input_reader=_Input(("Start", "exit")),
            writer=_Writer(),
        )
    )
    await producer.delta_seen.wait()

    running.cancel()
    await running
    producer_task.cancel()
    await asyncio.gather(producer_task, return_exceptions=True)

    assert control.cancel_calls == 1
