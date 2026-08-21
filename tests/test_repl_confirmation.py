from __future__ import annotations

import asyncio
from collections import deque
from uuid import UUID

import pytest

from myclaw.agent.message_bus import MessageBus, OutboundMessage
from myclaw.terminal.repl import run_repl
from myclaw.tools.tool_gateway import ConfirmationRequest

CONFIRMATION_UUID = UUID("16fd2706-8baf-4334-8c7f-ada847da0314")


class _Input:
    def __init__(self, values: tuple[str | None, ...]) -> None:
        self._values = deque(values)

    async def read(self) -> str | None:
        return self._values.popleft()


class _Writer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.deltas: list[str] = []
        self.finished = 0

    async def write_delta(self, delta: str) -> None:
        self.deltas.append(delta)

    async def finish_turn(self) -> None:
        self.finished += 1

    async def write_line(self, content: str) -> None:
        self.lines.append(content)


class _Control:
    has_active_run = False
    has_pending_confirmation = True

    def __init__(self) -> None:
        self.callback: object | None = None
        self.responses: list[tuple[UUID, str]] = []
        self.response_ready = asyncio.Event()

    def bind_confirmation_callback(self, callback: object) -> None:
        self.callback = callback

    async def cancel_active_run(self) -> None:
        return None

    def respond_to_confirmation(self, confirmation_id: UUID, decision: str) -> None:
        self.responses.append((confirmation_id, decision))
        self.response_ready.set()


async def _confirmation_producer(
    bus: MessageBus,
    control: _Control,
    request: ConfirmationRequest,
) -> None:
    inbound = await bus.get_inbound()
    assert inbound.content == "original request"
    callback = control.callback
    assert callable(callback)
    callback(request)
    await control.response_ready.wait()
    await bus.put_outbound(OutboundMessage("model_response", "", {"_streamed": True}))


def _request() -> ConfirmationRequest:
    return ConfirmationRequest(
        confirmation_id=CONFIRMATION_UUID,
        tool_call_id="call_schedule",
        tool_name="schedule",
        summary="Add a Schedule Job",
        details={"message": "Check the build", "every_seconds": 60},
        warnings=("This changes Workspace State.",),
    )


@pytest.mark.asyncio
async def test_repl_displays_normalized_confirmation_and_accepts_only_yes_or_no_contract() -> None:
    bus = MessageBus()
    control = _Control()
    producer = asyncio.create_task(_confirmation_producer(bus, control, _request()))
    writer = _Writer()

    await run_repl(
        bus=bus,
        control=control,
        input_reader=_Input(("original request", "not sure", " y ", "exit")),
        writer=writer,
    )
    await producer

    assert control.responses == [(CONFIRMATION_UUID, "approved")]
    assert writer.deltas == []
    assert writer.finished == 1
    assert any("Add a Schedule Job" in line for line in writer.lines)
    assert any('"every_seconds":60' in line for line in writer.lines)
    assert any("This changes Workspace State." in line for line in writer.lines)
    assert sum("yes/y" in line for line in writer.lines) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ("no", "n", ""))
async def test_repl_confirmation_declines_on_no_or_empty_input(reply: str) -> None:
    bus = MessageBus()
    control = _Control()
    producer = asyncio.create_task(_confirmation_producer(bus, control, _request()))
    writer = _Writer()

    await run_repl(
        bus=bus,
        control=control,
        input_reader=_Input(("original request", reply, "exit")),
        writer=writer,
    )
    await producer

    assert control.responses == [(CONFIRMATION_UUID, "declined")]
