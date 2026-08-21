from __future__ import annotations

from collections import deque
from typing import Literal
from uuid import UUID

import pytest

from myclaw.agent.loop import ConfirmationCallback
from myclaw.agent.message_bus import MessageBus, OutboundMessage
from myclaw.terminal.repl import run_repl
from myclaw.tools.tool_gateway import ConfirmationRequest


class _Input:
    def __init__(self, values: tuple[str | None, ...]) -> None:
        self._values = deque(values)

    async def read(self) -> str | None:
        return self._values.popleft()


class _Writer:
    def __init__(self) -> None:
        self.deltas: list[str] = []
        self.finishes = 0
        self.lines: list[str] = []

    async def write_delta(self, delta: str) -> None:
        self.deltas.append(delta)

    async def finish_turn(self) -> None:
        self.finishes += 1

    async def write_line(self, content: str) -> None:
        self.lines.append(content)


class _Control:
    has_active_run = False
    has_pending_confirmation = False

    async def cancel_active_run(self) -> None:
        return None

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        del callback

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: Literal["approved", "declined"],
    ) -> None:
        del confirmation_id, decision


class _ImmediateConfirmationControl(_Control):
    def __init__(self) -> None:
        self.request = ConfirmationRequest(
            UUID("16fd2706-8baf-4334-8c7f-ada847da0314"),
            "call-read",
            "read_file",
            "Read outside the Workspace.",
            {"path": "C:/outside.txt"},
        )
        self.responses: list[tuple[UUID, Literal["approved", "declined"]]] = []

    def bind_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        callback(self.request)

    def respond_to_confirmation(
        self,
        confirmation_id: UUID,
        decision: Literal["approved", "declined"],
    ) -> None:
        self.responses.append((confirmation_id, decision))


@pytest.mark.asyncio
async def test_repl_preserves_tool_output_when_confirmation_is_ready_together() -> None:
    bus = MessageBus()
    raw_arguments = '{"path":"C:/outside.txt"}'
    await bus.put_outbound(
        OutboundMessage(
            "tool_call",
            "read_file",
            {"tool_call_id": "call-read", "arguments": raw_arguments},
        )
    )
    await bus.put_outbound(OutboundMessage("model_response", "", {"_streamed": True}))
    control = _ImmediateConfirmationControl()
    writer = _Writer()

    await run_repl(
        bus=bus,
        control=control,
        input_reader=_Input(("inspect", "n", "exit")),
        writer=writer,
    )

    assert f"Tool: read_file\nArguments: {raw_arguments}" in writer.lines
    assert control.responses == [(control.request.confirmation_id, "declined")]
    assert writer.finishes == 1
