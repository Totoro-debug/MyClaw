from collections import deque
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from uuid import UUID

import pytest

from myclaw.agent.events import (
    AgentEvent,
    ConfirmationRequestedPayload,
    TurnCompletedPayload,
    TurnStartedPayload,
)
from myclaw.provider.models import ModelUsage
from myclaw.terminal.repl import run_repl

NOW = datetime(2026, 8, 7, 12, 0, 0, tzinfo=UTC)
TURN_UUID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")
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


class _ConfirmationConversation:
    def __init__(self) -> None:
        self.submitted: list[str] = []
        self.responses: list[tuple[UUID, str]] = []
        self.request = ConfirmationRequestedPayload(
            confirmation_id=CONFIRMATION_UUID,
            turn_id=TURN_UUID,
            tool_call_id="call_schedule",
            tool_name="schedule",
            summary="Add a Schedule Job",
            details={"message": "Check the build", "every_seconds": 60},
            warnings=("This changes Workspace State.",),
        )

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        self.submitted.append(text)
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        yield AgentEvent(
            type="confirmation_requested",
            event_id=1,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=self.request,
        )
        yield AgentEvent(
            type="turn_completed",
            event_id=2,
            turn_id=TURN_UUID,
            created_at=NOW,
            payload=TurnCompletedPayload(
                content="Done.",
                usage=ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            ),
        )

    def respond_to_confirmation(self, confirmation_id: UUID, decision: str) -> None:
        self.responses.append((confirmation_id, decision))

    async def cancel_active_turn(self) -> None:
        return None


@pytest.mark.asyncio
async def test_repl_displays_normalized_confirmation_and_accepts_only_yes_or_no_contract() -> None:
    conversation = _ConfirmationConversation()
    writer = _Writer()

    await run_repl(
        conversation=conversation,
        input_reader=_Input(("original request", "not sure", " y ", "exit")),
        writer=writer,
    )

    assert conversation.submitted == ["original request"]
    assert conversation.responses == [(CONFIRMATION_UUID, "approved")]
    assert writer.deltas == []
    assert writer.finished == 1
    assert any("Add a Schedule Job" in line for line in writer.lines)
    assert any('"every_seconds":60' in line for line in writer.lines)
    assert any("This changes Workspace State." in line for line in writer.lines)
    assert sum("yes/y" in line for line in writer.lines) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ("no", "n", ""))
async def test_repl_confirmation_declines_on_no_or_empty_input(reply: str) -> None:
    conversation = _ConfirmationConversation()
    writer = _Writer()

    await run_repl(
        conversation=conversation,
        input_reader=_Input(("original request", reply, "exit")),
        writer=writer,
    )

    assert conversation.responses == [(CONFIRMATION_UUID, "declined")]
