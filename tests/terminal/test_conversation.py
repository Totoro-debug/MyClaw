from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import UUID
from xml.etree import ElementTree

import pytest
from textual.widgets import TextArea

from myclaw.agent.events import (
    AgentEvent,
    ConversationPort,
    TextDeltaPayload,
    TurnCompletedPayload,
    TurnStartedPayload,
)
from myclaw.agent.runtime import PreparedReplRuntime
from myclaw.provider.models import ModelUsage
from myclaw.terminal.conversation import TerminalConversationApp
from tests.agent.test_fixed_catalog_runtime import (
    _response,
    _RuntimeProvider,
)
from tests.agent.test_fixed_catalog_runtime import (
    _runtime as _prepared_runtime,
)

NOW = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
TURN_ID = UUID("0f8fad5b-d9cb-469f-a165-70867728950e")


class ScriptedConversation:
    def __init__(
        self,
        *,
        pause_before_events: bool = False,
        pause_after_first_delta: bool = False,
    ) -> None:
        self.submissions: list[str] = []
        self.before_events = asyncio.Event()
        self.first_delta_emitted = asyncio.Event()
        self._continue_to_events = asyncio.Event()
        self._continue_after_first_delta = asyncio.Event()
        if not pause_before_events:
            self._continue_to_events.set()
        if not pause_after_first_delta:
            self._continue_after_first_delta.set()

    def continue_to_events(self) -> None:
        self._continue_to_events.set()

    def continue_turn(self) -> None:
        self._continue_after_first_delta.set()

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        self.submissions.append(text)
        self.before_events.set()
        await self._continue_to_events.wait()
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=TURN_ID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        yield AgentEvent(
            type="text_delta",
            event_id=1,
            turn_id=TURN_ID,
            created_at=NOW,
            payload=TextDeltaPayload(delta="First "),
        )
        self.first_delta_emitted.set()
        await self._continue_after_first_delta.wait()
        yield AgentEvent(
            type="text_delta",
            event_id=2,
            turn_id=TURN_ID,
            created_at=NOW,
            payload=TextDeltaPayload(delta="answer."),
        )
        yield AgentEvent(
            type="turn_completed",
            event_id=3,
            turn_id=TURN_ID,
            created_at=NOW,
            payload=TurnCompletedPayload(
                content="First answer.",
                usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3),
            ),
        )


class FakeRuntime:
    def __init__(self, conversation: ScriptedConversation) -> None:
        self.conversation = cast(ConversationPort, conversation)
        self.start_calls = 0
        self.close_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


def _runtime(conversation: ScriptedConversation) -> FakeRuntime:
    return FakeRuntime(conversation)


def _visible_screen_text(app: TerminalConversationApp) -> str:
    screenshot = ElementTree.fromstring(app.export_screenshot(simplify=True))
    text = "".join(
        element.text or ""
        for element in screenshot.iter()
        if element.tag.endswith("text") and not element.attrib.get("class", "").endswith("-title")
    )
    return text.replace("\xa0", " ")


@pytest.mark.asyncio
async def test_terminal_conversation_starts_blank_and_focuses_input() -> None:
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)):
        visible_text = _visible_screen_text(app)

        assert "Message MyClaw" in visible_text
        assert "Welcome" not in visible_text
        assert "Session" not in visible_text
        assert "model" not in visible_text.casefold()
        assert isinstance(app.screen.focused, TextArea)
        assert runtime.start_calls == 1

    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_nonblank_enter_echoes_user_before_consuming_agent_events() -> None:
    app: TerminalConversationApp
    conversation = ScriptedConversation(pause_before_events=True)
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press("h", "i", "enter"))
        try:
            await asyncio.wait_for(conversation.before_events.wait(), timeout=1)
            await asyncio.sleep(0.05)

            assert conversation.submissions == ["hi"]
            assert "hi" in _visible_screen_text(app)
        finally:
            conversation.continue_to_events()
            await asyncio.wait_for(submission, timeout=1)


@pytest.mark.asyncio
async def test_text_deltas_update_one_assistant_markdown_and_completion_wins() -> None:
    conversation = ScriptedConversation(pause_after_first_delta=True)
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press("h", "i", "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            await asyncio.sleep(0.05)
            partial_text = _visible_screen_text(app)

            assert partial_text.count("First") == 1
            assert "answer." not in partial_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)

        await asyncio.sleep(0.05)
        completed_text = _visible_screen_text(app)
        assert completed_text.count("First answer.") == 1
        assert completed_text.count("First") == 1

    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_terminal_conversation_uses_the_prepared_runtime_lifecycle(
    agent_home: Path,
    workspace: Path,
) -> None:
    provider = _RuntimeProvider((_response(content="Prepared runtime answer."),))
    runtime = _prepared_runtime(agent_home, workspace, provider)
    app = TerminalConversationApp(runtime)

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("h", "i", "enter")

        visible_text = _visible_screen_text(app)
        assert "hi" in visible_text
        assert "Prepared runtime answer." in visible_text

    assert provider.closed
