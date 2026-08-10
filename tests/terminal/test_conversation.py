from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast
from uuid import UUID
from xml.etree import ElementTree

import pytest
from textual.events import MouseScrollRight
from textual.widgets import Markdown, TextArea

from myclaw.agent.events import (
    AgentEvent,
    ConversationPort,
    TextDeltaPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
    TurnStartedPayload,
)
from myclaw.agent.runtime import PreparedReplRuntime
from myclaw.errors import ErrorInfo
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
        deltas: tuple[str, ...] = ("First ", "answer."),
        completed_content: str = "First answer.",
        outcomes: tuple[Literal["completed", "cancelled", "failed"], ...] = ("completed",),
        cancelled_content: str = "",
        failure_message: str = "The turn failed.",
    ) -> None:
        self.submissions: list[str] = []
        self._deltas = deltas
        self._completed_content = completed_content
        self._outcomes = outcomes
        self._cancelled_content = cancelled_content
        self._failure_message = failure_message
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
        outcome = self._outcomes[min(len(self.submissions) - 1, len(self._outcomes) - 1)]
        self.before_events.set()
        await self._continue_to_events.wait()
        yield AgentEvent(
            type="turn_started",
            event_id=0,
            turn_id=TURN_ID,
            created_at=NOW,
            payload=TurnStartedPayload(),
        )
        for event_id, delta in enumerate(self._deltas, start=1):
            yield AgentEvent(
                type="text_delta",
                event_id=event_id,
                turn_id=TURN_ID,
                created_at=NOW,
                payload=TextDeltaPayload(delta=delta),
            )
            if event_id == 1:
                self.first_delta_emitted.set()
                await self._continue_after_first_delta.wait()
        event_id = len(self._deltas) + 1
        if outcome == "completed":
            yield AgentEvent(
                type="turn_completed",
                event_id=event_id,
                turn_id=TURN_ID,
                created_at=NOW,
                payload=TurnCompletedPayload(
                    content=self._completed_content,
                    usage=ModelUsage(input_tokens=1, output_tokens=2, total_tokens=3),
                ),
            )
        elif outcome == "cancelled":
            yield AgentEvent(
                type="turn_cancelled",
                event_id=event_id,
                turn_id=TURN_ID,
                created_at=NOW,
                payload=TurnCancelledPayload(partial_content=self._cancelled_content),
            )
        else:
            yield AgentEvent(
                type="turn_failed",
                event_id=event_id,
                turn_id=TURN_ID,
                created_at=NOW,
                payload=TurnFailedPayload(
                    error=ErrorInfo(code="model_failed", message=self._failure_message)
                ),
            )


class FailingConversation:
    def __init__(self) -> None:
        self.closed = asyncio.Event()

    async def submit(self, text: str) -> AsyncIterator[AgentEvent]:
        del text
        try:
            yield AgentEvent(
                type="text_delta",
                event_id=1,
                turn_id=TURN_ID,
                created_at=NOW,
                payload=TextDeltaPayload(delta="partial"),
            )
        finally:
            self.closed.set()


class FailingMarkdownStream:
    async def write(self, markdown_fragment: str) -> None:
        del markdown_fragment
        raise RuntimeError("markdown write failed")

    async def stop(self) -> None:
        raise RuntimeError("markdown stop failed")


class FakeRuntime:
    def __init__(self, conversation: object) -> None:
        self.conversation = cast(ConversationPort, conversation)
        self.start_calls = 0
        self.close_calls = 0

    async def start(self) -> None:
        self.start_calls += 1

    async def close(self) -> None:
        self.close_calls += 1


def _runtime(conversation: object) -> FakeRuntime:
    return FakeRuntime(conversation)


def _visible_screen_text(app: TerminalConversationApp) -> str:
    text = "".join(element.text or "" for element in _screenshot_text_elements(app))
    return text.replace("\xa0", " ")


def _screenshot(app: TerminalConversationApp) -> ElementTree.Element:
    return ElementTree.fromstring(app.export_screenshot(simplify=True))


def _screenshot_text_elements(app: TerminalConversationApp) -> list[ElementTree.Element]:
    return [
        element
        for element in _screenshot(app).iter()
        if element.tag.endswith("text")
        and element.text
        and not element.attrib.get("class", "").endswith("-title")
    ]


def _content_text_nodes(app: TerminalConversationApp, marker: str) -> list[str]:
    return [
        (element.text or "").replace("\xa0", " ")
        for element in _screenshot_text_elements(app)
        if all(character in marker for character in (element.text or "").replace("\xa0", " "))
    ]


def _screenshot_text_nodes(app: TerminalConversationApp) -> list[tuple[str, float, float]]:
    return [
        (
            (element.text or "").replace("\xa0", " "),
            float(element.attrib["x"]),
            float(element.attrib["y"]),
        )
        for element in _screenshot_text_elements(app)
    ]


def _screenshot_width(app: TerminalConversationApp) -> float:
    return float(_screenshot(app).attrib["viewBox"].split()[2])


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
async def test_terminal_conversation_inherits_the_terminal_background() -> None:
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)):
        input_area = app.screen.focused

        assert isinstance(input_area, TextArea)
        assert app.screen.styles.background.a == 0
        assert input_area.styles.background.a == 0


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
    conversation = ScriptedConversation(
        pause_after_first_delta=True,
        completed_content="Final answer.",
    )
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
        assert completed_text.count("Final answer.") == 1
        assert "First answer." not in completed_text

    assert runtime.close_calls == 1


@pytest.mark.asyncio
async def test_streamed_markdown_preserves_reading_structure_and_link_urls() -> None:
    content = (
        "# Heading\n\n"
        "- first item\n"
        "- second item\n\n"
        "> quoted text\n\n"
        "```python\n"
        "long_value = " + "'x'" * 30 + "\n```\n\n"
        "[documentation](https://example.com/docs)\n"
        "![architecture image](https://example.com/asset.png)\n"
    )
    conversation = ScriptedConversation(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(100, 40)) as pilot:
        await pilot.press(*list("read"), "enter")
        await asyncio.sleep(0.1)

        visible_text = _visible_screen_text(app)
        assert "Heading" in visible_text
        assert "first item" in visible_text
        assert "second item" in visible_text
        assert "quoted text" in visible_text
        assert "long_value" in visible_text
        assert "documentation (https://example.com/docs)" in visible_text
        assert "architecture image" in visible_text
        assert "https://example.com/asset.png" in visible_text
        assert "@click" not in app.export_screenshot()


@pytest.mark.asyncio
async def test_markdown_structure_is_visible_while_the_fenced_block_is_incomplete() -> None:
    partial = "# Heading\n\n- first item\n\n> quoted text\n\n```python\nvalue = 1"
    content = partial + "\n```\n"
    conversation = ScriptedConversation(
        pause_after_first_delta=True,
        deltas=(partial, "\n```\n"),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("progress"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            await asyncio.sleep(0.05)

            visible_text = _visible_screen_text(app)
            assert "Heading" in visible_text
            assert "first item" in visible_text
            assert "quoted text" in visible_text
            assert "value=1" in visible_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)


@pytest.mark.asyncio
async def test_high_frequency_deltas_do_not_delay_the_exact_terminal_content() -> None:
    streamed_content = "draft-" * 300
    conversation = ScriptedConversation(
        deltas=tuple(streamed_content),
        completed_content="# Complete\n\nExact final content.",
    )
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await asyncio.wait_for(pilot.press(*list("burst"), "enter"), timeout=5)
        await asyncio.sleep(0.1)

        visible_text = _visible_screen_text(app)
        assert "Complete" in visible_text
        assert "Exact final content." in visible_text
        assert "draft-" not in visible_text


@pytest.mark.asyncio
async def test_long_markdown_code_lines_remain_unwrapped_in_a_narrow_terminal() -> None:
    code_line = "very_long_variable_name = " + "0123456789" * 8 + "TAIL"
    content = f"```python\n{code_line}\n```"
    conversation = ScriptedConversation(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(48, 20)) as pilot:
        await pilot.press(*list("code"), "enter")
        await asyncio.sleep(0.1)

        nodes = [
            (text, y)
            for text, _, y in _screenshot_text_nodes(app)
            if text and all(character in code_line for character in text)
        ]
        assert "".join(text for text, _ in nodes).startswith("very_long_variable_name")
        assert any("0" in text for text, _ in nodes)
        assert len({y for _, y in nodes}) == 1

        assert "TAIL" not in _visible_screen_text(app)
        await pilot._post_mouse_events([MouseScrollRight], offset=(10, 5), times=30)
        assert "TAIL" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_incomplete_streamed_markdown_remains_visible_before_completion() -> None:
    content = "[documentation](https://example.com/docs)\n\n```python\nvalue = 1\n```"
    conversation = ScriptedConversation(
        pause_after_first_delta=True,
        deltas=("[documentation](", "https://example.com/docs)\n\n```python\n", "value = 1\n```"),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        submission = asyncio.create_task(pilot.press(*list("partial"), "enter"))
        try:
            await asyncio.wait_for(conversation.first_delta_emitted.wait(), timeout=1)
            await asyncio.sleep(0.05)

            partial_text = _visible_screen_text(app)
            assert "[documentation](" in partial_text
            assert "partial" in partial_text
        finally:
            conversation.continue_turn()
            await asyncio.wait_for(submission, timeout=1)

        await asyncio.sleep(0.05)
        assert "documentation (https://example.com/docs)" in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_streamed_markdown_reflows_cjk_content_after_resize() -> None:
    content = "界" * 20
    conversation = ScriptedConversation(
        deltas=tuple(content),
        completed_content=content,
    )
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("cjk"), "enter")
        await asyncio.sleep(0.1)
        assert _content_text_nodes(app, content) == [content]

        await pilot.resize_terminal(40, 18)
        await asyncio.sleep(0.05)
        narrow_lines = _content_text_nodes(app, content)
        assert len(narrow_lines) == 2
        assert "".join(narrow_lines) == content

        await pilot.resize_terminal(80, 24)
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "cancelled_content", "expected_partial", "expected_status"),
    [
        ("cancelled", "Cancelled exact content.", "Cancelled exact content.", None),
        ("failed", "", "draft content", "Model unavailable."),
    ],
)
async def test_terminal_outcomes_settle_markdown_and_allow_a_subsequent_turn(
    outcome: Literal["cancelled", "failed"],
    cancelled_content: str,
    expected_partial: str,
    expected_status: str | None,
) -> None:
    conversation = ScriptedConversation(
        deltas=("draft ", "content"),
        completed_content="Recovered response.",
        outcomes=(outcome, "completed"),
        cancelled_content=cancelled_content,
        failure_message="Model unavailable.",
    )
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("first"), "enter")
        await asyncio.sleep(0.05)
        visible_text = _visible_screen_text(app)
        assert expected_partial in visible_text
        if expected_status is not None:
            assert expected_status in visible_text

        await pilot.press(*list("again"), "enter")
        await asyncio.sleep(0.05)
        assert conversation.submissions == ["first", "again"]
        assert "Recovered response." in _visible_screen_text(app)


@pytest.mark.asyncio
async def test_markdown_failure_still_closes_the_conversation_event_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conversation = FailingConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))
    markdown_stream = FailingMarkdownStream()
    monkeypatch.setattr(
        Markdown,
        "get_stream",
        classmethod(lambda cls, markdown: markdown_stream),
    )

    with pytest.raises(RuntimeError, match="markdown stop failed"):
        async with app.run_test(size=(80, 24)) as pilot:
            await pilot.press(*list("fail"), "enter")

    assert conversation.closed.is_set()


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


@pytest.mark.asyncio
async def test_narrow_terminal_uses_full_message_width_for_readable_content() -> None:
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))
    content = "0123456789ABCDEFGHIJ"

    async with app.run_test(size=(30, 16)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)

        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
async def test_wide_terminal_constrains_messages_to_a_comfortable_line_width() -> None:
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))
    terminal_width = 80
    content = "X" * 100

    async with app.run_test(size=(terminal_width, 24)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)

        lines = _content_text_nodes(app, content)
        display_width = terminal_width - 4
        assert "".join(lines) == content
        assert max(map(len, lines)) / display_width == pytest.approx(0.72, abs=0.08)


@pytest.mark.asyncio
async def test_messages_are_side_aligned_with_role_accents_on_wide_terminals() -> None:
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list("user"), "enter")
        await asyncio.sleep(0.05)

        nodes = _screenshot_text_nodes(app)
        user_x = next(x for text, x, _ in nodes if text == "user")
        assistant_x = next(x for text, x, _ in nodes if text == "First answer.")
        accents = [(x, y) for text, x, y in nodes if text == "│"]
        screenshot_width = _screenshot_width(app)

        assert user_x > screenshot_width * 0.65
        assert assistant_x < screenshot_width * 0.25
        assert min(x for x, _ in accents) < screenshot_width * 0.25
        assert max(x for x, _ in accents) > screenshot_width * 0.75


@pytest.mark.asyncio
async def test_cjk_double_width_content_reflows_when_terminal_is_resized() -> None:
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))
    content = "界" * 20

    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press(*list(content), "enter")
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]

        await pilot.resize_terminal(40, 18)
        await asyncio.sleep(0.05)
        narrow_lines = _content_text_nodes(app, content)
        assert len(narrow_lines) == 2
        assert "".join(narrow_lines) == content

        await pilot.resize_terminal(80, 24)
        await asyncio.sleep(0.05)
        assert _content_text_nodes(app, content) == [content]


@pytest.mark.asyncio
async def test_role_accents_remain_visible_with_limited_ansi_colors() -> None:
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))
    app.ansi_color = True

    async with app.run_test(size=(40, 18)) as pilot:
        await pilot.press(*list("hello"), "enter")
        await asyncio.sleep(0.05)

        visible_text = _visible_screen_text(app)
        assert app.native_ansi_color
        assert "hello│" in visible_text
        assert "│First answer." in visible_text


@pytest.mark.asyncio
async def test_role_accents_remain_visible_without_terminal_color(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NO_COLOR", "1")
    conversation = ScriptedConversation()
    runtime = _runtime(conversation)
    app = TerminalConversationApp(cast(PreparedReplRuntime, runtime))

    async with app.run_test(size=(40, 18)) as pilot:
        await pilot.press(*list("hello"), "enter")
        await asyncio.sleep(0.05)

        visible_text = _visible_screen_text(app)
        assert "nocolor" in app.screen.pseudo_classes
        assert "hello│" in visible_text
        assert "│First answer." in visible_text
