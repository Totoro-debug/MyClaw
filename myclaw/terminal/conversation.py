"""Full-screen Textual host for the Terminal Conversation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from markdown_it import MarkdownIt
from markdown_it.rules_core.state_core import StateCore
from markdown_it.token import Token
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.events import Key, Resize, Unmount
from textual.message import Message
from textual.widgets import Markdown, Static, TextArea

from myclaw.agent.events import (
    AgentEvent,
    TextDeltaPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
)
from myclaw.agent.runtime import PreparedReplRuntime

__all__ = ["TerminalConversationApp", "run_terminal_conversation"]

_COMPACT_MESSAGE_MAX_WIDTH = 60


class _ConversationInput(TextArea):
    """Multiline input whose ordinary Enter key submits the current draft."""

    class Submitted(Message):
        def __init__(self, text_area: _ConversationInput, text: str) -> None:
            super().__init__()
            self.text_area = text_area
            self.text = text

    async def _on_key(self, event: Key) -> None:
        if self.read_only:
            event.stop()
            event.prevent_default()
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
            return
        if event.key == "ctrl+j":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


@runtime_checkable
class _ClosableEventStream(Protocol):
    async def aclose(self) -> None: ...


class _MarkdownStream(Protocol):
    async def write(self, markdown_fragment: str) -> None: ...

    async def stop(self) -> None: ...


def _make_links_visible(state: StateCore) -> None:
    """Render Markdown links as ordinary label-and-URL text."""
    for token in state.tokens:
        if token.type != "inline" or token.children is None:
            continue

        children: list[Token] = []
        link_starts: list[tuple[str, int]] = []
        for child in token.children:
            if child.type == "link_open":
                href = child.attrGet("href")
                link_starts.append((str(href) if href is not None else "", len(children)))
                continue
            if child.type == "link_close":
                if link_starts:
                    href, start = link_starts.pop()
                    label = "".join(
                        item.content
                        for item in children[start:]
                        if item.type in {"text", "code_inline"}
                    )
                    if href and label != href:
                        children.append(Token("text", "", 0, content=f" ({href})"))
                continue
            if child.type == "image":
                source = child.attrGet("src")
                href = str(source) if source is not None else ""
                alt = child.content
                content = alt if not href else f"{alt} ({href})"
                children.append(Token("text", "", 0, content=content))
                continue
            children.append(child)
        token.children = children


def _markdown_parser() -> MarkdownIt:
    parser = MarkdownIt("gfm-like")
    parser.core.ruler.after("linkify", "myclaw_visible_links", _make_links_visible)
    return parser


class _ConversationDisplay(VerticalScroll):
    """Conversation viewport that reports terminal width changes to its host."""

    class Resized(Message):
        def __init__(self, display: _ConversationDisplay, width: int) -> None:
            super().__init__()
            self.display = display
            self.width = width

    def on_resize(self, event: Resize) -> None:
        self.post_message(self.Resized(self, event.size.width))


class TerminalConversationApp(App[None]):
    """The two-region Textual application for one prepared Runtime."""

    CSS = """
    Screen {
        layout: vertical;
        background: transparent;
    }

    #conversation-display {
        height: 1fr;
        width: 100%;
        padding: 1 2;
        scrollbar-size-vertical: 1;
        background: transparent;
    }

    #conversation-input {
        height: 5;
        min-height: 3;
        max-height: 6;
        width: 100%;
        border-top: solid $panel;
        padding: 0 1;
        background: transparent;
    }

    .message {
        width: 72%;
        max-width: 100%;
        min-width: 0;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        background: transparent;
    }

    .message-compact {
        width: 100%;
    }

    .user-message {
        text-align: right;
        border-right: solid $foreground;
    }

    .assistant-message {
        border-left: solid $foreground;
    }

    .message-row {
        width: 100%;
        height: auto;
    }

    .user-row {
        align: right top;
    }

    .assistant-row {
        align: left top;
    }

    .turn-status {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        color: $text-muted;
    }
    """

    def __init__(self, runtime: PreparedReplRuntime) -> None:
        super().__init__()
        self._runtime = runtime
        self._runtime_started = False

    def compose(self) -> ComposeResult:
        yield _ConversationDisplay(id="conversation-display")
        yield _ConversationInput(id="conversation-input", placeholder="Message MyClaw")

    async def on_mount(self) -> None:
        self._runtime_started = True
        await self._runtime.start()
        self.query_one(_ConversationInput).focus()

    async def on_unmount(self, event: Unmount) -> None:
        del event
        if self._runtime_started:
            self._runtime_started = False
            await self._runtime.close()

    @on(_ConversationDisplay.Resized)
    def _display_resized(self, message: _ConversationDisplay.Resized) -> None:
        compact = message.width <= _COMPACT_MESSAGE_MAX_WIDTH
        for content in message.display.query(".message"):
            content.set_class(compact, "message-compact")

    @on(_ConversationInput.Submitted)
    async def _submit_input(self, message: _ConversationInput.Submitted) -> None:
        text = message.text
        if not text.strip() or message.text_area.read_only:
            return

        message.text_area.text = ""
        message.text_area.read_only = True
        display = self.query_one("#conversation-display", _ConversationDisplay)
        row = Horizontal(classes="message-row user-row")
        await display.mount(row)
        await row.mount(
            Static(
                text,
                markup=False,
                classes=self._message_classes("user-message", display),
            )
        )
        display.scroll_end(animate=False, immediate=True)

        try:
            await self._consume_turn(self._runtime.conversation.submit(text))
        finally:
            message.text_area.read_only = False
            message.text_area.focus()

    async def _consume_turn(self, events: AsyncIterator[AgentEvent]) -> None:
        assistant: Markdown | None = None
        stream: _MarkdownStream | None = None
        streamed_fragments: list[str] = []
        terminal_content: str | None = None
        try:
            async for event in events:
                if event.type == "text_delta":
                    payload = event.payload
                    if not isinstance(payload, TextDeltaPayload):
                        raise TypeError("text_delta event has an invalid payload")
                    streamed_fragments.append(payload.delta)
                    if assistant is None:
                        assistant = await self._mount_assistant()
                        stream = Markdown.get_stream(assistant)
                    assert stream is not None
                    await stream.write(payload.delta)
                    self._scroll_to_latest()
                elif event.type == "turn_completed":
                    payload = event.payload
                    if not isinstance(payload, TurnCompletedPayload):
                        raise TypeError("turn_completed event has an invalid payload")
                    terminal_content = payload.content
                elif event.type == "turn_cancelled":
                    payload = event.payload
                    if not isinstance(payload, TurnCancelledPayload):
                        raise TypeError("turn_cancelled event has an invalid payload")
                    if payload.partial_content:
                        terminal_content = payload.partial_content
                elif event.type == "turn_failed":
                    payload = event.payload
                    if not isinstance(payload, TurnFailedPayload):
                        raise TypeError("turn_failed event has an invalid payload")
                    await self._mount_status(payload.error.message)
        finally:
            try:
                if stream is not None:
                    await stream.stop()
                final_content = (
                    terminal_content
                    if terminal_content is not None
                    else "".join(streamed_fragments)
                )
                if final_content or (assistant is not None and terminal_content is not None):
                    if assistant is None:
                        assistant = await self._mount_assistant()
                    await assistant.update(final_content)
                    self._scroll_to_latest()
            finally:
                if isinstance(events, _ClosableEventStream):
                    await events.aclose()

    async def _mount_assistant(self) -> Markdown:
        display = self.query_one("#conversation-display", _ConversationDisplay)
        assistant = Markdown(
            classes=self._message_classes("assistant-message", display),
            open_links=False,
            parser_factory=_markdown_parser,
        )
        row = Horizontal(classes="message-row assistant-row")
        await display.mount(row)
        await row.mount(assistant)
        return assistant

    def _scroll_to_latest(self) -> None:
        display = self.query_one("#conversation-display", _ConversationDisplay)
        display.scroll_end(animate=False, immediate=True)

    async def _mount_status(self, content: str) -> None:
        display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.mount(Static(content, markup=False, classes="turn-status"))
        display.scroll_end(animate=False, immediate=True)

    @staticmethod
    def _message_classes(role: str, display: _ConversationDisplay) -> str:
        compact = display.size.width <= _COMPACT_MESSAGE_MAX_WIDTH
        suffix = " message-compact" if compact else ""
        return f"message {role}{suffix}"


def run_terminal_conversation(runtime: PreparedReplRuntime) -> None:
    """Run a Terminal Conversation application around a prepared Runtime."""
    TerminalConversationApp(runtime).run()
