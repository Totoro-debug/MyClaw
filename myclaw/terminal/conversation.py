"""Full-screen Textual host for the Terminal Conversation."""

from __future__ import annotations

from asyncio import CancelledError, current_task
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from markdown_it import MarkdownIt
from markdown_it.rules_core.state_core import StateCore
from markdown_it.token import Token
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.driver import Driver
from textual.events import Key, MouseScrollDown, MouseScrollUp, Resize, Unmount
from textual.message import Message
from textual.scrollbar import ScrollTo
from textual.widget import Widget
from textual.widgets import Markdown, Static, TextArea
from textual.worker import Worker, WorkerError

from myclaw.agent.events import (
    AgentEvent,
    TextDeltaPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
)
from myclaw.agent.runtime import PreparedReplRuntime
from myclaw.terminal.keyboard import EnhancedKeyboardAction, EnhancedKeyboardAdapter

__all__ = ["TerminalConversationApp", "run_terminal_conversation"]

_COMPACT_MESSAGE_MAX_WIDTH = 60
_CONVERSATION_NAVIGATION_KEYS = frozenset({"pageup", "pagedown", "ctrl+home", "ctrl+end"})
type _ControlAction = Literal["cancel_active_turn", "clear_draft", "exit"]


class _ConversationInput(TextArea):
    """Multiline input whose ordinary Enter key submits the current draft."""

    class ControlAction(Message):
        def __init__(
            self,
            text_area: _ConversationInput,
            action: _ControlAction,
            *,
            turn_token: object | None = None,
        ) -> None:
            super().__init__()
            self.text_area = text_area
            self.action = action
            self.turn_token = turn_token

    class Submitted(Message):
        def __init__(self, text_area: _ConversationInput, text: str) -> None:
            super().__init__()
            self.text_area = text_area
            self.text = text

    def on_mount(self) -> None:
        self._history: list[str] = []
        self._history_index: int | None = None
        self._history_draft = ""
        self.active_turn_token: object | None = None

    def remember_submission(self, text: str) -> None:
        """Keep accepted input only for this live application instance."""
        self._history.append(text)
        self._history_index = None
        self._history_draft = ""

    def _navigate_history(self, direction: int) -> bool:
        if not self._history or (self.text and self._history_index is None):
            return False

        if self._history_index is None:
            self._history_draft = self.text
            self._history_index = len(self._history) - 1
        else:
            next_index = self._history_index + direction
            if next_index < 0:
                next_index = 0
            if next_index >= len(self._history):
                self._history_index = None
                self.text = self._history_draft
                return True
            self._history_index = next_index

        self.text = self._history[self._history_index]
        self.move_cursor((len(self.document.lines) - 1, len(self.document.lines[-1])))
        return True

    def _leave_history(self) -> None:
        self._history_index = None
        self._history_draft = ""

    async def _on_key(self, event: Key) -> None:
        if event.key in _CONVERSATION_NAVIGATION_KEYS:
            event.stop()
            event.prevent_default()
            self.app.query_one("#conversation-display", _ConversationDisplay).navigate(event.key)
            return
        control_action: _ControlAction | None = None
        turn_token: object | None = None
        if event.key == "ctrl+c":
            if self.read_only:
                control_action = "cancel_active_turn"
                turn_token = self.active_turn_token
            elif self.text:
                control_action = "clear_draft"
            else:
                control_action = "exit"
        elif event.key == "ctrl+d" and not self.text:
            control_action = "exit"
        if control_action is not None:
            event.stop()
            event.prevent_default()
            self.post_message(self.ControlAction(self, control_action, turn_token=turn_token))
            return
        if self.read_only:
            event.stop()
            event.prevent_default()
            return
        if event.key in {"up", "down"} and self._navigate_history(-1 if event.key == "up" else 1):
            event.stop()
            event.prevent_default()
            return
        action = EnhancedKeyboardAdapter.action_for_key(event.key)
        if self._history_index is not None and (
            event.is_printable
            or event.key in {"backspace", "delete", "ctrl+backspace", "ctrl+delete"}
            or action is not None
        ):
            self._leave_history()

        if action is EnhancedKeyboardAction.NEWLINE:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if action is EnhancedKeyboardAction.SUBMIT:
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self, self.text))
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


@dataclass(frozen=True, slots=True)
class _ScrollAnchor:
    widget: Widget
    relative_position: float


class _ConversationDisplay(VerticalScroll):
    """Conversation viewport with bottom-following and historical navigation."""

    class Resized(Message):
        def __init__(self, display: _ConversationDisplay, width: int) -> None:
            super().__init__()
            self.display = display
            self.width = width

    _following = True
    _new_content = False
    _last_scroll_state: tuple[bool, bool] | None = None
    _historical_anchor: _ScrollAnchor | None = None
    _resize_anchor: _ScrollAnchor | None = None

    @property
    def following(self) -> bool:
        """Whether new content should keep the viewport at the bottom."""
        return self._following

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._sync_after_user_scroll()

    def _on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        super()._on_mouse_scroll_up(event)
        self._sync_after_user_scroll()

    def _on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        super()._on_mouse_scroll_down(event)
        self._sync_after_user_scroll()

    def _on_scroll_to(self, message: ScrollTo) -> None:
        super()._on_scroll_to(message)
        self.call_after_refresh(self._sync_after_user_scroll)

    def on_resize(self, event: Resize) -> None:
        self._resize_anchor = None if self.following else self._historical_anchor
        self.post_message(self.Resized(self, event.size.width))

    def restore_resize_anchor(self) -> None:
        anchor = self._resize_anchor
        self._resize_anchor = None
        if self.following:
            self.scroll_end(animate=False, immediate=True)
            self.call_after_refresh(self._follow_latest)
            return
        if anchor is None:
            return

        def restore() -> None:
            if anchor.widget.parent is not self:
                return
            self.scroll_to(
                y=round(
                    anchor.widget.virtual_region.y
                    + anchor.relative_position * anchor.widget.virtual_region.height
                ),
                animate=False,
                immediate=True,
            )
            self._emit_scroll_state()

        self.call_after_refresh(restore)

    def navigate(self, key: str) -> None:
        if key == "pageup":
            page_height = max(1, self.scrollable_content_region.height)
            self.scroll_to(
                y=self.scroll_y - page_height,
                animate=False,
                immediate=True,
            )
        elif key == "pagedown":
            page_height = max(1, self.scrollable_content_region.height)
            self.scroll_to(
                y=self.scroll_y + page_height,
                animate=False,
                immediate=True,
            )
        elif key == "ctrl+home":
            self.scroll_home(animate=False, immediate=True)
        elif key == "ctrl+end":
            self.scroll_end(animate=False, immediate=True)
        else:
            raise ValueError(f"Unsupported conversation navigation key: {key}")
        self._sync_after_user_scroll()

    def content_changed(self) -> None:
        """Follow new content only while the user is at the conversation bottom."""
        if self.following:
            self.scroll_end(animate=False, immediate=True)
            self.call_after_refresh(self._follow_latest)
        else:
            self._new_content = True
        self._emit_scroll_state()

    def _follow_latest(self) -> None:
        if self.following:
            self.scroll_end(animate=False, immediate=True)

    def _sync_after_user_scroll(self) -> None:
        self._following = self.is_vertical_scroll_end
        if self._following:
            self._new_content = False
            self._historical_anchor = None
        else:
            self._historical_anchor = self._capture_scroll_anchor()
        self._emit_scroll_state()

    def _capture_scroll_anchor(self) -> _ScrollAnchor | None:
        scroll_top = round(self.scroll_y)
        last_child: Widget | None = None
        for child in self.children:
            region = child.virtual_region
            if region.height <= 0:
                continue
            last_child = child
            if region.bottom > scroll_top:
                return _ScrollAnchor(child, (scroll_top - region.y) / region.height)
        if last_child is None:
            return None
        region = last_child.virtual_region
        return _ScrollAnchor(last_child, (scroll_top - region.y) / region.height)

    def _emit_scroll_state(self) -> None:
        at_bottom = self.is_vertical_scroll_end
        if at_bottom and self.following:
            self._new_content = False
        state = (self.following, self._new_content)
        if state == self._last_scroll_state:
            return
        self._last_scroll_state = state
        try:
            self.app.query_one("#new-content", Static).display = state[1]
        except (NoMatches, NoScreen, ScreenStackError):
            pass


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
        height: auto;
        min-height: 3;
        max-height: 8;
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

    #conversation-input-region {
        height: auto;
        min-height: 5;
        max-height: 8;
        width: 100%;
    }

    #new-content {
        display: none;
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
    }

    #turn-status {
        display: none;
        height: 1;
        width: 100%;
        padding: 0 1;
        color: $text-muted;
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
        self._keyboard_adapter = EnhancedKeyboardAdapter()
        self._runtime_started = False
        self._turn_worker: Worker[None] | None = None
        self._cancel_requested_turn: object | None = None
        self._closing = False

    def _build_driver(
        self,
        headless: bool,
        inline: bool,
        mouse: bool,
        size: tuple[int, int] | None,
    ) -> Driver:
        driver = super()._build_driver(headless, inline, mouse, size)
        if driver.is_headless:
            return driver

        # Textual has no public hook around its Kitty push/pop writes. Keep this
        # compatibility boundary narrow and exercise it through App.run_async tests.
        self._keyboard_adapter = EnhancedKeyboardAdapter.install_on_driver(driver)
        return driver

    def compose(self) -> ComposeResult:
        yield _ConversationDisplay(id="conversation-display")
        yield Vertical(
            Static("New content below", id="new-content", markup=False),
            Static("Working", id="turn-status", markup=False),
            _ConversationInput(id="conversation-input", placeholder="Message MyClaw"),
            id="conversation-input-region",
        )

    async def on_mount(self) -> None:
        self._runtime_started = True
        await self._runtime.start()
        self.query_one(_ConversationInput).focus()

    async def on_unmount(self, event: Unmount) -> None:
        del event
        self._closing = True
        if self._turn_worker is not None:
            self._turn_worker.cancel()
            with suppress(WorkerError):
                await self._turn_worker.wait()
        if self._runtime_started:
            self._runtime_started = False
            await self._runtime.close()

    @on(_ConversationDisplay.Resized)
    def _display_resized(self, message: _ConversationDisplay.Resized) -> None:
        compact = message.width <= _COMPACT_MESSAGE_MAX_WIDTH
        for content in message.display.query(".message"):
            content.set_class(compact, "message-compact")
        message.display.restore_resize_anchor()

    @on(_ConversationInput.Submitted)
    async def _submit_input(self, message: _ConversationInput.Submitted) -> None:
        text = message.text
        if not text.strip() or message.text_area.read_only:
            return
        if text.strip().casefold() in {"exit", "quit"}:
            message.text_area.text = ""
            self.exit()
            return

        message.text_area.remember_submission(text)
        message.text_area.text = ""
        turn_token = object()
        message.text_area.active_turn_token = turn_token
        message.text_area.read_only = True
        self._set_working(True)
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
        display.content_changed()

        self._turn_worker = self.run_worker(
            self._run_turn(text, message.text_area, turn_token),
            name="conversation-turn",
            group="conversation-turn",
            exclusive=True,
            exit_on_error=False,
        )

    @on(_ConversationInput.ControlAction)
    async def _handle_control_action(self, message: _ConversationInput.ControlAction) -> None:
        message.stop()
        text_area = message.text_area
        if message.action == "cancel_active_turn":
            turn_token = message.turn_token
            if (
                turn_token is None
                or turn_token is not text_area.active_turn_token
                or self._cancel_requested_turn is turn_token
            ):
                return
            self._cancel_requested_turn = turn_token
            try:
                await self._runtime.conversation.cancel_active_turn()
            except Exception as error:
                if self._cancel_requested_turn is turn_token:
                    self._cancel_requested_turn = None
                self._handle_exception(error)
            return
        if message.action == "clear_draft":
            text_area.text = ""
            return
        self.exit()

    async def _run_turn(
        self,
        text: str,
        text_area: _ConversationInput,
        turn_token: object,
    ) -> None:
        try:
            await self._consume_turn(self._runtime.conversation.submit(text))
        except CancelledError:
            if self._closing:
                raise
            _clear_current_task_cancellation()
            await self._mount_status("Turn cancelled.")
        except Exception as error:
            self._handle_exception(error)
        finally:
            if text_area.active_turn_token is turn_token:
                text_area.active_turn_token = None
                if self._cancel_requested_turn is turn_token:
                    self._cancel_requested_turn = None
                self._set_working(False)
                text_area.read_only = False
                if not self._closing:
                    with suppress(Exception):
                        text_area.focus()

    async def _consume_turn(self, events: AsyncIterator[AgentEvent]) -> None:
        assistant: Markdown | None = None
        stream: _MarkdownStream | None = None
        streamed_fragments: list[str] = []
        terminal_content: str | None = None
        terminal_status: str | None = None
        cancelled = False
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
                    terminal_status = "Turn cancelled."
                elif event.type == "turn_failed":
                    payload = event.payload
                    if not isinstance(payload, TurnFailedPayload):
                        raise TypeError("turn_failed event has an invalid payload")
                    terminal_status = payload.error.message
        except CancelledError:
            cancelled = True
            raise
        finally:
            try:
                if stream is not None:
                    try:
                        await stream.stop()
                    except BaseException:
                        if not cancelled:
                            raise
                final_content = (
                    terminal_content
                    if terminal_content is not None
                    else "".join(streamed_fragments)
                )
                if not cancelled and (
                    final_content or (assistant is not None and terminal_content is not None)
                ):
                    if assistant is None:
                        assistant = await self._mount_assistant()
                    await assistant.update(final_content)
                    self._scroll_to_latest()
                if not cancelled and terminal_status is not None:
                    await self._mount_status(terminal_status)
            finally:
                if isinstance(events, _ClosableEventStream):
                    try:
                        await events.aclose()
                    except BaseException:
                        if not cancelled:
                            raise

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
        display.content_changed()

    async def _mount_status(self, content: str) -> None:
        display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.mount(Static(content, markup=False, classes="turn-status"))
        self._scroll_to_latest()

    def _set_working(self, working: bool) -> None:
        with suppress(NoMatches, NoScreen, ScreenStackError):
            self.query_one("#turn-status", Static).display = working

    @staticmethod
    def _message_classes(role: str, display: _ConversationDisplay) -> str:
        compact = display.size.width <= _COMPACT_MESSAGE_MAX_WIDTH
        suffix = " message-compact" if compact else ""
        return f"message {role}{suffix}"


def run_terminal_conversation(runtime: PreparedReplRuntime) -> None:
    """Run a Terminal Conversation application around a prepared Runtime."""
    TerminalConversationApp(runtime).run()


def _clear_current_task_cancellation() -> None:
    task = current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()
