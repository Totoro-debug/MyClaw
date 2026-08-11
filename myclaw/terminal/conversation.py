"""Full-screen Textual host for the Terminal Conversation."""

from __future__ import annotations

import re
from asyncio import CancelledError, current_task
from collections.abc import AsyncIterator, Awaitable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import ClassVar, Literal, Protocol, cast, runtime_checkable
from urllib.parse import urlsplit
from uuid import UUID

from markdown_it import MarkdownIt
from markdown_it.rules_core.state_core import StateCore
from markdown_it.token import Token
from textual import on
from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Center, Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.dom import NoScreen
from textual.driver import Driver
from textual.events import Key, MouseScrollDown, MouseScrollUp, Resize, Unmount
from textual.message import Message
from textual.screen import ModalScreen
from textual.scrollbar import ScrollTo
from textual.widget import Widget
from textual.widgets import Button, Markdown, Static, TextArea
from textual.worker import Worker, WorkerError

from myclaw.agent.events import (
    AgentEvent,
    ConfirmationDecision,
    ConfirmationRequestedPayload,
    TextDeltaPayload,
    ToolCompletedPayload,
    ToolStartedPayload,
    TurnCancelledPayload,
    TurnCompletedPayload,
    TurnFailedPayload,
)
from myclaw.agent.runtime import PreparedReplRuntime
from myclaw.terminal.keyboard import EnhancedKeyboardAction, EnhancedKeyboardAdapter

__all__ = ["TerminalConversationApp", "run_terminal_conversation"]

_COMPACT_MESSAGE_MAX_WIDTH = 60
_CONVERSATION_NAVIGATION_KEYS = frozenset({"pageup", "pagedown", "ctrl+home", "ctrl+end"})
_FAILURE_REASON_MAX_CHARS = 120
_TOOL_NAME_MAX_CHARS = 80
_GENERIC_TOOL_FAILURE_REASON = "The operation did not complete."
_UNSAFE_TOOL_DETAIL_PATTERN = re.compile(
    r"(?:^\s*[\[{])|(?:[\"'][^\"']+[\"']\s*:)|"
    r"(?:\b(?:api[_-]?key|authorization|bearer|password|secret|token)\b)|"
    r"(?:\b(?:arguments?|parameters?|result|output|content)\b\s*[:=])|"
    r"(?:\bcall[-_][A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
type _ControlAction = Literal["cancel_active_turn", "clear_draft", "exit"]
type _ToolRowStatus = Literal["running", "success", "error", "refused"]


@dataclass(frozen=True, slots=True)
class _ToolRowKey:
    turn_id: UUID
    tool_call_id: str


@dataclass(slots=True)
class _ToolRowState:
    widget: Static
    tool_name: str
    status: _ToolRowStatus


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


class _ToolConfirmationScreen(ModalScreen[ConfirmationDecision]):
    """Present one normalized Tool Confirmation without leaving the conversation."""

    CSS = """
    _ToolConfirmationScreen {
        align: center middle;
        padding: 1 2;
    }

    #confirmation-panel {
        width: 80%;
        max-width: 72;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        border: round $warning;
        background: $surface;
        overflow-y: auto;
    }

    #confirmation-heading {
        width: 100%;
        margin-bottom: 1;
        text-style: bold;
    }

    #confirmation-tool,
    #confirmation-reason,
    .confirmation-details,
    .confirmation-warning {
        width: 100%;
        height: auto;
        margin-bottom: 1;
    }

    .confirmation-warning {
        color: $text-warning;
    }

    #confirmation-actions {
        width: 100%;
        height: auto;
        align: center middle;
        margin-top: 1;
    }

    #confirmation-actions Button {
        margin: 0 1;
        height: 3;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "decline", "Decline", show=False),
        Binding("ctrl+c", "decline", "Decline", show=False, priority=True),
        Binding("left,up", "focus_decline", "Decline", show=False),
        Binding("right,down", "focus_approve", "Approve", show=False),
    ]

    def __init__(self, request: ConfirmationRequestedPayload) -> None:
        super().__init__(id=f"confirmation-{request.confirmation_id.hex}")
        self._request = request

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="confirmation-panel"):
                yield Static("Tool Confirmation", id="confirmation-heading", markup=False)
                yield Static(
                    f"Tool: {_friendly_tool_name(self._request.tool_name)}",
                    id="confirmation-tool",
                    markup=False,
                )
                reason = self._request.reason or self._request.summary
                yield Static(f"Reason: {reason}", id="confirmation-reason", markup=False)
                for warning in self._request.warnings:
                    yield Static(
                        f"Warning: {warning}",
                        markup=False,
                        classes="confirmation-warning",
                    )
                for detail in _confirmation_detail_lines(self._request):
                    yield Static(detail, markup=False, classes="confirmation-details")
                with Horizontal(id="confirmation-actions"):
                    yield Button("Decline", id="confirmation-decline")
                    yield Button("Approve", variant="success", id="confirmation-approve")

    def on_mount(self) -> None:
        self.query_one("#confirmation-decline", Button).focus()

    @on(Button.Pressed)
    def _button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        decision: ConfirmationDecision = (
            "approved" if event.button.id == "confirmation-approve" else "declined"
        )
        self.dismiss(decision)

    def action_decline(self) -> None:
        self.dismiss("declined")

    def action_focus_decline(self) -> None:
        self.query_one("#confirmation-decline", Button).focus()

    def action_focus_approve(self) -> None:
        self.query_one("#confirmation-approve", Button).focus()


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

    .tool-row {
        width: 100%;
        min-width: 0;
        height: auto;
        margin: 0 0 1 0;
        padding: 0 1;
        color: $text-muted;
        background: transparent;
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
        self._tool_rows: dict[_ToolRowKey, _ToolRowState] = {}
        self._closed_tool_turns: set[UUID] = set()
        self._active_confirmation_id: UUID | None = None
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
        observed_tool_turns: set[UUID] = set()
        resolved_confirmations: set[UUID] = set()
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
                elif event.type == "tool_started":
                    payload = event.payload
                    if not isinstance(payload, ToolStartedPayload):
                        raise TypeError("tool_started event has an invalid payload")
                    observed_tool_turns.add(event.turn_id)
                    await self._show_tool_started(event.turn_id, payload)
                elif event.type == "confirmation_requested":
                    payload = event.payload
                    if not isinstance(payload, ConfirmationRequestedPayload):
                        raise TypeError("confirmation_requested event has an invalid payload")
                    if payload.confirmation_id in resolved_confirmations:
                        continue
                    await self._request_confirmation(payload)
                    resolved_confirmations.add(payload.confirmation_id)
                elif event.type == "tool_completed":
                    payload = event.payload
                    if not isinstance(payload, ToolCompletedPayload):
                        raise TypeError("tool_completed event has an invalid payload")
                    observed_tool_turns.add(event.turn_id)
                    await self._show_tool_completed(event.turn_id, payload)
                elif event.type == "turn_completed":
                    payload = event.payload
                    if not isinstance(payload, TurnCompletedPayload):
                        raise TypeError("turn_completed event has an invalid payload")
                    await self._finish_tool_turn(
                        event.turn_id,
                        "Tool completion was not reported.",
                    )
                    terminal_content = payload.content
                elif event.type == "turn_cancelled":
                    payload = event.payload
                    if not isinstance(payload, TurnCancelledPayload):
                        raise TypeError("turn_cancelled event has an invalid payload")
                    await self._finish_tool_turn(event.turn_id, "The Tool call was interrupted.")
                    if payload.partial_content:
                        terminal_content = payload.partial_content
                    terminal_status = "Turn cancelled."
                elif event.type == "turn_failed":
                    payload = event.payload
                    if not isinstance(payload, TurnFailedPayload):
                        raise TypeError("turn_failed event has an invalid payload")
                    await self._finish_tool_turn(
                        event.turn_id,
                        "The Tool call ended with the turn failure.",
                    )
                    terminal_status = payload.error.message
        except CancelledError:
            cancelled = True
            raise
        finally:
            try:
                if not cancelled:
                    for turn_id in observed_tool_turns - self._closed_tool_turns:
                        await self._finish_tool_turn(
                            turn_id,
                            "Tool completion was not reported.",
                        )
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

    async def _request_confirmation(self, request: ConfirmationRequestedPayload) -> None:
        if self._active_confirmation_id is not None:
            self._respond_to_confirmation_if_pending(request.confirmation_id, "declined")
            return
        self._active_confirmation_id = request.confirmation_id
        try:
            dismissed = self.push_screen(
                _ToolConfirmationScreen(request),
                wait_for_dismiss=True,
            )
            decision = await cast(Awaitable[ConfirmationDecision | None], dismissed)
            if decision not in {"approved", "declined"}:
                decision = "declined"
        except BaseException:
            with suppress(Exception):
                self._respond_to_confirmation_if_pending(request.confirmation_id, "declined")
            raise
        else:
            self._respond_to_confirmation_if_pending(request.confirmation_id, decision)
        finally:
            if self._active_confirmation_id == request.confirmation_id:
                self._active_confirmation_id = None

    def _respond_to_confirmation_if_pending(
        self,
        confirmation_id: UUID,
        decision: ConfirmationDecision,
    ) -> bool:
        try:
            self._runtime.conversation.respond_to_confirmation(confirmation_id, decision)
        except ValueError:
            return False
        return True

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

    async def _show_tool_started(self, turn_id: UUID, payload: ToolStartedPayload) -> None:
        key = _ToolRowKey(turn_id, payload.tool_call_id)
        if turn_id in self._closed_tool_turns or key in self._tool_rows:
            return
        await self._mount_tool_row(
            key,
            tool_name=payload.tool_name,
            status="running",
            summary=payload.summary,
        )

    async def _show_tool_completed(self, turn_id: UUID, payload: ToolCompletedPayload) -> None:
        if turn_id in self._closed_tool_turns:
            return
        key = _ToolRowKey(turn_id, payload.tool_call_id)
        state = self._tool_rows.get(key)
        if state is None:
            await self._mount_tool_row(
                key,
                tool_name=payload.tool_name,
                status=payload.status,
                summary=payload.summary,
            )
            return
        if state.status != "running":
            return
        state.status = payload.status
        state.widget.update(_tool_row_content(payload.status, state.tool_name, payload.summary))
        self._scroll_to_latest()

    async def _mount_tool_row(
        self,
        key: _ToolRowKey,
        *,
        tool_name: str,
        status: _ToolRowStatus,
        summary: str,
    ) -> None:
        row = Static(
            _tool_row_content(status, tool_name, summary),
            markup=False,
            classes="tool-row",
        )
        self._tool_rows[key] = _ToolRowState(row, tool_name, status)
        await self.query_one("#conversation-display", _ConversationDisplay).mount(row)
        self._scroll_to_latest()

    async def _finish_tool_turn(self, turn_id: UUID, failure_reason: str) -> None:
        self._closed_tool_turns.add(turn_id)
        changed = False
        for key, state in self._tool_rows.items():
            if key.turn_id != turn_id or state.status != "running":
                continue
            state.status = "error"
            state.widget.update(_tool_row_content("error", state.tool_name, failure_reason))
            changed = True
        if changed:
            self._scroll_to_latest()

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


_FRIENDLY_INITIALISMS = {"cwd": "CWD", "id": "ID", "url": "URL"}


def _friendly_name(name: str, *, fallback: str) -> str:
    words = name.replace("_", " ").split()
    return (
        " ".join(
            _FRIENDLY_INITIALISMS.get(word.casefold(), word[:1].upper() + word[1:])
            for word in words
        )
        or fallback
    )


def _friendly_tool_name(tool_name: str) -> str:
    return _friendly_name(tool_name, fallback="Tool")


def _friendly_parameter_name(name: str) -> str:
    return _friendly_name(name, fallback="Parameter")


def _friendly_parameter_value(value: object) -> str:
    if isinstance(value, dict):
        if not value:
            return "None"
        return "; ".join(
            f"{_friendly_parameter_name(str(name))}: {_friendly_parameter_value(item)}"
            for name, item in value.items()
        )
    if isinstance(value, list):
        if not value:
            return "None"
        return ", ".join(_friendly_parameter_value(item) for item in value)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


_VISIBLE_CONFIRMATION_PARAMETERS: dict[str, tuple[str, ...]] = {
    "read_file": ("path", "offset", "limit"),
    "list_dir": ("path", "recursive", "max_entries"),
    "glob": ("pattern", "path", "kind", "head_limit", "offset"),
    "grep": (
        "pattern",
        "path",
        "glob",
        "type",
        "output_mode",
        "context",
        "head_limit",
        "offset",
    ),
    "web_fetch": ("url", "format"),
}


def _selected_parameter_lines(
    details: Mapping[str, object],
    names: tuple[str, ...],
) -> list[str]:
    lines: list[str] = []
    for name in names:
        if name not in details:
            continue
        value = details[name]
        if name == "url":
            value = _safe_confirmation_url(value)
        lines.append(f"{_friendly_parameter_name(name)}: {_friendly_parameter_value(value)}")
    return lines


def _text_size_line(label: str, value: object) -> str | None:
    if not isinstance(value, str):
        return None
    unit = "character" if len(value) == 1 else "characters"
    return f"{label}: {len(value)} {unit}"


def _safe_confirmation_url(value: object) -> object:
    if not isinstance(value, str):
        return value
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return "Invalid URL"
    if not parsed.scheme or hostname is None:
        return "Invalid URL"
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None:
        host = f"{host}:{port}"
    rendered = f"{parsed.scheme}://{host}{parsed.path}"
    if parsed.query:
        rendered = f"{rendered}?<redacted>"
    return rendered


def _confirmation_detail_lines(request: ConfirmationRequestedPayload) -> tuple[str, ...]:
    details = request.details
    tool_name = request.tool_name.casefold()
    if tool_name == "exec" and "command" in details:
        lines = [f"Command: {_friendly_parameter_value(details['command'])}"]
        for name in ("cwd", "timeout"):
            if name in details:
                lines.append(
                    f"{_friendly_parameter_name(name)}: {_friendly_parameter_value(details[name])}"
                )
        return tuple(lines)
    if tool_name == "write_file":
        lines = _selected_parameter_lines(details, ("path",))
        content_size = _text_size_line("Content", details.get("content"))
        if content_size is not None:
            lines.append(content_size)
        return tuple(lines) or ("Parameters: None",)
    if tool_name == "edit_file":
        lines = _selected_parameter_lines(details, ("path", "replace_all"))
        for label, name in (("Existing Text", "old_text"), ("Replacement Text", "new_text")):
            text_size = _text_size_line(label, details.get(name))
            if text_size is not None:
                lines.append(text_size)
        return tuple(lines) or ("Parameters: None",)
    visible_names = _VISIBLE_CONFIRMATION_PARAMETERS.get(tool_name)
    if visible_names is None:
        return ("Parameters: Not displayed",)
    return tuple(_selected_parameter_lines(details, visible_names)) or ("Parameters: None",)


def _tool_row_content(status: _ToolRowStatus, tool_name: str, summary: str) -> str:
    display_name = _concise_tool_name(tool_name)
    if status == "running":
        return f"Running: {display_name}"
    if status == "success":
        return f"Completed: {display_name}"
    if status == "refused":
        return f"Rejected: {display_name}"
    return f"Failed: {display_name} - {_safe_failure_reason(summary, display_name)}"


def _concise_tool_name(tool_name: str) -> str:
    display_name = " ".join(tool_name.split()) or "Tool"
    if len(display_name) <= _TOOL_NAME_MAX_CHARS:
        return display_name
    return f"{display_name[: _TOOL_NAME_MAX_CHARS - 3].rstrip()}..."


def _safe_failure_reason(summary: str, tool_name: str) -> str:
    detail = " ".join(summary.split())
    if not detail or _UNSAFE_TOOL_DETAIL_PATTERN.search(detail):
        return _GENERIC_TOOL_FAILURE_REASON

    for prefix in ("Failed", "Error", "Finished", "Completed"):
        if detail.casefold().startswith(prefix.casefold()):
            detail = detail[len(prefix) :].lstrip(" :-")
            break
    if detail.casefold().startswith(tool_name.casefold()):
        detail = detail[len(tool_name) :].lstrip(" :-")
    if not detail:
        return _GENERIC_TOOL_FAILURE_REASON
    if len(detail) <= _FAILURE_REASON_MAX_CHARS:
        return detail
    return f"{detail[: _FAILURE_REASON_MAX_CHARS - 3].rstrip()}..."


def run_terminal_conversation(runtime: PreparedReplRuntime) -> None:
    """Run a Terminal Conversation application around a prepared Runtime."""
    TerminalConversationApp(runtime).run()


def _clear_current_task_cancellation() -> None:
    task = current_task()
    if task is None:
        return
    while task.cancelling():
        task.uncancel()
