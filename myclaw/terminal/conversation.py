"""Full-screen Textual host for the Terminal Conversation."""

from __future__ import annotations

import re
import sys
from asyncio import CancelledError, Event, Task, create_task, sleep
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from functools import partial
from typing import ClassVar, Final, Literal, Protocol, cast
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
from textual.widgets import Button, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option
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
from myclaw.management.commands import SUPPORTED_MANAGEMENT_COMMANDS, ManagementCommandResult
from myclaw.management.service import SessionListingEntry
from myclaw.terminal._turn_stream import (
    clear_current_task_cancellation,
    close_event_stream,
)
from myclaw.terminal.keyboard import EnhancedKeyboardAction, EnhancedKeyboardAdapter

__all__ = [
    "TerminalConversationApp",
    "TerminalConversationError",
    "is_interactive_terminal",
    "run_terminal_conversation",
]

_COMPACT_MESSAGE_MAX_WIDTH = 60
_MIN_TERMINAL_WIDTH = 20
_MIN_TERMINAL_HEIGHT = 10
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
_TERMINAL_MODE_RESETS: Final = (
    ("\x1b[?2004h", "\x1b[?2004l"),
    ("\x1b[?1000h", "\x1b[?1000l"),
    ("\x1b[?1003h", "\x1b[?1003l"),
    ("\x1b[?1015h", "\x1b[?1015l"),
    ("\x1b[?1006h", "\x1b[?1006l"),
    ("\x1b[?1004h", "\x1b[?1004l"),
    ("\x1b[?1049h", "\x1b[?1049l"),
    ("\x1b[?25l", "\x1b[?25h"),
)
type _ControlAction = Literal["cancel_active_turn", "clear_draft", "exit"]
type _ToolRowStatus = Literal["running", "success", "error", "refused"]


class _DriverLifecycleHooks(Protocol):
    write: Callable[[str], None]
    flush: Callable[[], None]
    start_application_mode: Callable[[], None]
    stop_application_mode: Callable[[], None]


class _ConsoleRestoreHooks(Protocol):
    _restore_console: Callable[[], None] | None


class TerminalConversationError(RuntimeError):
    """A Terminal Conversation cannot run with the supplied terminal streams."""


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
        self._ctrl_c_turn_token: object | None = None

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
        completion = cast(_CommandCompletionHost, self.app)
        if event.key != "ctrl+c":
            self._ctrl_c_turn_token = None
        if event.key in {"up", "down", "left", "right"} and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            if event.key in {"up", "down"}:
                completion.move_command_completion(-1 if event.key == "up" else 1)
            return
        if event.key == "escape" and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            completion.dismiss_command_completion()
            return
        if event.key == "enter" and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            completion.accept_command_completion()
            return
        if event.key == "ctrl+c" and completion.command_completion_visible:
            event.stop()
            event.prevent_default()
            completion.dismiss_command_completion()
            return
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
                if turn_token is not None:
                    self._ctrl_c_turn_token = turn_token
            elif self._ctrl_c_turn_token is not None:
                control_action = "cancel_active_turn"
                turn_token = self._ctrl_c_turn_token
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
        action = EnhancedKeyboardAdapter.parse(event.key)
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


class _CommandCompletionHost(Protocol):
    @property
    def command_completion_visible(self) -> bool: ...

    def move_command_completion(self, direction: int) -> None: ...

    def dismiss_command_completion(self) -> None: ...

    def accept_command_completion(self) -> None: ...


class _CommandCompletion(OptionList):
    class Dismissed(Message):
        pass

    async def _on_key(self, event: Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.Dismissed())
            return
        await super()._on_key(event)


class _SessionPickerScreen(ModalScreen[str | None]):
    """Choose one validated Conversation Session without changing it yet."""

    CSS = """
    _SessionPickerScreen {
        align: center middle;
        padding: 1 2;
    }

    #session-picker-panel {
        width: 80%;
        max-width: 72;
        height: 80%;
        max-height: 90%;
        padding: 1 2;
        border: round $panel;
        background: $surface;
    }

    #session-picker-heading {
        width: 100%;
        margin-bottom: 1;
        text-style: bold;
    }

    .session-picker-notice {
        width: 100%;
        margin-bottom: 1;
        color: $text-muted;
    }

    #session-picker-options {
        width: 100%;
        height: 1fr;
        min-height: 3;
        overflow-y: auto;
    }
    """

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+c", "cancel", "Cancel", show=False, priority=True),
    ]

    def __init__(
        self,
        sessions: tuple[SessionListingEntry, ...],
        *,
        skipped_count: int,
    ) -> None:
        super().__init__(id="session-picker")
        self._sessions = sessions
        self._skipped_count = skipped_count

    def compose(self) -> ComposeResult:
        with Vertical(id="session-picker-panel"):
            yield Static("Resume Session", id="session-picker-heading", markup=False)
            if self._skipped_count:
                noun = "Session" if self._skipped_count == 1 else "Sessions"
                yield Static(
                    f"Skipped {self._skipped_count} corrupt Conversation {noun}.",
                    markup=False,
                    classes="session-picker-notice",
                )
            if not self._sessions:
                yield Static(
                    "No resumable Conversation Sessions.",
                    markup=False,
                    classes="session-picker-notice",
                )
            yield OptionList(
                *(
                    Option(_session_picker_label(session), id=session.id)
                    for session in self._sessions
                ),
                id="session-picker-options",
                markup=False,
            )

    def on_mount(self) -> None:
        self.query_one("#session-picker-options", OptionList).focus()

    @on(OptionList.OptionSelected, "#session-picker-options")
    def _option_selected(self, message: OptionList.OptionSelected) -> None:
        message.stop()
        self.dismiss(message.option_id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class _MarkdownStream(Protocol):
    async def write(self, markdown_fragment: str) -> None: ...

    async def stop(self) -> None: ...


class _CoalescedMarkdownStream:
    """Batch adjacent provider deltas into one Textual Markdown refresh."""

    def __init__(
        self,
        stream: _MarkdownStream,
        *,
        content_changed: Callable[[], None],
    ) -> None:
        self._stream = stream
        self._content_changed = content_changed
        self._pending: list[str] = []
        self._flush_task: Task[None] | None = None
        self._flush_error: BaseException | None = None
        self._stopped = False

    def write(self, fragment: str) -> None:
        if self._stopped:
            raise RuntimeError("Markdown stream is already stopped")
        self._pending.append(fragment)
        if self._flush_task is None and self._flush_error is None:
            self._flush_task = create_task(self._flush_after_event_loop_turn())

    async def stop(self) -> None:
        self._stopped = True
        try:
            if self._flush_task is not None:
                await self._flush_task
            if self._pending and self._flush_error is None:
                await self._flush_pending()
        except BaseException as error:
            self._flush_error = error

        try:
            await self._stream.stop()
        except BaseException as stop_error:
            if self._flush_error is not None:
                raise self._flush_error from stop_error
            raise
        if self._flush_error is not None:
            raise self._flush_error

    async def _flush_after_event_loop_turn(self) -> None:
        try:
            await sleep(0)
            while self._pending:
                await self._flush_pending()
        except BaseException as error:
            self._flush_error = error
        finally:
            self._flush_task = None

    async def _flush_pending(self) -> None:
        fragments = self._pending
        self._pending = []
        await self._stream.write("".join(fragments))
        self._content_changed()


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
        def __init__(self, display: _ConversationDisplay, width: int, generation: int) -> None:
            super().__init__()
            self.display = display
            self.width = width
            self.generation = generation

    _following = True
    _new_content = False
    _last_scroll_state: tuple[bool, bool] | None = None
    _historical_anchor: _ScrollAnchor | None = None
    _resize_anchor: _ScrollAnchor | None = None
    _resize_generation = 0
    _resize_seen = False
    _size_suspended = False
    _size_restore_pending = False
    _resize_restore_pending = False
    _restoring_resize = False
    _suspended_following = True
    _suspended_new_content = False
    _suspended_anchor: _ScrollAnchor | None = None

    @property
    def following(self) -> bool:
        """Whether new content should keep the viewport at the bottom."""
        return self._following

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        self._sync_after_user_scroll()

    def _on_mouse_scroll_up(self, event: MouseScrollUp) -> None:
        self._take_over_resize_restore()
        super()._on_mouse_scroll_up(event)
        self._sync_after_user_scroll()

    def _on_mouse_scroll_down(self, event: MouseScrollDown) -> None:
        self._take_over_resize_restore()
        super()._on_mouse_scroll_down(event)
        self._sync_after_user_scroll()

    def _on_scroll_to(self, message: ScrollTo) -> None:
        self._take_over_resize_restore()
        super()._on_scroll_to(message)
        self.call_after_refresh(self._sync_after_user_scroll)

    def on_resize(self, event: Resize) -> None:
        self._resize_generation += 1
        if not self._size_restore_pending:
            self._restoring_resize = False
        first_resize = not self._resize_seen
        self._resize_seen = True
        if not self._size_suspended and not self._size_restore_pending:
            self._resize_anchor = None if self.following else self._historical_anchor
            if not first_resize:
                self._resize_restore_pending = True
        self.post_message(self.Resized(self, event.size.width, self._resize_generation))

    def restore_resize_anchor(self, generation: int | None = None) -> None:
        if self._size_suspended:
            return
        if generation is None and not (self._size_restore_pending or self._resize_restore_pending):
            return
        if generation is not None and generation != self._resize_generation:
            return
        current_generation = self._resize_generation
        anchor = self._resize_anchor
        self._restoring_resize = True
        if self.following:
            self.scroll_end(
                animate=False,
                immediate=not self._size_restore_pending,
            )

            def follow_latest(attempt: int = 0) -> None:
                if current_generation != self._resize_generation:
                    return
                self.scroll_end(animate=False, immediate=True)
                if attempt < 2:
                    self.call_after_refresh(follow_latest, attempt + 1)
                    return
                self._finish_resize_restore(emit_state=True)

            self.call_after_refresh(follow_latest)
            return
        if anchor is None:
            self._finish_resize_restore()
            return

        def restore(attempt: int = 0) -> None:
            if current_generation != self._resize_generation or self._resize_anchor is not anchor:
                return
            if anchor.widget.parent is not self:
                self._finish_resize_restore()
                return
            target = round(
                anchor.widget.virtual_region.y
                + anchor.relative_position * anchor.widget.virtual_region.height
            )
            self.scroll_to(
                y=target,
                animate=False,
                immediate=True,
            )
            if attempt < 2:
                self.call_after_refresh(restore, attempt + 1)
                return
            self._finish_resize_restore(emit_state=True)

        self.call_after_refresh(restore)

    def schedule_resize_anchor_retry(self, generation: int | None = None) -> None:
        retry_generation = self._resize_generation if generation is None else generation
        self.set_timer(0.001, partial(self.restore_resize_anchor, retry_generation))

    def navigate(self, key: str) -> None:
        self._take_over_resize_restore()
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

    def _take_over_resize_restore(self) -> None:
        if self._size_suspended or not (
            self._size_restore_pending or self._resize_restore_pending or self._restoring_resize
        ):
            return
        self._finish_resize_restore()
        self._resize_generation += 1

    def _finish_resize_restore(self, *, emit_state: bool = False) -> None:
        self._resize_anchor = None
        self._size_restore_pending = False
        self._resize_restore_pending = False
        self._restoring_resize = False
        if emit_state:
            self._emit_scroll_state()

    def content_changed(self) -> None:
        """Follow new content only while the user is at the conversation bottom."""
        if self._size_suspended or self._size_restore_pending:
            if not self._suspended_following:
                self._suspended_new_content = True
            self._emit_scroll_state()
            return
        if self.following:
            self.scroll_end(animate=False, immediate=True)
            self._following = True
            self._new_content = False
            self._historical_anchor = None
            self.call_after_refresh(self._follow_latest)
        else:
            self._new_content = True
        self._emit_scroll_state()

    def reset_to_latest(self) -> None:
        """Start a replaced Session at its latest persisted content."""
        if self._size_suspended:
            self._suspended_following = True
            self._suspended_new_content = False
            self._suspended_anchor = None
        self._size_restore_pending = False
        self._resize_restore_pending = False
        self._restoring_resize = False
        self._following = True
        self._new_content = False
        self._historical_anchor = None
        self._resize_anchor = None
        self._resize_generation += 1
        self.scroll_end(animate=False, immediate=True)
        self._following = True
        self.call_after_refresh(self._follow_latest)
        self._emit_scroll_state()

    def _follow_latest(self, attempt: int = 0) -> None:
        if not self.following:
            return
        self.scroll_end(animate=False, immediate=True)
        self._following = True
        self._new_content = False
        self._historical_anchor = None
        self._emit_scroll_state()
        if attempt < 2:
            self.call_after_refresh(self._follow_latest, attempt + 1)

    def _sync_after_user_scroll(self) -> None:
        if self._size_suspended or self._restoring_resize:
            return
        if self._size_restore_pending:
            self._size_restore_pending = False
            self._resize_anchor = None
            self._resize_generation += 1
        elif self._resize_restore_pending:
            self._resize_restore_pending = False
            self._resize_anchor = None
            self._resize_generation += 1
        self._following = self.is_vertical_scroll_end
        if self._following:
            self._new_content = False
            self._historical_anchor = None
        else:
            self._historical_anchor = self._capture_scroll_anchor()
        self._emit_scroll_state()

    def suspend_for_size(self) -> None:
        """Freeze the visible scroll state while the normal layout is hidden."""
        if self._size_suspended:
            return
        self._size_suspended = True
        self._resize_restore_pending = False
        self._restoring_resize = False
        self._suspended_following = self._following
        self._suspended_new_content = self._new_content
        self._suspended_anchor = (
            None if self._following else (self._resize_anchor or self._historical_anchor)
        )
        self._resize_anchor = self._suspended_anchor
        self._resize_generation += 1

    def resume_from_size(self) -> None:
        """Restore the scroll state captured before the layout was hidden."""
        if not self._size_suspended:
            return
        self._size_suspended = False
        self._size_restore_pending = True
        self._resize_restore_pending = False
        self._restoring_resize = False
        self._following = self._suspended_following
        self._new_content = self._suspended_new_content
        self._historical_anchor = self._suspended_anchor
        self._resize_anchor = None if self._following else self._suspended_anchor
        self._resize_generation += 1
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


class _SizeInsufficientScreen(ModalScreen[None]):
    """Block all interaction until the terminal can render the application."""

    CSS = """
    _SizeInsufficientScreen {
        align: center middle;
        background: $background;
    }

    #size-insufficient-modal {
        width: 100%;
        height: 100%;
        padding: 1 2;
        content-align: center middle;
        text-align: center;
        background: $background;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static(
            "Terminal window is too small. Resize to continue.",
            id="size-insufficient-modal",
            markup=False,
        )

    async def _on_key(self, event: Key) -> None:
        event.stop()
        event.prevent_default()


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

    def restore_after_size(self) -> None:
        """Reveal the confirmation heading after a constrained background layout."""
        self.query_one("#confirmation-panel", Vertical).scroll_home(
            animate=False,
            immediate=True,
        )

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

    #command-completion {
        display: none;
        overlay: screen;
        offset: 0 -7;
        width: 100%;
        max-height: 7;
        background: transparent;
        border: round $panel;
    }

    .management-row {
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
        min-height: 3;
        max-height: 8;
        width: 100%;
    }

    #size-insufficient {
        display: none;
        width: 100%;
        height: 1fr;
        padding: 1 2;
        align: center middle;
        content-align: center middle;
        text-align: center;
        background: transparent;
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
        self._runtime_started = False
        self._size_insufficient = False
        self._driver_mode_started = False
        self._driver_mode_stopped = True
        self._viable_size = Event()
        self._viable_size.set()
        self._size_screen: _SizeInsufficientScreen | None = None
        self._turn_worker: Worker[None] | None = None
        self._resume_worker: Worker[None] | None = None
        self._cancel_requested_turn: object | None = None
        self._tool_rows: dict[_ToolRowKey, _ToolRowState] = {}
        self._closed_tool_turns: set[UUID] = set()
        self._active_confirmation_id: UUID | None = None
        self._completion_options: tuple[str, ...] = ()
        self._completion_dismissed_text: str | None = None
        self._closing = False
        self._application_error: Exception | None = None

    def _handle_exception(self, error: Exception) -> None:
        if self._application_error is None:
            self._application_error = error
        super()._handle_exception(error)

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
        EnhancedKeyboardAdapter.install_on_driver(driver)
        self._install_driver_lifecycle(driver)
        return driver

    def _install_driver_lifecycle(self, driver: Driver) -> None:
        hooks = cast(_DriverLifecycleHooks, driver)
        original_write = hooks.write
        original_flush = hooks.flush
        original_start = hooks.start_application_mode
        original_stop = hooks.stop_application_mode
        self._driver_mode_started = False
        self._driver_mode_stopped = True
        active_mode_resets: set[str] = set()

        def write(value: str) -> None:
            original_write(value)
            transitions: list[tuple[int, str, bool]] = []
            for enable, reset in _TERMINAL_MODE_RESETS:
                enable_at = value.find(enable)
                if enable_at >= 0:
                    transitions.append((enable_at, reset, True))
                reset_at = value.find(reset)
                if reset_at >= 0:
                    transitions.append((reset_at, reset, False))
            for _, reset, enabled in sorted(transitions):
                if enabled:
                    active_mode_resets.add(reset)
                else:
                    active_mode_resets.discard(reset)

        def restore_terminal_modes() -> None:
            if active_mode_resets:
                for _, reset in _TERMINAL_MODE_RESETS:
                    if reset in active_mode_resets:
                        write(reset)
                original_flush()

            restore_console = getattr(driver, "_restore_console", None)
            if callable(restore_console):
                restore_console()
                cast(_ConsoleRestoreHooks, driver)._restore_console = None

        def stop_application_mode() -> None:
            if not self._driver_mode_started or self._driver_mode_stopped:
                return
            primary_error = sys.exception()
            try:
                original_stop()
            except BaseException as stop_error:
                try:
                    restore_terminal_modes()
                except BaseException as restore_error:
                    stop_error.__cause__ = restore_error
                else:
                    self._driver_mode_stopped = True
                if primary_error is not None:
                    if stop_error is primary_error:
                        raise
                    raise primary_error from stop_error
                raise stop_error
            try:
                restore_terminal_modes()
            except BaseException as cleanup_error:
                if primary_error is not None:
                    raise primary_error from cleanup_error
                raise
            self._driver_mode_stopped = True

        def start_application_mode() -> None:
            self._driver_mode_started = True
            self._driver_mode_stopped = False
            try:
                original_start()
            except BaseException as primary_error:
                try:
                    stop_application_mode()
                except BaseException as cleanup_error:
                    if cleanup_error is primary_error:
                        raise
                    raise primary_error from cleanup_error
                raise

        hooks.write = write
        hooks.start_application_mode = start_application_mode
        hooks.stop_application_mode = stop_application_mode

    def compose(self) -> ComposeResult:
        yield _ConversationDisplay(id="conversation-display")
        yield Vertical(
            _CommandCompletion(
                id="command-completion",
                markup=False,
                compact=True,
            ),
            Static("New content below", id="new-content", markup=False),
            Static("Working", id="turn-status", markup=False),
            _ConversationInput(id="conversation-input", placeholder="Message MyClaw"),
            id="conversation-input-region",
        )
        yield Static(
            "Terminal window is too small. Resize to continue.",
            id="size-insufficient",
            markup=False,
        )

    async def on_mount(self) -> None:
        self._runtime_started = True
        await self._runtime.start()
        if not self._size_insufficient:
            self.query_one(_ConversationInput).focus()

    async def on_unmount(self, event: Unmount) -> None:
        del event
        self._closing = True
        cleanup_errors: list[BaseException] = []
        try:
            if self._turn_worker is not None:
                self._turn_worker.cancel()
                with suppress(WorkerError):
                    await self._turn_worker.wait()
            if self._resume_worker is not None:
                self._resume_worker.cancel()
                with suppress(WorkerError):
                    await self._resume_worker.wait()
        except BaseException as worker_error:
            cleanup_errors.append(worker_error)
        finally:
            if self._runtime_started:
                self._runtime_started = False
                try:
                    await self._runtime.close()
                except BaseException as runtime_error:
                    cleanup_errors.append(runtime_error)

        primary_error = self._application_error
        if primary_error is not None and cleanup_errors:
            causes: list[BaseException] = []
            if primary_error.__cause__ is not None:
                causes.append(primary_error.__cause__)
            causes.extend(cleanup_errors)
            unique_causes: list[BaseException] = []
            for error in causes:
                if not any(error is existing for existing in unique_causes):
                    unique_causes.append(error)
            cause: BaseException = (
                unique_causes[0]
                if len(unique_causes) == 1
                else BaseExceptionGroup("Terminal Conversation cleanup failed", unique_causes)
            )
            raise primary_error from cause
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise cleanup_errors[0] from BaseExceptionGroup(
                "Additional Terminal Conversation cleanup failures",
                cleanup_errors[1:],
            )

    @property
    def command_completion_visible(self) -> bool:
        return bool(self._completion_options)

    @on(TextArea.Changed, "#conversation-input")
    def _input_changed(self, message: TextArea.Changed) -> None:
        if isinstance(message.text_area, _ConversationInput):
            message.text_area._ctrl_c_turn_token = None
        self._refresh_command_completion(message.text_area.text)

    @on(_CommandCompletion.OptionSelected)
    def _completion_selected(self, message: _CommandCompletion.OptionSelected) -> None:
        if message.option_list.id != "command-completion":
            return
        self._select_command_completion(message.option_index)

    @on(_CommandCompletion.Dismissed)
    def _completion_dismissed(self, message: _CommandCompletion.Dismissed) -> None:
        message.stop()
        self.dismiss_command_completion()

    def on_resize(self, event: Resize) -> None:
        too_small = (
            event.size.width < _MIN_TERMINAL_WIDTH or event.size.height < _MIN_TERMINAL_HEIGHT
        )
        if too_small == self._size_insufficient:
            return

        display = self.query_one("#conversation-display", _ConversationDisplay)
        if too_small:
            display.suspend_for_size()
        self._size_insufficient = too_small
        if too_small:
            self._viable_size.clear()
        else:
            self._viable_size.set()
        input_region = self.query_one("#conversation-input-region", Vertical)
        size_state = self.query_one("#size-insufficient", Static)
        display.display = not too_small
        input_region.display = not too_small
        size_state.display = too_small
        if too_small:
            size_screen = _SizeInsufficientScreen()
            self._size_screen = size_screen
            self.push_screen(size_screen)
            return

        active_size_screen = self._size_screen
        self._size_screen = None
        if active_size_screen is not None and active_size_screen is self.screen:
            active_size_screen.dismiss()
        if not too_small and not self._closing:
            self.refresh(layout=True)
            display.resume_from_size()
            display.restore_resize_anchor()
            # Let the layout-driven scroll range settle before the final retry.
            display.schedule_resize_anchor_retry()
            self.call_after_refresh(self._restore_input_focus_after_size)

    def _restore_input_focus_after_size(self) -> None:
        if self._size_insufficient or self._closing:
            return
        self.screen.refresh(layout=True)
        if isinstance(self.screen, _ToolConfirmationScreen):
            self.screen.restore_after_size()
        if len(self.screen_stack) != 1:
            return
        with suppress(Exception):
            self.query_one(_ConversationInput).focus()

    def move_command_completion(self, direction: int) -> None:
        if not self._completion_options:
            return
        completion = self.query_one("#command-completion", _CommandCompletion)
        highlighted = completion.highlighted
        current = 0 if highlighted is None else highlighted
        completion.highlighted = max(
            0,
            min(len(self._completion_options) - 1, current + direction),
        )

    def dismiss_command_completion(self) -> None:
        input_area = self.query_one("#conversation-input", _ConversationInput)
        self._hide_command_completion(remember_text=input_area.text)
        input_area.focus()

    def accept_command_completion(self) -> None:
        if not self._completion_options:
            return
        completion = self.query_one("#command-completion", _CommandCompletion)
        highlighted = completion.highlighted
        index = 0 if highlighted is None else highlighted
        selected = self._completion_options[index]
        input_area = self.query_one("#conversation-input", _ConversationInput)
        should_submit = input_area.text == selected
        self._select_command_completion(index)
        if should_submit:
            self.post_message(_ConversationInput.Submitted(input_area, selected))

    def _select_command_completion(self, index: int) -> None:
        if not self._completion_options:
            return
        selected = self._completion_options[index]
        input_area = self.query_one("#conversation-input", _ConversationInput)
        input_area.text = selected
        input_area.move_cursor(
            (len(input_area.document.lines) - 1, len(input_area.document.lines[-1]))
        )
        self._hide_command_completion(remember_text=selected)
        input_area.focus()

    def _refresh_command_completion(self, text: str) -> None:
        if text == self._completion_dismissed_text:
            self._hide_command_completion()
            return
        self._completion_dismissed_text = None
        candidates = _management_command_candidates(text)
        if not candidates:
            self._hide_command_completion()
            return
        completion = self.query_one("#command-completion", _CommandCompletion)
        completion.set_options(candidates)
        completion.highlighted = 0
        completion.display = True
        self._completion_options = candidates

    def _hide_command_completion(self, *, remember_text: str | None = None) -> None:
        self._completion_options = ()
        if remember_text is not None:
            self._completion_dismissed_text = remember_text
        with suppress(NoMatches, NoScreen, ScreenStackError):
            completion = self.query_one("#command-completion", _CommandCompletion)
            completion.set_options(())
            completion.display = False

    @on(_ConversationDisplay.Resized)
    def _display_resized(self, message: _ConversationDisplay.Resized) -> None:
        compact = message.width <= _COMPACT_MESSAGE_MAX_WIDTH
        for content in message.display.query(".message"):
            content.set_class(compact, "message-compact")
        if not self._size_insufficient:
            message.display.restore_resize_anchor(message.generation)
            message.display.schedule_resize_anchor_retry(message.generation)

    @on(_ConversationInput.Submitted)
    async def _submit_input(self, message: _ConversationInput.Submitted) -> None:
        text = message.text
        if (
            not text.strip()
            or message.text_area.read_only
            or self._size_insufficient
            or (self._resume_worker is not None and not self._resume_worker.is_finished)
        ):
            return
        if text.strip().casefold() in {"exit", "quit"}:
            message.text_area.text = ""
            self.exit()
            return

        dispatcher = self._runtime.management_dispatcher
        if dispatcher is not None:
            result = cast(ManagementCommandResult, await dispatcher.dispatch(text))
            if result.handled:
                message.text_area.remember_submission(text)
                message.text_area.text = ""
                if result.resume_sessions is not None:
                    await self._open_resume_picker(
                        result.resume_sessions,
                        message.text_area,
                        skipped_count=result.resume_skipped_count,
                    )
                else:
                    await self._mount_management_rows(text, result.output)
                return

        message.text_area.remember_submission(text)
        message.text_area.text = ""
        turn_token = object()
        message.text_area.active_turn_token = turn_token
        message.text_area.read_only = True
        self._set_working(True)
        display = self.query_one("#conversation-display", _ConversationDisplay)
        await self._mount_user_message(text, display)
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
            clear_current_task_cancellation()
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
        stream: _CoalescedMarkdownStream | None = None
        streamed_fragments: list[str] = []
        observed_tool_turns: set[UUID] = set()
        resolved_confirmations: set[UUID] = set()
        terminal_content: str | None = None
        terminal_status: str | None = None
        cancelled = False
        primary_error: BaseException | None = None
        try:
            async for event in events:
                if event.type == "text_delta":
                    payload = event.payload
                    if not isinstance(payload, TextDeltaPayload):
                        raise TypeError("text_delta event has an invalid payload")
                    streamed_fragments.append(payload.delta)
                    if assistant is None:
                        assistant = await self._mount_assistant()
                        stream = _CoalescedMarkdownStream(
                            Markdown.get_stream(assistant),
                            content_changed=self._scroll_to_latest,
                        )
                    assert stream is not None
                    stream.write(payload.delta)
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
        except BaseException as error:
            primary_error = error
            cancelled = isinstance(error, CancelledError)

        cleanup_errors: list[BaseException] = []
        if not cancelled and not self._closing and primary_error is None:
            try:
                for turn_id in observed_tool_turns - self._closed_tool_turns:
                    await self._finish_tool_turn(
                        turn_id,
                        "Tool completion was not reported.",
                    )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if stream is not None:
            try:
                await stream.stop()
            except BaseException as cleanup_error:
                if not cancelled:
                    cleanup_errors.append(cleanup_error)
        final_content = (
            terminal_content if terminal_content is not None else "".join(streamed_fragments)
        )
        if (
            not cancelled
            and not self._closing
            and primary_error is None
            and not cleanup_errors
            and (final_content or (assistant is not None and terminal_content is not None))
        ):
            try:
                if assistant is None:
                    assistant = await self._mount_assistant()
                await assistant.update(final_content)
                self._scroll_to_latest()
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if (
            not cancelled
            and not self._closing
            and primary_error is None
            and not cleanup_errors
            and terminal_status is not None
        ):
            try:
                await self._mount_status(terminal_status)
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            await close_event_stream(events)
        except BaseException as cleanup_error:
            if not cancelled:
                cleanup_errors.append(cleanup_error)

        if primary_error is not None:
            if cleanup_errors:
                cause: BaseException = (
                    cleanup_errors[0]
                    if len(cleanup_errors) == 1
                    else BaseExceptionGroup("Turn cleanup failed", cleanup_errors)
                )
                raise primary_error from cause
            raise primary_error
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise cleanup_errors[0] from BaseExceptionGroup(
                "Additional turn cleanup failures",
                cleanup_errors[1:],
            )

    async def _request_confirmation(self, request: ConfirmationRequestedPayload) -> None:
        if self._active_confirmation_id is not None:
            self._respond_to_confirmation_if_pending(request.confirmation_id, "declined")
            return
        self._active_confirmation_id = request.confirmation_id
        try:
            await self._viable_size.wait()
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

    async def _mount_user_message(
        self,
        content: str,
        display: _ConversationDisplay,
    ) -> None:
        row = Horizontal(classes="message-row user-row")
        await display.mount(row)
        await row.mount(
            Static(
                content,
                markup=False,
                classes=self._message_classes("user-message", display),
            )
        )

    async def _mount_assistant(
        self,
        content: str = "",
        display: _ConversationDisplay | None = None,
    ) -> Markdown:
        if display is None:
            display = self.query_one("#conversation-display", _ConversationDisplay)
        assistant = Markdown(
            content,
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
        display = self.query_one("#conversation-display", _ConversationDisplay)
        row = await self._mount_tool_message(tool_name, status, summary, display)
        self._tool_rows[key] = _ToolRowState(row, tool_name, status)
        self._scroll_to_latest()

    async def _mount_tool_message(
        self,
        tool_name: str,
        status: _ToolRowStatus,
        summary: str,
        display: _ConversationDisplay,
    ) -> Static:
        row = Static(
            _tool_row_content(status, tool_name, summary),
            markup=False,
            classes="tool-row",
        )
        await display.mount(row)
        return row

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

    async def _mount_management_rows(self, command: str, output: str | None) -> None:
        display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.mount(
            Static(
                f"Command: {command}",
                markup=False,
                classes="management-row",
            )
        )
        if output is not None:
            await self._mount_management_output(output, scroll=False)
        self._scroll_to_latest()

    async def _mount_management_output(self, output: str, *, scroll: bool = True) -> None:
        display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.mount(Static(output, markup=False, classes="management-row"))
        if scroll:
            self._scroll_to_latest()

    async def _replace_display_from_session(self, expected_session_id: str) -> bool:
        authority = self._runtime.session
        if authority.session_id != expected_session_id:
            return False
        projected_messages = tuple(
            (message, *_persisted_role_and_content(message)) for message in authority.messages
        )
        if self._runtime.session is not authority:
            return False

        display = self.query_one("#conversation-display", _ConversationDisplay)
        await display.remove_children()
        self._tool_rows.clear()
        self._closed_tool_turns.clear()
        for message, role, content in projected_messages:
            await self._mount_persisted_message(message, role, content, display)
        display.reset_to_latest()
        return self._runtime.session is authority

    async def _mount_persisted_message(
        self,
        message: Mapping[str, object],
        role: str,
        content: str,
        display: _ConversationDisplay,
    ) -> None:
        if role == "user":
            await self._mount_user_message(content, display)
            return
        if role == "assistant":
            if content:
                await self._mount_assistant(content, display)
            status = _persisted_assistant_status(message)
            if status is not None:
                await display.mount(Static(status, markup=False, classes="turn-status"))
            return
        if role == "tool":
            await self._mount_tool_message(
                cast(str, message["name"]),
                cast(_ToolRowStatus, message["status"]),
                content,
                display,
            )

    async def _open_resume_picker(
        self,
        sessions: tuple[SessionListingEntry, ...],
        input_area: _ConversationInput,
        *,
        skipped_count: int,
    ) -> None:
        await self._viable_size.wait()
        await self.push_screen(
            _SessionPickerScreen(sessions, skipped_count=skipped_count),
            callback=lambda session_id: self._resume_picker_dismissed(
                session_id,
                input_area,
            ),
        )

    def _resume_picker_dismissed(
        self,
        session_id: str | None,
        input_area: _ConversationInput,
    ) -> None:
        if session_id is None:
            with suppress(Exception):
                input_area.focus()
            return
        if self._resume_worker is not None and not self._resume_worker.is_finished:
            return
        input_area.read_only = True
        self._resume_worker = self.run_worker(
            self._resume_selected_session(session_id, input_area),
            name="resume-session",
            group="resume-session",
            exclusive=False,
            exit_on_error=False,
        )

    async def _resume_selected_session(
        self,
        session_id: str,
        input_area: _ConversationInput,
    ) -> None:
        try:
            dispatcher = self._runtime.management_dispatcher
            if dispatcher is None:
                await self._mount_management_rows("/resume", "Session resume is unavailable.")
                return
            try:
                result = cast(ManagementCommandResult, await dispatcher.resume(session_id))
            except Exception:
                await self._mount_management_rows("/resume", "Session resume failed.")
                return
            resumed_session_id = result.resumed_session_id
            if resumed_session_id != session_id:
                await self._mount_management_rows("/resume", result.output)
                return
            authority = self._runtime.session
            if authority.session_id != resumed_session_id:
                await self._mount_management_rows(
                    "/resume",
                    "Session resume did not select the requested Conversation Session.",
                )
                return
            if not await self._replace_display_from_session(resumed_session_id):
                await self._mount_management_rows(
                    "/resume",
                    "Conversation Session authority changed before display replacement.",
                )
        finally:
            input_area.read_only = False
            if not self._closing:
                with suppress(Exception):
                    input_area.focus()

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


def _management_command_candidates(text: str) -> tuple[str, ...]:
    if not text.startswith("/") or "\n" in text or " " in text or "\t" in text:
        return ()
    return tuple(command for command in SUPPORTED_MANAGEMENT_COMMANDS if command.startswith(text))


def _session_picker_label(session: SessionListingEntry) -> str:
    local_updated_at = session.updated_at.astimezone().strftime("%Y-%m-%d %H:%M")
    return f"{session.title} | {local_updated_at}"


def _persisted_role_and_content(message: Mapping[str, object]) -> tuple[str, str]:
    role = message.get("role")
    content = message.get("content")
    if role not in {"user", "assistant", "tool"}:
        raise TypeError("Unsupported persisted Session message role")
    if not isinstance(content, str):
        raise TypeError("Persisted Session message content must be a string")
    if role == "assistant" and message.get("status") not in {
        "completed",
        "interrupted",
        "error",
    }:
        raise TypeError("Persisted Assistant message is malformed")
    if role == "tool" and (
        not isinstance(message.get("name"), str)
        or message.get("status") not in {"success", "error", "refused"}
    ):
        raise TypeError("Persisted Tool message is malformed")
    return role, content


def _persisted_assistant_status(message: Mapping[str, object]) -> str | None:
    status = message.get("status")
    if status == "interrupted":
        return "Turn cancelled."
    if status != "error":
        return None
    error = message.get("error")
    if isinstance(error, dict):
        detail = error.get("message")
        if isinstance(detail, str) and detail:
            return detail
    return "Turn failed."


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


def _stream_is_tty(stream: object) -> bool:
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def is_interactive_terminal() -> bool:
    """Return whether the standard streams can support a full-screen conversation."""

    def stream_is_interactive(current: object, driver_stream: object | None) -> bool:
        return _stream_is_tty(current) and _stream_is_tty(driver_stream)

    return all(
        (
            stream_is_interactive(sys.stdin, sys.__stdin__),
            stream_is_interactive(sys.stdout, sys.__stdout__),
            stream_is_interactive(sys.stderr, sys.__stderr__),
        )
    )


def run_terminal_conversation(runtime: PreparedReplRuntime) -> None:
    """Run a Terminal Conversation application around a prepared Runtime."""
    if not is_interactive_terminal():
        raise TerminalConversationError(
            "Terminal Conversation requires interactive stdin, stdout, and stderr TTYs."
        )
    TerminalConversationApp(runtime).run()
