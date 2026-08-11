# Full-Screen Terminal Conversation Design

Status: confirmed

This document records the agreed user-facing contract for MyClaw's Command-line Conversation. It complements the domain glossary and ADRs; it does not redefine Runtime Core, Conversation Session, or Agent Event semantics.

## Product Boundary

- Running `myclaw` without arguments starts a full-screen terminal UI and no longer starts the plain scrolling REPL.
- The plain REPL is not retained as an alternative user interface. Existing non-interactive Management Commands remain available.
- A non-TTY or a terminal that cannot start the full-screen UI receives a clear error and a nonzero exit status; there is no REPL fallback.
- The UI has exactly two primary regions: a conversation display above and an input area at the bottom. Temporary modal overlays are allowed.

## Conversation Display

- User messages align right and assistant messages align left. Neither role labels nor timestamps are shown.
- An accepted user submission appears immediately on the right without waiting for the first Agent Event.
- Messages use unframed text blocks with subtle backgrounds and a maximum width of about 72 percent. On narrow terminals they expand to the available width; a right-side accent marks user messages and a left-side accent marks assistant messages so roles remain distinguishable without labels or color.
- Assistant content renders as Markdown, including headings, lists, quotations, and code blocks. Incomplete streaming syntax may temporarily render as plain text and reflow as more content arrives.
- Ordinary messages and Markdown links are not clickable. A rendered link includes its URL but the UI never opens a browser.
- Tool activity, Management Command output, failures, and cancellation appear as neutral full-width rows.
- One live Tool row changes from running to its final status instead of appending separate start and completion rows. Tool arguments and full results are hidden by default; errors may show a wrapped reason.
- The display supports scrolling. It follows new content only while already at the bottom. Manual scrolling pauses that behavior and exposes a new-content indicator until the user returns to the bottom.
- The mouse wheel scrolls the conversation display. `PageUp` and `PageDown` move by one viewport, `Ctrl+Home` moves to the earliest content, and `Ctrl+End` returns to the latest content without moving input focus.
- While follow mode is paused, a non-interactive `New content below` state appears at the bottom of the conversation region and disappears after the view returns to the latest content.
- Ordinary prose wraps to the message width. Markdown code blocks preserve their lines and scroll horizontally.
- Ordinary message blocks do not respond to clicks. `Shift` plus drag delegates selection to the host terminal, and MyClaw does not own a clipboard.
- The UI inherits the terminal background and uses a limited adaptive ANSI palette. Low-color and monochrome terminals retain meaning through position, accent placement, and text style.
- A new empty Conversation Session leaves the conversation display blank and focuses the input. No welcome page or shortcut instructions are shown.
- The conversation display has no full-text search in the first version.

## Input Area

- The input is multiline, grows to at most six rows, and then scrolls internally.
- `Enter` submits. Whitespace-only content is not submitted.
- `Shift+Enter` and `Alt+Enter` insert a newline when the host terminal reports either chord distinctly. `Ctrl+J` is the reliable newline fallback.
- Bracketed paste inserts the complete pasted text, including newlines, without submitting it.
- Input history lasts for the current Runtime only and is not persisted. When the input is empty, `Up` and `Down` browse that history.
- Typing `/` offers completion only for `/config`, `/status`, `/resume`, `/memory`, and `/dream`.
- Direction keys always control the topmost open surface. When command completion is open, `Up` and `Down` move through candidates, `Enter` selects, and `Esc` closes it; input history receives `Up` and `Down` only when no overlay or completion is open and the input is empty.
- Interactive completion candidates are clickable.
- The input keeps the primary focus. Conversation scrolling does not move focus away from it.
- The UI does not display the active Session title or model name. Model and Session details remain available through Management Commands.
- During an active foreground turn the input remains visible but read-only, hides its editing cursor, and shows `Working` inside the input region. A second user message cannot be queued.
- During an active turn, `Ctrl+C` cancels that turn. While idle, `Ctrl+C` clears a nonempty draft; with an empty draft it exits the UI. Entering `exit` or `quit` also exits.
- With an empty input, `Ctrl+D` exits. With a nonempty draft it keeps the TextArea's normal delete behavior.

## Confirmation And Commands

- Tool Confirmation uses a blocking modal with one approve-once option and one decline option. Decline is selected initially. It supports mouse clicks, arrow-key selection, and `Enter` to confirm; both `Esc` and `Ctrl+C` decline the Tool call without cancelling the enclosing foreground turn.
- Tool Confirmation presents the confirmation reason and any safety warning without overwhelming the user. For Exec it shows the exact final shell command that will run. For every other Tool it shows a user-friendly Tool name and only the parsed effective parameters needed to judge the operation; it does not show raw JSON or a Tool result.
- `/resume` uses a scrollable selection modal ordered by most recently updated Session. Each clickable option shows only the Session title and updated time formatted in system-local `YYYY-MM-DD HH:mm`; the list has no search or filtering. Selecting a Conversation Session clears the display and rebuilds it from that Session's persisted user, assistant, and final Tool records, then positions the view at the latest content.
- `Esc` closes the `/resume` modal without changing the current Session, draft, or conversation display.
- Clicking outside Tool Confirmation or `/resume` does not close the modal.
- `Ctrl+C` is handled by the topmost surface before global behavior: it declines Tool Confirmation, cancels `/resume`, or closes command completion. Only when no such surface is open does it cancel an active turn, clear a draft, or exit.
- A resumed view does not recreate transient Agent Events such as Tool-start notifications, event timestamps, or pending confirmations. No separate UI event log is added.
- A submitted Management Command appears as a neutral command row followed by its result. `/config`, `/status`, `/memory`, and `/dream` write their results to neutral full-width rows; a successful `/resume` replaces the display and therefore does not retain its command row.
- Partial assistant content remains visible when a turn is cancelled or fails, followed by a neutral `Cancelled` or error row. The UI does not offer a retry action.
- Failures, cancellation, final Tool states, and Management Command output remain in the live display until the Session changes or the UI exits. `Working`, new-content state, completion, and modal overlays are transient.

## Host And Lifecycle

- Textual `>=8.2.8,<9` hosts the full-screen UI. The existing headless `run_repl` seam remains available for Runtime and adapter tests, not as a user-selected UI.
- A narrow enhanced-keyboard adapter detects and enables modifier-aware Enter input where the terminal protocol supports it. Unsupported terminals use `Ctrl+J` for reliable multiline entry.
- Mouse reporting covers conversation-wheel events and Confirmation controls.
- Interactive lists and confirmation options accept clicks. Conversation and code-block scrollbars appear only on overflow and support pointer dragging; ordinary content remains non-interactive.
- When the terminal becomes too small for both primary regions, normal layout pauses and a size-insufficient state is shown. Restoring the window rebuilds the layout and preserves the conversation's scroll anchor.
- Exit, cancellation, startup failure, and unhandled failure must restore terminal state before Runtime shutdown completes.
- UI-owned labels, statuses, buttons, and errors are English. User and model content is shown unchanged.

## Related Decisions

- [Implementation spec: GitHub Issue #131](https://github.com/Totoro-debug/MyClaw/issues/131)
- [ADR-0011: Use a Full-Screen Terminal Conversation](./adr/0011-use-full-screen-terminal-conversation.md)
- [ADR-0012: Use Textual for the Terminal Conversation](./adr/0012-use-textual-for-terminal-conversation.md)
- [Terminal TUI library selection research](./research/terminal-tui-library-selection.md)
