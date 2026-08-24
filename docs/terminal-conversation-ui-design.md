# Full-Screen Terminal Conversation Design

Status: confirmed

This document records the agreed user-facing contract for MyClaw's Command-line Conversation. It complements the domain glossary and ADRs; it does not redefine Agent Loop, Conversation Session, Agent Runner, or the sparse Message Bus protocol.

## Product Boundary

- Running `myclaw` without arguments starts a full-screen terminal UI and no longer starts the plain scrolling REPL.
- The plain REPL is not retained as an alternative user interface. Existing non-interactive Management Commands remain available.
- A non-TTY or a terminal that cannot start the full-screen UI receives a clear error and a nonzero exit status; there is no REPL fallback.
- The UI has exactly two primary regions: a conversation display above and an input area at the bottom. Temporary modal overlays are allowed.

## Conversation Display

- User messages align right and assistant messages align left. Neither role labels nor timestamps are shown.
- An accepted user submission appears immediately on the right without waiting for the first outbound MessageBus item.
- Messages use unframed text blocks with subtle backgrounds and a maximum width of about 72 percent. On narrow terminals they expand to the available width; a right-side accent marks user messages and a left-side accent marks assistant messages so roles remain distinguishable without labels or color.
- Assistant content renders as Markdown, including headings, lists, quotations, and code blocks. Incomplete streaming syntax may temporarily render as plain text and reflow as more content arrives.
- Ordinary messages and Markdown links are not clickable. A rendered link includes its URL but the UI never opens a browser. The activity-group heading is the only clickable disclosure surface inside run content; clicking its Markdown, Tool rows, or scrollbars does not toggle the group.
- Tool activity, Management Command output, failures, and cancellation appear as neutral full-width rows.
- One live Tool row changes from running to its final status instead of appending separate start and completion rows. The running Tool row displays the Tool name and complete raw Provider argument text; Tool results never enter the foreground Message Bus or terminal activity stream, and errors may show a wrapped safe reason.
- Each foreground Agent Run has one activity group for the model and Tool work that precedes its final output. The group contains intermediate model output and Tool-call rows, including the complete raw arguments required by the Outbound contract; it never exposes Tool results, an Exec result, stdout, or stderr.
- Every model call initially reuses the existing streamed Markdown presentation and delta-coalescing behavior in the normal conversation display. The AgentLoop emits a response `_stream_end` marker when that streamed segment is sealed; if the run then emits a `tool_call`, the sealed assistant message moves into the Agent Run activity group before Tool activity is added, while a run that reaches the terminal `_streamed` marker remains outside the group as the final assistant output. Provider-specific response details remain behind AgentRunner and the Model Router.
- A response `_stream_end` marker is emitted before any corresponding `tool_call` output. It is nonterminal and does not change the Agent Run contract of exactly one terminal `_streamed` marker. The accumulated streamed text remains the candidate presented by the UI until the terminal outcome resolves the run.
- Segment-end and Tool-call metadata expose no Provider response details or usage. The terminal `model_response` `_streamed` marker is the authoritative successful completion; a `system_control` `_streamed` marker is the authoritative failure or cancellation outcome.
- When a response segment is followed by a `tool_call`, the UI immediately moves any nonempty candidate output into the activity group rather than waiting for another model delta. An empty model segment does not create a visible group by itself; if the Run fails before Tool activity appears, only the external failure status is shown.
- After an activity group exists, each later model call still streams as a candidate final assistant message in the normal display after the group. If it continues with Tools, that same widget moves to the end of the group before the new Tool rows are added.
- Production outbound order guarantees response `_stream_end` before corresponding `tool_call` output. The UI nevertheless tolerates a missing or out-of-order marker from an abnormal or test source: a `tool_call` moves any current candidate into the group, and a terminal `_streamed` marker resolves any candidate still in the normal display according to the terminal outcome.
- Response segment markers belong to the lane-neutral Agent Run execution and are projected on the MessageBus for the foreground consumer. Schedule and internal headless consumers do not render nonterminal foreground output.
- Activity-group entries preserve occurrence order across intermediate model output and Tool calls. Each Tool call keeps one row that updates in place from running to its terminal status.
- Intermediate model output keeps the existing assistant Markdown width and left accent after it moves into the activity group, while Tool rows keep their neutral full-width presentation. The activity group adds only its disclosure heading and content container, with no surrounding border, card background, or nested scroll area; expanded content participates directly in the main conversation scroll.
- Sealed non-final assistant widgets move from the normal conversation display into the activity group without duplication or animation. Reparenting preserves the display's current follow or historical scroll anchor.
- A run that produces no intermediate model output and no Tool activity does not render an empty activity group; it displays only the final assistant message.
- A model call that requests Tools but produces no text does not add an empty assistant item to the activity group; its Tool activity still creates and populates the group.
- An Agent Run activity group remains expanded while the run is active. On successful completion, the final model output is rendered as a separate assistant message outside the group and the group then collapses automatically. Failed and cancelled runs keep their activity groups expanded, with their partial content and terminal status visible.
- Each Run is presented in the fixed order user message, optional activity group, then final assistant output or terminal status. The activity group is inserted before the final outcome even though it may not be created until the first model call is identified as non-final.
- The activity-group heading displays the Agent Run's execution time. Live timing uses a monotonic clock beginning when AgentLoop accepts the inbound message, includes time awaiting Tool Confirmation, updates once per second using elapsed whole seconds, and freezes when the run reaches its first terminal `_streamed` outbound; historical estimates use persisted message timestamps. It does not display a Tool count or terminal-state label. Formatting is `59s`, `1min 0s`, `1min 5s`, or `1h 0min 5s` as applicable; hours accumulate beyond 24 and lower-order fields remain present once their unit has been introduced.
- If the group first appears after the Run has already been active, its first heading value uses the full elapsed time since AgentLoop accepted the inbound message rather than restarting from zero.
- A terminal `_streamed` outbound recomputes elapsed whole seconds immediately before freezing the heading, so the frozen duration is not limited to the last periodic refresh. The first terminal marker is authoritative; later terminal markers from an abnormal source are ignored.
- Timing continues from the monotonic clock while the group is outside the viewport, hidden by the size-insufficient state, or covered by Tool Confirmation. Repainting may wait, but the next visible or terminal update uses the correct accumulated duration.
- The heading includes a disclosure symbol: `▼ 12s` while expanded and `▶ 12s` while collapsed. The whole heading is clickable but is not keyboard-focusable and has no keyboard toggle behavior. While the Agent Run is active, the group is forced open and clicking its heading cannot collapse it; mouse toggling becomes available only after a terminal `_streamed` outbound.
- Expanded and collapsed headings contain only the disclosure symbol and formatted duration, with no Tool count, terminal state, or content preview. Each discrete click toggles once; the UI defines no separate double-click behavior or debounce policy.
- Durations shorter than one second display as `0s`; the UI does not round them up.
- A user's manual expanded or collapsed choice remains in effect while the current display remains mounted. Only the group's own first successful completion triggers automatic collapse; later Agent Runs do not reset an older group's state. Rebuilding a Session through `/resume` restores complete successful groups as collapsed.
- Collapsing or expanding an activity group follows the conversation display's existing scroll contract. At the latest content the display continues following the final output; while the user is reading history, it preserves the visual anchor and reports new content below instead of jumping.
- Moving a candidate message into the activity group is treated as a conversation content change for scrolling and the existing `New content below` state; the display does not distinguish movement from newly appended content.
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
- During an active foreground turn the input remains visible and editable. Enter submits each ordinary message to the Message Bus Inbound FIFO; pending messages appear only in a dedicated queue display while the current Agent Run continues. The `Working` status does not make the input read-only.
- During an active turn, `Ctrl+C` cancels that turn. While idle, `Ctrl+C` clears a nonempty draft; with an empty draft it exits the UI. Entering `exit` or `quit` also exits.
- With an empty input, `Ctrl+D` exits. With a nonempty draft it keeps the TextArea's normal delete behavior.

## Pending Input Queue

- The Terminal Conversation is the sole consumer of Message Bus Outbound and the queue
  display is driven by the Inbound snapshot callback. The consumed head is removed from
  the pending display and promoted to the conversation display when Agent Loop accepts it;
  the currently executing message is never returned to input.
- `Up` with non-empty input keeps normal TextArea behavior. `Up` with empty input and a
  non-empty pending queue atomically calls `drain_inbound()`, discards the old metadata,
  joins all drained contents with one newline in FIFO order, and places one editable value
  in the input. A later Enter submits exactly one fresh empty-metadata Inbound Message.
- `Up` with both empty input and an empty pending queue retains input-history navigation.
  Management Commands, Tool Confirmation decisions, cancellation, and Session replacement
  remain separate control paths and never become Inbound Messages.

## Confirmation And Commands

- Tool Confirmation uses a blocking modal with one approve-once option and one decline option. Decline is selected initially. It supports mouse clicks, arrow-key selection, and `Enter` to confirm; both `Esc` and `Ctrl+C` decline the Tool call without cancelling the enclosing foreground turn.
- Tool Confirmation presents the confirmation reason and any safety warning without overwhelming the user. For Exec it shows the exact final shell command that will run. For every other Tool it shows a user-friendly Tool name and only the parsed effective parameters needed to judge the operation; it does not show raw JSON or a Tool result.
- Resolving Tool Confirmation does not append a separate approval or refusal record to the activity group. The Tool row's eventual completed, failed, or rejected status remains the durable visible outcome.
- `/resume` uses a scrollable selection modal ordered by most recently updated Session. Each clickable option shows only the Session title and updated time formatted in system-local `YYYY-MM-DD HH:mm`; the list has no search or filtering. Selecting a Conversation Session clears the display and rebuilds it from that Session's persisted user, assistant, and final Tool records, groups each user-message-bounded Agent Run into the same activity-group and final-output presentation used by the live display, then positions the view at the latest content. This grouping is a UI projection inferred from each user message and the messages that follow it until the next user message; Session storage does not gain a persisted Agent Run or turn identifier.
- Selecting the already active Session is a no-op with no confirmation or display rebuild. A different Session with pending input but no active run replaces the complete Runtime Generation without prompting and discards the old pending input. When a different Session would replace an active foreground run, the UI presents exactly one Approve/Decline confirmation. Decline leaves the old Session, queue, display, and active run unchanged; Approve replaces the generation and rebuilds the selected Session without waiting for old-run repair or Provider shutdown. Target preparation happens first; if it fails, the old generation remains authoritative and the UI renders one safe error. After a successful replacement, the old Outbound consumer and references are discarded, so late output can reach only the old Message Bus and cannot render in the new Session.
- `Esc` closes the `/resume` modal without changing the current Session, draft, or conversation display.
- Clicking outside Tool Confirmation or `/resume` does not close the modal.
- `Ctrl+C` is handled by the topmost surface before global behavior: it declines Tool Confirmation, cancels `/resume`, or closes command completion. Only when no such surface is open does it cancel an active turn, clear a draft, or exit.
- A resumed view does not recreate transient MessageBus output such as Tool-call notifications, stream timing, or pending confirmations. No separate UI event log is added.
- A submitted Management Command appears as a neutral command row followed by its result. `/config`, `/status`, `/memory`, and `/dream` write their results to neutral full-width rows; a successful `/resume` replaces the display and therefore does not retain its command row.
- Partial assistant content remains visible when a turn is cancelled or fails, followed by a neutral `Cancelled` or error row. The UI does not offer a retry action.
- Model content from the active call is not a final output when the run fails or is cancelled. If there is process activity to group, that partial content moves into the expanded activity group and the cancellation or failure status remains outside it.
- On cancellation, the accumulated streamed candidate is retained before the `system_control` `_streamed` cancellation marker is rendered; it moves into the expanded activity group and does not create an empty assistant item. On failure, the `system_control` `_streamed` failure marker remains the visible terminal outcome and the UI can retain only candidate text already received through streaming.
- A model call is considered final only after the successful `model_response` `_streamed` marker arrives. If a failure occurs after a no-Tool model response has been displayed but before successful completion, that response moves into the expanded activity group and the failure status remains outside it.
- Any successfully completed Agent Run with empty final model output displays a neutral `Completed with no response.` row, whether or not it has an activity group. If it has a group, the group still collapses.
- A successful Run with nonempty final assistant output does not add a redundant `Completed` status row.
- When `/resume` encounters a historical user-message-bounded run whose persisted messages do not reveal whether it completed, failed, or was cancelled, the UI rebuilds its known intermediate model and final Tool records as an expanded activity group without inventing a terminal outcome.
- When persisted assistant state does explicitly identify an interrupted or failed Run, `/resume` rebuilds its known process content as an expanded activity group and places the cancellation or failure status outside it. The no-invention rule applies only when persisted state does not identify an outcome.
- Rebuilt successful groups default to collapsed; rebuilt failed, cancelled, and unknown-outcome groups default to expanded. All are mouse-toggleable after reconstruction.
- Historical projection is tolerant of valid individual messages in an abnormal Run order. It uses the last recognizable terminal assistant state to choose the default group state, preserves other recognizable messages in persisted order inside the activity group, and falls back to the existing flat projection for content it cannot safely classify rather than treating the whole Session as corrupt.
- Persisted assistant or Tool messages before the first user message remain in the existing flat projection because the UI cannot safely assign them to an Agent Run.
- When an abnormal historical Run has multiple completed assistant messages without Tool calls, the last is projected as the final output and earlier nonempty assistant messages are retained as activity. A recognized successful historical Run with empty final assistant content displays `Completed with no response.` just like a live Run.
- `/resume` estimates historical execution time from the persisted timestamps within each user-message-bounded run instead of changing the Session storage format to persist a measured duration. A complete run uses the interval from its user message to its final assistant message; a run without a known terminal outcome uses the interval from its user message to its last known message. Both display the same integer-seconds format without an uncertainty marker.
- Historical duration estimates clamp negative or reversed timestamp intervals to `0s`.
- A run that fails or is cancelled before producing model content or Tool activity does not render an empty activity group solely to show elapsed time; only its terminal status remains in the normal display.
- If live MessageBus consumption fails without a terminal `_streamed` outbound, the UI freezes timing, treats the visible unconfirmed model output as process content, keeps any nonempty activity group expanded and mouse-toggleable, and places the existing generic failure status outside it. Application shutdown does not mount replacement failure UI.
- A direct successful `model_response` `_streamed` marker from an abnormal or test source updates and retains the current candidate as final output, collapses an existing activity group, and otherwise displays only the final outcome. Once any terminal marker has resolved the Run, subsequent terminal markers cannot rewrite its content, status, duration, or disclosure state.
- Failures, cancellation, final Tool states, and Management Command output remain in the live display until the Session changes or the UI exits. `Working`, new-content state, completion, and modal overlays are transient.

## Host And Lifecycle

- Textual `>=8.2.8,<9` hosts the full-screen UI. The existing headless `run_repl` seam remains available for Runtime and adapter tests, not as a user-selected UI.
- A narrow enhanced-keyboard adapter detects and enables modifier-aware Enter input where the terminal protocol supports it. Unsupported terminals use `Ctrl+J` for reliable multiline entry.
- Mouse reporting covers conversation-wheel events and Confirmation controls.
- Interactive lists and confirmation options accept clicks. Conversation and code-block scrollbars appear only on overflow and support pointer dragging; ordinary content remains non-interactive.
- The activity-group disclosure uses a custom non-focusable title rather than Textual's keyboard-operable default Collapsible title. It handles mouse clicks only, uses the same muted neutral styling as Tool activity, and adds neither a border nor a tooltip; hover may apply a subtle text emphasis.
- Replacing the conversation display during `/resume` destroys the previous Session's activity-group timers, click state, widgets, and internal references before the rebuilt Session display becomes authoritative.
- When the terminal becomes too small for both primary regions, normal layout pauses and a size-insufficient state is shown. Restoring the window rebuilds the layout and preserves the conversation's scroll anchor.
- Exit, cancellation, startup failure, and unhandled failure must restore terminal state before Runtime shutdown completes.
- UI-owned labels, statuses, buttons, and errors are English. User and model content is shown unchanged.

## Related Decisions

- [Implementation spec: GitHub Issue #131](https://github.com/Totoro-debug/MyClaw/issues/131)
- [ADR-0011: Use a Full-Screen Terminal Conversation](./adr/0011-use-full-screen-terminal-conversation.md)
- [ADR-0012: Use Textual for the Terminal Conversation](./adr/0012-use-textual-for-terminal-conversation.md)
