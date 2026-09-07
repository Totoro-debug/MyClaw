---
status: accepted
---

# Use a Full-Screen Terminal Conversation

Running `myclaw` without arguments starts one full-screen terminal UI with a scrollable conversation display and a bottom multiline input area. There is no plain REPL or non-TTY fallback. Non-interactive Management Commands remain separate; automated UI tests use Textual's headless test harness.

User input is right-aligned and assistant Markdown is left-aligned. Intermediate model output and Tool activity belong to one ordered Agent Run Activity Group; the final answer stays outside it. Active groups remain open, successful groups collapse, and failed or cancelled groups remain open. Completed groups support mouse disclosure. Scrolling preserves a historical anchor until the user returns to the latest content; activity changes use the same scroll behavior.

Each Tool call has one row keyed by its call ID. It starts with the complete raw arguments and updates in place to Completed, Failed, or Rejected when a status-only completion notification arrives. Neither Tool Result content nor artifact details enter these notifications. Cancellation marks still-pending rows Cancelled; an abnormal run ending without a Tool outcome marks them Status unavailable, without inferring success from the run. Duplicate completion notices cannot rewrite the first observed outcome. Historical rows use persisted Tool status without showing Tool Result content.

Only streamed model content supplies the live assistant text; the terminal `_streamed` message is an outcome marker, not a replacement answer. Exactly one terminal outcome freezes elapsed timing. A successful empty answer displays `Completed with no response.`. UI-owned labels are English; user, model, and runtime error content is displayed as supplied.

Management Commands and confirmation remain separate from ordinary queued input. `/resume` rebuilds the display from the selected Session and clears the shared queues after the old generation is drained. Terminal modes must be restored on normal exit and failure before runtime shutdown finishes.

Detailed product requirements: [Issue #131](https://github.com/Totoro-debug/myclaw/issues/131) and [Issue #146](https://github.com/Totoro-debug/myclaw/issues/146). Framework and keyboard choice: [ADR-0012](0012-use-textual-for-terminal-conversation.md).
