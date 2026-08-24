---
status: accepted
---

# Use Session Blackboard Task Framing Before Foreground Agent Runs

## Context

A user's task can be continued, narrowed, expanded, replaced, or cancelled across
successive inputs. The foreground Agent currently receives raw conversation context but
has no compact, explicit statement of the one task it is trying to finish or the boundary
that makes the task complete. Putting this state in a visible task manager, updating it
inside the ReAct loop, or letting it control execution would add a workflow product that
MyClaw does not need.

## Decision

Before every nonempty ordinary foreground user input, MyClaw performs an isolated Task
Framing call through the configured `chat` Model Route with no Tools. The call receives
only the previous Blackboard, the complete content of the latest assistant Session
message, and the new raw user input. Its strict JSON decision keeps, replaces, or clears
one Blackboard containing exactly `goal` and `completion_boundary`. The output parser may
extract JSON from a raw response, one Markdown fence, or surrounding prose, but the
resulting object and action invariants remain strict. Prompts require concise values, but
the Runtime imposes no Blackboard character limit.

The resolved Blackboard is staged for one foreground Agent Run. The model-visible current
User Message retains the raw input first and appends the Blackboard in a distinct final
`<blackboard>` block. The persisted user Session message remains the exact raw input.
Blackboard state lives only in the active Conversation Session's `metadata.blackboard` and
has no history, user-facing command, Tool, event, or presentation surface. It is state for
task interpretation only; it cannot authorize operations, bypass Tool Confirmation, order
work, or otherwise control execution.

Blackboard state and its model usage are committed only with a successfully accepted
foreground Agent Run Session increment. A normal failed, maximum-iteration, or cancelled
Runner result still has an accepted repaired increment and therefore commits the staged
state. Preparation cancellation or failure retains the previous state. A completed Task
Framing call whose response is invalid clears the state if the main increment commits;
a Task Framing model failure also degrades to no Blackboard and the raw input. Task
Framing is excluded from Schedule and Memory execution.

## Considered Options

- **A visible task store or Blackboard Tool** was rejected because the state is an
  internal interpretation aid, not a user-managed task product.
- **Updating Blackboard during Agent Runner iterations** was rejected because intermediate
  reasoning and Tool results must not turn passive state into workflow control or cause
  task drift within one Agent Run.
- **A Workspace-level task ledger** was rejected because the state follows one Conversation
  Session and only annotates that Session's foreground runs.
- **Full Session history as framing input** was rejected in favor of the previous state,
  latest assistant content, and new input, which are the minimum facts needed to interpret
  task continuity.
- **Persisting the enriched User Message** was rejected because it would misrepresent
  Runtime-generated state as the user's original words and expose the mechanism on resume.

## Consequences

- Every ordinary foreground input adds one synchronous Chat Route call, increasing latency,
  model usage, and cost. Its usage is included in Session cumulative usage when the turn
  commits.
- The complete latest assistant content can overflow the framing route's context. This is
  accepted; the call then degrades to no Blackboard and the raw Agent Run continues.
- A malformed persisted Blackboard does not make the Session unloadable. It is treated as
  absent and is naturally replaced or cleared by a later committed turn.
- The final `<blackboard>` block is not a security boundary because it remains inside a
  `user` role message. The foreground system prompt identifies the final Runtime-appended
  block, while all existing permission and Tool Confirmation boundaries remain authoritative.
- Session messages, Schedule behavior, Memory behavior, Message Bus output, Agent Runner,
  and Tool Gateway contracts otherwise remain unchanged.

The complete implementation contract and acceptance plan are recorded in
[GitHub Issue #174](https://github.com/Totoro-debug/MyClaw/issues/174).
