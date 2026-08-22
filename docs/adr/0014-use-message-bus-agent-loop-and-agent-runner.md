---
status: accepted
---

# Use Message Bus, Agent Loop, and Agent Runner

> Historical context boundary: the Context paragraph records the pre-migration
> architecture that motivated this decision. `AgentRun` in that paragraph is the old
> implementation type; current `Agent Run` is only the domain execution concept.

## Context

The current interactive path combines product orchestration, the model-and-Tool loop,
foreground presentation events, Session ownership, and terminal transport across
`AgentRun`, `ConversationPort`, `AgentEvent`, and `PreparedReplRuntime`. User input is
therefore coupled to one active turn, the reusable ReAct boundary is not explicit, and
Session switching cannot replace all Session-bound runtime state as one unit.

The product needs editable FIFO input while a foreground run is active, one terminal
projection of selected execution output, a reusable bounded ReAct engine, and Schedule
execution that can reuse runtime capabilities without leaking output or state into the
foreground conversation.

## Decision

One `AgentLoop` owns one transient `MessageBus`. The bus contains only an unbounded
Inbound FIFO and an unbounded Outbound FIFO and has no independent close state.
Inbound accepts ordinary user messages only. Cancellation, Tool Confirmation, Session
switching, and Management Commands use separate control interfaces. Outbound has one
Terminal Conversation consumer and contains only `model_reasoning`, `model_response`,
`tool_call`, and `system_control` messages. Tool results, Schedule Job output, and
Memory Task output never enter Outbound, and Outbound is not persisted.

`AgentLoop` is the product orchestration layer. It owns the active Session and Message
Bus, assembles context, invokes summaries and title generation, creates the fixed Tool
catalog and shared Tool Gateway, invokes `AgentRunner`, commits the returned increment,
schedules Session persistence, publishes foreground Outbound messages, and exposes the
independent foreground control operations.

`AgentRunner` is a reusable, Session-independent ReAct engine whose constructor owns
only `ModelRouter`. Each call receives initial model messages, a Model Route, a Tool
Gateway, callbacks, confirmation and cancellation controls, result externalization,
and the iteration limit. One iteration is one model call followed by every Tool call
from that response, executed sequentially in provider order. The default limit is 50,
configuration values below 50 are invalid, and a fiftieth response that still requests
Tools is completed through those Tools before the run terminates without a fifty-first
model call.

The Runner result contains only the generated assistant and Tool message increment,
not the current user message or initial context. It also returns final content,
cumulative usage, a `completed`, `failed`, `cancelled`, or `max_iterations` finish
reason, and optional structured `ErrorInfo`. Agent Loop commits the current user
message together with that increment. Normal cancellation repairs an atomic provider
message sequence before persistence; a forced Runtime Generation replacement does not.

Provider-visible reasoning may be streamed to Outbound when the Provider actually
returns it. Opaque reasoning continuation state required by a Provider remains inside
the current Runner call and is passed back only as required for the Tool-use loop. The
design does not claim access to raw hidden chain-of-thought and does not persist
reasoning as Session messages.

Tool Confirmation is a blocking await on one pending Future, not a queue. The Terminal
temporarily replaces the input UI with the confirmation UI and resolves that Future
through the Agent Loop control interface. Schedule runs receive no confirmation
channel, so confirmation-required Tools refuse.

Schedule Service is created before Agent Loop. Agent Loop constructs `ScheduleTool`
from the Service, creates the shared Tool Gateway, and then its Schedule callback is
assigned to the Service before start. Schedule Service alone owns Schedule Store
access. Its callback performs a complete isolated Schedule Agent Run with Agent Loop
resources and no foreground Message Bus. A task-local `ContextVar` on `ScheduleTool`
prevents recursive Schedule Job creation while preserving concurrent foreground use.

Every successful Session switch replaces the complete active Runtime Generation. The
replacement is prepared and validated before the old generation is synchronously
aborted. Abort abandons the old Session, cancels owned work without awaiting it,
discards its Message Bus and pending input, and performs only detached best-effort
Provider cleanup. Normal process shutdown remains an awaited close. This split is
intentional.

Ordinary Session persistence remains an ordered complete-snapshot operation, but each
asynchronous snapshot now receives at most three attempts with 100 ms and 200 ms
backoff. Exhaustion is silent and a later snapshot includes the authoritative in-memory
state. Forced Session abandonment cancels pending snapshots and performs no final
write. Tool Artifacts keep the existing
`.myclaw/artifacts/<session_id>/<tool_call_id>.txt` path for both foreground and
Schedule Sessions.

`ConversationPort`, `AgentEvent`, their payloads and emitters, and
`PreparedReplRuntime` are removed after the terminal cutover; no compatibility aliases
remain. `ManagementPort` is unchanged.

## Consequences

- Foreground input and output have explicit ownership and FIFO semantics, while
  operational controls cannot be mistaken for user messages.
- Agent Runner can be tested and reused without Session, Message Bus, terminal, memory,
  or summary dependencies.
- One shared Tool Gateway keeps a fixed Tool catalog, but Schedule execution is isolated
  by per-run arguments and task-local recursion state rather than a second Gateway.
- Outbound is a live projection, not an audit log. Restart and Session resume rebuild the
  display from Session messages and cannot recreate transient reasoning or Tool calls.
- Forced Session switching favors immediate isolation over graceful completion. It can
  lose unpersisted state, leave Tool Artifacts or side effects, skip Memory work under
  the existing cursor contract, and cause at-least-once Schedule side effects.
- Unbounded FIFOs provide no backpressure. This is accepted for the local single-user
  product and can be revisited only with a separate overload policy.

## Amendment: Runtime Generation replacement (#172)

Issue #172 supersedes only the earlier statement that `ManagementPort` is unchanged.
Its `resume` operation may carry the Terminal-confirmed `force` flag to the outer
Runtime Host. Management remains a Management-only boundary: it neither constructs
Runtime components nor reaches into MessageBus internals.

This decision supersedes the Conversation Port and Agent Event portions of ADR-0011
and ADR-0013, and the ordinary no-retry and close-only persistence portions of
ADR-0009. ADR-0010's fixed Tool catalog remains in force; only construction ownership
moves into Agent Loop.

## Amendment: final post-#172 runtime state (#173)

Issue #172 is the accepted implementation of the Runtime Generation replacement, and
this amendment records the final current facts rather than reopening the decision:

- `RuntimeHost` prepares and validates an unstarted target generation before synchronously
  aborting the old one; terminal unbind/rebind and target start occur only after target
  preparation succeeds. Same-session replacement is a no-op.
- `PreparedRuntime.close()` is the normal awaited lifecycle and `PreparedRuntime.abort()`
  is the synchronous forced-switch lifecycle. Abort calls `Session.abandon()`, cancels
  generation-owned work without awaiting repair or persistence, and leaves detached
  Provider cleanup best effort with failure logging only.
- Message Bus exposes only its six async queue operations and one synchronous Inbound
  callback binding. It has no independent close, abort, replay, broadcast, version, or
  backpressure lifecycle.
- If cancellation is requested after the fiftieth response's Tools finish, normal
  cancellation wins; otherwise the Runner returns `agent_iteration_limit` without a
  fifty-first model call.
- The generation owns fresh Schedule Store/Service, Message Bus, Router, Runtime Memory,
  Memory scheduler, Agent Loop, fixed Tools, shared Gateway/Runner, and Management
  services. Agent Home, Workspace, configuration, Provider factory, clocks, and Terminal
  application are process-level inputs reused across generations.
- Schedule Service remains the only Store/management owner. Foreground and Schedule use
  the same Gateway/Runner identities but isolated Session, context, cancellation, and
  externalizer state. Schedule has no foreground Message Bus projection, passes
  `confirmation=None`, and resets its recursive-add ContextVar token in `finally`.

ADR-0010's fixed Tool catalog remains authoritative. No compatibility alias restores the
deleted transport or lifecycle names, and no Python `AgentRun` class is part of the
current package.
