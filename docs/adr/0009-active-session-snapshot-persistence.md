# Use an Active In-Memory Session with Ordered Snapshots

---
status: accepted
---

## Context

The foreground Runtime needs one authoritative Conversation Session while a
turn is running. A file-first design based on message-by-message persistence
split that authority between an identifier, immutable persistence objects, and
recovery journals. It also made an ordinary turn depend
on metadata rewrites and crash repair even though the user-facing operation is
an in-memory conversation.

## Decision

The foreground Runtime owns exactly one active `Session` instance. `Session` is
the state and persistence boundary for that Conversation Session: callers pass
the instance through Runtime Core, Conversation, Conversation Summary,
Management status, Tool Artifact integration, and Session Log context. Session
does not own model calls, Tool execution, Agent Events, cancellation, or the
Conversation Port.

`Session` exposes synchronous `create`, `load`, `add_message`,
`append_messages`, `update_metadata`, `persist`, and `close` methods. Its public
mutable state is JSON-native `list[dict[str, Any]]` messages, `dict[str, Any]`
metadata, and the integer `last_consolidated`. `session_id`, `created_at`, and
`updated_at` are read-only properties. Message and metadata values must remain a
JSON-compatible tree; provider-specific values are converted at the provider seam
before they enter Session.

`append_messages` is the atomic in-memory boundary for an Agent Run increment. It
deep-copies, timestamps, and validates every input message and calculates the complete
assistant usage delta before changing Session state. A failure leaves messages and
cumulative usage unchanged; success appends all prepared messages in input order and
applies the usage delta once. `add_message` remains the compatible single-message
operation.

Creation and loading are synchronous. A new empty Session is memory-only and
does not create a history file. Its first message makes it eligible for
materialization. Session IDs use a system-local timestamp plus UUID4. In-memory
Session timestamps are timezone-aware; persisted timestamps and message
`timestamp` values are ISO 8601 strings with the host's current local offset.

### Current JSONL shape

The history path remains `<workspace>/.myclaw/sessions/<session_id>.jsonl`.
Every successful write replaces the complete compact UTF-8 JSONL document
atomically. The first line is a strict header with exactly these fields:

```json
{"session_id":"<session_id>","created_at":"<local-time>","updated_at":"<local-time>","last_consolidated":0,"metadata":{"title":"Untitled session","token_usage":{"model_calls":0,"input_tokens":0,"output_tokens":0,"total_tokens":0}}}
```

Each later line is one message dictionary with the common `role`, `content`,
and `timestamp` fields. Supported roles are `user`, `assistant`, and `tool`;
role-specific provider fields and unknown JSON-compatible extensions are
preserved. There is no message identifier, line type marker, or schema-version
field. The current header field set is strict, while future structural fields
must be explicitly accepted by a later implementation without version
dispatch.

Existing schema-versioned Session files are unsupported. There is no migration,
compatibility reader, lazy upgrade, repair mode, or persistence version
dispatch.

### Turn persistence

Agent operations mutate only the active Session in memory. After all work for a
turn is complete, AgentTurn calls `persist()` without awaiting file I/O.
`persist()` immediately captures the complete state, including a deep copy of
messages and metadata, assigns a new local `updated_at`, and schedules one
asynchronous atomic replacement. Pending snapshots are chained so they finish
in call order; an older snapshot cannot overwrite a newer one. The method
returns without a task, acknowledgement, or result.

An ordinary asynchronous write failure is swallowed. It produces no Agent
Event, diagnostic log, failure acknowledgement, retry record, or change to the
turn outcome. A later turn may write a newer complete snapshot containing the
earlier in-memory state; this is a new ordinary attempt, not an explicit retry.

`close()` first prevents queued work from publishing stale state, then makes a
bounded synchronous best-effort save of the latest nonempty Session. It makes
at most three attempts, waiting 100 ms and 200 ms between failures. Shutdown
failures are swallowed after that budget and never logged or raised by Session.

### Summary and ownership

Conversation Summary directly assigns `Session.last_consolidated` after its
summary stream operation. There is no Session-specific position method,
monotonicity check, extra journal, or crash-recovery protocol
coordinating the two files. A crash or failed snapshot can therefore leave the
Summary content and `last_consolidated` divergent, causing repeated or omitted
summary work.

Workspace State continues to own Session history, Tool Artifacts, Conversation
Summary, Scheduled Work, and lazily-created Session Logs. Tool Artifact
publication remains Runtime Core behavior, and Session Log remains a separate
diagnostic facility. No cross-process coordination is added for active Session
snapshots, Summary state, or other Workspace-owned state; separate REPL
processes may race.

## Accepted tradeoffs

- Silent ordinary persistence failure keeps Agent Events and turn outcomes
  independent of best-effort disk work, but no persistence acknowledgement,
  failure logging, or user-visible save status exists.
- An abnormal process exit after an asynchronous failure can lose the most
  recent in-memory turn; bounded `close()` is only a final opportunity, not a
  strong crash-consistency guarantee.
- Title generation remains asynchronous after the first user message. A late
  title can remain memory-only because the first-turn snapshot may already be
  queued; a later turn or `close()` can persist it.
- Summary content and `last_consolidated` can diverge after a crash or failed
  Session snapshot, so summary work may repeat or be omitted.
- Direct mutation of messages, metadata, and `last_consolidated` is permitted;
  callers own the risk of bypassing convenience normalization.
- Multiple processes have no Session lock or coordination. The design supports
  one active Session per foreground Runtime, not a shared multi-process
  Conversation Session.

## Consequences

The Session module is a small, deep public seam. Tests observe its in-memory
state and complete JSONL snapshots through the public interface. The old
per-message write, incomplete-line repair, typed persistence-object, and extra
journal contracts are not compatibility surfaces and must not be reintroduced
without reopening this decision and ADR-0001.
