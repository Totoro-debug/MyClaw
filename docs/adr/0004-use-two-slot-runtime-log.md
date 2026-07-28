# Use a two-slot Runtime Log

MyClaw will maintain one Agent Home-level Runtime Log shared by all Conversation
Sessions. The log uses two equal rotating slots named `run.log.0` and `run.log.1`
rather than a current file plus a backup file.

The initial slot is `run.log.0`. After every record is written, the runtime checks
the size of the slot that received the record. When that slot reaches the configured
threshold, subsequent records switch to the other slot. Each slot has a fixed
10,485,760-byte (10 MiB) target limit. The record that crosses the threshold remains
intact, so a slot may exceed the target by at most one encoded record.

A durable global cursor identifies the active slot across runtime restarts. The cursor
is Agent Home-level state shared by every Conversation Session; startup does not infer
the active slot from log-file timestamps.

When the cursor and both slots are absent on first use, the writer atomically creates a
`0\n` cursor without emitting a warning. If the cursor is missing while either slot
exists, or its bytes are not exactly `0\n` or `1\n`, a locked writer atomically resets
it to `0\n` and emits a `WARNING` without including the invalid bytes. If `run.log.0`
has already reached the threshold, recovery completes the normal rotation to
`run.log.1` before appending the triggering record.

Runtime Log state is created on demand with this layout:

```text
~/.myclaw/logs/
  run.log.0
  run.log.1
  run.log.cursor
  run.log.lock
```

`run.log.cursor` contains only the canonical ASCII value `0\n` or `1\n`. Slot changes
atomically replace the cursor file. The two log slot files are also created on demand.

Runtime Log storage remains private Agent Home state. On POSIX, `logs/` is restricted
to mode `0700` and the slot, cursor, and lock files to `0600`; existing broader modes
are narrowed before use. On Windows, files inherit the user's Agent Home ACL. All
platforms require the log directory to resolve inside Agent Home and accept only
unaliased regular control and slot files: symbolic links, Windows reparse points,
hard-linked files, and paths resolving outside Agent Home are rejected. A validation or
permission failure affects only logging and follows its failure-isolation policy.

On the normal path, every record append, resulting size check, target-slot reset, and
cursor change is one cross-process exclusive operation. This makes Runtime Log
coordination a narrow, explicit exception to ADR-0001: Conversation Session and
background persistence remain uncoordinated between REPL processes, while Runtime Log
operations normally serialize access to the shared files.

If lock acquisition fails, logging remains available by appending the record to the
original log slot without the lock. This fallback does not rotate files or update the
cursor. It is deliberately best effort: a concurrent successful rotation may remove
the fallback record, and the target slot may temporarily exceed its size threshold.
The original slot is the canonical cursor value read atomically immediately before the
lock attempt. If that cursor cannot be read, the fallback slot is `run.log.0`.

The implementation uses only Python's platform-specific standard-library locking
primitives: `fcntl.flock` on POSIX and `msvcrt.locking` on Windows. Both backends lock a
stable `~/.myclaw/logs/run.log.lock` control file. MyClaw does not add a third-party
locking dependency.

Lock acquisition uses non-blocking attempts with a monotonic one-second deadline.
Failure to acquire the lock by that deadline activates the unlocked append fallback
described above; logging never waits indefinitely for another runtime process.

While holding the lock, a writer appends one complete record and checks the active
slot's resulting byte size. Once it reaches the threshold, the writer first atomically
replaces the other slot with an empty file and then atomically changes the cursor to
that slot. Before a normal-path append, a writer also completes this same rotation if
the cursor still identifies a slot that already reached the threshold. This repairs a
crash between resetting the target slot and publishing the new cursor, and ensures the
cursor never intentionally points to uncleared content from an older cycle.

Runtime Log records use UTF-8 plain text rather than JSON Lines. The slot names remain
`run.log.0` and `run.log.1`.

When a WARNING or ERROR record is caused by an exception, its plain-text body includes
the exception type, message, complete traceback, and chained causes. A warning without
an associated exception remains a single-line record. A multi-line record is fully
formatted and UTF-8 encoded before the writer performs one append operation.

The header is always one physical line. Traceback formatter output follows immediately,
with every continuation line indented by four spaces. Each complete record ends in
exactly one newline and records are not separated by blank lines. Carriage returns,
NUL, ESC, and other non-structural control characters from exception text are rendered
as visible escapes; only formatter-owned newlines create physical continuation lines.

Every record starts with a fixed plain-text header in this shape:

```text
2026-07-28T14:32:10.123+08:00 ERROR pid=18420 session=<session_id|-> myclaw.agent.turn: Message
```

The timestamp is local time in ISO 8601 form with millisecond precision and an explicit
UTC offset. The level uses the standard `WARNING` or `ERROR` name. The process ID,
Conversation Session ID when one exists, and component logger name provide correlation
inside the global log; process-level and Memory Task records use `session=-`. Runtime
Log headers do not include the full Workspace path.

Absolute filesystem paths in exception messages and tracebacks are retained without
normalization. MyClaw is a local, single-user Personal Agent, and complete paths are
considered useful diagnostic metadata. This does not permit logging file contents or
authentication material.

The Runtime Log records technical failures, not normal control outcomes. User
cancellation, EOF and normal exit, Tool permission refusal, invalid Tool arguments,
an empty Memory Task batch, and a background trigger skipped because equivalent work is
already running do not produce WARNING or ERROR records. Unreadable or invalid startup
configuration, corrupt persisted state, model or provider failures, unexpected Tool
failures, and background task exceptions remain in diagnostic scope.

Severity follows the outcome of the affected unit of work rather than the lifetime of
the process. A technical failure is `WARNING` when retry, route fallback, safe
degradation, or selective skipping allows the main operation to complete successfully.
It is `ERROR` when a startup, foreground turn, Tool execution, Memory Task, Scheduled
Work run, persistence operation, or shutdown ultimately fails, even if the runtime can
continue serving other work afterward.

Records contain diagnostic metadata but not business content. Error codes, exception
types and sanitized tracebacks, component names, Provider and Model Route identifiers,
model IDs, Tool names, attempt counts, process IDs, and Conversation Session IDs are
allowed. User and assistant messages, System Prompts, Long-term Memory, Tool arguments
and results, file contents, web content, provider request or response bodies, API keys,
and authentication or cookie headers are forbidden. Before writing, the logger redacts
all configured API-key values and common Bearer, API-key, and Cookie credential patterns
from the complete formatted record, including exception messages and tracebacks.

Each non-terminal failed Provider attempt produces one `WARNING` with its one-based
attempt number out of five, safe error code, Provider and Model Route metadata, and
planned retry delay. Switching to the default Model Route produces a separate
`WARNING`. A terminal attempt does not produce another retry warning; its exception
continues to the enclosing work-unit boundary for final classification.

Recoverable failures are recorded as `WARNING` at the point that decides to retry,
fallback, skip, or degrade because they do not propagate farther. Terminal failures are
recorded once as `ERROR` only at the highest boundary of each independent unit of work,
such as CLI startup, an Agent Turn, Tool Gateway call, Memory Task, Scheduled Work run,
Session title task, or Runtime shutdown. Provider, persistence, and other lower layers
raise terminal failures without logging them. A distinct failure raised while cleaning
up the original operation is a separate `ERROR` with its own traceback.

The handler is attached only to MyClaw's dedicated `myclaw` logger hierarchy. MyClaw
does not configure the root logger, capture Python `warnings`, or directly persist
third-party SDK log records. Failures from Provider and transport libraries enter the
Runtime Log only after a MyClaw boundary has converted them to safe metadata and applied
the Runtime Log redaction policy.

The diagnostic ownership matrix is:

| Outcome | Owning boundary and examples |
| --- | --- |
| `WARNING` | Model Router retry or fallback; Session listing skips corrupt state; cursor recovery; Session title or stream-close degradation that still completes its parent operation. |
| `ERROR` | CLI startup or configuration command cannot complete; Agent Turn ultimately fails; Tool Gateway execution ultimately fails; Tool Artifact persistence fails; Session, Memory, or Scheduled Work persistence prevents its operation; Memory Task or Scheduled Work run fails; Runtime startup or shutdown fails; an independent background task terminates unexpectedly. |
| Excluded | User cancellation and normal exit; Tool refusal or invalid arguments; empty Memory Task input; an overlapping background trigger skipped by policy. |

Provider, filesystem, and persistence adapters preserve exception chains but do not log
terminal failures themselves. Correlation context is captured before queue submission:
foreground Turns, Session title tasks, Tools, and Scheduled Work use their owning Session
ID, while CLI startup, configuration, Runtime-wide shutdown, and Memory Tasks use
`session=-`.

The CLI installs the dedicated handler before loading User Configuration for both the
default REPL command and the `config` Management Command. Handler installation is lazy:
the `logs/` directory and its files are not created until the first record is emitted.
Before configuration loads, generic credential-pattern redaction applies. After a
successful load, every configured Provider API key is added to the redactor so later
records can remove exact secret values as well.

Runtime Log failures never change the result of a conversation, Tool, background task,
Management Command, startup, or shutdown. Disk exhaustion, permission failures,
directory or cursor replacement errors, and failure of the unlocked append fallback
are caught inside the handler. It writes one non-recursive diagnostic containing only
the internal exception type directly to `stderr`, suppresses repeats during the same
continuous failure, and becomes eligible to report again after any successful file-log
append.

The Runtime Log is always enabled at the fixed `WARNING` threshold. User Configuration
cannot disable it or change its level, paths, slot count, or 10 MiB threshold. This
keeps diagnostics available when User Configuration itself is missing or invalid and
does not expand the `config.toml` contract.

Runtime logging supplements rather than replaces existing user-facing CLI errors and
Agent Events. This change adds no `/logs` command, `myclaw logs` command, search, export,
or manual cleanup interface; users inspect the two Agent Home slot files directly.

Each MyClaw process owns one dedicated Runtime Log writer thread. The main thread and
asyncio tasks capture the record timestamp and correlation context, enqueue a write
task, and return without performing Runtime Log filesystem I/O. The writer thread alone
formats and redacts records, acquires the cross-process lock, appends and synchronizes
bytes, checks slot size, rotates slots, and updates the cursor. Records from one process
retain FIFO submission order; order between processes is determined by lock acquisition.

Submission uses a bounded FIFO with capacity 1024 and never waits. When the queue is
full, the submitter silently removes the oldest pending record and retains the newest
record. Queue overflow does not produce a Runtime Log record or `stderr` diagnostic.

The writer is a daemon thread. On normal REPL shutdown, Management Command completion,
or a handled top-level failure, the outermost CLI lifetime stops accepting records and
waits up to ten seconds for the queue to drain. Any tasks left after the deadline are
silently abandoned so logging cannot prevent process exit. Forced termination or a
process crash may also lose queued records.

The writer flushes and `fsync`s every complete normal-path and unlocked-fallback append
before checking size or reporting the task complete. Filesystems that explicitly do not
support `fsync` use the repository's existing best-effort compatibility behavior; other
synchronization failures follow the Runtime Log failure-isolation policy.

This accepted decision supersedes the former domain statement that the first version
does not maintain a persistent Runtime Log and extends ADR-0002's Agent Home layout with
the on-demand `logs/` directory. It is the only cross-process coordination exception to
ADR-0001; Conversation Session and background state remain uncoordinated between REPL
processes.
