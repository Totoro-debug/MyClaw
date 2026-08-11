# MyClaw Schedule Release Readiness

Schedule release status: **READY**

Terminal Conversation real-terminal validation: **PENDING**

This is the current evidence index for GitHub issues
[#69](https://github.com/Totoro-debug/myclaw/issues/69), the final Session
architecture gate [#92](https://github.com/Totoro-debug/myclaw/issues/92), and the
Schedule end-to-end gate [#117](https://github.com/Totoro-debug/myclaw/issues/117),
which completes parent issue [#104](https://github.com/Totoro-debug/myclaw/issues/104).
The READY status above applies to that recorded Schedule release candidate; it does not
claim that the later Terminal Conversation real-terminal matrix has been completed.
The active Tool boundary decision is [ADR-0010](adr/0010-fixed-tool-catalog-and-base-tool-boundaries.md),
which supersedes the affected parts of ADR-0003, ADR-0005, and ADR-0007 without
rewriting those historical decisions.

## Release Contract

- The installed command enters the Typer application directly with no platform gate,
  operating-system version check, or architecture allowlist.
- Packaging emits exactly one pure-Python `py3-none-any` wheel containing the
  filesystem adapter and direct Exec implementation plus the Loguru Session Log
  dependency; no owned-process Tool adapter is part of the package.
- Windows x64 is the currently validated environment.
- macOS Intel and Apple Silicon are intended compatibility targets but remain
  unverified until the same suite and installed-wheel smoke run natively there.
- Linux and other POSIX hosts may attempt the POSIX adapters, but receive no formal
  support claim from this release.
- Agent Home remains host-local and owns User Configuration. Legacy Agent Home Runtime
  Log files remain untouched and are no longer opened by MyClaw.
- Workspace State remains `<workspace>/.myclaw/` and owns Conversation Session history,
  Conversation Summary, Long-term Memory, Schedule state and Jobs, Tool Artifacts, and
  Session Logs. Active Session format and lifecycle follow ADR-0009: JSON-native state, strict
  five-field header, complete atomic JSONL snapshots, ordered async `persist()`, and
  bounded synchronous `close()`.
- Schedule Jobs use the shared Agent Run through one dispatcher. Add/list/remove,
  confirmation, at/every/cron timing, Schedule Session ownership, Summary/Memory
  behavior, fault recovery, and shutdown are covered through the Runtime composition
  seam documented in [Windows x64 validation](release/windows-validation.md).

## Terminal Conversation Acceptance Matrix

The default `myclaw` command is the full-screen Terminal Conversation. A non-TTY
launch is rejected before Textual or Runtime startup with the stable
`interactive_terminal_required` outcome; it never falls back to the internal headless
REPL seam and it does not create Conversation Session history.

| Host input path | Validation status | Shift+Enter | Alt+Enter | Ctrl+J | Expected result |
| --- | --- | --- | --- | --- | --- |
| Windows Terminal target | **PENDING real-terminal run** | Newline only when the enhanced-keyboard report is delivered | Newline only when the host delivers it; Windows Terminal may reserve the chord | Always newline | Capability-aware multiline composition |
| Other terminal with enhanced-keyboard reporting | Capability contract only | Newline when the supported report is negotiated and parsed | Newline only when delivered by the host | Always newline | Capability-gated behavior; no inferred modifier |
| Basic or unsupported terminal | Capability contract only | Ordinary Enter keeps submit semantics | No assumption about delivery | Always newline | Safe ordinary submission plus portable multiline fallback |

The current Windows Python environment cannot import the standard-library `pty` module
because `termios` is unavailable, and it has no `winpty` or `pexpect` harness. A native
pseudo-terminal process smoke therefore cannot run on this host. The nearest executable
process seam is the installed console-entry-point test for non-TTY rejection, while
Textual headless lifecycle tests cover application startup, exit, cancellation, and
terminal restoration. No Windows Terminal version, settings snapshot, observation date,
or physical Shift+Enter/Alt+Enter/Ctrl+J result has been recorded, so this document does
not claim that path as validated. Run the installed-wheel PTY smoke on a POSIX host and
record the Windows Terminal real-terminal matrix before release.

The accepted Terminal Conversation architecture is recorded in
[ADR-0011](adr/0011-use-terminal-conversation-as-the-interactive-cli.md), and the
Textual/enhanced-keyboard choice is recorded in
[ADR-0012](adr/0012-use-textual-and-capability-gated-enhanced-keyboard-input.md).

## Delivery Evidence

| Area | Evidence | Result |
| --- | --- | --- |
| Host filesystem | Windows-native characterization plus POSIX contract/fault injection | PASS |
| Workspace State | Native identity, ownership, containment, redirection, and persistence suites | PASS |
| Active Session | JSONL replacement, lazy materialization, snapshot ordering, silent failure, close retry, and status vocabulary | PASS |
| Session Log | Workspace path safety, routing, rotation, retention, terminal behavior, failure isolation, and drain suites | PASS |
| Exec lifecycle | Direct Bash, safety confirmation, cancellation, and shutdown suites | PASS |
| Schedule and shared Agent Run | Runtime composition E2E plus Schedule model, Store, Tool, Service, Session, Summary, and Memory suites | PASS |
| CLI and package | Direct Typer entry, clean universal tag, clean installation, dependency check, and Unicode smoke | PASS |
| Complete Windows gate | Full warning-strict pytest, repository Ruff lint, strict Mypy, clean artifact rebuild, and final clean install | PASS |

Artifact identity, host details, exact commands, and final counts are recorded in
[Windows x64 validation](release/windows-validation.md).

## Evidence Boundaries

- POSIX adapter tests run on Windows with synthetic capabilities and fault injection.
  They are not native macOS validation.
- No macOS CI or manual macOS evidence was added by this release.
- No paid or live Provider conversation is required by this offline release gate.
- File-first persistence and Session Logs remain uncoordinated across runtime processes.
- Ordinary Session persistence has no acknowledgement or failure logging; a crash can
  lose the latest turn, and Conversation Summary can diverge from `last_consolidated`.
- Existing Session schemas are unsupported; no migration or version dispatch is provided.
- Exec command checks and direct-process cleanup are not an operating-system
  filesystem, network, or process sandbox.

## Session Log Accepted Risks

- Same-Session concurrency is unsupported; duplicate, interleaved, rotated, or damaged
  output is undefined when that precondition is violated.
- Loguru uses an unbounded queue and normal sink removal performs an infinite drain.
- There is no per-record fsync, so crashes, power loss, and forced termination can lose
  recent records.
- No active redaction and no control escaping are performed; caller-supplied credentials,
  exception text, and control characters may be persisted verbatim.
- Rotation provides per-Session retention only: one active file and at most one history
  file per Session, with no Workspace-wide size bound.
- Legacy Agent Home Runtime Log files remain untouched, byte-for-byte, and are never
  updated by the packaged application.

Every required Windows gate passed for the issue-117 universal-wheel candidate.
Native macOS validation remains outstanding and is not implied by this release
evidence. The exact Schedule acceptance matrix and artifact identity are recorded
in [Windows x64 validation](release/windows-validation.md).
