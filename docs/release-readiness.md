# MyClaw Host-Neutral Release Readiness

Status: **READY**

This is the current evidence index for GitHub issue
[#69](https://github.com/Totoro-debug/myclaw/issues/69). The active platform decision
is [ADR-0007](adr/0007-use-host-adapters.md), which supersedes ADR-0006 without
rewriting that historical Windows-only decision.

## Release Contract

- The installed command enters the Typer application directly with no platform gate,
  operating-system version check, or architecture allowlist.
- Packaging emits exactly one pure-Python `py3-none-any` wheel containing the
  filesystem and owned-process adapters plus the Loguru Session Log dependency.
- Windows x64 is the currently validated environment.
- macOS Intel and Apple Silicon are intended compatibility targets but remain
  unverified until the same suite and installed-wheel smoke run natively there.
- Linux and other POSIX hosts may attempt the POSIX adapters, but receive no formal
  support claim from this release.
- Agent Home remains host-local and owns User Configuration. Legacy Agent Home Runtime
  Log files remain untouched and are no longer opened by MyClaw.
- Workspace State remains `<workspace>/.myclaw/` and keeps every existing layout,
  record format, Permission Policy rule, and lifecycle guarantee.

## Delivery Evidence

| Area | Evidence | Result |
| --- | --- | --- |
| Host filesystem | Windows-native characterization plus POSIX contract/fault injection | PASS |
| Workspace State | Native identity, ownership, containment, redirection, and persistence suites | PASS |
| Session Log | Workspace path safety, routing, rotation, retention, terminal behavior, failure isolation, and drain suites | PASS |
| Shell lifecycle | Direct argv, trusted Git, Windows Job, POSIX process group, cancellation, and shutdown suites | PASS |
| CLI and package | Direct Typer entry, universal tag, clean installation, dependency check, and Unicode smoke | PASS |
| Complete Windows gate | Full warning-strict pytest, Ruff, strict Mypy, artifact rebuild, and final clean install | PASS |

Artifact identity, host details, exact commands, and final counts are recorded in
[Windows x64 validation](release/windows-validation.md).

## Evidence Boundaries

- POSIX adapter tests run on Windows with synthetic capabilities and fault injection.
  They are not native macOS validation.
- No macOS CI or manual macOS evidence was added by this release.
- No paid or live Provider conversation is required by this offline release gate.
- File-first persistence and Session Logs remain uncoordinated across runtime processes.
- Shell command policy and owned-process cleanup are not an operating-system
  filesystem or network sandbox.

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

Every required Windows gate passed for the final universal-wheel candidate. Native
macOS validation remains outstanding and is not implied by this release evidence.
