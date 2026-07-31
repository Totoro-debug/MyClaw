# MyClaw Windows x64 Release Readiness

Status: **READY**

This is the current evidence index for GitHub issue
[#56](https://github.com/Totoro-debug/myclaw/issues/56). The accepted storage and
platform decisions are [ADR-0005](adr/0005-store-workspace-state-in-workspace.md)
and [ADR-0006](adr/0006-support-windows-only.md). Historical ADR text remains an
unchanged record of the decisions that applied when it was accepted.

## Release Contract

- Supported runtime: 64-bit Windows on x64 hardware with Python 3.12 or later.
- Published artifact: exactly one `py3-none-win_amd64` wheel.
- Agent Home owns only user configuration and runtime logs.
- Workspace State is rooted at `<workspace>\.myclaw` and owns sessions, memory,
  summaries, scheduled work, and tool artifacts.
- Existing state under the former Agent Home layout is preserved and ignored. No
  automatic discovery, copy, move, or conversion is performed.
- Unsupported platforms fail at the lightweight console entry point before the
  runtime or CLI implementation is imported.

## Delivery Evidence

| Area | Evidence | Result |
| --- | --- | --- |
| Workspace State | Root-conflict, ownership, and non-migration validation | PASS |
| Runtime storage | Session, memory, summary, scheduled-work, artifact, management, and background validation | PASS |
| Platform gate | Windows x64 acceptance and deterministic rejection validation | PASS |
| Process model | Windows Job Object ownership, process-tree shutdown, console handling, and cancellation validation | PASS |
| Filesystem model | Windows file attributes, reparse-point, hard-link, and containment validation | PASS |
| Public surface | Retired workspace-slug, compatibility, and single-implementation layers scan | PASS |
| Release artifact | Exactly one tagged Windows x64 wheel; 73 packaged Python files match source | PASS |
| Clean install | Isolated wheel installation, dependency check, and package-origin validation | PASS |
| Unicode CLI | Clean Agent Home and Workspace first-start and config smoke | PASS |

The artifact identity, host details, and clean-install transcript are recorded in
[Windows x64 validation](release/windows-validation.md).

## Final Gates

| Gate | Result |
| --- | --- |
| Root-conflict focused suite | PASS: `5 passed` |
| Storage and filesystem safety suites | PASS: `231 passed` |
| Complete offline suite | PASS: `842 passed`; zero skips |
| Ruff lint | PASS: all checks passed |
| Ruff format check over complete `myclaw tests` | PASS: `162 files already formatted` |
| Strict Mypy | PASS: no issues in 162 source files |
| Diff hygiene and retired-surface scan | PASS |

## Manual And External Boundaries

| Check | Status | Reason |
| --- | --- | --- |
| Clean-wheel installation and installed CLI smoke | PASS | Performed in a new environment outside the checkout with Unicode paths. |
| Real Anthropic conversation | NOT RUN | No dedicated release credential was supplied. |
| Real OpenAI-compatible conversation | NOT RUN | No dedicated endpoint, model, and credential were supplied. |
| Public web adapter smoke | NOT RUN | This candidate gate intentionally remained offline at the application boundary. |

Automated Provider tests cover routing, streaming, tool calls, timeout conversion,
and retry behavior with injected clients. They are not represented as live service
acceptance.

## Known Boundaries

- Provider credentials remain plaintext in the user-owned configuration file;
  display paths redact them, while operating-system account access remains the
  protection boundary.
- File-first persistence serializes work within one runtime but does not promise
  coordination between multiple running MyClaw processes.
- Shell policy is a narrow command policy, not a general filesystem or network
  sandbox.
- Tool artifacts and Long-term Memory have no automatic retention limit.
- Same-user filesystem replacement can still race checks that precede an operating
  system file operation.

Every required gate passed; this Windows x64 candidate is ready for release.
