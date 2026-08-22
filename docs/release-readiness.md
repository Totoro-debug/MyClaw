# MyClaw Runtime Release Readiness (#173)

Status: **evidence recorded for the final runtime migration; no release was uploaded**

Verification date: 2026-08-22
Repository: `Totoro-debug/myclaw`
Fixed code baseline: `047e6894039d6942ee4b8e488f17b378f459fa24`
Verification host: Windows, Python 3.12
Review scope: the final #173 documentation update and one baseline format correction over
the fixed baseline; verification was recorded before the task's sole final commit.

This file is the current release evidence index for Issues #163–#173. The former Schedule
candidate record is retained as historical context, but its old READY/PENDING labels are
not the current runtime status. No CI result, live Provider conversation, manual Windows
Terminal observation, or package upload is implied by this document.

## Current runtime contract

The current domain vocabulary is deliberately narrow:

- **Message Bus** is the Agent Loop-owned transient transport with one Inbound FIFO and
  one Outbound FIFO. Its six async operations are `inbound_snapshot`, `put_inbound`,
  `get_inbound`, `drain_inbound`, `put_outbound`, and `get_outbound`. It has no independent
  close, abort, replay, broadcast, version, or backpressure lifecycle.
- **Inbound Message** is an ordinary user message entering the FIFO. Inbound mutations
  invoke the single callback after queue coordination is released and preserve FIFO order.
- **Outbound Message** is a live projection of selected foreground output. Its only types
  are `model_reasoning`, `model_response`, `tool_call`, and `system_control`. The only
  sparse markers are mutually exclusive `_stream_delta`, `_stream_end`, and `_streamed`.
  There is one Outbound consumer. Tool results never enter Outbound.
- **Agent Loop** owns foreground orchestration, the active Session, Message Bus, fixed Tool
  catalog, shared Gateway, Runner invocation, persistence, and control boundaries.
- **Agent Runner** is the reusable Session-independent bounded model/Tool executor; its
  constructor owns only the Model Router.
- **Agent Run** is the domain execution concept from input acceptance through
  Summary/context, Runner execution, Session increment, and persistence request. It is
  not a Python `AgentRun` class or a compatibility alias for a deleted legacy type.
- **Runtime Generation** is the process-level replacement unit containing all
  Session-bound runtime state. `RuntimeHost` prepares and validates a target before
  synchronously aborting and replacing the old generation.
- **Tool Confirmation** is one pending Future for one foreground Tool call. Schedule runs
  pass `confirmation=None`, so confirmation-required Tools refuse.
- **Schedule Service** is the sole Schedule Store and management owner. It calls the Agent
  Loop callback for isolated Schedule execution.
- **Tool Artifact** is the successful oversized-result externalization at
  `.myclaw/artifacts/<session_id>/<tool_call_id>.txt`, with the unchanged
  `path`/`total_chars`/`preview_chars` reference shape.

Runner and lifecycle invariants:

- One iteration is one model call followed, in provider order, by all sequential Tool
  calls from that response. Provider retries do not consume iterations.
- `runtime.max_iterations` defaults to 50 and values below 50 are invalid. The fiftieth
  iteration completes its Tools and, unless normal cancellation has been requested,
  returns `agent_iteration_limit` without a fifty-first model call. Cancellation takes
  priority. The user-facing limit text is:
  `MyClaw 本轮对话已经达到最大循环次数，仍没有输出最终结果。可以再次尝试本次请求或者尝试给出更明确的任务目标。`
  Normal cancellation returns `turn_cancelled` and:
  `MyClaw 已取消本轮对话。`
- Provider-visible reasoning can be streamed when returned by the Provider. Opaque
  continuation state remains inside the Runner's current Tool loop; neither is a claim of
  raw hidden chain-of-thought, and continuation is not persisted in Session or Outbound.
- Foreground and Schedule share Gateway/Runner identity but isolate Session, context,
  cancellation, and externalizer state. Schedule has no confirmation channel or
  foreground Message Bus projection, sets the recursive-add `ContextVar`, resets its token
  in `finally`, and uses `confirmation=None`.
- Ordinary Session snapshots are ordered complete replacements with at most three async
  attempts and `100 ms`/`200 ms` backoff. Normal awaited close performs the bounded final
  save. `Session.abandon()` is synchronous and idempotent, cancels pending snapshots, and
  performs no final save or retry.
- `PreparedRuntime.close()` is normal awaited shutdown. `PreparedRuntime.abort()` is the
  forced synchronous replacement path. Detached Provider cleanup is best effort with
  failure logging; this is an accepted tradeoff of immediate generation isolation.

## Migration acceptance evidence: Issues #163–#172

Each row names the focused public seams and the regression suites that cover the issue's
acceptance. The final complete-suite result below is the gate for all rows.

| Issue | Final acceptance evidence in the working tree |
| --- | --- |
| #163 Message Bus | `tests/agent/test_message_bus.py`, `tests/agent/test_loop.py`, `tests/terminal/test_repl_bus.py`; six operations, FIFO/callback behavior, one consumer, four types, sparse markers, and no Tool-result Outbound are asserted. |
| #164 Provider reasoning | `tests/provider/test_models.py`, `tests/test_anthropic_provider.py`, `tests/test_openai_compatible_provider.py`, `tests/test_model_router.py`; visible reasoning and provider-owned continuation are covered, including same-provider retry and explicit non-replay after fallback, plus usage/errors/shutdown. |
| #165 Session retry/abandon | `tests/sessions/test_session.py`, `tests/sessions/test_session_io_security.py`, `tests/test_release_contract.py`; ordered JSONL retry, later authoritative recovery, cancellation of every pending snapshot, normal final save, abandon no-save, and filesystem safety are covered. |
| #166 Schedule Service | `tests/scheduling/test_schedule_service.py`, `tests/scheduling/test_schedule_service_boundary.py`, `tests/scheduling/test_schedule_store.py`, `tests/scheduling/test_schedule_model.py`, `tests/tools/core/test_schedule.py`, `tests/test_session_log.py`; sole Store/management ownership, task-local recursion, concurrent actions, cancellation, Job state, and Schedule Session Log partition are covered. |
| #167 bounded Agent Runner | `tests/agent/test_runner.py`, `tests/configuration/test_agent_runner_config.py`; result/usage invariants, sequential Tools, provider retry accounting, Provider-valid atomic cancellation repair, exact text, 50th-Tool cancellation priority, and no call 51 are covered. |
| #168 Agent Loop | `tests/agent/test_loop.py`, `tests/agent/test_fixed_catalog_runtime.py`, `tests/agent/test_context.py`; single FIFO consumer, preparation/cancellation, title coordination, fixed catalog, commit/persist, controls, and sparse projection are covered. |
| #169 Schedule through Agent Loop | `tests/agent/test_schedule_loop.py`, `tests/scheduling/test_schedule_runtime.py`, `tests/scheduling/test_schedule_service.py`, `tests/tools/core/test_schedule.py`; callback and shared identities, isolated state, Schedule clock/Session Log, cancellation, `confirmation=None`, Memory, and Artifact behavior are covered. |
| #170 Terminal over Message Bus | `tests/terminal/test_conversation.py`, `tests/terminal/test_repl_bus.py`, `tests/terminal/test_keyboard.py`, `tests/test_repl_confirmation.py`, `tests/test_cli.py`; active input, atomic drain/queue races, sparse markers, sole consumer, confirmation restoration, Ctrl+C, exit, and installed non-TTY behavior are covered. |
| #171 legacy transport removal | `tests/architecture/test_module_boundaries.py`, `tests/test_release_contract.py`, the exact-symbol/module audit below, and wheel-content audit; deleted transports, payloads, emitter/bridge, modules, exports, and Python `AgentRun` type are absent without misclassifying `AgentRunner` or domain `Agent Run`. |
| #172 Runtime Generation replacement | `tests/test_runtime_generation.py`, `tests/test_runtime_active_session.py`, `tests/test_runtime_session_title.py`, `tests/test_runtime_shutdown.py`, `tests/test_runtime.py`, `tests/sessions/test_session_resume.py`; preflight, same-session no-op, pending-only discard, active prompt/decline/approve, abort task ownership, late old output isolation, identities, detached cleanup, and normal close are covered. |

## Release commands and results

Commands were run in this worktree on the fixed baseline plus the final #173
documentation/format diff:

| Gate | Command | Result |
| --- | --- | --- |
| Complete regression | `python -m pytest -q -ra` | **PASS** — `1150 passed, 8 skipped in 263.14s` |
| Lint | `python -m ruff check .` | **PASS** — `All checks passed!` |
| Format | `python -m ruff format --check .` | **PASS** — `162 files already formatted` |
| Production/tests typing | `python -m mypy --strict myclaw tests` | **PASS** — no issues in 162 source files |
| Patch hygiene | `git diff --check` | **PASS** — no whitespace errors |
| Source distribution/wheel | `python -m build` | **PASS** — `myclaw-0.1.0.tar.gz` and `myclaw-0.1.0-py3-none-any.whl` built |
| Archive contents | exact `zipfile`/`tarfile` command below | **PASS** — wheel 88 entries, sdist 140 entries; no cache/build pollution or deleted #171 modules |
| Clean installation | `python -m venv .release-verify/venv`; venv Python `-m pip install` of the wheel | **PASS** — dependency resolution and installation completed |
| Dependency consistency | `.release-verify/venv/Scripts/python.exe -m pip check` | **PASS** — `No broken requirements found.` |
| Installed import | exact outside-checkout import command below | **PASS** — final public imports resolved from the venv, not the checkout |
| Installed configuration/CLI | exact installed-process commands below | **PASS** — `--help` and `config` exit 0; missing config generates once and exits 2 with `config_missing` |
| Installed non-TTY | second installed `myclaw` launch after generated config | **PASS** — exit 2 with one `interactive_terminal_required`; no Workspace State was created |

The package/install/CLI rows use this reproducible PowerShell sequence from the repository
root (the final cleanup removes `.release-verify`, `build`, `dist`, and `*.egg-info`):

```powershell
python -m build
$wheel = (Get-ChildItem -LiteralPath dist -Filter 'myclaw-*.whl' -File).FullName
python -c 'from pathlib import Path; import tarfile, zipfile; wheel=next(Path("dist").glob("myclaw-*.whl")); sdist=next(Path("dist").glob("myclaw-*.tar.gz")); wn=zipfile.ZipFile(wheel).namelist(); tn=tarfile.open(sdist).getnames(); stale=("myclaw/agent/events.py","myclaw/agent/run.py","myclaw/agent/ports.py","myclaw/session/conversation.py","myclaw/session/session_resume.py","myclaw/terminal/_turn_stream.py"); assert not [x for x in wn if any(x.endswith(y) for y in stale)]; assert not [x for x in tn if any(x.endswith(y) for y in stale)]; assert not [x for x in wn+tn if x.endswith(".pyc") or "__pycache__" in x or "/build/" in x or "/dist/" in x]; print(len(wn), len(tn))'
$verify = Join-Path (Resolve-Path .) '.release-verify'
python -m venv (Join-Path $verify 'venv')
$venvPython = Join-Path $verify 'venv\Scripts\python.exe'
& $venvPython -m pip install $wheel
& $venvPython -m pip check
$outside = New-Item -ItemType Directory -Force -Path (Join-Path $verify 'outside')
Push-Location $outside
$env:PYTHONPATH = $null
& $venvPython -c 'from pathlib import Path; import sys, myclaw; from myclaw.agent.message_bus import MessageBus; from myclaw.agent.loop import AgentLoop; from myclaw.agent.runner import AgentRunner; from myclaw.agent.runtime import PreparedRuntime, RuntimeHost; from myclaw.schedule.service import ScheduleService; assert Path(sys.prefix).resolve() in Path(myclaw.__file__).resolve().parents; print(myclaw.__file__)'
$installedMyClaw = Join-Path $verify 'venv\Scripts\myclaw.exe'
& $installedMyClaw --help
$verifyHomeDir = New-Item -ItemType Directory -Force -Path (Join-Path $verify 'home')
$verifyWorkspaceDir = New-Item -ItemType Directory -Force -Path (Join-Path $verify 'workspace')
$env:HOME = $verifyHomeDir.FullName
$env:USERPROFILE = $verifyHomeDir.FullName
Push-Location $verifyWorkspaceDir
& $installedMyClaw                       # exit 2; config_missing; creates config.toml
& $installedMyClaw config                # exit 0; redacted generated configuration
& $installedMyClaw                       # exit 2; interactive_terminal_required
Pop-Location
Pop-Location
```

### PTY and terminal condition

The existing PTY smoke is `tests/test_cli.py::test_installed_wheel_terminal_conversation_pseudo_terminal_smoke`.
It was not falsely marked as passed: on this Windows Python runtime the test is skipped
because `termios`/`pty` are unavailable. The full suite's exact skip is:

`The Windows Python runtime has no termios/pty harness; use the Windows Terminal matrix.`

The executable alternative on this host is the installed non-TTY check above plus the
Textual headless startup, product-path exit, cancellation, Ctrl+C, and restoration tests
covered by the full suite. A POSIX PTY run and a physical Windows Terminal matrix remain
environmental follow-ups; no full-screen startup or terminal restoration claim is made
for them here.

The other seven skips are Windows symlink/file-link characterization cases, all reported
by pytest with the Windows `WinError 1314` privilege condition. There are no xfails or
expected failures, and no new skip was added for #173.

## Regression coverage index

| Required regression area | Evidence |
| --- | --- |
| Session JSONL, usage, persist, abandon | `tests/sessions/test_session.py`, `tests/sessions/test_session_io_security.py`, `tests/test_release_contract.py` |
| Title and Summary | `tests/test_runtime_session_title.py`, `tests/memory/test_conversation_summary.py` |
| Fixed Tool catalog and confirmation safety | `tests/agent/test_fixed_catalog_runtime.py`, `tests/tools/test_fixed_tool_gateway.py`, `tests/test_repl_confirmation.py`, `tests/test_permission_loop.py` |
| Artifact root and reference shape | `tests/tools/test_base_tool.py`, `tests/tools/test_fixed_tool_gateway.py`, `tests/agent/test_schedule_loop.py` |
| Provider streaming/reasoning/continuation/errors/shutdown | `tests/provider/test_models.py`, `tests/test_anthropic_provider.py`, `tests/test_openai_compatible_provider.py`, `tests/test_model_router.py`, `tests/test_runtime_shutdown.py` |
| Schedule Store/Job/shared execution | `tests/scheduling/`, `tests/agent/test_schedule_loop.py`, `tests/scheduling/test_schedule_runtime.py` |
| Memory independence | `tests/memory/`, `tests/scheduling/test_schedule_runtime.py` |
| Management commands | `tests/management/`, `tests/test_repl.py`, `tests/terminal/test_conversation.py` |
| Ctrl+C and normal cancellation | `tests/test_repl.py`, `tests/test_repl_confirmation.py`, `tests/terminal/test_conversation.py`, `tests/test_runtime_shutdown.py` |
| Resume and generation replacement | `tests/sessions/test_session_resume.py`, `tests/test_runtime_generation.py`, `tests/test_runtime_active_session.py` |
| Normal shutdown and terminal cleanup | `tests/test_runtime_shutdown.py`, `tests/test_runtime.py`, `tests/terminal/test_conversation.py`, `tests/test_cli.py` |

## Stale-document and removal audit

The first audit was intentionally RED. It found missing final contract details in
`CONTEXT.md` and runtime contracts, stale read-only/hidden-Tool-argument claims in the
Terminal design, old Runtime Core/Schedule ownership wording, a migration-status message
bus design, historical ADRs without a clear current boundary, and a Schedule-only
release-readiness record. The baseline `python -m ruff format --check .` was also RED for
one pre-existing line-wrap mismatch in `myclaw/memory/conversation_summary.py`.
The independent review's first complete pytest run was RED only because this record had
dropped the exact Session Log risk-contract wording asserted by
`tests/test_release_contract.py`; the wording was restored before the final complete run.

The minimal GREEN audit was then rerun with these reproducible checks. The regex uses the
exact deleted class/symbol spellings, so it does not misclassify `AgentRunner` or the legal
domain term `Agent Run`:

```powershell
$legacy = 'ConversationPort|AgentEvent|PreparedReplRuntime|AgentRunEmitter|_ConversationEventBridge|(^|[^A-Za-z])AgentRun([^A-Za-z]|$)'
rg -n $legacy myclaw tests
rg -n -i $legacy CONTEXT.md docs
Test-Path myclaw/agent/events.py,myclaw/agent/run.py,myclaw/agent/ports.py,myclaw/session/conversation.py,myclaw/session/session_resume.py,myclaw/terminal/_turn_stream.py
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict myclaw tests
```

The source/test `rg` command returned no matches and every `Test-Path` result was `False`.
The document command's remaining exact legacy-name hits are negative removal statements or
confined to migration plans/ADR text with an explicit historical/superseded boundary; none
claims an active API. ADR-0010's fixed Tool catalog remains authoritative. ADR-0011 and
ADR-0013 retain their original historical transport text, while ADR-0014 explicitly
supersedes that transport and adds the final post-#172 amendment.

The only code change in this release-closure task is the one-file Ruff formatting correction
required to turn the baseline format gate GREEN. No production behavior, test assertion,
schema, or data migration was changed.

## Source, wheel, and composition audit

- The foreground composition has one production `AgentLoop(...)` construction in
  `myclaw/agent/runtime.py`; `AgentLoop.__init__` constructs the fixed Gateway, Runner,
  and Message Bus. Schedule Service is assigned exactly one
  `on_schedule_job = agent_loop.run_schedule_job` callback.
- Source search confirms the intended call path: Terminal Conversation -> Agent Loop ->
  Agent Runner, and Schedule Service -> Agent Loop callback -> the same Gateway/Runner
  identities with isolated per-run state.
- Memory Task's `ToolGateway._for_memory(...)` is the intentional separate boundary with
  only its dedicated memory capabilities; it does not create a third foreground/Schedule
  execution path.
- Exact deleted symbols/modules/exports are absent from `myclaw/` and `tests/`; no
  compatibility alias or Python `AgentRun` class was restored.
- The built wheel and sdist contain only current package/source entries. Their archive
  audits found no `.pyc`, `__pycache__`, stale legacy modules, or build directories.
- Build output, the temporary venv, and temporary release workspaces were removed after
  verification and are not part of the final worktree.

## Accepted risks and unverified environments

- Windows cannot provide the repository's POSIX PTY harness in this environment. Native
  Windows Terminal behavior, macOS Intel/Apple Silicon, and a POSIX PTY smoke remain
  unverified.
- No live Provider/network conversation, CI job, manual keyboard matrix, package upload,
  or GitHub release was performed.
- Runtime replacement intentionally favors immediate isolation: unpersisted state can be
  lost, detached Provider cleanup is best effort, and Schedule side effects are at-least-once.
- Message Bus FIFOs are unbounded; same-Session concurrency remains unsupported.
- Session Log accepted risks remain: same-Session concurrency is unsupported;
  unbounded queue and infinite drain; no per-record fsync; no active redaction;
  no control escaping; per-Session retention;
  legacy Agent Home Runtime Log files remain untouched.

The older Schedule candidate evidence for Issues #69/#92/#104/#117 remains in Git history
and prior documentation for traceability. It is not used as evidence for this final #173
runtime gate.
