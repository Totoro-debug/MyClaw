# MyClaw Issue #2-#36 Delivery Procedure

Last updated: 2026-07-12 (Asia/Shanghai)

## Objective

Complete GitHub issues #2 through #36 with test-driven vertical slices, parallel execution on every independent frontier, batch-level verification and commits, and immediate GitHub issue updates.

This file is the delivery ledger. It was created before any implementation file. Every sub-agent must update its own issue row and append an Agent Log entry before reporting completion. The coordinating agent owns batch verification, commits, and GitHub comments/closure.

## TDD Contract

The public seams are already agreed by the user through issue #1, each child issue's acceptance criteria, and the canonical testing decisions in `docs/myclaw-runtime-contracts.md` and `docs/myclaw-personal-agent-prd.md`.

- Test only observable behavior at these public seams: CLI/REPL, Conversation Port, Management Port, Runtime Core, Memory Manager, Tool Gateway, Session Store, Model Router, and Provider Adapter.
- Work one vertical slice at a time: add one behavior test, run it and capture the expected failure (RED), add only enough implementation for that behavior, then run it to green (GREEN).
- Prefer real in-process integration through public interfaces. Mock only system boundaries such as provider SDKs, HTTP/DNS, clocks, subprocesses, and fault-injected filesystems.
- Do not test private methods, internal call counts, or implementation layout. Expected values must be literals or contract examples rather than recomputing the implementation.
- Each Agent Log entry must record the RED command/failure, GREEN command/result, changed files, and any residual risk.

## Status Protocol

1. Before editing code, the assigned agent changes only its row from `TODO` to `IN_PROGRESS` and adds its agent name.
2. The agent performs strict red-green cycles and runs the narrow issue suite.
3. Before returning, the agent changes its row to `AGENT_DONE` and appends an Agent Log entry with TDD evidence. Sub-agents do not commit or mutate GitHub issues.
4. The coordinator reviews the diff, runs the batch's narrow tests plus the full quality gate, and changes rows to `VERIFIED`.
5. The coordinator creates one batch commit, records its SHA below, comments on each completed issue with tests and commit evidence, then closes it and changes the row to `CLOSED`.
6. A dependent batch cannot start until every dependency row is `CLOSED`. Independent issues on an open frontier are dispatched to separate agents concurrently, subject only to the four-agent runtime limit; overflow starts immediately in the next parallel wave.

Status legend: `TODO` -> `IN_PROGRESS` -> `AGENT_DONE` -> `VERIFIED` -> `CLOSED`; use `BLOCKED` only with a concrete blocker in the Agent Log.

## Dependency Batches

| Batch | Parallel frontier | Gate |
|---|---|---|
| B01 | #2 | None |
| B02 | #3 | #2 |
| B03 | #4 | #3 |
| B04 | #5 | #4 |
| B05 | #6, #13 | #5 |
| B06 | #7 | #6 |
| B07 | #8 | #7 |
| B08 | #9, #11 | #8 |
| B09 | #10, #12, #15, #21, #22, #27 | #9 and/or #11 as listed below |
| B10 | #14, #16 | #12 / #15 |
| B11 | #17, #20, #25, #28 | #16 / #14 + #27 |
| B12 | #18, #19, #23, #26, #31 | #17 and issue-specific gates |
| B13 | #24, #29 | #23 / #13 + #18 + #28 |
| B14 | #30, #32 | #29 / #11 + #18 + #24 + #26 + #31 |
| B15 | #33 | #30 + #32 |
| B16 | #34 | #19 + #21 + #22 + #33 |
| B17 | #35 | #20 + #34 |
| B18 | #36 | #35 |

## Issue Ledger

| Issue | Ticket | Depends on | Batch | Public test seam | Agent | Status | Commit |
|---|---|---|---|---|---|---|---|
| #2 | T01 Installable Python project | None | B01 | installed `myclaw` CLI and reusable public test fixtures | b01_issue_2 | CLOSED | 7b73e26 |
| #3 | T02 Runtime Contracts | #2 | B02 | frozen schemas and Port/Store/Provider/Tool protocols | b02_issue_3 | CLOSED | f32893f |
| #4 | T03 Agent Home and atomic writes | #3 | B03 | Agent Home initializer, atomic file API, Workspace identity | b03_issue_4 | CLOSED | 2865d46 |
| #5 | T04 Configuration | #4 | B04 | `myclaw`, `myclaw config`, validated/redacted config API | b04_issue_5 | CLOSED | ef73b6a |
| #6 | T05 First streaming turn | #5 | B05 | Conversation Port through REPL and Session Store reload | b05_issue_6 | CLOSED | e18725d |
| #7 | T06 Multi-turn Short-term Memory | #6 | B06 | one REPL session across Conversation Port turns | b06_issue_7 | CLOSED | 9e95271 |
| #8 | T07 Model route fallback | #7 | B07 | Model Router route resolution and startup validation | - | TODO | - |
| #9 | T08 Retry budget | #8 | B08 | Model Router with fake Provider and fake Clock | - | TODO | - |
| #10 | T09 Failed/cancelled turns | #9 | B09 | Conversation Port terminal events and reloadable Session Store | - | TODO | - |
| #11 | T10 Session metadata/status | #8 | B08 | Session Store metadata and Management Port `/status` | - | TODO | - |
| #12 | T11 Async session title | #9, #11 | B09 | Conversation Port latency/history plus Session Store metadata | - | TODO | - |
| #13 | T12 Config/memory views | #5 | B05 | Management Port `/config` and `/memory` | b05_issue_13 | CLOSED | e18725d |
| #14 | T13 Resume session | #12 | B10 | Management Port `/resume` and Session Store recovery | - | TODO | - |
| #15 | T14 Read-only file tool loop | #9 | B09 | Tool Gateway through full Conversation Port model loop | - | TODO | - |
| #16 | T15 Tool result/failure semantics | #15 | B10 | Tool Gateway normalization and Agent Events | - | TODO | - |
| #17 | T16 Permission request loop | #16 | B11 | Conversation Port permission response and Tool Gateway | - | TODO | - |
| #18 | T17 Safe Workspace writes | #17 | B12 | Tool Gateway write/edit behavior at filesystem boundary | - | TODO | - |
| #19 | T18 Interrupted tool-call repair | #10, #17 | B12 | cancelled Conversation Port turn and reloaded Session Store | - | TODO | - |
| #20 | T19 Tool Artifacts | #16 | B11 | Tool Gateway result plus Session Store/artifact files | - | TODO | - |
| #21 | T20 Anthropic Provider | #9 | B09 | Provider Adapter with fake official SDK | - | TODO | - |
| #22 | T21 OpenAI-compatible Provider | #9 | B09 | Provider Adapter with fake official SDK | - | TODO | - |
| #23 | T22 Shell permission policy | #17 | B12 | Tool Gateway Shell policy/catalog | - | TODO | - |
| #24 | T23 Shell process lifecycle | #23 | B13 | Shell tool subprocess boundary | - | TODO | - |
| #25 | T24 WebSearch | #16 | B11 | Tool Gateway WebSearch with fake search boundary | - | TODO | - |
| #26 | T25 SSRF-resistant WebFetch | #25 | B12 | WebFetch HTTP/DNS/peer boundary through Tool Gateway | - | TODO | - |
| #27 | T26 Conversation Summary | #9, #11 | B09 | Runtime Core/Memory Manager with fake Provider | - | TODO | - |
| #28 | T27 Consolidation recovery | #14, #27 | B11 | Session Store/summary journal fault-injection API | - | TODO | - |
| #29 | T28 Dream memory update | #13, #18, #28 | B13 | Management Port `/dream` and restricted Memory Manager | - | TODO | - |
| #30 | T29 Periodic Memory Task | #29 | B14 | Memory scheduler with fake Clock and Management Port | - | TODO | - |
| #31 | T30 Scheduled Work persistence | #17 | B12 | Tool Gateway permission plus Scheduled Work Store | - | TODO | - |
| #32 | T31 Scheduled Work execution | #11, #18, #24, #26, #31 | B14 | scheduler through Runtime Core and dedicated Session Store | - | TODO | - |
| #33 | T32 Background coordination | #30, #32 | B15 | Runtime event ordering with foreground/background tasks | - | TODO | - |
| #34 | T33 Runtime shutdown | #19, #21, #22, #33 | B16 | REPL/Runtime lifetime and owned resource boundaries | - | TODO | - |
| #35 | T34 Security/fault review | #20, #34 | B17 | end-to-end public seams with boundary fault injection | - | TODO | - |
| #36 | T35 Cross-platform release | #35 | B18 | built wheel, clean installs, CLI smoke, release evidence | - | TODO | - |

## Quality Gate

Every batch must pass the commands established by #2. Until #2 freezes exact commands, the intended gate is:

```text
pytest
ruff check .
ruff format --check .
mypy src tests
python -m build
```

Issue #36 additionally requires Windows and POSIX evidence, clean-wheel installation, `myclaw`/`myclaw config` smoke tests, traceability for all 48 user stories and required tests, manual acceptance status, real-provider smoke status, and known risks.

## Batch Commit And Issue Log

| Batch | Issues | Verification | Commit | GitHub update | Status |
|---|---|---|---|---|---|
| B01 | #2 | 11 pytest + Ruff lint/format + strict mypy + sdist/wheel + CLI smoke | 7b73e26 | #2 commented and closed | CLOSED |
| B02 | #3 | 66 pytest + Ruff lint/format + strict mypy + sdist/wheel | f32893f | #3 commented and closed | CLOSED |
| B03 | #4 | 77 pytest + Ruff lint/format + strict mypy + sdist/wheel | 2865d46 | #4 commented and closed | CLOSED |
| B04 | #5 | 141 pytest + Ruff lint/format + strict mypy + sdist/wheel + installed CLI smoke | ef73b6a | #5 commented and closed | CLOSED |
| B05 | #6, #13 | 169 pytest + Ruff lint/format + strict mypy + sdist/wheel + combined REPL/management integration | e18725d | #6 and #13 commented and closed | CLOSED |
| B06 | #7 | 173 pytest + Ruff lint/format + strict mypy + sdist/wheel + independent contract review | 9e95271 | #7 commented and closed | CLOSED |
| B07 | #8 | - | - | - | TODO |
| B08 | #9, #11 | - | - | - | TODO |
| B09 | #10, #12, #15, #21, #22, #27 | - | - | - | TODO |
| B10 | #14, #16 | - | - | - | TODO |
| B11 | #17, #20, #25, #28 | - | - | - | TODO |
| B12 | #18, #19, #23, #26, #31 | - | - | - | TODO |
| B13 | #24, #29 | - | - | - | TODO |
| B14 | #30, #32 | - | - | - | TODO |
| B15 | #33 | - | - | - | TODO |
| B16 | #34 | - | - | - | TODO |
| B17 | #35 | - | - | - | TODO |
| B18 | #36 | - | - | - | TODO |

## Agent Log

Append one row per completed or blocked assignment. Keep evidence concise but reproducible.

| Timestamp | Agent | Issue | RED evidence | GREEN evidence | Files / notes |
|---|---|---|---|---|---|
| 2026-07-12 09:16 +08:00 | b01_issue_2 | #2 | `python -m pytest tests/test_package.py -q` -> `ModuleNotFoundError: myclaw`; `python -m pytest tests/test_cli.py -q` -> console executable absent; fixture slices failed first with missing modules/fixtures, missing provider complete/close, and Tool failure not raised; `ruff format --check .` -> 1 file and `python -m mypy src tests` -> 2 errors. | Offline editable install succeeded; `pytest` -> 11 passed; `ruff check .` -> passed; `ruff format --check .` -> 15 files formatted; `mypy src tests` -> 15 files clean; `python -m build --no-isolation` -> wheel + sdist; offline wheel install and `myclaw --help` -> exit 0. | `.gitignore`, `README.md`, `pyproject.toml`, `src/myclaw/`, `tests/`; residual: host setuptools emits an unrelated deprecated `upload_docs` entry-point warning during no-isolation builds, but artifacts build successfully. |
| 2026-07-12 09:35 +08:00 | contract_audit | #3 | N/A - read-only audit. | N/A - read-only audit. | Checked runtime contracts: UUID4 rule conflicts with the Scheduled Work session example; accepted contract invalidates the whole Scheduled Work file despite implementation-plan tolerance; Management/model result shapes and exact event ordering remain underdefined. |
| 2026-07-12 11:24 +08:00 | b02_issue_3 | #3 | `python -m pytest tests/contract/test_common_contracts.py -q` -> `ModuleNotFoundError: myclaw.contracts`; subsequent vertical slices failed first with missing contract exports, invalid counters/UUID1/status combinations accepted (`DID NOT RAISE`), unimplemented error/artifact branches, and missing terminal validators; `python -m mypy src/myclaw/contracts tests/contract tests/fixtures/provider.py tests/fixtures/tool.py` -> 37 errors including Provider/Tool Protocol incompatibility. | Narrow contract + fake suite -> 42 passed; static Protocol slice -> 23 files clean; `python -m pytest -q` -> 49 passed; `python -m ruff check .` -> passed; `python -m ruff format --check .` -> 36 files formatted; `python -m mypy src tests` -> 36 files clean; `python -m build --no-isolation` -> wheel + sdist. | `src/myclaw/contracts/`, `tests/contract/`, typed provider/Tool fakes and tests, `docs/myclaw-runtime-contracts.md`; corrected the Scheduled Work UUID1 example to normative UUID4. Residual: non-persisted model/management shapes use the smallest documented fields and underdefined event status/action/kind values remain strings; Accepted Scheduled Work whole-file invalidation overrides the stale implementation-plan tolerance; build retains the known host `upload_docs` warning. |
| 2026-07-12 11:30 +08:00 | standards_axis | #3 | N/A - read-only standards review. | N/A - read-only standards review. | Findings: invalid five-token cron accepted; Artifact path and Session title invariants not enforced; background completion can interleave with foreground streaming; possible duplicated usage validation. |
| 2026-07-12 11:48 +08:00 | b02_issue_3 | #3 | Cron: `python -m pytest tests/contract/test_memory_scheduling_contracts.py::test_scheduled_work_rejects_range_invalid_five_field_cron -q` -> `DID NOT RAISE` for `99 99 99 99 99`; artifact path matrix -> 4 failed/3 passed (wrong root, invalid Session ID, raw unsafe character, empty encoded ID accepted); title matrix -> 6 failed (metadata/update accepted empty, unnormalized, and 61-code-point titles); event `-k background` -> 1 failed/1 passed (interleaving accepted, background-only valid). | Cron contract -> 5 passed; artifact/session contracts -> 23 passed; title/session contracts -> 20 passed; event contracts -> 6 passed; `python -m pytest -q` -> 66 passed; `python -m ruff check .` -> passed; `python -m ruff format --check .` -> 36 files formatted; `python -m mypy src tests` -> 36 files clean; `python -m build --no-isolation` -> wheel + sdist. | Added `croniter>=6,<7`; exact artifact path/encoding grammar; normalized 60-code-point Session titles; foreground/background event exclusion. Residual: croniter 6.2.2 lacks typing metadata, isolated with `# type: ignore[import-untyped]`; build retains the known host `upload_docs` warning. |
| 2026-07-12 12:18 +08:00 | b03_issue_4 | #4 | RED command form was `python -m pytest <file>::<node> -q`. Atomic nodes `test_failed_atomic_bytes_replace_preserves_official_state` and `test_atomic_text_replace_writes_exact_utf8_bytes` failed with missing module/export. Agent Home nodes `test_production_agent_home_is_fixed`, `test_first_initialization_creates_only_base_state`, and `test_repeated_initialization_preserves_long_term_memory` failed with missing module, missing `initialize`, and the observed overwrite. Workspace nodes `test_windows_drive_workspace_has_the_accepted_identity_and_slug`, `test_posix_workspace_omits_the_root_from_its_slug`, `test_unc_workspace_has_the_accepted_identity_and_slug`, and `test_native_workspace_is_absolutized_and_lexically_normalized` failed with missing module, `ValueError`, wrong UNC slug, and relative-path `ValueError`. | Each identical narrow node command -> 1 passed after its minimal slice; `python -m pytest tests/test_agent_home.py tests/test_atomic_files.py tests/test_workspace.py -q` -> 11 passed; `python -m pytest -q` -> 77 passed; `python -m ruff check .` -> passed; `python -m ruff format --check .` -> 42 files already formatted; `python -m mypy src tests` -> 42 files clean; `python -m build --no-isolation` -> wheel + sdist. | `src/myclaw/{agent_home,atomic_files,workspace}.py`, matching three public-seam test modules, and this ledger. Residual: parent-directory fsync is intentionally best-effort and unavailable on this Windows host, so POSIX directory-fsync durability remains platform-dependent; no cross-process locking by accepted design; build retains the known host `upload_docs` warning. |
| 2026-07-12 13:03 +08:00 | b04_issue_5 | #5 | Vertical RED commands used `python -m pytest <node> -q`: missing template -> `ModuleNotFoundError: myclaw.config`; typed load -> missing `load`; 35-case schema matrix -> missing `ConfigError`; 13-case route matrix -> missing `resolve_route`; parsed/malformed/schema-invalid views -> missing `view`, uncaught `TOMLDecodeError`, then absent validation error; installed bare/config CLIs -> exit 0 placeholder and `No such command 'config'`; atomic publication fault -> raw `OSError`. | Each node passed after its minimal slice; `python -m pytest tests/test_config.py tests/test_cli.py -q` -> 63 passed; `python -m pytest -q` -> 139 passed; `python -m ruff check .` and `python -m ruff format --check .` -> passed/44 files formatted; `python -m mypy src tests` -> 44 files clean; `python -m build --no-isolation` -> wheel + sdist; isolated offline wheel smokes -> missing 2, config-missing 0, valid 0, config-valid 0, invalid 2, config-invalid 2. | `src/myclaw/{config,cli,atomic_files}.py`, `tests/test_{config,cli}.py`, and this ledger. Residual: no cross-process locking by accepted design; atomic create-without-overwrite depends on same-filesystem hard-link support; template IDs use the accepted full-schema examples `anthropic-default` and `openai-local`; build retains the known host `upload_docs` warning. |
| 2026-07-12 13:19 +08:00 | b04_issue_5 | #5 | Review REDs: `python -m pytest tests/test_config.py::test_config_view_redacts_only_the_api_key_when_its_text_is_reused -q` showed global substitution corrupting the comment, provider table, URL, catalog, and routes; `python -m pytest tests/test_config.py::test_config_view_redacts_multiline_api_key_in_valid_dotted_toml -q` safely failed with `ConfigView leaked a multiline plaintext provider API key` for tomlkit's dotted-key proxy. | Both identical node commands -> passed after targeted structured edits; config/CLI suite -> 65 passed; full pytest -> 141 passed; Ruff lint/format -> passed/44 files; strict mypy -> 44 files clean; offline wheel + sdist build -> passed; isolated installed-wheel overlap/multiline CLI smokes -> exit 0/0 with no secret fragments and preserved unrelated text. | `pyproject.toml`, `src/myclaw/config.py`, `tests/test_config.py`, and this ledger. Added source-preserving `tomlkit>=0.13,<0.14` (verified 0.13.3) while retaining `tomllib` for accepted schema validation; installed wheel metadata verified the dependency. Residual: build retains the known host `upload_docs` warning. |
| 2026-07-12 13:51 +08:00 | b05_issue_13 | #13 | Public-seam nodes failed in sequence with missing `myclaw.management`, absent `memory_view`, raw decode/read exceptions, missing `myclaw.management_commands`, `/memory` and unknown slash `NotImplementedError`, omitted config diagnostics, and escaped persistence errors. | Each identical node passed after its minimal slice; `python -m pytest tests/test_management_views.py tests/test_management_commands.py -q` -> 17 passed; config + management suite -> 74 passed; management contract suite -> 4 passed; owned Ruff lint/format and strict mypy -> clean; combined B05 `pytest -q` -> 165 passed, format -> 54 files, offline wheel + sdist -> passed. | `src/myclaw/{management,management_commands}.py`, `tests/test_management_{views,commands}.py`, and this ledger. Complete fresh disk views, stable persistence errors, exact dispatch, and conversation/provider bypass are covered; no CLI/session/runtime/config edits. Residual: combined Ruff import-order and mypy failures were exclusively in concurrently unfinished #6 files and were handed to that agent for the authoritative B05 rerun; build retains the known host `upload_docs` warning. |
| 2026-07-12 14:02 +08:00 | b05_issue_6 | #6 | Public-seam slices failed first with missing `myclaw.session_store`, unsupported assistant reload, missing `myclaw.conversation`, missing REPL/runtime modules, nonblank REPL `NotImplementedError`, absent management hook, absent console input/writer and fail-closed provider factory; first full static gate then exposed 2 format files and 1 callback typing error. | Each identical node passed after its minimal implementation; owned Session Store/Conversation/REPL/runtime/CLI suite -> 19 passed; `python -m pytest -q` -> 169 passed; `python -m ruff check .` -> passed; `python -m ruff format --check .` -> 56 files formatted; `python -m mypy src tests` -> 56 files clean; `python -m build --no-isolation` -> wheel + sdist. | `src/myclaw/{cli,conversation,repl,runtime,session_store}.py`, `tests/test_{conversation,repl,runtime,session_store}.py`, and this ledger. Covers lazy exact JSONL, valid reload, progressive ordered deltas, one completed assistant/terminal, whitespace/EOF, structural #13 dispatch hook, and deferred injectable provider factory. Residual: failure/cancellation, history/system context, metadata updates/listing, and real provider adapters remain their assigned later tickets; Windows extended-length paths are internal only; build retains the known host `upload_docs` warning. |
| 2026-07-12 14:33 +08:00 | b06_issue_7 | #7 | `python -m pytest tests/test_conversation.py::test_consecutive_turns_send_raw_short_term_memory_and_wrap_only_current_input -q` -> request 1 contained only raw current input and request 2 omitted history; `python -m pytest tests/test_runtime.py::test_prepared_repl_reuses_one_session_and_its_startup_system_context -q` -> system prompt was empty; `python -m pytest tests/test_repl.py::test_repl_exit_and_quit_ignore_case_and_whitespace_without_materializing_messages -q` -> padded mixed-case `ExIt` entered the Conversation Port; `python -m pytest tests/test_conversation.py::test_conversation_port_rejects_an_overlapping_foreground_submit -q` -> `DID NOT RAISE RuntimeError`. | Each identical node -> 1 passed after its minimal slice; combined Conversation Port/REPL/runtime/Session Store/management suite -> 27 passed; `python -m pytest -q` -> 173 passed; `python -m ruff check .` -> passed; `python -m ruff format --check .` -> 57 files formatted; `python -m mypy src tests` -> 57 files clean; `python -m build --no-isolation` -> wheel + sdist. | `src/myclaw/{conversation,prompts,repl,runtime}.py`, `tests/test_{conversation,repl,runtime}.py`, and this ledger. Covers raw ordered Short-term Memory, current-only Runtime Context, fixed System Prompt composition and startup memory snapshot, exact exit aliases, unknown slash persistence, and one foreground submit. Residual: Tool guidance is empty until the Tool Gateway slice; tool history, compression, routing/retry, failure/cancellation, title/status, and background work remain their assigned tickets; build retains the known host `upload_docs` warning. |
| 2026-07-12 14:40 +08:00 | b06_review | #7 | N/A read-only review; no implementation RED/GREEN cycle. | CodeGraph-first call-path review against `CONTEXT.md`, `docs/myclaw-runtime-contracts.md`, and GitHub #7; `python -m pytest tests/test_conversation.py tests/test_repl.py tests/test_runtime.py -q` -> 13 passed. | Scope: ordered raw Short-term Memory and current-only Runtime Context, startup System Prompt/Workspace/memory snapshot, case-insensitive padded `exit`/`quit`, unknown slash pass-through, and foreground concurrency guard. No blocking findings. Residual for #10: keep the guard active until the cancelled stream emits its terminal outcome or unwinds, and retain an active stream/task handle for `cancel_active_turn`; the current `finally` correctly releases the guard on completion, error, or generator cancellation. |
