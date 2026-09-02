# MyClaw Release Readiness

Status: **Issues #194, #195, #199, #201, and #202 verified through `7c84fcb8a05e015f8ffcbb530b7c801de19a4a70` on 2026-08-28; no release was uploaded**

This document is the evidence index for the current implementation. It records active
contracts, current test locations, current verification results, and present release risks.

## Current Runtime Contract

| Area | Current boundary | Primary evidence |
| --- | --- | --- |
| Process and Terminal | `myclaw.terminal.cli` composes one lifetime-scoped Message Bus, Model Router, Memory Manager, Dream, Schedule Service, Management service, and current `AgentLoop`, then starts the full-screen Textual Terminal Conversation; non-TTY input is rejected without a plain REPL fallback. | `tests/test_cli.py`, `tests/test_cli_replacement_contract.py`, `tests/terminal/test_conversation.py` |
| Foreground execution | The CLI Runtime Lifetime owns one FIFO `MessageBus`; the current `AgentLoop` consumes it while owning the active Session, fixed Tool Gateway, and serial foreground consumer. | `tests/agent/test_message_bus.py`, `tests/agent/test_loop.py`, `tests/test_cli_replacement_contract.py` |
| Bounded ReAct execution | `AgentRunner.run()` is the sole bounded model/Tool ReAct boundary. One Session-independent Runner instance is shared by foreground `chat` and User Schedule `schedule` work in the current Agent Loop; Dream owns a separate instance and restricted Gateway for its `memory` lane while reusing the same loop implementation. | `tests/agent/test_runner.py`, `tests/scheduling/test_schedule_agent_loop.py`, `tests/memory/test_dream.py` |
| Task interpretation | Every nonempty ordinary foreground input is evaluated by isolated Task Framing before context assembly. One staged Blackboard may be committed to Session metadata and projected only into the current model-visible user input. | `tests/agent/test_blackboard.py`, `tests/agent/test_context.py`, `tests/agent/test_loop.py` |
| Skills | Each AgentLoop owns a Skill Loader; each foreground Agent Run captures the current immutable state while constructing its initial messages. Startup and `/resume` perform an initial load; successful `/reload_skill` atomically publishes new state and synchronizes later foreground metadata, manual/always projection, and Terminal completion in the same Runtime Generation, while failure preserves all prior views. Autonomous `read_file` calls retain live filesystem semantics, and raw-input Session persistence does not change the fixed Tool Catalog. | `tests/skills/test_catalog.py`, `tests/agent/test_fixed_catalog.py`, `tests/agent/test_context.py`, `tests/agent/test_loop.py`, `tests/management/test_management_commands.py`, `tests/terminal/test_conversation.py`, `tests/tools/core/test_file_tools.py` |
| Session persistence | The active in-memory `Session` commits one validated Agent Run increment, then schedules an ordered complete JSONL snapshot with three attempts. Normal close performs a bounded final save; forced replacement uses `abandon()` without a final save. | `tests/sessions/test_session.py`, `tests/sessions/test_session_io_security.py`, `tests/agent/test_loop.py`, `tests/test_cli.py` |
| Schedule and Memory | Schedule Service is the only Schedule Store owner. Schedule Agent Runs use isolated Sessions and no foreground output or confirmation channel. Dream consumes Conversation Summary independently of foreground Blackboard state through its dedicated Runner instance and `memory` lane. | `tests/scheduling/test_schedule_service.py`, `tests/agent/test_schedule_loop.py`, `tests/memory/test_dream.py`, `tests/memory/test_conversation_summary.py` |
| Tools and safety | The fixed ten-Tool Catalog executes only through `ToolGateway.call()` and the final BaseTool preparation pipeline. Exact-call confirmation protects external paths and bounded Exec/Web risks without claiming an OS sandbox. | `tests/tools/test_fixed_tool_gateway.py`, `tests/tools/core/test_file_tools.py`, `tests/tools/core/test_exec.py`, `tests/tools/core/test_web_fetch.py` |
| Providers | Anthropic and OpenAI-compatible adapters normalize streaming, Tool calls, usage, retryable failures, fallback, and cancellation behind `ModelRouter`. | `tests/test_anthropic_provider.py`, `tests/test_openai_compatible_provider.py`, `tests/test_model_router.py` |
| Packaging | The installed command is `myclaw.terminal.process_entry:run`; distribution metadata produces one host-neutral wheel and declares all runtime dependencies directly. | `tests/test_release_contract.py`, `pyproject.toml` |

## Verification Gates

| Gate | Command | Current result |
| --- | --- | --- |
| Full behavior suite | `python -m pytest -q` | Passed: 1,412 passed and 10 conditionally skipped on Windows; 1,422 nodes total |
| Lint | `python -m ruff check .` | Passed: 0 violations |
| Format | `python -m ruff format --check .` | Baseline check remains informational; no unrelated bulk formatting is made |
| Types | `python -m mypy` | Passed: 164 source files checked |
| Distribution build | `python -m build --no-isolation` | Passed: one sdist and one `myclaw-0.1.0-py3-none-any.whl` built |
| Tracked Markdown local links and active-design stale checks | `python -m pytest -q tests/test_release_contract.py` | Passed: 25 tests; 42 tracked Markdown files, no unresolved local inline/reference targets, and no stale structural findings |

Current full-suite evidence: 1,412 passed and 10 conditionally skipped on Windows; 1,422 nodes total. Release contract tests: 25 passed; 42 tracked Markdown files. Mapped owner-node execution: 14 passed after 14 nodes were collected. `git diff --check` returned `0` for the current patch; the earlier staged prospective patch check also returned `0`.

## Spec Remediation Verification

Commit `65b24d13748327ff46126563928897a67d2c4ee8` contains the production and owner-test
cutover for issues #194, #199, and #201. The only follow-up code change is confined to
`tests/terminal/test_conversation.py`: the always-raising unavailable helper returns
`Never`, and one redundant dispatcher cast is removed. No file under `myclaw/` changed.

The final gates ran serially on 2026-08-28 without a concurrent Python verification
process. The full suite accounted for exactly 1,417 nodes: 1,407 passed, 10 were skipped
under existing Windows link/terminal platform conditions, and none failed. Ruff reported
zero violations, Mypy reported zero errors across 164 source files, and the build produced
exactly one sdist and one wheel. The 25 release contract tests passed, including clean
distribution checks and structural absence checks for the retired Memory, Terminal/REPL,
Schedule, and Management compatibility surfaces.

## Issue #202 Closure Evidence

The closure is evaluated only against a clean `d60b96d1beed98b4325d2913b674be32d669adb3`
checkout plus the staged prospective documentation patch. ADR-0017 and the CLI composition
implementation plan are tracked authoritative documents in that tree. The active mapping
contains no separate Skill module implementation plan.

The prospective patch adopts these deferred architecture documents: `CONTEXT.md`,
`docs/myclaw-personal-agent-prd.md`, `docs/myclaw-runtime-contracts.md`, the superseded
notes in `docs/adr/0014-use-message-bus-agent-loop-and-agent-runner.md` and
`docs/adr/0016-use-agent-home-skill-catalog-and-progressive-loading.md`, the full
`docs/adr/0017-use-cli-composition-root-and-session-scoped-agent-loop.md`, and the full
`docs/cli-composition-root-implementation-plan.md`. The historical bodies of ADR-0014
and ADR-0016 remain historical context; only their explicit superseded notes change the
active ownership mapping.

The final linearization refinement formed during later implementation review is recorded as
an implementation fact, not as a claim about the original parent issue wording. In this
contract, target preparation is a precondition. The successful replacement sequence is
`quiesce_for_rebind -> pause_and_drain -> current unavailable -> old abort/drain -> bus.reset() -> rebind_agent_loop -> target.start() -> publish current -> schedule_service.resume()`.
Target preparation failure is fatal; successful target activation precedes publication of the
current reference. The real CLI/Terminal fatal boundary, exit code `1`, and exactly one safe
error are covered by `tests/test_cli.py::test_cli_resume_constructor_failure_terminates_safely`
and `tests/test_cli.py::test_cli_resume_preflight_failure_terminates_safely`. After Terminal
exit, the actual shutdown chain is
`Management deactivate -> Schedule pause_and_drain + close -> Loop close/abort -> Dream close -> Model Router close`.

The release meta-test uses one fixed owner-node tuple of 14 nodes. It starts exactly two
subprocesses: one `--collect-only` command and one targeted execution command covering the
complete tuple. Each uses `[sys.executable, "-m", "pytest", ...]`, repository-root cwd,
`-p no:cacheprovider`, explicitly loaded `pytest_asyncio.plugin`, disabled third-party plugin
autoload, the default non-shell process mode, and a sanitized environment. The
collect result requires return code `0`, an explicit count of 14 and every expected real
pytest node id. The execution result requires return code `0` and 14 passed JUnit test cases;
the mapping excludes the release meta-test itself. Its temporary report uses a pytest tmp
fixture, and mapped owner tests use temporary fixtures without mutating shared repository
state. Failure diagnostics retain subprocess stdout/stderr but this document does not record
machine-specific paths.

### Architecture Consistency

| Machine-checkable claim | Passing test node |
| --- | --- |
| The CLI owns the Runtime-Lifetime components and performs async shutdown. | `tests/test_cli.py::test_cli_async_root_owns_lifetime_components_and_async_shutdown` |
| Agent Loop construction owns one set of Runtime-Generation collaborators without constructor side effects. | `tests/agent/test_loop.py::test_agent_loop_constructs_each_generation_collaborator_once_without_side_effects` |
| Message Bus reset clears both FIFOs and publishes exactly one empty snapshot. | `tests/agent/test_message_bus.py::test_reset_clears_both_fifos_and_publishes_one_empty_snapshot` |
| Replacement publishes the target generation only after activation, preserving stable lifetime ownership. | `tests/test_cli.py::test_cli_resume_publishes_current_only_after_target_activation` |
| The deleted legacy runtime module is absent from the import surface. | `tests/test_cli.py::test_legacy_runtime_module_is_not_discoverable` |

The release contract also checks architecture facts directly from production source AST:
the complete `LoadedSkill` field set and loader publication method, the nine async and two
synchronous Message Bus operations, the Memory Manager surface, the Dream constructor and
abort/close methods, the Schedule Clock and Service signatures, the replacement event order,
and the CLI shutdown call order. These source facts are separate from the behavior-node
execution evidence above.

### Persistence and Dream Records

The six compatibility persistence surfaces keep their current exact schemas. Each row
names the behavior-level contract that must pass; the table is not a claim that a path
exists without executing its node.

| Surface | Exact contract evidence |
| --- | --- |
| Session | `tests/sessions/test_session.py::test_persist_writes_one_complete_compact_utf8_snapshot_atomically` |
| Summary | `tests/memory/test_records.py::test_summary_entry_serializes_with_exactly_three_keys` |
| Cursor | `tests/memory/test_memory_manager.py::test_manager_appends_and_claims_summaries_with_cursor_preadvance` |
| Long-term Memory | `tests/memory/test_memory_manager.py::test_manager_reads_disk_and_refreshes_snapshot_after_an_edit` |
| Schedule | `tests/scheduling/test_schedule_model.py::test_schedule_job_round_trips_the_strict_persisted_shape` |
| Artifact | `tests/tools/test_base_tool.py::test_base_tool_result_handler_writes_a_bounded_workspace_artifact` |
| Dream System Job | `tests/scheduling/test_schedule_dream.py::test_dream_registration_persists_a_hidden_recurring_system_job`; `tests/scheduling/test_schedule_dream.py::test_exact_dream_registration_performs_zero_store_writes`; `tests/scheduling/test_schedule_dream.py::test_due_dream_job_dispatches_directly_without_user_or_session_execution` |

The Dream System Job is the only intentionally new persisted record type. Dream
registration creates no foreground Session or Schedule Session.

The active stale scan visits every Python file under `myclaw` and `tests` with structural AST
checks for imports, classes, names, calls, and attributes; it does not exclude a whole test
file. A separate text check is limited to this release-readiness document. Superseded ADR
history, plan removal vocabulary, and negative absence assertions are not treated as
production residue. Valid names such as `RuntimeStatus` remain outside the forbidden
structural boundary.

## Skill Closeout Evidence

- Focused regression passed across `tests/skills/test_catalog.py`, `tests/test_templates.py`, `tests/agent/test_loop.py`, `tests/test_cli.py`, `tests/test_cli_replacement_contract.py`, `tests/terminal/test_conversation.py`, `tests/management/test_management_commands.py`, `tests/tools/core/test_file_tools.py`, `tests/tools/test_fixed_tool_gateway.py`, and `tests/test_release_contract.py`.
- The fixed Tool Catalog contract remains covered by `tests/tools/test_fixed_tool_gateway.py::test_fixed_catalog_order_and_detached_definitions`, which asserts exactly ten Tool schemas.
- The active documentation contract test structurally verifies ADR frontmatter, the CONTEXT glossary definitions, ordered Management Commands, same-generation `/reload_skill` lifecycle, scoped PRD routing, complete-document host projection, strict raw Skill names, real startup-budget inputs, restored Tab behavior, and explicit obsolete markers across the authoritative active documents.
- The tracked-link test derives its source set from `git ls-files -- '*.md'` and audits the repository's simple inline/reference local-link forms, not full CommonMark: external schemes and pure fragments are excluded, while local targets are URL-decoded, stripped of fragments, and resolved relative to their source document.

## Issue #195 Terminal Commit Cancellation Evidence

Verification executed on Windows x64 on 2026-08-28:

- `python -m pytest tests/scheduling -q`: `186 passed`.
- Runtime Generation, CLI replacement, Dream and Schedule focused command: `254 passed, 1 skipped`;
  the skip is the Windows Python runtime's missing `termios/pty` harness.
- `python -m pytest -q`: `1412 passed, 10 skipped`; all skips are explained Windows
  symlink-privilege or `termios/pty` capability gates, with `0` failures.
- `python -m ruff check .`: return code `0`.
- `python -m mypy myclaw tests`: return code `0`, `164` source files checked.
- `git diff --check`: return code `0`.
- `python -m build`: return code `0`; both sdist and wheel built successfully.
- Schedule persisted field-set changes: `0`; Store schema and serialization were not modified.

## Accepted Risks and Unverified Environments

- There is no platform gate. Windows x64 is currently validated; macOS Intel and Apple
  Silicon are intended compatibility targets but remain unverified. Other POSIX hosts may
  attempt the POSIX filesystem adapter without a formal support claim. Packaging is
  `py3-none-any`.
- Skill instructions are user-authored prompt material. The tested boundaries intentionally
  expose canonical Skill paths to the foreground provider, persist autonomous `read_file`
  results, keep manual bodies transient, and require explicit opt-in for always-loaded bodies.
- Skill symlink/reparse-point cases requiring link creation are conditionally skipped on this
  Windows environment when the process lacks the required privilege; ordinary in-root and
  escaping-path logic remains covered where the platform permits it.
- Existing Textual timing-sensitive tests failed intermittently during focused and full-suite
  verification, then passed in isolated reruns and the final full suite; no unrelated production
  code was changed to mask that environment-sensitive behavior.
- Same-Session concurrency is unsupported.
- Session Log uses an unbounded queue and infinite drain.
- Session Log provides no per-record fsync, no active redaction, and no control escaping.
- Session Log uses per-Session retention without a Workspace-wide size bound.
- Legacy Agent Home Runtime Log files remain untouched.
- Workspace-owned state has no cross-process coordination. Separate Terminal Conversation
  processes can race on Session snapshots, Summary/Cursor state, Schedule state, Artifacts,
  and Session Logs.
- Exec is not an OS sandbox. It launches one Bash process with best-effort direct-process
  cleanup; approved commands may still affect files or networks outside the Workspace.
- Forced Runtime Generation replacement abandons the old Session without repair or a final
  save. Unpersisted conversation state can be lost, and accepted Tool, Artifact, Memory, or
  Schedule side effects are not rolled back.
- Conversation Summary and `last_consolidated` are not committed in one filesystem
  transaction. A crash can cause summary work to repeat or be omitted.
- Real paid-provider and live-network acceptance are not implied by the fake boundary
  suites. They require explicitly supplied release credentials and endpoints.

## Authoritative Documents

- [Domain language](../CONTEXT.md)
- [Product requirements](myclaw-personal-agent-prd.md)
- [Runtime contracts](myclaw-runtime-contracts.md)
- [ADR-0017: CLI composition root and Session-scoped Agent Loop](adr/0017-use-cli-composition-root-and-session-scoped-agent-loop.md)
- [ADR-0018: Centralized Model Request Context construction](adr/0018-centralize-model-request-context-construction.md)
- [CLI composition root implementation plan](cli-composition-root-implementation-plan.md)
- [Terminal Conversation design](terminal-conversation-ui-design.md)
- [Security and fault review](security-fault-review.md)
- [Current architectural decisions](adr/)
