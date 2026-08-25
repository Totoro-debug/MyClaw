# MyClaw Release Readiness

Status: **current worktree verified on 2026-08-26; no release was uploaded**

This document is the evidence index for the current implementation. It records active
contracts, current test locations, current verification results, and present release risks.

## Current Runtime Contract

| Area | Current boundary | Primary evidence |
| --- | --- | --- |
| Process and Terminal | `myclaw` prepares one `RuntimeHost` and starts the full-screen Textual Terminal Conversation; non-TTY input is rejected without a plain REPL fallback. | `tests/test_cli.py`, `tests/terminal/test_conversation.py`, `tests/test_runtime_generation.py` |
| Foreground execution | One `AgentLoop` owns one FIFO `MessageBus`, active Session, fixed Tool Gateway, and serial foreground consumer. One Session-independent `AgentRunner` performs the bounded model and Tool loop. | `tests/agent/test_message_bus.py`, `tests/agent/test_loop.py`, `tests/agent/test_runner.py` |
| Task interpretation | Every nonempty ordinary foreground input is evaluated by isolated Task Framing before context assembly. One staged Blackboard may be committed to Session metadata and projected only into the current model-visible user input. | `tests/agent/test_blackboard.py`, `tests/agent/test_context.py`, `tests/agent/test_loop.py`, `tests/test_runtime.py` |
| Skills | The Runtime Lifetime Skill Catalog provides foreground metadata discovery, ordinary `read_file` loading, host-side manual/always loading, raw-input Session persistence, and presentation-only shared completion without changing the fixed Tool Catalog. | `tests/skills/test_catalog.py`, `tests/agent/test_fixed_catalog_runtime.py`, `tests/agent/test_context.py`, `tests/agent/test_loop.py`, `tests/terminal/test_conversation.py`, `tests/tools/core/test_file_tools.py` |
| Session persistence | The active in-memory `Session` commits one validated Agent Run increment, then schedules an ordered complete JSONL snapshot with three attempts. Normal close performs a bounded final save; forced replacement uses `abandon()` without a final save. | `tests/sessions/test_session.py`, `tests/sessions/test_session_io_security.py`, `tests/test_runtime_active_session.py`, `tests/test_runtime_shutdown.py` |
| Schedule and Memory | Schedule Service is the only Schedule Store owner. Schedule Agent Runs use isolated Sessions and no foreground output or confirmation channel. Memory Tasks consume Conversation Summary independently of foreground Blackboard state. | `tests/scheduling/test_schedule_service.py`, `tests/agent/test_schedule_loop.py`, `tests/memory/test_memory_task.py`, `tests/memory/test_conversation_summary.py` |
| Tools and safety | The fixed ten-Tool Catalog executes only through `ToolGateway.call()` and the final BaseTool preparation pipeline. Exact-call confirmation protects external paths and bounded Exec/Web risks without claiming an OS sandbox. | `tests/tools/test_fixed_tool_gateway.py`, `tests/tools/core/test_file_tools.py`, `tests/tools/core/test_exec.py`, `tests/tools/core/test_web_fetch.py` |
| Providers | Anthropic and OpenAI-compatible adapters normalize streaming, Tool calls, usage, retryable failures, fallback, and cancellation behind `ModelRouter`. | `tests/test_anthropic_provider.py`, `tests/test_openai_compatible_provider.py`, `tests/test_model_router.py` |
| Packaging | The installed command is `myclaw.terminal.process_entry:run`; distribution metadata produces one host-neutral wheel and declares all runtime dependencies directly. | `tests/test_release_contract.py`, `pyproject.toml` |

## Verification Gates

| Gate | Command | Current result |
| --- | --- | --- |
| Full behavior suite | `python -m pytest -q` | Passed: 1,430 passed and 11 conditionally skipped on Windows |
| Lint | `python -m ruff check .` | Passed |
| Format | `python -m ruff format --check .` | Passed: 172 files already formatted |
| Types | `python -m mypy myclaw tests` | Passed: 172 source files checked |
| Distribution build | `python -m build --no-isolation` | Passed: sdist and `myclaw-0.1.0-py3-none-any.whl` built; wheel contains `myclaw/skills/__init__.py`, `myclaw/skills/catalog.py`, and 17 templates including both Skill templates |
| Tracked Markdown local links and active-design stale checks | `python -m pytest -q tests/test_release_contract.py` | Passed: 15 tests; 39 tracked Markdown files, no unresolved local inline/reference targets, and no stale findings in the explicit active contract set |

## Skill Closeout Evidence

- Focused regression passed across `tests/skills/test_catalog.py`, `tests/agent/test_fixed_catalog_runtime.py`, `tests/agent/test_context.py`, `tests/agent/test_loop.py`, `tests/test_runtime_generation.py`, `tests/terminal/test_conversation.py`, `tests/tools/core/test_file_tools.py`, `tests/tools/test_fixed_tool_gateway.py`, `tests/configuration/test_config.py`, `tests/test_cli.py`, `tests/test_templates.py`, and `tests/test_release_contract.py`: 460 passed and 4 conditionally skipped on Windows.
- The fixed Tool Catalog contract remains covered by `tests/tools/test_fixed_tool_gateway.py::test_fixed_catalog_order_and_detached_definitions`, which asserts exactly ten Tool schemas.
- The active documentation contract test structurally verifies ADR frontmatter, the three CONTEXT glossary definitions, the scoped PRD routing contract, complete-document revalidation versus instruction-body projection, and explicit obsolete markers across the authoritative active documents.
- The tracked-link test derives its source set from `git ls-files -- '*.md'` and explicitly excludes the untracked `docs/skill-module-implementation-plan.md`. It audits the repository's simple inline/reference local-link forms, not full CommonMark: external schemes and pure fragments are excluded, while local targets are URL-decoded, stripped of fragments, and resolved relative to their source document.

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
- [Terminal Conversation design](terminal-conversation-ui-design.md)
- [Security and fault review](security-fault-review.md)
- [Current architectural decisions](adr/)
