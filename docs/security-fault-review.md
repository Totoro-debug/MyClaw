# Security and Fault Review

This document records the issue #35 review of the current implementation and its
public-boundary tests. It is an evidence map, not a claim that release testing is
complete. The review follows the repository's RED -> GREEN TDD practice: injected
or reproduced boundary failures were first observed through public interfaces,
then the same tests were made green with the smallest scoped correction.

The governing contracts are [TOOL_SCHEMA](myclaw-runtime-contracts.md#tool_schema-tool-gateway),
[Tool Artifacts](myclaw-runtime-contracts.md#117-tool-artifact), the
[fail-closed capability matrix](myclaw-runtime-contracts.md#12-fail-closed-capability-矩阵), and the
[Error Contract](myclaw-runtime-contracts.md#13-error-contract). Persistence and
sandbox boundaries are further defined by
[ADR-0001](adr/0001-file-first-local-persistence.md),
[ADR-0002](adr/0002-fixed-agent-home.md), and
[ADR-0010](adr/0010-fixed-tool-catalog-and-base-tool-boundaries.md), with active Session
snapshot behavior defined by [ADR-0009](adr/0009-active-session-snapshot-persistence.md).
Issue #130 removed the superseded File/Security/Shell/Web worker, owned-process, and
separate Artifact modules. The current-behavior summaries below reflect the fixed
Core Catalog while the recorded #35 gate counts remain historical evidence.

## Acceptance Coverage

This first matrix maps each acceptance criterion to the domains and public
evidence that satisfy it.

| ID | Acceptance criterion | Covered domains | Public evidence |
| --- | --- | --- | --- |
| AC1 | Cover path traversal, Workspace State access, Exec safety, SSRF redirects, and secret redaction. | File Tools / Workspace; Exec; Web / SSRF / secrets | Fixed file Tools resolve Workspace paths and request confirmation for external paths. Exec checks cwd, destructive patterns, and URL targets. WebFetch validates DNS targets and each redirect. Configuration and adapter failures expose safe messages. See [Core file tools](../tests/tools/core/test_file_tools.py), [Exec](../tests/tools/core/test_exec.py), [WebFetch](../tests/tools/core/test_web_fetch.py), and [configuration](../tests/configuration/test_config.py). |
| AC2 | Cover disk failure, corrupt TOML/JSON/JSONL, consecutive provider failures, and cancellation during stream/tool/metadata phases. | Web / secrets; Conversation / provider / Session / cancellation / publication; Runtime Core artifact externalization | Fault-injected complete Session snapshots and bounded close attempts produce safe persistence outcomes; corrupt configuration, Schedule state, and Session files fail closed; consecutive retryable Provider failures stop at the five-attempt logical-call budget; provider stream failures and unsupported events become model failures. Cancellation is exercised before provider work and during stream, Tool execution, publication, model-error publication, and metadata work. Runtime Core may leave an accepted orphan artifact if later Session persistence fails; no rollback path can delete a file after publication. See [Model Router](../tests/test_model_router.py), [Session snapshot tests](../tests/sessions/test_session.py), [Agent Run](../tests/agent/test_run_awaitable.py), [Schedule Store](../tests/scheduling/test_schedule_store.py), and [runtime shutdown](../tests/test_runtime_shutdown.py). |
| AC3 | Ensure user-visible errors contain no API key, sensitive raw Tool data, or unexpected traceback. | Exec / Web / configuration; Conversation / provider / Session / cancellation / publication; Tool Results and artifacts | Exec and Web failures use safe Tool messages; malformed configuration is redacted; Tool failures are flat results; Provider or Schedule Service failures crossing a public boundary use safe summaries. Ordinary Session snapshot failures remain silent. See [Exec](../tests/tools/core/test_exec.py), [WebFetch](../tests/tools/core/test_web_fetch.py), [configuration](../tests/configuration/test_config.py), [Agent Run](../tests/agent/test_run_awaitable.py), and [Schedule Service](../tests/scheduling/test_schedule_service.py). |

## Domain Coverage

This reverse matrix starts from each reviewed domain and maps it back to the
acceptance criteria, public behavior, implementation, and primary tests.

| Domain | Acceptance IDs | Key public behavior | Implementation | Primary tests |
| --- | --- | --- | --- | --- |
| Filesystem / Workspace / artifacts | AC1, AC2, AC3 | Fixed file Tools resolve and canonicalize Workspace paths, allow Workspace State, request confirmation outside Workspace, and write direct UTF-8 artifacts at the BaseTool-owned path. | [base.py](../myclaw/tools/base.py), [read_file.py](../myclaw/tools/core/read_file.py), [write_file.py](../myclaw/tools/core/write_file.py), [host_filesystem.py](../myclaw/utils/host_filesystem.py), [run.py](../myclaw/agent/run.py) | [file Tools](../tests/tools/core/test_file_tools.py), [artifact seam](../tests/agent/test_run_awaitable.py) |
| Exec safety / process | AC1, AC3 | One Bash process accepts arbitrary commands, checks cwd and bounded destructive/DNS patterns, requests exact-call confirmation when needed, and does not claim process-tree ownership or OS sandboxing. | [exec.py](../myclaw/tools/core/exec.py), [tool_gateway.py](../myclaw/tools/tool_gateway.py) | [Exec](../tests/tools/core/test_exec.py), [runtime shutdown](../tests/test_runtime_shutdown.py) |
| Web / SSRF / secrets | AC1, AC2, AC3 | WebFetch rejects invalid schemes and userinfo, and requests exact-call confirmation for DNS failures or non-public targets. Each redirect repeats target validation; redirect count, total timeout, media type, and output processing are bounded. Web adapter failures hide raw URLs/queries and secrets. Malformed configuration paths redact API keys or return safe errors without traceback. | [web_fetch.py](../myclaw/tools/core/web_fetch.py), [config.py](../myclaw/config/config.py), [cli.py](../myclaw/terminal/cli.py), [tool_gateway.py](../myclaw/tools/tool_gateway.py) | [WebFetch](../tests/tools/core/test_web_fetch.py), [configuration](../tests/configuration/test_config.py) |
| Conversation / provider / Session / cancellation / publication | AC2, AC3 | Initial user, assistant, Tool, and model-error publication faults end in one safe terminal event. Corrupt Session JSONL fails before the provider call. A logical model stream stops after five consecutive retryable Provider failures; unexpected Provider exceptions, unsupported events, and empty completions become normalized model errors. Main and title iterators are closed deterministically. Active Session messages stay in memory through the turn, then a complete snapshot is scheduled; ordinary snapshot failures are silent, and shutdown uses bounded retries. | [conversation.py](../myclaw/session/conversation.py), [session.py](../myclaw/session/session.py), [model_router.py](../myclaw/provider/model_router.py), [tool_gateway.py](../myclaw/tools/tool_gateway.py) | [test_model_router.py](../tests/test_model_router.py), [test_session.py](../tests/sessions/test_session.py), [test_runtime_active_session.py](../tests/test_runtime_active_session.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |
| Tool publication and artifacts | AC2, AC3 | Runtime Core externalizes only oversized successful results through BaseTool, writes direct UTF-8 artifacts, retains success on write failure with a bounded marker, and accepts an orphan after later Session persistence failure. Cancellation and custom `BaseException` identity survive publication handling, and one Schedule Job failure does not stop the Schedule Service. | [base.py](../myclaw/tools/base.py), [run.py](../myclaw/agent/run.py), [service.py](../myclaw/schedule/service.py), [session.py](../myclaw/session/session.py) | [test_run_awaitable.py](../tests/agent/test_run_awaitable.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |

## RED to GREEN Fault Review

| Fault category | RED observation or injected fault | Current GREEN behavior | Evidence |
| --- | --- | --- | --- |
| Filesystem boundary confusion | Traversal, links, external aliases, device paths, and artifact IDs could cross intended scope. | BaseTool path resolution uses host semantics, external targets use exact confirmation, Workspace State has no extra MyClaw restriction, and invalid artifact IDs fall back to UUID4. | [test_file_tools.py](../tests/tools/core/test_file_tools.py), [test_run_awaitable.py](../tests/agent/test_run_awaitable.py) |
| Exec safety and lifecycle | Destructive commands, unsafe URLs, or cancellation could cross the visible safety boundary. | Concrete Exec checks request exact-call confirmation; timeout/cancellation clean up the direct Bash best effort, with no process-tree ownership or OS sandbox claim. | [test_exec.py](../tests/tools/core/test_exec.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |
| SSRF and secret parsing | Redirects, unusual address forms, malformed TOML key spellings, or invalid UTF-8 could expose secret text or a traceback. | Every visible WebFetch target is checked before access, DNS failures and non-public targets require exact-call confirmation, and redirects repeat checks; structured and fallback configuration redaction remains safe. | [test_web_fetch.py](../tests/tools/core/test_web_fetch.py), [test_config.py](../tests/configuration/test_config.py) |
| Disk and publication failure | Session mutations occur in memory, then an ordinary atomic snapshot or the bounded close attempts can fail independently of the completed Agent Run. | Agent Run outcomes and Agent Events do not wait for ordinary Session persistence; snapshot failures are silent. `close()` gets three bounded attempts and swallows final failure. Artifact writes are not rolled back after later Session failure. | [test_session.py](../tests/sessions/test_session.py), [test_runtime_active_session.py](../tests/test_runtime_active_session.py), [test_run_awaitable.py](../tests/agent/test_run_awaitable.py) |
| Corrupt persisted data | Malformed or schema-invalid TOML, Schedule state JSON, and Session JSONL are supplied at their public load/use boundaries. | Corrupt Schedule state blocks Runtime startup before schedulers or REPL input; unusable Session history fails before Provider work; one independent Schedule Job failure does not terminate the dispatcher. No raw secret or traceback crosses the user-visible boundary. | [configuration](../tests/configuration/test_config.py), [test_session.py](../tests/sessions/test_session.py), [test_schedule_store.py](../tests/scheduling/test_schedule_store.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py) |
| Provider failure and iterator ownership | Consecutive Provider failures, unexpected exceptions, unsupported stream events, empty completion, and abandoned streams could leak implementation detail or retain iterators until garbage collection. | A logical stream is capped at five Provider attempts; attempt exhaustion and malformed streams become safe model failures. Iterators close deterministically on completion, explicit close, cancellation, and error; partial streamed content is added to the active Session as interrupted where required. | [test_model_router.py](../tests/test_model_router.py), [test_runtime.py](../tests/test_runtime.py), [test_run_awaitable.py](../tests/agent/test_run_awaitable.py) |
| Cancellation by phase | Cancellation is injected before Provider start and during stream or Tool execution, including states with completed and unfinished Tool calls in the active Session. | The original cancellation is preserved, exactly one terminal outcome is emitted, completed messages and Tool results remain in memory, unfinished calls receive correlated error results, and one best-effort snapshot is requested after terminal work. Runtime shutdown cancellation keeps pending Schedule definitions while the Schedule Service settles owned tasks before shared dependencies close. | [test_run_awaitable.py](../tests/agent/test_run_awaitable.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |
| User-visible error disclosure | API keys, raw Web arguments, sensitive Tool output, private exception messages, and fake tracebacks are injected into boundary failures. | Tool Results use flat safe messages; Agent Events, CLI output, Agent Run failures, Schedule Service outcomes, and persisted errors expose safe summaries only. Sensitive oversized success data remains in its controlled artifact and is not copied into an error message. | [test_exec.py](../tests/tools/core/test_exec.py), [test_web_fetch.py](../tests/tools/core/test_web_fetch.py), [test_run_awaitable.py](../tests/agent/test_run_awaitable.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py) |

## Verified Gates

The following focused gates were run during the #35 work and recorded in
`Procedure.md`:

- Filesystem security: 24 passed.
- Historical #35 Scheduled Work execution gate: 23 passed.
- Historical #35 Scheduled Work/background/Session/artifact/security/shutdown gate: 114 passed.
- Shell-focused regression: 82 passed.
- Web/config/CLI/Management regression: 153 passed, including the warning-strict run.
- Conversation/Tool/artifact/title regression: 79 passed, including the warning-strict run.
- Strict Mypy covered all 111 source files; scoped Ruff lint and format checks passed.
- `git diff --check` passed for the implementation batches.

These are focused security and regression gates. A final repository-wide test run
and release smoke are not claimed here.

## Residual Risks

- Same-user filesystem TOCTOU remains possible between final path validation and
  operating-system I/O. Eliminating that race requires handle-relative or
  OS-specific confinement rather than another string/path check.
- Exec is not an OS sandbox. Workspace `cwd` and concrete command/DNS checks do
  not provide general child-process confinement, as documented by ADR-0010.
- Cross-process coordination is out of scope. Separate
  MyClaw runtimes can race on file-first persistence, Session metadata, summary
  allocation, or Schedule Job triggers; only in-runtime serialization is promised
  for those state domains.
- Real provider and live-network smoke evidence is recorded separately from the
  Windows x64 release candidate. This review uses injected Provider/Web boundaries
  plus the Windows-focused tests available in the current environment.
- Artifact externalization intentionally has no rollback. A later Session
  persistence failure can retain an unreferenced artifact; orphan cleanup remains
  out of scope.
