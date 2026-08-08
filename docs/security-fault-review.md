# Security and Fault Review

This document records the issue #35 review of the current implementation and its
public-boundary tests. It is an evidence map, not a claim that release testing is
complete. The review follows the repository's RED -> GREEN TDD practice: injected
or reproduced boundary failures were first observed through public interfaces,
then the same tests were made green with the smallest scoped correction.

The governing contracts are [file tools](myclaw-runtime-contracts.md#113-内置-file-tools),
[Tool Artifacts](myclaw-runtime-contracts.md#117-tool-artifact), the
[fail-closed capability matrix](myclaw-runtime-contracts.md#12-fail-closed-capability-矩阵), and the
[Error Contract](myclaw-runtime-contracts.md#13-error-contract). Persistence and
sandbox boundaries are further defined by
[ADR-0001](adr/0001-file-first-local-persistence.md),
[ADR-0002](adr/0002-fixed-agent-home.md), and
[ADR-0003](adr/0003-shell-permission-is-not-os-sandbox.md), with active Session
snapshot behavior defined by [ADR-0009](adr/0009-active-session-snapshot-persistence.md).
Issue #38 later removed broad file/Shell confirmation and artifact ownership
machinery. Issue #104 then introduced the shared Agent Run, Schedule Service, and
invocation-scoped Tool Confirmation for Schedule add/remove. The current-behavior
summaries below reflect those later contracts while the recorded #35 gate counts
remain historical evidence.

## Acceptance Coverage

This first matrix maps each acceptance criterion to the domains and public
evidence that satisfy it.

| ID | Acceptance criterion | Covered domains | Public evidence |
| --- | --- | --- | --- |
| AC1 | Cover path traversal, Agent Home write protection, Shell policy, SSRF redirects, and secret redaction. | Filesystem / Agent Home / artifacts; Shell policy / process; Web / SSRF / secrets | File Tools and injected Security helpers reject traversal, aliases, hard links, device/ADS paths, and protected Agent Home state. Shell executes only the five exact safe forms and refuses every other command. WebFetch validates DNS and the connected peer on every hop. Configuration and Web adapter failures redact API-key aliases, raw URLs/queries, and tracebacks. See [filesystem security](../tests/test_security_filesystem.py), [Shell policy](../tests/tools/shell/test_shell_policy.py), [Shell security](../tests/test_security_shell.py), [WebFetch](../tests/tools/web/test_web_fetch.py), and [Web/secret security](../tests/test_security_web_secrets.py). |
| AC2 | Cover disk failure, corrupt TOML/JSON/JSONL, consecutive provider failures, and cancellation during stream/tool/metadata phases. | Web / secrets; Conversation / provider / Session / cancellation / publication; Runtime Core artifact externalization | Fault-injected complete Session snapshots and bounded close attempts produce safe persistence outcomes; corrupt configuration, Schedule state, and Session files fail closed; consecutive retryable Provider failures stop at the five-attempt logical-call budget; provider stream failures and unsupported events become model failures. Cancellation is exercised before provider work and during stream, Tool execution, publication, model-error publication, and metadata work. Runtime Core may leave an accepted orphan artifact if later Session persistence fails; no rollback path can delete a file after publication. See [Model Router](../tests/test_model_router.py), [Session snapshot tests](../tests/sessions/test_session.py), [Agent Run](../tests/agent/test_run.py), [Schedule Store](../tests/scheduling/test_schedule_store.py), and [runtime shutdown](../tests/test_runtime_shutdown.py). |
| AC3 | Ensure user-visible errors contain no API key, sensitive raw Tool data, or unexpected traceback. | Shell policy / process; Web / SSRF / secrets; Conversation / provider / Session / cancellation / publication; Tool Results and artifacts | Nonzero Shell exits discard unsafe process output; WebSearch/WebFetch adapter failures hide raw arguments; malformed configuration redaction fails closed; Tool failures produce flat message-only results, and Provider or Schedule Service failures that cross a public boundary use safe summaries. Ordinary Session snapshot failures remain silent and emit no boundary result. Persisted messages use the same safe details. See [Shell security](../tests/test_security_shell.py), [Web/secret security](../tests/test_security_web_secrets.py), [Agent Run](../tests/agent/test_run.py), and [Schedule Service](../tests/scheduling/test_schedule_service.py). |

## Domain Coverage

This reverse matrix starts from each reviewed domain and maps it back to the
acceptance criteria, public behavior, implementation, and primary tests.

| Domain | Acceptance IDs | Key public behavior | Implementation | Primary tests |
| --- | --- | --- | --- | --- |
| Filesystem / Agent Home / artifacts | AC1, AC2, AC3 | Workspace reads enforce canonical containment and reject traversal, external aliases, hard links, Windows devices, and alternate data streams; foreground writes and edits are refused. Agent Home classification takes priority even when nested inside the Workspace; only Long-term Memory and current-session artifacts receive their documented read exceptions. Runtime Core creates oversized successful-result artifacts atomically and never owns a later commit/discard/rollback lifecycle. | [agent_home.py](../myclaw/config/agent_home.py), [file_tools.py](../myclaw/tools/files/file_tools.py), [host_filesystem.py](../myclaw/utils/host_filesystem.py), [tool_artifacts.py](../myclaw/tools/tool_artifacts.py), [tool_gateway.py](../myclaw/tools/tool_gateway.py) | [test_security_filesystem.py](../tests/test_security_filesystem.py), [test_workspace_inspection_tools.py](../tests/tools/files/test_workspace_inspection_tools.py), [test_tool_artifacts.py](../tests/tools/test_tool_artifacts.py) |
| Shell policy / process | AC1, AC3 | Only `pwd`, `git status`, `git status --short`, `git diff --stat`, and `git diff --name-only` execute. Every other command is refused before process creation; invalid cwd, timeout, NUL, or control characters fail closed. Safe Git uses a trusted startup-captured executable, disables hooks/external processors, and refuses filter/attribute ambiguity. Timeout, cancellation, I/O failure, and runtime close retain process-tree ownership until cleanup settles; transient cleanup can be retried. Nonzero process output is not returned as a successful or raw user-visible result. | [shell_tool.py](../myclaw/tools/shell/shell_tool.py), [owned_process.py](../myclaw/tools/shell/owned_process.py) | [test_shell_policy.py](../tests/tools/shell/test_shell_policy.py), [test_shell_process.py](../tests/tools/shell/test_shell_process.py), [test_security_shell.py](../tests/test_security_shell.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |
| Web / SSRF / secrets | AC1, AC2, AC3 | WebFetch rejects invalid schemes, userinfo, localhost, port zero, and every non-public address category. Every DNS answer must be public, the connected peer must belong to the validated set, and each redirect repeats validation before the next request. Redirect count, total timeout, media type, and decompressed body size are bounded. Web adapter failures hide raw URLs/queries and secrets. Valid, schema-invalid, malformed, escaped-key, and invalid-UTF-8 configuration paths either redact API keys or return a safe persistence/configuration error without traceback. | [web_fetch.py](../myclaw/tools/web/web_fetch.py), [config.py](../myclaw/config/config.py), [cli.py](../myclaw/terminal/cli.py), [tool_gateway.py](../myclaw/tools/tool_gateway.py) | [test_web_fetch.py](../tests/tools/web/test_web_fetch.py), [test_security_web_secrets.py](../tests/test_security_web_secrets.py) |
| Conversation / provider / Session / cancellation / publication | AC2, AC3 | Initial user, assistant, Tool, and model-error publication faults end in one safe terminal event. Corrupt Session JSONL fails before the provider call. A logical model stream stops after five consecutive retryable Provider failures; unexpected Provider exceptions, unsupported events, and empty completions become normalized model errors. Main and title iterators are closed deterministically. Active Session messages stay in memory through the turn, then a complete snapshot is scheduled; ordinary snapshot failures are silent, and shutdown uses bounded retries. | [conversation.py](../myclaw/session/conversation.py), [session.py](../myclaw/session/session.py), [model_router.py](../myclaw/provider/model_router.py), [tool_gateway.py](../myclaw/tools/tool_gateway.py) | [test_model_router.py](../tests/test_model_router.py), [test_session.py](../tests/sessions/test_session.py), [test_runtime_active_session.py](../tests/test_runtime_active_session.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |
| Tool publication and artifacts | AC2, AC3 | Runtime Core externalizes only oversized successful results and persists the derived immutable Tool Result. A later Session snapshot failure does not trigger artifact deletion; the possible orphan is accepted. Cancellation and custom `BaseException` identity survive publication handling, and one Schedule Job failure does not stop the Schedule Service. | [run.py](../myclaw/agent/run.py), [service.py](../myclaw/schedule/service.py), [session.py](../myclaw/session/session.py), [tool_artifacts.py](../myclaw/tools/tool_artifacts.py) | [test_run.py](../tests/agent/test_run.py), [test_tool_artifacts.py](../tests/tools/test_tool_artifacts.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |

## RED to GREEN Fault Review

| Fault category | RED observation or injected fault | Current GREEN behavior | Evidence |
| --- | --- | --- | --- |
| Filesystem boundary confusion | Hard links, aliases/junctions, protected Agent Home paths nested in the Workspace, reused Tool IDs, device names, and alternate data streams crossed or confused the intended scope. | Classification is canonical and fail-closed, documented exceptions require exact unaliased paths, duplicate artifact IDs are invalid, and artifacts are create-once without rollback ownership. | [test_security_filesystem.py](../tests/test_security_filesystem.py) |
| Shell trust and lifecycle | Safe Git could run a Workspace/PATH shadow or repository hooks/filters; a nonzero process result exposed raw output; cleanup ownership could be lost after transient stop failures or cancellation during spawn. | Safe Git is pinned to a trusted executable and hardened environment; ambiguous repositories and all non-allowlisted commands are refused; nonzero output is normalized; cleanup remains owned and retryable until the process tree is reaped. | [test_security_shell.py](../tests/test_security_shell.py), [test_shell_process.py](../tests/tools/shell/test_shell_process.py) |
| SSRF and secret parsing | Redirects or unusual address forms could evade validation; port zero and deprecated site-local IPv6 reached later boundaries; malformed/schema-invalid TOML key spellings and invalid UTF-8 could expose secret text or a traceback. | DNS sets and actual peers are public and pinned on every hop; all disallowed address/port forms fail before I/O; structured and fallback configuration redaction recognize equivalent API-key spellings and fail closed; CLI output remains safe. | [test_web_fetch.py](../tests/tools/web/test_web_fetch.py), [test_security_web_secrets.py](../tests/test_security_web_secrets.py) |
| Disk and publication failure | Session mutations occur in memory, then an ordinary atomic snapshot or the bounded close attempts can fail independently of the completed Agent Run. | Agent Run outcomes and Agent Events do not wait for ordinary Session persistence; snapshot failures are silent. `close()` gets three bounded attempts and swallows final failure. Artifact writes are not rolled back after later Session failure. | [test_session.py](../tests/sessions/test_session.py), [test_runtime_active_session.py](../tests/test_runtime_active_session.py), [test_run.py](../tests/agent/test_run.py) |
| Corrupt persisted data | Malformed or schema-invalid TOML, Schedule state JSON, and Session JSONL are supplied at their public load/use boundaries. | Corrupt Schedule state blocks Runtime startup before schedulers or REPL input; unusable Session history fails before Provider work; one independent Schedule Job failure does not terminate the dispatcher. No raw secret or traceback crosses the user-visible boundary. | [test_security_web_secrets.py](../tests/test_security_web_secrets.py), [test_session.py](../tests/sessions/test_session.py), [test_schedule_store.py](../tests/scheduling/test_schedule_store.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py) |
| Provider failure and iterator ownership | Consecutive Provider failures, unexpected exceptions, unsupported stream events, empty completion, and abandoned streams could leak implementation detail or retain iterators until garbage collection. | A logical stream is capped at five Provider attempts; attempt exhaustion and malformed streams become safe model failures. Iterators close deterministically on completion, explicit close, cancellation, and error; partial streamed content is added to the active Session as interrupted where required. | [test_model_router.py](../tests/test_model_router.py), [test_runtime.py](../tests/test_runtime.py), [test_run.py](../tests/agent/test_run.py) |
| Cancellation by phase | Cancellation is injected before Provider start and during stream or Tool execution, including states with completed and unfinished Tool calls in the active Session. | The original cancellation is preserved, exactly one terminal outcome is emitted, completed messages and Tool results remain in memory, unfinished calls receive correlated error results, and one best-effort snapshot is requested after terminal work. Runtime shutdown cancellation keeps pending Schedule definitions while the Schedule Service settles owned tasks before shared dependencies close. | [test_run.py](../tests/agent/test_run.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py) |
| User-visible error disclosure | API keys, raw Web arguments, sensitive Tool/process output, private exception messages, and fake tracebacks are injected into boundary failures. | Tool Results use a flat safe message without nested exception data; Agent Events, CLI output, Agent Run failures, Schedule Service outcomes, and persisted errors expose safe summaries only. Sensitive oversized success data remains in its controlled artifact and is not copied into an error message. | [test_security_shell.py](../tests/test_security_shell.py), [test_security_web_secrets.py](../tests/test_security_web_secrets.py), [test_run.py](../tests/agent/test_run.py), [test_schedule_service.py](../tests/scheduling/test_schedule_service.py) |

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
- Shell is not an OS sandbox. The current exact allowlist and Workspace `cwd`
  validation do not provide general child-process confinement, as documented by
  ADR-0003; Issue #38 temporarily refuses the broader commands that ADR expected
  foreground confirmation to authorize.
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
