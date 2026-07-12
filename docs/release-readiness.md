# MyClaw v0.1 Release Readiness

This document is the release evidence index for GitHub
[#1](https://github.com/Totoro-debug/myclaw/issues/1) and
[#36](https://github.com/Totoro-debug/myclaw/issues/36). The local
[PRD](myclaw-personal-agent-prd.md) remains the product source of truth and the
[runtime contracts](myclaw-runtime-contracts.md) remain the accepted behavior and
schema source of truth.

Evidence is separated deliberately:

- Automated coverage links show that a public behavior has a maintained test.
- Platform results come only from the Windows and POSIX validation reports.
- Manual and external smoke rows use only `PASS`, `NOT RUN`, or `BLOCKED`.
- `NOT RUN` is not treated as a pass. No local credential was read or used merely
  to improve this report.

## Platform Evidence

| Environment | Evidence | Status in this synthesis |
| --- | --- | --- |
| Windows | [Windows validation](release/windows-validation.md) | **PASS:** 651 offline tests; Ruff lint/format; strict Mypy; wheel/sdist build; clean-wheel install; Unicode `myclaw`/`myclaw config` and redaction smoke. |
| POSIX | [POSIX validation](release/posix-validation.md) | **PENDING AUTHORITATIVE RUN:** local WSL registration points to a missing filesystem; the permanent `ubuntu-latest` workflow is the accepted authority. |

The security/fault baseline immediately before B18 is recorded in
[Security and Fault Review](security-fault-review.md): 651 tests passed on the
Windows host, including a warning-strict run, with Ruff, Mypy, package build, and
`git diff --check` also passing. That baseline does not replace the two B18
platform reports.

## User Story Traceability

The following table contains exactly the 48 User Stories from the PRD, using an
accurate short form where the full sentence would only repeat the actor and
motivation. Implementation links identify the current production boundary;
automated links identify public or contract test evidence.

| ID | User Story | Implementation evidence | Automated test evidence |
| --- | --- | --- | --- |
| US-01 | Run MyClaw locally without a multi-tenant platform. | The installable Python entry point and one-process Runtime are defined by [pyproject.toml](../pyproject.toml), [cli.py](../src/myclaw/cli.py), and [runtime.py](../src/myclaw/runtime.py). | Installed package/entry behavior: [test_package.py](../tests/test_package.py), [test_cli.py](../tests/test_cli.py). |
| US-02 | Running `myclaw` enters the REPL by default. | [cli.py](../src/myclaw/cli.py) composes [repl.py](../src/myclaw/repl.py). | Installed console entry and REPL behavior: [test_cli.py](../tests/test_cli.py), [test_repl.py](../tests/test_repl.py). |
| US-03 | One REPL supports continuous multi-turn conversation with Short-term Memory. | [runtime.py](../src/myclaw/runtime.py), [conversation.py](../src/myclaw/conversation.py), and [session_store.py](../src/myclaw/session_store.py) reuse the active Session and its unconsolidated suffix. | Multi-turn history and same-session reuse: [test_runtime.py](../tests/test_runtime.py), [test_conversation.py](../tests/test_conversation.py), [test_conversation_summary.py](../tests/test_conversation_summary.py). |
| US-04 | Every valid REPL start prepares a new Session by default. | Session identity preparation is in [session_store.py](../src/myclaw/session_store.py) and Runtime composition in [runtime.py](../src/myclaw/runtime.py). | Prepared Session behavior: [test_session_store.py](../tests/test_session_store.py), [test_repl.py](../tests/test_repl.py), [test_runtime.py](../tests/test_runtime.py). |
| US-05 | Exiting an empty REPL leaves no Session file. | [session_store.py](../src/myclaw/session_store.py) separates preparation from first durable append; [repl.py](../src/myclaw/repl.py) ignores blank/exit-only input. | Empty and exit-only REPL cases: [test_repl.py](../tests/test_repl.py). |
| US-06 | Sessions are grouped and isolated by Workspace. | [workspace.py](../src/myclaw/workspace.py), [agent_home.py](../src/myclaw/agent_home.py), and [session_store.py](../src/myclaw/session_store.py) define Workspace identity, slug, and directory. | Windows/POSIX/UNC slugs and Workspace-scoped stores: [test_workspace.py](../tests/test_workspace.py), [test_session_store.py](../tests/test_session_store.py). |
| US-07 | `/resume` provides an interactive picker for Sessions in the current Workspace. | [session_resume.py](../src/myclaw/session_resume.py), [management.py](../src/myclaw/management.py), and [repl.py](../src/myclaw/repl.py). | Listing, selection, switch, and history continuation: [test_session_resume.py](../tests/test_session_resume.py). |
| US-08 | The picker shows Session titles and times. | Picker summaries are produced by [session_store.py](../src/myclaw/session_store.py) and rendered through [session_resume.py](../src/myclaw/session_resume.py). | Stable summary fields/order and picker rendering: [test_session_resume.py](../tests/test_session_resume.py), [test_management_contracts.py](../tests/contract/test_management_contracts.py). |
| US-09 | Session titles are generated automatically. | [session_titles.py](../src/myclaw/session_titles.py) starts the asynchronous chat-route title task. | Non-blocking title generation and Session ownership: [test_session_title.py](../tests/test_session_title.py), [test_runtime_session_title.py](../tests/test_runtime_session_title.py). |
| US-10 | Title-generation failure still leaves a readable fallback title. | Fallback normalization and truncation are in [session_titles.py](../src/myclaw/session_titles.py). | Failure, empty output, Unicode limit, and `Untitled session`: [test_session_title.py](../tests/test_session_title.py). |
| US-11 | Main chat always streams output. | Streaming Model events are consumed by [conversation.py](../src/myclaw/conversation.py) and progressively rendered by [repl.py](../src/myclaw/repl.py). | Streaming contract, progressive output, and durable completion: [test_conversation.py](../tests/test_conversation.py), [test_repl.py](../tests/test_repl.py), [test_model_contracts.py](../tests/contract/test_model_contracts.py). |
| US-12 | Ctrl+C cancels only the active turn, not background work. | [interrupts.py](../src/myclaw/interrupts.py), [runtime.py](../src/myclaw/runtime.py), and [background_coordination.py](../src/myclaw/background_coordination.py). | Foreground-only repeated/idle interrupt and background survival: [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py), [test_repl.py](../tests/test_repl.py). |
| US-13 | `exit` or `quit` exits the REPL clearly. | [repl.py](../src/myclaw/repl.py) handles case-insensitive, whitespace-tolerant exit tokens; Runtime owns shutdown. | Exit/quit and all-settled shutdown: [test_repl.py](../tests/test_repl.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py). |
| US-14 | `/config` displays the current configuration. | [management_commands.py](../src/myclaw/management_commands.py) uses [management.py](../src/myclaw/management.py) and [config.py](../src/myclaw/config.py). | Complete renderable configuration view: [test_management_commands.py](../tests/test_management_commands.py), [test_management_views.py](../tests/test_management_views.py). |
| US-15 | Configuration output redacts API keys by default. | Recursive and malformed-text redaction are in [config.py](../src/myclaw/config.py) and [cli.py](../src/myclaw/cli.py). | Valid, invalid, escaped-key, and CLI redaction: [test_config.py](../tests/test_config.py), [test_security_web_secrets.py](../tests/test_security_web_secrets.py), [test_cli.py](../tests/test_cli.py). |
| US-16 | `/status` shows version, chat model, uptime, token estimate, and Session state. | [management.py](../src/myclaw/management.py), [management_commands.py](../src/myclaw/management_commands.py), and [runtime.py](../src/myclaw/runtime.py). | Exact management fields, actual values, and fallback route: [test_management_contracts.py](../tests/contract/test_management_contracts.py), [test_management_views.py](../tests/test_management_views.py), [test_runtime.py](../tests/test_runtime.py). |
| US-17 | `/memory` shows the complete latest on-disk Long-term Memory. | [management.py](../src/myclaw/management.py) reads the store instead of the Runtime snapshot. | Repeated latest-disk view and complete command rendering: [test_management_views.py](../tests/test_management_views.py), [test_management_commands.py](../tests/test_management_commands.py). |
| US-18 | `/dream` manually processes pending summaries. | [memory_task.py](../src/myclaw/memory_task.py) is exposed through [management_commands.py](../src/myclaw/management_commands.py). | Manual route, result summary, cursor success/failure: [test_memory_task.py](../tests/test_memory_task.py). |
| US-19 | `/dream` does not call a model when no summaries are pending. | The zero-work path is in [memory_task.py](../src/myclaw/memory_task.py). | Exact no-pending output and zero Provider calls: [test_memory_task.py](../tests/test_memory_task.py). |
| US-20 | Long-term Memory is maintained automatically. | [memory_scheduler.py](../src/myclaw/memory_scheduler.py) triggers [memory_task.py](../src/myclaw/memory_task.py) inside the Runtime. | Periodic execution, custom cron, silence, failure isolation, and close: [test_memory_scheduler.py](../tests/test_memory_scheduler.py). |
| US-21 | Long-term Memory has User Info, User Preference, Project Fact, and Lesson sections. | The fixed template is created by [agent_home.py](../src/myclaw/agent_home.py). | Exact first-start template and preservation: [test_agent_home.py](../tests/test_agent_home.py). |
| US-22 | Conversation Summary compresses early messages automatically. | [conversation_summary.py](../src/myclaw/conversation_summary.py) performs pre-chat threshold/budget consolidation. | Message-count/token triggers, cutoff, cursor, and summary generation: [test_conversation_summary.py](../tests/test_conversation_summary.py), [test_consolidation_recovery.py](../tests/test_consolidation_recovery.py). |
| US-23 | Original Session messages remain after summarization. | [conversation_summary.py](../src/myclaw/conversation_summary.py) advances metadata without deleting message records; [session_store.py](../src/myclaw/session_store.py) preserves them. | Repeated consolidation and byte-preserving metadata rewrite: [test_conversation_summary.py](../tests/test_conversation_summary.py), [test_session_store.py](../tests/test_session_store.py). |
| US-24 | Long-term Memory changes only when the model decides an update is needed. | [memory_task.py](../src/myclaw/memory_task.py) treats absence of `edit_file` as no update. | No-edit cursor advance, exact edit, and failed edit behavior: [test_memory_task.py](../tests/test_memory_task.py). |
| US-25 | File read, list, and search are available by default. | [file_tools.py](../src/myclaw/file_tools.py), [permission_policy.py](../src/myclaw/permission_policy.py), and [tool_gateway.py](../src/myclaw/tool_gateway.py). | Public conversation Tool loop and path boundaries: [test_readonly_tool_loop.py](../tests/test_readonly_tool_loop.py), [test_security_filesystem.py](../tests/test_security_filesystem.py). |
| US-26 | File creation and editing require confirmation. | [workspace_write_tools.py](../src/myclaw/workspace_write_tools.py) and [permission_policy.py](../src/myclaw/permission_policy.py). | Foreground approval/refusal, background refusal, execute-once, and protected paths: [test_workspace_write_tools.py](../tests/test_workspace_write_tools.py), [test_permission_loop.py](../tests/test_permission_loop.py). |
| US-27 | Shell automatically allows only a tiny built-in read-only list. | [shell_policy.py](../src/myclaw/shell_policy.py), [shell_process.py](../src/myclaw/shell_process.py), and [shell_tool.py](../src/myclaw/shell_tool.py). | Five exact forms, near misses, cwd/timeout, trusted Git, and process ownership: [test_shell_policy.py](../tests/test_shell_policy.py), [test_security_shell.py](../tests/test_security_shell.py), [test_shell_process.py](../tests/test_shell_process.py). |
| US-28 | WebSearch and WebFetch are enabled by default for public Internet access. | [web_search.py](../src/myclaw/web_search.py), [web_fetch.py](../src/myclaw/web_fetch.py), and Runtime catalog wiring. | Enable/disable catalogs and Provider-neutral results: [test_web_search.py](../tests/test_web_search.py), [test_web_fetch.py](../tests/test_web_fetch.py). Live status is recorded under External Smoke Evidence. |
| US-29 | WebFetch blocks local and private networks. | DNS, peer, and redirect validation are in [web_fetch.py](../src/myclaw/web_fetch.py). | Every non-public category, all DNS answers, peer pinning, and redirect revalidation: [test_web_fetch.py](../tests/test_web_fetch.py), [test_security_web_secrets.py](../tests/test_security_web_secrets.py). |
| US-30 | Large Tool results become Tool Artifacts. | [tool_artifacts.py](../src/myclaw/tool_artifacts.py) is applied centrally by [tool_gateway.py](../src/myclaw/tool_gateway.py). | Threshold, raw bytes, preview, safe name, and failure normalization: [test_tool_artifacts.py](../tests/test_tool_artifacts.py), [test_security_filesystem.py](../tests/test_security_filesystem.py). |
| US-31 | Tool Artifacts remain with their Session and are readable after resume. | [agent_home.py](../src/myclaw/agent_home.py), [tool_artifacts.py](../src/myclaw/tool_artifacts.py), and [file_tools.py](../src/myclaw/file_tools.py) enforce current-session artifact scope. | Durable references, commit/discard ownership, and exact read exception: [test_tool_artifacts.py](../tests/test_tool_artifacts.py), [test_security_filesystem.py](../tests/test_security_filesystem.py), [test_session_resume.py](../tests/test_session_resume.py). |
| US-32 | A user can create natural-language Scheduled Work. | [scheduled_work.py](../src/myclaw/scheduled_work.py) persists definitions and [scheduled_work_execution.py](../src/myclaw/scheduled_work_execution.py) runs them. | Exact seven-field record and complete cron turns: [test_scheduled_work.py](../tests/test_scheduled_work.py), [test_scheduled_work_execution.py](../tests/test_scheduled_work_execution.py). |
| US-33 | Creating Scheduled Work requires confirmation. | Creation flows through [permission_policy.py](../src/myclaw/permission_policy.py) and [tool_gateway.py](../src/myclaw/tool_gateway.py). | Approval creates one record; refusal/background ASK creates none: [test_scheduled_work.py](../tests/test_scheduled_work.py), [test_permission_loop.py](../tests/test_permission_loop.py). |
| US-34 | Scheduled Work uses a task-specific Session. | [scheduled_work_execution.py](../src/myclaw/scheduled_work_execution.py) prepares and reuses the task's Session ID. | First/repeated trigger history, title, route, and failure isolation: [test_scheduled_work_execution.py](../tests/test_scheduled_work_execution.py). |
| US-35 | Scheduled Work completion is shown when the REPL is idle, never over foreground streaming. | [background_coordination.py](../src/myclaw/background_coordination.py) brokers foreground and background events for [repl.py](../src/myclaw/repl.py). | Idle display, global ordering, and queue-until-terminal behavior: [test_background_coordination.py](../tests/test_background_coordination.py), [test_event_contracts.py](../tests/contract/test_event_contracts.py). |
| US-36 | `default`, `chat`, `memory`, and `cron` models can be configured separately. | Typed configuration is in [config.py](../src/myclaw/config.py), routing in [model_router.py](../src/myclaw/model_router.py), and consumers in Runtime/Memory/Scheduled Work. | Route schema and route-specific calls: [test_config.py](../tests/test_config.py), [test_model_router.py](../tests/test_model_router.py), [test_memory_task.py](../tests/test_memory_task.py), [test_scheduled_work_execution.py](../tests/test_scheduled_work_execution.py). |
| US-37 | An unavailable specific route falls back to `default`. | [model_router.py](../src/myclaw/model_router.py) owns fallback and the shared attempt budget. | Static/dynamic fallback, recovery, and terminal no-fallback cases: [test_model_router.py](../tests/test_model_router.py), [test_runtime.py](../tests/test_runtime.py). |
| US-38 | Anthropic and OpenAI-compatible providers are supported. | Official-SDK adapters are [providers/anthropic.py](../src/myclaw/providers/anthropic.py) and [providers/openai_compatible.py](../src/myclaw/providers/openai_compatible.py); factory wiring is in [runtime.py](../src/myclaw/runtime.py). | Adapter streaming/tool/error contracts and factory selection: [test_anthropic_provider.py](../tests/test_anthropic_provider.py), [test_openai_compatible_provider.py](../tests/test_openai_compatible_provider.py), [test_provider_factory.py](../tests/test_provider_factory.py). Real-provider status is recorded below. |
| US-39 | First run generates a configuration template. | [config.py](../src/myclaw/config.py) creates the template and [cli.py](../src/myclaw/cli.py) exits with edit guidance. | Create-once, atomic failure, and installed CLI behavior: [test_config.py](../tests/test_config.py), [test_cli.py](../tests/test_cli.py). |
| US-40 | Invalid configuration makes `myclaw` exit clearly instead of entering a partial REPL. | Startup gating is in [cli.py](../src/myclaw/cli.py), [config.py](../src/myclaw/config.py), and [runtime.py](../src/myclaw/runtime.py). | Parse/schema/default-route failures and exit behavior: [test_cli.py](../tests/test_cli.py), [test_config.py](../tests/test_config.py). |
| US-41 | `myclaw config` remains usable for inspecting bad configuration. | [cli.py](../src/myclaw/cli.py) exposes the non-interactive command through [config.py](../src/myclaw/config.py). | Parse-invalid and schema-invalid content remains safely inspectable: [test_cli.py](../tests/test_cli.py), [test_management_commands.py](../tests/test_management_commands.py), [test_security_web_secrets.py](../tests/test_security_web_secrets.py). |
| US-42 | Runtime Core only orchestrates replaceable model, Tool, and store boundaries. | [runtime.py](../src/myclaw/runtime.py) composes Protocol boundaries from [contracts/ports.py](../src/myclaw/contracts/ports.py). | Structural substitutability and injected boundary Runtime tests: [test_protocol_contracts.py](../tests/contract/test_protocol_contracts.py), [test_runtime.py](../tests/test_runtime.py). |
| US-43 | Conversation Port emits typed Agent Events so CLI only interacts and renders. | Event/Port types are in [contracts/events.py](../src/myclaw/contracts/events.py) and [contracts/ports.py](../src/myclaw/contracts/ports.py); [conversation.py](../src/myclaw/conversation.py) emits them. | Exact event shapes/order plus REPL rendering: [test_event_contracts.py](../tests/contract/test_event_contracts.py), [test_conversation.py](../tests/test_conversation.py), [test_repl.py](../tests/test_repl.py). |
| US-44 | Management Port handles management commands instead of disguising them as chat. | [contracts/management.py](../src/myclaw/contracts/management.py), [management.py](../src/myclaw/management.py), and [management_commands.py](../src/myclaw/management_commands.py). | Built-ins bypass Conversation; unknown slash input remains chat: [test_management_commands.py](../tests/test_management_commands.py), [test_repl.py](../tests/test_repl.py), [test_management_contracts.py](../tests/contract/test_management_contracts.py). |
| US-45 | Tool Gateway uniformly validates arguments, permission, and results. | [tool_gateway.py](../src/myclaw/tool_gateway.py) composes [permission_policy.py](../src/myclaw/permission_policy.py) and artifact/result normalization. | Invalid schema, unknown Tool, safe failure, permission loop, and normalized contracts: [test_tool_failure_semantics.py](../tests/test_tool_failure_semantics.py), [test_permission_loop.py](../tests/test_permission_loop.py), [test_tool_contracts.py](../tests/contract/test_tool_contracts.py). |
| US-46 | Provider adapters use official SDKs for streaming, Tool calls, and error semantics. | SDK dependencies are pinned in [pyproject.toml](../pyproject.toml); adapters are [providers/anthropic.py](../src/myclaw/providers/anthropic.py) and [providers/openai_compatible.py](../src/myclaw/providers/openai_compatible.py). | Injected official-client boundaries and factory selection: [test_anthropic_provider.py](../tests/test_anthropic_provider.py), [test_openai_compatible_provider.py](../tests/test_openai_compatible_provider.py), [test_provider_factory.py](../tests/test_provider_factory.py). |
| US-47 | Fake Provider and fake Tool tests avoid real API dependencies. | Reusable boundaries are [fixtures/provider.py](../tests/fixtures/provider.py) and [fixtures/tool.py](../tests/fixtures/tool.py). | Their scripted behavior is verified by [test_fake_provider.py](../tests/test_fake_provider.py) and [test_fake_tool.py](../tests/test_fake_tool.py), then reused throughout the suite. |
| US-48 | v0.1 excludes MCP and subagent spawning to stabilize the core Runtime boundary. | The accepted negative scope is explicit in [runtime contracts](myclaw-runtime-contracts.md) and the production catalog is fixed in [runtime.py](../src/myclaw/runtime.py) / [tool_gateway.py](../src/myclaw/tool_gateway.py); there is no MCP/subagent production module or catalog entry. | Exact built-in catalogs are characterized by [test_readonly_tool_loop.py](../tests/test_readonly_tool_loop.py), [test_web_search.py](../tests/test_web_search.py), [test_shell_policy.py](../tests/test_shell_policy.py), and [test_scheduled_work.py](../tests/test_scheduled_work.py). This is negative-scope evidence, not a claim that an absent feature was manually exercised. |

## Required Test Traceability

The PRD contains 35 Required test bullets. Each row maps one bullet to maintained
public or contract evidence; grouping multiple test files in one row does not
merge or omit the requirement.

| ID | PRD Required test | Evidence |
| --- | --- | --- |
| RT-M01 | Short-term Memory is the suffix after the Consolidation Cursor. | Exact suffix contract: [test_session_contracts.py](../tests/contract/test_session_contracts.py); assembled history: [test_conversation_summary.py](../tests/test_conversation_summary.py). |
| RT-M02 | Both token-budget and total-message-count summary triggers. | [test_conversation_summary.py](../tests/test_conversation_summary.py). |
| RT-M03 | Cutoff aligns to a user message on the main and fallback paths. | [test_conversation_summary.py](../tests/test_conversation_summary.py). |
| RT-M04 | Summary JSONL has only `index`, `timestamp`, and `content`. | Exact keys: [test_memory_scheduling_contracts.py](../tests/contract/test_memory_scheduling_contracts.py); persistence/index behavior: [test_conversation_summary.py](../tests/test_conversation_summary.py). |
| RT-M05 | Summary generation excludes Long-term Memory and does not immediately trigger Memory Task. | [test_conversation_summary.py](../tests/test_conversation_summary.py), [test_memory_scheduler.py](../tests/test_memory_scheduler.py). |
| RT-M06 | Chat fails when memory route and default fallback both fail. | [test_conversation_summary.py](../tests/test_conversation_summary.py), [test_model_router.py](../tests/test_model_router.py). |
| RT-M07 | `/dream` with no pending summaries does not call a model. | [test_memory_task.py](../tests/test_memory_task.py). |
| RT-M08 | Summary Cursor advances for no update/edit success and does not advance for edit failure. | [test_memory_task.py](../tests/test_memory_task.py). |
| RT-M09 | Memory Task batch size, cron, non-reentrancy, and restricted `edit_file` path. | [test_memory_task.py](../tests/test_memory_task.py), [test_memory_scheduler.py](../tests/test_memory_scheduler.py). |
| RT-M10 | Runtime Long-term Memory cache differs intentionally from `/memory` latest-disk view. | [test_memory_scheduler.py](../tests/test_memory_scheduler.py), [test_management_views.py](../tests/test_management_views.py). |
| RT-S01 | Workspace slug generation. | Windows drive, POSIX root, UNC, normalization, and Unicode: [test_workspace.py](../tests/test_workspace.py). |
| RT-S02 | Session ID/path, metadata first line, and OpenAI-style messages. | [test_common_contracts.py](../tests/contract/test_common_contracts.py), [test_session_contracts.py](../tests/contract/test_session_contracts.py), [test_session_store.py](../tests/test_session_store.py). |
| RT-S03 | Empty REPL does not persist a Session. | [test_repl.py](../tests/test_repl.py). |
| RT-S04 | `/resume` lists only current-Workspace Sessions and switches correctly. | [test_session_resume.py](../tests/test_session_resume.py). |
| RT-S05 | Ordinary messages append as one line; metadata rewrites atomically. | [test_session_store.py](../tests/test_session_store.py), [test_atomic_files.py](../tests/test_atomic_files.py). |
| RT-S06 | Same-Runtime writes to one Session are serialized. | **PASS:** `test_same_runtime_concurrent_session_writes_preserve_every_record_and_usage` in [test_session_store.py](../tests/test_session_store.py) races 12 assistant appends after one user record, then proves all 13 unique records and exact cumulative usage reload without loss or duplication. |
| RT-S07 | Title generation is asynchronous, has fallback, counts usage, and shares the Session write lock. | [test_session_title.py](../tests/test_session_title.py), [test_runtime_session_title.py](../tests/test_runtime_session_title.py). |
| RT-S08 | Completed stream, interrupted partial, model failure, and Tool failure persistence rules. | [test_conversation.py](../tests/test_conversation.py), [test_repl.py](../tests/test_repl.py), [test_interrupted_tool_repair.py](../tests/test_interrupted_tool_repair.py), [test_security_fault_injection.py](../tests/test_security_fault_injection.py). |
| RT-T01 | File defaults, internal-file write protection, and out-of-scope denial. | [test_readonly_tool_loop.py](../tests/test_readonly_tool_loop.py), [test_workspace_write_tools.py](../tests/test_workspace_write_tools.py), [test_security_filesystem.py](../tests/test_security_filesystem.py). |
| RT-T02 | Shell exact allowlist, Workspace cwd, and 60-600 second timeout validation. | [test_shell_policy.py](../tests/test_shell_policy.py), [test_shell_process.py](../tests/test_shell_process.py). |
| RT-T03 | WebSearch enablement plus WebFetch private-network block, redirect recheck, and five-hop limit. | [test_web_search.py](../tests/test_web_search.py), [test_web_fetch.py](../tests/test_web_fetch.py). |
| RT-T04 | Scheduled Work creation confirmation and background ASK-as-refusal. | [test_scheduled_work.py](../tests/test_scheduled_work.py), [test_scheduled_work_execution.py](../tests/test_scheduled_work_execution.py), [test_permission_loop.py](../tests/test_permission_loop.py). |
| RT-T05 | Tool Artifact threshold, path, raw content, preview, and atomic write. | [test_tool_artifacts.py](../tests/test_tool_artifacts.py), [test_atomic_files.py](../tests/test_atomic_files.py), [test_security_filesystem.py](../tests/test_security_filesystem.py). |
| RT-T06 | Tool calls are not automatically retried after failure. | Execute-once safe failure: [test_tool_failure_semantics.py](../tests/test_tool_failure_semantics.py). |
| RT-P01 | Fake Anthropic/OpenAI-compatible adapter tests cover streaming, Tool calls, timeout, and error conversion. | [test_anthropic_provider.py](../tests/test_anthropic_provider.py), [test_openai_compatible_provider.py](../tests/test_openai_compatible_provider.py). |
| RT-P02 | Chat route is streaming; memory/cron may complete without streaming. | [test_model_contracts.py](../tests/contract/test_model_contracts.py), [test_anthropic_provider.py](../tests/test_anthropic_provider.py), [test_openai_compatible_provider.py](../tests/test_openai_compatible_provider.py). |
| RT-P03 | Route fallback, unknown route, unknown-protocol Provider, and unusable default. | [test_config.py](../tests/test_config.py), [test_model_router.py](../tests/test_model_router.py), [test_runtime.py](../tests/test_runtime.py). |
| RT-P04 | Five-attempt exponential backoff and `retry-after`. | [test_model_router.py](../tests/test_model_router.py). |
| RT-P05 | First run generates configuration and exits. | [test_config.py](../tests/test_config.py), [test_cli.py](../tests/test_cli.py). |
| RT-P06 | Invalid configuration makes `myclaw` exit directly. | [test_cli.py](../tests/test_cli.py), [test_config.py](../tests/test_config.py). |
| RT-P07 | `myclaw config` generation, complete display, and API-key redaction. | [test_cli.py](../tests/test_cli.py), [test_management_commands.py](../tests/test_management_commands.py), [test_security_web_secrets.py](../tests/test_security_web_secrets.py). |
| RT-P08 | `/config`, `/status`, `/resume`, `/memory`, and `/dream`. | [test_management_commands.py](../tests/test_management_commands.py), [test_management_views.py](../tests/test_management_views.py), [test_session_resume.py](../tests/test_session_resume.py), [test_memory_task.py](../tests/test_memory_task.py). |
| RT-P09 | Non-built-in `/...` input is sent to the model. | [test_repl.py](../tests/test_repl.py), [test_management_commands.py](../tests/test_management_commands.py). |
| RT-P10 | Ctrl+C, `exit`, `quit`, and REPL exit cancellation of background work. | [test_repl.py](../tests/test_repl.py), [test_runtime_shutdown.py](../tests/test_runtime_shutdown.py), [test_background_coordination.py](../tests/test_background_coordination.py). |
| RT-P11 | Tests do not imply a cross-process coordination guarantee for multiple REPLs. | The explicit negative boundary is [ADR-0001](adr/0001-file-first-local-persistence.md) and [runtime contracts](myclaw-runtime-contracts.md). Automated tests cover separate in-process Runtime ownership in [test_runtime.py](../tests/test_runtime.py) and [test_background_coordination.py](../tests/test_background_coordination.py); no cross-process lock guarantee is asserted. |

## Manual Acceptance Checklist

These statuses describe actual manual or production-boundary execution available
to this report, not automated substitutes.

| Check | Status | Evidence / reason |
| --- | --- | --- |
| Install the built wheel into a clean Windows environment; run `myclaw` and `myclaw config`. | PASS | [Windows validation](release/windows-validation.md) records an isolated venv outside the checkout, `pip check`, first-start config generation, Unicode paths, `myclaw config`, and secret redaction. |
| Install the built wheel into a clean POSIX environment; run the offline gates and CLI smoke. | BLOCKED | Local WSL registration points to a missing filesystem and no container/VM alternative was available. The authoritative `ubuntu-latest` run is pending in [POSIX validation](release/posix-validation.md). |
| Start `myclaw` with a dedicated real Provider configuration and complete a streamed multi-turn conversation. | NOT RUN | No dedicated release Provider credential/endpoint was supplied; local potential secrets were not inspected or used. |
| Verify automatic title generation, exit, restart, `/resume`, picker title/time, and resumed history with a real Provider. | NOT RUN | Depends on the dedicated real-Provider setup above. Automated coverage is listed under US-07 through US-10. |
| Exercise `/config`, `/status`, `/memory`, and `/dream` interactively against a release Agent Home. | NOT RUN | No dedicated manual release Agent Home/provider session was provisioned. |
| Approve and refuse Workspace file changes; confirm Agent Home and traversal denial. | NOT RUN | No separate manual destructive-safety Workspace was provisioned. Automated coverage is listed under US-25/US-26 and the [security review](security-fault-review.md). |
| Exercise all five automatic Shell forms, one approved non-automatic command, and one background refusal. | NOT RUN | No dedicated manual Shell acceptance Workspace was provisioned. Platform process evidence belongs in the validation reports. |
| Trigger Ctrl+C during streaming while Scheduled Work/Memory scheduling remains alive, then exit cleanly. | NOT RUN | Requires a real interactive Provider run and controlled background schedule. |
| Create and observe Scheduled Work without interleaving its completion over foreground streaming. | NOT RUN | Requires a real interactive Provider run and controlled cron schedule. |
| Run production WebSearch against a public query. | PASS | 2026-07-13 Windows host: `DuckDuckGoSearchBoundary` query `OpenAI official documentation` returned `results=2`; both result URLs were HTTPS. |
| Run production WebFetch through real DNS and HTTP against a public URL. | PASS | 2026-07-13 Windows host: `PublicWebFetchBoundary(SocketDNSResolver + AioHttp client)` fetched `https://example.com/`; normalized text had `chars=142` and contained `Example`. |

## External Smoke Evidence

| Boundary | Status | Credential / network conditions | Evidence |
| --- | --- | --- | --- |
| Anthropic real Provider | NOT RUN | No dedicated release API key was supplied. The task did not authorize discovery or use of a local user's potential credential. | Official-SDK behavior is automated in [test_anthropic_provider.py](../tests/test_anthropic_provider.py), but that is not a paid live-API smoke. |
| OpenAI-compatible real Provider | NOT RUN | No dedicated release base URL, model, and API key were supplied. No arbitrary local endpoint was assumed. | Official-SDK behavior is automated in [test_openai_compatible_provider.py](../tests/test_openai_compatible_provider.py), but that is not a live endpoint smoke. |
| Production WebSearch | PASS | Credential-free DuckDuckGo adapter; outbound public HTTPS available from the Windows host on 2026-07-13. Search result ordering/content remains backend-dependent. | Query `OpenAI official documentation` returned two Provider-neutral results with HTTPS URLs. |
| Production WebFetch | PASS | Credential-free request to public `https://example.com/`; real DNS and HTTP client used. No private/local target was contacted as part of release smoke. | Returned 142 normalized characters and contained `Example`. |

The network PASS rows prove the production adapters could reach one public target
from the recorded Windows environment. They do not guarantee future third-party
availability, every proxy/DNS environment, or a live POSIX network path.

## Known Risks

- Real Anthropic and OpenAI-compatible API smoke is `NOT RUN`. Adapter contracts,
  routing, retry, streaming, and Tool calls are tested with injected official SDK
  clients and fake Providers, but credentials, remote account policy, model
  availability, and billing behavior remain unverified for this release record.
- Same-user filesystem replacement can race final validation and OS I/O. Removing
  that TOCTOU class requires handle-relative or OS-specific confinement.
- Foreground-approved Shell is not an OS filesystem/network sandbox. Workspace
  `cwd` and confirmation do not confine child effects; see
  [ADR-0003](adr/0003-shell-permission-is-not-os-sandbox.md).
- File-first persistence has in-Runtime serialization but no cross-process
  coordination. Multiple REPLs can race Session/summary state or duplicate
  Scheduled Work triggers; see [ADR-0001](adr/0001-file-first-local-persistence.md).
- Indeterminate unreadable/conflicting publication preserves a Tool Artifact and
  releases rollback ownership rather than risk deleting a durable reference. It
  can conservatively retain an unreferenced artifact.
- Live WebSearch depends on a third-party backend, and public network/DNS/proxy
  behavior varies by environment. WebFetch SSRF tests are injected and exhaustive
  at the boundary, while the live smoke intentionally contacted only a public URL.
- Cross-platform release claims are valid only to the extent recorded in
  [Windows validation](release/windows-validation.md) and
  [POSIX validation](release/posix-validation.md). This document does not infer a
  platform PASS from another environment.

## Release Decision Inputs

Issue #36 can be closed only after the coordinator confirms that both platform
reports satisfy the offline test/lint/format/type/package gates, the wheel clean
install and CLI smoke criteria are evidenced, this document still contains
exactly 48 User Story rows and all 35 Required test rows, and every external/manual
status remains truthful. A real paid Provider PASS is not fabricated; its
explicit `NOT RUN` status and credential limitation are part of the acceptance
record.
