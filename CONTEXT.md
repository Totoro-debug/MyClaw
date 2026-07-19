# Personal Agent Runtime

This context defines the language for a local-first, single-user personal Agent runtime inspired by nanobot.

## Language

**Personal Agent**:
A local-first, single-user Agent runtime that can run continuously for one person and expose a command-line conversation interface as its primary entry point. Remote APIs or chat channels, if present, are adapters to the same runtime rather than separate products.
_Avoid_: Bot platform, multi-tenant assistant, channel-first agent, agent platform

**Agent Home**:
The fixed `~/.myclaw/` location that stores global Personal Agent data such as User Configuration, Conversation Summary, and Long-term Memory. Writes under Agent Home must be atomic so cancellation cannot corrupt existing files; first startup creates the base directories `memory/` and `sessions/`, Long-term Memory is initialized at startup if missing, Scheduled Work is stored in the root `scheduled-work.json`, consolidation recovery journals are created on demand under `memory/pending-consolidations/`, other runtime files are created on demand, and the first version does not maintain a separate persistent runtime log.
_Avoid_: Project workspace, session directory, install directory, configurable data root, runtime log

**Workspace**:
The current user-selected working directory for a Personal Agent interaction, identified by its normalized absolute path. Its Agent Home directory name is a lowercase path slug formed by joining path segments with `-` and replacing existing `-` characters inside segments with `_`, such as `D:\desktop\project\Demo-one` becoming `d-desktop-project-demo_one`. Conversation Sessions are grouped by Workspace under Agent Home; file capabilities are bounded to the Workspace plus explicitly allowed Agent Home paths, while Shell requires a Workspace cwd but is not an operating-system filesystem or network sandbox after foreground user confirmation.
_Avoid_: Agent Home, install directory, session directory, project ID

**Conversation Port**:
A code-level boundary through which user-facing interfaces submit conversational input and receive Agent Events. Interfaces use this port without depending on sessions, tools, memory, or model providers; it is not a network port.
_Avoid_: CLI service, message bus API, run facade, network port

**Agent Event**:
A typed runtime event emitted through the Conversation Port, such as streamed text, progress, tool activity, permission requests, final output, or errors. Main chat conversations using the chat Model Route must support streamed text events in the first version; memory and cron routes are not required to stream; streamed assistant text is written to the session only after completion, and tool activity events expose tool name and status summary rather than full arguments or results by default.
_Avoid_: Callback, log line, text chunk

**Command-line Conversation**:
The primary user-facing way to talk with the Personal Agent, available as an interactive REPL in the first version; running `myclaw` without arguments enters the REPL. Each valid REPL start prepares a new Conversation Session by default; a session is not persisted if the REPL exits before any user message, and one REPL process reuses its session across inputs. Only built-in slash commands are handled by the Management Port; other inputs beginning with `/` are treated as ordinary user messages and sent to the model. Users can run `/resume` inside the REPL to choose and switch to a prior session from the current Workspace using an interactive picker. The previous session is retained if it has messages and can be discarded if empty.
_Avoid_: Terminal session, shell command, chat channel, one-shot command

**Management Command**:
An explicit command for managing configuration, status, session resume, or memory without relying on natural-language conversation. In the first version, non-interactive management supports only `myclaw config` for config inspection and does not require `myclaw --help` as a product feature; if the configuration is missing, it generates the default configuration and displays its redacted content, while REPL slash commands are `/config`, `/status`, `/resume`, `/memory`, and `/dream`. `/config` is read-only, displays configuration fields completely except for redacted plaintext API keys by default; when configuration parsing fails, it shows the parse error, path, and raw file content with text-level redaction of obvious API key lines; `/status` reports version, chat model, runtime uptime, estimated token state, current session message count, consolidation cursor, and current-session cumulative model usage. Memory management can fully display the latest on-disk Long-term Memory through `/memory` without pagination and manually trigger a Memory Task through `/dream`, but cannot directly edit Long-term Memory in the first version; Scheduled Work has no explicit management commands in the first version, and Session title rename or regeneration commands are out of scope.
_Avoid_: Tool call, chat instruction, task management, one-shot conversation

**Management Port**:
A code-level boundary used by Management Commands to inspect or modify configuration, session metadata, scheduled work, and memory without depending on storage implementation details.
_Avoid_: Conversation Port, direct file access, admin API

**Runtime Lifetime**:
The lifecycle of a Personal Agent runtime process. Each REPL invocation creates one MyClaw runtime instance, and the interactive REPL is the long-running foreground runtime; multiple REPL instances in the same Workspace are allowed but not coordinated in the first version, and each runtime independently starts Memory Task and Scheduled Work schedulers. Background work only lives while the runtime process is running. User messages are queued and processed serially in the foreground conversation lane, while Memory Tasks and Scheduled Work run as asynchronous background tasks inside the same runtime; Ctrl+C cancels only the active foreground turn; entering `exit` or `quit` with surrounding whitespace ignored and case-insensitive matching exits the REPL and immediately cancels running background tasks, and there is no detached daemon runtime or one-shot runtime in the first version.
_Avoid_: Detached mode, daemon mode, persistent background process, one-shot command

**Runtime Core**:
The orchestration layer for a Personal Agent turn. It coordinates session state, context assembly, model routing, tool execution, and memory processing through code-level boundaries rather than owning their concrete implementations.
_Avoid_: Agent loop implementation, service container, microservice

**Runtime Context**:
Dynamic metadata added to a model call. Workspace path is assembled into the built-in identity system prompt, while current time, session ID, and similar per-turn metadata are prepended to the current user input.
_Avoid_: User instruction, long-term memory, session message

**System Prompt**:
The chat and cron model's system-level context composed of the built-in identity prompt, Long-term Memory, and tool guidance; Conversation Summary generation does not inject Long-term Memory. If these system-level parts exceed the model context budget, the request fails with a user-facing configuration or memory-size error rather than silently trimming Long-term Memory.
_Avoid_: Runtime context, user message, conversation summary

**Conversation Session**:
A durable conversational thread within a Workspace for a Personal Agent, identified by an automatic local-time timestamp-plus-UUID ID and a readable title generated asynchronously after the first user input without blocking the first chat response, falling back to a truncated first-user-message title if title generation fails; title generation is not written to conversation history but counts toward session cumulative model usage. Each session is persisted as `<session_id>.jsonl` under its Workspace session directory, with the first JSONL line holding metadata for ID, title, timestamps, consolidation cursor, and cumulative model usage, followed by OpenAI-style user, assistant, and tool messages; writes within the same session are serialized to preserve message order, ordinary messages are appended as single JSONL lines, and metadata updates atomically rewrite the session file; session write serialization is only within one runtime and does not coordinate multiple REPL processes. Final model failures are persisted as assistant error messages, and tool execution failures are persisted as tool error results, keeping session roles limited to user, assistant, and tool. If a turn is interrupted, completed assistant or tool messages are retained, streamed partial assistant content is persisted as an interrupted/error assistant message, and unfinished tool calls are materialized as tool error results. Assistant messages may carry both content and tool call requests in the same OpenAI-style message, and tool messages carry the corresponding result or refusal.
_Avoid_: Chat ID, terminal session, workspace, runtime checkpoint, background task

**Memory System**:
The three-layer memory structure of a Personal Agent: Short-term Memory, Conversation Summary, and Long-term Memory.
_Avoid_: Single memory store, vector memory, raw transcript archive

**Short-term Memory**:
The unconsolidated suffix of a Conversation Session used to continue the current thread.
_Avoid_: Chat log, transcript, prompt history, full session file

**Consolidation Cursor**:
The position in a Conversation Session through which earlier messages have already been summarized into Conversation Summary.
_Avoid_: Checkpoint, bookmark, session ID

**Conversation Summary**:
A global ordered summary stream stored at `~/.myclaw/memory/summary.jsonl` for earlier conversation messages that have crossed the Consolidation Cursor after the context window limit or a configured turn-count boundary is reached. Summary compression runs synchronously before the chat model call when either the context window limit or the configured total-message-count boundary is reached; it uses the memory Model Route with default fallback, and if the fallback model also fails, the chat request fails with a user-facing error. Token-budget compression selects roughly half the budget worth of early messages, and message-count compression selects roughly half the threshold worth of early messages; in both cases the cutoff is advanced to the next user message, falling back to the nearest previous user message if no later user exists, and messages before that user are summarized so the retained suffix starts at a user message. The original session messages may remain stored, but summary entries do not retain source session or message-range identity, do not immediately trigger Memory Task, and are not directly injected into the main chat model context.
_Avoid_: Long-term memory, raw history, manual note, session memory

**Long-term Memory**:
Agent-level durable memory stored at `~/.myclaw/memory/memory.md`, initialized at startup as a template with its four sections if missing, periodically considered from Conversation Summary by background processing, and updated only when the model judges that stable information should be retained. It is one structured Markdown document shared across sessions by default, loaded at runtime startup and fully injected as part of the system prompt for main chat without a first-version size cap; updates made by Memory Task take effect for chat only after runtime restart. It is divided into User Info, User Preference, Project Fact, and Lesson.
_Avoid_: Raw history, session archive, manual notes, vector database, conversation summary

**Memory Task**:
A task scheduled by User Configuration using a cron expression in the system local timezone or manually triggered by the `/dream` Management Command that uses the memory Model Route and a restricted Tool Gateway that can read current Long-term Memory and edit only `~/.myclaw/memory/memory.md`, while the Memory Manager reads the Summary Cursor and a globally configured batch size of Conversation Summary entries before constructing the memory prompt; the default batch size is 10. Scheduled runs execute silently in the background while manual runs block in the foreground and report summary status without full memory diffs, a run that does not call edit_file is treated as no update, the default schedule is hourly, and Memory Task does not run concurrently with another Memory Task; scheduled triggers are skipped while a prior run is active, and manual triggers are rejected with a user-facing message if a Memory Task is already running; `/dream` returns `No pending summaries` without calling the model when there are no unprocessed Conversation Summary entries.
_Avoid_: Chat turn, session compression, full agent run

**Summary Cursor**:
The persisted Conversation Summary index stored as plaintext at `~/.myclaw/memory/.cursor` through which a Memory Task has already processed summary entries. It advances when the Memory Task completes without calling edit_file, or after a needed edit succeeds; it does not advance when a required Long-term Memory edit fails.
_Avoid_: Session cursor, consolidation cursor, checkpoint

**Lesson**:
A reusable experience that should change future Agent behavior or design judgment.
_Avoid_: Conversation summary, activity log, task note

**Tool Gateway**:
The boundary that exposes tools to the Agent Runtime as declared capabilities and returns normalized tool results. It owns tool resolution, parameter validation, permission policy, sandboxing, and audit semantics, but does not serialize concurrent foreground and background tool calls. Tool calls are not retried by the gateway; failures are returned as tool results.
_Avoid_: Direct tool registry, plugin executor, shell wrapper

**Tool Catalog**:
The set of tool capabilities exposed through a Tool Gateway. The main-agent catalog always includes built-in file read/write/search and scheduled work, may include shell execution plus WebSearch and WebFetch depending on User Configuration, and does not include subagent spawning or MCP tools. Web and shell enablement applies to both foreground conversation and Scheduled Work.
_Avoid_: Plugin list, command list, model tools, subagent registry, MCP registry

**Tool Artifact**:
An external `.txt` file named from the tool_call_id and stored under `artifacts/<session_id>/` beside the Conversation Session file for a tool result that exceeds the globally configured `max_tool_result_chars` threshold, which defaults to 50000 characters. The artifact stores the raw tool result, while the session tool message stores an artifact reference plus a simple truncated preview rather than the full result; the artifact is retained with its session by default.
_Avoid_: Tool result, attachment, memory entry

**Scheduled Work**:
A persisted natural-language Agent task stored in the Agent Home root `scheduled-work.json` JSON array file as an element with id, title, cron, prompt, created_at, enabled, and session_id, and scheduled to run later or repeatedly using the system local timezone. Scheduled Work is created with enabled=true in the first version and has no management entry point to change it; creating Scheduled Work requires user confirmation because it is a persistent side effect, but creation does not statically reject tasks that may later need disabled tools. When triggered, it writes the task prompt as a user message and starts an Agent turn through the cron Model Route with Long-term Memory injected in a task-specific Conversation Session, where the final result is stored as an assistant message rather than directly running shell commands; the same Scheduled Work does not run concurrently within a single runtime and skips a trigger if its previous run is still active in that runtime, with no cross-process coordination. It only triggers while the runtime process is alive. Results are saved to the task session and surfaced through the connected REPL via Agent Events when available; if a foreground turn is streaming, background completion prompts are queued until the foreground turn ends, and the first version does not implement notification adapters.
_Avoid_: Shell cron job, reminder, detached background process

**Permission Policy**:
An internal rule set that decides whether a tool capability is allowed, denied, or requires confirmation for a given scope. Required confirmations block the current turn until answered; accepted confirmations are not recorded as separate history entries, refusals are returned as tool results, and unanswered background confirmations are treated as refusals. File read, listing, and search default to allowed, while file write and edit default to confirmation except in restricted memory tasks; the main Agent may read Long-term Memory but its file edit tool cannot edit Long-term Memory, User Configuration, or Agent Home internal state files such as sessions, summary, cursor, and scheduled work definitions; file access outside Workspace and allowed Agent Home areas is denied rather than escalated to confirmation. Shell execution allows a cwd parameter only within the Workspace, allows the model to request a timeout that must be validated within the hardcoded 60–600 second range, automatically allows only the exact read-only forms `pwd`, `git status`, `git status --short`, `git diff --stat`, and `git diff --name-only`, requires foreground confirmation for other syntactically valid commands, and treats those confirmation requirements as refusals in background work; it does not claim operating-system filesystem or network isolation for an approved foreground process. Directory and file inspection should prefer file tools. Web access defaults to allowed; WebSearch uses a credential-free built-in adapter, initially DuckDuckGo, with no additional first-version restrictions, while WebFetch blocks localhost, private network ranges, and link-local addresses, including after each followed redirect, with a maximum of five redirects.
_Avoid_: Tool switch, safety flag, enablement

**Model Route**:
A named model purpose strictly limited to default, chat, memory, or cron; unknown route names in User Configuration are configuration errors. User configuration maps each route to a provider ID, one model from that provider's model catalog, context window, max output, temperature, reasoning effort, and timeout; provider adapters silently ignore reasoning effort when unsupported, and each logical model call uses at most five provider attempts with exponential backoff and provider retry-after support. The requested route and default fallback share that attempt budget; fallback occurs when a specific route is missing, statically unusable, or returns a permanent route/provider-unavailable error, while invalid requests, context overflow, and cancellation do not fallback. The chat route covers main conversation and Session title generation, the memory route covers both Conversation Summary generation and Memory Task updates, and the cron route covers Scheduled Work.
_Avoid_: Model string, provider selection, backend, ad hoc route

**Model Provider**:
A concrete LLM backend identified by a kebab-case provider ID and configured with protocol, required base URL, plaintext API key, and a model catalog represented as a list of model IDs. The first version supports the `anthropic` and `openai-compatible` protocols; providers with unknown protocols are ignored.
_Avoid_: Model route, model string, gateway

**User Configuration**:
The single global TOML configuration for one Personal Agent installation, stored at `~/.myclaw/config.toml`. It is organized around runtime, models, memory, and tools sections, with model providers under `[models.providers.<provider_id>]`, model routes under `[models.routes.<route>]`, and tool enablement under `[tools.web]` and `[tools.shell]`, both enabled by default, and defines runtime lifecycle settings, model providers, model routes, memory behavior, configurable web and shell tool enablement, and enabled capabilities, but not the built-in Agent identity; if missing, the CLI generates a default configuration with one `openai-local` OpenAI-compatible provider scaffold whose base URL, API key, and model catalog are empty plus explicit unusable scaffolds for the default, chat, memory, and cron Model Routes, then exits and asks the user to replace the placeholder values or remove purpose-specific routes before starting. If the configuration cannot be parsed, model configuration is incomplete, or the default Model Route is unusable, running `myclaw` to start the REPL exits with a user-facing configuration error rather than entering a repair mode; a missing default Model Route is reported with its required TOML table name, and non-interactive `myclaw config` remains available for inspecting configuration.
_Avoid_: Agent profile, session override, per-chat settings, identity prompt, repair mode
