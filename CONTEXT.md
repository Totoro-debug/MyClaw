# Personal Agent Runtime

This context defines the language for a local-first, single-user personal Agent runtime inspired by nanobot.

## Language

**Personal Agent**:
A host-neutral, local-first, single-user Agent runtime that can run continuously for one person and expose a command-line conversation interface as its primary entry point. Windows x64 is the currently validated environment; macOS Intel and Apple Silicon are intended compatibility targets but remain unverified. Remote APIs or chat channels, if present, are adapters to the same runtime rather than separate products.
_Avoid_: Bot platform, multi-tenant assistant, channel-first agent, agent platform

**Agent Home**:
The fixed `~/.myclaw/` location for the current operating-system account that stores global User Configuration and legacy Runtime Log files for the Personal Agent. Session technical diagnostics belong to Workspace State rather than Agent Home.
_Avoid_: Project workspace, session directory, install directory, configurable data root

**Workspace**:
The current user-selected working directory for a Personal Agent interaction, identified by its normalized absolute path under the host's native path semantics without inferring a Git root or searching ancestor directories. It owns the Personal Agent's non-global persistent state in its reserved `.myclaw/` directory; file capabilities resolve paths within the Workspace, including Workspace State, while operating-system permissions remain authoritative and Workspace cwd validation remains neither a filesystem sandbox nor a network sandbox.
_Avoid_: Agent Home, install directory, session directory, project ID

**Workspace State**:
The persistent Personal Agent state owned by exactly one Workspace and stored in `<workspace>/.myclaw/`. It includes Conversation Session history, the Memory System, Schedule Job state, Tool Artifacts, and lazily-created Session Logs, but never User Configuration; it is reserved local runtime state rather than ordinary Workspace content, is initialized with a Git ignore rule while remaining portable when the whole Workspace directory is copied, and is accessible to the fixed file Tools through normal Workspace path resolution.
_Avoid_: Agent Home, project source, global state, cache

**Conversation Port**:
A code-level boundary through which user-facing interfaces submit conversational input, receive Agent Events, and answer Tool Confirmation requests. Interfaces use this port without depending on sessions, tools, memory, or model providers; it is not a network port.
_Avoid_: CLI service, message bus API, run facade, network port

**Agent Event**:
A typed runtime event emitted through the Conversation Port, such as streamed text, tool activity, a nonterminal Tool Confirmation request, final output, or errors. Main chat conversations using the chat Model Route must support streamed text events in the first version; memory and Schedule Agent runs are not required to stream; completed turn messages are added to the active Session in memory and a complete snapshot is scheduled after terminal work, while ordinary tool activity events expose tool name and status summary rather than full arguments or results by default.
_Avoid_: Callback, log line, text chunk

**Command-line Conversation**:
The primary user-facing way to talk with the Personal Agent, available as an interactive REPL in the first version; running `myclaw` without arguments enters the REPL. Each valid REPL start prepares a new Conversation Session by default; a session is not persisted if the REPL exits before any user message, and one REPL process reuses its session across inputs. Only built-in slash commands are handled by the Management Port; other inputs beginning with `/` are treated as ordinary user messages and sent to the model. Users can run `/resume` inside the REPL to choose and switch to a prior session from the current Workspace using an interactive picker. The previous session is retained if it has messages and can be discarded if empty.
_Avoid_: Terminal session, shell command, chat channel, one-shot command

**Management Command**:
An explicit command for managing configuration, status, session resume, or memory without relying on natural-language conversation. In the first version, non-interactive management supports only `myclaw config` for config inspection and does not require `myclaw --help` as a product feature; if the configuration is missing, it generates the default configuration and displays its redacted content, while REPL slash commands are `/config`, `/status`, `/resume`, `/memory`, and `/dream`. `/config` is read-only, displays configuration fields completely except for redacted plaintext API keys by default; when configuration parsing fails, it shows the parse error, path, and raw file content with text-level redaction of obvious API key lines; `/status` reports version, chat model, runtime uptime, estimated token state, current Session message count, `last_consolidated`, and current-Session cumulative model usage. Memory management can fully display the latest on-disk Long-term Memory through `/memory` without pagination and manually trigger a Memory Task through `/dream`, but cannot directly edit Long-term Memory in the first version; Schedule Jobs have no explicit management commands in the first version, and Session title rename or regeneration commands are out of scope.
_Avoid_: Tool call, chat instruction, task management, one-shot conversation

**Management Port**:
A code-level boundary used by Management Commands to inspect or modify configuration, session metadata, Schedule Jobs, and memory without depending on storage implementation details.
_Avoid_: Conversation Port, direct file access, admin API

**Runtime Lifetime**:
The lifecycle of a Personal Agent runtime process. Each REPL invocation creates one MyClaw runtime instance, and the interactive REPL is the long-running foreground runtime; multiple REPL instances in the same Workspace are allowed and remain uncoordinated, including Session Log writes, and each runtime independently starts a Memory Task scheduler and a Schedule Service. Background work only lives while the runtime process is running. User messages are queued and processed serially in the foreground conversation lane, while Memory Tasks and Schedule Jobs run asynchronously inside the same runtime; Ctrl+C cancels only the active foreground turn; entering `exit` or `quit` with surrounding whitespace ignored and case-insensitive matching exits the REPL and immediately cancels running background tasks, and there is no detached daemon runtime or one-shot runtime in the first version.
_Avoid_: Detached mode, daemon mode, persistent background process, one-shot command

**Session Log**:
Workspace-owned technical diagnostics for one Conversation Session, stored lazily under `<workspace>/.myclaw/logs/<session_id>.log` through an explicit validated Session context. WARNING and ERROR records use a Loguru file sink with an unbounded enqueue queue, UTF-8 output, exact 10 MiB rotation, and at most one retained historical file per Session. Sink setup and writes are fail-open, setup retries on the next context, and context exit removes the sink after an infinite drain. Same-Session concurrency is intentionally unsupported; no registry, lock, or cross-process coordination is provided. There is no per-record fsync, active redaction, or control escaping. Legacy Agent Home Runtime Log files remain untouched and are not updated.
_Avoid_: Runtime Log, Conversation log, chat transcript, audit log, activity feed

**Runtime Core**:
The orchestration layer for a Personal Agent turn. It coordinates session state, context assembly, model routing, tool execution, and memory processing through code-level boundaries rather than owning their concrete implementations. After a Tool Gateway call, Runtime Core asks BaseTool to externalize an oversized successful result before Conversation Session persistence; it does not roll artifacts back if later persistence fails.
_Avoid_: Agent loop implementation, service container, microservice

**Agent Run**:
One complete Agent execution for one input against one Conversation Session, from accepting the input through Conversation Summary, context assembly, the model and Tool loop, Session persistence, and exactly one terminal payload. Foreground Conversation and Schedule Service call the same run function with caller-specific Model Route, streaming, and payload consumption; an Agent Run is not a Runtime Lifetime, individual model request, or Tool call.
_Avoid_: Agent Turn, Runtime, Model call, Tool call

**Runtime Context**:
Dynamic metadata added to a model call. Workspace path is assembled into the built-in identity system prompt, while current time, session ID, and similar per-turn metadata are prepended to the current user input.
_Avoid_: User instruction, long-term memory, session message

**System Prompt**:
The chat and schedule Model Routes' system-level context composed of the built-in identity prompt, Long-term Memory, and tool guidance; Conversation Summary generation does not inject Long-term Memory. If these system-level parts exceed the model context budget, the request fails with a user-facing configuration or memory-size error rather than silently trimming Long-term Memory.
_Avoid_: Runtime context, user message, conversation summary

**Conversation Session**:
A durable conversational thread owned by one Workspace and represented during a foreground Runtime by one active `Session` instance. The Session ID uses an automatic system-local timestamp plus UUID4 and its title is generated asynchronously after the first user input without blocking the first chat response, falling back to a truncated first-user-message title if title generation fails; title generation is not a message but counts toward cumulative usage. Each Session is persisted as `.myclaw\sessions\<session_id>.jsonl` only after it has a message. The first JSONL line is a strict header with exactly `session_id`, `created_at`, `updated_at`, `last_consolidated`, and `metadata`; later lines are JSON-native user, assistant, and tool message dictionaries with `role`, `content`, and local-time `timestamp` plus provider-relevant fields and JSON-compatible extensions. During a turn the active Session is the in-memory authority. After terminal turn work, `persist()` captures a complete deep-copied state and schedules an ordered atomic replacement; `close()` makes at most three bounded synchronous attempts. Ordinary persistence failures are silent: there is no acknowledgement, failure logging, or stronger crash consistency guarantee. A late title may wait for a later turn or close, and Summary state may diverge from `last_consolidated` after a crash. Existing schema-versioned files are unsupported, with no migration or version dispatch, and separate REPL processes are not coordinated. Final model failures are persisted as assistant error messages, and tool execution failures are persisted as flat tool results containing status, content, and an optional artifact reference without a nested error field. Assistant Tool calls preserve the provider's raw JSON argument text. If a turn is interrupted, completed assistant or tool messages are retained, streamed partial assistant content is persisted as an interrupted/error assistant message, and unfinished tool calls are materialized as tool error results. Assistant messages may carry both content and tool call requests in the same OpenAI-style message, and tool messages carry the corresponding result or refusal. The pre-#38 Tool-call object and nested Tool-error JSONL shapes are intentionally not accepted as backward-compatible input.
_Avoid_: Chat ID, terminal session, workspace, runtime checkpoint, background task

**Memory System**:
The three-layer memory structure owned independently by each Workspace: Short-term Memory, Conversation Summary, and Long-term Memory.
_Avoid_: Single memory store, vector memory, raw transcript archive

**Short-term Memory**:
The unconsolidated suffix of a Conversation Session used to continue the current thread.
_Avoid_: Chat log, transcript, prompt history, full session file

**`last_consolidated`**:
The mutable nonnegative message position held by the active Conversation Session. Messages before this position have already been summarized into Conversation Summary; Short-term Memory is the suffix beginning at this position. Conversation Summary assigns it directly and does not coordinate it with a Session snapshot through a journal or transaction.
_Avoid_: checkpoint, bookmark, session ID

**Conversation Summary**:
A Workspace-owned ordered summary stream stored at `.myclaw\memory\summary.jsonl` for earlier Conversation Session messages that have crossed `last_consolidated` after the context window limit or a configured turn-count boundary is reached. Summary compression runs synchronously before each model request in foreground and Schedule Agent runs when either the effective Agent run Model Route's context window limit or the configured total-message-count boundary is reached; it uses the memory Model Route with default fallback, and if the fallback model also fails, the Agent run fails with a user-facing error. Token-budget compression selects roughly half the budget worth of early messages, and message-count compression selects roughly half the threshold worth of early messages; in both cases the cutoff is advanced to the next user message, falling back to the nearest previous user message if no later user exists, and messages before that user are summarized so the retained suffix starts at a user message. Conversation Summary updates `last_consolidated` directly after its own stream operation. The original Session messages may remain stored, but summary entries do not retain source Session or message-range identity, do not immediately trigger Memory Task, and are not directly injected into the main chat model context. A crash or failed Session snapshot may leave summary content and `last_consolidated` divergent, so summary work can repeat or be omitted.
_Avoid_: Long-term memory, raw history, manual note, session memory

**Long-term Memory**:
A Workspace-level durable memory stored at `.myclaw\memory\memory.md`, initialized at startup as a template with its four sections if missing, periodically considered from that Workspace's Conversation Summary by background processing, and updated only when the model judges that stable information should be retained. It is one structured Markdown document shared only across sessions in the same Workspace, loaded into Runtime at startup and fully injected as part of the system prompt for Agent runs without a first-version size cap; a successful Memory Task modification refreshes the Runtime's cached copy, while each Agent run keeps the one cached snapshot captured at its start so a refresh affects only later runs. It is divided into User Info, User Preference, Project Fact, and Lesson.
_Avoid_: Raw history, session archive, manual notes, vector database, conversation summary

**Memory Task**:
A task scheduled by User Configuration using a cron expression in the system local timezone or manually triggered by the `/dream` Management Command that uses the memory Model Route and a standard Tool Gateway containing only dedicated `read_file` and `edit_file` capabilities for the current Workspace's Long-term Memory, while the Memory Manager reads that Workspace's Summary Cursor and a globally configured batch size of Conversation Summary entries before constructing the memory prompt; the default batch size is 10. For a nonempty batch it persists the batch's final Summary Cursor before any model call; a Cursor failure aborts before model or Memory work, while any later model, Tool, cancellation, or Long-term Memory failure leaves the advanced Cursor unchanged and does not retry that batch. The dedicated edit capability is independent of the refused foreground Workspace edit Tool. Memory Task does not receive a Conversation Session ID, persist Tool Results, or create Tool Artifacts. Scheduled runs execute silently in the background while manual runs block in the foreground and report summary status without full memory diffs, a run that does not call edit_file is treated as no update, the default schedule is hourly, and Memory Task does not run concurrently with another Memory Task; scheduled triggers are skipped while a prior run is active, and manual triggers are rejected with a user-facing message if a Memory Task is already running; `/dream` returns `No pending summaries` without calling the model when there are no unprocessed Conversation Summary entries.
_Avoid_: Chat turn, session compression, full agent run

**Summary Cursor**:
The Workspace-owned persisted Conversation Summary index stored at `.myclaw\memory\.cursor` through which a Memory Task has consumed that Workspace's summary entries. A Memory Task persists the new Cursor before attempting any corresponding Long-term Memory update, and the Cursor never rolls back even when that update fails, so consumed summaries may permanently produce no Long-term Memory change. This Memory Task position is separate from the active Session's `last_consolidated`.
_Avoid_: Session position, checkpoint

**Lesson**:
A reusable experience that should change future Agent behavior or design judgment.
_Avoid_: Conversation summary, activity log, task note

**Tool Gateway**:
The only public Tool invocation boundary. Runtime Core constructs one fixed catalog of `BaseTool` instances at startup; the Gateway caches their annotation-derived OpenAI Function Calling schemas and returns defensive schema snapshots. `ToolGateway.call()` parses raw JSON argument text, resolves the Tool, projects declared parameters, performs the allowed safe conversions and schema validation, runs concrete argument and safety checks, obtains any required one-shot Tool Confirmation through an injected per-run channel, executes once, and returns an immutable normalized result. Shared path, DNS, traversal, truncation, and Artifact behavior lives in BaseTool or small shared helpers; capability-specific rules remain in concrete Tools. The Gateway has no dynamic registration, plugin or MCP hooks, generic retry, global timeout, concurrency lock, Workspace ownership, persistence, or Artifact externalization.
_Avoid_: Direct tool registry, plugin executor, shell wrapper

**Tool Confirmation**:
A host-mediated, one-shot user decision bound to one validated Tool call in one live Agent Run. Interactive conversations present the normalized effective operation and accept `approved` or `declined` outside ordinary Session messages, while a caller without an interactive confirmation channel receives the standard Tool refusal behavior.
_Avoid_: Permission Policy, model approval, chat reply, persistent approval

**Tool Catalog**:
The ordered set of concrete `BaseTool` capabilities registered once with a Tool Gateway. The main catalog always contains, in order, Read File, Write File, Edit File, List Dir, Glob, Grep, Exec, Web Search, Web Fetch, and Schedule. There is no User Configuration enablement switch, dynamic registration, plugin, MCP, or subagent entry. Memory Task may construct a separate Gateway containing only its dedicated Long-term Memory Tools; it does not mutate the main catalog. Schedule add/list/remove actions do not request confirmation, while unsafe paths and Exec/Web safety boundaries use the common one-shot confirmation protocol.
_Avoid_: Plugin list, command list, model tools, subagent registry, MCP registry

**Tool Artifact**:
An external `.txt` file named from the tool_call_id and stored under `.myclaw\artifacts\<session_id>\` when a successful Tool result exceeds the global `runtime.max_tool_result_chars` threshold, which defaults to 4096 characters. BaseTool writes the raw result and returns an immutable content/reference pair with a bounded prefix preview; errors and refusals remain inline. There is no separate Artifact module, commit, rollback, callback, cleanup, or ownership lifecycle, so a successful artifact write may leave an accepted orphan file if later Conversation Session persistence fails.
_Avoid_: Tool result, attachment, memory entry

**Schedule**:
The domain module encompassing persistent scheduled tasks and the service that manages and runs them.
_Avoid_: Scheduled Work, a single scheduled task, the scheduling service

**Schedule Job**:
One Workspace-owned persistent task in the Schedule module, expressed as an `at`, `every`, or `cron` schedule and identified by a UUID `job_id`. A user-sourced Job is managed through natural-language Agent Tools, while a system-sourced Job is Runtime-owned and hidden from those public management operations; every Job derives one persistent Conversation Session ID as `schedule_<job_id>`, reuses that Session across executions, and retains it when the Job is deleted. Its Session and Memory behavior is identical to every other Conversation Session, with no separate lifecycle. Schedule-owned Sessions are not listed or selectable through the foreground `/resume` command.
_Avoid_: Schedule, Scheduled Work, shell cron job, reminder

**Schedule Service**:
The service that manages and triggers Schedule Jobs, then submits each triggered Job to the same complete Agent flow used by foreground conversation. Users manage Schedule Jobs through natural-language Agent Tools.
_Avoid_: Schedule, Schedule Job, detached background process

**Permission Policy**:
The concrete Tool safety checks for one normalized invocation; there is no separate Security or Permission Policy module. Tool Confirmation supplies explicit per-invocation consent but does not replace capability checks or create persistent approval. Relative and absolute file paths resolve under the current Workspace, including Workspace State; a resolved external path requires confirmation and actual operating-system permissions decide the result. Exec checks its bounded destructive-command, working-directory, and DNS rules before launching one Bash process. WebSearch and WebFetch are fixed Catalog capabilities; WebFetch validates every resolved target and redirect. Confirmation is never a claim of OS-level filesystem, network, or process isolation.
_Avoid_: Tool switch, safety flag, enablement

**Model Route**:
A named model purpose strictly limited to default, chat, memory, or schedule; route tables with other names in User Configuration are undefined fields and are discarded. User configuration maps each route to a provider ID, one model from that provider's model catalog, context window, max output, temperature, reasoning effort, and timeout; provider adapters silently ignore reasoning effort when unsupported, and each logical model call uses at most five provider attempts with exponential backoff and provider retry-after support. The requested route and default fallback share that attempt budget; fallback occurs when a specific route is missing, statically unusable, or returns a permanent route/provider-unavailable error, while invalid requests, context overflow, and cancellation do not fallback. The chat route covers main conversation and Session title generation, the memory route covers both Conversation Summary generation and Memory Task updates, and the schedule route covers Schedule Jobs.
_Avoid_: Model string, provider selection, backend, ad hoc route

**Model Provider**:
A concrete LLM backend identified by a kebab-case provider ID and configured with protocol, required base URL, plaintext API key, and a model catalog represented as a list of model IDs. The first version supports the `anthropic` and `openai-compatible` protocols; providers with unknown protocols are ignored.
_Avoid_: Model route, model string, gateway

**User Configuration**:
The single global TOML configuration for one Personal Agent installation, stored at `~/.myclaw/config.toml`. It is organized around runtime, models, and memory sections, with model providers under `[models.providers.<provider_id>]` and model routes under `[models.routes.<route>]`; the fixed Tool Catalog is not configurable. Fields and tables outside that defined schema are discarded during runtime loading, while `myclaw config` reports undefined fields for inspection. If missing, the CLI generates a default configuration with one `openai-local` OpenAI-compatible provider scaffold whose base URL, API key, and model catalog are empty plus explicit unusable scaffolds for the default, chat, memory, and schedule Model Routes, then exits and asks the user to replace the placeholder values or remove purpose-specific routes before starting. If the configuration cannot be parsed, a defined field is structurally invalid, or `[models.routes.default]` is absent, running `myclaw` to start the REPL exits with a user-facing configuration error rather than entering a repair mode; Model Route provider/model usability is evaluated only when that route is first used, while non-interactive `myclaw config` remains available for inspecting configuration.
_Avoid_: Agent profile, session override, per-chat settings, identity prompt, repair mode
