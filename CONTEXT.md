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

**Message Bus**:
The transient, in-process pair of unbounded FIFO message queues owned by one Agent Loop, with no independent lifecycle or close state. Inbound stores ordinary user input waiting for serial processing and provides locked snapshot, enqueue, dequeue, and drain operations; each mutation synchronously pushes the resulting immutable snapshot to one registered queue UI callback after releasing the lock. Outbound stores the foreground reasoning, response, Tool-call, and system-control presentation messages selected for one Terminal Conversation consumer; Tool results, Schedule Job output, and Memory Task output never enter this bus.
_Avoid_: persistent event log, broadcast bus, Schedule queue

**Inbound Message**:
One ordinary user input waiting in Message Bus Inbound, represented only by textual `content` and an empty `metadata` dictionary in the current product. Pending messages may be drained together into the terminal input, permanently removed from Inbound, joined with newlines for editing, and submitted again as one new message with fresh empty metadata; the currently executing message is unaffected.
_Avoid_: Management Command, Tool Confirmation, cancellation command, Agent Event

**Outbound Message**:
A transient foreground execution message placed in Message Bus Outbound for the Terminal Conversation's sole consumer. Its type is model reasoning, model response, Tool call, or system control, represented by textual `content` and runtime `metadata`; `_stream_delta` alone marks incremental content, `_stream_end` alone marks the end of one streamed segment, and `_streamed` alone marks the end of an Agent Run. System control represents a failed, cancelled, or maximum-iterations result. Tool calls expose the Tool name and arguments, while Tool results remain inside Agent Runner and the Conversation Session increment and never enter Outbound; Outbound Messages are not persisted.
_Avoid_: Agent Event, Session message, diagnostic log, broadcast event

**Agent Run Activity Group**:
A user-visible group of non-final model output and Tool activity belonging to one foreground Agent Run, kept distinct from that run's final output. It is a presentation of Agent Run progress rather than a separate conversation or persisted Session entity.
_Avoid_: Conversation, Agent Run, event log, transcript

**Command-line Conversation**:
The primary user-facing way to talk with the Personal Agent, presented as a full-screen terminal UI when `myclaw` runs without arguments. It has two primary regions: a scrollable conversation display above and a bottom input area; user messages align right, assistant messages align left, neutral full-width rows show operational events, and Tool Confirmation temporarily replaces the input area with a blocking decision UI. The input area remains editable during an active foreground run so Enter can queue more ordinary messages and Up can drain all pending messages back for newline-joined editing; Ctrl+C cancels only the active run. Each valid start prepares a new Conversation Session by default; a session is not persisted if the UI exits before any user message, and one process reuses its session across inputs. Only built-in slash commands are handled by the Management Port; other inputs beginning with `/` are treated as ordinary user messages and sent to the model. Users can run `/resume` to choose and switch to a prior session from the current Workspace; switching clears the conversation display and rebuilds it from the target Session's persisted messages and final Tool states rather than attempting to recreate transient Outbound Messages. The previous session is retained if it has messages and can be discarded if empty.
_Avoid_: Terminal session, shell command, chat channel, one-shot command, plain REPL

**Management Command**:
An explicit command for managing configuration, status, session resume, or memory without relying on natural-language conversation. In the first version, non-interactive management supports only `myclaw config` for config inspection and does not require `myclaw --help` as a product feature; if the configuration is missing, it generates the default configuration and displays its redacted content, while Terminal Conversation slash commands are `/config`, `/status`, `/resume`, `/memory`, and `/dream`. `/config` is read-only, displays configuration fields completely except for redacted plaintext API keys by default; when configuration parsing fails, it shows the parse error, path, and raw file content with text-level redaction of obvious API key lines; `/status` reports version, chat model, runtime uptime, estimated token state, current Session message count, `last_consolidated`, and current-Session cumulative model usage. Memory management can fully display the latest on-disk Long-term Memory through `/memory` without pagination and manually trigger a Memory Task through `/dream`, but cannot directly edit Long-term Memory in the first version; Schedule Jobs have no explicit management commands in the first version, and Session title rename or regeneration commands are out of scope.
_Avoid_: Tool call, chat instruction, task management, one-shot conversation

**Management Port**:
A code-level boundary used by Management Commands to inspect or modify configuration, session metadata, Schedule Jobs, and memory without depending on storage implementation details.
_Avoid_: Message Bus, direct file access, admin API

**Runtime Lifetime**:
The lifecycle of one Terminal Conversation process. It owns exactly one active Runtime Generation at a time and may replace that generation when the user switches Conversation Session; separate Terminal Conversation processes in the same Workspace remain uncoordinated, including Session Log writes. Background work only lives while the process is running. User messages are queued and processed serially in the foreground Agent Loop, while Memory Tasks and Schedule Jobs run asynchronously inside the active generation; Ctrl+C normally cancels only the active foreground run, while entering `exit` or `quit` with surrounding whitespace ignored and case-insensitive matching exits Terminal Conversation and immediately cancels running background tasks. There is no detached daemon runtime or one-shot runtime in the first version.
_Avoid_: Detached mode, daemon mode, persistent background process, one-shot command

**Runtime Generation**:
One complete prepared set of runtime components for the currently selected Conversation Session, including Message Bus, Model Router, Runtime Memory, Schedule Service, Memory Task scheduler, Agent Loop, Tool Gateway, and management services. A successful Session switch first prepares and validates a replacement generation, then synchronously abandons the old generation without waiting for active runs, persistence, scheduled work, memory work, or Provider shutdown; old Message Bus output and pending input are discarded, while detached cleanup is best effort. Forced replacement may lose unpersisted Session state, skip Memory updates under the existing Cursor contract, or repeat Schedule Job side effects.
_Avoid_: Runtime Lifetime, Conversation Session, Agent Run

**Session Log**:
Workspace-owned technical diagnostics for one Conversation Session, stored lazily under `<workspace>/.myclaw/logs/<session_id>.log` through an explicit validated Session context. WARNING and ERROR records use a Loguru file sink with an unbounded enqueue queue, UTF-8 output, exact 10 MiB rotation, and at most one retained historical file per Session. Sink setup and writes are fail-open, setup retries on the next context, and context exit removes the sink after an infinite drain. Same-Session concurrency is intentionally unsupported; no registry, lock, or cross-process coordination is provided. There is no per-record fsync, active redaction, or control escaping. Legacy Agent Home Runtime Log files remain untouched and are not updated.
_Avoid_: Runtime Log, Conversation log, chat transcript, audit log, activity feed

**Runtime Core**:
The product-level orchestration performed by Agent Loop for foreground messages. It coordinates Message Bus consumption, session state, Conversation Summary, context assembly, reusable Agent Runner execution, recovery, output publication, and memory access through injected runtime components rather than owning their concrete implementations. After a Tool Gateway call, Runtime Core asks BaseTool to externalize an oversized successful result before Conversation Session persistence; it does not roll artifacts back if later persistence fails.
_Avoid_: service container, microservice

**Agent Loop**:
The long-lived foreground orchestrator that owns one Message Bus and serially consumes its Inbound Messages. It holds injected runtime components such as Context Builder, active Conversation Session, Model Router, Runtime Memory, and Schedule Service; initializes the fixed Tool instances and shared Tool Gateway; prepares and recovers each Agent Run; invokes Agent Runner; persists the resulting Session increment; and publishes foreground Outbound Messages. Schedule Service may invoke a separately isolated Schedule execution callback that reuses Agent Loop resources without using its foreground Session, Message Bus, or cancellation state; Memory Tasks remain independent.
_Avoid_: Agent Runner, Runtime Lifetime, model loop, Schedule Service

**Agent Runner**:
A reusable, Session-independent ReAct execution engine that owns only its Model Router and is invoked by Agent Loop with initial model messages, a Model Route, a Tool Gateway, output callbacks, and control parameters. It reports live execution through callbacks and returns only this run's generated assistant and Tool message increment, final content, cumulative usage, a completed, failed, cancelled, or maximum-iterations finish reason, and optional structured error information. One iteration consists of one model call followed by all Tool calls requested by that response in their original sequential order; the configurable maximum defaults to 50 and cannot be lower than 50.
_Avoid_: Agent Loop, Conversation Session, Runtime Lifetime, Provider retry loop

**Agent Run**:
One complete Agent execution for one input against one Conversation Session, from accepting the input through Conversation Summary, context assembly, Agent Runner's model and Tool loop, Session increment, and persistence request. Agent Loop performs this flow for foreground input and publishes exactly one terminal Outbound Message; Schedule Service invokes an isolated callback that performs the same complete flow without using the foreground Message Bus. An Agent Run is not a Runtime Lifetime, Runtime Generation, individual model request, or Tool call.
_Avoid_: Agent Turn, Runtime, Model call, Tool call

**Runtime Context**:
Dynamic metadata added to a model call. Workspace path is assembled into the built-in identity system prompt, while current time, session ID, and similar per-turn metadata are prepended to the current user input.
_Avoid_: User instruction, long-term memory, session message

**System Prompt**:
The chat and schedule Model Routes' system-level context composed of the built-in identity prompt, Long-term Memory, and tool guidance; Conversation Summary generation does not inject Long-term Memory. If these system-level parts exceed the model context budget, the request fails with a user-facing configuration or memory-size error rather than silently trimming Long-term Memory.
_Avoid_: Runtime context, user message, conversation summary

**Conversation Session**:
A durable conversational thread owned by one Workspace and represented during a foreground Runtime by one active `Session` instance. The Session ID uses an automatic system-local timestamp plus UUID4 and its title is generated asynchronously after the first user input without blocking the first chat response, falling back to a truncated first-user-message title if title generation fails; title generation is not a message but counts toward cumulative usage. Each Session is persisted as `.myclaw\sessions\<session_id>.jsonl` only after it has a message. The first JSONL line is a strict header with exactly `session_id`, `created_at`, `updated_at`, `last_consolidated`, and `metadata`; later lines are JSON-native user, assistant, and tool message dictionaries with `role`, `content`, and local-time `timestamp` plus provider-relevant fields and JSON-compatible extensions. During a turn the active Session is the in-memory authority. After terminal turn work, `persist()` captures a complete deep-copied state and schedules an ordered atomic replacement with at most three asynchronous attempts; exhausted failures remain silent and a later complete snapshot may persist the still-authoritative in-memory state. `close()` makes at most three bounded synchronous attempts. A user-confirmed forced Session switch instead abandons the active Session: pending snapshots and active work are cancelled without waiting, no final save is attempted, and any state not already on disk may be lost. There is no acknowledgement, user-facing persistence error, or stronger crash consistency guarantee. A late title may wait for a later turn or close, and Summary state may diverge from `last_consolidated` after a crash. Existing schema-versioned files are unsupported, with no migration or version dispatch, and separate Terminal Conversation processes are not coordinated. Final model failures are persisted as assistant error messages, and tool execution failures are persisted as flat tool results containing status, content, and an optional artifact reference without a nested error field. Assistant Tool calls preserve the provider's raw JSON argument text. If a turn is interrupted normally, completed assistant or tool messages are retained, streamed partial assistant content is persisted as an interrupted/error assistant message, and unfinished tool calls are materialized as tool error results; a forced Session switch deliberately skips this repair and does not roll back completed Tool side effects or orphaned Artifacts. Assistant messages may carry both content and tool call requests in the same OpenAI-style message, and tool messages carry the corresponding result or refusal. Provider-visible reasoning may be retained transiently inside one Tool-use loop when required by the Provider protocol, but it is not persisted as a Conversation Session message. The pre-#38 Tool-call object and nested Tool-error JSONL shapes are intentionally not accepted as backward-compatible input.
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
A Workspace-owned ordered summary stream stored at `.myclaw\memory\summary.jsonl` for earlier Conversation Session messages that have crossed `last_consolidated` after the context window limit or a configured turn-count boundary is reached. Summary compression runs synchronously in the outer Runtime before each foreground or Schedule Agent Run when either the effective Agent Run Model Route's context window limit or the configured total-message-count boundary is reached; it never runs from inside the model and Tool loop. It uses the memory Model Route with default fallback, and if the fallback model also fails, the Agent Run fails with a user-facing error. Token-budget compression selects roughly half the budget worth of early messages, and message-count compression selects roughly half the threshold worth of early messages; in both cases the cutoff is advanced to the next user message, falling back to the nearest previous user message if no later user exists, and messages before that user are summarized so the retained suffix starts at a user message. Conversation Summary updates `last_consolidated` directly after its own stream operation. The original Session messages may remain stored, but summary entries do not retain source Session or message-range identity, do not immediately trigger Memory Task, and are not directly injected into the main chat model context. A crash or failed Session snapshot may leave summary content and `last_consolidated` divergent, so summary work can repeat or be omitted.
_Avoid_: Long-term memory, raw history, manual note, session memory

**Long-term Memory**:
A Workspace-level durable memory stored at `.myclaw\memory\memory.md`, initialized at startup as a template with its four sections if missing, periodically considered from that Workspace's Conversation Summary by background processing, and updated only when the model judges that stable information should be retained. It is one structured Markdown document shared only across sessions in the same Workspace, loaded into Runtime at startup and fully injected as part of the system prompt for Agent runs without a first-version size cap; a successful Memory Task modification refreshes the Runtime's cached copy, while each Agent run keeps the one cached snapshot captured at its start so a refresh affects only later runs. It is divided into User Info, User Preference, Project Fact, and Lesson.
_Avoid_: Raw history, session archive, manual notes, vector database, conversation summary

**Memory Task**:
A task scheduled by User Configuration using a cron expression in the system local timezone or manually triggered by the `/dream` Management Command that uses the memory Model Route and a standard Tool Gateway containing only dedicated `read_file` and `edit_file` capabilities for the current Workspace's Long-term Memory, while the Memory Manager reads that Workspace's Summary Cursor and a globally configured batch size of Conversation Summary entries before constructing the memory prompt; the default batch size is 10. For a nonempty batch it persists the batch's final Summary Cursor before any model call; a Cursor failure aborts before model or Memory work, while any later model, Tool, cancellation, or Long-term Memory failure leaves the advanced Cursor unchanged and does not retry that batch. The dedicated edit capability is limited to the exact Long-term Memory file and remains separate from the main catalog's Workspace-scoped `edit_file` capability. Memory Task does not receive a Conversation Session ID, persist Tool Results, or create Tool Artifacts. Scheduled runs execute silently in the background while manual runs block in the foreground and report summary status without full memory diffs, a run that does not call edit_file is treated as no update, the default schedule is hourly, and Memory Task does not run concurrently with another Memory Task; scheduled triggers are skipped while a prior run is active, and manual triggers are rejected with a user-facing message if a Memory Task is already running; `/dream` returns `No pending summaries` without calling the model when there are no unprocessed Conversation Summary entries.
_Avoid_: Chat turn, session compression, full agent run

**Summary Cursor**:
The Workspace-owned persisted Conversation Summary index stored at `.myclaw\memory\.cursor` through which a Memory Task has consumed that Workspace's summary entries. A Memory Task persists the new Cursor before attempting any corresponding Long-term Memory update, and the Cursor never rolls back even when that update fails, so consumed summaries may permanently produce no Long-term Memory change. This Memory Task position is separate from the active Session's `last_consolidated`.
_Avoid_: Session position, checkpoint

**Lesson**:
A reusable experience that should change future Agent behavior or design judgment.
_Avoid_: Conversation summary, activity log, task note

**Tool Gateway**:
The only public Tool invocation boundary. Agent Loop constructs one fixed catalog of `BaseTool` instances during initialization; the Gateway caches their annotation-derived OpenAI Function Calling schemas and returns defensive schema snapshots. `ToolGateway.call()` parses raw JSON argument text, resolves the Tool, projects declared parameters, performs the allowed safe conversions and schema validation, runs concrete argument and safety checks, obtains any required one-shot Tool Confirmation through an injected per-run channel, executes once, and returns an immutable normalized result. Shared path, DNS, traversal, truncation, and Artifact behavior lives in BaseTool or small shared helpers; capability-specific rules remain in concrete Tools. The Gateway has no dynamic registration, plugin or MCP hooks, generic retry, global timeout, concurrency lock, Workspace ownership, persistence, or Artifact externalization.
_Avoid_: Direct tool registry, plugin executor, shell wrapper

**Tool Confirmation**:
A host-mediated, one-shot user decision bound to one validated Tool call in one live Agent Run. Interactive conversations present the normalized effective operation and accept `approved` or `declined` outside ordinary Session messages, while a caller without an interactive confirmation channel receives the standard Tool refusal behavior.
_Avoid_: Permission Policy, model approval, chat reply, persistent approval

**Tool Catalog**:
The ordered set of concrete `BaseTool` capabilities registered once with a Tool Gateway. The main catalog always contains, in order, Read File, Write File, Edit File, List Dir, Glob, Grep, Exec, Web Search, Web Fetch, and Schedule. There is no User Configuration enablement switch, dynamic registration, plugin, MCP, or subagent entry. Memory Task may construct a separate Gateway containing only its dedicated Long-term Memory Tools; it does not mutate the main catalog. Foreground and Schedule executions share the main Gateway; Schedule Tool uses task-local Schedule execution context to refuse recursive Job creation without affecting concurrent foreground calls. Schedule add/list/remove actions do not request confirmation, while unsafe paths and Exec/Web safety boundaries use the common one-shot confirmation protocol.
_Avoid_: Plugin list, command list, model tools, subagent registry, MCP registry

**Tool Artifact**:
An external `.txt` file named from the tool_call_id and stored under `.myclaw\artifacts\<session_id>\` when a successful Tool result exceeds the global `runtime.max_tool_result_chars` threshold, which defaults to 4096 characters. Foreground and Schedule Sessions use the same Artifact root; Schedule Session IDs carry their canonical `schedule_` prefix. The per-run externalizer supplies the active Session ID, while the shared Tool Gateway does not infer Artifact ownership or read Schedule execution context. BaseTool writes the raw result and returns an immutable content/reference pair with a bounded prefix preview; errors and refusals remain inline. There is no separate Artifact module, commit, rollback, callback, cleanup, or ownership lifecycle, so a successful artifact write may leave an accepted orphan file if later Conversation Session persistence fails.
_Avoid_: Tool result, attachment, memory entry

**Schedule**:
The domain module encompassing persistent scheduled tasks and the service that manages and runs them.
_Avoid_: Scheduled Work, a single scheduled task, the scheduling service

**Schedule Job**:
One Workspace-owned persistent task in the Schedule module, expressed as an `at`, `every`, or `cron` schedule and identified by a UUID `job_id`. A user-sourced Job is managed through natural-language Agent Tools, while a system-sourced Job is Runtime-owned and hidden from those public management operations; every Job derives one persistent Conversation Session ID as `schedule_<job_id>`, reuses that Session across executions, and retains it when the Job is deleted. Its Session and Memory behavior is identical to every other Conversation Session, with no separate lifecycle. Schedule-owned Sessions are not listed or selectable through the foreground `/resume` command.
_Avoid_: Schedule, Scheduled Work, shell cron job, reminder

**Schedule Service**:
The service that is the only Schedule Job management boundary, owns access to Schedule persistence, and triggers due Jobs through an `on_schedule_job` callback assigned after Agent Loop initialization and before the Service starts. The callback reuses Agent Loop runtime resources without publishing to the foreground Message Bus. Schedule Tool receives this Service directly and cannot access the Schedule Store or scheduled execution resources; users manage Schedule Jobs through natural-language Agent Tools.
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
The single global TOML configuration for one Personal Agent installation, stored at `~/.myclaw/config.toml`. It is organized around runtime, models, and memory sections, with model providers under `[models.providers.<provider_id>]` and model routes under `[models.routes.<route>]`; the fixed Tool Catalog is not configurable. Fields and tables outside that defined schema are discarded during runtime loading, while `myclaw config` reports undefined fields for inspection. If missing, the CLI generates a default configuration with one `openai-local` OpenAI-compatible provider scaffold whose base URL, API key, and model catalog are empty plus explicit unusable scaffolds for the default, chat, memory, and schedule Model Routes, then exits and asks the user to replace the placeholder values or remove purpose-specific routes before starting. If the configuration cannot be parsed, a defined field is structurally invalid, or `[models.routes.default]` is absent, running `myclaw` to start Terminal Conversation exits with a user-facing configuration error rather than entering a repair mode; Model Route provider/model usability is evaluated only when that route is first used, while non-interactive `myclaw config` remains available for inspecting configuration.
_Avoid_: Agent profile, session override, per-chat settings, identity prompt, repair mode
