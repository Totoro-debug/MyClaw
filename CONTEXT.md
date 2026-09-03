# Personal Agent Runtime

This context defines the language for an independently designed, local-first, single-user Personal Agent runtime.

## Language

**Personal Agent**:
A local-first, single-user Agent runtime that works continuously for one person through a Command-line Conversation.
_Avoid_: Bot platform, multi-tenant assistant, channel-first agent, agent platform

**Agent Home**:
The fixed account-global home of one Personal Agent installation, separate from every Workspace and Conversation Session.
_Avoid_: Project workspace, session directory, install directory, configurable data root

**Workspace**:
The user-selected directory that scopes one Personal Agent interaction and owns its non-global state and file capabilities.
_Avoid_: Agent Home, install directory, session directory, project ID

**Workspace State**:
Persistent Personal Agent state owned by exactly one Workspace rather than by the installation or operating-system account.
_Avoid_: Agent Home, project source, global state, cache

**Message Bus**:
The transient pair of Inbound and Outbound queues shared by one Command-line Conversation and its current Agent Loop for foreground conversation flow. Its public async operations are snapshot, put/get, pause/resume delivery, drain, and reset; its only synchronous operations bind and unbind the Inbound-changed callback. Its contents are cleared whenever the conversation resumes a Conversation Session, including its current one.
_Avoid_: persistent event log, broadcast bus, Schedule queue

**Inbound Message**:
One ordinary user input waiting in the Message Bus for serial foreground processing.
_Avoid_: Management Command, Tool Confirmation, cancellation command, Agent Event

**Outbound Message**:
One transient presentation message emitted by an Agent Loop for its Command-line Conversation.
_Avoid_: Agent Event, Session message, diagnostic log, broadcast event

**Agent Run Activity Group**:
A user-visible group of non-final model output and Tool activity belonging to one foreground Agent Run, distinct from that run's final output.
_Avoid_: Conversation, Agent Run, event log, transcript

**Command-line Conversation**:
The primary user-facing conversation with the Personal Agent, presented as a full-screen terminal experience.
_Avoid_: Terminal session, shell command, chat channel, one-shot command, plain REPL

**Management Command**:
An explicit user command for inspecting or changing runtime-managed state without relying on natural-language conversation.
_Avoid_: Tool call, chat instruction, task management, one-shot conversation

**Management Port**:
The boundary through which Management Commands use runtime capabilities without knowing their storage or implementation details.
_Avoid_: Message Bus, direct file access, admin API

**Runtime Lifetime**:
The lifetime of one Command-line Conversation process, including every Runtime Generation it owns over time.
_Avoid_: Detached mode, daemon mode, persistent background process, one-shot command

**Runtime Generation**:
One replaceable set of Session-bound runtime components owned by a Runtime Lifetime.
_Avoid_: Runtime Lifetime, Conversation Session, Agent Run

**Session Log**:
Workspace-owned technical diagnostics associated with one Conversation Session rather than with the whole installation.
_Avoid_: Runtime Log, Conversation log, chat transcript, audit log, activity feed

**Agent Loop**:
The Session-scoped product orchestrator that owns foreground state, consumes Inbound Messages serially, invokes Agent Runs, and publishes Outbound Messages through the Command-line Conversation's Message Bus.
_Avoid_: Agent Runner, Runtime Lifetime, model loop, Schedule Service

**Agent Runner**:
A reusable, Session-independent engine that performs one bounded model-and-Tool ReAct loop for any request that requires ReAct.
_Avoid_: Agent Loop, Conversation Session, Runtime Lifetime, Provider retry loop, single model call

**Agent Run**:
One complete Agent execution for one input against one Conversation Session, from input acceptance through its final outcome and persistence request.
_Avoid_: Agent Turn, Runtime, Model call, Tool call

**Blackboard**:
A hidden task definition attached to a Conversation Session and its foreground Agent Runs, containing one current goal and one completion boundary. It supports interpretation without controlling execution or exposing a task-management surface.
_Avoid_: Task list, plan, workflow state, progress tracker

**Task Framing**:
The interpretation of an eligible new foreground user input against the current Blackboard and latest assistant response to keep, replace, or clear the current task definition. It is mutually exclusive with a Manual Skill Invocation in the same Agent Run.
_Avoid_: Task decomposition, planning, orchestration, progress update

**Runtime Context**:
Dynamic facts about the current runtime and Agent Run supplied to a model call but not authored by the user.
_Avoid_: User instruction, Long-term Memory, Session message

**System Prompt**:
The stable system-level context that establishes Personal Agent identity, memory, and capability guidance for a model call.
_Avoid_: Runtime Context, user message, Conversation Summary

**Model Request Context**:
The provider-neutral ordered messages assembled for one model call, including its System Prompt, Runtime Context, projected conversational messages, and any model-visible execution continuation.
_Avoid_: Prompt, Conversation Session, raw Session transcript, Provider request payload

**Skill**:
A named, discoverable instruction package that guides an Agent Run through existing capabilities without registering Tools or expanding permissions.
_Avoid_: Tool, Plugin, Management Command, capability extension

**Skill Catalog**:
The ordered set of valid Skill metadata available for discovery without loading the corresponding Skill instructions.
_Avoid_: Tool Catalog, command list, loaded Skill content

**Skill Snapshot**:
The immutable set of validated Skill metadata and complete instructions captured and exposed by the Skill Loader at one successful load, retained until another successful reload or Runtime Generation replacement.
_Avoid_: live Skill directory, Tool Catalog, Runtime Lifetime cache

**Skill Invocation**:
The selection and application of one Skill's instructions to a foreground Agent Run, initiated explicitly by the user or autonomously by the model.
_Avoid_: Management Command, Tool capability, Skill discovery

**Manual Skill Invocation**:
A Skill Invocation initiated explicitly when the user names an available Skill in the foreground input.
_Avoid_: Autonomous Skill Invocation, always-loaded Skill, Management Command

**Conversation Session**:
A durable conversational thread owned by one Workspace and represented by one active in-memory Session authority during foreground execution.
_Avoid_: Chat ID, terminal session, Workspace, runtime checkpoint, background task

**Memory System**:
The three-layer memory structure owned by a Workspace: Short-term Memory, Conversation Summary, and Long-term Memory.
_Avoid_: Single memory store, vector memory, raw transcript archive

**Short-term Memory**:
The unconsolidated suffix of a Conversation Session used to continue its current thread.
_Avoid_: Chat log, transcript, prompt history, full Session file

**`last_consolidated`**:
The position in a Conversation Session separating messages already represented by Conversation Summary from Short-term Memory.
_Avoid_: checkpoint, bookmark, Session ID

**Conversation Summary**:
A Workspace-owned ordered stream of compact summaries derived from earlier Conversation Session messages.
_Avoid_: Long-term Memory, raw history, manual note, Session memory

**Long-term Memory**:
A Workspace-level durable memory of stable information intended to influence later Agent behavior across Conversation Sessions.
_Avoid_: Raw history, Session archive, manual notes, vector database, Conversation Summary

**Dream**:
A background or manually triggered memory process that turns new Conversation Summary entries into Long-term Memory.
_Avoid_: Memory Task, Chat turn, Session compression, full Agent Run

**Summary Cursor**:
The Workspace-owned position through which Dream has consumed the Conversation Summary stream.
_Avoid_: Session position, checkpoint

**Lesson**:
A reusable experience that should change future Agent behavior or design judgment.
_Avoid_: Conversation Summary, activity log, task note

**Tool Gateway**:
The sole public boundary for resolving, validating, authorizing, executing, and normalizing Tool calls.
_Avoid_: Direct Tool registry, plugin executor, shell wrapper

**Tool Confirmation**:
A host-mediated, one-shot user decision bound to one validated Tool call in one live Agent Run.
_Avoid_: Permission Policy, model approval, chat reply, persistent approval

**Tool Catalog**:
The ordered set of concrete capabilities available through a Tool Gateway.
_Avoid_: Plugin list, command list, model tools, subagent registry, MCP registry

**Tool Artifact**:
A durable external representation of an oversized successful Tool result associated with one Conversation Session.
_Avoid_: Tool result, attachment, memory entry

**Schedule**:
The domain encompassing persistent scheduled tasks and the service that manages and runs them.
_Avoid_: Scheduled Work, a single scheduled task, the scheduling service

**Schedule Job**:
One Workspace-owned persistent task in the Schedule domain with its own execution timing. A user-created Schedule Job has its own Conversation Session; a System Schedule Job may use a dedicated internal executor.
_Avoid_: Schedule, Scheduled Work, shell cron job, reminder

**System Schedule Job**:
A Schedule Job created and maintained by the Personal Agent for internal Runtime work rather than by the user. It is hidden from user Schedule listing and mutation.
_Avoid_: User Schedule Job, public Schedule, shell cron job

**Dream Schedule Job**:
The unique System Schedule Job that invokes Dream through its dedicated execution path without creating a Schedule Session or entering an Agent Loop.
_Avoid_: User Schedule Job, Memory Task scheduler, scheduled Agent Run

**Schedule Service**:
The sole management and execution boundary for Schedule Jobs within one Runtime Lifetime.
_Avoid_: Schedule, Schedule Job, detached background process

**Permission Policy**:
The Tool-specific safety rules that determine whether one normalized invocation can run directly, requires Tool Confirmation, or must be refused.
_Avoid_: Tool switch, safety flag, enablement

**Model Route**:
A named model purpose that resolves a model request without exposing Provider selection to its caller.
_Avoid_: Model string, provider selection, backend, ad hoc route

**Reasoning Effort**:
A five-level intent attached to one Model Route that asks its Model Provider to trade response capability and thoroughness against latency and cost for each request. Its built-in level is `medium` when User Configuration omits it.
_Avoid_: Thinking level, token budget, Conversation Session override

**Model Provider**:
A configured backend that implements model calls for one or more Model Routes.
_Avoid_: Model Route, model string, gateway

**User Configuration**:
The single account-global configuration that selects runtime, model, and memory behavior for a Personal Agent installation.
_Avoid_: Agent profile, Session override, per-chat settings, identity prompt, repair mode
