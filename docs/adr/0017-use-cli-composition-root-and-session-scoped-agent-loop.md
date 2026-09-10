---
status: accepted
---

# Use the CLI as Composition Root and Agent Loop as Runtime Generation

The CLI asynchronous root is the sole composition root. It owns the Workspace path/state, lifetime-scoped Message Bus, Model Router, MCP Runtime Manager, Memory Manager, Dream, Schedule Service, Management services, Terminal application, and current Agent Loop reference. Each Agent Loop is one Session-scoped Runtime Generation and constructs its own Session, Skill Loader, Context Builder, Conversation Summary Manager, Tool Gateway, and Agent Runner. Blackboard owns Task Framing generation. No Host, container, or differently named Runtime aggregate proxies this ownership.

The MCP Runtime Manager connects configured Servers before the first Agent Loop is created and supplies an immutable MCP Tool Snapshot for each generation. The Skill Loader instead publishes atomically replaceable frozen Skill state within a generation. Complete request preflight includes the candidate Skill projection and the Runtime Generation's corresponding initial Tool Exposure; their separate refresh rules follow [ADR-0016](0016-use-agent-home-skill-catalog-and-progressive-loading.md) and [ADR-0020](0020-expose-configured-mcp-tools-through-tool-gateway.md).

As the composition root, `myclaw/terminal/cli.py` may import its narrow MCP Runtime Manager boundary and Built-in Tool name reservation directly from their defining `myclaw.agent.tools` modules. This permission does not extend to other Terminal presentation modules, and the architecture does not provide a top-level MCP re-export module.

Agent Runner is the sole bounded ReAct implementation. Foreground and User Schedule work share the current Agent Loop's Runner and generation-owned Tool instances, but each uses an independent Agent Run Gateway view with isolated Catalog and Tool Activation state. Dream owns a separate Runner and restricted Gateway using that same engine. Single-completion requests call Model Router directly. Context construction follows [ADR-0018](0018-centralize-model-request-context-construction.md).

Session replacement first prepares a candidate MCP generation, pauses old foreground admission, waits for required Session persistence, and constructs and synchronously preflights an unstarted target Agent Loop. Active foreground work requires the existing force confirmation before destructive replacement. The successful replacement order is:

`Terminal quiesce -> Schedule pause_and_drain -> current unavailable -> old Loop abort/drain -> bus.reset -> Terminal rebind -> target.start -> MCP candidate activation -> publish current -> release admission barrier -> Schedule resume`

The target is started before it becomes current, and Schedule dispatch resumes only afterward. Target construction or preflight failure terminates the Terminal Conversation with a safe error; the CLI still owns and closes all prepared components. Failure after destructive replacement begins leaves Management unavailable and triggers shutdown rather than restoring a partially aborted generation.

After the Terminal returns and restores terminal state, shutdown awaits `Management deactivate -> Schedule pause_and_drain + close -> pending/active Loop abort or close -> MCP close -> Dream close -> Model Router close`. Cleanup continues after individual failures and preserves the primary error. Normal Loop closure persists and drains; forced abort may lose unpersisted state and does not undo accepted Tool, Artifact, Memory, or Schedule side effects.

Schedule pause cancels and drains dispatcher work, Agent Runs, and terminal commits. An uncommitted one-shot Job remains pending for later dispatch; recurring schedules keep their accepted cursor and do not replay the interrupted occurrence. Direct Schedule Service closure waits for already-started terminal commits, while CLI replacement and shutdown first invoke the cancelling pause operation.

Memory Manager owns Summary, Summary Cursor, Long-term Memory persistence, and the live memory snapshot, without model work. Dream consumes Summary batches through its dedicated `memory` lane and can only read or edit the exact Long-term Memory file. It advances the Summary Cursor before model execution, has no automatic retry or rollback, and refreshes the live memory view after edits. Schedule Service owns its Store and dispatches the persisted `job_id="dream", source="system"` Job directly to Dream without a Schedule Session. User Jobs reach the current Agent Loop through a CLI-owned callable.

Requirements: [CLI ownership](https://github.com/Totoro-debug/myclaw/issues/188), [Schedule terminal commits](https://github.com/Totoro-debug/myclaw/issues/195).
