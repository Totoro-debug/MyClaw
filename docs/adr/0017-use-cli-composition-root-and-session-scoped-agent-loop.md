---
status: accepted
---

# Use the CLI as Composition Root and Agent Loop as Runtime Generation

MyClaw keeps Runtime Lifetime and Runtime Generation as domain concepts but removes the code-level Runtime layer. The CLI asynchronous root is the sole composition root: it owns the Workspace path/state, one lifetime-scoped Message Bus, Model Router, Memory Manager, Dream, Schedule Service, Management services, Terminal application, and the current Agent Loop reference. Each Agent Loop is one Session-scoped Runtime Generation and constructs its own Session, Skill Loader and immutable full Skill Snapshot, Context Builder, Conversation Summary Manager, Task Framer, Tool Gateway, and Agent Runner. No Host, container, or differently named Runtime aggregate may proxy that ownership.

In this contract, target preparation is a precondition: the CLI constructs and synchronously preflights the target Agent Loop before destructive cutover. The final linearization refinement formed during later implementation review is not a claim about the original parent issue wording. After preparation succeeds, the successful cutover is:

`quiesce_for_rebind -> pause_and_drain -> current unavailable -> old abort/drain -> bus.reset() -> rebind_agent_loop -> target.start() -> publish current -> schedule_service.resume()`

The target is successfully started and activated before the CLI publishes it as current; Schedule dispatch resumes only after that publication. Any target construction or preflight failure is fatal, terminates the Terminal Conversation with a safe error, and leaves the CLI `finally` block to shut down the still-owned components. Normal process shutdown remains awaited.

The actual CLI shutdown chain, after Terminal `run_async()` has returned and Terminal exit cleanup has run, is `Management deactivate -> Schedule pause_and_drain + close -> pending/active Agent Loop abort or close -> Dream close -> Model Router close`. This records the call dependency order rather than inventing a Terminal business-component close call.

Long-lived memory and scheduling responsibilities follow the same boundary. Memory Manager owns Summary, Summary Cursor, Long-term Memory persistence, and the live memory snapshot but performs no model work. Dream owns the dedicated memory Agent execution path. Schedule Service owns its Store and dispatches the persisted `job_id="dream", source="system"` Job directly to `Dream.run()` without a Schedule Session; user Jobs still use the current Agent Loop through a CLI-owned callable. Memory Task Scheduler and the `Workspace` path wrapper are removed.

This decision supersedes the ownership and replacement portions of ADR-0014 and the Runtime-Lifetime Skill loading scope in ADR-0016. The Message Bus protocol, Agent Runner boundary, Skill path permissions, and other unaffected decisions in those ADRs remain accepted.
