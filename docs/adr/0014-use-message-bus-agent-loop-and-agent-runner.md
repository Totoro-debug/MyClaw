---
status: accepted
---

# Use Message Bus, Agent Loop, and Agent Runner

One `AgentLoop` owns one transient `MessageBus`, one active foreground `Session`, the fixed Tool Gateway, and the serial foreground execution path. Inbound is an editable FIFO of ordinary user messages; Outbound has one Terminal Conversation consumer and carries only sparse reasoning, response, Tool-call, and system-control presentation messages. Tool results, Schedule output, and Memory Task output never enter Outbound.

`AgentRunner` is a reusable Session-independent ReAct engine whose constructor owns only `ModelRouter`. Each invocation receives initial messages, a `chat` or `schedule` route, the Tool Gateway, output and confirmation callbacks, a result externalizer, cancellation state, and an iteration limit; it returns only the invocation's Provider-valid assistant/Tool increment, final content, four-field usage, finish reason, and optional `ErrorInfo`.

One iteration is one model call followed by every requested Tool call in Provider order. Provider retries do not consume iterations. The default and minimum limit is 50; the fiftieth response completes its Tool calls and then returns `agent_iteration_limit` without a fifty-first model call unless normal cancellation takes priority.

Schedule execution is invoked through the Schedule Service's isolated callback. It shares the main Gateway and Runner identities but uses its own Session, context, cancellation, externalizer, and Session Log, passes `confirmation=None`, and publishes no foreground Outbound messages.

`RuntimeHost` owns exactly one active prepared Runtime Generation. Session replacement validates an unstarted target before synchronously aborting the old generation, rebinding the Terminal, and starting the target. Normal shutdown awaits component closure; forced abort abandons the old Session and may lose unpersisted state or leave accepted Tool, Artifact, Memory, or Schedule side effects.
