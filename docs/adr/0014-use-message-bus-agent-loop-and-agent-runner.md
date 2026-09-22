---
status: accepted
---

# Use Message Bus, Agent Loop, and Agent Runner

The CLI owns one transient Message Bus for the Runtime Lifetime. Each Agent Loop binds one active foreground Conversation Session to that bus and serializes ordinary foreground inputs. Inbound is an editable FIFO. Outbound has one Terminal Conversation consumer and carries sparse reasoning, response, Tool-call, and system-control presentation messages. Ownership and Session replacement follow [ADR-0017](0017-use-cli-composition-root-and-session-scoped-agent-loop.md).

Foreground model text is published as stream deltas. Exactly one terminal marker ends an Agent Run; it is not a replacement copy of the final response. Tool start notifications carry the Tool name as content and `tool_call_id` plus raw `arguments` as metadata. Tool completion notifications carry the same name and identifier with only `status` (`success`, `error`, or `refused`). They update the existing Tool row without exposing Tool Result content or Artifact references. Missing completion notifications never imply success. Tool Results remain in the Runner's model transcript and Session persistence. Schedule and Dream output never enter foreground Outbound.

Agent Runner is a reusable Session-independent ReAct engine whose constructor owns a Model Router and the narrow request-preparation collaborator required by its owning lane. Each invocation receives complete initial messages, a route, Tool Gateway, output and confirmation callbacks, result externalizer, cancellation policy, iteration limit, and failure policy. It returns the invocation's Provider-valid assistant/Tool increment, final content, four-field usage, finish reason, and optional Error Info. Foreground uses `chat` and User Schedule Jobs use `schedule`. Requests needing only a single completion call the Model Router directly; Dream follows that path as recorded in [ADR-0024](0024-use-one-shot-dream-model-request.md).

One iteration is one model call followed by every requested Tool call in Provider order. Provider retries do not consume iterations. The default and minimum limit is 50; the last allowed response completes its Tool calls before reporting `agent_iteration_limit`, unless normal completion or cancellation takes priority. Tool errors and refusals normally return to the model for continuation. Cancellation repairs incomplete assistant/Tool pairs without undoing accepted side effects.

Foreground and User Schedule execution use the same Agent Runner implementation through separate run-local instances, each bound to that run's Router and request preparer. User Schedule also uses its own Session, context, cancellation, externalizer, and Session Log. [ADR-0021](0021-defer-tool-schema-exposure-per-agent-run.md) replaces the requirement to share the foreground Gateway identity with a Schedule-specific available Catalog implemented by `ToolGateway.for_run()` that excludes `ScheduleTool`, while retaining the generation's MCP capabilities. Its run-local confirmation callback submits background envelopes to the Runtime Lifetime coordinator; it has no foreground output callback. Dream directly owns its one-shot Memory Model request and restricted edit execution.

Requirements: [Message Bus and Runner](https://github.com/Totoro-debug/myclaw/issues/162), [CLI ownership](https://github.com/Totoro-debug/myclaw/issues/188).
