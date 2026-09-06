# MCP Tool Support Specification

## Problem Statement

MyClaw currently exposes only its fixed Built-in Tool Catalog. A user who has an MCP Server cannot make the Server's Tools available to the Model through the existing Tool Gateway, even though MCP provides a standard discovery and invocation protocol. Adding a second invocation path would fragment parameter handling, safety boundaries, result normalization, Runtime Generation behavior, and failure semantics.

## Solution

Allow User Configuration to declare trusted MCP Servers using the official MCP Python SDK v2. The CLI-owned MCP Runtime Manager connects enabled Servers over stdio or Streamable HTTP, discovers their Tools, and creates an immutable MCP Tool Snapshot for each Runtime Generation. Each discovered capability is wrapped as an `MCPTool`, a `BaseTool` implementation that enters the same Tool Catalog and Tool Gateway as a Built-in Tool.

The Model receives all Built-in and MCP Tool schemas through the normal `tools` request argument. MCP Tool descriptions are not added to the System Prompt. MCP calls receive the complete argument dictionary without local schema validation, type conversion, or removal of additional fields. Results read only MCP `content` blocks and are converted to the existing text-only Tool Result boundary.

## User Stories

1. As a Personal Agent user, I want to configure an MCP Server, so that its Tools are available to my Model.
2. As a Personal Agent user, I want stdio MCP Servers, so that I can use local MCP processes.
3. As a Personal Agent user, I want Streamable HTTP MCP Servers, so that I can use network-hosted MCP processes.
4. As a Personal Agent user, I want one configuration item per `mcp_name`, so that each Server has one clear source of truth.
5. As a Personal Agent user, I want invalid MCP Server configuration to be ignored independently, so that one bad Server does not prevent MyClaw from starting.
6. As a Personal Agent user, I want a malformed TOML document to remain a fatal configuration error, so that unrelated configuration is never guessed.
7. As a Personal Agent user, I want stdio processes to inherit MyClaw's environment, so that the first version does not create a second environment-secret configuration system.
8. As a Personal Agent user, I want to set a stdio working directory relative to the Workspace, so that local Server commands can use Workspace files predictably.
9. As a Personal Agent user, I want to provide static HTTP headers for one Server, so that authenticated non-OAuth HTTP endpoints can be used.
10. As a Personal Agent user, I want connection and call timeouts, so that an unavailable Server cannot block the runtime indefinitely.
11. As a Personal Agent user, I want failed MCP connections reported briefly in the terminal, so that I know why configured capabilities are missing.
12. As a Personal Agent user, I want failures logged without command, URL, header, or secret content, so that diagnostics do not leak configuration.
13. As a Personal Agent user, I want `myclaw config` and `/config` to show sanitized MCP diagnostics, so that I can locate ignored configuration without exposing headers.
14. As a Personal Agent user, I want the default configuration to show commented stdio and HTTP examples, so that I can discover the feature without enabling a Server accidentally.
15. As a Model, I want Built-in and MCP Tools to have the same function schema shape, so that I can select either kind through one tools interface.
16. As a Model, I want MCP Tool schemas supplied in the request `tools` argument, so that MCP descriptions do not pollute the System Prompt.
17. As a Model, I want every Tool schema represented as a dictionary, so that provider adapters receive one uniform data shape.
18. As a Model, I want each request to rebuild the complete schema list from the current Tools, so that a Runtime Generation cannot accidentally reuse a stale aggregate schema.
19. As a Model, I want MCP Tool names to be deterministic and provider-safe, so that repeated requests address the same remote capability.
20. As a Personal Agent user, I want collisions between Built-in and MCP names resolved deterministically, so that one call never targets an ambiguous capability.
21. As a Personal Agent user, I want excessively long MCP names to use the documented fallback name, so that usable remote Tools are not discarded unnecessarily.
22. As a Model, I want an MCP Tool's original name used when it has no description, so that the schema still identifies the capability.
23. As a Model, I want nullable JSON Schema forms normalized to OpenAI-compatible forms, so that nullable MCP arguments remain expressible to the provider.
24. As an MCP Server, I want MyClaw to forward the complete argument object, so that nested values and additional fields reach my Tool unchanged.
25. As a Personal Agent user, I want MCP arguments not locally validated against a possibly incomplete schema, so that the MCP Server remains authoritative for argument semantics.
26. As a Personal Agent user, I want enabled MCP Tools to execute without an extra confirmation prompt, so that my explicit User Configuration trust decision is respected.
27. As a Personal Agent user, I want MCP Tool calls to use the same Gateway preparation and result pipeline as Built-in Tools, so that Sessions and Tool Artifacts remain consistent.
28. As a Personal Agent user, I want multiple text result blocks joined in order with newlines, so that no textual output is lost.
29. As a Personal Agent user, I want non-text result blocks converted with their string representation, so that the first version has a deterministic text boundary for every content block.
30. As a Personal Agent user, I want an empty MCP result represented as `(no output)`, so that successful empty calls remain visible to the Model.
31. As a Personal Agent user, I want `isError` content returned as the original ToolError message, so that Server-provided failure details are preserved.
32. As a Personal Agent user, I want call timeouts and Server Tool errors to affect only that Tool Result, so that transient business failures do not remove the Server from the Catalog.
33. As a Personal Agent user, I want a truly closed MCP session detected as unavailable, so that the next Runtime Generation can reconnect it.
34. As a Personal Agent user, I want cancelled MCP calls to follow Built-in cancellation behavior, so that Ctrl+C and shutdown remain predictable.
35. As a Personal Agent user, I want healthy MCP connections reused across `/resume`, so that resuming a Session does not reconnect every Server.
36. As a Personal Agent user, I want failed Servers retried before a new Runtime Generation, so that `/resume` can recover capabilities without disturbing the old Generation.
37. As a Personal Agent user, I want a failed reconnection not to invalidate the old Generation, so that an unavailable Server cannot make an otherwise usable Session unusable.
38. As a Personal Agent user, I want the Dream Agent to retain its restricted Tool Catalog, so that MCP capabilities do not silently widen memory maintenance operations.
39. As a Personal Agent user, I want User Schedule Agent Runs to share the foreground MCP Tool Snapshot, so that scheduled work sees the same configured capabilities.
40. As a Personal Agent user, I want `/status` to avoid MCP-specific fields, so that status remains a stable runtime view.
41. As a Personal Agent user, I want context budgeting to include the System Prompt, all messages, and all Tool schemas, so that the provider request is never budgeted against an incomplete projection.
42. As a Personal Agent user, I want local context overflow to use one stable error code and message, so that callers do not need separate handling for Memory, Skill, or Tool schema size.
43. As a developer, I want the Gateway to invoke a polymorphic prepared-execution seam, so that it does not branch on whether a Tool is Built-in or MCP.
44. As a developer, I want existing Built-in Tool casting, defaults, filtering, validation, and safety checks preserved, so that MCP support does not regress established capabilities.
45. As a developer, I want SDK-specific types isolated inside the MCP adapter, so that the Agent Runner, Provider adapters, Session, and Gateway retain narrow contracts.
46. As a developer, I want paginated Tool discovery, so that Servers with more than one page of Tools are complete.
47. As a developer, I want local stdio and Streamable HTTP integration tests, so that transport support is verified without network-dependent tests.
48. As a maintainer, I want the MCP dependency constrained to the official v2 line, so that protocol behavior does not drift across incompatible SDK generations.

## Implementation Decisions

- Use the official MCP Python SDK v2 with dependency constraint `mcp>=2,<3`.
- Add a user-level MCP configuration section with one Server item keyed by `mcp_name`. Support `enabled`, `transport`, stdio `command`/`args`/optional `cwd`, Streamable HTTP `url`/`headers`, `connect_timeout`, and `call_timeout`. Do not support `env`, `secret_env`, OAuth, or MCP reload commands.
- Validate MCP Server items independently. A valid TOML document with an invalid MCP item keeps the remaining configuration; an unparseable TOML document remains fatal. Invalid items and connection failures are omitted from runtime memory, logged with sanitized metadata, and briefly reported in the terminal.
- Make the CLI composition root own MCP Runtime Manager lifetime. Connect enabled Servers concurrently before the initial Agent Loop starts. Before `/resume` pauses the old Generation, reconnect only failed Servers concurrently and construct the candidate MCP Tool Snapshot. Close MCP clients after Agent Loop shutdown and before Dream and Model Router shutdown.
- Reuse healthy clients and discovered Tool definitions across `/resume`. Only SDK-confirmed session/transport closure enters the failed set. A timeout or `isError` response does not change availability.
- Freeze one immutable MCP Tool Snapshot per Runtime Generation. Include only successfully connected, discovered, valid, and uniquely named Tools. Keep Built-in Tools first, then sort MCP Servers and remote Tools by name.
- Keep each Tool's `parameters` as `dict[str, Any]`; do not cache a Gateway-level schema list. Build the Model request's complete Tool list by calling each Tool's `to_schema()` for every request.
- Keep Built-in Tool behavior by rebuilding a temporary internal Schema during preparation. Do not retain `PreparedToolCall`; `prepare()` returns prepared arguments and an optional safety reason as a tuple.
- Make `execute_prepared(arguments)` the polymorphic execution seam. Built-in Tools use the default keyword expansion; `MCPTool` sends the complete argument dictionary to the remote Tool.
- Do not locally validate, cast, or filter MCP invocation arguments. At load time, require each remote `inputSchema` to be a JSON-serializable dictionary with an object root; otherwise ignore that Tool.
- Recursively normalize only the nonstandard nullable form. Remove the nullable keyword, add `null` to string or array `type` values, and use an `anyOf` null wrapper when no type is present. Preserve existing `anyOf`, `oneOf`, `$ref`, and other JSON Schema constructs.
- Allocate `mcp_<mcp_name>_<remote_tool_name>` first. If it exceeds the provider-safe 64-character limit, try `mcp_<remote_tool_name>`; do not replace characters or truncate. Ignore candidates that remain invalid or collide with an earlier Tool.
- Use the remote Tool name as its description when the Server omits a description. Do not add MCP descriptions to the System Prompt.
- Read only `CallToolResult.content`. Join TextContent blocks in order with newlines, convert other blocks with `str(block)`, and use `(no output)` for empty content. Ignore structured content and output schemas.
- Convert `isError` content and timeout/closed-session conditions to ToolError according to the Tool contract. Let other protocol, SDK, and programming exceptions use the Gateway's generic failure path. Propagate cancellation unchanged.
- Treat all enabled MCP Tools as trusted capabilities and skip Tool Confirmation for them in foreground and User Schedule Agent Runs. Keep Dream on its restricted Gateway.
- Compute every request and compression budget from the actual System Prompt, complete message list, and complete Tool schema list. Use only `model_context_overflow` with the shared local overflow message.
- Keep `/status`'s existing shape and omit MCP-specific health, count, or listing fields. Its context estimate may include the active Tool schemas because they are part of the real request.

## Testing Decisions

- Tests assert externally observable behavior at the highest available seam; they do not assert SDK implementation details or private helper structure.
- The primary unit seam is the Tool Gateway with fake BaseTool/MCP client adapters. It covers dict preparation, Built-in compatibility, MCP full-object forwarding, confirmation bypass, result normalization, ToolError mapping, cancellation, and dynamic schema construction.
- Configuration tests cover valid and invalid Server items, per-item isolation, TOML parse failure, path resolution, timeout bounds, forbidden fields, header redaction, diagnostics, and the disabled default examples.
- MCP adapter tests use a fake SDK seam for pagination, schema normalization, naming, collisions, content blocks, `isError`, timeout, closed session, generic exception, and cancellation matrices.
- Separate local end-to-end tests exercise one stdio Server and one Streamable HTTP Server through the real MCP SDK v2. These tests must not access the public Internet.
- CLI lifecycle tests use spies/fakes to verify concurrent startup, sanitized terminal notices, `/resume` healthy reuse and failed reconnection, candidate Snapshot isolation, and shutdown ordering.
- Context-budget tests grow System Prompt, messages, and Tool schemas independently, then verify trigger, non-compressible checks, compression cutoff, `/status` estimates, unified code, and unified message.
- Preserve all existing Built-in Tool, Agent Runner, Session, Schedule, Provider, and CLI regression tests. The final gate is the full test suite plus static type, lint, and format checks.

## Out of Scope

- MCP Resources, Prompts, Sampling, Tasks, `input_required`, list-changed subscriptions, dynamic Tool reload, and long-running task polling.
- OAuth, client credential flows, token storage, user-configurable subprocess environments, and secret-manager integration.
- Automatic MCP call retries, global Tool locks, Tool result multimodal persistence, structured-content projection, outputSchema validation, and provider-specific MCP adapters.
- MCP use by Dream, new `/status` fields, new MCP management commands, and any second Agent runtime.
- Silent truncation of Tool schemas, automatic removal of Tools to fit context, or fallback to a different Server when one fails.

## Further Notes

This specification supersedes the fixed-Catalog and no-MCP portions of the earlier Tool boundary decision while retaining the existing Tool Gateway, Built-in Tool, confirmation, Session, Artifact, and Provider boundaries wherever this specification does not explicitly change them. The implementation must follow the reviewed MCP Tool support implementation plan and must stop for re-review if it changes configuration shape, trust policy, lifecycle ownership, result projection, or error mapping.
