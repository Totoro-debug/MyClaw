---
status: accepted
---

# Fix the Built-in Tool Catalog and BaseTool Boundaries

The Built-in Tool Catalog is fixed in this order: Read File, Write File, Edit File, List Dir, Glob, Grep, Exec, Web Search, Web Fetch, and Schedule. User Configuration cannot enable, disable, register, or replace these Built-in capabilities. Configured MCP Tools extend the generation Catalog as defined by [ADR-0020](0020-expose-configured-mcp-tools-through-tool-gateway.md).

`ToolGateway.call()` is the sole public invocation boundary. It parses raw Provider arguments, resolves a Tool, calls the final `BaseTool.prepare()` pipeline, obtains any one-shot confirmation, and invokes `execute_prepared()` before normalizing the result. Built-in preparation casts, defaults, filters, and validates with a temporary restricted Schema; MCP preparation preserves the complete argument object without those transformations. Each Tool exposes a `parameters` dictionary, and `to_schema()` returns a detached projection for every Model request.

File Tools use normal Workspace path resolution, including Workspace State, and external targets require exact-call confirmation. Exec runs one direct Bash process with bounded destructive-command, cwd, timeout, and DNS checks; MyClaw does not claim process-tree ownership or OS-level filesystem, network, or process isolation.

`BaseTool` externalizes oversized successful results beneath `.myclaw/artifacts/<session_id>/<tool_call_id>.txt`, using a UUID fallback for an invalid call ID. Artifacts have no separate module, commit, rollback, cleanup, or ownership lifecycle.

ADR-0016 defines a confirmation-free `read_file` boundary for canonical paths beneath `~/.myclaw/skills`; it does not add or dynamically register a Tool.

Tool execution has no generic retry or Gateway-wide lock. Successful results exceeding the configured character limit (default 4096) are externalized; errors and refusals remain inline. An artifact write failure retains success with a bounded failure marker. Tool Result content is model/session data, not a sanitized terminal diagnostic: trusted MCP `isError` text is retained, while foreground Tool activity carries status only.
