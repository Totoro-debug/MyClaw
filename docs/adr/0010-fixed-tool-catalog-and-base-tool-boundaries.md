---
status: accepted
---

# Fix the Tool Catalog and BaseTool Boundaries

The main Tool Catalog is fixed in this order: Read File, Write File, Edit File, List Dir, Glob, Grep, Exec, Web Search, Web Fetch, and Schedule. User Configuration cannot enable, disable, register, or replace these capabilities, and the product has no plugin, MCP, subagent, generic retry, or per-invocation Tool-plan surface.

`ToolGateway.call()` is the sole public invocation boundary. It parses raw Provider arguments, resolves a Tool, and delegates the final cast, restricted Schema validation, concrete argument validation, safety evaluation, one-shot confirmation, execution, and normalized result pipeline to `BaseTool` and the concrete capability.

File Tools use normal Workspace path resolution, including Workspace State, and external targets require exact-call confirmation. Exec runs one direct Bash process with bounded destructive-command, cwd, timeout, and DNS checks; MyClaw does not claim process-tree ownership or OS-level filesystem, network, or process isolation.

`BaseTool` externalizes oversized successful results beneath `.myclaw/artifacts/<session_id>/<tool_call_id>.txt`, using a UUID fallback for an invalid call ID. Artifacts have no separate module, commit, rollback, cleanup, or ownership lifecycle.
