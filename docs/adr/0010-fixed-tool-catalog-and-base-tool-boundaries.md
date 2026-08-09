---
status: accepted
---

# Fix the Tool Catalog and BaseTool boundaries

The Tool subsystem uses one fixed Core Catalog in this order: Read File, Write
File, Edit File, List Dir, Glob, Grep, Exec, Web Search, Web Fetch, and
Schedule. Runtime configuration does not enable, disable, register, or replace
these capabilities. The Gateway constructs the catalog once, exposes detached
provider schemas, and is the only public invocation boundary. Plugins, MCP
hooks, subagent entries, dynamic registration, generic Tool retries, and
per-invocation Tool plans are not part of the product contract.

`BaseTool` owns the final preparation pipeline: safe argument casting, Schema
validation, concrete argument validation, concrete safety evaluation, and the
one-shot confirmation decision when a safety boundary is crossed. Concrete
Tools do not override preparation or confirmation-finished hooks. Tool
execution returns text; expected failures become safe Tool Errors and
unexpected failures are logged once and normalized by the Gateway.

File Tools resolve relative and absolute paths using the current Workspace.
Workspace State is part of that resolved Workspace and has no additional
MyClaw-specific access block; operating-system permissions remain authoritative.
A path resolving outside the Workspace requires confirmation. Exec runs one
direct Bash process with bounded input checks and best-effort DNS/destructive
command checks. MyClaw does not claim process-tree ownership or OS-level
filesystem or network isolation.

Successful Tool content exceeding `runtime.max_tool_result_chars` is handled by
`BaseTool` beneath `.myclaw/artifacts/<session_id>/<tool_call_id>.txt` (with a
UUID fallback for an invalid call ID). The result contains a bounded prefix and
an optional Workspace-relative reference. There is no separate Artifact
module, rollback, cleanup, or ownership lifecycle.

This decision supersedes the affected parts of:

- ADR-0003, which required the historical Shell allowlist and owned permission
  policy. Exec now uses its fixed concrete safety checks and direct Bash
  lifecycle while retaining the statement that cwd checks are not an OS
  sandbox.
- ADR-0005, which reserved Workspace State from generic file Tools and placed
  artifacts beside Session files. Workspace State is now accessible through
  the fixed file Tools, and artifacts use the unified Workspace-level path.
- ADR-0007, which made an owned process-tree adapter a required portability
  boundary. Exec no longer exposes that adapter; the host filesystem adapter
  remains only for unrelated Workspace State and persistence behavior.

The fixed catalog and the BaseTool/Gateway boundary are covered through the
Gateway seam and concrete Core Tool suites. Historical tests for the removed
modules and contracts are deleted rather than retained as compatibility
aliases.
