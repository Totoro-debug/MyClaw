---
status: accepted
---

# Use Run-Local Tool Permission Levels and Foreground Snapshots

MyClaw exposes three configured Tool Permission Levels through
`[runtime].permission_level`:

- `read-only` permits internal Workspace File reads directly. Every File write
  and every external File access requires one confirmation for that call.
- `workspace-write` permits internal Workspace File reads and writes directly.
  Every external File access requires one confirmation for that call.
- `full-access` permits valid File accesses directly. It removes permission
  prompts, not argument validation, business refusals, capability errors, or
  execution errors.

The default remains `workspace-write`. The Runtime Lifetime owns one
`RuntimePermissionControl` containing the configured level and the current
selected level. Every Runtime Generation receives the same control object, so
generation replacement does not reset the current value. A foreground Agent
Run captures one immutable `PermissionSnapshot` before Session title work,
manual Skill resolution, or Task Framing. The snapshot contains the current
Tool Permission Level and the process-lifetime resolved Exec Shell. Changing
the control during a run affects only a later run. Selection and capture are
synchronous operations confined to the CLI event-loop thread, so there is no
await boundary at which a caller can observe a partially updated selection.

## Authorization Boundary

`ToolGateway.call()` remains the only public Tool invocation seam. The
Gateway parses and prepares normalized arguments, performs validation and
business refusal checks, collects detached invocation facts, opens one
authorization session, requests at most one confirmation for that normalized
call, and only then executes it. Confirmation state is run-local and
call-local; it is never cached in a shared Tool instance.

The foreground Runtime Context and the foreground run Gateway receive the
same snapshot object. Runtime Context reports the permission level, resolved
Exec Shell family, and that permission checks may require confirmation. The
Gateway uses the snapshot for File policy while preserving the existing Exec
Host and Web confirmation behavior.

## Foreground Management and Runtime Lifetime

The CLI owns one `RuntimePermissionControl` for the process Runtime Lifetime
and reuses it for every foreground generation. The `/permission` command
selects one of the three levels for later foreground Agent Runs only. The
selection is not persisted to User Configuration, Conversation Session, or
Schedule; a successful generation replacement retains it, a failed replacement
does not change it, and a new process starts from the configured value.

The selector reports the current level and leaves it unchanged when the same
level is submitted. An upgrade to `full-access` opens a warning with Cancel
focused by default on every attempt. Full-Access removes ordinary permission
confirmation for foreground File Tools only; it is not an operating-system
sandbox and does not bypass validation, capability checks, business refusals,
or Tool errors. Exec, Web, MCP, and Schedule retain their existing behavior. A
process startup notice is shown at most once when the configured level is
Full-Access. `/config` reports the configured level, and `/status` reports both
configured and current foreground levels.

## File Facts and Host Paths

Read File, List Dir, Glob, and Grep emit canonical host `FileAccess` facts with
role `read`. Write File emits a `write` fact, while Edit File emits both `read`
and `write` facts for its target. Canonicalization uses the declared base path,
host path case and drive/UNC rules, symlink/junction/reparse resolution, and the
nearest existing ancestor for a missing write target. The policy compares
canonical paths with the canonical Workspace root; string prefixes are not
used.

Foreground model File Tools do not inherit the Skill Root exemption used by
internal Skill loading. A foreground model-issued File access beneath
`~/.myclaw/skills` is an ordinary external access unless it is also beneath
the Workspace. User Schedule Agent Runs retain their pre-change authorization
behavior until the later Schedule permission phase. Dream's private memory
Tool and Runtime persistence writes remain outside this model File policy.

Permission classification happens after preparation and business refusal, so
invalid arguments, hard errors, capability errors, and execution errors do
not become permission prompts. A declined confirmation never reaches the
execution boundary.

## Scope Boundaries

This decision does not alter the fixed Tool Catalog, Tool Exposure, Tool
Activation, Tool Search, MCP Tool behavior, or Schedule permission behavior.
It does not provide an operating-system sandbox. Full-Access removes ordinary
foreground File permission confirmation only; validation, capability checks,
business refusals, and Tool errors remain enforced. Exec, Web, MCP, and
Schedule permission behavior is unchanged by this decision.

The existing Skill Loader remains responsible for its own internal reads, and
ADR-0016 no longer defines a confirmation-free boundary for model-issued
`read_file` calls.

Consequences: permission policy is now expressed with structured, detached
facts while legacy safety reasons continue to support non-File Tools during
the contract transition. The foreground run has a stable authorization view
even if runtime control state changes while the model is working.
