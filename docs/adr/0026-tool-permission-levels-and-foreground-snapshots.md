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
Gateway uses the snapshot for File policy and the strict PowerShell Exec
policy described below; Web confirmation behavior remains unchanged.

## Exec Permission Mapping

Exec inspection is a detached Host fact collection that happens after
argument normalization and before authorization. A missing selected shell is
a capability error at every permission level. When the selected shell exists,
an unavailable, failed, timed-out, malformed, or inconsistent inspector
requires one confirmation at every level. Catastrophic Exec matches also
require one confirmation at every level.

For PowerShell 5.1 and 7, Read-Only and Workspace-Write direct execution is
limited to the fixed candidate command and parameter grammars recorded in
ADR-0010. The Host returns canonical command identity, expected Microsoft
module, resolution count, and static path-role facts. The policy canonicalizes
those paths using the Exec cwd and Workspace root, accepts only the FileSystem
Provider, and applies Read-Only or Workspace-Write path rules. Unknown or
dynamic syntax, untrusted identity, pipeline-fed paths, unknown parameters,
and external paths require one confirmation. Full-Access directly executes
parseable non-catastrophic dynamic commands, while still enforcing normal
argument validation, capability checks, business refusals, and Tool errors.

Approved Git read forms use a unique native executable outside the Workspace.
The Host fixes Git configuration, pager, external-diff/textconv, fsmonitor,
hook, prompt, and optional-lock behavior through the process environment and
adds `--no-ext-diff --no-textconv` to `diff` and `show`. Static inspection does
not launch Git. After Workspace-aware canonicalization proves the executable
is external and every repository directory is internal, the exact resolved
Git executable audits effective local/worktree include and clean/process
filter configuration. Positive, failed, or ineligible audits, unlisted Git
forms, and Workspace-resident Git executables require confirmation at the
lower levels.

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
confirmation for foreground File Tools and parseable non-catastrophic
PowerShell Exec calls; it is not an operating-system sandbox and does not
bypass validation, capability checks, business refusals, catastrophic or
uncertain Exec confirmation, or Tool errors. Web, MCP, and Schedule retain
their existing behavior. A
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
foreground File and parseable non-catastrophic PowerShell Exec permission
confirmation; validation, capability checks, business refusals, catastrophic
or uncertain Exec confirmation, and Tool errors remain enforced. Web, MCP,
and Schedule permission behavior is unchanged by this decision.

The existing Skill Loader remains responsible for its own internal reads, and
ADR-0016 no longer defines a confirmation-free boundary for model-issued
`read_file` calls. Exec uses the same structured, detached authorization facts
for its PowerShell policy; legacy safety reasons remain only for the existing
non-PowerShell compatibility path.

Consequences: permission policy is now expressed with structured, detached
facts while legacy safety reasons continue to support non-File Tools during
the contract transition. The foreground run has a stable authorization view
even if runtime control state changes while the model is working.
