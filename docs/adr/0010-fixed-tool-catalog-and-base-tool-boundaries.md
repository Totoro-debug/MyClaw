---
status: accepted
---

# Fix the Built-in Tool Catalog and BaseTool Boundaries

## Strict Bash Exec Boundary

On POSIX Bash, Read-Only and Workspace-Write use the shared typed assessment
with Bash-native facts. Direct read candidates are `pwd`, `ls`, `cat`, `head`,
`tail`, `wc`, `stat`, `file`, `grep`, `rg`, `find`, `sort`, `uniq`, `cut`,
and `diff`; direct write candidates are `mkdir`, `touch`, `cp`, `mv`, and
`rm`. Each command has a fixed option grammar and explicit path roles.
`grep`/`rg` patterns must be fixed, `find` cannot use action or delegation
expressions, and `cp`/`mv` distinguish source reads from destination writes.
A simple fixed pipeline is allowed when every command identity and operand is
classified. Bash identities must have one matching PATH entry for a native
executable outside the Workspace, or be the approved `pwd` builtin. A single
hit remains eligible when its parent PATH directory is a symlink. Each matching
PATH entry counts separately: a repeated directory, two directories where one
symlinks to the other, or two distinct executable targets is ambiguous and
requires one confirmation, even when the hits resolve to the same target. A
symlink at the executable's final path component is a shim and requires one
confirmation even with one hit; a symlink in a parent directory alone is not a
shim. Aliases, functions, scripts, other shims, unknown identities, Workspace
executables, dynamic expansions, redirection, command lists, control flow,
glob/home expansion, follow/watch modes, external preprocessors, and unlisted
commands also require one confirmation. Read-Only permits
only Workspace reads; Workspace-Write permits Workspace reads and writes.
Full-Access allows parseable non-catastrophic dynamic Bash commands directly,
but never bypasses catastrophic confirmation, inspector uncertainty, argument
validation, capability errors, or Tool execution errors.

The Runtime Generation Tool Catalog contains these ten Built-in Tools in fixed order: Read File, Write File, Edit File, List Dir, Glob, Grep, Exec, Web Search, Web Fetch, and Schedule. Configured MCP Tools follow them as defined by [ADR-0020](0020-expose-configured-mcp-tools-through-tool-gateway.md). Foreground and User Schedule Agent Run Tool Catalogs add the run-local Built-in Tool Search; the User Schedule Catalog also excludes Schedule, as defined by [ADR-0021](0021-defer-tool-schema-exposure-per-agent-run.md). User Configuration cannot enable, disable, register, or replace any Built-in capability.

`ToolGateway.call()` is the sole public invocation boundary. It parses raw Provider arguments, resolves a Tool, calls the final `BaseTool.prepare()` pipeline, obtains any one-shot confirmation, and invokes `execute_authorized(arguments, authorization)` before normalizing the result; the default authorized seam delegates to `execute_prepared(arguments)`. Built-in preparation casts, defaults, filters, and validates with a temporary restricted Schema; MCP preparation preserves the complete argument object without those transformations. Each Tool exposes a `parameters` dictionary, and `to_schema()` returns a detached projection whenever that Tool is exposed in a Model request.

`BaseTool.prepare()` returns one detached `ToolInvocationFacts` value after argument normalization, validation, and capability-specific fact collection. This structured contract supersedes the former tuple and free-form safety-reason return contract. `ToolPermissionPolicy` is the only authorization decision point; Tools do not return free-form confirmation reasons or open an alternate authorization path. The Gateway retains the call-local authorization session through execution so Host-backed Tools can authorize each audited hop with the same typed state.

File Tools use normal Workspace path resolution, including Workspace State, and apply the Tool Permission Level matrix in [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md) to canonical internal and external targets. Exec remains a fixed catalogued capability backed by a process-lifetime Host: Windows `auto` selects available PowerShell 7 (`pwsh`) and otherwise Windows PowerShell 5.1, while POSIX always uses Bash and ignores the Windows selector. Explicit `powershell` and `pwsh` never cross-fallback. Inspection and execution share the resolved executable, canonical cwd, minimal environment, timeout/cancellation handling, and no-profile flags; PowerShell uses `-NoLogo -NoProfile -NonInteractive`, and Bash uses no login, profile, or rc loading. The Host returns raw outcomes and typed assessments; only the Exec Tool/Gateway boundary creates Tool Results. A missing selected shell reports one sanitized startup diagnostic, remains catalogued, and returns a stable capability error at invocation. Inspector failure is conservative uncertainty and retains the existing confirmation semantics. This Host policy supersedes ADR-0007's direct-Bash-only clause and narrows ADR-0017's CLI import boundary to admit this Runtime Lifetime-owned Host contract. MyClaw does not claim process-tree ownership or OS-level filesystem, network, or process isolation.
At Read-Only and Workspace-Write, the PowerShell policy adds a stricter low-permission Exec boundary: only the fixed read candidates (`Get-ChildItem`, `Get-Content`, `Get-Item`, `Get-Location`, `Get-FileHash`, `Measure-Object`, `Select-Object`, `Sort-Object`, `Select-String`, `Test-Path`, `Resolve-Path`, `Format-List`, `Format-Table`, and `Out-String`) and the fixed write candidates (`New-Item`, `Set-Content`, `Add-Content`, `Clear-Content`, `Copy-Item`, `Move-Item`, `Rename-Item`, `Remove-Item`, and `Out-File`) can execute directly. Each candidate has a fixed parameter grammar and path-role table. Direct cmdlets must resolve uniquely to their canonical name in the expected Microsoft PowerShell module; aliases, functions, scripts, shims, ambiguous identities, Workspace executables, unknown parameters, dynamic operands, pipeline-fed paths, and non-FileSystem Provider paths require one confirmation. Read-Only permits only Workspace read paths; Workspace-Write permits Workspace read and write paths; Full-Access bypasses ordinary classification confirmation for valid non-catastrophic commands but never bypasses catastrophic confirmation, parser uncertainty, argument validation, capability errors, or Tool execution errors.
Approved Git read forms are limited to `status`, `diff`, `log`, `show`, `branch --list`, `rev-parse`, and `ls-files`. Git identities must resolve uniquely to a native executable outside the Workspace. Git inspection and execution use fixed environment controls that disable system/global configuration, pagers, external diff, fsmonitor, hooks, terminal prompts, and optional locks; `diff` and `show` also receive `--no-ext-diff --no-textconv` execution flags. Static AST and identity inspection never launches Git. After the Workspace boundary is available, and only when the canonical executable is outside the Workspace and every effective repository directory is canonicalized inside it, the Host uses that exact executable to audit local/worktree includes and clean/process filters. A positive, failed, or ineligible audit requires confirmation, as do unlisted forms.

`BaseTool` externalizes oversized successful results beneath `.myclaw/artifacts/<session_id>/<tool_call_id>.txt`, using a UUID fallback for an invalid call ID. Artifacts have no separate module, commit, rollback, cleanup, or ownership lifecycle.

ADR-0016 defines the Skill Loader's internal confirmation-free filesystem boundary; model-issued File accesses follow [ADR-0026](0026-tool-permission-levels-and-foreground-snapshots.md). This does not add or dynamically register a Tool.

Tool execution has no generic retry or Gateway-wide lock. Successful results exceeding the configured character limit (default 4096) are externalized; errors and refusals remain inline. An artifact write failure retains success with a bounded failure marker. Tool Result content is model/session data, not a sanitized terminal diagnostic: trusted MCP `isError` text is retained, while foreground Tool activity carries status only.
