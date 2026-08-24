---
status: accepted
---

# Store Workspace State in the Workspace

All persistent state except global User Configuration belongs to `<workspace>/.myclaw/`. Workspace identity is the normalized absolute startup directory; MyClaw does not infer a Git root, search ancestors, or fall back to Agent Home or ephemeral storage when Workspace State cannot be initialized safely.

The current layout is:

```text
.myclaw/
  .gitignore
  memory/
    memory.md
    summary.jsonl
    .cursor
  sessions/
    <session_id>.jsonl
  schedule-sessions/
    schedule_<job_id>.jsonl
  artifacts/
    <session_id>/
      <tool_call_id>.txt
  logs/
    <session_id>.log
  schedule.json
```

Startup creates the root, internal Git ignore rule, `memory/`, `sessions/`, and missing Long-term Memory template; the remaining paths are created on demand. Known records validate their own formats, unknown entries and legacy scheduled-work state remain untouched, and the fixed file Tools can access Workspace State through normal Workspace path resolution subject to operating-system permissions.
