---
status: accepted
---

# Store Workspace State in the Workspace

All persistent runtime state belongs to `<workspace>/.myclaw/`; global User Configuration and user-authored Skills belong to Agent Home. Workspace identity is the normalized absolute startup directory; MyClaw does not infer a Git root, search ancestors, or fall back to Agent Home or ephemeral storage when Workspace State cannot be initialized safely.

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

Startup creates the root, internal Git ignore rule, `memory/`, `sessions/`, and missing Long-term Memory template. Registration of the Dream System Job also creates or reconciles `schedule.json`; the remaining paths are created on demand. Known records validate their own formats, unknown entries and legacy scheduled-work state remain untouched, and the fixed file Tools can access Workspace State through normal Workspace path resolution subject to operating-system permissions.

Each persisted Schedule Job uses one strict canonical object shape with a required `title`. The decoder accepts a document containing only the exact pre-title object shape as a migration boundary and derives each title from the first non-empty message line using the Session title normalization rule. The in-memory Job is canonical immediately, but the file is rewritten only by the next successful Store mutation; mixed schema versions, partial hybrids, and unknown fields are rejected, and a failed write leaves the previous document authoritative.
