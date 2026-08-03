---
status: accepted
---

# Store Workspace State in the Workspace

Session technical diagnostics are Workspace-owned and stored lazily in `.myclaw\logs\<session_id>.log`; ADR-0008 supersedes the former Agent Home Runtime Log ownership.

MyClaw stores all persistent state except global User Configuration in `<workspace>\.myclaw\`. Each Workspace owns isolated Conversation Sessions, Memory System state, Scheduled Work, Tool Artifacts, and Session Logs; copying the complete Workspace carries this state, while initialization creates an internal Git ignore rule when absent. MyClaw never reads, validates, or replaces an existing `.myclaw\.gitignore`, so a later user edit may intentionally expose Workspace State to version control. Workspace State is reserved from generic file inspection, with explicit read access only for Long-term Memory and the current Conversation Session's Tool Artifacts. The `.myclaw` name itself reserves MyClaw's ownership without a directory-level marker or schema version: an existing ordinary directory is accepted, known files validate their own formats, and unknown entries are preserved without being read, while a file, symbolic link, Junction, or Reparse Point at that path fails startup. Runtime startup also fails rather than falling back to Agent Home or an ephemeral mode when Workspace State cannot otherwise be initialized safely. Existing non-global files and legacy Runtime Log files under Agent Home are preserved but are no longer read or migrated. This decision supersedes ADR-0002's ownership and layout of non-global state while retaining its fixed Agent Home for User Configuration and preserved legacy files.

The Workspace State layout removes the former Workspace slug layer without changing individual persistence formats:

```text
.myclaw\
  .gitignore
  scheduled-work.json
  memory\
    memory.md
    summary.jsonl
    .cursor
    pending-consolidations\
      <session_id>.json
  sessions\
    <session_id>.jsonl
    artifacts\
      <session_id>\
        <encoded_tool_call_id>.txt
  logs\
    <session_id>.log
```

After global User Configuration loads and validates, REPL startup initializes `.myclaw\`, its internal ignore rule, `memory\`, `sessions\`, and a missing `memory\memory.md` template before accepting input. It does not create a Conversation Session JSONL file until the first message is persisted, and creates `artifacts\` only when a successful oversized Tool result is first externalized; summary, cursor, consolidation-journal, and Scheduled Work files also remain on demand. The `myclaw config` Management Command never initializes Workspace State.

Workspace identity remains the normalized absolute startup working directory. MyClaw neither infers a Git repository root nor searches ancestors for another `.myclaw\`, so starting from a parent directory and one of its subdirectories intentionally selects independent Workspace State.

On Windows, Workspace State inherits the Workspace's existing ACL and MyClaw does not rewrite its DACL. Git exclusion and Tool access rules do not prevent operating-system users who can read the Workspace from reading `.myclaw\` directly.

MyClaw does not set the Windows Hidden attribute on `.myclaw\`; the reserved directory remains normally visible to operating-system file browsers.
