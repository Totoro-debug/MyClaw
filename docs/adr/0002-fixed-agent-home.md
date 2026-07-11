# Fix Agent Home at `~/.myclaw/`

The Personal Agent uses the fixed `~/.myclaw/` Agent Home instead of supporting configurable data roots or Agent profiles. It contains the global `config.toml`, memory files under `memory/`, Workspace-grouped session files under `sessions/`, Tool Artifacts beside their session files, and Scheduled Work definitions in the root `scheduled-work.json` JSON array file. First startup creates the `memory/` and `sessions/` directories and initializes the Long-term Memory template; other runtime files are created on demand. This keeps the local-first single-user mental model simple, with the trade-off that migration, test isolation, and multi-environment workflows must work around one canonical location.

The accepted first-version layout is:

```text
~/.myclaw/
  config.toml
  scheduled-work.json
  memory/
    memory.md
    summary.jsonl
    .cursor
    pending-consolidations/
      <session_id>.json
  sessions/
    <workspace_slug>/
      <session_id>.jsonl
      artifacts/
        <session_id>/
          <encoded_tool_call_id>.txt
```

Only `memory/`, `sessions/`, and a missing `memory/memory.md` template are created during first startup. The other files and directories are created on demand.
