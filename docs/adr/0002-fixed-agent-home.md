# Fix Agent Home at `~/.myclaw/`

The Personal Agent uses the fixed `~/.myclaw/` Agent Home instead of supporting
configurable data roots or Agent profiles. Agent Home owns global User
Configuration and preserves legacy Runtime Log files. ADR-0005 supersedes the
former Agent Home ownership of non-global runtime state: Conversation Sessions,
Conversation Summary, Long-term Memory, Scheduled Work, Tool Artifacts, and
Session Logs belong to the current Workspace's `.myclaw/` directory.

The current Agent Home layout is intentionally small:

```text
~/.myclaw/
  config.toml
```

Workspace State initialization and its on-demand Session materialization are
defined by ADR-0005 and ADR-0009. Existing legacy files are preserved without
being read, migrated, or deleted.

---
status: accepted
---
