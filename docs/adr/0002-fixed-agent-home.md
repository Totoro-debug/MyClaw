---
status: accepted
---

# Fix Agent Home at `~/.myclaw/`

MyClaw uses one fixed Agent Home for the current operating-system account and does not support profiles or configurable data roots. Agent Home owns only global User Configuration at `~/.myclaw/config.toml`; all active non-global state belongs to the current Workspace, while legacy Runtime Log files under Agent Home remain untouched.

ADR-0016 extends Agent Home ownership with user-authored Skills under `~/.myclaw/skills` while leaving Workspace-owned runtime state unchanged.
