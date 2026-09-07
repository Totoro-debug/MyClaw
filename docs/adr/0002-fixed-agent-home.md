---
status: accepted
---

# Fix Agent Home at `~/.myclaw/`

MyClaw uses one fixed Agent Home for the current operating-system account and does not support profiles or configurable data roots. Agent Home owns global User Configuration at `~/.myclaw/config.toml` and user-authored Skills under `~/.myclaw/skills`. All active non-global runtime state belongs to the current Workspace; legacy Runtime Log files under Agent Home remain untouched.

Skill loading and the narrowly scoped read permission are defined by [ADR-0016](0016-use-agent-home-skill-catalog-and-progressive-loading.md). Agent Home as a whole is not an exemption from external-path Tool Confirmation.
