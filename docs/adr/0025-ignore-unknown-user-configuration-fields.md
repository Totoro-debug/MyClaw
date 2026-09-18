---
status: accepted
---

# Ignore Unknown User Configuration Fields

User Configuration loading and `/config` interpret the document as a projection of fields understood by the running MyClaw version. Unknown fields and static tables, including removed settings such as `compaction_message_threshold`, are ignored so configuration can survive upgrades, downgrades, and stale keys; the raw file is not rewritten merely because such a field is present. Known fields still apply their declared validation; a missing required field uses its declared default or fails when no default exists. Malformed TOML still fails, every declared Provider remains eagerly validated, and an invalid MCP Server retains its existing isolated diagnostic behavior.

Field-specific fallback is explicit rather than implicit. In particular, missing or invalid `[runtime].compact_ratio` uses `0.9` with exactly one safe warning, while other known invalid fields continue to fail unless their own contract declares a fallback. `/config` exposes the effective ratio before the redacted raw content, and status exposes the same effective value. No environment-variable overlay or Route-driven lazy configuration parsing is introduced. This trades typo detection for forward and backward compatibility; effective configuration and status views expose the value actually in use.

Requirements: [Unified Agent Run token budgeting and context compaction](https://github.com/Totoro-debug/MyClaw/issues/234).
