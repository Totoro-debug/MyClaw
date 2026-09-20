---
status: accepted
---

# Ignore Unknown User Configuration Fields

User Configuration loading and `/config` interpret the document as a projection of fields understood by the running MyClaw version. Unknown fields and static tables, including removed settings such as `compaction_message_threshold`, are ignored so configuration can survive upgrades, downgrades, and stale keys; the raw file is not rewritten merely because such a field is present. Known fields still apply their declared validation; a missing required field uses its declared default or fails when no default exists. Malformed TOML still fails, every declared Provider remains eagerly validated, and an invalid MCP Server retains its existing isolated diagnostic behavior.

Default-value fallback is one explicit contract for every defaultable leaf: `runtime.max_tool_result_chars` (`4096`), `runtime.max_iterations` (`50`), `runtime.enable_skill_always_load` (`false`), `runtime.compact_ratio` (`0.9`), `runtime.permission_level` (`workspace-write`), `runtime.exec_shell` (`auto`), `memory.batch_size` (`10`), `memory.schedule` (`0 * * * *`), and each `models.routes.*.reasoning_effort` (`medium`). A missing value silently uses its default. An explicitly invalid value uses its default and produces exactly one sanitized diagnostic containing only the field and effective default value; the invalid source value is never copied into that diagnostic. This supersedes the earlier compact-ratio-only fallback wording. `/config` exposes the effective defaultable runtime values before the redacted raw content, and startup reports the same diagnostics. No environment-variable overlay or Route-driven lazy configuration parsing is introduced. This trades typo detection for forward and backward compatibility; effective configuration and status views expose the value actually in use.

Requirements: [Multi-level Tool permissions](https://github.com/Totoro-debug/MyClaw/issues/245), Phase 1 tracked by [T03](https://github.com/Totoro-debug/MyClaw/issues/247).
