---
status: accepted
---

# Persist Runtime Reasoning Effort Best-Effort After Memory Commit

The shared `ModelRouter` remains the Runtime Lifetime authority for the current Reasoning Effort. A successful
`/effort` update publishes that in-memory value before it performs any User Configuration I/O. The Management
Port owns this ordering, so a configuration failure cannot prevent the next logical model request from using the
committed value or cancel an active Agent Run.

`ConfigLoader.update_reasoning_effort()` is a narrow domain operation. It rereads the latest `config.toml`, uses a
`tomlkit` round-trip to preserve comments, table order, credentials, and unrelated settings, updates the
`default` route, and updates `chat` only when that table is explicitly present. It validates the complete candidate
configuration before making exactly one same-directory atomic replacement. It never materializes a missing `chat`
route and does not use the startup `UserConfiguration` snapshot as a write source.

Persistence is deliberately best effort. Parse, validation, and replacement failures leave the published runtime
value intact; Management records one safe diagnostic containing only the stable operation and exception type, then
returns the normal successful selection result. It does not log configuration contents, credentials, or a traceback.
Temporary divergence between Runtime Lifetime status and the on-disk User Configuration is accepted until a later
successful update or process restart.

This decision does not add configuration locks, a general mutation framework, rollback transactions, or a global
mutable User Configuration aggregate. Provider retry/fallback behavior, Session state, and Agent Loop ownership are
unchanged.
