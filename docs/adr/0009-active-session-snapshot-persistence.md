---
status: accepted
---

# Use an Active In-Memory Session with Ordered Snapshots

One active `Session` is the foreground Conversation Session authority. It owns JSON-native messages, metadata, `last_consolidated`, identity, timestamps, and the strict compact JSONL snapshot boundary; it does not own model calls, Tool execution, foreground presentation, or runtime lifecycle.

`append_messages()` validates and deep-copies a complete Agent Run increment and applies its usage delta atomically in memory. New empty Sessions remain memory-only. A persisted Session contains one exact header followed by user, assistant, and Tool message dictionaries; schema-versioned or malformed histories are rejected without migration, repair, or version dispatch.

`persist()` captures a complete deep-copied snapshot and schedules ordered atomic replacement with at most three asynchronous attempts and 100 ms then 200 ms backoff. Exhausted ordinary failures are silent and do not change the Agent Run outcome. `close()` performs the same bounded three-attempt final save synchronously, while `abandon()` cancels pending snapshots, rejects later mutation, and performs no final save.

Conversation Summary updates `last_consolidated` without a cross-file transaction, so summary and Session state may diverge after a crash. Separate processes are not coordinated, and forced Runtime Generation replacement may discard state that was not already persisted.
