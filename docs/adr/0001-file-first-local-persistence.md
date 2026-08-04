# Use file-first local persistence

The first version of the Personal Agent stores User Configuration, Workspace
State, Conversation Summary, Long-term Memory, Scheduled Work definitions, Tool
Artifacts, and Session Logs as local files rather than starting with SQLite or a
mixed storage model. This keeps the local-first system transparent and
inspectable. The implementation relies on atomic writes and in-runtime
ordering where required, but intentionally provides no cross-process
coordination for Workspace-owned state.

ADR-0009 partially supersedes this decision for the foreground Conversation
Session lifecycle. The current Session authority is one active in-memory
`Session`, and its history is written as a complete atomic JSONL snapshot after
each completed turn. The former per-message write, incomplete-line repair,
typed persistence-object, and cross-file consolidation-journal contracts are
retired; they are not current file-first guarantees.

File-first remains the storage choice for Workspace State and for the separate
Conversation Summary stream. Summary state and `Session.last_consolidated` are
not committed as one filesystem transaction. Their possible divergence after a
crash or failed Session snapshot is accepted by ADR-0009. Separate REPL
processes remain uncoordinated.

---
status: accepted
---
