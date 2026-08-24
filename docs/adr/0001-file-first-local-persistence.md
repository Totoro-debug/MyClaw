---
status: accepted
---

# Use File-First Local Persistence

MyClaw stores User Configuration and Workspace-owned runtime state as inspectable local files instead of using a database or mixed storage model. Each store defines its own publication guarantees: Conversation Sessions and other declared stores use atomic replacement where specified, Tool Artifacts use direct writes, and no Workspace-owned state has cross-process coordination.

The active in-memory `Session` is authoritative during an Agent Run, while Conversation Summary and Session snapshots remain independent files whose state may diverge after a crash. Existing legacy Agent Home Runtime Log files are preserved without being read, migrated, or updated.
