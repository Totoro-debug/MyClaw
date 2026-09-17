---
status: accepted
---

# Use One-shot Dream Model Requests

Dream is a single-completion Memory responsibility rather than a ReAct Agent Run. After claiming Conversation Summary entries and pre-advancing the Summary Cursor, it reads the complete Long-term Memory into one logical `memory` Model Router request that exposes only `edit_file`, then applies every returned edit sequentially without a second model confirmation. Router-level automatic retries remain available, while Dream adds no retry, context compaction, or cursor rollback.

A response with no Tool call is a successful unchanged result; all returned edits succeeding is a successful update. Unknown Tools, edit failures, and `length` or `cancelled` model finishes fail the Dream run, while edits completed before a later failure remain durable and are reported through `memory_updated`. A fixed request at or above the resolved Memory Model Route's Available Context fails instead of entering Agent Run compaction, and every automatic fallback attempt repeats that hard check for its actual route.

This keeps Agent Runner as the sole bounded ReAct engine while avoiding a second model request whose only purpose was to confirm edits the first response had already specified.

Requirements: [Unified Agent Run token budgeting and context compaction](https://github.com/Totoro-debug/MyClaw/issues/234).
