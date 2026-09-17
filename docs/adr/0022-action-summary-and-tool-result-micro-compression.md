---
status: accepted
---

# Persist Action Summary and Omit Repeated Tool Results from Model Context

Conversation compaction continues to persist its existing fact-oriented Conversation Summary in the Workspace summary stream, and also generates an independent Action Summary with the same message range and `memory` route. The Action Summary is stored under the active Session metadata key `summary`, projected by `ContextBuilder` as one unwrapped `user` message after the System Prompt and before retained history, and replaced only after a later successful compaction; it is not appended to `summary.jsonl` or `Session.messages`. If either Summary model call fails, compaction fails and does not advance `last_compacted`; an already-appended fact Summary may remain because the existing persistence boundary is non-transactional.

To bound repeated context cost without changing Agent-visible Tool history, an Agent Run enables Tool Result Micro-compression only after more than ten eligible Tool Calls in that run. On the next ReAct request, the provider-only message projection replaces content longer than 512 characters in eligible Tool messages from earlier ReAct cycles, including prior Agent Runs, with `[tool_name result omitted from context]`, substituting the actual Tool name. The immediately completed cycle remains unmodified, and `AgentRunnerResult.messages`, AgentLoop increments, Session persistence, Tool Artifacts, and Dream requests retain the original Tool content. Eligible built-ins are `exec`, `glob`, `grep`, `list_dir`, `read_file`, `web_fetch`, and `web_search`; every `MCPTool` is eligible.

This separates durable task-oriented context from the global Memory System and keeps omission reversible at the provider boundary, while preserving complete operational history for AgentLoop consumers and Session recovery.
