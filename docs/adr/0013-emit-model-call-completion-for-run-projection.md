---
status: accepted
---

# Emit model-call completion for Agent Run projection

Agent Run emits an explicit nonterminal model-call completion payload containing the call's complete text and whether execution continues with Tools, before any corresponding Tool-start payloads. This lets foreground adapters distinguish intermediate model output from the final candidate without guessing from Tool events, including when a Provider completes with text but emitted no deltas. The signal is lane-neutral, Schedule consumers may ignore it, the Conversation Port maps it for the Terminal Conversation, and Session persistence remains unchanged because historical groups are inferred from user-message boundaries.
