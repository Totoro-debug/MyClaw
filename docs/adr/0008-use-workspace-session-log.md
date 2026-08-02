---
status: accepted
---

# Use a Workspace-owned Session Log

Session technical diagnostics belong to the owning Workspace and Conversation Session.
An explicit Session context validates the existing Session ID rule, prepares
`.myclaw/logs` lazily, and registers a Loguru file sink at `<session_id>.log`. The
sink uses WARNING threshold, enqueue, UTF-8, catch isolation, 10,485,760-byte
rotation, and retention of one historical file. Sink registration failures are
fail-open and retried on the next context; context exit removes the sink and lets
Loguru drain its queue without a custom deadline. Same-Session concurrency is
deliberately unsupported and is not coordinated by a registry or lock. This decision
supersedes ADR-0004; legacy Agent Home Runtime Log files are preserved but no longer
updated by the production entry point.
