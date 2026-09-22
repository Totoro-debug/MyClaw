---
status: accepted
---

# Use a Runtime Lifetime Tool Confirmation Coordinator

The CLI composition root owns exactly one `ToolConfirmationCoordinator` for the Runtime Lifetime. Every Agent Loop generation submits its foreground confirmations through that coordinator, and the Terminal Conversation is the one presenter. The coordinator survives successful Session replacement and is closed before Runtime shutdown proceeds.

Each request carries an immutable Runtime-only `ConfirmationEnvelope` containing the exact normalized `ConfirmationRequest`, a typed foreground or background origin, and an immutable owner. Foreground owners identify the generation and Agent Run; background owners identify the generation, `job_id`, and occurrence. The envelope is a presentation boundary: it has no persistence serializer and is never stored in Session, Schedule, or public JSON. The exact normalized request remains the Gateway's existing Tool Confirmation metadata.

The coordinator has one active item globally. Foreground and background requests each use FIFO queues. Foreground is selected before background whenever the active slot becomes free, but an active item is never preempted. User decisions are accepted only through the opaque active token; duplicate, late, and unknown tokens are ignored. There is no timeout while the Runtime Lifetime remains alive. Producer cancellation removes its queued item or dismisses its active modal without turning lifecycle cancellation into a user decline.

Owner and generation cancellation complete pending producers with typed `ConfirmationAborted`. Closing or unbinding the current presenter aborts active and queued items and dismisses the active modal; binding a different presenter is rejected while one is current. Rebinding the Terminal keeps the coordinator and presenter alive, while replacement cancels the old generation before the old UI is quiesced. A missing presenter raises typed `ConfirmationUnavailable`; the Gateway maps that condition to its existing fail-closed confirmation-availability Tool Error and never executes the Tool.

Synchronous and asynchronous presenter invocation failures complete the active producer with typed `ConfirmationUnavailable` and let the coordinator advance to the next queued request. Coordinator-owned lifecycle cancellation stops and drains the presentation pump while preserving producer cancellation and `ConfirmationAborted` semantics.

The Textual adapter projects foreground and background envelopes into the same modal. Foreground uses `Tool Confirmation`; background uses `Background Tool Confirmation` and shows `Source: job_id + title`. The modal defaults focus to Decline, has no timeout, and treats lifecycle dismissal as no decision. The adapter captures the mounted conversation display and input widgets once and reuses those references for streamed output and modal restoration, avoiding stale widget queries across replacement.

User Schedule occurrences use this same coordinator when their admission
snapshot requires confirmation. Schedule Service binds a
`BackgroundConfirmationOwner(generation_id, job_id, occurrence_id)` to the
active occurrence; the Agent Loop wraps each exact normalized call in a
background envelope carrying only the canonical `job_id` and title for the
modal source. Replacement and shutdown cancel old-generation owners and drain
their typed `ConfirmationAborted` failures while the Schedule Store is still
writable. The typed outcome passes through the Gateway, Runner, and Agent Loop
to Schedule Service; a failed terminal Store write fails that drain. The old
generation cannot admit a later confirmation between drain and Schedule pause;
the pause barrier retains any late gate-aborted occurrence until terminal
persistence finishes before it broadly cancels ordinary Schedule work.
A recurring Job retains an error terminal state, while a one-shot Job follows
its existing terminal removal path. A successful Job deletion first persists
absence, then cancels and drains its exact active occurrence and confirmation
owner. No owner or envelope is persisted, and Dream/System Schedule work
remains outside this confirmation path.
