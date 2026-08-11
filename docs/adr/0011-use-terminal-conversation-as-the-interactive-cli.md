---
status: accepted
---

# Use Terminal Conversation as the Interactive CLI

Running `myclaw` without arguments starts the full-screen Terminal Conversation.
There is no user-selectable plain REPL and no REPL fallback for redirected streams or
unsupported terminals. Non-interactive Management Commands, currently `myclaw config`,
remain outside the full-screen host and retain their process outcomes.

The CLI loads User Configuration before checking stdin, stdout, and stderr for TTY
support. A valid non-TTY launch returns the stable `interactive_terminal_required`
outcome before preparing a Runtime or creating Workspace State. A valid TTY launch
prepares exactly one Runtime and gives it to `run_terminal_conversation`, whose
application lifecycle owns Runtime start and close exactly once.

The injectable `run_repl` loop remains an internal headless seam for Runtime,
Conversation Port, Session resume, Tool Confirmation, cancellation, and shutdown
regression coverage. It is not a product adapter, console entry point, or production
fallback. Production console-input and progressive-writer adapters are removed.

Terminal Conversation continues to use the prepared Runtime, Conversation Port,
Management dispatcher, and active Conversation Session authority. This decision does
not add a UI-specific Runtime port or change Session persistence, Model Provider, Tool,
Memory, Schedule, User Configuration, or Workspace State contracts.

Installed-process tests cover configuration and non-TTY outcomes. The Textual
application seam covers interactive behavior and Runtime lifecycle, while an installed
wheel pseudo-terminal smoke must observe full-screen startup, exit through a product
key path, and terminal restoration when a PTY harness is available.
