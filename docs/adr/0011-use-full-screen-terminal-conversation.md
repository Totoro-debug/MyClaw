---
status: accepted
---

# Use a Full-Screen Terminal Conversation

Running `myclaw` without arguments starts one full-screen terminal UI with a scrollable conversation display and a bottom input area. There is no plain REPL product mode or non-TTY fallback; non-interactive Management Commands remain separate, and the internal headless REPL seam exists only for runtime regression tests.
