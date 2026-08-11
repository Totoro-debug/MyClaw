---
status: accepted
---

# Use Textual and Capability-Gated Enhanced Keyboard Input

Terminal Conversation uses `textual>=8.2.8,<9` with `rich>=14.2.0,<15` for its
full-screen application lifecycle, layout, scrolling, Markdown, modal screens, mouse
input, and headless application tests. Production prompt-toolkit adapters and the
prompt-toolkit direct dependency are retired.

A narrow internal enhanced-keyboard adapter owns conservative capability detection,
Kitty keyboard-mode enablement, Enter-family report parsing, and restoration. It treats
Shift+Enter and Alt+Enter as newline only when the host delivers a distinct supported
report. It never infers a modifier from ordinary carriage return. Ctrl+J always inserts
a newline and remains the portable, zero-configuration fallback.

Windows Terminal may reserve Alt+Enter, and terminal capability detection does not
prove that a particular host configuration delivers either modified Enter chord.
Those results require a dated real-terminal acceptance record with the terminal version
and relevant settings. Until such evidence exists, Windows Terminal modifier behavior
is pending rather than validated.

Textual owns application mode, alternate screen, cursor, bracketed paste, mouse, and
focus reporting. Terminal Conversation tracks modes written by the driver and restores
all still-active modes before prepared Runtime shutdown, including startup and cleanup
failures. The enhanced-keyboard adapter likewise restores a mode it enabled. Headless
driver tests cover fault paths; an installed-wheel pseudo-terminal smoke verifies each
observed enable sequence has a later reset where the host supplies a PTY harness.

This decision does not expand formal platform support. Windows x64 remains the
currently validated environment for the repository as a whole; other host adapters are
attempted according to ADR-0007 without a support promise.
