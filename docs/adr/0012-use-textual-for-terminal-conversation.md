---
status: accepted
---

# Use Textual for the Terminal Conversation

MyClaw uses Textual `>=8.2.8,<9` and Rich `>=14.2.0,<15` for the full-screen application lifecycle, Markdown, scrolling, modal interaction, mouse input, and headless UI tests. A narrow capability-gated keyboard adapter handles modifier-aware Enter reports and restoration; unsupported terminals use `Ctrl+J` as the reliable newline fallback.
