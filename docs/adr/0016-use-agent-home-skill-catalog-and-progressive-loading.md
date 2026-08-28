---
status: accepted
---

# Use an Agent Home Skill Catalog with Progressive Loading

> Runtime-Lifetime snapshot ownership and on-demand manual Skill body loading in this ADR are superseded by [ADR-0017](0017-use-cli-composition-root-and-session-scoped-agent-loop.md). Skill discovery rules, path permissions, and prompt-safety constraints remain accepted. This historical decision and its consequences are immutable; the superseded scope is not an active Skill contract.

MyClaw discovers Skills from the direct child directories of `~/.myclaw/skills`, keeps only valid YAML-frontmatter metadata in a Runtime Lifetime catalog, and exposes every retained Skill's name, description, and absolute `SKILL.md` path to foreground model calls and the existing command-completion surface. Skill names are validated without trimming, descriptions are trimmed before validation, canonical path order makes the first valid duplicate win, and Management Command names remain reserved. Skills guide existing capabilities without registering Tools or expanding their authority.

By default, Skill documents are loaded only after selection. A user slash invocation makes the host read and revalidate the complete UTF-8 `SKILL.md` once, then project that complete document, including its frontmatter delimiters, frontmatter, body, and original line endings, together with the extracted request as Runtime-owned content in that foreground Agent Run's current `user` message. The Conversation Session persists only the original slash input. An autonomous model selection uses ordinary `read_file` calls and therefore retains normal pagination, Tool Artifact, persistence, and retry-by-further-call behavior. MyClaw does not cache these reads or verify that the model reached end of file. When `[runtime].enable_skill_always_load` is true, a Skill whose optional YAML `always` value is the boolean `true` is read at Runtime Lifetime startup and its complete document is included in every foreground System Prompt; otherwise `always` is ignored and only metadata is included. Always-loaded content is frozen until restart, has no fixed file-size limit, and fails startup if the minimum real foreground request—including the Runtime Context wrapper and fixed Tool schemas—cannot fit the chat input budget.

`read_file` treats canonical paths beneath `~/.myclaw/skills` as confirmation-free in every lane, while paths that escape through links or otherwise resolve outside that subtree keep the existing Workspace-external confirmation behavior. Other Tool permissions, writes, Exec working directories, Schedule behavior, and the fixed ten-Tool Catalog are unchanged.

## Consequences

Agent Home now owns user-authored Skills in addition to User Configuration. Automatic Skill reads may persist complete or partial Skill content in Conversation Session Tool messages, whereas host-loaded manual content exists only in the current model projection and always-loaded content exists only in the foreground System Prompt. Absolute Agent Home Skill paths are disclosed to the configured Model Provider, and enabling always-load deliberately permits user-authored Skill instructions to extend the foreground System Prompt.
