---
status: accepted
---

# Use an Agent Home Skill Catalog with Progressive Loading

MyClaw discovers Skills from the direct child directories of `~/.myclaw/skills`, keeps only valid YAML-frontmatter metadata in a Runtime Lifetime catalog, and exposes every retained Skill's name, description, and absolute `SKILL.md` path to foreground model calls and the existing command-completion surface. Skills guide existing capabilities without registering Tools or expanding their authority; invalid entries are skipped with safe diagnostics, canonical path order makes the first valid duplicate win, and Management Command names remain reserved.

By default, Skill bodies are loaded only after selection. A user slash invocation makes the host read and revalidate the complete `SKILL.md` once, then project only the instruction body after its frontmatter, together with the extracted request, as Runtime-owned content in that foreground Agent Run's current `user` message. The raw frontmatter, absolute path, and complete raw document are not projected, and the Conversation Session persists only the original slash input. An autonomous model selection uses ordinary `read_file` calls and therefore retains normal pagination, Tool Artifact, persistence, and retry-by-further-call behavior. MyClaw does not cache these reads or verify that the model reached end of file. When `[runtime].enable_skill_always_load` is true, a Skill whose optional YAML `always` value is the boolean `true` is read at Runtime Lifetime startup and its complete body is included in every foreground System Prompt; otherwise `always` is ignored and only metadata is included. Always-loaded content is frozen until restart, has no fixed file-size limit, and fails startup if it cannot be loaded or exceeds the chat input budget.

`read_file` treats canonical paths beneath `~/.myclaw/skills` as confirmation-free in every lane, while paths that escape through links or otherwise resolve outside that subtree keep the existing Workspace-external confirmation behavior. Other Tool permissions, writes, Exec working directories, Schedule behavior, and the fixed ten-Tool Catalog are unchanged.

## Consequences

Agent Home now owns user-authored Skills in addition to User Configuration. Automatic Skill reads may persist complete or partial Skill content in Conversation Session Tool messages, whereas host-loaded manual content exists only in the current model projection and always-loaded content exists only in the foreground System Prompt. Absolute Agent Home Skill paths are disclosed to the configured Model Provider, and enabling always-load deliberately permits user-authored Skill instructions to extend the foreground System Prompt.
