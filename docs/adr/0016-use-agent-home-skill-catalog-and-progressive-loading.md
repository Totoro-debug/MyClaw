---
status: accepted
---

# Use an Agent Home Skill Catalog with Progressive Loading

Each Agent Loop owns a Skill Loader that scans direct child directories of `~/.myclaw/skills`, validates UTF-8 `SKILL.md` documents, and freezes their full contents and metadata. Skill names match `[a-z_-][a-z0-9_-]{0,63}` without trimming; descriptions are trimmed and must contain 1-1024 characters. Canonical path order makes the first valid duplicate win, and Management Command names remain reserved. Invalid documents are omitted. Skills guide existing capabilities without registering Tools or expanding their authority.

Initial startup, each `/resume`, and `/reload_skill` construct a candidate Skill state. Publication is atomic and follows complete Model Request Context budget preflight, including the Tool Catalog. Failed reloads retain the previous model catalog, manual invocation content, and terminal completion metadata. An active Agent Run retains its captured projection; later requests use the new state. Initial startup or `/resume` preflight failure is fatal to the Terminal Conversation. Context assembly follows [ADR-0018](0018-centralize-model-request-context-construction.md).

A Manual Skill Invocation is an exact leading slash name, optionally followed by whitespace and a request. The host projects the frozen complete document, including frontmatter and original line endings, with the extracted request into the current foreground `user` message. It does not reread the file. The Conversation Session persists only the original slash input. Unknown names, partial names, and case mismatches remain ordinary input.

The foreground System Prompt exposes the Skill Catalog's names, descriptions, and absolute document paths. Autonomous model selection uses ordinary `read_file` calls against the live filesystem, retaining pagination, Tool Artifact, persistence, and retry-by-further-call behavior. The host does not verify that these reads reach end of file. Frozen host content and autonomous filesystem reads may therefore differ until the next successful reload.

When `[runtime].enable_skill_always_load` is true, documents whose YAML `always` value is the boolean `true` also appear in every newly constructed foreground System Prompt. Otherwise only metadata appears. There is no fixed Skill file-size limit, but the complete request must fit the chat input budget. Schedule and Dream do not receive manual or always-loaded Skill projections.

`read_file` treats canonical paths beneath `~/.myclaw/skills` as confirmation-free in every lane. Paths escaping that subtree through links retain the ordinary Workspace-external confirmation behavior. The exemption grants no permission to write Skills, read the rest of Agent Home, bypass Exec/Web checks, or expand the Dream Tool Catalog. Configured MCP authority is independent and follows [ADR-0020](0020-expose-configured-mcp-tools-through-tool-gateway.md).

Absolute Skill paths and selected contents are disclosed to the configured Model Provider. Autonomous reads may persist full or partial Skill content in Session Tool messages; host-loaded manual content exists in the current model projection, and always-loaded content extends the foreground System Prompt.

Requirements: [Skills](https://github.com/Totoro-debug/myclaw/issues/180), [context construction and reload](https://github.com/Totoro-debug/myclaw/issues/203).
