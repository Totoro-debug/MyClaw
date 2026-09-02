---
status: accepted
---

# Centralize Agent Loop Model Request Context Construction

> This decision supersedes ADR-0016's requirement that always-loaded Skill content remain frozen until restart and ADR-0017's requirement that one immutable full Skill Snapshot last for the entire Agent Loop generation. Skill discovery, validation, prompt-safety, path permissions, and Agent Loop ownership remain accepted.

`ContextBuilder` is the sole construction boundary for System Prompts and initial Model Request Contexts used by `AgentLoop`, including foreground chat, Schedule Jobs, Session title generation, preflight, and foreground summary-budget projections. It owns the Workspace, Agent Home, timezone, `MemoryManager`, and `SkillLoader`; it reads the current Long-term Memory snapshot when constructing a System Prompt and reads the Skill Loader's current frozen Skill state. It exposes both focused prompt methods and complete message-building methods, does not accept Tool schemas, and uses `datetime.now(self._timezone)` when constructing every current User Message, including Schedule requests.

`SkillLoader` owns the frozen Skill state directly; the separate `SkillSnapshot` type is removed. Each successful `load()` atomically replaces that state, while failure preserves the previous state. The `/reload_skill` Management Command calls the current Agent Loop's loader directly: an active Agent Run keeps its already-built messages, and later requests use the reloaded state without rebuilding the Agent Loop or clearing its Conversation Session. A successful reload returns the new Skill metadata to the Terminal so its completion cache changes with the model-visible catalog and Manual Skill resolution; a failed reload leaves all three views unchanged.

`Blackboard` owns Task Framing generation, prompt construction, model invocation, parsing, and reduction through an asynchronous class method; the separate `TaskFramer` and `TaskFramingEvaluator` abstractions are removed. Conversation Summary and Dream retain locally owned System Prompt and request assembly because they are separate execution responsibilities. `AgentRunner` remains independent from `ContextBuilder`, consumes complete initial messages, and continues to own ReAct transcript increments and repair messages. Tool schemas remain owned by Tool Gateway and request/status call sites rather than Context Builder.

Model-visible structural wrappers use Markdown. Substantive versioned System Prompt content remains in the `myclaw.templates` package and is not asserted verbatim by tests; single-use assembly-only templates are inlined at their construction sites. Tests verify message roles, ordering, dynamic projection, isolation, escaping, Markdown structure, and execution behavior instead of static prompt wording.
