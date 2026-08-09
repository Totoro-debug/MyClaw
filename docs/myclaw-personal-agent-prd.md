# Build MyClaw local-first Personal Agent runtime

## Problem Statement

用户需要一个参考 nanobot、但边界更加清晰的通用 Personal Agent。它应当是 local-first、single-user 的本地运行时，以命令行 REPL 为主要入口，支持持续前台运行、三层记忆、工具调用、Workspace 级会话隔离、模型路由、用户配置和自然语言定时任务。

首版必须避免演变为多租户 Agent 平台、微服务系统、插件平台或后台 daemon。上层只依赖明确的代码接口边界，不需要了解底层模型 SDK、文件格式或工具实现。

## Solution

实现一个 Python 编写的 MyClaw Personal Agent runtime：

- 用户运行 `myclaw` 进入长驻前台 REPL。
- 每个有效 REPL 启动准备一个新的 Workspace-scoped Conversation Session；未发送用户消息就退出时不持久化空 session。
- Runtime Core 通过 Conversation Port、Management Port、Model Route、Tool Gateway 和 memory/session 存储边界编排 Agent Run。
- Agent Home 固定为 `~/.myclaw/`，采用 file-first persistence。
- REPL 支持 streaming、后台 Memory Task、Schedule Jobs、session resume 和管理 slash commands；Schedule actions 不请求 Tool Confirmation，Exec、Web 和 Workspace 外部路径按具体目标执行一次性确认。
- 首版非交互管理只支持 `myclaw config`，不支持 one-shot 对话。

## User Stories

1. 作为个人用户，我想在本地运行 MyClaw，所以我的个人 Agent 不依赖多租户平台。
2. 作为个人用户，我想运行 `myclaw` 直接进入 REPL，所以命令行对话是自然的默认入口。
3. 作为个人用户，我想在一个 REPL 中连续多轮对话，所以当前线程可以保留 Short-term Memory。
4. 作为个人用户，我想每次有效 REPL 启动默认准备新 session，所以不同对话不会自动混在一起。
5. 作为个人用户，我不想空 REPL 留下 session 文件，所以未发送消息时退出不会产生垃圾会话。
6. 作为个人用户，我想 session 按 Workspace 分组，所以不同项目的对话彼此隔离。
7. 作为个人用户，我想通过 `/resume` 的交互式 picker 恢复当前 Workspace 的历史 session，所以可以继续旧对话。
8. 作为个人用户，我想在 picker 中看到 session 标题和时间，所以可以识别需要恢复的对话。
9. 作为个人用户，我想 Session title 自动生成，所以无需手动命名每个会话。
10. 作为个人用户，我想标题生成失败时仍有可读标题，所以 session 创建不会因辅助模型调用失败。
11. 作为个人用户，我想主对话始终 streaming，所以能及时看到模型输出。
12. 作为个人用户，我想 Ctrl+C 只取消当前 turn，所以后台任务不会被误取消。
13. 作为个人用户，我想输入 `exit` 或 `quit` 退出 REPL，所以退出行为明确。
14. 作为个人用户，我想 `/config` 查看当前配置，所以可以确认 runtime 和模型设置。
15. 作为个人用户，我想 API key 在配置输出中默认脱敏，所以终端内容不会轻易泄露密钥。
16. 作为个人用户，我想 `/status` 查看版本、chat model、运行时间、token 和 session 状态，所以可以理解当前 runtime 状况。
17. 作为个人用户，我想 `/memory` 完整查看最新磁盘 Long-term Memory，所以可以知道 Agent 保存了什么。
18. 作为个人用户，我想 `/dream` 手动处理待总结记忆，所以可以主动刷新 Long-term Memory。
19. 作为个人用户，我想没有待处理摘要时 `/dream` 不调用模型，所以不会产生无意义成本。
20. 作为个人用户，我想 Long-term Memory 自动维护，所以 Agent 能逐渐积累稳定信息。
21. 作为个人用户，我想长期记忆分为 User Info、User Preference、Project Fact、Lesson，所以内容有清晰结构。
22. 作为个人用户，我想 Conversation Summary 自动压缩早期消息，所以长对话能继续进行。
23. 作为个人用户，我想原始 session 消息仍保留，所以摘要不会破坏历史可追溯性。
24. 作为个人用户，我想长期记忆只在模型判断需要时更新，所以不会每次后台任务都产生无意义修改。
25. 作为个人用户，我想文件读取、列举和搜索默认可用，所以 Agent 能理解 Workspace。
26. 作为个人用户，我想 Workspace 内文件新建和编辑默认可用、外部路径精确确认，所以项目修改边界清晰可见。
27. 作为个人用户，我想 Exec 对 destructive 命令和非公网 URL 目标请求精确确认，所以命令执行有清晰安全边界。
28. 作为个人用户，我想 WebSearch 和 WebFetch 默认可用，所以 Agent 能访问公网资料。
29. 作为个人用户，我想 WebFetch 对本地、私有网络和 DNS 失败目标请求精确确认，所以这些目标不会被默认访问。
30. 作为个人用户，我想大型工具结果存为 Tool Artifact，所以 session 和上下文不会被大结果撑爆。
31. 作为个人用户，我想 Tool Artifact 随 session 保留，所以恢复旧 session 时仍能读取完整结果。
32. 作为个人用户，我想通过自然语言创建 at、every 或 cron Schedule Job，所以无需手写状态文件。
33. 作为个人用户，我想 Schedule Job 使用专属 Session，所以定时任务不污染当前会话。
34. 作为个人用户，我想 Schedule Job 运行结果留在其 Session 中，所以后台结果不会打断前台 streaming。
35. 作为个人用户，我想 Schedule Job 与前台使用同一个 Agent Run，所以模型、Tool、Summary 和 Artifact 语义一致。
36. 作为个人用户，我想分别配置 default、chat、memory、schedule 模型，所以不同任务可使用不同模型。
37. 作为个人用户，我想具体 route 不可用时 fallback 到 default，所以系统有统一兜底。
38. 作为个人用户，我想支持 Anthropic 和 OpenAI-compatible provider，所以可以使用不同模型服务。
39. 作为个人用户，我想首次运行自动生成配置模板，所以能知道需要填写哪些内容。
40. 作为个人用户，我想配置无效时 `myclaw` 明确退出，所以不会进入半可用 REPL。
41. 作为个人用户，我想仍可运行 `myclaw config` 查看坏配置，所以可以定位配置问题。
42. 作为开发者，我想 Runtime Core 只负责编排，所以具体模型、工具和存储实现可以替换。
43. 作为开发者，我想 Conversation Port 输出 typed Agent Events，所以 CLI 只负责交互和渲染。
44. 作为开发者，我想 Management Port 处理管理命令，所以管理操作不会伪装成聊天。
45. 作为开发者，我想 Tool Gateway 统一解析、prepare、拒绝、单次执行和结果封装，所以工具行为保持一致。
46. 作为开发者，我想 provider adapter 使用官方 SDK，所以 streaming、tool calls 和错误语义更可控。
47. 作为开发者，我想用 fake provider 和 fake tool 测试，所以自动化测试不依赖真实 API。
48. 作为开发者，我想首版不支持 MCP 和 subagent，所以能先稳定核心 runtime 边界。

## Implementation Decisions

### Product and runtime

- 产品边界是 local-first、single-user 的 Personal Agent，不是多租户平台。
- 首版实现语言为 Python 3.12+，采用 `pyproject.toml` 和仓库根目录下的 `myclaw/` 包布局，CLI 使用 Typer + Rich，并以 asyncio 作为并发基础。
- 运行 `myclaw` 不带参数直接进入 REPL。
- 首版没有 one-shot 对话命令、detached daemon、HTTP server 或 IPC server。
- 每个 REPL invocation 创建一个独立 runtime。
- 用户消息进入前台队列并串行执行；Memory Task 和 Schedule Jobs 在同一 runtime 内异步运行。
- Ctrl+C 只取消当前前台 turn。
- 输入 `exit` 或 `quit`（忽略前后空白、大小写不敏感）退出 REPL并立即取消后台任务。
- 多个 REPL 可在同一 Workspace 运行，但首版不做跨进程协调；每个 runtime 独立启动后台调度器。

### CLI and management

- 非交互管理首版只支持 `myclaw config`。
- 首版不要求 `myclaw --help` 作为产品能力。
- REPL 内置 slash commands：`/config`、`/status`、`/resume`、`/memory`、`/dream`。
- 只有内置 slash commands 进入 Management Port；其它 `/` 开头文本作为普通用户消息发送给模型。
- `/config` 完整显示配置，但脱敏 plaintext API key。
- 配置语法错误时，`myclaw config` 显示解析错误、配置路径和原文，并对明显 API key 行做文本级脱敏。
- `/status` 显示版本、chat model、runtime uptime、估算 token 状态、当前 Session 消息数、`last_consolidated` 和当前 Session 累计 model usage。
- `/status` 的 provider-neutral token estimate 使用 `ceil(UTF-8 byte length / 4)`，展示估算输入 token、context window 和占比；实际 cumulative usage 不混入估算值。
- `/resume` 只展示当前 Workspace 的 sessions；选择后直接切换。原 session 有消息则保留，无消息可丢弃。
- `/memory` 不分页，完整读取并显示磁盘最新 `memory.md`。
- `/dream` 前台阻塞执行 Memory Task，并显示处理条数、是否更新及 cursor 状态等摘要，不显示完整 diff。
- `/dream` 没有 pending summary 时返回 `No pending summaries`，不调用模型。

### Agent Home and persistence

- Agent Home 固定为 `~/.myclaw/`，不支持覆盖或多个 profile。
- 有效 REPL 启动在当前 Workspace 创建 `.myclaw/`、`memory/`、`sessions/` 和 Long-term Memory template；`myclaw config` 只处理 Agent Home。
- 技术诊断按 Conversation Session 写入 Workspace-owned Session Log；不维护 Agent Home 或 Workspace 级全局 Runtime Log。
- 每种 Workspace State 写入遵守自身明确的持久化契约；只有声明 atomic replacement 的 Store 承诺原子发布，Tool Artifact 直接写入。Conversation Session 在 turn 完成后以完整 JSONL snapshot 一次 replacement。
- User Configuration 位于 `~/.myclaw/config.toml`。
- Long-term Memory 位于 `<workspace>/.myclaw/memory/memory.md`。
- Conversation Summary 位于 `<workspace>/.myclaw/memory/summary.jsonl`。
- Summary Cursor 是纯文本文件 `<workspace>/.myclaw/memory/.cursor`。
- Schedule state 保存在 Workspace State 根目录的 `schedule.json` 严格 JSON 数组文件中；Schedule Session 使用独立的 `schedule-sessions/` 分区。
- Long-term Memory 缺失时，runtime 启动会创建包含 User Info、User Preference、Project Fact、Lesson 四个空分区的模板。
- Conversation Summary、Summary Cursor、Schedule state 和 Schedule Session 文件按需创建；legacy scheduled-work state 原样保留且不读取、不迁移、不删除。

### Workspace and sessions

- Workspace 由规范化绝对路径识别。
- Workspace State 不派生 slug；Session history directory 为 `<workspace>/.myclaw/sessions/`。
- Session ID 使用系统本地时间戳 + UUID4，固定格式为 `YYYYMMDD-HHMMSS-ffffff_<uuid4>`。
- Session 文件为 `<workspace>/.myclaw/sessions/<session_id>.jsonl`。
- JSONL 第一行是严格 header，恰好包含 `session_id`、`created_at`、`updated_at`、`last_consolidated` 和 `metadata`；metadata 当前包含 title 与 cumulative `token_usage`。
- 后续行是 JSON-native OpenAI-style user、assistant、tool message dictionaries，公共字段为 role、content、local-time timestamp；不含通用 message ID、line type marker 或 schema version。
- 每次 Session mutation 的持久化都是完整 compact JSONL atomic replacement；turn 内只改 active Session memory，terminal work 后 `persist()` ordered async scheduling，shutdown 由 `close()` bounded synchronous retry。
- 普通异步 persistence failure 静默吞掉，不提供 acknowledgement、failure logging 或 stronger crash consistency；同一 runtime 内保证 snapshot order，不提供跨进程保护。
- 新 REPL 准备新 session；第一条 user message 之前退出不持久化 session。
- Session title 在首条用户输入后异步使用 chat route 生成，不阻塞首轮回复。
- 标题生成失败时使用截断的首条用户消息作为 fallback。
- 生成标题和 fallback 都折叠空白并截断为最多 60 个 Unicode code points；结果为空时使用 `Untitled session`。
- 标题调用不写入对话历史，但 usage 计入 Session 累计 usage；late title 可等到后续 turn 或 close 才落盘。
- 首版不支持 title rename 或 regenerate。
- assistant streaming 完成后一次性写入 session。
- streaming 被中断时保存 partial assistant，并标记 interrupted/error。
- assistant content 和 tool_calls 可存在于同一 assistant message。
- 模型最终失败写 assistant error message；工具失败或拒绝写 tool error/result message。
- 旧 Session schema unsupported；无 migration、compatibility reader、lazy upgrade 或 version dispatch。缺少完整 snapshot trailing newline 或包含非法 header/message 时 Session 不可恢复且不会被自动删除。

### Memory system

- Memory System 包含 Short-term Memory、Conversation Summary 和 Long-term Memory。
- Short-term Memory 是 active Session 中 `last_consolidated` 之后的消息后缀。
- Conversation Summary 是全局 JSONL 流，每条只包含自增 index、timestamp、content，不保存 source session 或 message range。
- Conversation Summary index 从 1 开始；缺失的 Summary Cursor 文件等价于 0。
- Conversation Summary 不直接进入 chat 上下文，也不会在新增后立即触发 Memory Task。
- 达到 chat route context budget 或配置的总消息数阈值时，在下一次 chat 调用前同步压缩。
- 总消息数阈值默认 40。
- Token 触发时初始选择约半个预算的早期消息；消息数触发时选择约半个阈值的早期消息。
- Cutoff 向后推进到下一个 user message；如果不存在，则回退到最近的前一个 user message。保留后缀必须从 user message 开始。
- 摘要生成使用 memory route，具体 route 不可用时 fallback default；fallback 也失败则当前 chat 请求失败。
- 生成 Conversation Summary 时不注入 Long-term Memory。
- Long-term Memory 是单个 Markdown 文件，完整注入 chat 和 Schedule Job 系统提示词，不做相关性筛选，也不设首版大小上限。
- Long-term Memory 在 runtime 启动时加载并缓存；Memory Task 修改后，新的 Agent Run 使用刷新后的快照，正在运行的 Agent Run 保留启动时快照。
- `/memory` 读取磁盘最新内容，不读取 runtime 缓存。
- Memory Task 使用系统本地时区 cron，默认每小时一次。
- Memory Task 的 `batch_size` 是全局配置，默认 10。
- Memory Manager 读取 Summary Cursor 和 summary batch，然后构造 memory prompt。
- memory 模型通过标准 Tool Gateway 和专用 read/edit Tool 读取 Long-term Memory，并且只能编辑 `<workspace>/.myclaw/memory/memory.md`；它不使用 Conversation Session 或 Tool Artifact。
- Memory Task 没有调用 `edit_file` 时视为 no update 并推进 Summary Cursor。
- 需要编辑且成功时推进 cursor；编辑失败时不推进。
- Memory Task 在单 runtime 内不重入：周期触发遇到运行中任务则跳过，`/dream` 遇到运行中任务则拒绝并提示。
- 周期 Memory Task 成功或失败都不通知 REPL；手动 `/dream` 输出摘要状态。
- 主 Agent 的固定 file Tools 可访问 Workspace 内全部路径，包括 Workspace State；实际文件结果服从操作系统权限。Memory Task 仍只使用它的专用 Long-term Memory Tool。
- Conversation Summary 在 global summary lock 内完成自己的 summary stream operation 后直接更新 active Session 的 `last_consolidated`；没有 pending journal 或启动恢复协议。crash 后 Summary 与 Session snapshot 可 divergence，且不提供跨进程协调。

### Model routing and providers

- Route 名称严格固定为 `default`、`chat`、`memory`、`schedule`；其它 route table 按未定义字段投影掉。
- `chat` 用于主对话和 Session title。
- `memory` 用于 Conversation Summary 和 Memory Task。
- `schedule` 用于 Schedule Jobs。
- 具体 route 缺失或不可用时总是 fallback 到 `default`。
- `default` 不可用时，REPL 启动失败。
- 每个 route 配置 provider_id、model、context_window、max_output、temperature、reasoning_effort、timeout。
- provider adapter 对不支持的 reasoning_effort 静默忽略。
- 每个逻辑 model call 最多执行 5 个 provider attempts，不是首次调用后再重试 5 次；requested route 与 default fallback 共享该 attempt budget。
- 临时 provider 错误在当前 route 的剩余 attempt budget 内指数退避并尊重 retry-after；具体 route 缺失、配置不可用或返回永久 route/provider 不可用错误时使用 default 的剩余 budget，context overflow、invalid request 和 cancellation 不 fallback。
- Model Provider 配置包含 kebab-case provider_id、protocol、required base_url、plaintext api_key 和 model ID 列表。
- 首版 protocol 只支持 `anthropic` 与 `openai-compatible`。
- 未知 protocol 的 provider 被忽略；引用它的 route 视为不可用并执行 fallback。
- Anthropic adapter 使用官方 Anthropic SDK；OpenAI-compatible adapter 使用官方 OpenAI SDK。
- chat route 的每次输出都必须 streaming；schedule 不要求 streaming。

### User Configuration

- TOML 顶层围绕 runtime、models、memory 组织；Tool Catalog 不进入 User Configuration。
- Provider 使用 `[models.providers.<provider_id>]`。
- Route 使用 `[models.routes.<route>]`。
- User Configuration 不控制 Tool enablement；固定十工具 Catalog 始终可用。
- Runtime loading 投影掉未知顶层 table、未知字段和未知 route；`myclaw config` 报告这些未定义字段，未知 protocol provider 仍按既定规则忽略。
- 配置缺失时，只创建一个 ID 为 `openai-local` 的 OpenAI-compatible provider 模板（base URL、API key 和 model list 为空），并为 `default`、`chat`、`memory`、`schedule` 创建显式但不可用的 route 待填写段；四个 route 初始都引用 `openai-local`。随后退出并提示用户替换 Provider、model 和模型限制，或删除不需要定制的具体 route 以回退到 default；旧配置完全缺少 default route 时，错误消息必须指出 `[models.routes.default]`。
- OpenAI-compatible provider 模板的 base_url 为空；所有 provider 的有效配置都要求 base_url。
- `myclaw config` 在配置缺失时创建默认配置并显示脱敏内容。
- 配置无法解析、模型配置不完整或 default route 不可用时，`myclaw` 启动 REPL 直接退出并显示用户可见错误。
- 不支持 Agent profile、session override、per-chat settings 或用户配置 identity/system prompt。

### Tool Gateway and fail-closed security

- 所有 capability 都是具体 `BaseTool`；Runtime Core 在启动时注入稳定依赖并将完整 Tool Catalog 一次性注册到 Tool Gateway。
- `BaseTool.to_schema()` 从直接公开注解、显式 required、默认值和 `ToolParam` 生成 OpenAI Function Calling schema；Model Request 保存缓存的 typed snapshot，Anthropic adapter 在内部转换。
- `ToolGateway.call()` 是唯一公开入口，负责 raw JSON 解析、调用 BaseTool 固定 cast/Schema/参数/安全管线、一次性 Tool Confirmation、执行和扁平 Tool Result 封装；没有 dynamic registration、plugin/MCP、generic retry、per-call execution context 或 approval flag。
- `ModelToolCall.arguments` 保留原始 JSON string；Tool Result 仅含 call ID、name、status、content 和可选 artifact/confirmation metadata，不含 nested error。
- Tool 执行不重试，取消继续向上传播；Tool Gateway 不设置统一 timeout、不持久化结果、不持有 Workspace。
- 没有独立 `Security` 模块；公共路径、DNS、截断和 Artifact 能力由 BaseTool 或共享 helper 提供，capability-specific 规则保留在具体 Tool。
- Tool Gateway 不序列化前台和后台工具调用。
- File read/list/search 默认 allow。
- Workspace 内新建文件和编辑已有文件直接执行，实际结果服从操作系统权限；解析到 Workspace 外的路径请求一次性确认。
- Exec cwd 默认 Workspace，timeout 范围为 1–600 秒；它执行单次 Bash，没有 allowlist、persistent process 或 OS sandbox。
- Exec 对已知 destructive 命令形状及 URL DNS/private-target 风险请求确认。
- WebSearch 无额外首版限制，使用无凭据的内置 adapter，首选 DuckDuckGo；实际后端不进入持久化配置契约。
- WebFetch 对 DNS 失败、localhost、私有网段和其他非公网地址请求一次性确认；没有确认通道或拒绝时 refused。
- WebFetch 最多跟随 5 次重定向，并对每个目标重新执行地址检查。
- Schedule add/list/remove 不请求确认；Scheduled Agent context 拒绝 add，但允许 list/remove。Exec、Web 和 Workspace 外部路径才使用精确绑定的一次性确认。

### Tool Artifacts

- BaseTool 只在成功工具结果严格超过 `max_tool_result_chars = 4096` 时外部化；error 和 refused 保持 inline。
- Artifact 路径为 `<workspace>/.myclaw/artifacts/<session_id>/<id>.txt`；合法 ASCII call ID 直接使用，其他 ID 使用 UUID4。
- Artifact 保存原始工具结果，直接 UTF-8 写入并允许覆盖，不执行 atomic/identity/rollback/cleanup。
- Session tool message 保存 artifact 引用和受 4096 字符限制的 prefix-plus-marker preview；写失败保留成功状态并返回受限失败 marker。
- 合法的 ASCII tool call ID 直接作为 Artifact 文件名；其他 ID 使用 UUID4，逻辑 `tool_call_id` 保持不变。
- Artifact 随 session 保留，首版不自动清理。
- Artifact 没有 commit、rollback、callback 或 ownership lifecycle；写入后若 Session persistence 失败，允许留下 orphan file。

### Schedule

- Schedule Job 是自然语言 Agent 任务，不是 shell cron job。
- Schedule state 是 Workspace-owned 的严格 JSON 数组，字段为 `job_id`、`source`、`message`、`schedule`、`state`、`created_at_ms` 和 `updated_at_ms`。
- Schedule 支持 `at`、`every` 和 `cron`；Cron Job 固化 IANA timezone，at 使用带 UTC offset 的绝对时间。
- User Job 通过 `schedule` Tool 的 add、list、remove action 管理；三种 action 都不请求 Tool Confirmation，Scheduled Agent context 拒绝 add。
- 每个 Job 使用由 `schedule_<job_id>` 派生的 Schedule Session，并使用专属 `schedule-sessions/` 分区；该 Session 不出现在 `/resume`。
- Schedule Service 是唯一的 Job dispatcher，所有触发都通过共享 Agent Run 使用 `schedule` route，结果保留在 Schedule Session。
- 同一 Job 在单 runtime 内不重入；不同 Job 可并发；不做跨进程协调，因此多个 runtime 可能重复触发。
- Schedule 只在 runtime 存活时运行，不发送 Agent Event、完成提示或通知。
- legacy scheduled-work state 原样保留且永不读取、检测、迁移、重命名或删除。
- 损坏的 Schedule state 在 scheduler 和 REPL 启动前以 `schedule_state_error`、修复提示和路径阻止 Runtime 启动。

## Testing Decisions

- 测试只验证外部可观察行为，不绑定内部实现细节。
- 优先测试最高 seam：Conversation Port、Management Port、Runtime Core、Memory Manager、Tool Gateway、active Session、Model Router/Provider Adapter。
- 测试使用 fake provider、fake tool、临时 Agent Home 和临时 Workspace，不调用真实模型 API。

### Required memory tests

- Short-term Memory 是 `last_consolidated` 后缀。
- Token 和总消息数两种摘要触发条件。
- Cutoff 对齐 user message 的主路径和 fallback 路径。
- Summary JSONL schema 只有 index、timestamp、content。
- Summary 生成不注入 Long-term Memory，且不会立即触发 Memory Task。
- memory route/default fallback 全部失败时 chat 请求失败。
- `/dream` 无 pending summaries 时不调用模型。
- Summary Cursor 在 no update、edit success、edit failure 下的推进规则。
- Memory Task batch size、cron、不重入和受限 edit_file 路径。
- Long-term Memory runtime cache 与 `/memory` 磁盘最新视图的差异。

### Required session tests

- Session ID、Workspace-owned 文件路径、strict header 第一行和 JSON-native OpenAI-style messages。
- 空 REPL 不持久化 session。
- `/resume` 只列当前 Workspace sessions，并正确切换。
- 每轮一次完整 JSONL replacement、snapshot freeze 和同 runtime ordered persistence。
- empty Session lazy materialization、silent ordinary failure、bounded close retry 与 host filesystem fault seam。
- title 异步生成、fallback、usage 计入、late title 和后续 snapshot/close 落盘。
- streaming 完成、中断 partial、模型失败和工具失败写入规则。

### Required tool tests

- File 默认行为、Workspace State 访问、外部确认和越界处理。
- Exec Workspace cwd、destructive/DNS confirmation 和 1–600 秒 timeout 校验。
- Fixed Catalog WebSearch、WebFetch 私网确认、重定向复检和 5 次上限。
- Workspace write/edit 的 OS 权限行为；Schedule add/list/remove 不请求确认，Scheduled Agent add 固定 refused。
- Tool Artifact 的 4096 阈值、路径、原始内容、prefix preview 和直接覆盖写入。
- Tool Gateway raw arguments、projection/coercion/refusal、cancellation 和扁平结果；Tool 执行不做 generic retry。

### Required provider and CLI tests

- Anthropic 和 OpenAI-compatible fake adapters 的 streaming、tool_calls、timeout 和错误转换。
- chat route 必须 streaming；schedule 可非 streaming。
- route fallback、未知 route、未知 protocol provider 和 default unusable。
- 模型固定 5 次指数退避 retry 和 retry-after。
- 首次运行生成配置并退出。
- 配置无效时 `myclaw` 直接退出。
- `myclaw config` 的生成、完整展示和 API key 脱敏。
- `/config`、`/status`、`/resume`、`/memory`、`/dream`。
- 非内置 `/...` 文本发送给模型。
- Ctrl+C、`exit`、`quit` 和 REPL 退出取消后台任务。
- 多 REPL 不协调行为不应被测试误认为有跨进程锁保证。

## Out of Scope

- 多租户或多用户 Agent 平台。
- channel-first bot 平台。
- one-shot 对话命令或 one-shot runtime。
- detached daemon、系统服务、HTTP 或 IPC server。
- 多 REPL、同 session、Memory Task、Schedule Job 的跨进程协调或锁。
- 微服务拆分。
- MCP 工具扩展。
- subagent/spawn 或多 Agent 编排。
- 用户可配置 identity/system prompt。
- Agent profiles、session overrides、per-chat settings。
- Session list/view/delete/rename 管理命令。
- 独立 Schedule CLI、Job 修改、pause/resume/run-now 管理命令。
- Agent Home 或 Workspace 级全局 Runtime Log；按 Session 归属的 Session Log 由 ADR-0008 约束。
- SQLite、混合数据库或向量数据库记忆。
- Long-term Memory 相关性筛选或大小上限。
- Tool Gateway 对前台和后台工具调用加全局锁。
- 用户不能扩展 Exec 的安全规则或固定 Tool Catalog。
- Exec 子进程的 OS 级文件系统或网络 sandbox。
- API key 环境变量引用或系统密钥链。
- Tool Artifact 自动清理。
- notification adapter 或系统桌面通知。
- 非交互 status 或 `myclaw --help` 产品能力。
- nanobot 同等广度的 provider registry。

## Further Notes

- ADR 0001 记录 file-first local persistence。
- ADR 0002 记录固定 Agent Home `~/.myclaw/`。
- ADR-0010 记录 Exec 的 Workspace cwd、destructive/DNS 确认边界及首版不提供 OS 级 sandbox；Exec 没有 allowlist，已知风险形状请求一次性确认。
- `docs/myclaw-runtime-contracts.md` 是已接受的首版 schema、Port、事件和错误契约。
- `CONTEXT.md` 是最终 canonical language；本 PRD 的实现术语应与其保持一致。
- GitHub issue：<https://github.com/Totoro-debug/myclaw/issues/1>；本文件仍是需求的本地 canonical source。
