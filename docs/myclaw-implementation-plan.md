# MyClaw 首版分阶段实现计划

## 1. 计划目标与需求基线

本计划用于把 MyClaw 首版从已确认的设计文档推进到可安装、可运行、可验收的 Python 产品。实现边界以以下文档为准：

1. `CONTEXT.md`：canonical domain language 与边界规则，发生冲突时优先级最高。
2. `docs/myclaw-personal-agent-prd.md`：产品行为、实现决策、必测场景和 Out of Scope。
3. `docs/adr/0001-file-first-local-persistence.md`：file-first persistence 与不做跨进程协调的边界。
4. `docs/adr/0002-fixed-agent-home.md`：固定 `~/.myclaw/` 与 User Configuration ownership。
5. `docs/adr/0005-store-workspace-state-in-workspace.md` 与 `docs/adr/0009-active-session-snapshot-persistence.md`：Workspace State layout、active Session authority 和 snapshot persistence。

需求基线是提交 `d6b6e00` 及工作区中 PRD 对 GitHub issue 链接的后续更新。开始编码前应把本计划与当前 canonical 文档一起确认；后续需求变化必须先更新对应文档，再调整实现和测试。

首版完成的判断标准不是“模块都已创建”，而是以下用户路径能够在隔离环境中端到端运行：

- 首次运行生成配置并退出，配置有效后运行 `myclaw` 进入 REPL。
- 主对话通过 chat Model Route streaming，支持多轮、工具调用、取消与持久化。
- 当前 Workspace 的 session 可以恢复，异常、中断和大工具结果不会破坏历史。
- 长对话可以压缩为 Conversation Summary，Memory Task 可以按规则更新 Long-term Memory。
- Schedule Jobs 只在 runtime 存活期间运行，结果写入 Schedule Session，不发送 Agent Event 或通知。
- 所有 required tests 使用 fake provider、fake tool、临时 Agent Home 和临时 Workspace 通过，不依赖真实模型 API。

## 2. 实施原则

- 采用纵向切片推进。每个阶段都产出可运行行为，不按“先写完所有接口、最后统一集成”的方式实施。
- 测试外部可观察行为。优先从 Conversation Port、Management Port、Runtime Core、Memory Manager、Tool Gateway、active Session、Model Router/Provider Adapter 验证契约。
- 核心层不依赖 Typer、Rich、Anthropic SDK、OpenAI SDK 或具体文件路径实现；这些依赖通过 adapter 接入。
- Workspace State 写入采用各自的原子写策略；active Session 在 turn 结束后冻结完整 JSON-native state，并通过 ordered async JSONL replacement 持久化，shutdown 使用 bounded synchronous close。
- 所有并发保证仅限单 runtime。不得意外引入或在测试中承诺跨进程锁、去重或一致性。
- 先用 fake provider 打通 Runtime Core，再接真实 provider，避免把 SDK 行为和编排错误混在一起定位。
- 任何阶段都不得重新引入 one-shot、daemon、HTTP/IPC、MCP、subagent、可配置 Permission Policy 或用户自定义 identity。

## 3. 建议的包与模块边界

以下是当前模块布局，不是要求每个文件都形成一层抽象。若实际代码更简单，可合并同一边界内的小文件，但不得让 CLI 直接读写 session、memory 或 provider SDK。

```text
myclaw/
  agent/               # Runtime Core、Agent Event、Conversation Port、prompts、Workspace
  config/              # Agent Home、TOML、ConfigView、脱敏、route 解析
  errors.py            # 稳定 ErrorCode 与 ErrorInfo
  management/          # Management models/Port、service 与命令分发
  memory/              # Summary records、Memory models/ports、任务与 scheduler
  provider/            # Model models/Port、Router、errors 与 Provider adapters
  schedule/            # Schedule model、state、Tool 与 Service
  session/             # active Session、Conversation 与 resume
  terminal/            # Typer CLI、REPL 与 foreground interrupts
  tools/
    models.py          # Tool 定义、调用、结果与执行上下文
    artifacts.py       # ArtifactReference 与文件名编码
    ports.py           # Tool Protocol
    files/             # Workspace 文件读写工具
    shell/             # Shell policy、process 与 Tool adapter
    web/               # WebSearch 与 WebFetch
  utils/               # JSON 类型、通用校验、时间格式与 atomic file helpers
tests/
  agent/               # Agent Event 约束
  architecture/        # Protocol 替换性与 AST 模块边界
  configuration/       # 配置行为
  management/          # Management Port 行为
  memory/              # Conversation Summary 与 Memory Task
  provider/            # Provider-neutral models
  scheduling/          # Schedule state、Tool 与 Service 行为
  sessions/            # Conversation、active Session、resume 与 title
  tools/               # files、shell、web 与 artifact 行为
  utils/               # 时间格式与 Session identifier 行为
  fixtures/            # fake provider、fake tool、clock、Agent Home builders
  test_*.py            # 跨模块、Runtime、CLI 与安全行为
```

关键依赖方向：

```text
CLI -> Conversation Port / Management Port -> Runtime Core
Runtime Core -> active Session / Memory Manager / Model Router / Tool Gateway
Model Router -> Provider Adapter
Tool Gateway -> Permission Policy / Built-in Tools / Artifact Store
Schedulers -> Runtime Core or Memory Manager
Persistence adapters -> Workspace State paths and atomic persistence helpers
```

反向依赖不允许出现。例如 provider adapter 不知道 REPL，Session 不渲染 Rich 输出，Tool Gateway 不直接处理用户输入。

## 4. 阶段总览与依赖

| 阶段 | 可演示里程碑 | 主要依赖 |
| --- | --- | --- |
| Phase 0 | 契约、schema、工程骨架可执行 | 无 |
| Phase 1 | 配置与 Agent Home 行为可独立验收 | Phase 0 |
| Phase 2 | fake provider 的 streaming REPL 纵向切片 | Phase 1 |
| Phase 3 | session 生命周期与管理命令完整 | Phase 2 |
| Phase 4 | 文件工具、权限、tool loop 和 artifact 完整 | Phase 3 |
| Phase 5 | 真实 providers、Shell 和 Web 接入 | Phase 4 |
| Phase 6 | 三层 Memory System 完整 | Phase 5 |
| Phase 7 | Schedule Jobs、共享 Agent Run 与 Runtime 生命周期完整 | Phase 6 |
| Phase 8 | 全量验收、真实冒烟与发布准备 | Phase 7 |

Phase 0 至 Phase 2 是第一条 tracer bullet；完成后已经具备真实 CLI、真实 session 文件和可替换的模型边界。Phase 3 至 Phase 5 完成主 Agent 闭环。Phase 6 至 Phase 7 加入后台能力。Phase 8 只做收敛，不承接新的产品范围。

## 5. Phase 0：冻结契约并建立工程骨架

阶段状态：D01-D16 已于 2026-07-11 接受，并固化在 `docs/myclaw-runtime-contracts.md`；PRD 与 ADR 已同步。工程骨架与 fixtures 已实现；运行时约束已于 2026-07-18 按领域所有权拆分，模块边界由 AST 测试约束。

### 目标

消除会导致不同模块各自猜测的 schema 空白，建立可安装的 Python 项目、质量门和共用测试夹具。

### 实现任务

1. 确定 Python 最低版本、构建后端、依赖管理方式和 `src/` 布局，配置 `myclaw` console script。
2. 加入 Typer、Rich、TOML/schema、async 测试、cron、HTTP、Anthropic SDK 和 OpenAI SDK 的依赖；锁定兼容版本范围。
3. 定义核心值对象和 typed contracts：Agent Event、Agent Run、model message/tool call、model usage、normalized tool result、permission decision、JSON-native Session state、summary entry、Schedule state。
4. 定义 Conversation Port、Management Port、active Session、Model Provider、Summary Store、Memory Store 和 Tool 接口的最小方法集合。
5. 确认所有时间字段格式、local timezone 语义、UUID 形式、错误分类和用户可见错误映射。
6. 建立 `pytest`、async 测试支持、lint、format、type check 和覆盖率命令；提供 fake clock、scripted fake provider、fake tool、临时 Agent Home/Workspace fixtures。
7. 为内置 prompts 建立版本可追踪的独立资源，避免 prompt 字符串散落在编排代码中。

### 已冻结的设计门

- `config.toml` 的完整字段名、类型、默认值、范围以及未知字段策略。
- session metadata 与 user/assistant/tool JSONL 的精确 schema，包括 interrupted/error 和 artifact reference 的表达。
- Schedule state 根 JSON 文件名为 `schedule.json`；legacy scheduled-work state 原样保留且不读取、不迁移、不删除。
- 内置 file tools 的名称和参数 schema，以及哪些 Agent Home 路径允许主 Agent 读取。
- Shell 极小只读 allowlist 的精确命令和参数判定规则。
- WebSearch 的实际后端、凭据和 normalized result schema。
- token estimate 的算法及 `/status` 中“估算 token 状态”的展示口径。
- Session title fallback 的截断长度和 Unicode/空白处理。

以上结论以 `docs/myclaw-runtime-contracts.md` 为精确契约。后续若改变产品行为，必须先更新契约、PRD 或对应 ADR，而不是只修改代码。

### 测试与退出条件

- `myclaw` 包可安装，console script 可启动，测试、lint 和 type check 在干净环境执行成功。
- fake provider 能脚本化输出 text chunks、tool calls、usage、retryable error、final error 和 cancellation。
- contract 测试能证明 CLI/core 测试不需要真实 SDK 或真实用户目录。
- schema fixtures 固化在测试中，后续阶段不能无迁移意识地改变持久化格式。

## 6. Phase 1：Agent Home、配置与持久化基础

### 目标

先建立所有后续功能共享的路径、安全写入和配置真相源，并交付完整的 `myclaw config` 行为。

### 实现任务

1. 实现固定 Agent Home 路径解析，并将其所有权限制为 User Configuration 与 untouched legacy Runtime Log files；新的 Session Logs 属于 Workspace State，不得暴露为用户配置或 profile。
2. 有效启动在当前 Workspace 创建 `.myclaw`、`memory/`、`sessions/` 和四分区 `memory/memory.md` 模板；其他运行态文件按需创建。
3. 实现宿主原生 normalized absolute Workspace identity，直接在 Workspace State 存储非全局状态，不派生 slug。
4. 实现同目录临时文件、flush、必要时 fsync、原子 replace 的写入助手，并清理失败临时文件。
5. 实现 TOML 默认模板、解析、schema 校验、provider/route 可用性校验和用户可见错误。
6. 实现 API key 结构化脱敏，以及 TOML 无法解析时对原始文本中明显 API key 行的保守脱敏。
7. 实现 `myclaw config`：缺失时生成并显示脱敏内容；有效或无效时均可检查；不得启动 runtime。
8. 实现 `myclaw` 启动前校验：缺配置时生成后退出，解析失败、模型配置不完整或 default route 不可用时明确退出。

### 测试与退出条件

- 在临时 HOME 中验证首次目录和 memory template 创建的幂等性。
- 模拟写入中断，旧文件仍完整，临时文件不会被误读为正式状态。
- 覆盖未知 route、未知 protocol provider、空 model catalog、model 不在 catalog、缺/空 base URL、不可用 default。
- API key 在正常配置输出和解析失败原文中都不泄露，非密钥内容仍完整显示。
- `myclaw config` 和 `myclaw` 的缺配置/坏配置退出码及错误信息可由 CLI 测试稳定断言。

## 7. Phase 2：Model Router 与最小 streaming REPL 纵向切片

### 目标

用 fake provider 打通“用户输入 -> Runtime Core -> streaming Agent Events -> session 持久化 -> 终端显示”的最小闭环。

### 实现任务

1. 实现 Model Router，只接受 default/chat/memory/schedule；具体 route 缺失或不可用时 fallback default。
2. 实现统一 model request/stream event/response/usage/error contract，以及固定最多 5 次的 retry coordinator。
3. retry 仅处理明确的临时 provider/model 错误，使用可注入 clock 的指数退避，并尊重 retry-after；取消和永久错误不得重试。
4. 实现 active Session：延迟物化、严格五字段 header、JSON-native message dictionaries、完整 atomic JSONL replacement、ordered async `persist()` 和 bounded synchronous `close()`。
5. 实现 typed Agent Events 的最小集合：streamed text、final output、error；明确事件顺序和每个 turn 只有一个终态。
6. 实现 Conversation Port 和 Runtime Core 的单轮 chat 路径；chat 必须 streaming，完成后一次写入 assistant message。
7. 实现 system prompt 和 Runtime Context 组装：内置 identity + 启动时缓存的 Long-term Memory + Workspace；每轮 user input 带当前时间和 session ID。
8. 实现 Rich REPL：`myclaw` 无参数进入、同一进程串行多轮、`exit`/`quit` 退出；未知 `/...` 暂按普通消息发送。
9. 首条 user message 才物化 session，空 REPL 退出不留下 session 文件。

### 测试与退出条件

- scripted fake provider 连续产生多个 chunks 时，CLI 渐进显示而 session 只保存一个完整 assistant message。
- route fallback、未知 route、default unusable、5 次 retry、retry-after 和 cancellation 均有确定性测试。
- 连续两轮输入复用同一 session，第二轮能够看到正确 Short-term Memory。
- 空 REPL 不创建 session；模型最终失败保存 assistant error；Ctrl+C 保存 partial interrupted/error assistant 后仍可继续下一轮。
- 演示命令可以在完全离线条件下通过 fake provider 完成一轮 streaming 对话。

## 8. Phase 3：完整 Session 生命周期与 Management Port

### 目标

补齐 session title、恢复、状态查询和 REPL 管理体验，使对话基础能力达到产品要求。

### 实现任务

1. 完成 Session ID、JSON-native metadata、cumulative usage、`last_consolidated`、message count 与 local-time updated time 管理。
2. 首条用户输入后异步调用 chat route 生成 title，不阻塞首轮回复；失败时使用确定性的截断标题。
3. title 调用不写入 conversation history，但 usage 计入 Session；late title 可由后续 ordered snapshot 或 bounded close 落盘。
4. 实现当前 Workspace session 枚举、排序、损坏文件隔离和 Rich 交互式 picker。
5. 实现 `/resume` 切换：只列当前 Workspace；保留有消息的原 session，可丢弃空 session；新输入写入被选择的 session。
6. 实现 Management Port 和 `/config`、`/status`、`/memory` 的只读路径。
7. `/memory` 每次读取磁盘最新值；`/status` 使用 runtime 缓存和 session 状态计算要求字段。
8. 完善退出语义：忽略 `exit`/`quit` 周围空白与大小写，runtime shutdown 关闭资源。

### 测试与退出条件

- 覆盖 Workspace-owned Session 路径、strict header、三种 role、完整 replacement、snapshot freeze 和同 runtime ordered persistence。
- 验证 title 的异步性、fallback、usage 累计、late completion、silent ordinary failure、bounded close retry 和正确 Session 归属。
- `/resume` 不泄露其他 Workspace session，切换后上下文和新消息归属正确。
- `/config` 完整但脱敏；`/status` 字段齐全；`/memory` 能看到 runtime 启动后发生的磁盘修改。
- 未知 slash 输入仍进入 Conversation Port，绝不能被 CLI 吞掉或当作错误命令。

## 9. Phase 4：Tool Gateway、文件工具与 Tool Artifact

### 目标

形成可扩展但边界固定的 agent tool loop，先交付风险较低的文件能力和统一权限/结果处理。

### 实现任务

1. 实现 Tool Catalog、参数 schema 校验、tool resolution、Tool Confirmation 与 normalized result。
2. Runtime Core 支持 assistant content 与 tool calls 共存，按协议持久化 assistant/tool messages，并继续模型循环直到 final output。
3. 实现 tool activity 和 `confirmation_requested` Agent Events；默认只显示工具名与状态摘要，不泄露完整参数/结果。
4. 实现 file read/list/search，并固定拒绝 Workspace create/write/edit；路径在操作前规范化并检查 symlink/reparse 后的真实边界。
5. 主 Agent 对 config、memory、sessions、summary、cursor 和 Schedule state 内部文件的写入一律拒绝；越界 fail closed，不升级为 confirmation。
6. Schedule add/remove 的前台 Tool Confirmation 阻塞当前 Agent Run；批准或拒绝不增加独立 history，非交互 Schedule Agent Run 自动拒绝。其他 refused capability 不进入确认流程。
7. 工具异常、参数错误和拒绝都转换为 tool result；Tool Gateway 执行 concrete Tool 声明的有界 retry，取消继续向上传播。
8. 超过 `max_tool_result_chars` 的原始结果原子写入 Tool Artifact，session 保存引用和截断 preview。
9. 中断时保留已完成 messages，并把未完成 tool calls 物化为 tool error results，使 session 可恢复。

### 测试与退出条件

- file allow/refused/error 矩阵完整覆盖 Workspace、允许读取的状态别名、内部写保护、`..`、symlink/reparse 和不存在目标的父目录。
- tool call 参数错误和执行失败只执行一次；assistant/tool 关联 ID 可完整恢复。
- Schedule confirmation 接受、拒绝、非交互自动拒绝的事件和 session 历史符合要求。
- 阈值边界、artifact 路径、原始内容、preview、atomicity 和随 session 恢复均通过测试。
- streaming 中断和 tool execution 中断后，JSONL 仍是合法、语义完整的对话。

## 10. Phase 5：真实 Provider、Shell 与 Web 工具

### 目标

接入首版全部外部能力，并在统一 contracts 后面隔离 SDK、进程和网络差异。

### 实现任务

1. 使用官方 Anthropic SDK 实现 adapter，转换 streaming text、tool use、usage、timeout、retry-after 和错误类别。
2. 使用官方 OpenAI SDK 实现 openai-compatible adapter，支持 required base URL、streaming、tool calls、usage 和错误转换。
3. provider 不支持 reasoning effort 时静默忽略；chat 强制 streaming，memory/schedule 允许非 streaming。
4. 实现 Shell tool：配置 enablement、Workspace cwd、60-600 秒 timeout、固定只读 allowlist和非 allowlist refusal；安全终止超时/取消的子进程。
5. 实现 WebSearch adapter 和 normalized results；配置关闭时不进入 catalog。
6. 实现 WebFetch：仅公网 HTTP(S)、DNS/IP 校验、阻止 localhost/private/link-local、每次 redirect 重新校验、最多 5 次跳转、响应大小/超时限制。
7. web/shell enablement 同时作用于前台 chat 和 Schedule Jobs 的 catalog 组装。
8. 增加可选的手工真实 API smoke tests，必须通过环境标志显式启用，默认测试套件仍完全离线。

### 测试与退出条件

- 用 fake SDK/client contract tests 覆盖两类 adapter 的 stream、mixed content/tool calls、usage、timeout、retryable/permanent error。
- 验证 route 层负责 5 次 model retry，而 adapter 与 Tool Gateway 不重复 retry。
- Shell 覆盖 allowlist、refusal、cwd 边界、timeout 下限/上限、超时和 Ctrl+C 的进程清理。
- WebFetch 覆盖直接私网、DNS 解析到私网、公开 URL 重定向到私网、循环/超过 5 次重定向。
- 至少各完成一次 Anthropic 与 OpenAI-compatible 的人工 streaming 冒烟；若无凭据，记录为发布前待执行项，不让单元测试依赖凭据。

## 11. Phase 6：三层 Memory System

### 目标

实现长对话压缩、全局 summary 流、Long-term Memory 缓存语义和可手动/周期执行的 Memory Task。

### 实现任务

1. 实现 Short-term Memory 为 active Session `last_consolidated` 之后的 suffix，并纳入 chat context assembly。
2. 在 chat 调用前同时评估 token budget 与总消息数阈值，达到任一条件则同步执行 consolidation。
3. 实现选择约半预算/半阈值早期消息、cutoff 向下一个 user 对齐、无后继 user 时回退最近前一个 user 的算法。
4. 使用 memory route 生成 summary，不注入 Long-term Memory；memory 与 default 都失败时使当前 chat 明确失败。
5. 以 `{index, timestamp, content}` 写入全局 `summary.jsonl`，之后直接更新 active Session `last_consolidated`；接受两者在 crash 或 Session snapshot failure 后 divergence，不引入 pending journal。
6. 实现 Summary Cursor 纯文本 store、batch 读取和 Memory Manager prompt。
7. 建立受限 memory Tool Gateway：可以读取当前 Long-term Memory，只能编辑 `memory/memory.md`，无需用户确认。
8. 实现 Memory Task 状态机：no edit 与 edit success 推进 cursor，required edit failure 不推进；单 runtime 内不重入。
9. 实现 `/dream`：无 pending 时零模型调用；运行中时拒绝；其他情况前台阻塞并返回摘要状态。
10. 按本地时区和配置 cron 启动后台 Memory Task，默认 hourly；重叠 trigger 跳过，周期结果不通知 REPL。
11. 保持 Long-term Memory 启动时缓存：Memory Task 修改后 chat 仍使用旧缓存直到 runtime 重启，而 `/memory` 读取磁盘最新内容。

### 测试与退出条件

- PRD 的全部 required memory tests 通过，包括两种触发、两条 cutoff 路径和 summary 不带 source identity。
- 验证 summary 不直接注入 chat、不立即触发 Memory Task、生成时不注入 Long-term Memory。
- 对 Summary 写入和 Session snapshot 注入故障，验证 failure silence、`last_consolidated` 保持内存语义，以及 crash divergence 的已接受边界；不得声称跨文件强一致或启动恢复。
- Summary Cursor 在 no update、edit success、edit failure 下严格按规则推进。
- batch size、手工/周期不重入、受限 edit 路径、runtime cache 与磁盘视图差异均有 integration tests。

## 12. Phase 7：Schedule Jobs 与完整 Runtime 生命周期

### 目标

完成自然语言 Schedule Job、Schedule Session、共享 Agent Run 与 Schedule Service 生命周期，并收口 asyncio 资源关闭。

### 实现任务

1. 实现 `schedule.json` strict array Store、schema 校验、copy-on-write 与 atomic replacement；损坏 state 在 scheduler/REPL 前阻止启动。
2. 实现 `schedule` Tool 的 add/list/remove：支持 at/every/cron，add/remove 通过 Tool Confirmation，list 只返回 user Job。
3. 每个 Job 创建并复用 `schedule_<job_id>` Schedule Session，首次产生消息时写入 `schedule-sessions/` 分区。
4. 每次触发通过共享 Agent Run 的 `schedule` route 执行，注入启动时 Memory snapshot，保存完整 Session history。
5. 每个 Job 在单 runtime 内加运行态 guard；重叠 trigger 跳过，不实现跨进程去重。
6. runtime 启动 Memory scheduler 与唯一 Schedule Service；前台 Conversation 和 Schedule Job 共享资源关闭顺序。
7. Schedule Service 不发 Agent Event、completion prompt 或 notification；结果以 Schedule Session 为真相源。
9. Ctrl+C 只取消当前前台 turn；`exit`/`quit` 立即取消并 await 全部后台任务，关闭 provider/HTTP/子进程资源。
10. 明确异常隔离：单个后台任务失败不能终止 REPL 或 scheduler loop，失败写入对应 task session。

### 测试与退出条件

- 创建确认接受后只写一次，拒绝和非交互 confirmation refusal 不写；JSON array 的并发写在单 runtime 内串行。
- at/every/cron trigger、Schedule Session 归属、schedule route/fallback、Long-term Memory snapshot 和 final result 持久化正确。
- 同 Job 不重入，不同 Job 可并发；多 runtime 不协调的行为不被测试成强保证。
- legacy scheduled-work state 无论内容或 path type 都保持原样，且不被读取、检测、迁移或删除。
- Ctrl+C、正常退出、异常退出路径没有遗留 asyncio tasks；退出时后台取消不会破坏已有文件。

## 13. Phase 8：全量验收、加固与发布准备

### 目标

只修复集成缺口和发布问题，以需求矩阵证明首版完成，不在本阶段新增范围。

### 实现任务

1. 建立“48 条 User Stories -> 测试/演示”的追踪矩阵，以及“Required tests -> 测试文件”的反向索引。
2. 在 Windows x64 发布候选上运行完整测试，重点核对路径、原子 replace、subprocess cancellation 和终端中断行为。
3. 执行安全复核：路径穿越与 symlink、Agent Home 内部写保护、Shell policy、SSRF/redirect、secret redaction、artifact 泄露面。
4. 执行故障注入：磁盘写失败、损坏 JSONL/TOML/JSON、provider 连续失败、网络超时、取消发生在 stream/tool/metadata update 各阶段。
5. 执行 REPL 手工验收：首次配置、streaming、Schedule 确认/拒绝、resume、长对话 consolidation、`/dream`、Schedule 静默后台执行和退出清理。
6. 补齐安装、配置、Agent Home 文件说明、已知限制和故障排查文档；明确 API key 是 plaintext 风险。
7. 确认打包元数据、版本展示、license、console entry point 和干净环境安装。
8. 输出首版已知风险：多 REPL 重复调度、同 session 跨进程并发、Long-term Memory 无大小上限、artifact 不自动清理。

### 最终退出条件

- 所有 PRD Required tests 通过，lint、format、type check 和 packaging check 通过。
- fake provider 的完整端到端套件稳定、无网络依赖；真实 provider smoke test 有明确结果记录。
- 临时 Agent Home 的验收不会读写真实 `~/.myclaw/`。
- 所有用户可见错误不包含 API key、完整敏感 tool 参数或意外 traceback。
- 文档与实现术语一致，无 one-shot、daemon、MCP、subagent 或跨进程保证残留。
- 安装后仅通过 `myclaw` 和 `myclaw config` 即可完成首版规定入口的验收。

## 14. 建议的测试分层与持续集成门禁

### 每个工作包提交前

- 运行受影响模块的 unit/contract tests。
- 运行 format、lint 和 type check。
- 新增持久化字段、事件或错误类型时同步更新 schema fixture 和兼容性测试。

### 每个 Phase 合并前

- 运行完整离线测试套件。
- 运行当前阶段的 CLI/integration 演示脚本。
- 对 Agent Home 文件做快照检查，确认路径、JSONL/JSON/TOML/Markdown 格式和 secret redaction。
- 检查任务泄漏、未关闭 client、未 await coroutine 和 flaky timing；调度测试使用 fake clock，不依赖真实 sleep。

### 发布候选

- Windows x64 测试与 Python 3.12 门禁。
- 只构建通用 `py3-none-any` wheel，不产生额外发行物。
- 在空 Windows x64 虚拟环境安装唯一 wheel，再执行 Unicode CLI smoke tests。
- 手工真实 provider 与真实网络测试从默认 CI 隔离，凭据只通过 CI secret 或本地环境提供，不写入 fixture/log。

## 15. 风险与控制措施

| 风险 | 可能后果 | 控制措施 |
| --- | --- | --- |
| file-first 多文件更新不是事务 | summary 已写但 cursor 未更新，重启后重复处理 | 明确写入顺序、幂等恢复规则和故障注入测试 |
| 多 REPL 无跨进程协调 | 同 session 交错、重复 Memory/Scheduled trigger | 文档明确接受；只保证单 runtime 内锁与不重入 |
| SDK streaming 事件语义不同 | partial、tool calls 或 usage 丢失 | 统一 adapter contract，用 fake SDK transcript 做契约测试 |
| Ctrl+C 与后台 task ownership 不清 | 误取消 scheduler 或遗留任务 | Runtime 统一拥有 task，前台 cancellation scope 独立 |
| 文件边界只做字符串前缀检查 | 路径穿越或 symlink 越权 | 规范化/解析真实路径后比较目录归属，覆盖 Windows reparse 场景 |
| WebFetch DNS/redirect 检查不完整 | SSRF 访问本机或内网 | 每跳解析和校验，限制 scheme、redirect 次数、响应大小和 timeout |
| 配置/错误输出泄露 API key | 终端历史或 CI log 泄密 | 结构化脱敏 + parse-failure 文本脱敏 + 错误快照测试 |
| Long-term Memory 无首版大小上限 | system prompt 超预算导致 chat 失败 | 按需求显式报配置/记忆过大错误，不静默裁剪 memory |
| Schedule Job 与前台 streaming 竞争 | 共享资源关闭或 Session 写入顺序错误 | Schedule Service 不向 Conversation Port 发送事件，结果只保留在 Schedule Session |
| 测试依赖 wall clock 或真实网络 | 慢、flaky、无法离线 | fake clock、fake provider、fake HTTP transport，真实 smoke 独立运行 |

## 16. 明确不进入首版的工作

以下事项不得作为“顺手优化”加入任何 Phase：one-shot 对话、daemon/系统服务、HTTP/IPC、跨进程锁或调度去重、MCP、subagent、多 Agent 编排、用户配置 identity、profiles、session 管理扩展、独立 Schedule CLI、Job 修改/pause/resume/run-now、Agent Home 或 Workspace 级全局 Runtime Log、SQLite/向量数据库、Long-term Memory 相关性筛选或大小上限、可扩展 Shell allowlist、密钥链或环境变量 key reference、artifact 自动清理、notification adapter、非交互 status，以及 nanobot 等宽 provider registry。

如确需其中任一能力，应先进入下一版本需求与 ADR，不应延后当前 Phase 的退出条件。

## 17. 推荐执行顺序

1. 先完成 Phase 0 的决策门和骨架，不要同时开始真实 SDK、Web 或 scheduler。
2. 把 Phase 1-2 作为第一里程碑，实现可重复演示的离线 streaming REPL。
3. 把 Phase 3-5 作为第二里程碑，完成可实际使用的主 Agent 对话和工具闭环。
4. 把 Phase 6 作为独立里程碑，因为 memory 的多文件一致性和失败语义需要集中验证。
5. 完成 Phase 7 后才宣称 runtime 生命周期闭环；随后 Phase 8 冻结功能，只做验收和加固。

每个 Phase 可拆成若干小 issue，但 issue 必须以可观察行为收尾，并声明依赖的前置 issue。不要按目录或类名拆成缺少端到端价值的任务。
