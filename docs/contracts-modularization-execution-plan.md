# Contracts 模块化拆分执行计划

## 文档状态

- 状态：Completed（2026-07-18）
- 对应 Issue：[#37 将集中式 contracts 约束拆分到各领域模块](https://github.com/Totoro-debug/MyClaw/issues/37)
- 变更性质：内部架构重构
- 产品行为变化：无

本文件保留迁移前路径与分阶段任务，作为 Issue #37 的历史执行记录。下文中的
`myclaw/contracts/`、`tests/contract/` 和 `myclaw.contracts` 均指已经删除的旧结构，
不是当前有效的源码或导入路径。

2026-08-04 的 Session 架构替换已删除下文记录的 Session 类型和 persistence
ports；当前 active Session 与 snapshot behavior 由
`docs/adr/0009-active-session-snapshot-persistence.md` 约束。为保留 Issue #37 的
历史证据，下文的中间目标和迁移步骤不回写为当前 API。

2026-07-29 的后续整理进一步删除了通用 `ports.py` 模块：`ConversationPort` 与
Agent Event 同归 `myclaw/agent/events.py`，`ModelProvider` 与 Provider 中立模型同归
`myclaw/provider/models.py`，Store 协议则与各自的持久化或编排实现同模块。下文仍
保留 Issue #37 执行时采用的中间目标路径，不代表当前导入路径。

## 1. 目标

将 `myclaw/contracts/` 中集中维护的值对象、持久化记录、事件、错误和
Protocol 拆分到拥有相应生命周期与不变量的领域模块中，使导入路径直接表达依赖
方向，并最终删除集中式 `contracts` 包。

本次重构不得改变以下已接受行为：

- User Configuration、Conversation Session、Conversation Summary、Scheduled Work 和
  Tool Artifact 的持久化格式。
- JSON/JSONL 字段、紧凑编码、换行和时间表示。
- Session ID、UUID、计数器和标题校验规则。
- 稳定 ErrorCode 集合及错误重试语义。
- Agent Event 的 envelope、payload 和事件序列。
- Conversation Port、Management Port、Store、Provider 和 Tool 的方法行为。

## 2. 目标依赖结构

```text
utils / errors
       |
       v
session.identifiers
       |
       v
tools.models
       |
       v
provider.models
       |
       v
session.records
       |
       +----------+----------+----------+
       v          v          v          v
    memory     schedule    agent    management
```

依赖规则：

1. `utils` 和根级错误结构不得依赖任何领域模块。
2. `tools` 可以依赖 `session.identifiers`，不得依赖 `session.records`。
3. `provider.models` 可以依赖 Tool 声明和 ModelToolCall，Tools 不得反向依赖
   Provider 模型。
4. `session.records` 可以引用 Provider usage/message 类型和 Tool result/artifact
   类型。
5. Port 只能依赖其方法签名实际暴露的模型。
6. 生产代码必须直接从约束所有者导入，包级 `__init__.py` 不得重新建立跨领域
   聚合入口。

## 3. 目标文件归属

| 当前内容 | 目标位置 |
| --- | --- |
| JSON 类型别名 | `myclaw/utils/json_types.py` |
| 通用数值、datetime、UUID 校验 | `myclaw/utils/validation.py` |
| RFC3339 毫秒格式化 | `myclaw/utils/time.py` |
| ErrorCode、STABLE_ERROR_CODES、ErrorInfo | `myclaw/errors.py` |
| Session ID 生成与校验 | `myclaw/session/identifiers.py` |
| Session 持久化记录 | `myclaw/session/records.py` |
| ResumeResult 等非持久化 Session 结果 | `myclaw/session/models.py` |
| SessionStore | `myclaw/session/ports.py` |
| Tool 定义、调用、结果与执行上下文 | `myclaw/tools/models.py` |
| ArtifactReference 与文件名编码 | `myclaw/tools/artifacts.py` |
| PermissionDecision | `myclaw/tools/permission_policy.py` |
| Tool Protocol | `myclaw/tools/ports.py` |
| Model 请求、响应、消息、usage 与 stream | `myclaw/provider/models.py` |
| ModelProvider | `myclaw/provider/ports.py` |
| ModelCallError | `myclaw/provider/errors.py` |
| SummaryEntry | `myclaw/memory/records.py` |
| MemoryTaskResult | `myclaw/memory/models.py` |
| SummaryStore、MemoryStore | `myclaw/memory/ports.py` |
| ScheduledWork 与序列化 | `myclaw/schedule/records.py` |
| Agent Event 与序列校验 | `myclaw/agent/events.py` |
| ConversationPort | `myclaw/agent/ports.py` |
| ConfigView | `myclaw/config/models.py` |
| RuntimeStatus | `myclaw/management/models.py` |
| ManagementPort | `myclaw/management/ports.py` |

`ModelToolCall` 归入 `tools.models`。它代表模型提出的 Tool 调用，同时被 Provider
adapter、Tool Gateway 和 Session 记录消费；由 Tools 所有可以消除 Tools 对 Provider
模型的反向依赖。

## 4. 执行原则

每个迁移步骤先创建新定义，再将旧模块改为对新定义的再导出。兼容层不得复制
dataclass、Enum 或 Protocol 定义，确保新旧路径引用同一个 Python 对象，避免
`isinstance`、event payload 校验和 runtime-checkable Protocol 出现差异。

每个 PR 合并前必须满足：

```powershell
python -m pytest
python -m ruff check .
python -m mypy
```

## 5. PR 1：基础类型与 Session ID

### 5.1 新增文件

- `myclaw/errors.py`
- `myclaw/utils/json_types.py`
- `myclaw/utils/validation.py`
- `myclaw/utils/time.py`
- `myclaw/session/identifiers.py`

### 5.2 迁移任务

1. 将 JsonScalar、JsonValue、JsonObject 移入 JSON 类型模块。
2. 将非负整数、非负有限数值、aware datetime 和 UUID4 校验移入通用校验模块。
3. 将 RFC3339 毫秒格式化移入时间工具模块。
4. 将 Session ID 正则、生成和校验移入 Session identifiers。
5. 将 ErrorCode、稳定错误码集合和 ErrorInfo 移入根级错误模块。
6. 将原 common、json_types 和 errors 模块改为兼容性再导出。
7. 将对应生产消费者逐步改为直接导入新模块。
8. 增加旧路径与新路径对象一致性的兼容测试。

### 5.3 验收条件

- Common contract 和 Error contract 测试通过。
- Session ID 和时间格式化的输出逐字符不变。
- 稳定错误码集合没有增加、删除或重命名。
- 新基础模块不导入任何领域模块。

## 6. PR 2：Tools 与 Provider

### 6.1 新增文件

- `myclaw/tools/models.py`
- `myclaw/tools/artifacts.py`
- `myclaw/tools/ports.py`
- `myclaw/provider/models.py`
- `myclaw/provider/ports.py`
- `myclaw/provider/errors.py`

### 6.2 迁移任务

1. 迁移 ToolResultStatus、ToolExecutionLane、ToolDefinition、ModelToolCall、ToolResult
   和 ToolExecutionContext。
2. 迁移 ArtifactReference 和 artifact tool_call_id 编码规则。
3. 将 PermissionDecision 移入 Permission Policy 模块。
4. 将 Tool Protocol 移入 Tools ports。
5. 迁移 ModelRoute、ReasoningEffort、FinishReason、ModelUsage 和响应消息类型。
6. 迁移 Provider 直接调用字段、ModelResponse、ModelStreamEvent 和 stream 序列校验。
7. 将 ModelProvider 和 ModelCallError 分别移入 Provider ports 与 errors。
8. 将原 models、tools 和 ports 中相应名称改为兼容性再导出。
9. 更新 Model Router、Provider adapters、Tool Gateway、内置 Tools 和测试 fixtures
   的导入。

### 6.3 验收条件

- Model request/response 的字典结构不变。
- Streaming transcript 的终止事件规则不变。
- Tool result 和 ArtifactReference 的字典结构不变。
- Provider adapter 的请求转换、streaming、Tool call 和错误语义测试通过。
- Tool Gateway、权限、安全和 artifact 测试通过。
- `myclaw/tools` 不导入 `myclaw/provider`。

## 7. PR 3：Session、Memory 与 Schedule

### 7.1 新增文件

- `myclaw/session/records.py`
- `myclaw/session/models.py`
- `myclaw/session/ports.py`
- `myclaw/memory/records.py`
- `myclaw/memory/models.py`
- `myclaw/memory/ports.py`
- `myclaw/schedule/records.py`

### 7.2 迁移任务

1. 迁移 SessionMetadata、UserSessionMessage、AssistantSessionMessage 和
   ToolSessionMessage。
2. 迁移 CumulativeUsage、SessionError、SessionMessage、ConversationSession、
   MetadataUpdate 和 SessionSummary。
3. 将 ResumeResult 移入 Session models，将 SessionStore 移入 Session ports。
4. 迁移 SummaryEntry、MemoryTaskResult、SummaryStore 和 MemoryStore。
5. 迁移 ScheduledWork 及完整 JSON 数组序列化。
6. 更新 Session Store、Conversation、Conversation Summary、Memory Task、Scheduled
   Work persistence 和 execution 的导入。
7. 将原 sessions、memory、scheduling 和 ports 中相应名称改为兼容性再导出。

### 7.3 验收条件

- Session metadata 和三种 message JSONL 输出逐字段不变。
- Session title、cursor、usage 和损坏恢复规则不变。
- Summary JSONL 仍只有 index、timestamp 和 content。
- Consolidation recovery 测试通过。
- Scheduled Work JSON 数组结构和校验规则不变。
- Session、Memory 和 Scheduling 测试全部通过。

## 8. PR 4：Agent Event 与 Management Port

### 8.1 新增文件

- `myclaw/agent/events.py`
- `myclaw/agent/ports.py`
- `myclaw/config/models.py`
- `myclaw/management/models.py`
- `myclaw/management/ports.py`

### 8.2 迁移任务

1. 迁移 AgentEventType、所有 payload、AgentEvent 和事件序列校验。
2. 将 ConversationPort 移入 Agent ports。
3. 将 ConfigView 移入 Config models。
4. 将 RuntimeStatus 移入 Management models。
5. 将 ManagementPort 移入 Management ports。
6. 让 ManagementPort 直接引用 ConfigView、ResumeResult、MemoryTaskResult、
   RuntimeStatus 和 SessionSummary 的所有者模块。
7. 更新 Runtime、Conversation、REPL、Management services、commands、Session resume
   和后台协调模块的导入。
8. 将原 events、management 和 ports 中相应名称改为兼容性再导出。

### 8.3 验收条件

- Agent Event envelope 和每种 payload 的字典结构不变。
- foreground terminal event 和 background completion 排序规则不变。
- REPL 只通过 Conversation Port 和 Management Port 访问运行时能力。
- Protocol runtime check、Runtime、REPL、Management 和 Background Coordination
  测试通过。

## 9. PR 5：移除兼容层并更新文档

### 9.1 清理任务

1. 将剩余生产代码、fixtures 和测试全部改为从约束所有者直接导入。
2. 将 `tests/contract/` 中的测试迁移到对应领域测试目录。
3. 增加基于 Python AST 的架构测试，禁止生产代码导入 `myclaw.contracts`。
4. 增加包边界检查，避免 `__init__.py` 跨领域聚合不相关类型。
5. 删除 `myclaw/contracts/` 及兼容性测试。
6. 更新 implementation plan 和 release-readiness 中的源码路径与测试追踪关系。
7. 检查 README 和其他文档，不得保留已删除路径。

### 9.2 最终验收

```powershell
rg "myclaw\.contracts" myclaw tests docs
python -m pytest
python -m ruff check .
python -m mypy
python -m build
```

完成时应满足：

- `myclaw` 和 `tests` 中不存在旧 contracts 导入。
- 文档中不存在失效的 contracts 源码链接；历史说明文字可以保留，但必须明确其
  历史性质。
- 全量测试、lint、类型检查和构建通过。
- 持久化 fixtures、ErrorCode 集合、Agent Event fixtures 和 Protocol 行为没有变化。
- 没有循环导入或跨领域聚合 barrel。

## 10. 测试迁移建议

| 原测试 | 建议目标 |
| --- | --- |
| common contracts | Session identifiers、utils validation/time 测试 |
| error contracts | 根级 errors 测试 |
| model contracts | Provider models 测试 |
| tool contracts | Tools models/artifacts 测试 |
| session contracts | Session records 测试 |
| memory/scheduling contracts | Memory records 与 Schedule records 测试 |
| event contracts | Agent events 测试 |
| management contracts | Management models/ports 测试 |
| protocol contracts | 各领域 ports 测试及一个跨模块架构测试 |

测试应继续验证外部可观察行为，不应仅验证类位于某个文件。文件归属通过独立的
AST 架构测试约束。

## 11. 风险与控制

| 风险 | 控制措施 |
| --- | --- |
| 新旧路径定义了两个不同 dataclass | 旧路径只再导出新对象，并测试对象 identity |
| Session 与 Tools 循环依赖 | Tools 只依赖 Session identifiers；ModelToolCall 由 Tools 所有 |
| Provider 与 Tools 循环依赖 | Provider models 依赖 Tools models，Tools 不导入 Provider |
| 顶层 `__init__.py` 形成新的 contracts | 使用直接模块导入和 AST 架构测试 |
| 机械移动意外改变序列化 | 每个 PR 运行 exact-shape contract tests |
| Protocol 移动破坏 fake 实现 | 保留 runtime-checkable 和结构替换性测试 |
| 文档链接失效 | 最终 PR 搜索旧路径并更新 release traceability |
| 已发布的 Python 导入 API 被破坏 | 若需要外部兼容，保留一个 minor version 的弃用再导出层 |

## 12. 范围外事项

- 修改任何持久化 schema 或 schema version。
- 增删 ErrorCode 或改变错误重试语义。
- 修改 Agent Event、Port 方法或 Runtime Core 编排行为。
- 修改 Permission Policy、Tool、Provider、Memory、Session 或 Scheduled Work 的产品
  行为。
- 引入依赖注入框架、插件系统、MCP、subagent、daemon、网络 API 或跨进程协调。
- 借机重构与类型迁移无关的实现代码。
