# MyClaw Runtime Contracts

## 文档状态

- 状态：`Accepted`
- 接受日期：`2026-07-11`
- 目标版本：MyClaw `v0.1`
- canonical source：`CONTEXT.md`
- 产品行为来源：`docs/myclaw-personal-agent-prd.md`
- 历史实施顺序记录：`docs/myclaw-implementation-plan.md`

本文把已确认的产品行为细化为可直接实现的类型、文件 schema 和代码边界。来自 canonical 文档的行为与 D01-D16 已于 2026-07-11 一并接受；Issue #118/#130 固化了固定 Core Tool Catalog、BaseTool 管线、Workspace State 访问、Exec/Web/Schedule 行为和 Artifact 边界。本文对应章节以这些已批准规格为准。后续变更必须先更新本契约及受影响的 PRD/ADR。

本文不新增 one-shot、daemon、HTTP/IPC、MCP、subagent、跨进程协调、用户可配置安全策略或用户自定义 identity。

## 1. 契约通则

### 1.1 兼容性

- `config.toml`、Session JSONL、Summary JSONL、Summary Cursor、Long-term Memory、Schedule JSON 和 Tool Artifact 是持久化契约。
- Session JSONL 的第一行是当前严格 header，后续行是 JSON-native message dictionaries；不包含 schema version 或 line type marker。
- 旧 Session schema unsupported；不提供 migration、compatibility reader、lazy upgrade 或 version dispatch。
- summary JSONL 必须严格保持 `index`、`timestamp`、`content` 三个字段，因此不增加版本字段。
- 内部 Python 类型、类名和文件拆分不是持久化契约，可在不改变外部行为的情况下调整。

### 1.2 编码、换行与数值

- 所有文本文件使用 UTF-8，无 BOM。
- JSON/JSONL 使用紧凑 JSON；每条 JSONL record 以单个 `\n` 结束。
- JSON 字段名使用 `snake_case`。
- token、cursor、index、计数和字符数均为非负十进制整数。
- “字符数”统一指 Unicode code point 数量，即 Python `len(str)`，不是 UTF-8 byte 数。
- 持久化路径字段一律使用 `/` 分隔的相对路径；不把用户主目录绝对路径写进 artifact reference。

### 1.3 时间

- 所有持久化时间使用 RFC 3339，精确到毫秒，并带系统本地 UTC offset，例如 `2026-07-11T15:30:12.123+08:00`。
- runtime 内部使用 timezone-aware `datetime`。
- Schedule Tool 的 cron 默认使用 UTC，显式 timezone 必须是规范 IANA 时区；Memory Task 的独立 cron 仍使用 runtime 启动时的系统本地时区。夏令时行为交给选定 cron library，并通过 fake clock 固化测试。
- elapsed time、timeout 和 retry backoff 使用 monotonic clock，不使用 wall clock 差值。

### 1.4 ID

- UUID 均使用小写、带连字符的 UUID4。
- Session ID 使用 `<local_timestamp>_<uuid4>`：`YYYYMMDD-HHMMSS-ffffff_550e8400-e29b-41d4-a716-446655440000`。
- Schedule Job ID 和 turn ID 使用 UUID4；Conversation Session message dictionaries 没有通用 message ID。
- provider 返回的 tool call ID 原样保存在 message 中，不在业务层重新命名。
- Tool Artifact 文件名直接使用只含 ASCII 字母、数字、下划线和连字符的 tool call ID；其他 ID 使用 UUID4。Artifact 路径固定为 Workspace State 下 `.myclaw/artifacts/<session_id>/<id>.txt`。

## 2. Phase 0 已接受决策表

| ID | 已接受的决定 | 理由 | 同步文档 |
| --- | --- | --- | --- |
| D01 | Python 最低版本为 3.12，采用 `pyproject.toml` + 根目录 `myclaw/` 包布局 | asyncio、typing 和 timezone 能力成熟，降低兼容分支 | 实施计划 |
| D02 | Runtime loading 投影掉未知配置字段；`myclaw config` 报告未定义字段；未知 provider protocol 仍按 PRD 忽略 | 保持运行时配置兼容，同时让配置命令暴露拼写错误 | PRD 可补充 |
| D03 | memory message threshold 默认 `40` 条 | 足够早地覆盖长会话，又不会在短对话频繁摘要 | PRD 可补充 |
| D04 | Schedule state 文件名为 `schedule.json` | 与 canonical Schedule module 一致，legacy state 不参与升级 | PRD/Issue #116 |
| D05 | Session title 最长 `60` 个 Unicode code points | picker 可读且不需要终端宽度参与持久化 | PRD 可补充 |
| D06 | Summary index 从 `1` 开始，缺失 `.cursor` 等价于 `0` | cursor 语义是“已处理到的最大 index”，直观且易恢复 | PRD 可补充 |
| D07 | model retry 的“最多 5 次”解释为每个逻辑 model call 最多 `5 attempts`，不是首次调用后再重试 5 次 | 限制最坏延迟和费用，与英文 canonical 的 five attempts 一致 | PRD |
| D08 | route fallback 发生在具体 route 缺失、配置不可用或返回永久 route/provider 不可用错误时；同一 route 的临时错误先在 5-attempt budget 内重试 | 避免两个 route 各重试 5 次导致不可控延迟 | PRD |
| D09 | token estimate 使用 `ceil(UTF-8 byte length / 4)`，展示估算输入 token、context window 和占比 | provider-neutral，明确它只是估算值 | PRD 可补充 |
| D10 | WebSearch 使用无凭据的内置 adapter，首选 DuckDuckGo；后端和 enablement 不进入持久化配置契约 | 固定 Catalog 保持可预测，adapter 可替换 | PRD |
| D11 | Exec 使用单次 Bash 和固定 destructive/DNS safety checks；没有 allowlist，也不宣称 OS sandbox | 规则可审计，同时不把 cwd 检查误称为进程隔离 | PRD/ADR-0010 |
| D12 | Session 只接受完整当前 JSONL snapshot；缺少尾随换行、非法 header/message 或旧 schema 都拒绝加载，不做尾行修复 | 完整 atomic replacement 简化 Session authority；不静默修复或迁移不支持的历史格式 | ADR-0009 |
| D13 | Conversation Summary 直接更新 Session 的 `last_consolidated`，不创建 pending journal 或跨文件恢复协议 | 接受 Summary 与 Session snapshot 在 crash 后 divergence，以保持 Session 接口小且无 persistence acknowledgement | ADR-0009 |
| D14 | Tool Artifact 由 BaseTool 在成功结果超过全局 `4096` 字符阈值时直接写入 `.myclaw/artifacts/<session>/<id>.txt`，preview 受同一阈值限制 | 让模型上下文有统一硬上限，同时保留完整结果 | PRD/ADR-0010 |
| D15 | 配置、持久化、模型与服务级错误使用稳定 error code；`ToolError` 与 Tool Result 仅使用安全 message | Tool 消息与模型消费格式保持扁平，其他上层逻辑仍不依赖易变终端文案 | 实施契约/Issue #38 |
| D16 | 首版 Exec 不宣称提供 OS 级文件系统/网络或进程隔离；只对 cwd、destructive 命令和 URL DNS 做具体检查 | 仅校验 cwd 无法约束 Bash 子进程访问绝对路径，不能制造虚假的 sandbox 保证 | PRD/ADR-0010 |

D01-D16 均为首版实现契约，其中 D04、D07、D08、D10、D11、D12、D13、D16 是已显式接受的产品或风险边界；Session snapshot 细节由 ADR-0009 补充。

## 3. Agent Home 与 Workspace

### 3.1 固定布局

```text
~/.myclaw/
  config.toml

<workspace>/.myclaw/
  .gitignore
  schedule.json
  memory/
    memory.md
    summary.jsonl
    .cursor
  sessions/
    <session_id>.jsonl
  artifacts/
    <session_id>/
      <tool_call_id_or_uuid4>.txt
  schedule-sessions/
    <schedule_session_id>.jsonl
  logs/
    <session_id>.log
```

已确定：Agent Home 拥有 User Configuration；其中既有的 `logs/run.log.0`、`run.log.1`、`run.log.cursor` 与 `run.log.lock` 仅作为 legacy Runtime Log 文件逐字节保留，MyClaw 不再读取、移动、删除、截断或更新它们。有效 REPL 启动初始化 Workspace State root、`.gitignore`、`memory/`、`sessions/` 和缺失的 `memory/memory.md`；`summary.jsonl`、`.cursor`、`schedule.json`、Session、`.myclaw/artifacts/`、`schedule-sessions/` 和 `logs/` 按需创建。legacy scheduled-work state 原样保留且不读取、不检测、不迁移、不重命名或删除。`myclaw config` 不初始化 Workspace State。

`memory.md` 初始内容固定为：

```markdown
# Long-term Memory

## User Info

## User Preference

## Project Fact

## Lesson
```

### 3.2 Workspace identity 与 Workspace State

1. Workspace identity 是启动 cwd 按当前 host 原生路径语义进行词法规范化后的绝对路径；不解析 Git root，也不搜索祖先目录或通过 filesystem alias 改变归属。
2. Windows 保留 Drive、UNC 与 extended path 行为；POSIX 使用原生绝对路径语义。相对路径被拒绝为 Workspace identity。
3. Workspace 不派生第二层名称。非全局状态直接位于 `<workspace>/.myclaw/`。
4. Agent Home 仅保留全局 User Configuration 与 untouched legacy Runtime Log files；旧的非全局数据不读取、不迁移、不删除。

### 3.3 Session Log

Session technical diagnostics 位于 `<workspace>/.myclaw/logs/<session_id>.log`。显式 Session context 先验证 Session ID，再按需准备 `logs/`，并注册只接受相同 Session ID 的 Loguru Sink。WARNING 与 ERROR 写入；DEBUG 与 INFO 不写入。第三方标准库 `logging` record 不桥接到 Session Log，只有 MyClaw 边界捕获的异常可以成为 MyClaw Loguru event。

文件 Sink 固定使用 `enqueue=True`、UTF-8、`catch=True`、精确 10,485,760-byte rotation 与最多一个历史文件的 per-Session retention。Sink 初始化 fail-open，下一次 context 重试；context exit 移除 Sink 并无限等待队列排空。

以下风险已明确接受：Loguru queue 无界；正常退出是 infinite drain；没有 per-record fsync，崩溃、断电或强制终止可丢失最近记录；没有 active redaction 或 control escaping，异常消息中的 credential 与控制字符可能原样持久化；retention 仅限单个 Session，Workspace 总日志量无上限；同一 Session 的单进程重叠或跨进程并发不受支持，不检测且不协调。

### 3.4 原子写

本节只约束明确声明 atomic replacement 的持久化 Store；Session Log、Tool Artifact 和用户请求的 file Tool 写入遵守各自章节的非原子契约。

- 新文件或整体更新：在目标同目录创建唯一临时文件，写入完整内容，flush，尽可能 fsync，再 atomic replace。
- 文件内容 flush 后尽可能 fsync；POSIX 在发布后同步 parent directory，host 明确不支持同步时保留可测试的 best-effort 分支。
- Session snapshot：当前 runtime 的 active Session 在 turn 结束后捕获完整 JSON-native state，按 `persist()` 调用顺序异步进行一次 UTF-8 JSONL atomic replacement；取消不得打断已开始的 filesystem operation。
- `Session.close()` 在 shutdown 中对最新非空 state 做最多三次同步 replacement attempt，普通异步失败和 close 最终失败均 silent，不产生 acknowledgement 或 failure log。
- 不创建 Session 跨进程 lock file，不依赖文件锁，不承诺两个 REPL 写同一 Session 的顺序。

## 4. User Configuration 契约

### 4.1 完整 schema

```toml
[runtime]
max_tool_result_chars = 4096

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[models.providers.anthropic-default]
protocol = "anthropic"
base_url = "https://api.anthropic.com"
api_key = "..."
models = ["model-id"]

[models.providers.openai-local]
protocol = "openai-compatible"
base_url = "http://127.0.0.1:8000/v1"
api_key = "..."
models = ["model-id"]

[models.routes.default]
provider_id = "anthropic-default"
model = "model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120

[models.routes.chat]
provider_id = "anthropic-default"
model = "model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
timeout = 120

[models.routes.memory]
provider_id = "anthropic-default"
model = "model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
timeout = 120

[models.routes.schedule]
provider_id = "anthropic-default"
model = "model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
timeout = 120
```

`models.routes.chat`、`models.routes.memory` 和 `models.routes.schedule` 均可省略，省略时使用 default。`reasoning_effort` 可省略。

### 4.2 字段规则

| 字段 | 类型/范围 | 默认/要求 |
| --- | --- | --- |
| `runtime.max_tool_result_chars` | integer，`1000..1000000` | 默认 `4096` |
| `memory.consolidation_message_threshold` | integer，`4..10000` | 默认 `40` |
| `memory.batch_size` | integer，`1..1000` | 默认 `10` |
| `memory.schedule` | 5-field cron string | 默认 `0 * * * *` |
| provider ID | `^[a-z0-9]+(?:-[a-z0-9]+)*$` | 必须唯一 |
| `protocol` | string | `anthropic` 或 `openai-compatible`；其他值的 provider 被忽略 |
| `base_url` | absolute HTTP(S) URL | 所有可用 provider 必填、非空 |
| `api_key` | string | plaintext；可为空模板值，不可用于可用 route |
| `models` | unique string array | 模板可空；可用 route 引用的 model 必须存在 |
| route name | table key | 仅 `default`、`chat`、`memory`、`schedule` |
| `provider_id` | string | 必填，必须引用可用 provider |
| `model` | string | 必填，必须在 provider catalog 中 |
| `context_window` | integer，`1024..10000000` | 必填 |
| `max_output` | integer，`1..context_window-1` | 必填 |
| `temperature` | number，`0..2` | 必填 |
| `reasoning_effort` | `low`、`medium`、`high` | 可省略；不支持时 adapter 静默忽略 |
| `timeout` | integer seconds，`1..600` | 必填 |

启动时将未知顶层 table、未知字段和未知 route table 投影掉；`myclaw config` 仍报告这些未定义字段。未知 protocol provider 按 canonical 要求忽略；如果 default 因此不可用，REPL 启动失败。Tool Catalog 不接受用户配置的 enablement 或 replacement。

### 4.3 首次生成模板

缺少 `config.toml` 时，生成模板包含 runtime、memory、一个 ID 为 `openai-local` 的 OpenAI-compatible provider template，以及 `default`、`chat`、`memory`、`schedule` 四个显式但不可用的 route scaffold。Provider 的 `base_url`、`api_key` 和 `models` 均为空。四个 route 初始都指向 `openai-local`，model 使用待替换值，并提供完整的模型限制字段；用户可删除不需要定制的具体 route，使其回退到 default。生成后 `myclaw` 退出，`myclaw config` 则显示脱敏模板；旧配置完全缺少 default route 时，启动错误指出 `[models.routes.default]`。

### 4.4 脱敏

- TOML 成功解析时，所有 provider 的 `api_key` 显示为 `"***REDACTED***"`；空值仍显示为空，便于诊断未配置状态。
- TOML 解析失败时，对匹配 `(?i)^\s*api[-_]?key\s*=` 的行只保留 key、等号和 `"***REDACTED***"`。
- exception、Rich renderable、测试 snapshot 和日志替代输出都不得包含未脱敏 key。
- 首版不支持环境变量引用或系统密钥链。

## 5. Conversation Session 契约

### 5.1 Header and state

第一行必须是：

```json
{"session_id":"20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000","created_at":"2026-07-11T15:30:12.123+08:00","updated_at":"2026-07-11T15:31:02.456+08:00","last_consolidated":0,"metadata":{"title":"MyClaw implementation","token_usage":{"model_calls":0,"input_tokens":0,"output_tokens":0,"total_tokens":0}}}
```

规则：

- `last_consolidated` 是从 `0` 开始的 message boundary，表示前多少条 Session messages 已被 Conversation Summary 覆盖；Short-term Memory 是 `messages[last_consolidated:]`。Conversation Summary 直接赋值，不调用 cursor-specific method，也不通过 journal 与 Session snapshot 协调。
- `metadata` 当前拥有 `title` 与 `token_usage`；`token_usage` 包含主 chat、Tool loop、title、Conversation Summary 和与当前 Session 直接相关的辅助调用。Memory Task 不接收 Conversation Session；Schedule Job 的模型调用计入其 Schedule Session，不计入当前前台 Session。
- `total_tokens` 必须等于 `input_tokens + output_tokens`。provider 未返回某项时该项为 `0`，不得用估算值混入实际 usage。
- 每次成功持久化都是完整 compact UTF-8 JSONL atomic replacement，header 与所有 message lines 一起提交；不存在逐消息写入或 metadata-only rewrite。
- 当前 header 必须恰好包含 `session_id`、`created_at`、`updated_at`、`last_consolidated`、`metadata`。旧 schema unsupported，不 migration、不 version dispatch。

### 5.2 User message

```json
{"role":"user","content":"Help me inspect this project.","timestamp":"2026-07-11T15:30:12.200+08:00"}
```

- `content` 是非空 string；只包含空白的 REPL 输入不创建 message，也不调用模型。`timestamp` 使用 system local timezone 的 ISO 8601 string。
- Runtime Context 是发给模型时临时 prepend 的内容，不写回 `content`。

### 5.3 Assistant message

```json
{"role":"assistant","content":"I will inspect the files.","timestamp":"2026-07-11T15:30:13.000+08:00","tool_calls":[{"id":"call_123","name":"list_dir","arguments":"{\"path\":\".\"}"}],"status":"completed","error":null,"token_usage":{"model_calls":1,"input_tokens":120,"output_tokens":24,"total_tokens":144}}
```

字段规则：

- `content` 是 string，可为空；`tool_calls` 是 array，可为空；两者至少一个非空，除非 `status=error`。通用 message ID 不存在。
- tool call `arguments` 必须保留 provider 的原始 JSON string。Tool Gateway 无法解析时追加 flat `tool` error message，不执行具体 Tool。
- `status` 仅为 `completed`、`interrupted`、`error`。
- `error` 在 `completed` 时必须为 `null`；其他状态为 `{code, message}`，message 必须可安全展示。
- `token_usage` 使用四字段结构 `model_calls`、`input_tokens`、`output_tokens`、`total_tokens`；每个 assistant model result 的 `model_calls` 为 `1`，并累计到 Session metadata 的 `token_usage`。
- streaming 正常完成后才把一条 `completed` assistant message 加入 active Session；`persist()` 在 turn terminal work 之后一次写入完整 snapshot。
- Ctrl+C 时有 partial text 就写 `interrupted` assistant message；没有 partial text 且没有 tool call 时不写空 assistant message。
- 最终 model failure 写 `error` assistant message；恢复对话构建 model context 时省略纯 error message，保留 interrupted partial content 并追加内部中断标记。
- 如果 assistant 已产生 tool calls 后 turn 被取消，必须为每个尚无结果的 call 添加 tool error message。

错误示例：

```json
{"role":"assistant","content":"Partial answer","timestamp":"2026-07-11T15:30:13.000+08:00","tool_calls":[],"status":"interrupted","error":{"code":"turn_cancelled","message":"Turn interrupted by user."},"token_usage":{"model_calls":1,"input_tokens":120,"output_tokens":8,"total_tokens":128}}
```

### 5.4 Tool message

普通结果：

```json
{"role":"tool","content":"CONTEXT.md\ndocs/","timestamp":"2026-07-11T15:30:13.500+08:00","tool_call_id":"call_123","name":"list_dir","status":"success","artifact":null}
```

写入成功：

```json
{"role":"tool","content":"File written successfully.","timestamp":"2026-07-11T15:30:13.500+08:00","tool_call_id":"call_124","name":"write_file","status":"success","artifact":null}
```

字段规则：

- `status` 仅为 `success`、`error`、`refused`。
- tool 参数校验失败、执行异常和未完成 call 都使用 `error`，不新增 role；这些字段仍作为 JSON-native tool message 保存在完整 Session snapshot 中。
- tool result 会回传模型，因此 `content` 必须是 normalized、可读的 UTF-8 text，不包含 Python repr 或 traceback。
- Tool Gateway 不重试 Tool 执行；`ToolError` 是可见的安全 error，普通异常被记录一次并规范化，取消继续向上传播。
- 旧的 structured arguments 和 nested Tool error JSONL shape 不兼容且不恢复读取。

### 5.5 Session title

固定规则：

1. 首条 user message 加入 active Session 后异步使用 chat route 生成 title，不阻塞首轮 terminal response 或 end-of-turn snapshot。
2. 模型输出取第一条非空行，去除成对引号和首尾空白，内部连续空白折叠为一个空格，截断到 60 个 Unicode code points。
3. 模型失败、空输出或非法输出时，对首条 user content 应用同样的规范化和截断。
4. fallback 仍为空时使用 `Untitled session`。
5. title 调用 usage 计入 Session metadata，但不新增 message；late title 可能只存在于内存，直到后续 turn 或 `close()` 再落盘。

### 5.6 格式边界与失败语义

- 第一行缺失、字段不是当前严格五字段 header、日期不是带 offset 的 ISO 8601、`last_consolidated` 为负数或 metadata 不合法：Session 不可恢复。
- 任一 message line 非法、包含旧 line-marker/version fields、缺少 trailing `\n`，或文件为空：Session 不可恢复，不静默跳过、不做 partial-line repair，也不自动迁移。
- `Session.load()` 是同步且严格的当前格式读取；picker 跳过不可恢复 Session，同时显示一条汇总警告；不得把损坏文件自动删除。
- `persist()` 不等待 filesystem operation，不返回 task、acknowledgement 或 failure；普通写入异常不产生 `OutboundMessage`、Session Log 或其他诊断记录。
- `close()` 标记 Session closed 后抑制过期异步 snapshot，最多同步尝试三次（间隔 `100 ms`、`200 ms`），最终失败静默吞掉。

## 6. Conversation Summary、Cursor 与 Long-term Memory

### 6.1 Summary entry

每行严格只有三个字段：

```json
{"index":1,"timestamp":"2026-07-11T16:00:00.000+08:00","content":"The user is implementing MyClaw and prefers a file-first architecture."}
```

- `index` 全局严格递增，从 `1` 开始。
- `content` 非空。
- 不保存 source session、message ID、message range、route 或 usage。
- Summary entries use the runtime's global summary lock and remain a separate ordered stream from Session snapshots。

### 6.2 Summary Cursor

- `memory/.cursor` 内容是一个非负 ASCII decimal integer 和尾随 `\n`，例如 `12\n`。
- 文件缺失等价于 `0`。
- cursor 表示 Memory Task 已成功处理的最大 summary index。
- no update 或 edit success 后原子写入 batch 的最后 index；required edit failure 不写。

### 6.3 Summary and `last_consolidated` consistency

Conversation Summary generation appends its one summary entry under the
single-runtime Summary lock, then directly assigns the active Session's
`last_consolidated`. There is no pending journal, startup recovery path, or
cross-file transaction between `summary.jsonl` and Session JSONL. A crash or
failed Session snapshot may leave the summary entry and `last_consolidated`
divergent; subsequent work may therefore repeat or omit a summary range. This
is accepted by ADR-0009 and does not provide cross-process coordination.

### 6.4 Long-term Memory cache

- runtime startup 原子创建缺失模板并读取一次，保存 immutable string snapshot。
- chat 和 schedule 的 System Prompt 使用该 snapshot。
- `/memory` 和 Memory Task 每次读取磁盘最新文件。
- Memory Task 成功编辑后刷新 runtime snapshot；正在运行的 Agent Run 保持启动时快照。
- system-level prompt 超过 route context budget 时返回 `memory_context_too_large`，不得裁剪 Long-term Memory。

## 7. Schedule 契约

固定文件名：`<workspace>/.myclaw/schedule.json`。legacy
`scheduled-work.json` 无论内容或 path type 都不读取、不检测、不迁移、不重命名或删除。

顶层必须是严格 JSON array；文件缺失或 `[]` 表示空状态。每条 Job 恰好包含：
`job_id`、`source`、`message`、`schedule`、`state`、`created_at_ms` 和 `updated_at_ms`。

```json
[
  {
    "job_id": "550e8400-e29b-41d4-a716-446655440000",
    "source": "user",
    "message": "Review the current project and summarize open risks.",
    "schedule": {
      "kind": "cron",
      "at_time": null,
      "every_seconds": null,
      "cron_expr": "0 9 * * 1",
      "timezone": "Asia/Shanghai"
    },
    "state": {
      "last_finished_at_ms": null,
      "last_status": null,
      "last_error": null
    },
    "created_at_ms": 1783776000000,
    "updated_at_ms": 1783776000000
  }
]
```

规则：

- `source` 只能是 `user` 或 `system`；System Job 不进入公开 list/remove。
- `schedule.kind` 只能是 `at`、`every` 或 `cron`；未选择的 schedule 字段必须显式为 `null`。
- `at` 使用带 UTC offset 的 RFC 3339 毫秒时间；`every` 使用正整数秒；`cron` 使用规范五字段表达式和规范 IANA timezone。
- `state` 只记录 `never-run`、`latest-ok` 或 `latest-error` 允许的字段组合；running、next run 和 history 不持久化。
- 解析拒绝重复 object key、非 canonical 值、非法 schedule/state、重复 Job ID 和非精确 field set；任何损坏都阻止 Runtime 启动并输出 `schedule_state_error` 与路径。
- Store 在 Runtime-local lock 内构造 immutable candidate，严格序列化后通过同目录 atomic replacement 发布；写失败保留旧 authority 并 fault Store。
- User Job 的 add/list/remove 由 `schedule` Tool 管理且不请求确认；Schedule Agent context 拒绝 add，list/remove 仍可用，公开定义按创建时间和 Job ID 稳定排序。
- 每次触发通过共享 Agent Run 的 `schedule` route 运行；Schedule Session ID 派生为 `schedule_<job_id>`，Session 首次产生消息时才落盘到 `schedule-sessions/`。
- Schedule Service 在一个 dispatcher 中处理动态变更、at/every/cron、重叠跳过、不同 Job 并发和 Runtime shutdown；不发送前台 `OutboundMessage` 或通知。

## 8. Prompt、Runtime Context 与预算

### 8.1 Chat 与 Schedule System Prompt

chat 和 schedule 的 system-level context 按以下固定顺序组装：

1. 内置 identity prompt，其中包含 normalized absolute Workspace。
2. 完整的 runtime-startup Long-term Memory snapshot，以明确的 `<long_term_memory>` delimiter 包裹。
3. 固定 Tool Catalog 的 guidance，以明确的 `<tool_guidance>` delimiter 包裹。

User Configuration 不得插入或替换 identity/system prompt。缓存的 OpenAI-format Tool schema snapshots 通过 provider 的结构化 tools 字段发送，不把 JSON schema 重复拼入自然语言 guidance。

### 8.2 当前 user input 的 Runtime Context

发给 chat/schedule model 的当前 user message临时转换为：

```text
<runtime_context>
current_time: 2026-07-11T15:30:12.123+08:00
session_id: <session_id>
</runtime_context>

<user_input>
<raw user content>
</user_input>
```

- session JSONL 只保存 raw user content，不保存上述 wrapper。
- 历史 user messages 不重复添加新的 Runtime Context。
- Workspace 已在 identity prompt 中，不在每轮 wrapper 重复。
- Schedule Job 不额外注入旧任务 ID 字段；Memory Task 使用专门 prompt，不伪装成 chat user input。

### 8.3 专用 prompts

- Session title：只接收规范化后的首条 user content，不注入 Long-term Memory、tools 或 conversation history。
- Conversation Summary：只接收本次选中的早期 Session messages，不注入 Long-term Memory 或 Tool Catalog。
- Memory Task：接收 Summary Cursor 后的 batch 和四分区维护规则，并只暴露 restricted memory tools。
- Schedule Job：使用 chat/schedule system composition，把 Job message 作为 Schedule Session 的普通 user message。

prompt 文本存放在独立、可版本追踪的 package resources；测试断言组成部分和是否注入，不锁死整段自然语言文案。

### 8.4 Context budget 与 consolidation

- 可用输入预算为 `context_window - max_output`。
- 估算对象包含 system prompt、retained session messages、当前 Runtime Context、user input 和结构化 tool definitions。
- 在每次 chat route model call 前检查预算和 `consolidation_message_threshold`，包括一个 tool loop 中后续的 chat model call。
- consolidation 只能选择当前 turn 之前的早期消息，不得拆走正在执行的 assistant/tool call chain。
- token 触发先选择约输入预算一半的早期消息；message threshold 触发先选择约 threshold 一半的早期消息，再按 canonical 规则把 retained suffix 对齐到 user message。
- System Prompt 自身超过预算时直接返回 `memory_context_too_large` 或相应配置错误，不裁剪 Long-term Memory。
- 没有足够的完整历史 turn 可供 consolidation 时返回 `model_context_overflow`，不得丢弃当前 user message或 tool chain。

## 9. Model Contracts

### 9.1 Route resolution

```text
logical purpose -> requested route -> usable route config -> provider adapter
                    | unavailable
                    +-----------------------> default route
```

- route purpose 是 `default | chat | memory | schedule`。
- chat 用于主对话和 title；memory 用于 summary 和 Memory Task；schedule 用于 Schedule Jobs。
- chat request 必须调用 streaming provider contract；memory/schedule 可调用 complete contract。
- requested route 与 default 指向同一配置时只尝试一次。
- default 不可用时 runtime startup 失败。

固定失败/fallback 顺序：

1. route 缺失、provider 被忽略、model 不在 catalog等静态不可用：直接使用 default。
2. route 调用发生 rate limit、timeout 或 unavailable：在总共 5 attempts 内对同一 route 指数退避。
3. route 返回 auth、model-not-found 或 unsupported 等永久不可用错误：若不是 default，切到 default，并只使用剩余 attempt budget。
4. invalid request、context overflow、cancellation 和本地 schema error 不 fallback。

一个逻辑 model call 在 requested route 和 default 之间共享 5-attempt budget，最坏不超过 5 次 provider 调用。若第 5 次才暴露永久错误，则没有剩余预算调用 default，当前逻辑调用失败。

### 9.2 Provider 直接调用

```python
class ModelProvider(Protocol):
    def stream(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: Literal["low", "medium", "high"] | None,
        timeout: int,
    ) -> AsyncIterator[ModelStreamEvent]: ...

    async def complete(
        self,
        *,
        messages: Sequence[dict[str, object]],
        tools: Sequence[OpenAIToolSchema],
        model: str,
        max_output: int,
        temperature: float,
        reasoning_effort: Literal["low", "medium", "high"] | None,
        timeout: int,
    ) -> ModelResponse: ...
```

Router 只在调用边界接收逻辑 route、message dictionaries 和 Tool schemas，并负责 route resolution；具体 Provider 只接收上面列出的已解析字段。调用方通过 `stream` 或 `complete` 方法选择同步/流式语义，不携带请求 ID，也不把 route 传给 concrete Provider。`OpenAIToolSchema` 是由 Tool Catalog 缓存的 OpenAI Function Calling 格式快照。OpenAI-compatible adapter 直接传递该格式；Anthropic adapter 在内部转换字段。Provider adapter 不自行 fallback、不读取 User Configuration、不处理 Session。

### 9.3 统一响应与 streaming

```python
ModelResponse(
    message: AssistantModelMessage,
    usage: ModelUsage,
    finish_reason: Literal["stop", "tool_calls", "length", "cancelled"],
)

ModelUsage(
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
)
```

成功的 `ModelResponse` 必须包含非空白 `message.content` 或至少一个 Tool call；两者都没有时，provider adapter 将响应规范化为不可重试的 `model_failed`。

streaming contract 只向 Runtime Core 暴露：

- `text_delta(delta)`：非空 text chunk，按顺序到达。
- `completed(response)`：恰好一次，包含聚合后的 content、完整 tool calls 和 usage。
- exception：在 `completed` 前终止 stream，由 router 转成统一错误。

adapter 内部负责聚合 provider-specific tool call deltas。Runtime Core 不解析 Anthropic content block 或 OpenAI chunk。

### 9.4 Retry

- 一个逻辑 model call 默认最多 5 attempts。
- delay 为 `min(30, 0.5 * 2^(attempt-1))` seconds，并加入可注入、可关闭的 bounded jitter。
- 有合法 `retry_after` 时使用 `max(backoff, retry_after)`，再限制到 60 seconds。
- auth、invalid request、context overflow、unsupported、cancelled 不 retry。
- rate limit、timeout、connection error、provider unavailable 可 retry。
- tool calls、title fallback 本地处理和 Tool Gateway 不复用 model retry。

## 10. MessageBus、AgentLoop 与 AgentRunner 契约

### 10.1 Public foreground boundary

`AgentLoop` owns one foreground `MessageBus`. Terminal, headless REPL and
other foreground consumers submit `InboundMessage` values and consume
`OutboundMessage` values; they do not reach into Session, provider or Tool
objects. The independent `AgentLoop.control` surface owns cancellation,
confirmation callbacks and the current foreground projection. Management Port
and its dispatcher remain a separate management boundary.

### 10.2 AgentRunner

`AgentRunner` is the bounded, Session-independent model/Tool execution shared
by the foreground loop and Schedule. It receives the selected route and Tool
Gateway dependencies and returns an `AgentRunnerResult`. One Agent Run remains
the domain term for one Runner execution; it is not a transport or a public
event envelope. Repair construction, Tool Result externalization and iterator
cleanup are private implementation helpers and are not package interfaces.

### 10.3 Sparse outbound protocol

`OutboundMessage` has one of the following types:

| type | metadata/use |
| --- | --- |
| `model_reasoning` | streamed reasoning deltas and `_stream_end` |
| `model_response` | streamed response deltas, `_stream_end`, or `_streamed` completion |
| `tool_call` | a Tool call projection with `tool_call_id` and raw `arguments` |
| `system_control` | a `_streamed` terminal marker with `finish_reason` such as `cancelled`, `failed` or `max_iterations` |

Foreground streaming may contain multiple delta messages, but each completed
foreground run has exactly one terminal `_streamed` marker. Tool results and
confirmation replies are not foreground bus payloads. A Tool confirmation is
delivered through the `AgentLoop.control` callback and resolved by its direct
Future-bound response; wrong, late and duplicate decisions do not authorize a
different call. Schedule uses the same Runner/Gateway execution through
`AgentLoop.run_schedule_job` and does not publish foreground messages.

### 10.4 Management Port

最小接口：

```python
class ManagementPort(Protocol):
    async def config_view(self) -> ConfigView: ...
    async def status(self) -> RuntimeStatus: ...
    async def resumable_sessions(self) -> tuple[SessionSummary, ...]: ...
    async def resume(self, session_id: str, *, force: bool = False) -> ResumeResult: ...
    async def memory_view(self) -> str: ...
    async def dream(self) -> MemoryTaskResult: ...
```

- `resumable_sessions` 仅返回当前 Workspace 的 id、title、created_at、updated_at、message_count。
- `resume` 再次验证 session 属于当前 Workspace，不信任 UI 传入值；`force` 仅由
  Terminal 在确认 active foreground replacement 后传入，Management 不构造 Runtime。
- `config_view` 已脱敏；Management Port 永不返回 plaintext API key。
- `memory_view` 读取磁盘，不返回 runtime cache。

## TOOL_SCHEMA：Tool Gateway 契约

### 11.1 Tool schema、调用与结果

```python
OpenAIToolSchema = {
    "type": "function",
    "function": {"name": str, "description": str, "parameters": JsonObject},
}

ModelToolCall(id: str, name: str, arguments: str)  # arguments 是原始 JSON 文本

ToolResult(
    tool_call_id: str,
    name: str,
    status: Literal["success", "error", "refused"],
    content: str,
    artifact: ArtifactReference | None,
    confirmation: ToolConfirmationMetadata | None,
)
```

`BaseTool.to_schema()` 从具体 Tool 的固定 `parameters` Schema 生成 detached OpenAI Function Calling schema，具体 Tool 不得覆盖；普通 Tool 的 Schema 由公开注解、显式 `required`、默认值和 `ToolParam` 派生，复杂的固定 Schema 可由 Tool 显式提供。`ToolGateway.call()` 是唯一公开调用入口，顺序固定为：parse raw JSON -> resolve -> final BaseTool cast/Schema validation/argument validation/safety check -> one-shot confirmation when required -> execute -> normalize。未声明参数被忽略；只允许规格定义的 string-to-integer、integral-float-to-integer 和 string-to-boolean 转换。Tool 不重试，解析、准备、确认和拒绝也不重试；取消继续向上传播。

### 11.2 Catalog 与依赖所有权

- Runtime Core 在启动时以 Workspace、具体 Schedule Store 和 scheduled-agent 标志构造 `ToolGateway`；Gateway 自己一次性构造固定十工具 Catalog，不暴露注册入口。
- Tool 调用不接收 session ID、Agent Home、lane、approval flag 或通用 execution context。
- 没有独立 `Security` 模块；公共路径、DNS、截断和 Artifact 边界由 BaseTool 或共享小 helper 提供，具体 Tool 保留 capability-specific 规则。
- Memory Task 使用标准 Tool Gateway 和仅含专用 Long-term Memory read/edit Tool 的 catalog。
- Tool Gateway 不在前台/后台之间加全局执行锁。
- Tool Gateway 不设置统一 timeout、持久化 Tool Result 或持有 Workspace；Artifact 写入由 BaseTool 的结果处理能力完成。

### 11.3 内置 file tools

固定名称和 input schema：

| Tool | 参数 | 行为 |
| --- | --- | --- |
| `read_file` | `path: str`, `offset: int = 1`, `limit: int = 2000` | 读取 UTF-8 文本行；二进制/解码失败返回 error |
| `list_dir` | `path: str = "."`, `recursive: bool = false`, `max_entries: int = 200` | 稳定排序的相对路径列表，目录以 `/` 结尾 |
| `glob` | `pattern: str`, `path: str = "."`, `head_limit: int = 200`, `offset: int = 0`, `kind: str = "files"` | 使用固定 Glob dialect 的路径列表 |
| `grep` | `pattern: str`, `path: str = "."`, `glob: str | null`, `type: str | null`, `output_mode: str = "content"`, `context: int = 0`, `head_limit: int = 0`, `offset: int = 0`, `fixed_string: bool = false`, `ignore_case: bool = false` | 通过 Python regex 的文本搜索 |
| `write_file` | `path: str`, `content: str` | 直接写入精确 UTF-8 文本，外部路径需要确认 |
| `edit_file` | `path: str`, `old_text: str`, `new_text: str`, `replace_all: bool = false` | 精确文本替换，外部路径需要确认 |

参数上限：`limit 1..10000`、`max_entries 1..10000`、`head_limit 0..1000`。`old_text` 为空无效；`replace_all=false` 时匹配数必须恰好为 1。

主 Agent file access：

- Workspace 内 read/list/search：allow。
- Workspace 及其中 Workspace State 的 read/list/write/edit：allow，实际结果服从操作系统权限。
- 解析到 Workspace 外的 file path：请求一次性确认；无确认通道或拒绝时 refused。
- 当前 session 的 artifact directory：按相同 Workspace 路径规则处理，没有额外 MyClaw 权限层。

共享路径 helper 使用 host path semantics 解析 Workspace root 和请求目标，再按 canonical path 判断是否在 Workspace 内；内部目标直接交给具体 OS 操作，外部目标先确认。没有中央 `Security` 类型筛选或额外 device/named-pipe/non-regular blanket policy。

### 11.4 Exec Tool

```json
{
  "name": "exec",
  "arguments": {
    "command": "git status --short",
    "cwd": ".",
    "timeout": 60
  }
}
```

- `timeout` 必须在 `1..600` 秒，默认 `60`。
- `cwd` 缺省为 Workspace，解析到 Workspace 外时请求确认。
- Exec 通过异步子进程启动一次 Bash；不提供 persistent session、PTY、stdin、streaming、PowerShell、CMD、allowlist 或 OS sandbox。
- 已知递归/强制删除、磁盘格式化、直接磁盘写入、关机和 fork bomb 形状请求确认；HTTP(S) URL 的 DNS 失败或非 global 地址同样请求确认。
- 输出包含 exit code 和非空 stdout/stderr block；非零退出仍是 successful text result。超出 4000 字符时按共享 prefix-plus-marker 规则截断。

### 11.5 Web tools

`web_search`：

```json
{"query":"MyClaw agent runtime","count":5}
```

- `count` 范围 `1..10`，默认 `5`。
- normalized result 是按序号分隔的 text block，每项包含 `Title`、`URL`、`Snippet` 三行。
- 无结果返回空 string，网络失败返回 tool error；同步 DDGS 调用经 worker thread 执行，单次调用 timeout 为 30 秒且不重试。

`web_fetch`：

```json
{"url":"https://example.com/page"}
```

- 仅允许 `http` 和 `https`，禁止 URL userinfo。
- 请求前解析全部 IP；DNS 失败或任一地址属于 loopback、private、link-local、unspecified、multicast 或 reserved 时请求一次性确认，无确认通道或拒绝时 refused。
- 公网目标先尝试 Jina Reader，失败或空结果才 direct fetch；已确认的非公网目标跳过 Jina Reader。
- 每次 redirect 重新执行 scheme、hostname、DNS/IP 检查，最多跟随 5 次；需要确认的新 redirect target 必须作为新的独立调用确认。
- 不承诺 pin 实际 peer 或防止 DNS rebinding；每个可见 URL 仍在请求前执行 DNS/IP 公网检查。
- connect timeout 10 秒、整个调用 total timeout 30 秒；direct response 不设独立 raw-byte 上限，最终输出由 `maxChars` 限制，默认 `50000` 字符。
- HTML 转为可读 text；其他 textual media 保留 text；缺失 Content-Type 按 text 处理，明确二进制 media 返回 unsupported media error。
- WebSearch 和 WebFetch 始终在固定 Catalog 中，不受 User Configuration enablement 控制。

### 11.6 Schedule Tool

```json
{
  "name": "schedule",
  "arguments": {
    "action": "add",
    "message": "Review the project and summarize open risks.",
    "cron_expr": "0 9 * * 1",
    "timezone": "Asia/Shanghai"
  }
}
```

- `action` 必须精确为 `add`、`list` 或 `remove`。
- `add` 按 `every_seconds`、`cron_expr`、`at_time` 顺序选择 schedule；低优先级和 action 无关字段忽略，仅校验所选字段。
- cron 默认 `UTC`，显式 timezone 必须是规范 IANA zone；at-time 接受带 timezone 的 ISO 输入并规范化到 RFC 3339 毫秒。
- `list` 和 `remove` 不需要确认；list 只返回 user Job，remove 只处理 canonical UUID4 的 user Job。
- Scheduled Agent context 拒绝 `add`，但允许 `list`/`remove`；Schedule Tool 不请求 Tool Confirmation。

### 11.7 Tool Artifact

BaseTool 在 `status == "success"` 且结果长度严格超过 `runtime.max_tool_result_chars` 时：

1. 把完整 raw content 直接以 UTF-8 写入 `.myclaw/artifacts/<session_id>/<id>.txt`，创建父目录并允许覆盖。
2. 返回原始结果的 prefix 加 marker，marker 计入全局限制；不保留尾部：

```text

...[truncated; full result stored at .myclaw/artifacts/<session_id>/<id>.txt]
```

3. `artifact` 写为：

```json
{"path":".myclaw/artifacts/<session_id>/<id>.txt","total_chars":73421,"preview_chars":4050}
```

合法 call ID 直接作为文件名，其他 ID 使用 UUID4。Artifact 写失败时保留成功状态，返回受限制的 prefix 加 artifact-write-failed marker；不回退写入完整 raw content。没有 atomic、identity、commit、rollback 或 cleanup 生命周期；文件写成功后若 Session persistence 失败，允许留下 orphan artifact。

## 12. Fail-closed capability 矩阵

| Capability | Foreground | Schedule Job | Memory Task |
| --- | --- | --- | --- |
| Workspace read/list/search | allow | allow | 不在 catalog |
| Workspace write/edit | allow, subject to OS permissions | allow, subject to OS permissions | 不在 catalog |
| Long-term Memory read | allow | allow | allow |
| Long-term Memory edit | allow, subject to OS permissions | allow, subject to OS permissions | allow，仅精确文件 |
| Current-session artifact read | allow | allow | 不在 catalog |
| Agent Home internal state read/write | 按普通 Workspace path rules | Workspace 内 allow；外部 refused/error | 不在 catalog；owned stores 自行操作 |
| Exec | allow or one-shot confirmation by concrete safety check | allow；需要确认时 refused | 不在 catalog |
| WebSearch/WebFetch | allow or one-shot confirmation by concrete target check | allow；需要确认时 refused | 不在 catalog |
| Schedule add/list/remove | allow | allow, except `add` is refused | 不在 catalog |
| Workspace 之外 | one-shot confirmation or OS error | refused/error（无确认通道） | refused/error |

当前 Tool 契约没有 centralized Permission Policy、`ask`、approval flag 或 invocation ContextVar。前台 Exec、Web 和 Workspace 外部路径的确认是一次性、精确绑定到当前 normalized call 的 Tool Confirmation；Schedule Agent 没有确认通道，因此需要确认的操作会被拒绝，而 Schedule actions 自身不请求确认。无效参数、越界访问和执行失败返回 message-only `error`。Tool Result 不携带 error code 或嵌套 `ErrorInfo`，可携带 confirmation metadata。

## 13. Error Contract

### 13.1 稳定结构

```python
ErrorInfo(
    code: str,
    message: str,
    retryable: bool = False,
    retry_after_seconds: float | None = None,
)
```

`ErrorInfo` 仍用于 model、Agent Run 和 service-level error contract，不用于 `ToolError` 或 Tool Result。`ToolError` 只有安全的 message；cause、traceback、SDK response body 不写 Tool message 或 foreground `OutboundMessage`，但在拥有明确 Session 的 MyClaw 边界可进入 Session Log。Session Log 不执行主动脱敏。

### 13.2 稳定 code

| code | 用途 | model retry |
| --- | --- | --- |
| `config_missing` | 已生成模板，需要编辑 | 否 |
| `config_parse_error` | TOML 语法错误 | 否 |
| `config_invalid` | schema/route 无效 | 否 |
| `persistence_error` | 原子写、append、损坏文件 | 否 |
| `route_unavailable` | requested/default route 不可用 | 按 D08 |
| `provider_auth_error` | API key/认证失败 | 否 |
| `provider_rate_limited` | provider 限流 | 是 |
| `provider_timeout` | model timeout | 是 |
| `provider_unavailable` | 网络/服务临时不可用 | 是 |
| `model_invalid_request` | provider 拒绝请求 | 否 |
| `model_context_overflow` | request 超 context | 否 |
| `memory_context_too_large` | system prompt/Long-term Memory 超预算 | 否 |
| `model_failed` | 无更具体映射的最终失败 | 否 |
| `turn_cancelled` | Ctrl+C 或 shutdown cancellation | 否 |
| `tool_not_found` | 未注册工具 | 否 |
| `tool_invalid_arguments` | JSON/schema 参数无效 | 否 |
| `tool_denied` | 内置 policy 禁止 | 否 |
| `tool_refused` | service-level Tool refusal projection | 否 |
| `tool_failed` | 工具执行失败 | 否 |
| `memory_task_running` | Memory Task 不重入 | 否 |
| `schedule_state_error` | Schedule state 损坏或不安全 | 否 |

CLI exit code：成功 `0`，配置/用法 `2`，runtime startup/persistence `1`，Ctrl+C 结束当前 turn 但 REPL 继续时不退出进程。

## 14. `/status` 契约

`RuntimeStatus` 至少包含：

```json
{
  "version": "0.1.0",
  "chat_model": "anthropic-default/model-id",
  "uptime_seconds": 123,
  "estimated_input_tokens": 4200,
  "context_window": 200000,
  "context_used_percent": 2.1,
  "session_message_count": 12,
  "last_consolidated": 4,
  "cumulative_usage": {
    "model_calls": 5,
    "input_tokens": 6100,
    "output_tokens": 900,
    "total_tokens": 7000
  }
}
```

- `chat_model` 显示解析后的实际 route；chat fallback default 时显示 default 对应 provider/model。
- token estimate 对下一次 chat request 的 system prompt、retained messages、tool definitions 和 Runtime Context 的 UTF-8 bytes 求和后除以 4 向上取整。
- 没有已持久化 message 的准备中 Session：message count/`last_consolidated`/usage 都为 0。
- scheduler 初始化失败可附加非持久化 warning，但不得取代 required fields。

## 15. 最小 Session/Provider/Tool 接口

以下签名用于限定职责，不要求使用特定 ABC library。Conversation Session 的
identity、messages、metadata、`last_consolidated` 和 complete snapshot
persistence 由同一个 active `Session` instance 负责；Session 不暴露
filesystem acknowledgement，也不承担 MessageBus/AgentLoop 或 Model
Provider 职责：

```python
class Session:
    @classmethod
    def create(cls, workspace_state: WorkspaceState) -> "Session": ...
    @classmethod
    def load(cls, workspace_state: WorkspaceState, session_id: str) -> "Session": ...
    def add_message(self, role: str, content: str, **fields: JsonValue) -> None: ...
    def update_metadata(self, metadata: dict[str, JsonValue] | None = None, **updates: JsonValue) -> None: ...
    def persist(self) -> None: ...
    def close(self) -> None: ...

class SummaryStore(Protocol):
    async def append(self, content: str, timestamp: datetime) -> SummaryEntry: ...
    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...

class MemoryStore(Protocol):
    async def read_long_term(self) -> str: ...
    async def replace_long_term(self, content: str) -> None: ...
    async def read_summary_cursor(self) -> int: ...
    async def write_summary_cursor(self, index: int) -> None: ...

class ModelProvider(Protocol):
    def stream(self, *, messages, tools, model, max_output, temperature, reasoning_effort, timeout) -> AsyncIterator[ModelStreamEvent]: ...
    async def complete(self, *, messages, tools, model, max_output, temperature, reasoning_effort, timeout) -> ModelResponse: ...
    async def close(self) -> None: ...

class ConcreteTool(BaseTool):
    name = "capability_name"
    description = "Model-visible capability description."
    required = ("value",)
    value: Annotated[str, ToolParam(description="Declared parameter.")]

    async def execute(self, *, value: str) -> str: ...
```

Schema casting、参数校验、安全检查和结果截断/Artifact 写入属于 BaseTool；Tool Gateway 负责固定 Catalog、raw call 解析、一次性确认和 Tool Result 封装。没有 generic Tool retry、Security module、invocation ContextVar 或 Tool plan。route resolution 与 model retry 不由 Tool Gateway 或 ModelProvider 实现；它们属于 Model Router。

## 16. 契约测试清单

Phase 0 应先把以下内容固化为 fixtures/snapshots：

- 默认 config template 与一个完整有效 config。
- config unknown field、unknown route、unknown protocol 和 redaction cases。
- 当前 Session header 与 user/assistant/tool message shape 的 exact-key assertion。
- 完成、中断、model failure、tool failure 后的完整 Session JSONL snapshots，以及 ordered async persist 和 bounded close。
- summary schema exact-key assertion、index/cursor 起点和 batch 行为。
- Schedule model strict round-trip、Schedule state strict-load、legacy state untouched 和 atomic mutation。
- MessageBus sparse outbound schema、terminal marker 以及 AgentLoop control/Future 语义。
- Model Provider scripted transcript：text deltas、tool call deltas、usage、retry-after、timeout、cancellation。
- 固定 Catalog、BaseTool preparation order、file path boundary、Exec/Web confirmation 和 WebFetch redirect/IP cases。
- complete atomic JSONL replacement、缺少 trailing newline、middle corruption、旧 schema rejection，以及 Summary/`last_consolidated` crash divergence。

契约测试断言稳定 code、结构和文件内容；终端文案除脱敏与必需信息外不做全文 snapshot，以免实现被展示细节锁死。

## 17. 确认记录

D01-D16 已于 2026-07-11 全部接受；Session snapshot 契约由 ADR-0009 于 2026-08-04 接受。ADR-0010（Issue #130）明确 supersede ADR-0003、ADR-0005 和 ADR-0007 中受影响的 Tool、Workspace State 和 owned-process 条款。本文 `TOOL_SCHEMA` 是 Python 类型、持久化实现与 contract fixtures 的直接输入。
