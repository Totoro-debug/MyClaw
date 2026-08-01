# MyClaw Runtime Contracts

## 文档状态

- 状态：`Accepted`
- 接受日期：`2026-07-11`
- 目标版本：MyClaw `v0.1`
- canonical source：`CONTEXT.md`
- 产品行为来源：`docs/myclaw-personal-agent-prd.md`
- 实施顺序来源：`docs/myclaw-implementation-plan.md`

本文把已确认的产品行为细化为可直接实现的类型、文件 schema 和代码边界。来自 canonical 文档的行为与 D01-D16 已于 2026-07-11 一并接受；Issue #38 于 2026-07-27 收缩了 Tool 契约，Issue #69 于 2026-08-01 用 host adapter 取代 Windows-only 实现约束，本文对应章节以这些已批准规格为准。后续变更必须先更新本契约及受影响的 PRD/ADR。

本文不新增 one-shot、daemon、HTTP/IPC、MCP、subagent、跨进程协调、用户可配置安全策略或用户自定义 identity。

## 1. 契约通则

### 1.1 兼容性

- `config.toml`、session JSONL、summary JSONL、Summary Cursor、Long-term Memory、Scheduled Work JSON 和 Tool Artifact 是持久化契约。
- session metadata 的 `schema_version` 固定为 `1`。同一文件中的 message records 按该版本解释。
- 首版不实现旧 schema migration；遇到不支持的 `schema_version` 时只读失败并给出用户可见错误，不自动重写。
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
- cron 解释使用 runtime 启动时的系统本地时区。夏令时行为交给选定 cron library，并通过 fake clock 固化测试。
- elapsed time、timeout 和 retry backoff 使用 monotonic clock，不使用 wall clock 差值。

### 1.4 ID

- UUID 均使用小写、带连字符的 UUID4。
- Session ID 使用 `<local_timestamp>_<uuid4>`：`YYYYMMDD-HHMMSS-ffffff_550e8400-e29b-41d4-a716-446655440000`。
- Scheduled Work ID、turn ID 和 message ID 使用 UUID4。
- provider 返回的 tool call ID 原样保存在 message 中，不在业务层重新命名。
- Tool Artifact 文件名使用 tool call ID 的 UTF-8 percent-encoding，通常为 `safe="-_."`；Windows 保留 basename（`CON`、`PRN`、`AUX`、`NUL`、`COM1`-`COM9`、`LPT1`-`LPT9`）全量编码，避免保留名以及 `/`、`\\`、`:` 等字符破坏路径；逻辑路径仍对应原 tool call ID。

## 2. Phase 0 已接受决策表

| ID | 已接受的决定 | 理由 | 同步文档 |
| --- | --- | --- | --- |
| D01 | Python 最低版本为 3.12，采用 `pyproject.toml` + 根目录 `myclaw/` 包布局 | asyncio、typing 和 timezone 能力成熟，降低兼容分支 | 实施计划 |
| D02 | 配置 schema 严格校验未知字段；未知 provider protocol 仍按 PRD 忽略 | 尽早暴露拼写错误，同时保留既定 fallback 语义 | PRD 可补充 |
| D03 | memory message threshold 默认 `40` 条 | 足够早地覆盖长会话，又不会在短对话频繁摘要 | PRD 可补充 |
| D04 | Scheduled Work 文件名为 `scheduled-work.json` | 与 canonical term 一致，避免含糊的 `tasks.json` | PRD/ADR 0002 |
| D05 | Session title 最长 `60` 个 Unicode code points | picker 可读且不需要终端宽度参与持久化 | PRD 可补充 |
| D06 | Summary index 从 `1` 开始，缺失 `.cursor` 等价于 `0` | cursor 语义是“已处理到的最大 index”，直观且易恢复 | PRD 可补充 |
| D07 | model retry 的“最多 5 次”解释为每个逻辑 model call 最多 `5 attempts`，不是首次调用后再重试 5 次 | 限制最坏延迟和费用，与英文 canonical 的 five attempts 一致 | PRD |
| D08 | route fallback 发生在具体 route 缺失、配置不可用或返回永久 route/provider 不可用错误时；同一 route 的临时错误先在 5-attempt budget 内重试 | 避免两个 route 各重试 5 次导致不可控延迟 | PRD |
| D09 | token estimate 使用 `ceil(UTF-8 byte length / 4)`，展示估算输入 token、context window 和占比 | provider-neutral，明确它只是估算值 | PRD 可补充 |
| D10 | WebSearch 使用无凭据的内置 adapter，首选 DuckDuckGo；后端不进入持久化配置契约 | 保持 User Configuration 只控制 enablement，adapter 可替换 | PRD |
| D11 | Shell allowlist 只包含本文列出的精确只读命令形状；Issue #38 期间其他命令一律 refused | 规则可审计，不做不可靠的“只读意图”推断；这是对 ADR-0003 的临时偏离 | PRD/Issue #38 |
| D12 | session 仅允许恢复单个不完整尾行；中间损坏或完整但非法的尾行视为损坏并拒绝加载 | 兼顾 append 崩溃恢复与不静默跳过历史 | PRD/ADR 0001 可补充 |
| D13 | 使用 `memory/pending-consolidations/` journal 协调 summary append 与 session cursor 更新 | summary schema 不能保存 source identity，持久化 journal 才能确定性恢复崩溃窗口 | ADR 0001 |
| D14 | Tool Artifact preview 保留前 `2000` 字符，加固定截断提示和相对路径 | 给模型足够上下文，同时稳定控制 session 大小 | PRD 可补充 |
| D15 | 配置、持久化、模型与服务级错误使用稳定 error code；`ToolError` 与 Tool Result 仅使用安全 message | Tool 消息与模型消费格式保持扁平，其他上层逻辑仍不依赖易变终端文案 | 实施契约/Issue #38 |
| D16 | 首版 Shell 不宣称提供 OS 级文件系统/网络隔离；强制 Workspace cwd，执行仅限精确只读命令 | 仅校验 cwd 无法约束子进程访问绝对路径，不能制造虚假的 sandbox 保证 | PRD/ADR 0003/Issue #38 |

D01-D16 均为首版实现契约，其中 D04、D07、D08、D10、D11、D12、D13、D16 是已显式接受的产品或风险边界。

## 3. Agent Home 与 Workspace

### 3.1 固定布局

```text
~/.myclaw/
  config.toml
  logs/
    run.log.0
    run.log.1
    run.log.cursor
    run.log.lock

<workspace>/.myclaw/
  .gitignore
  scheduled-work.json
  memory/
    memory.md
    summary.jsonl
    .cursor
    pending-consolidations/
      <session_id>.json
  sessions/
    <session_id>.jsonl
    artifacts/
      <session_id>/
        <encoded_tool_call_id>.txt
```

已确定：Agent Home 只拥有 User Configuration 与 Runtime Logs。有效 REPL 启动初始化 Workspace State root、`.gitignore`、`memory/`、`sessions/` 和缺失的 `memory/memory.md`；`summary.jsonl`、`.cursor`、`scheduled-work.json`、Session、artifacts 与 `pending-consolidations/` 按需创建。`myclaw config` 不初始化 Workspace State。

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
4. Agent Home 仅保留全局 User Configuration 与 Runtime Logs；旧的非全局数据不读取、不迁移、不删除。

### 3.3 原子写

- 新文件或整体更新：在目标同目录创建唯一临时文件，写入完整内容，flush，尽可能 fsync，再 atomic replace。
- 文件内容 flush 后尽可能 fsync；POSIX 在发布后同步 parent directory，host 明确不支持同步时保留可测试的 best-effort 分支。
- session 普通 message append：持有当前 runtime 的 session lock，一次写入完整 UTF-8 record + `\n`，flush 后返回；取消不得打断临界区。
- session metadata 更新：持有同一 session lock，读取现有 records，以新 metadata + 原 message records 原子重写。
- 不创建跨进程 lock file，不依赖文件锁，不承诺两个 REPL 写同一 session 的顺序。

## 4. User Configuration 契约

### 4.1 完整 schema

```toml
[runtime]
max_tool_result_chars = 50000

[memory]
consolidation_message_threshold = 40
batch_size = 10
schedule = "0 * * * *"

[tools.web]
enabled = true

[tools.shell]
enabled = true

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

[models.routes.cron]
provider_id = "anthropic-default"
model = "model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
timeout = 120
```

`models.routes.chat`、`models.routes.memory` 和 `models.routes.cron` 均可省略，省略时使用 default。`reasoning_effort` 可省略。

### 4.2 字段规则

| 字段 | 类型/范围 | 默认/要求 |
| --- | --- | --- |
| `runtime.max_tool_result_chars` | integer，`1000..1000000` | 默认 `50000` |
| `memory.consolidation_message_threshold` | integer，`4..10000` | 默认 `40` |
| `memory.batch_size` | integer，`1..1000` | 默认 `10` |
| `memory.schedule` | 5-field cron string | 默认 `0 * * * *` |
| provider ID | `^[a-z0-9]+(?:-[a-z0-9]+)*$` | 必须唯一 |
| `protocol` | string | `anthropic` 或 `openai-compatible`；其他值的 provider 被忽略 |
| `base_url` | absolute HTTP(S) URL | 所有可用 provider 必填、非空 |
| `api_key` | string | plaintext；可为空模板值，不可用于可用 route |
| `models` | unique string array | 模板可空；可用 route 引用的 model 必须存在 |
| route name | table key | 仅 `default`、`chat`、`memory`、`cron` |
| `provider_id` | string | 必填，必须引用可用 provider |
| `model` | string | 必填，必须在 provider catalog 中 |
| `context_window` | integer，`1024..10000000` | 必填 |
| `max_output` | integer，`1..context_window-1` | 必填 |
| `temperature` | number，`0..2` | 必填 |
| `reasoning_effort` | `low`、`medium`、`high` | 可省略；不支持时 adapter 静默忽略 |
| `timeout` | integer seconds，`1..600` | 必填 |
| `tools.web.enabled` | boolean | 默认 `true` |
| `tools.shell.enabled` | boolean | 默认 `true` |

严格拒绝未知顶层 table、未知已知-schema 字段和未知 route name。未知 protocol provider 按 canonical 要求忽略；如果 default 因此不可用，REPL 启动失败。

### 4.3 首次生成模板

缺少 `config.toml` 时，生成模板包含 runtime、memory、tools、一个 ID 为 `openai-local` 的 OpenAI-compatible provider template，以及 `default`、`chat`、`memory`、`cron` 四个显式但不可用的 route scaffold。Provider 的 `base_url`、`api_key` 和 `models` 均为空。四个 route 初始都指向 `openai-local`，model 使用待替换值，并提供完整的模型限制字段；用户可删除不需要定制的具体 route，使其回退到 default。生成后 `myclaw` 退出，`myclaw config` 则显示脱敏模板；旧配置完全缺少 default route 时，启动错误指出 `[models.routes.default]`。

### 4.4 脱敏

- TOML 成功解析时，所有 provider 的 `api_key` 显示为 `"***REDACTED***"`；空值仍显示为空，便于诊断未配置状态。
- TOML 解析失败时，对匹配 `(?i)^\s*api[-_]?key\s*=` 的行只保留 key、等号和 `"***REDACTED***"`。
- exception、Rich renderable、测试 snapshot 和日志替代输出都不得包含未脱敏 key。
- 首版不支持环境变量引用或系统密钥链。

## 5. Conversation Session 契约

### 5.1 Metadata record

第一行必须是：

```json
{"record_type":"metadata","schema_version":1,"id":"20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000","title":"MyClaw implementation","created_at":"2026-07-11T15:30:12.123+08:00","updated_at":"2026-07-11T15:31:02.456+08:00","consolidation_cursor":0,"cumulative_usage":{"model_calls":0,"input_tokens":0,"output_tokens":0,"total_tokens":0}}
```

规则：

- `consolidation_cursor` 是 canonical `Consolidation Cursor` 的持久化表示，是从 `0` 开始的 message boundary，表示前多少条 message records 已被 Conversation Summary 覆盖；Short-term Memory 是 `messages[consolidation_cursor:]`。
- `cumulative_usage` 包含主 chat、tool loop 中的模型调用、title、summary 和与当前 session 直接相关的辅助调用；Memory Task 与 Scheduled Work 计入各自执行所属 session，不计入当前前台 session。
- `total_tokens` 必须等于 `input_tokens + output_tokens`。provider 未返回某项时该项为 `0`，不得用估算值混入实际 usage。
- metadata 更新必须保留所有 message record 的字节语义，不排序或重排历史。

### 5.2 User message

```json
{"record_type":"message","id":"0f8fad5b-d9cb-469f-a165-70867728950e","created_at":"2026-07-11T15:30:12.200+08:00","role":"user","content":"Help me inspect this project."}
```

- `content` 是非空 string；只包含空白的 REPL 输入不创建 message，也不调用模型。
- Runtime Context 是发给模型时临时 prepend 的内容，不写回 `content`。

### 5.3 Assistant message

```json
{"record_type":"message","id":"7c9e6679-7425-40de-944b-e07fc1f90ae7","created_at":"2026-07-11T15:30:13.000+08:00","role":"assistant","content":"I will inspect the files.","tool_calls":[{"id":"call_123","name":"list_files","arguments":"{\"path\":\".\"}"}],"status":"completed","error":null,"usage":{"input_tokens":120,"output_tokens":24,"total_tokens":144}}
```

字段规则：

- `content` 是 string，可为空；`tool_calls` 是 array，可为空；两者至少一个非空，除非 `status=error`。
- tool call `arguments` 必须保留 provider 的原始 JSON string。Tool Gateway 无法解析时追加 flat `tool` error record，不执行具体 Tool。
- `status` 仅为 `completed`、`interrupted`、`error`。
- `error` 在 `completed` 时必须为 `null`；其他状态为 `{code, message}`，message 必须可安全展示。
- streaming 正常完成后才写一条 `completed` assistant record。
- Ctrl+C 时有 partial text 就写 `interrupted` record；没有 partial text 且没有 tool call 时不写空 assistant record。
- 最终 model failure 写 `error` assistant record；恢复对话构建 model context 时省略纯 error record，保留 interrupted partial content并追加内部中断标记。
- 如果 assistant 已产生 tool calls 后 turn 被取消，必须为每个尚无结果的 call 追加 tool error record。

错误示例：

```json
{"record_type":"message","id":"7c9e6679-7425-40de-944b-e07fc1f90ae7","created_at":"2026-07-11T15:30:13.000+08:00","role":"assistant","content":"Partial answer","tool_calls":[],"status":"interrupted","error":{"code":"turn_cancelled","message":"Turn interrupted by user."},"usage":{"input_tokens":120,"output_tokens":8,"total_tokens":128}}
```

### 5.4 Tool message

普通结果：

```json
{"record_type":"message","id":"9b2c3a42-1d2e-4a1e-a827-61f36dc54713","created_at":"2026-07-11T15:30:13.500+08:00","role":"tool","tool_call_id":"call_123","name":"list_files","content":"CONTEXT.md\ndocs/","status":"success","artifact":null}
```

拒绝或失败：

```json
{"record_type":"message","id":"9b2c3a42-1d2e-4a1e-a827-61f36dc54713","created_at":"2026-07-11T15:30:13.500+08:00","role":"tool","tool_call_id":"call_123","name":"write_file","content":"Writing Workspace files is unavailable because confirmation is not implemented.","status":"refused","artifact":null}
```

字段规则：

- `status` 仅为 `success`、`error`、`refused`。
- tool 参数校验失败、执行异常和未完成 call 都使用 `error`，不新增 role。
- tool result 会回传模型，因此 `content` 必须是 normalized、可读的 UTF-8 text，不包含 Python repr 或 traceback。
- Tool Gateway 按具体 Tool 的 `max_retries` 执行有界指数退避；WebSearch/WebFetch 为 2，其余内置 Tool 为 0。
- 旧的 structured arguments 和 nested Tool error JSONL shape 不兼容且不恢复读取。

### 5.5 Session title

固定规则：

1. 首条 user message 持久化后异步使用 chat route生成 title。
2. 模型输出取第一条非空行，去除成对引号和首尾空白，内部连续空白折叠为一个空格，截断到 60 个 Unicode code points。
3. 模型失败、空输出或非法输出时，对首条 user content 应用同样的规范化和截断。
4. fallback 仍为空时使用 `Untitled session`。
5. title 调用 usage 计入 session metadata，但不新增 message。

### 5.6 损坏恢复

- 第一行缺失、非法或不是 supported metadata：session 不可恢复。
- 中间 message record 非法：session 不可恢复，不静默跳过。
- 文件最后没有 `\n` 且最后一段不是合法完整 JSON：视为 append 中断，忽略该尾段并保留此前 records；首次后续成功写入前原子修复文件。
- 有 `\n` 的非法尾行视为完整但损坏，拒绝恢复。
- picker 跳过不可恢复 session，同时显示一条汇总警告；不得把损坏文件自动删除。

## 6. Conversation Summary、Cursor 与 Long-term Memory

### 6.1 Summary record

每行严格只有三个字段：

```json
{"index":1,"timestamp":"2026-07-11T16:00:00.000+08:00","content":"The user is implementing MyClaw and prefers a file-first architecture."}
```

- `index` 全局严格递增，从 `1` 开始。
- `content` 非空。
- 不保存 source session、message ID、message range、route 或 usage。
- append 使用 runtime 内全局 summary lock。

### 6.2 Summary Cursor

- `memory/.cursor` 内容是一个非负 ASCII decimal integer 和尾随 `\n`，例如 `12\n`。
- 文件缺失等价于 `0`。
- cursor 表示 Memory Task 已成功处理的最大 summary index。
- no update 或 edit success 后原子写入 batch 的最后 index；required edit failure 不写。

### 6.3 Consolidation crash consistency

file-first 无法原子提交“追加 global summary”和“更新 session metadata”。使用内部 journal：

```json
{"session_id":"<session_id>","old_cursor":0,"new_cursor":8,"summary_index":12,"timestamp":"2026-07-11T16:00:00.000+08:00","content":"..."}
```

提交协议：

1. 在 global summary lock 内确定下一个 `summary_index`，生成 summary content 和目标 session cutoff。
2. 原子写入 `memory/pending-consolidations/<session_id>.json`，其中保存即将写入的完整 summary record 信息。
3. 如果 `summary.jsonl` 尚无该 index，append journal 中的精确 record；如果已有相同 record，视为幂等成功；如果 index 已存在但内容不同，停止并报告 `persistence_error`。
4. 原子更新 session `consolidation_cursor = new_cursor`。
5. 删除 journal。删除前崩溃不会破坏正确性，恢复时重复步骤 3-5。

runtime 启动时先恢复所有 journal，再接受新 chat turn。journal 是 Agent Home 内部状态，主 Agent file tools 不可读写。该协议只保证单 runtime 正常恢复；两个 REPL 同时 consolidation 仍可能争用全局 index，这是首版明确接受的跨进程风险。

### 6.4 Long-term Memory cache

- runtime startup 原子创建缺失模板并读取一次，保存 immutable string snapshot。
- chat 和 cron 的 System Prompt 使用该 snapshot。
- `/memory` 和 Memory Task 每次读取磁盘最新文件。
- Memory Task 成功编辑后不刷新 runtime snapshot；下一次 runtime 启动才生效。
- system-level prompt 超过 route context budget 时返回 `memory_context_too_large`，不得裁剪 Long-term Memory。

## 7. Scheduled Work 契约

固定文件名：`~/.myclaw/scheduled-work.json`。

顶层必须是 JSON array；空状态可以由文件缺失或 `[]` 表示。记录示例：

```json
[
  {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "title": "Weekly project review",
    "cron": "0 9 * * 1",
    "prompt": "Review the current project and summarize open risks.",
    "created_at": "2026-07-11T16:00:00.000+08:00",
    "enabled": true,
    "session_id": "20260711-160000-000000_550e8400-e29b-41d4-a716-446655440000"
  }
]
```

规则：

- 首版 record 恰好包含 canonical 七个字段，不增加 `last_run_at`、`next_run_at` 或运行状态。
- `title`、`prompt` 非空；title 最长 120 字符，prompt 最长 20000 字符。
- `cron` 是合法 5-field cron，不接受 seconds 或 timezone field。
- `enabled` 创建时必须为 `true`。
- `session_id` 在创建记录时分配；session 文件在首次 trigger 写入 user message 时才创建。
- store update 在单 runtime lock 内执行整体 JSON array 原子 rewrite。
- 单条 record 非法使整个文件配置无效，scheduler 不启动；REPL 主对话仍可运行并通过 `/status` 显示 scheduler error。不得静默丢弃非法任务。
- 当前 Tool contract 固定拒绝创建 Scheduled Work；拒绝时不分配持久化 record。
- task-specific session 使用普通 session schema；每次 trigger 追加 prompt user message和最终 assistant/tool history。

## 8. Prompt、Runtime Context 与预算

### 8.1 Chat 与 cron System Prompt

chat 和 cron 的 system-level context 按以下固定顺序组装：

1. 内置 identity prompt，其中包含 normalized absolute Workspace。
2. 完整的 runtime-startup Long-term Memory snapshot，以明确的 `<long_term_memory>` delimiter 包裹。
3. 当前 Tool Catalog 的 guidance，以明确的 `<tool_guidance>` delimiter 包裹。

User Configuration 不得插入或替换 identity/system prompt。缓存的 OpenAI-format Tool schema snapshots 通过 provider 的结构化 tools 字段发送，不把 JSON schema 重复拼入自然语言 guidance。

### 8.2 当前 user input 的 Runtime Context

发给 chat/cron model 的当前 user message临时转换为：

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
- Scheduled Work 可额外加入 `scheduled_work_id`；Memory Task 使用专门 prompt，不伪装成 chat user input。

### 8.3 专用 prompts

- Session title：只接收规范化后的首条 user content，不注入 Long-term Memory、tools 或 conversation history。
- Conversation Summary：只接收本次选中的早期 session messages，不注入 Long-term Memory 或 Tool Catalog。
- Memory Task：接收 Summary Cursor 后的 batch 和四分区维护规则，并只暴露 restricted memory tools。
- Scheduled Work：使用 chat/cron system composition，把 task prompt 作为 task-specific session 的普通 user message。

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

- route purpose 是 `default | chat | memory | cron`。
- chat 用于主对话和 title；memory 用于 summary 和 Memory Task；cron 用于 Scheduled Work。
- chat request 必须调用 streaming provider contract；memory/cron 可调用 complete contract。
- requested route 与 default 指向同一配置时只尝试一次。
- default 不可用时 runtime startup 失败。

固定失败/fallback 顺序：

1. route 缺失、provider 被忽略、model 不在 catalog等静态不可用：直接使用 default。
2. route 调用发生 rate limit、timeout 或 unavailable：在总共 5 attempts 内对同一 route 指数退避。
3. route 返回 auth、model-not-found 或 unsupported 等永久不可用错误：若不是 default，切到 default，并只使用剩余 attempt budget。
4. invalid request、context overflow、cancellation 和本地 schema error 不 fallback。

一个逻辑 model call 在 requested route 和 default 之间共享 5-attempt budget，最坏不超过 5 次 provider 调用。若第 5 次才暴露永久错误，则没有剩余预算调用 default，当前逻辑调用失败。

### 9.2 统一请求

```python
ModelRequest(
    request_id: UUID,
    route: Literal["default", "chat", "memory", "cron"],
    system_prompt: str,
    messages: tuple[ModelMessage, ...],
    tools: tuple[OpenAIToolSchema, ...],
    stream: bool,
    model: str,
    max_output: int,
    temperature: float,
    reasoning_effort: Literal["low", "medium", "high"] | None,
    timeout_seconds: int,
)
```

`OpenAIToolSchema` 是由 Tool Catalog 缓存的 OpenAI Function Calling 格式快照。OpenAI-compatible adapter 直接传递该格式；Anthropic adapter 在内部转换字段。provider adapter 接收已解析的 concrete route config，不自行 fallback、不读取 User Configuration、不处理 session。

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

## 10. Agent Event 与 Port 契约

### 10.1 Event envelope

所有事件共享：

```json
{
  "type": "text_delta",
  "event_id": 3,
  "turn_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2026-07-11T15:30:13.123+08:00",
  "payload": {}
}
```

- `event_id` 是 runtime 内单调递增 integer，仅用于排序，不持久化。
- `turn_id` 对 background lifecycle 事件可为对应 background run ID。
- CLI 只依赖 event，不读取 session、tool 或 provider 对象。

### 10.2 Event types

| type | payload | 说明 |
| --- | --- | --- |
| `turn_started` | `{}` | 前台 turn 接受并开始 |
| `text_delta` | `{delta}` | chat streaming 文本 |
| `progress` | `{status, summary}` | 非敏感进度 |
| `tool_started` | `{tool_call_id, tool_name, summary}` | 不含完整 arguments |
| `tool_completed` | `{tool_call_id, tool_name, status, summary}` | 不含完整 raw result |
| `turn_completed` | `{content, usage}` | 一个 turn 恰好一个终态 |
| `turn_failed` | `{error}` | 安全的用户可见错误 |
| `turn_cancelled` | `{partial_content}` | Ctrl+C 终态 |
| `background_completed` | `{kind, title, session_id, status, summary}` | Scheduled Work 空闲提示；Memory Task 周期运行不发 |

事件状态 summary 最长 240 字符。tool argument 和 raw result 不进入普通 tool activity event。当前契约没有 permission request event 或等待确认状态。

### 10.3 Conversation Port

最小接口：

```python
class ConversationPort(Protocol):
    def submit(self, text: str) -> AsyncIterator[AgentEvent]: ...
    async def cancel_active_turn(self) -> None: ...
```

- 同一 port 同时只允许一个 foreground `submit`；REPL 自身串行化输入。
- Conversation Port 不接受 session ID、route、provider 或 tool catalog 参数。

### 10.4 Management Port

最小接口：

```python
class ManagementPort(Protocol):
    async def config_view(self) -> ConfigView: ...
    async def status(self) -> RuntimeStatus: ...
    async def resumable_sessions(self) -> tuple[SessionSummary, ...]: ...
    async def resume(self, session_id: str) -> ResumeResult: ...
    async def memory_view(self) -> str: ...
    async def dream(self) -> MemoryTaskResult: ...
```

- `resumable_sessions` 仅返回当前 Workspace 的 id、title、created_at、updated_at、message_count。
- `resume` 再次验证 session 属于当前 Workspace，不信任 UI 传入值。
- `config_view` 已脱敏；Management Port 永不返回 plaintext API key。
- `memory_view` 读取磁盘，不返回 runtime cache。

## 11. Tool Gateway 契约

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
)
```

`BaseTool.to_schema()` 从具体 Tool 类直接声明的公开注解、显式 `required`、默认值和 `ToolParam` 统一生成 schema，具体 Tool 不得覆盖。`ToolGateway.call()` 是唯一公开调用入口，顺序固定为：parse raw JSON -> resolve -> project/convert/validate -> refuse if required -> execute with bounded retry -> normalize。未声明参数被忽略；只允许规格定义的 string-to-integer、integral-float-to-integer 和 string-to-boolean 转换。解析、prepare 和 refusal 不重试，取消继续向上传播。

`max_retries` 必须在 `0..5`，表示首次执行之外的额外次数；重试前依次等待 1、2、4、8、16 秒。WebSearch 和 WebFetch 为 2，其余内置 Tool 为 0。执行阶段的普通异常都可消耗已声明的重试预算，最终 `ToolError` 返回其安全 message，其他异常返回 capability-specific generic message；`CancelledError` 等 `BaseException` 不被捕获。

### 11.2 Catalog 与依赖所有权

- Runtime Core 在启动时构造具体 `BaseTool`，注入 Workspace、`Security`、store、client、clock 等稳定专用依赖，并一次性注册完整 catalog。
- Tool 调用不接收 session ID、Agent Home、lane、approval flag 或通用 execution context。
- `Security` 实现公共访问边界，具体 Tool 保留 capability-specific 规则和同步 `refusal_reason()`。
- Memory Task 使用标准 Tool Gateway 和仅含专用 Long-term Memory read/edit Tool 的 catalog。
- Tool Gateway 不在前台/后台之间加全局执行锁。
- Tool Gateway 不设置统一 timeout，不持久化 Tool Result，也不创建 Tool Artifact。

### 11.3 内置 file tools

固定名称和 input schema：

| Tool | 参数 | 行为 |
| --- | --- | --- |
| `read_file` | `path: str`, `offset: int = 0`, `limit: int = 2000` | 读取 UTF-8 文本行；二进制/解码失败返回 error |
| `list_files` | `path: str = "."`, `recursive: bool = false`, `max_entries: int = 1000` | 稳定排序的相对路径列表 |
| `search_files` | `query: str`, `path: str = "."`, `glob: str | null`, `max_results: int = 200` | 文本搜索，返回 path/line/preview |
| `write_file` | `path: str`, `content: str` | 当前固定 refused，不执行 Workspace 写入 |
| `edit_file` | `path: str`, `old_text: str`, `new_text: str`, `replace_all: bool = false` | 当前固定 refused，不执行 Workspace 编辑 |

参数上限：`limit 1..10000`、`max_entries 1..10000`、`max_results 1..1000`。`old_text` 为空无效；`replace_all=false` 时匹配数必须恰好为 1。

主 Agent file access：

- Workspace 内 read/list/search：allow。
- Workspace 内 write/edit：refused。
- Agent Home 的 `memory/memory.md`：main catalog read allow，write/edit refused；Memory Task 专用 edit Tool 仅允许精确文件。
- 当前 session 的 artifact directory：read allow，write/edit refused。
- `config.toml`、session JSONL、summary、cursor、Scheduled Work JSON 和其他 Agent Home 内部路径：main file Tools read/write/edit 均 fail closed；这些内容只通过对应 Port/Store 暴露。
- 其他路径：fail closed，不升级为 confirmation。

路径检查必须基于解析后的目标和最近存在父目录，防止 `..`、symlink、junction/reparse point 越界。拒绝访问 device file、named pipe 和非普通文件。

### 11.4 Shell tool

```json
{
  "name": "shell",
  "arguments": {
    "command": "git status --short",
    "cwd": ".",
    "timeout": 60
  }
}
```

- `cwd` 缺省为 Workspace，解析后必须在 Workspace 内。
- `timeout` 必须在 `60..600` 秒。
- `tools.shell.enabled=false` 时不进入 catalog。
- `pwd` 不创建子进程，直接返回已验证的 host-native Workspace cwd。
- 四个 Git 操作以 trusted Git executable 和固定 hardened argv 通过 direct process execution 执行；不调用 command interpreter、login shell 或 quoting bootstrap。
- 含管道、重定向、命令替换、后台符号、多命令分隔符或控制字符的变体不匹配精确 allowlist，并在创建进程前 refused。

自动 allowlist 仅包括以下精确形状：

```text
pwd
git status
git status --short
git diff --stat
git diff --name-only
```

执行 Git 操作时固定 argv 禁用 pager 与 filesystem monitor；`git diff` 变体还禁用 external diff 和 textconv。其他命令一律 refused；cwd 越界、timeout 越界或命令包含 NUL/control characters 同样 fail closed。用户不能扩展 allowlist。Windows 使用 Job Object 拥有完整进程树，POSIX 使用新 session 与 process-group signals；timeout、cancellation 和 shutdown 都等待 owned process tree 清理完成。

重要边界：首版不宣称 Shell 子进程受到 OS 级文件系统或网络 sandbox。Workspace 限制严格作用于 `cwd` 和固定 allowlist。Issue #38 暂时拒绝所有非 allowlist 命令，明确偏离 ADR-0003 要求的前台确认；该偏离不增加 OS 隔离保证。若产品要求“绝不访问 Workspace 外路径”，必须先选择并记录 host-native 进程隔离方案，不能通过字符串扫描命令来伪造该保证。

### 11.5 Web tools

`web_search`：

```json
{"query":"MyClaw agent runtime","max_results":5}
```

- `max_results` 范围 `1..10`，默认 `5`。
- normalized result 是 JSON text array，每项恰好为 `{title, url, snippet}`。
- 无结果返回 `[]`，网络失败返回 tool error。

`web_fetch`：

```json
{"url":"https://example.com/page"}
```

- 仅允许 `http` 和 `https`，禁止 URL userinfo。
- 请求前解析全部 IP；任一地址属于 loopback、private、link-local、unspecified、multicast 或 reserved 时拒绝。
- 每次 redirect 重新执行 scheme、hostname、DNS/IP 检查，最多跟随 5 次。
- DNS rebinding 防护要求连接实际 peer IP 仍属于已校验公网集合；无法验证时拒绝。
- connect timeout 10 秒、total timeout 30 秒、响应 body 上限 10 MiB。
- HTML 转为可读 text；其他 textual media 保留 text；二进制返回 unsupported media error。
- `tools.web.enabled=false` 时 `web_search` 和 `web_fetch` 都不进入 catalog。

### 11.6 Scheduled Work tool

```json
{
  "name": "create_scheduled_work",
  "arguments": {
    "title": "Weekly project review",
    "cron": "0 9 * * 1",
    "prompt": "Review the project and summarize open risks."
  }
}
```

- 当前 foreground 与 Scheduled Work catalog 中调用均固定 refused，不分配 ID、不写 store，也不进入重试。
- 未来确认设计恢复创建能力后，成功执行才生成 id、created_at、enabled 和 session_id，并原子更新 store。
- 创建时不分析 prompt 未来是否需要 disabled tools。

### 11.7 Tool Artifact

Tool Gateway 返回以后，Runtime Core 仅在 `status == "success"` 且 `len(raw_content) > runtime.max_tool_result_chars` 时：

1. 把完整 raw content 原子写入 `artifacts/<session_id>/<encoded_tool_call_id>.txt`。
2. 以 immutable replacement 创建新的 Tool Result，其 `content` 为前 2000 字符加：

```text

...[truncated; full result stored at artifacts/<session_id>/<encoded_tool_call_id>.txt]
```

3. `artifact` 写为：

```json
{"path":"artifacts/<session_id>/<encoded_tool_call_id>.txt","total_chars":73421,"preview_chars":2000}
```

Artifact 写失败时整个 tool result 为 `error`，不得把超阈值 raw content 回退写入 session。Externalization 没有 callback、commit、rollback 或 cleanup 生命周期；文件写成功后若 Session persistence 失败，允许留下 orphan artifact。

## 12. Fail-closed capability 矩阵

| Capability | Foreground | Scheduled Work | Memory Task |
| --- | --- | --- | --- |
| Workspace read/list/search | allow | allow | 不在 catalog |
| Workspace write/edit | refused | refused | 不在 catalog |
| Long-term Memory read | allow | allow | allow |
| Long-term Memory edit | refused | refused | allow，仅精确文件 |
| Current-session artifact read | allow | allow | 不在 catalog |
| Agent Home internal state read/write | refused | refused | refused，memory store 自身操作除外 |
| Shell allowlist | allow | allow | 不在 catalog |
| 其他 Shell | refused | refused | 不在 catalog |
| WebSearch/WebFetch | allow if enabled | allow if enabled | 不在 catalog |
| Create Scheduled Work | refused | refused | 不在 catalog |
| Workspace/Agent Home 之外 | refused/error | refused/error | refused/error |

当前 Tool 契约没有 centralized Permission Policy、`ask`、approval flag 或 execution context。危险 capability 在执行前返回 `refused`；无效参数、越界访问和执行失败返回 message-only `error`。Tool Result 不携带 error code 或嵌套 `ErrorInfo`。

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

`ErrorInfo` 仍用于 model、assistant turn 和 service-level error contract，不用于 `ToolError` 或 Tool Result。`ToolError` 只有安全的 message；cause、traceback、SDK response body 只存在于瞬时诊断上下文，不写 Tool message 或 Agent Event。首版无持久化 runtime log。

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
| `scheduled_work_invalid` | cron/record 无效 | 否 |

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
  "consolidation_cursor": 4,
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
- 没有已持久化 message 的准备中 session：message count/cursor/usage 都为 0。
- scheduler 初始化失败可附加非持久化 warning，但不得取代 required fields。

## 15. 最小 Store/Provider/Tool 接口

以下签名用于限定职责，不要求使用特定 ABC library：

```python
class SessionStore(Protocol):
    async def append_message(self, session_id: str, message: SessionMessage) -> None: ...
    async def update_metadata(self, session_id: str, update: MetadataUpdate) -> None: ...
    async def load(self, session_id: str) -> ConversationSession: ...
    async def list_for_workspace(self, workspace: Path) -> tuple[SessionSummary, ...]: ...

class SummaryStore(Protocol):
    async def append(self, content: str, timestamp: datetime) -> SummaryEntry: ...
    async def after(self, cursor: int, limit: int) -> tuple[SummaryEntry, ...]: ...

class MemoryStore(Protocol):
    async def read_long_term(self) -> str: ...
    async def replace_long_term(self, content: str) -> None: ...
    async def read_summary_cursor(self) -> int: ...
    async def write_summary_cursor(self, index: int) -> None: ...

class ModelProvider(Protocol):
    def stream(self, request: ModelRequest) -> AsyncIterator[ModelStreamEvent]: ...
    async def complete(self, request: ModelRequest) -> ModelResponse: ...
    async def close(self) -> None: ...

class ConcreteTool(BaseTool):
    name = "capability_name"
    description = "Model-visible capability description."
    required = ("value",)
    value: Annotated[str, ToolParam(description="Declared parameter.")]

    async def execute(self, *, value: str) -> str: ...
```

参数解析、结果标准化和 bounded retry 属于 Tool Gateway；公共安全边界属于注入的 `Security`；capability-specific 规则属于具体 Tool；Artifact externalization 属于 Runtime Core。route resolution 与 retry 不由 ModelProvider 实现；它们属于 Model Router。

## 16. 契约测试清单

Phase 0 应先把以下内容固化为 fixtures/snapshots：

- 默认 config template 与一个完整有效 config。
- config unknown field、unknown route、unknown protocol 和 redaction cases。
- 四种 session records：metadata、user、assistant mixed content/raw tool calls、flat tool artifact/error。
- 完成、中断、model failure、tool failure 后的完整 session JSONL snapshots。
- summary schema exact-key assertion、index/cursor 起点和 batch 行为。
- Scheduled Work JSON exact-key assertion。
- 全部 Agent Event payload schema 与事件顺序。
- Model Provider scripted transcript：text deltas、tool call deltas、usage、retry-after、timeout、cancellation。
- Tool fail-closed matrix、file path boundary、Shell exact allowlist 和 WebFetch redirect/IP cases。
- atomic rewrite、incomplete final JSONL line、middle corruption 和 consolidation crash window。

契约测试断言稳定 code、结构和文件内容；终端文案除脱敏与必需信息外不做全文 snapshot，以免实现被展示细节锁死。

## 17. 确认记录

D01-D16 已于 2026-07-11 全部接受，Tool 契约由 Issue #38 于 2026-07-27 更新。ADR-0001 和 ADR-0002 继续有效；非 allowlist Shell 暂时直接拒绝的行为是对 ADR-0003 前台确认规则的已批准临时偏离。本文 schema 是 Python 类型、持久化实现与 contract fixtures 的直接输入。
