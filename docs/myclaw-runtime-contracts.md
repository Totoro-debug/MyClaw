# MyClaw Runtime Contracts

## 文档状态

- 状态：`Accepted`
- 接受日期：`2026-07-11`
- 目标版本：MyClaw `v0.1`
- 领域语言：`CONTEXT.md`
- 产品行为来源：`docs/myclaw-personal-agent-prd.md`

本文把已确认的产品行为细化为当前实现的类型、文件 schema 和代码边界。后续产品或架构变更必须先更新本契约及受影响的 PRD/ADR。

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
- Schedule Tool 的 cron 默认使用 UTC，显式 timezone 必须是规范 IANA 时区；Dream System Schedule Job 使用 Runtime 启动时的系统本地 IANA 时区。夏令时行为交给选定 cron library，并通过 fake clock 固化测试。
- elapsed time、timeout 和 retry backoff 使用 monotonic clock，不使用 wall clock 差值。

### 1.4 ID

- UUID 均使用小写、带连字符的 UUID4。
- Session ID 使用 `<local_timestamp>_<uuid4>`：`YYYYMMDD-HHMMSS-ffffff_550e8400-e29b-41d4-a716-446655440000`。
- User Schedule Job ID 和 turn ID 使用 UUID4；内置 System Schedule Job 可以使用保留 symbolic ID，当前唯一保留值为 `dream`。Conversation Session message dictionaries 没有通用 message ID。
- provider 返回的 tool call ID 原样保存在 message 中，不在业务层重新命名。
- Tool Artifact 文件名直接使用只含 ASCII 字母、数字、下划线和连字符的 tool call ID；其他 ID 使用 UUID4。Artifact 路径固定为 Workspace State 下 `.myclaw/artifacts/<session_id>/<id>.txt`。

## 2. Accepted Design Decisions

| ID | 已接受的决定 | 理由 |
| --- | --- | --- |
| D01 | Python 最低版本为 3.12，采用 `pyproject.toml` + 根目录 `myclaw/` 包布局 | asyncio、typing 和 timezone 能力成熟，降低兼容分支 |
| D02 | Runtime loading 投影掉未知配置字段；`myclaw config` 报告未定义字段；未知 provider protocol 按 PRD 忽略 | 保持运行时配置兼容，同时暴露拼写错误 |
| D03 | memory message threshold 默认 `40` 条 | 足够早地覆盖长会话，且不在短对话中频繁摘要 |
| D04 | Schedule state 文件名为 `schedule.json` | 与 Schedule domain 一致，legacy state 不参与升级 |
| D05 | Session title 最长 `60` 个 Unicode code points | picker 可读且不让终端宽度参与持久化 |
| D06 | Summary index 从 `1` 开始，缺失 `.cursor` 等价于 `0` | cursor 表示已处理的最大 index，语义直接 |
| D07 | 每个逻辑 model call 最多 `5 attempts` | 限制最坏延迟和费用 |
| D08 | 具体 route 缺失、不可用或返回永久 route/provider 错误时 fallback；临时错误在共享 5-attempt budget 内重试 | 避免 requested/default route 分别耗尽一套 budget |
| D09 | token estimate 使用 `ceil(UTF-8 byte length / 4)` | 提供 provider-neutral 估算且不混入真实 usage |
| D10 | WebSearch 使用无凭据的内置 adapter，后端和 enablement 不进入配置 | 固定 Catalog 保持可预测，adapter 可替换 |
| D11 | Exec 使用单次 Bash 和固定 destructive/DNS safety checks；没有 allowlist 或 OS sandbox 声明 | 规则可审计，不把 cwd 检查误称为进程隔离 |
| D12 | Session 只接受完整当前 JSONL snapshot，拒绝旧或损坏形状 | 完整 atomic replacement 简化 Session authority |
| D13 | Conversation Summary 直接更新 `last_consolidated`，不建立跨文件 journal | 接受 crash 后 divergence，保持 Session 接口小且无 persistence acknowledgement |
| D14 | BaseTool 在成功结果超过 `4096` 字符时写入 `.myclaw/artifacts/<session>/<id>.txt` | 限制模型上下文同时保留完整结果 |
| D15 | 配置、持久化、模型与服务错误使用稳定 code；Tool Error/Result 使用安全扁平 message | 上层逻辑不依赖易变终端文案 |
| D16 | Exec 不提供 OS 级文件系统、网络或进程隔离 | cwd 和字符串检查不能制造虚假 sandbox 保证 |
| D17 | 每个非空且不是 Manual Skill Invocation 的普通前台输入在 Agent Run 前做 Task Framing，Blackboard 与 usage 只随已接受 increment 提交；手动 Skill 轮为 metadata no-op | 在跨输入保留一个明确任务边界，不引入计划或执行控制产品 |
| D18 | CLI 是唯一 Runtime composition root；一个 Session-scoped Agent Loop 承担一个 Runtime Generation | 让创建位置、生命周期与真实 ownership 一致，不保留代理所有组件的 Runtime 聚合层 |

D01-D18 均为当前实现契约；精确持久化、Tool、Runtime 和 Task Framing 边界由本文后续章节与对应 ADR 定义。

## 3. Agent Home 与 Workspace

### 3.1 固定布局

```text
~/.myclaw/
  config.toml
  skills/
    <skill-directory>/
      SKILL.md

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

已确定：Agent Home 拥有 User Configuration 和可选的 user-authored Skill root；其中既有的 `logs/run.log.0`、`run.log.1`、`run.log.cursor` 与 `run.log.lock` 仅作为 legacy Runtime Log 文件逐字节保留，MyClaw 不再读取、移动、删除、截断或更新它们。有效 Terminal Conversation 启动初始化 Workspace State root、`.gitignore`、`memory/`、`sessions/`、缺失的 `memory/memory.md`，并在注册内置 Dream System Job 时创建或校正 `schedule.json`；`summary.jsonl`、`.cursor`、Session、`.myclaw/artifacts/`、`schedule-sessions/` 和 `logs/` 按需创建。legacy scheduled-work state 原样保留且不读取、不检测、不迁移、不重命名或删除。`myclaw config` 不初始化 Workspace State。`AgentHome.initialize()` 只创建 Agent Home root，不创建缺失的 `skills/`。

#### 3.1.1 Skill Catalog discovery

`~/.myclaw/skills/` 缺失或为空时，Skill Catalog 是空 snapshot。Catalog 只扫描其一级子目录中名为 `SKILL.md` 的 instruction file；不会把嵌套目录作为独立候选。frontmatter 必须从文件首行的独占 `---` 开始，并以之后的独占 `---` 结束；其内容使用安全 YAML mapping 解析，`name` 和 `description` 必须是字符串。原始 `name` 不做 trim，必须直接匹配 `[a-z_-][a-z0-9_-]{0,63}`；`description` trim 后必须为 1 到 1024 个 Unicode code points。

每个候选的 instruction path 必须是可读普通文件并在 canonical Skill root 内；每个候选都捕获一个 complete `SKILL.md` document，其 bytes 必须是 UTF-8。canonical root 外的 symlink/reparse target、缺失文件、非 UTF-8 document 和其他 malformed metadata/body 均跳过。跳过时只记录安全的 candidate path 与 reason，不记录 instruction document。候选按 canonical path 字符串升序评估；reserved Management Command names 和重复 Skill names 不进入 snapshot，同名时保留第一个有效候选。Catalog metadata 不注册 Tool。

`runtime.enable_skill_always_load` 是 boolean，默认 `false`。每个 Agent Loop 构造自己的 `SkillLoader`；Loader 对每个候选只读取一次完整 UTF-8 document，以同一份 bytes 完成 frontmatter、metadata、document、canonical path 和可选 `always` 校验，然后原子保存一个 Runtime Generation-scoped immutable `SkillSnapshot`。YAML non-boolean `always` 只产生一次安全 warning 并按非 always Skill 保留；`false`、缺失或关闭配置时不进入 always-loaded subset。Generation 不暴露中间 snapshot。

当前 Agent Loop 的 manual Skill invocation 只能使用 SkillLoader 保存的完整 snapshot document，不再次读取磁盘：

```python
class LoadedSkill:
    metadata: SkillMetadata
    document: str
    always: bool

class SkillSnapshot:
    root: Path
    skills: tuple[LoadedSkill, ...]

class SkillLoader:
    def load(self) -> SkillSnapshot: ...
```

`LoadedSkill.document` 逐字符保留完整 document，不剥离 opening delimiter、frontmatter、closing delimiter、正文或原始换行。`SkillSnapshot` 保存 immutable `tuple[LoadedSkill, ...]`，`SkillLoader` 只通过 `load()` 发布 snapshot；不存在可变的 `snapshot` 字段。当前 Agent Loop 内 manual invocation 和 always-loaded projection 使用 frozen document；启动或任何 `/resume` 构造新的 Agent Loop 时重新扫描。模型自主调用普通 `read_file` 仍遵守 Tool path 和实时文件读取契约。缺失、不可读、非 UTF-8、frontmatter/YAML 无效或 canonical containment 失效的单个候选被跳过并记录安全 warning；已发布 snapshot 的全局预算 preflight failure 使用稳定 `skill_context_too_large` 并终止 Terminal Conversation。

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
4. Agent Home 仅保留全局 User Configuration、user-authored Skill root 与 untouched legacy Runtime Log files；旧的非全局数据不读取、不迁移、不删除。

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
- 不创建 Session 跨进程 lock file，不依赖文件锁，不承诺两个 Terminal Conversation processes 写同一 Session 的顺序。

## 4. User Configuration 契约

### 4.1 完整 schema

```toml
[runtime]
max_tool_result_chars = 4096
max_iterations = 50
enable_skill_always_load = false

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
| `runtime.max_iterations` | integer，至少 `50` | 默认 `50` |
| `runtime.enable_skill_always_load` | boolean | 默认 `false`；仅控制 startup always-load freeze |
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

启动时将未知顶层 table、未知字段和未知 route table 投影掉；`myclaw config` 仍报告这些未定义字段。未知 protocol provider 按 canonical 要求忽略；如果 default 因此不可用，Terminal Conversation 启动失败。Tool Catalog 不接受用户配置的 enablement 或 replacement。

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
{"session_id":"20260711-153012-123456_550e8400-e29b-41d4-a716-446655440000","created_at":"2026-07-11T15:30:12.123+08:00","updated_at":"2026-07-11T15:31:02.456+08:00","last_consolidated":0,"metadata":{"title":"MyClaw implementation","token_usage":{"model_calls":0,"input_tokens":0,"output_tokens":0,"total_tokens":0},"blackboard":{"goal":"Inspect the project","completion_boundary":"Report the findings"}}}
```

规则：

- `last_consolidated` 是从 `0` 开始的 message boundary，表示前多少条 Session messages 已被 Conversation Summary 覆盖；Short-term Memory 是 `messages[last_consolidated:]`。Conversation Summary 直接赋值，不调用 cursor-specific method，也不通过 journal 与 Session snapshot 协调。
- `metadata` 必须拥有 `title` 与 `token_usage`，并可选拥有当前 `blackboard`。`token_usage` 包含主 chat、Tool loop、Task Framing、title、Conversation Summary 和与当前 Session 直接相关的辅助调用。Dream 不接收 Conversation Session；User Schedule Job 的模型调用计入其 Schedule Session，不计入当前前台 Session。
- `blackboard` 恰好包含 `goal` 与 `completion_boundary` 两个经过 trim 后的非空 string，不设字符数上限。Session load 将 malformed optional Blackboard 当作缺失并从内存 metadata 移除；`update_metadata()` 和 Agent Loop commit 必须拒绝非法形状。
- `total_tokens` 必须等于 `input_tokens + output_tokens`。provider 未返回某项时该项为 `0`，不得用估算值混入实际 usage。
- 每次成功持久化都是完整 compact UTF-8 JSONL atomic replacement，header 与所有 message lines 一起提交；不存在逐消息写入或 metadata-only rewrite。
- 当前 header 必须恰好包含 `session_id`、`created_at`、`updated_at`、`last_consolidated`、`metadata`。旧 schema unsupported，不 migration、不 version dispatch。

### 5.2 User message

```json
{"role":"user","content":"Help me inspect this project.","timestamp":"2026-07-11T15:30:12.200+08:00"}
```

- `content` 是非空 string；只包含空白的 Terminal Conversation 输入不创建 message，也不调用模型。`timestamp` 使用 system local timezone 的 ISO 8601 string。
- Runtime Context 与 Runtime-owned Blackboard block 是发给模型时临时组装的内容，不写回 `content`。

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
- `persist()` 不等待 filesystem operation，不返回 task、acknowledgement 或 failure；每个按调用顺序排列的完整 snapshot 最多尝试三次，失败后异步等待 `100 ms`、`200 ms` 再重试；普通写入异常不产生 `OutboundMessage`、Session Log 或其他诊断记录。
- `close()` 标记 Session closed 后抑制过期异步 snapshot，最多同步尝试三次（间隔 `100 ms`、`200 ms`），最终失败静默吞掉。
- `abandon()` 是 forced Runtime Generation replacement 的同步边界：它幂等地取消 pending snapshots、禁止后续 Session mutation，不等待、不重试且不做 final save；普通 `close()` 不使用该语义。

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
- cursor 表示已由 Dream 领取的最大 summary index。
- Dream 在模型工作前通过 Memory Manager 原子写入 batch 的最后 index；no update、edit success、model failure 和 Tool/edit failure 均保留已推进的 cursor，不自动重试该 batch。

### 6.3 Summary and `last_consolidated` consistency

Conversation Summary Manager generates one summary, asks Memory Manager to append it under the
single-runtime Summary lock, then directly assigns the active Session's `last_consolidated` only
after the append succeeds. There is no pending journal, startup recovery path, or
cross-file transaction between `summary.jsonl` and Session JSONL. A crash or
failed Session snapshot may leave the summary entry and `last_consolidated`
divergent; subsequent work may therefore repeat or omit a summary range. This
is accepted by ADR-0009 and does not provide cross-process coordination.

### 6.4 Long-term Memory cache

- Workspace State startup 原子创建缺失模板；Memory Manager 读取并保存当前 string snapshot。
- chat 和 user Schedule Job 的 System Prompt 使用该 snapshot。
- `/memory` 和 Dream 每次通过 Memory Manager 读取磁盘最新文件。
- Dream 成功编辑后由 Memory Manager 刷新 snapshot；正在运行的 Agent Run 保持启动时快照。
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

- `source` 只能是 `user` 或 `system`；User Job ID 必须是 UUID4，内置 System Job 可以使用保留 symbolic ID。当前唯一支持的 System Job 是 `job_id="dream", source="system"`；System Job 不进入公开 list/remove。
- `schedule.kind` 只能是 `at`、`every` 或 `cron`；未选择的 schedule 字段必须显式为 `null`。
- `at` 使用带 UTC offset 的 RFC 3339 毫秒时间；`every` 使用正整数秒；`cron` 使用规范五字段表达式和规范 IANA timezone。
- `state` 只记录 `never-run`、`latest-ok` 或 `latest-error` 允许的字段组合；running、next run 和 history 不持久化。
- 解析拒绝重复 object key、非 canonical 值、非法 schedule/state、重复 Job ID 和非精确 field set；任何损坏都阻止 Runtime 启动并输出 `schedule_state_error` 与路径。
- Store 在 Runtime-local lock 内构造 immutable candidate，严格序列化后通过同目录 atomic replacement 发布；写失败保留旧 authority 并 fault Store。
- User Job 的 add/list/remove 由 `schedule` Tool 管理且不请求确认；Schedule Agent context 拒绝 add，list/remove 仍可用，公开定义按创建时间和 Job ID 稳定排序。
- Schedule Service 初始化后、启动 dispatcher 前，CLI 注册 Dream System Job。Store 中不存在该 ID/source 时创建；ID/source 和配置 cron/本地 IANA 时区一致时跳过；cron 或时区变化时更新 schedule 并保留 ID/state；同 ID 不同 source 时以 `schedule_state_error` 阻止启动。Dream identity、注册和 dispatch 忽略 `message`。
- User Job 通过当前 Agent Loop 的共享 Agent Runner/Gateway 使用 `schedule` route；Schedule Session ID 派生为 `schedule_<job_id>`，Session 首次产生消息时才落盘到 `schedule-sessions/`。Dream System Job 不创建 Schedule Session、不进入 Agent Loop，直接调用 `Dream.run()`。
- Dream 无 pending summary 或成功处理记为 `ok`，安全失败记为 `error`；Dream 已运行时按重叠 occurrence 跳过且不修改 Job state。下一 cron 正常运行，不自动重试失败 batch。
- Schedule Service 在一个 dispatcher 中处理动态变更、at/every/cron、重叠跳过、不同 Job 并发、Session replacement pause/resume 和 Runtime shutdown；不发送前台 `OutboundMessage` 或通知。

## 8. Prompt、Runtime Context 与预算

### 8.1 Chat 与 Schedule System Prompt

chat 和 schedule 的共有 system-level context 按以下固定顺序组装：

1. 基础 Markdown System Prompt 直接声明内置 identity，并包含 normalized absolute Workspace 与 composition-root 注入的 Agent Home path。
2. host Runtime metadata，置于 `## Runtime` Markdown 章节，固定包含 `platform.system()`、`platform.machine()` 和当前 Python version 的结果。
3. 固定 Tool Catalog 的中文 guidance，置于 `## Tool 使用指南` Markdown 章节；Tool 名称保持不变。
4. 完整读取 runtime-startup Long-term Memory snapshot，并经固定 Markdown 投影后置于 `## Long-term Memory` 章节：删除首行精确的 `# Long-term Memory` 及其后一个 Markdown 分隔空行，再对剩余文本全局执行 `##` 到 `###` 的字面替换。

User Configuration 不得插入或替换 identity/system prompt。缓存的 OpenAI-format Tool schema snapshots 通过 provider 的结构化 tools 字段发送，不把 JSON schema 重复拼入自然语言 guidance。

Foreground chat 在上述共有部分之后按当前 Runtime Generation Skill Snapshot order 追加一个 `## Skill Catalog` Markdown section 和 fenced JSONL block。每个 Skill 是独占一行的 compact JSON object，字段顺序固定为 name、description、path；JSON 文本中的反引号以及 `&`、`<`、`>` 使用 Unicode escape，确保 metadata 不能产生 literal block delimiter。该 block 只指导模型使用普通 `read_file` 读取已知 canonical absolute path；模型需要更多内容时可按 `offset`/`limit` 继续分页，不需要证明 EOF。

当 Skill Snapshot 的 always-loaded subset 非空时，Foreground chat 随后按相同 Snapshot order 追加一个 `## Always-loaded Skills` Markdown section 和 fenced JSONL block。每个 opted-in Skill 是一行 compact JSON object，字段顺序固定为 name、body；body 字段承载逐字符一致的完整 `SKILL.md` document。document 的 frontmatter、换行、引号、反斜杠及任意文字均由 JSON 字符串编码承载，反引号以及 `&`、`<`、`>` 使用 Unicode escape。Runtime 不做 raw interpolation，也不截断 document。Foreground consolidation/budget projection 与最终 chat request 使用完全相同的编码后 prompt。该 always document 只进入 Foreground chat：Schedule、Session title、Task Framing、实际 Conversation Summary provider 和 Dream 均接收 `0` 个 Skill document；Summary 的 foreground budget projection 可包含同一完整 foreground prompt，但不把 document 发送给 Summary provider。Schedule prompt 不追加 Skill metadata 或 document。

Foreground 的 metadata projection 与最终 chat request 使用同一 Catalog block；这不改变 Conversation Summary provider 的独立 prompt，后者仍接收 `0` 个 Skill metadata 或 body。

Foreground System Prompt 不包含 Blackboard guidance。Blackboard 只通过合格普通 foreground turn 的 current user message 投影，Tool schema、Permission Policy、Tool Confirmation 与 Tool Gateway 继续作为代码级权威边界。

手动 Skill invocation 是独立的 foreground projection path：只有 raw input 在字符 `0` 以 `/` 开始、且第一个
Unicode whitespace 之前的 token 与 Catalog 中的 Skill name 完全一致（区分大小写）时才匹配。无 delimiter
时 request 为空；匹配 delimiter 只移除一个，其后的空格、换行和其它字符逐字保留。匹配后 Agent Loop
直接读取当前 Runtime Generation 的 immutable Skill Snapshot 中已经验证并冻结的完整 document，不再次访问文件系统。
磁盘上的 Skill 在当前 Session 中发生删除、修改或失效不会改变本次 projection；首次启动或任意 `/resume`
构造新 Agent Loop 时才由新的 Skill Loader 重新发现、完整读取并校验候选。

### 8.2 当前 user input 的 Runtime Context

发给 chat/schedule model 的当前 user message 临时转换为：

````text
## Runtime Context

- Current time: 2026-07-11T15:30:12.123+08:00
- Session ID: <session_id>

## User Input

<raw user content>

## Task goal

Inspect the project

## Completion boundary

Report the findings
````

成功的手动 invocation 不把 raw slash input 放进 model-visible current user，而是使用一个独立的
`## Skill Instructions` fenced JSON object（`name`、`body`）和一个独立的 `## User Request` fenced JSON
string。body 字段承载逐字符一致的完整 `SKILL.md` document；document、request 的换行、引号、反斜杠及 Markdown fence 内容均由 JSON 编码承载，反引号以及 `&`、`<`、`>`
使用 Unicode escape。该 ephemeral projection 只存在于本次
foreground user message，Skill body 不进入 System Prompt；raw slash input 仍是唯一持久化的 Session user
message。若同一 Skill 已在当前 Runtime Generation Skill Snapshot 中属于 always-loaded subset，它仍按 always System contract 出现，manual user
projection 不会去重、覆盖或额外修改该既有 block。

- session JSONL 只保存 raw user content，不保存上述 wrapper。
- 历史 user messages 不重复添加新的 Runtime Context。
- Workspace 已在 identity prompt 中，不在每轮 wrapper 重复。
- staged Blackboard 存在时，current user message 末尾只追加 `## Task goal` 与 `## Completion boundary` 两个 Markdown section；字段值在既有 outer-whitespace trim 后逐字符插入，不做 Markdown escaping 或内部归一化。
- 成功的 Manual Skill Invocation 不投影 Blackboard；已持久化 Blackboard 对该轮不可见且保持原样。
- User Schedule Job 不额外注入旧任务 ID 字段；Dream 使用专门 prompt，不伪装成 chat user input。

### 8.3 专用 prompts

- Session title：只接收规范化后的首条 user content，不注入 Long-term Memory、tools 或 conversation history。
- Task Framing：只接收 previous Blackboard、latest assistant content 和 current raw user input 组成的 compact JSON，使用独立 system prompt 且 `tools=()`。
- Manual Skill invocation：Title 仍可接收 current raw slash input；Task Framing call count、Blackboard projection 与 Task Framing usage 均为 `0`。只有最终 foreground context 接收 typed invocation。手动 body 与 extracted request 在同一个 current `user` message 的不同 JSON-delimited blocks 中各出现一次，不能写入 Session、Blackboard 或 System Prompt；已有 `metadata.blackboard` 不更新也不删除，unknown/non-matching slash input 继续 ordinary foreground input。
- Conversation Summary：只接收本次选中的早期 Session messages，不注入 Long-term Memory、Tool Catalog、Skill metadata 或 Skill body；system prompt 只允许提取 User facts、Decisions、Solutions、Events 和 Preferences 五类关键事实，要求每行一条 Markdown 无序列表，无有价值信息时输出 `None`，并忽略可从仓库源码或 Git 历史直接推断的代码模式。
- Dream：接收 Summary Cursor 后的 batch 和四分区维护规则，并只暴露 restricted memory tools。
- User Schedule Job：使用共有 chat/schedule system composition，把 Job message 作为 Schedule Session 的普通 user message，不接收 Skill metadata。Dream System Job 只触发 Dream，不生成该 prompt。Session title、Task Framing 和 Dream 同样不接收 Skill metadata。

prompt 文本存放在独立、可版本追踪的 package resources；测试断言组成部分和是否注入，不锁死整段自然语言文案。

### 8.3a Shared Slash Completion

Terminal Conversation reuses one presentation-only completion surface for Management Commands
and Skill metadata. It shows candidates only when the composer text starts with `/` at character
zero and contains no character for which `str.isspace()` is true. The five Management Commands
remain first in their fixed order, with these stable labels and descriptions: `/config - View User
Configuration`, `/status - View Runtime Status`, `/resume - Resume a Conversation Session`,
`/memory - View Long-term Memory`, and `/dream - Process pending Conversation Summaries`. Valid
Skills follow in the current Runtime Generation Skill Snapshot order and use `/name - description`.

The display label is independent from the insertion value. Skill descriptions are user-controlled:
their whitespace runs are folded to one ASCII space for a markup-disabled, single-line OptionList
label, with no change to the retained SkillMetadata. The UI applies no-wrap/ellipsis rendering so
long labels do not change candidate row height or obscure the composer. A Management Command
inserts its original command token; an exact Management Command Enter selection may submit through
the existing dispatcher. A Skill selection through mouse or Enter inserts exactly `/<name> `,
closes the popup, restores input focus, and creates zero Message Bus inbound messages; it never
submits or dispatches the Skill. Tab is not intercepted by the completion surface and does not
accept, replace, or submit any highlighted candidate. A Management prefix selection only
completes the composer. The current Agent Loop supplies only an ordered
`tuple[SkillMetadata, ...]` presentation projection; generation rebind replaces that UI projection
and clears old candidate state after the replacement Agent Loop has built and preflighted a new
Skill Snapshot.

### 8.4 Context budget 与 consolidation

- 可用输入预算为已解析 chat route 的 `context_window - max_output`。
- Agent Loop 同步构造在读取当前 Long-term Memory、构造 Skill Loader/Snapshot、`ContextBuilder`、固定 Tool schemas 和其他会话内组件之后，但在启动任何 task 前执行 Skill budget preflight。它以空 history 和空 current user content 调用 `ContextBuilder.build_status_messages()`，并与 `/status` 共享 compact JSON 序列化 seam，将真实 System Prompt、当前 user Runtime Context wrapper 和固定十个结构化 Tool schemas 投影为 `RuntimeStatusInput`，再调用现有 `estimate_input_tokens`（所有 UTF-8 bytes 合计后向上取整 `/ 4`）。`estimated == available` 允许，`estimated > available` 抛出独立 `SkillContextTooLargeError`，稳定 code 为 `skill_context_too_large`，document 不截断。每次启动或 `/resume` 都重扫完整 Skill documents 并对新 Snapshot 做 preflight；Session retained history 不参与该 startup configuration check。
- 估算对象包含 system prompt、retained session messages、当前 Runtime Context、user input 和结构化 tool definitions。
- foreground manual invocation 的 body/request 通过 transient typed projection 计入 retained-current budget 与 cutoff；实际 Summary provider 仍只接收选中的 raw historical Session records，不接收手动 Skill instructions 或 request。
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
- chat 用于主对话、Task Framing 和 title；memory 用于 Conversation Summary 和 Dream；schedule 用于 User Schedule Jobs。
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

streaming contract 只向 Agent Loop 暴露：

- `text_delta(delta)`：非空 text chunk，按顺序到达。
- `completed(response)`：恰好一次，包含聚合后的 content、完整 tool calls 和 usage。
- exception：在 `completed` 前终止 stream，由 router 转成统一错误。

adapter 内部负责聚合 provider-specific tool call deltas。Agent Loop 不解析 Anthropic content block 或 OpenAI chunk。

### 9.4 Retry

- 一个逻辑 model call 默认最多 5 attempts。
- delay 为 `min(30, 0.5 * 2^(attempt-1))` seconds，并加入可注入、可关闭的 bounded jitter。
- 有合法 `retry_after` 时使用 `max(backoff, retry_after)`，再限制到 60 seconds。
- auth、invalid request、context overflow、unsupported、cancelled 不 retry。
- rate limit、timeout、connection error、provider unavailable 可 retry。
- tool calls、title fallback 本地处理和 Tool Gateway 不复用 model retry。

## 10. MessageBus、AgentLoop 与 AgentRunner 契约

### 10.1 Public foreground boundary

The CLI composition root owns one foreground `MessageBus` for the whole Runtime Lifetime; the
current `AgentLoop` only uses it. The full-screen Terminal Conversation submits
`InboundMessage` values and consumes `OutboundMessage` values; it does not reach into Session, provider or Tool
objects. The independent `AgentLoop.control` surface owns cancellation,
confirmation callbacks and the current foreground projection. Management Port
and its dispatcher remain a separate management boundary.

Message Bus 的九个 async operations 恰好是 `inbound_snapshot()`、`put_inbound()`、
`get_inbound()`、`pause_inbound_delivery()`、`resume_inbound_delivery()`、
`drain_inbound()`、`put_outbound()`、`get_outbound()` 和 `reset()`；两个同步 public
operations 是 `set_inbound_changed_callback()` 与 `unbind_inbound_changed_callback()`。Inbound 与
Outbound 共享同一个异步 coordination boundary 并分别保持 FIFO。Inbound put/get/drain 在 coordination
内捕获操作后的 immutable tuple，释放 coordination 后同步调用一个可绑定或清除的 callback；Outbound
put/get 不调用该 callback。Snapshot read 不调用 callback；callback failure 只记录并忽略，不回滚 mutation。
`reset()` 在同一个临界区内清空
Inbound 和 Outbound，随后用空 Inbound snapshot 通知 callback；因此任意 `/resume`（包括当前 Session）
可以复用同一 Bus identity 而不会保留旧 generation 消息。Outbound 是无界 single-consumer FIFO。
Message Bus 不拥有独立 close、abort、replay、broadcast、version 或
backpressure lifecycle；Tool result 永不进入 Outbound。

### 10.2 Task Framing and Blackboard

`AgentLoop` 在每个非空且不是 Manual Skill Invocation 的普通 foreground `InboundMessage` 主 Agent Run 准备前调用一次 `TaskFramingEvaluator`。Manual Skill Invocation、Management Command、User Schedule execution、Conversation Summary 模型调用和 Dream 不自行做 Task Framing；Conversation Summary/context preparation 使用合格普通轮已 staged 的同一 Blackboard。

最小值与接口形状：

```python
@dataclass(frozen=True, slots=True)
class Blackboard:
    goal: str
    completion_boundary: str

@dataclass(frozen=True, slots=True)
class FramingResult:
    blackboard: Blackboard | None
    usage_delta: dict[str, int] | None
    status: Literal["resolved", "invalid_response", "model_failed"]

class TaskFramingEvaluator(Protocol):
    async def frame(
        self,
        *,
        previous: Blackboard | None,
        last_assistant_content: str,
        current_user_input: str,
    ) -> FramingResult: ...
```

Task Framing 使用 `ModelRouter.complete("chat", ..., tools=())`，因此复用 chat route 的总共五次 attempt budget、retry、default fallback 与 cancellation，但不传 Tools 或 continuation。`TaskFramer` 与 `AgentRunner` 接收并保存传给同一 `AgentLoop` 的同一个 Model Router 对象；Router 的构造、Runtime Lifetime ownership 与最终 close 仍只属于 CLI composition root。模型只接收 previous Blackboard、Session 中最新 assistant message 的完整 content 和 current raw user input 组成的 compact JSON。

返回 decision 只能是：

- `keep`：使用 previous Blackboard；previous 为 `None` 时该 decision 无效。
- `replace`：提供恰好 `action`、`goal`、`completion_boundary`；两个值 trim 后必须非空。
- `clear`：清除 Blackboard，且不得附带 goal/boundary。

解析器按顺序接受完整 raw JSON、一个 Markdown fenced JSON，或外层 prose 中的第一个平衡 JSON object；多个/破损 fence、多余字段、错误类型或违反 action invariant 都是 `invalid_response`。`resolved` 必须有四字段 usage，Blackboard 可为 `None`；`invalid_response` 必须有 usage 且 Blackboard 为 `None`；`model_failed` 的 Blackboard 和 usage 都为 `None`。

resolved Blackboard 被 staged 给当前 foreground Agent Run。Model-visible current user message 保留 raw input，并在末尾按以下逐字符模板追加 Markdown；Session user message 只保存 raw input。Blackboard 不进入 Outbound，不允许 Tool 读写，也不能改变 Tool Confirmation 或其他安全判断。

```markdown
## Task goal

{task_goal}

## Completion boundary

{completion_boundary}
```

Agent Loop 只通过一次 `Session.append_messages()` 把主 Runner increment、Task Framing usage 和 `metadata.blackboard` update/removal 一起提交。正常 `failed`、`cancelled` 和 `max_iterations` 结果含有已接受且修复后的 increment，因此仍提交 staged state。Context/Summary/Runner 准备失败或取消保留 previous Blackboard；`invalid_response` 或 `model_failed` 在主 increment 成功提交时清除当前 Blackboard。

Manual Skill Invocation 与以上 framing/commit 分支互斥：该轮不 decode previous Blackboard、不读取 latest assistant content 用于 Task Framing、不调用 evaluator，也不向 Context、Summary budget 或 Runner 投影 Blackboard；Task Framing usage 为 `None`，`metadata.blackboard` update/removal 均为 no-op。下一次合格普通输入才重新读取并评估此前保存的值。

### 10.3 AgentRunner

`AgentRunner.run()` is the sole bounded, Session-independent model/Tool ReAct execution
boundary. Every request that requires repeated model-and-Tool iteration invokes it using
that invocation's route, messages, Tool Gateway, output and confirmation callbacks, result
externalizer, cancellation policy, iteration limit, and failure policy. The current lanes are
foreground Agent Runs through `chat`, User Schedule Jobs through `schedule`, and Dream through
`memory`. Each Agent Loop owns one Runner shared by its foreground and User Schedule work;
Dream owns a separate Runner instance and restricted Gateway while reusing the same bounded
ReAct implementation. Model requests that require only one completion and no ReAct loop use
the Model Router directly.

Each Runner invocation returns an `AgentRunnerResult`. Agent Run is not a synonym for Runner
invocation: it remains the domain term for one complete execution from input acceptance through
Summary/context, one Runner invocation, Session increment and persistence request. Repair
construction, Tool Result externalization and iterator cleanup are private implementation
helpers and are not package interfaces.

### 10.4 Sparse outbound protocol

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
the current `AgentLoop.run_schedule_job` callback and does not publish foreground messages.

Streaming metadata is sparse and mutually exclusive on each message:
`{"_stream_delta": True}` marks one reasoning/response fragment,
`{"_stream_end": True}` marks the end of one segment, and
`{"_streamed": True}` marks the end of the whole foreground Agent Run. No marker is
combined with another marker.

### 10.5 Management Port

最小接口：

```python
class ManagementPort(Protocol):
    async def config_view(self) -> ConfigView: ...
    async def status(self) -> RuntimeStatus: ...
    async def resumable_listing(self) -> SessionListingReport: ...
    async def resume(self, session_id: str, *, force: bool = False) -> ResumeResult: ...
    async def memory_view(self) -> str: ...
    async def dream(self) -> DreamResult: ...
```

- `resumable_listing` 的 `sessions` 仅包含当前 Workspace 的 id、title、created_at、
  updated_at、message_count，并用 `skipped_count` 汇总损坏或不可读项。
- `resume` 再次验证 session 属于当前 Workspace，不信任 UI 传入值；`force` 仅由
  Terminal 在确认 active foreground replacement 后传入，Management 不构造 Runtime。
- `config_view` 已脱敏；Management Port 永不返回 plaintext API key。
- `memory_view` 读取磁盘，不返回 runtime cache。

### 10.6 Agent Runner and Provider Boundary

`AgentRunnerResult` 的 `messages` 只包含本次调用生成的 assistant/Tool increment，
不包含 initial messages 或当前 user message。`final_content` 是本次调用的最终文本；
`usage` 恰好包含 `model_calls`、`input_tokens`、`output_tokens`、`total_tokens`，均为
非负整数且 `total_tokens == input_tokens + output_tokens`。`completed` 必须没有 error；
`failed` 必须携带安全的 `ErrorInfo`；`cancelled` 必须使用 `turn_cancelled`；
`max_iterations` 必须使用 `agent_iteration_limit`。正常失败或取消后的 increment
必须是 Provider-valid 的 assistant/Tool 序列。

一次 iteration 恰好是一次 model call 加上该响应请求的全部 Tool calls，Tool 按
Provider 原始顺序逐个执行；Provider 内部 retry 不计入 iteration。默认和最小
`runtime.max_iterations` 都是 `50`。第 50 次响应请求 Tools 时必须完成全部 Tools，随后
若没有请求 normal cancellation，则返回 `max_iterations`，不进行第 51 次 Provider call，
并使用以下固定文案；normal cancellation 优先：

`MyClaw 本轮对话已经达到最大循环次数，仍没有输出最终结果。可以再次尝试本次请求或者尝试给出更明确的任务目标。`

正常取消使用固定文案 `MyClaw 已取消本轮对话。`。Provider 返回的 reasoning 通过
`ReasoningDelta` 只在 live callback/foreground Outbound 中可见；Anthropic thinking/signature
blocks 与 OpenAI-compatible `reasoning_content` 可形成 opaque `ModelContinuation`，但该
continuation 只在同一个 Tool loop 的下一次 Provider call 中使用，不进入 Session-shaped
messages、Outbound 或持久化。

### 10.7 Schedule and Lifecycle Boundary

`ScheduleService` 在 CLI composition root 中创建并拥有 Schedule Store。CLI 给它两个稳定 executor：
User Job executor 每次调用时解引用当前 Agent Loop；Dream executor 直接调用 CLI-owned `Dream.run()`。
CLI 在 Service 初始化后、dispatcher 启动前注册或校正内置 Dream System Job。Foreground 与 User
Schedule execution 使用当前 Agent Loop 的同一 Gateway identity 和同一 Runner identity，但每次
User Schedule execution 拥有独立的 Schedule Session、context、cancellation 和 externalizer Session ID。
Dream 使用自己的 Runner/Gateway，通过 `memory` lane 调用同一个 `AgentRunner.run()` 实现，且不创建
Schedule Session。所有 Schedule execution 都没有 confirmation channel 或 foreground Message Bus
projection，任何 Schedule output 都不进入 foreground Outbound。

Schedule execution 通过 ContextVar token 设置 `ScheduleTool._in_schedule_job`，并在
`finally` 中 reset；只有递归 `add` 被拒绝，foreground `add` 以及 Schedule `list`/`remove`
仍可用。Schedule Artifact root 保持
`.myclaw/artifacts/schedule_<job_id>/<tool_call_id>.txt`，reference shape 保持
`path`、`total_chars`、`preview_chars` 三个字段。

CLI 的 private async root 是唯一 composition root，拥有规范化绝对 Workspace `Path`、`WorkspaceState`、
一个稳定 Message Bus、Model Router、Memory Manager、Dream、Schedule Service、Management
Service/Dispatcher、Terminal application 和可替换的 current Agent Loop reference。同步 Typer entry
只完成参数/配置边界并进入该 async root；Terminal application 不启动或关闭业务组件。代码中不得存在
`RuntimeHost`、`PreparedRuntime`、`RuntimeBindings`、`prepare_runtime` 或承担相同聚合职责的改名容器。

一个 Runtime Generation 恰好是一个 Agent Loop。Agent Loop 同步构造 Session、Skill Loader、immutable
full Skill Snapshot、Context Builder、Conversation Summary Manager、Task Framer、Tool Gateway、Agent Runner
及其他 Session 内 task；它只接收并使用 CLI-owned outer objects/ports。构造完成后处于 prepared/unstarted
状态，并在启动 task 前完成同步 preflight。Skill Loader 在每个 generation 重新发现目录、完整读取/校验正文，
丢弃并安全记录无效候选，再保存包含 metadata 和全部正文的 immutable Snapshot；同一 generation 的 manual
和 always projection 都只使用 frozen document。普通模型主动调用 `read_file` 仍保持实时 Tool 语义。

The final linearization refinement formed during later implementation review is recorded here; it is not a restatement of the original parent issue wording. In this contract, target preparation is a precondition and includes target construction plus synchronous `preflight()` before destructive cutover. For a successful target, the exact source-backed sequence is:

`quiesce_for_rebind -> pause_and_drain -> current unavailable -> old abort/drain -> bus.reset() -> rebind_agent_loop -> target.start() -> publish current -> schedule_service.resume()`

`current unavailable` means the CLI clears its current reference before old-loop abort/drain. The target is successfully started and activated before that reference is published; Schedule resumes only after publication. Any target construction or preflight failure is fatal, terminates Terminal Conversation, and is handled by the CLI `finally` path. Selection of the current Session also performs the full flow. Schedule Service, Dream, Memory Manager, Model Router and Message Bus are not closed or rebuilt during replacement.

`pause_and_drain()` 在线性化 paused state 后取消并等待 dispatcher、所有 active User/System
Schedule Job run task 以及已登记或在 run cancellation 清理期间新建的 terminal-commit task；caller
cancellation 不会截断共享 pause barrier，只有 owned task 全部归零后才向 caller 传播。
未提交的 `at` terminal removal 保持 persisted pending，并在 `resume()` 后重试一次；取消的 `every`
terminal commit 保留 reservation 已推进的 monotonic deadline，取消的 `cron` terminal commit 保留已推进的
Cron cursor，因此二者都只在下一个正常 occurrence 执行。已经完成 Store publication 的 terminal commit
保持其 persisted state 与完成时刻 cadence。直接调用 `ScheduleService.close()` 仍等待已开始的 terminal
commit 而不取消；CLI shutdown 先调用 `pause_and_drain()`，因此使用 pause 的取消语义。

After Terminal `run_async()` returns, the actual CLI shutdown call chain is `Management deactivate -> Schedule pause_and_drain + close -> pending/active Agent Loop abort or close -> Dream close -> Model Router close`. Terminal exit/unmount cleanup has already run at the first boundary; the CLI does not call a separate Terminal business-component close. Accepted Tool/Artifact/Memory/Schedule side effects are not rolled back.

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

- Agent Loop 在初始化时以规范化绝对 Workspace `Path`、Schedule Service 和 canonical Agent Home Skill root 构造一个共享 `ToolGateway`；不存在 `Workspace` wrapper class。Gateway 自己一次性构造固定十工具 Catalog，不暴露注册入口。Schedule Service 自己创建并持有 Store/Job management ownership，Gateway 不接受第二套 Schedule Gateway。
- Tool 调用不接收 session ID、Agent Home、lane、approval flag 或通用 execution context。
- 没有独立 `Security` 模块；公共路径、DNS、截断和 Artifact 边界由 BaseTool 或共享小 helper 提供，具体 Tool 保留 capability-specific 规则。
- Dream 使用合法且独立的专用 Tool Gateway，只注册 Long-term Memory read/edit Tool；
  它不复用 foreground/User Schedule 的共享 Gateway，并拥有独立 Agent Runner instance 与
  `memory` lane，但复用统一的 `AgentRunner.run()` bounded ReAct implementation。
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
- 仅 `read_file` 对 canonical `~/.myclaw/skills` root 下的已知 path 免确认；Skill root 缺失时不创建目录，缺失目标继续返回普通 `read_file` error。
- `read_file` 的 Skill-root 判断在请求 path canonical resolve 后进行；解析到 root 外的 symlink/reparse target 仍请求一次性确认。
- Foreground model 读取已发现的 Skill 正文仍使用普通 `read_file`；长正文沿用既有 pagination 和 Tool Artifact/Session persistence 规则，不引入 Skill-specific Tool、body cache 或 EOF check。
- 解析到 Workspace 外且不在上述 canonical Skill root 的 file path：请求一次性确认；无确认通道或拒绝时 refused。
- 当前 session 的 artifact directory：按相同 Workspace 路径规则处理，没有额外 MyClaw 权限层。

共享路径 helper 使用 host path semantics 解析 Workspace root 和请求目标，再按 canonical path 判断是否在 Workspace 内；`read_file` 额外判断 canonical Skill root，内部目标直接交给具体 OS 操作，外部目标先确认。没有中央 `Security` 类型筛选或额外 device/named-pipe/non-regular blanket policy。

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

| Capability | Foreground | User Schedule Job | Dream |
| --- | --- | --- | --- |
| Workspace read/list/search | allow | allow | 不在 catalog |
| Workspace write/edit | allow, subject to OS permissions | allow, subject to OS permissions | 不在 catalog |
| Long-term Memory read | allow | allow | allow |
| Long-term Memory edit | allow, subject to OS permissions | allow, subject to OS permissions | allow，仅精确文件 |
| Current-session artifact read | allow | allow | 不在 catalog |
| Canonical Agent Home Skill root (`read_file` only) | allow | allow，限已知 path | 不在 catalog |
| Agent Home internal state read/write | 按普通 Workspace path rules | Workspace 内 allow；外部 refused/error | 不在 catalog；owned stores 自行操作 |
| Exec | allow or one-shot confirmation by concrete safety check | allow；需要确认时 refused | 不在 catalog |
| WebSearch/WebFetch | allow or one-shot confirmation by concrete target check | allow；需要确认时 refused | 不在 catalog |
| Schedule add/list/remove | allow | allow, except `add` is refused | 不在 catalog |
| Workspace 之外 | one-shot confirmation or OS error | refused/error（无确认通道） | refused/error |

上述 Skill-root carve-out 只扩展既有 `read_file` 的 generic path safety；它不提供 Skill discovery、Skill invocation、目录监听或新的 Tool API。Foreground 与 Schedule 共用同一 Gateway，因此 Schedule 可以读取已知 canonical Skill path，但没有 metadata/prompt 投影或确认通道。

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
| `memory_task_running` | Dream 不重入（保留既有稳定 code） | 否 |
| `schedule_state_error` | Schedule state 损坏或不安全 | 否 |
| `skill_context_too_large` | always-loaded Skill document 的最小真实 Foreground request projection 超出 `context_window - max_output` | 否 |

CLI exit code：成功 `0`，配置/用法 `2`，runtime startup/persistence `1`，Ctrl+C 结束当前 turn 但 Terminal Conversation 继续时不退出进程。首次启动或任意 `/resume` 的 Agent Loop 构造/同步 preflight failure 必须由 CLI composition root 捕获，终止 Terminal Conversation，并只通过 `_print_error_info` 输出稳定 code/message，不输出 traceback、底层异常、Skill 正文或敏感路径。

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
- `uptime_seconds` 是当前 Agent Loop/Conversation Session 的 generation uptime；任意 `/resume`（包括当前 Session）后重置，不表示 CLI process uptime。
- token estimate 对下一次 chat request 的 system prompt、retained messages、tool definitions 和 Runtime Context 的 UTF-8 bytes 求和后除以 4 向上取整。
- 没有已持久化 message 的准备中 Session：message count/`last_consolidated`/usage 都为 0。
- Schedule Service health 可附加非持久化 warning，但不得取代当前 Session required fields。

## 15. 最小 Session/Provider/Tool 接口

以下签名用于限定职责，不要求使用特定 ABC library。Conversation Session 的
identity、messages、metadata、`last_consolidated` 和 complete snapshot
persistence 由同一个 active `Session` instance 负责；Session 不暴露
filesystem acknowledgement，也不承担 MessageBus/AgentLoop 或 Model
Provider 职责：

```python
class Session:
    @classmethod
    def create(
        cls,
        workspace_state: WorkspaceState,
        *,
        now: Callable[[], datetime] | None = None,
        new_uuid: Callable[[], UUID] | None = None,
        partition: SessionStoragePartition = SessionStoragePartition.FOREGROUND,
        job_id: UUID | str | None = None,
    ) -> "Session": ...
    @classmethod
    def load(
        cls,
        workspace_state: WorkspaceState,
        session_id: str,
        *,
        partition: SessionStoragePartition | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> "Session": ...
    def add_message(self, role: str, content: str, **fields: JsonValue) -> None: ...
    def append_messages(
        self,
        messages: list[dict[str, JsonValue]],
        *,
        metadata_updates: dict[str, JsonValue] | None = None,
        metadata_removals: tuple[str, ...] = (),
        usage_delta: dict[str, int] | None = None,
    ) -> None: ...
    def update_metadata(self, metadata: dict[str, JsonValue] | None = None, **updates: JsonValue) -> None: ...
    def persist(self) -> None: ...
    def close(self) -> None: ...
    def abandon(self) -> None: ...

class MemoryManager:
    @property
    def long_term_path(self) -> Path: ...
    async def append_summary(self, content: str, timestamp: datetime) -> SummaryEntry: ...
    async def claim_summaries(self, limit: int) -> SummaryClaim: ...
    async def read_long_term(self) -> str: ...
    async def edit_long_term(
        self,
        *,
        old: str,
        new: str,
        replace_all: bool = False,
    ) -> str: ...
    def memory_snapshot(self) -> str: ...

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

当前契约测试固化以下 fixtures/snapshots：

- 默认 config template 与一个完整有效 config。
- config unknown field、unknown route、unknown protocol 和 redaction cases。
- 当前 Session header 与 user/assistant/tool message shape 的 exact-key assertion。
- 完成、中断、model failure、tool failure 后的完整 Session JSONL snapshots，以及 ordered async persist 和 bounded close。
- summary schema exact-key assertion、index/cursor 起点和 batch 行为。
- Schedule model strict round-trip、Schedule state strict-load、legacy state untouched 和 atomic mutation。
- MessageBus sparse outbound schema、terminal marker 以及 AgentLoop control/Future 语义。
- Blackboard/FramingResult strict shape、Task Framing decision table、current-input-only projection、usage/metadata atomic commit 和非前台路径排除。
- Model Provider scripted transcript：text deltas、tool call deltas、usage、retry-after、timeout、cancellation。
- 固定 Catalog、generation-scoped full Skill Snapshot/manual frozen projection、foreground-only metadata projection、BaseTool preparation order、file path boundary、Exec/Web confirmation 和 WebFetch redirect/IP cases。
- CLI composition ownership、Agent Loop session ownership、initial/resume preflight failure、same-Session replacement、stable Message Bus identity 和 atomic inbound/outbound reset。
- Dream System Job registration/reconciliation、direct Dream dispatch、User Job current-loop dispatch、replacement pause/cancel/await/rebind/resume 顺序和 no-Schedule-Session invariant。
- complete atomic JSONL replacement、缺少 trailing newline、middle corruption、旧 schema rejection，以及 Summary/`last_consolidated` crash divergence。

契约测试断言稳定 code、结构和文件内容；终端文案除脱敏与必需信息外不做全文 snapshot，以免实现被展示细节锁死。

## 17. 确认记录

D01-D18、Session snapshot、固定 Tool Catalog、CLI composition root、Message Bus/Agent Loop/Agent Runner、Dream System Job 以及 Session Blackboard Task Framing 均为当前已接受契约。本文 `TOOL_SCHEMA` 与各持久化 schema 是 Python 类型、实现和 contract fixtures 的直接输入。
