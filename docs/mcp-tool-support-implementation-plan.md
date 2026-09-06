---
status: accepted
---

# MCP Tool 支持实施方案

## 文档状态

- 架构状态：已完成 grilling，待实施评审
- Implementation status：未开始；本方案评审通过前不得修改生产实现
- 决策来源：[ADR-0020](adr/0020-expose-configured-mcp-tools-through-tool-gateway.md)
- 产品行为来源：[PRD](myclaw-personal-agent-prd.md)
- 运行时契约来源：[Runtime Contracts](myclaw-runtime-contracts.md)
- 领域语言来源：[CONTEXT.md](../CONTEXT.md)

本文把已确认的“用 `MCPTool` 将 MCP Tool 包装为内置 Tool”拆成可独立开发、独立测试、独立合并的 Task。实现顺序允许在每个 Task 合并后保持仓库可测试；不得在实施中重新引入动态注册、Tool 重载命令或另一套 Agent runtime。

## 1. 目标与非目标

### 1.1 目标

1. 使用官方 MCP Python SDK v2（依赖约束 `mcp>=2,<3`）消费 stdio 和 Streamable HTTP Server。
2. 在 CLI Runtime Lifetime 内建立和关闭 MCP Client；在每个 Runtime Generation 内冻结成功发现的 MCP Tool Snapshot。
3. 使 `MCPTool` 与 Built-in Tool 共用 `BaseTool`、Tool Gateway、Model `tools` 参数、确认/拒绝边界、执行边界和结果外化边界。
4. 保持 Built-in Tool 现有参数强转、默认值、未知字段过滤、Schema 校验和安全确认行为。
5. 让 MCP Tool 转发完整 `dict[str, Any]`，不做参数类型强转、不做 inputSchema 校验、不删除额外字段。
6. 在每次 Model 请求时逐个调用 Tool 的 `to_schema()`，构造新的 `list[dict[str, Any]]`；不缓存 Gateway 聚合 schema。
7. 统一所有本地上下文超限为 `model_context_overflow`，并按本次请求实际发送的完整内容计算预算。

### 1.2 非目标

- 不支持 MCP Resource、Prompt、Sampling、Tasks、`input_required`、动态 list-changed 订阅或 Tool reload 命令。
- 不读取 `CallToolResult.structured_content` 或 `outputSchema`；本次只读取 `content`。
- 不引入 OAuth、用户可配置进程环境、`env`/`secret_env` 配置项、自动重试或全局 Tool 执行锁。
- 不让 Dream 使用 MCP Tool；Dream 继续使用受限 Tool Gateway。
- 不在 `/status` 增加 MCP 字段、连接列表、健康状态或 Tool 计数。

## 2. 已确认的产品与领域决策

### 2.1 User Configuration

每个 Server 只允许一个逻辑配置项，名称即 `mcp_name`：

```toml
[mcp.servers.filesystem]
enabled = true
transport = "stdio"
command = "uvx"
args = ["mcp-server-filesystem", "."]
cwd = "."
connect_timeout = 30
call_timeout = 60

[mcp.servers.search]
enabled = true
transport = "streamable-http"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer secret" }
connect_timeout = 30
call_timeout = 60
```

- `mcp_name` 使用 `[a-z0-9][a-z0-9_-]{0,63}`。
- stdio 不接收 `env` 或 `secret_env`，直接继承 MyClaw 进程环境；`cwd` 省略时为 Workspace，相对路径也相对 Workspace，绝对路径原样使用。
- Streamable HTTP 可在同一 Server 项下声明静态 `headers` 字典。
- 合法配置但连接、初始化或发现失败的 Server 被忽略，不阻塞启动；终端给出一行脱敏提醒，process log 记录 `mcp_name`、阶段、异常类型和 traceback，不记录配置内容。
- MCP 字段错误只跳过对应 Server，不进入运行时配置内存；整个 TOML 无法解析时仍按现有 `config_parse_error` 阻止启动。
- `myclaw config` 和 `/config` 显示被忽略 Server 的脱敏诊断；headers 值不显示。
- 默认配置模板加入注释掉的 stdio 与 Streamable HTTP 示例，不启用任何 Server。

### 2.2 Tool schema 与参数

- `BaseTool.parameters`、`to_schema()` 返回值、Provider Tool schema 接口和 Gateway schema 列表全部使用 `dict[str, Any]`。
- 每个 Tool 保留自己的 `parameters` 字典；ToolGateway 不缓存 `_schemas` 聚合列表，`.schemas` 每次访问都按 Catalog 顺序调用每个 Tool 的 `to_schema()`。
- `to_schema()` 是纯投影：

```python
def to_schema(self) -> dict[str, Any]:
    """OpenAI function schema."""
    return {
        "type": "function",
        "function": {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        },
    }
```

- Built-in Tool 不保留 Schema 实例；每次 `prepare()` 从类声明重新构建临时 Schema，以保持已有 cast/default/filter/validate 行为。
- `MCPTool` 保存发现到的原始 inputSchema 字典的 nullable 归一化结果。加载时只接受可 JSON 序列化、根 `type == "object"` 的字典；不在调用时验证参数。
- `nullable` 递归处理 dict/list：删除关键字；`nullable=true` 且 `type` 为字符串时改成包含 `null` 的 type 数组；`type` 已为数组时追加 `null`；无 `type` 时包装为 `anyOf: [原节点, {"type": "null"}]`；其他 JSON Schema 结构原样保留。

### 2.3 BaseTool 与 Gateway 执行边界

`BaseTool.prepare()` 保持 `final`，固定顺序为：

```text
prepare(arguments: dict[str, Any])
  -> prepare_arguments(arguments)
  -> validate_arguments(**prepared)
  -> check_safety(**prepared)
  -> (prepared, safety_reason)
```

`PreparedToolCall` 删除。`execute()` 不再是强制 abstract method；`BaseTool.__init_subclass__()` 要求具体 Tool 至少覆盖 `execute()` 或 `execute_prepared()`。

```python
async def execute_prepared(self, arguments: dict[str, Any]) -> str:
    return await self.execute(**deepcopy(arguments))
```

Built-in Tool 使用默认实现；`MCPTool` 只覆盖 `execute_prepared()`，Gateway 不判断 Tool 类型。

### 2.4 MCP 生命周期与 Snapshot

```text
ConfigLoader
  -> valid MCPServerConfiguration + ephemeral diagnostics
CLI Runtime Lifetime
  -> concurrently connect enabled Servers
  -> paginate tools/list
  -> validate/normalize each Tool and allocate model name
  -> MCP Tool Snapshot
Runtime Generation
  -> ToolGateway(Built-in Tools + Snapshot)
  -> each Model request: [tool.to_schema() for tool in catalog]
  -> ToolGateway.call -> BaseTool.prepare -> execute_prepared
```

- 初次启动在创建初始 Agent Loop 前并发连接所有合法 enabled Server。
- `/resume` 在暂停旧 Generation 前并发重连失败 Server并生成候选 Snapshot；失败不影响旧 Generation，随后沿用既有 replacement transaction。
- 健康连接与已发现 Tool 定义跨 `/resume` 复用；仅失败 Server 重连并重新发现。健康列表变化需要进程重启。
- 只有 SDK 明确报告 session/transport closed 才把 Server 标记待重连；`isError=true` 和调用超时永不改变可用性。
- 关闭顺序固定为停止输入、pause/drain Schedule、abort/close Agent Loop、close MCP Client、close Dream、close Model Router；清理错误并入现有聚合错误。

### 2.5 命名、结果和错误

- 优先名称 `mcp_<mcp_name>_<remote_tool_name>`；若超过 64 个字符则尝试 `mcp_<remote_tool_name>`；不替换字符、不截断。候选必须匹配 `[A-Za-z0-9_-]{1,64}`，仍不可用则忽略 Tool。
- Built-in Tool 先排序；MCP Server 按 `mcp_name` 排序，Server 内 Tool 按远程名称排序。候选冲突保留排序中第一个，其余忽略。
- 缺少 description 时使用远程 Tool 原始名称。
- MCPTool 只读取 `result.content`：

```python
parts = []
for block in result.content:
    if isinstance(block, types.TextContent):
        parts.append(block.text)
    else:
        parts.append(str(block))
return "\n".join(parts) or "(no output)"
```

- `isError=true`：以上述文本直接包装为 `ToolError`。
- 调用超时：包装为带 timeout 原因的 `ToolError`，不改变 Server 状态。
- session/transport 已关闭：包装为连接不可用的 `ToolError`，并标记 Server 供下次 Generation 重连。
- 其他协议/SDK/代码异常：交给 Gateway 通用失败路径，不把原始异常返回模型。
- 取消沿用 Built-in Tool 语义，`CancelledError` 原样传播。

### 2.6 全量上下文预算

每次模型请求和每次压缩判断都计算本次请求将发送的全部内容：

```text
System Prompt + 全部历史/当前消息 + 全部 Tool schema
```

不可压缩内容超出 `context_window - max_output` 时，返回统一错误：

```text
code = "model_context_overflow"
message = "Model request context exceeds the available input budget."
```

所有本地上下文超限统一为 `model_context_overflow`；`/status` 保持既有字段形状，但估算值包含活动 Generation 的完整 Tool schema，不增加 MCP 专属字段。

## 3. 架构影响与接口定义

### 3.1 受影响模块

| 模块 | 影响 |
| --- | --- |
| `myclaw/config/config.py` | 增加 MCP Server 配置解析、逐项容错、脱敏诊断投影 |
| `myclaw/tools/base.py` | dict 参数契约、临时 Built-in Schema、final prepare 返回 tuple、执行扩展点 |
| `myclaw/tools/tool_gateway.py` | 动态 Snapshot 注入、取消 schema 缓存、统一 `execute_prepared` |
| `myclaw/tools/mcp_*.py` | SDK v2 transport/session、发现、命名、nullable、结果与连接状态 |
| `myclaw/agent/runner.py` | 每次 Model 请求重新构造 Tool schema list |
| `myclaw/agent/loop.py` | 接收 generation Snapshot、统一 context overflow、去除 MCP status 字段 |
| `myclaw/memory/conversation_summary.py` | 压缩 trigger/cutoff 预留完整 Tool schema 预算 |
| `myclaw/terminal/cli.py` | Runtime Lifetime MCP manager、启动失败终端提醒、resume 前重连、关闭顺序 |
| `myclaw/errors.py` | 删除两个旧 context overflow code |
| Provider/测试 fixture/契约文档 | `dict[str, Any]` schema 类型与动态 list 适配 |
| `pyproject.toml` | 增加 `mcp>=2,<3` |

### 3.2 建议的内部接口

```python
class MCPRuntimeManager:
    async def start(self, configuration: Mapping[str, MCPServerConfiguration]) -> MCPStartupReport: ...
    async def prepare_generation(self) -> MCPSnapshotReport: ...
    async def close(self) -> None: ...

@dataclass(frozen=True, slots=True)
class MCPToolSpec:
    server_name: str
    remote_name: str
    model_name: str
    description: str
    parameters: dict[str, Any]

class MCPTool(BaseTool):
    async def prepare_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]: ...
    async def execute_prepared(self, arguments: dict[str, Any]) -> str: ...
```

实际 SDK 类型只允许出现在 MCP adapter 模块；Tool Gateway、Agent Runner、Provider 和 Session 不直接 import SDK 类型。

## 4. 数据流与失败语义

### 4.1 启动

1. `ConfigLoader` 解析 TOML；全文件语法错误照旧终止。
2. MCP Server 表逐项校验；错误项生成脱敏诊断并跳过，合法项进入 `UserConfiguration.mcp`。
3. CLI 并发建立合法 enabled Server；每个 Server 有独立 connect/discovery timeout。
4. 对返回的分页 Tool 列表排序、校验根 schema、递归归一化 nullable、分配模型名；单个非法 Tool 跳过。
5. 输出每个失败 Server 一行简短提醒；对 connected 但有跳过 Tool 的 Server 输出一条聚合提醒。
6. 将成功 Tool Snapshot 注入初始 Agent Loop；没有成功 MCP Tool 仍正常使用 Built-in Catalog。

### 4.2 一次 Model Tool call

```text
Provider raw arguments (JSON string)
  -> json.loads + top-level dict check
  -> ToolGateway lookup
  -> BaseTool.prepare
       Built-in: reconstruct Schema -> cast/default/filter/validate
       MCP: deepcopy full dict, no local schema validation
  -> safety/confirmation (MCP always no safety reason)
  -> execute_prepared
       Built-in: execute(**arguments)
       MCP: client.call_tool(remote_name, arguments)
  -> content text conversion
  -> ToolResult / Tool Artifact externalization
```

### 4.3 `/resume` 与关闭

`/resume` 先请求 MCP manager 生成候选 Snapshot，再执行当前 CLI 的 pause/drain、旧 Loop abort、Bus reset、目标 Loop preflight、rebind/start；候选失败只令目标 Generation 缺少该 Server。Runtime shutdown 在 Agent Loop 完成后关闭 MCP manager，任何单个 close 失败都进入现有 cleanup error 聚合。

## 5. 原子实施 Task 与量化验收

Task 之间通过清晰接口衔接；每个 Task 都能独立提交、独立运行自身测试，并在合并后保持既有测试可执行。

### T1：配置模型与诊断

**边界**：只修改 User Configuration、默认模板、脱敏 ConfigView 和相关测试；不建立 MCP Client。

**验收**：

- 合法 stdio/HTTP 配置各至少 5 个解析断言；cwd 相对/绝对路径各 2 个断言。
- `env`、`secret_env`、transport 不适用字段、未知字段和非法 name 各至少 3 个用例，均只跳过对应 Server。
- 整文件 TOML 语法错误仍有 1 个 fatal regression test。
- headers 值在 `myclaw config` 与 `/config` 输出中 100% 脱敏；command、args、url、cwd 不被误删。
- 默认模板包含 2 个注释示例且启动后启用 Server 数为 0。

### T2：BaseTool/Gateway dict 边界

**边界**：重构 BaseTool、Gateway 和 fixtures；不接入真实 MCP SDK。

**验收**：

- 全部 Built-in Tool 既有测试通过，且新增 10 个用例覆盖 cast/default/unknown-field/Schema error 行为不变。
- `prepare()` 返回 `(dict, str | None)` 的顺序和取消传播各有断言；仓库中 `PreparedToolCall` 引用为 0。
- Gateway 每次 `.schemas` 访问都重新调用每个 Tool 的 `to_schema()`；新增计数 Tool 测试至少访问 3 次。
- Gateway 只调用 `execute_prepared()`；新增 Built-in 与对象参数 Tool 各 3 个调用断言。
- Tool 模块直接接口不再引用 `JsonObject`、`JsonValue`、typed OpenAI schema；无关 Session JSON 类型保持不变。

### T3：MCP SDK v2 单 Server adapter

**边界**：实现 MCPTool、单 Server transport/session、分页 discovery、nullable 和 result/error 转换；通过 fake SDK seam 测试。

**验收**：

- 官方 MCP Python SDK 依赖 `mcp>=2,<3` 在本 Task 中声明并成功解析；后续 Task 不再负责首次加入该依赖。
- 分页 discovery 至少 3 页、空列表、重复 cursor 各 1 个测试；最终顺序确定且无遗漏。
- stdio 与 Streamable HTTP 各至少 1 条本地真实 SDK 端到端测试，不访问公网。
- nullable 递归 golden cases 至少 10 个，覆盖字符串 type、type 数组、无 type、嵌套 list、anyOf/oneOf/$ref 原样保留。
- 完整参数透传测试至少 5 个：嵌套对象、数组、null、额外字段和远程未声明字段均逐字到达 Client。
- content 转换覆盖 TextContent、两个文本逐行拼接、非文本 `str(block)`、空 content 四类；structured_content 不被读取。
- `isError`、timeout、closed session、其他异常、CancelledError 各至少 2 个测试，错误状态和连接状态符合契约。

### T4：多 Server 命名与 Snapshot

**边界**：实现命名分配、冲突、排序、健康复用和 generation Snapshot；不改 CLI 交互。

**验收**：

- 两个 Server 暴露同名 Tool 时可确定性区分；built-in 优先、Server/tool 排序各有 3 个排列测试。
- 长名称覆盖 64、65、fallback 成功、fallback 失败和非法字符各至少 2 个测试。
- 相同配置重复构建 100 次得到完全相同 schema/name 顺序。
- 失败 Server 重连只发生在 failed 集合；healthy Server 的 client/discovery 调用次数在 `/resume` 后保持不变。
- Snapshot 为不可变 tuple；旧 Generation 在候选 Snapshot 失败时工具集合完全不变。

### T5：CLI Runtime Lifetime 集成

**边界**：把 MCP manager 接入 CLI 初始启动、`/resume` 和关闭；加入终端提示与 process log。

**验收**：

- 合法 Server 并发启动；3 个 Server 中 1 个超时不会阻塞另外 2 个成功 Server。
- 初始失败和 `/resume` 重连失败各有 3 个测试：旧 Generation 可继续、目标 Snapshot 不含失败 Server、终端每个失败 Server 恰好一行。
- close 顺序有 1 个生命周期 spy 测试，顺序严格符合 ADR；close failure 聚合仍保留。
- 日志字段只含 `mcp_name`、阶段、异常类型；敏感配置值和原始异常 message 出现次数为 0。
- `/status` 与 `/config` 现有字段/命令行为回归测试全部通过，status 输出不含 MCP 专属字段。

### T6：全量上下文预算与错误码收敛

**边界**：更新 Agent Runner、Loop、Summary、Provider error mapping、错误类型和契约测试；不改变 MCP transport。

**验收**：

- System Prompt、历史消息、当前消息、Tool schema 四项分别增大时，预算估算均单调增加；每项至少 3 个测试。
- 压缩 trigger、不可压缩检查和 cutoff 各至少 3 个测试，均为完整 Tool schema 预留预算。
- `model_context_overflow` 是唯一上下文超限 code，在代码、文档和测试中的引用均符合统一契约。
- 所有本地超限路径使用完全相同 message；Provider 现有错误映射回归测试通过。

### T7：契约文档与全回归

**边界**：更新 PRD、Runtime Contracts、ADR-0010 supersession、默认 README/开发依赖说明，并执行全套质量门禁；MCP SDK 依赖已由 T3 引入，本 Task 只核对其存在和解析结果。

**验收**：

- PRD、Runtime Contracts、ADR 和 CONTEXT 对 MCP 名称、配置、生命周期、错误、结果和安全语义无互相矛盾条目。
- `mcp>=2,<3` 已由 T3 加入 package metadata，且安装解析成功。
- `pytest` 全量通过；`mypy`、`ruff check`、`ruff format --check` 全量通过。
- 新增 MCP 测试全部不访问公网；真实 transport 测试仅使用本地 stdio/HTTP fixture。
- `rg` 检查生产代码中不存在 MCP reload command、Tool schema aggregate cache、`PreparedToolCall` 或 Tool-specific context overflow code。

## 6. 变更影响评估

- **Provider 请求**：Tool schema 从每轮请求动态生成，但 OpenAI-compatible 和 Anthropic 的字段转换保持不变；Anthropic 仍把 `function.parameters` 映射到 `input_schema`。
- **Session/Tool Result**：MCP 结果在进入现有 text-only Tool boundary 前完成 content 转换，Artifact 和 Session 持久化格式不变。
- **安全**：MCP Server 是 User Configuration 信任边界，所有启用 MCP Tool 无条件执行；不信任远端 annotation，也不新增确认 UI。
- **并发**：Foreground 与 User Schedule 可并发调用共享 MCP Client；不增加 Gateway 全局锁。Dream 不持有这些 Client。
- **失败隔离**：配置/连接/发现失败只减少当前 Snapshot；业务 Tool error、timeout 不改变可用性；真实 session closed 才进入下一 Generation 重连集合。
- **兼容性**：删除两个旧 context overflow code 会影响错误类型、CLI 文案和相关测试，是本次明确批准的全局契约变更；其他稳定错误码和持久化 schema 不变。

## 7. 评审门槛

在用户明确确认本方案后，才能开始 T1；任何生产代码 Task 开始前都必须先提交并评审对应测试矩阵。若实现中需要改变本 ADR 的配置字段、生命周期、信任边界、结果投影或错误映射，必须停止编码、更新 ADR 和本方案并重新评审。
