# MyClaw

## 项目简介

MyClaw 是面向单用户、本地优先的个人 Agent 运行时。通过全屏终端对话调用模型与工具，支持会话恢复、三层记忆、Skill、MCP 和定时任务，运行状态以文件形式保存在本地。

## 项目安装

需要 Python 3.12+ 和 Git。POSIX 主机使用 Bash；Windows 主机的 `auto` 选择可用的 PowerShell 7，否则使用 Windows PowerShell 5.1。

目前已验证 Windows x64；macOS 尚未完成原生验证，其他 POSIX 平台暂无正式支持承诺。

```bash
git clone https://github.com/Totoro-debug/myclaw.git
cd myclaw
```

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install .
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## 项目最小配置

执行 `myclaw config` 生成默认配置，再将 `~/.myclaw/config.toml` 的内容替换为以下配置。已有配置不会被该命令覆盖。`~` 表示当前用户主目录，Windows 下通常为 `C:\Users\<用户名>`。

选择支持工具调用的模型，替换服务地址、API Key 和两处模型 ID。按模型实际限制设置 `context_window` 与 `max_output`，单位均为 token，后者必须小于前者；`timeout` 单位为秒。

```toml
[models.providers.my-provider]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "replace-with-your-api-key"
models = ["your-model-id"]

[models.routes.default]
provider_id = "my-provider"
model = "your-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
timeout = 120
```

`protocol` 支持 `openai-compatible` 和 `anthropic`。只配置 `default` 路由即可供对话、记忆和定时任务使用，其余配置采用默认值。

API Key 直接保存在配置文件中，当前不支持环境变量引用；`myclaw config` 显示时会脱敏。更多可选配置及 MCP 示例见[配置模板](myclaw/templates/default-config.md)。

MCP Server 可在 `[mcp.servers.<name>.tool_keywords]` 下按远端 Tool 原名配置英文关键词；缺失或为空的关键词会在启动时通过现有 `chat` Model Route 生成并尽力保存。生成失败时，当前进程使用对应的远端 Tool 原名；生成成功但保存失败时，当前进程继续使用已生成的内存关键词。两类失败都不会阻止 Agent 启动。

`[runtime].permission_level` 接受 `read-only`、`workspace-write` 和 `full-access`，默认值为 `workspace-write`；`[runtime].exec_shell` 接受 `auto`、`powershell` 和 `pwsh`，默认值为 `auto`。Windows 下 `auto` 优先选择版本至少为 7 的 `pwsh`，否则选择 Windows PowerShell 5.1；显式选择不会交叉回退，POSIX 始终使用 Bash 并忽略该 Windows selector。Exec Host 的检查和执行都禁用 Profile/rc（PowerShell 使用 `-NoLogo -NoProfile -NonInteractive`，Bash 不使用 login/profile/rc），并共享同一个可执行文件、工作目录、最小环境和超时/取消行为。选定 Shell 缺失不会阻止启动，只显示一次安全诊断；Exec 仍在工具目录中，但调用时返回稳定的能力错误。Shell 存在但检查器失败时按不确定结果继续沿用确认语义。

前台 Agent Run 会在标题生成、Skill 解析和 Task Framing 之前捕获当前权限级别与已解析的 Exec Shell，并在本次运行内保持不变。Runtime Context 会显示该快照以及 Tool 调用可能需要确认；确认只绑定到一次规范化后的 Tool 调用，不会缓存到共享 Tool。File Tool 的权限矩阵和主机路径规范化见 [ADR-0026](docs/adr/0026-tool-permission-levels-and-foreground-snapshots.md)。模型发起的 `read_file` 不继承 Skill Root 的内部加载豁免；Dream 私有 Tool 和 Runtime 持久化写入不属于该模型 File 策略。

`/permission` 只修改当前 Runtime Lifetime 后续前台 Agent Run 的权限级别，不写入 User Configuration、Conversation Session 或 Schedule。成功的 `/resume` replacement 会保留当前选择；失败的 replacement 不会改变它；新进程从配置值开始。选择 `full-access` 前必须通过默认聚焦 Cancel 的警告；当前该级别只取消前台 File Tool 的普通权限确认，不是操作系统沙箱，也不绕过参数校验、能力错误、业务拒绝或 Tool 执行错误。Exec、Web、MCP 和 Schedule 保持现有行为。进程启动时最多显示一次 Full-Access 安全提示；`/config` 显示 configured level，`/status` 同时显示 configured/current foreground level。

安全默认值字段包括 Runtime 的 Tool 结果大小、迭代上限、Always-load Skill 开关、`compact_ratio`、权限级别和 Exec Shell，Memory 的 batch size 与 schedule，以及每个 Model Route 的 `reasoning_effort`。缺失字段静默使用默认值；显式非法值使用有效默认值并产生且只产生一条不包含原始值的安全诊断。`compact_ratio` 的默认值为 `0.9`，只接受有限且非布尔的数值 `0.5` 至 `0.95`（含边界）。原始配置不会被自动改写；`myclaw config` 与 `/config` 显示相同的有效值和诊断，以及脱敏后的原始 TOML。已移除的 `compaction_message_threshold` 等未知字段会被忽略。

## 项目启动

在已激活虚拟环境的交互式终端中，进入希望 Agent 操作的目录后启动（将 `<workspace>` 替换为实际路径）：

```bash
cd <workspace>
myclaw
```

新开终端后需重新激活安装目录中的虚拟环境，或使用其中 `myclaw` 可执行文件的绝对路径。启动需要交互式输入、输出，不能通过管道运行。

启动目录即 Workspace，不会自动切换到 Git 根目录。运行状态保存在该目录的 `.myclaw/` 中；定时任务仅在 MyClaw 进程运行期间执行。

Schedule Tool 的 `add` 可选接收用户可见 `title`；它会按 Conversation Session 的空白和成对引号规则规范化，并确定性截断到 60 个 Unicode code points。只有省略该字段时才从 `message` 第一条非空行派生；显式 `null` 或规范化后为空的值会被拒绝。`add` 和 `list` 的每个公开 Job 都返回 `job_id`、`title`、`message` 和 `schedule`。新建用户 Schedule Conversation Session 的初始 title 与 Job title 相同，Job 始终是 title 的权威来源。

Workspace 的 `.myclaw/schedule.json` 使用严格 canonical Schedule Job schema，当前记录必须包含 `title`。升级时仅接受整份文档都使用完整旧字段集的状态，并在读取时派生 title；下一次成功的 Schedule Store 写入会把全部记录改写为新 schema。旧新记录混用、单条记录混合字段、未知字段和失败写入都不会被静默修复。

`Enter` 提交输入，`Ctrl+J` 换行；`Ctrl+C` 取消当前回复，输入 `exit` 或 `quit` 退出。

以下管理命令需单独输入，不附带参数：

| 命令 | 用途 |
| --- | --- |
| `/resume` | 从当前 Workspace 的会话列表选择并恢复历史会话 |
| `/status` | 查看运行状态与上下文用量 |
| `/config` | 查看脱敏后的配置 |
| `/permission` | 选择前台 Tool 权限级别 |
| `/effort` | 选择对话模型的推理强度 |
| `/memory` | 查看长期记忆 |
| `/dream` | 将待处理的会话摘要整理为长期记忆 |
| `/reload_skill` | 重新加载 `~/.myclaw/skills/` 中的 Skill |

## 项目架构

CLI 负责组装运行时和管理组件生命周期。前台输入经终端与 Message Bus 进入 Agent Loop，由 Agent Runner 循环调用模型与工具，结果经 Message Bus 返回终端。

| 组件 | 职责 |
| --- | --- |
| Terminal / Message Bus | 基于 Textual 与 Rich 展示对话，通过输入、输出队列连接当前 Agent Loop |
| Agent Loop | 绑定一个会话，串行处理前台输入，管理上下文、任务目标和会话持久化 |
| Agent Runner | 执行有迭代上限的模型与工具循环，供前台和用户定时任务复用 |
| Model Router | 按用途选择模型，适配 OpenAI 兼容协议与 Anthropic，处理重试和回退 |
| Tool Gateway / MCP | 统一内置与 MCP 工具的调用入口；MCP 连接由 CLI 管理，每个 Agent Loop 使用固定工具快照，单次 Agent Run 通过 `tool_search` 按需暴露延迟 Tool schema |
| Memory / Dream | 管理短期记忆、会话摘要和长期记忆；Dream 通过一次受限的 `memory` 请求整理长期记忆 |
| Schedule Service | 持久化并调度任务；用户任务调用当前 Agent Loop，记忆整理任务直接调用 Dream |
| Skill Loader / Context Builder | 加载 Skill 快照，按需提供指令，统一构建 Agent Loop 的模型请求上下文 |

全局配置与 Skill 位于 `~/.myclaw/`；会话、记忆、定时任务、工具产物和日志归各 Workspace 的 `.myclaw/` 所有。

内置工具通过权限检查决定是否请求一次性确认；Exec 通过 Host 以当前用户权限执行，不提供操作系统沙箱。MCP 支持 stdio 和 Streamable HTTP，已启用的 Server 视为可信能力，其工具调用不再逐次确认。

架构决策见[现行 ADR](docs/adr/)，领域术语见[CONTEXT.md](CONTEXT.md)。

## 运行时合同

### `/status`

前台和用户定时任务共用同一套上下文预算；Dream 不进入这套 Agent Run 预算。`/status` 只基于当前已提交的 Conversation Session 和当前 User Configuration，投影下一次独立 Foreground Agent Run 的初始模型请求基线，不读取未提交的 run-local staged state。Route 字段来自当前配置解析出的初始 `chat` Model Route；未配置 `chat` 时可使用配置层面的静态 `default`，上一逻辑调用的动态 fallback 不会成为下一次调用的初始 Route。上下文字段定义如下：

- `context_window`：当前配置解析出的初始 `chat` Model Route 的总上下文窗口。
- `max_output`：为模型输出预留的 token 上限。
- `available_context`：可用于输入的预算，等于 `context_window - max_output`。
- `compact_ratio`：实际生效的配置比例，默认 `0.9`。
- `compact_context_window`：软压缩阈值，等于 `ceil(available_context * compact_ratio)`。
- `projected_next_request_tokens`：上述下一次独立 Foreground Agent Run 初始模型请求的投影 token 数；仅当最新 main-Agent assistant 的 Provider 用量 provenance 合法且兼容时使用报告增量，缺失、非法或不兼容时使用完整本地估算，不跨过该 assistant 复用更早锚点。
- `projection_source`：投影来源，`reported_delta` 表示报告用量增量，`estimated` 表示本地估算。
- `input_budget_used_percent`：`projected_next_request_tokens / available_context * 100`，分母是可用输入预算。

`cumulative_usage` 是 Session 生命周期内的累计模型调用、输入、输出和总 token 用量，仅用于观测，不代表当前请求上下文占用；Schedule 状态不混入前台 Session 的上下文字段。

### Agent Run

Foreground 和 User Schedule Agent Run 在启动及每次 ReAct 请求前都执行同一套预算检查。历史压缩优先于 Provider-only Tool Result Micro-compression；达到软阈值时按 10%/50% 保留规则选择历史，达到硬输入上限则以 `model_context_overflow` 结束，且每个实际 Provider 尝试都重新检查自己的 Route。超过十个符合条件的 Tool Call 后，才会对请求投影中的旧 Tool Result 做微压缩；持久化消息、摘要和 Tool Artifact 不写入占位符。终止时由 Agent Loop 一次性提交本次 Session 增量及 Agent Run 状态，摘要流保持在 Session 事务之外。

### Dream

Dream 不是 Agent Run，不创建 Agent Runner 或 Agent Run Context Controller，也不执行上下文压缩。没有待处理 Conversation Summary 时，Dream 发起 `0` 次模型请求；领取到待处理 Summary 后，恰好发起一个逻辑上的 `memory` 模型请求，Router 自动重试只增加 Provider attempt，不增加逻辑请求数。该请求只暴露 `edit_file`，返回的编辑按顺序执行；前面的编辑成功后，即使后续编辑失败，已完成的修改仍保留。

### 权威合同与公共证据

Ticket #234 的统一要求由 [ADR-0023](docs/adr/0023-manage-agent-run-context-by-projected-token-budget.md)、[ADR-0024](docs/adr/0024-use-one-shot-dream-model-request.md) 和 [ADR-0025](docs/adr/0025-ignore-unknown-user-configuration-fields.md) 共同定义；[ADR-0022](docs/adr/0022-action-summary-and-tool-result-micro-compression.md) 已明确标记为被 ADR-0023 取代。公共行为证据集中在[控制器测试](tests/memory/test_agent_run_context_controller.py)、[前台循环测试](tests/agent/test_loop.py)、[定时任务测试](tests/agent/test_schedule_loop.py)、[Dream 测试](tests/memory/test_dream.py)、[Session 测试](tests/sessions/test_session.py)和[发布合同测试](tests/test_release_contract.py)。
