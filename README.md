# MyClaw

MyClaw 是一个面向单用户的、本地优先的 Personal Agent 运行时，支持 Python 3.12及以上版本。它以当前目录作为 Agent 的 Workspace，通过全屏终端对话连接模型、工具、会话、记忆、Skill 与定时任务，并将运行状态保存为本地可检查的文件。

MyClaw 不是多租户 Agent 平台，也不是后台常驻服务。每次运行对应一个终端中的 Runtime Lifetime；用户可以在其中连续对话、调用工具、恢复历史 Conversation Session，并让计划任务在该进程存活期间执行。

## 核心能力

- **全屏终端对话**：基于 Textual 和 Rich，支持流式回复、推理与工具活动展示。
- **本地优先持久化**：配置、会话、记忆、日志、定时任务和工具产物均保存为本地文件。
- **多模型路由**：支持 `openai-compatible` 和 `anthropic` Provider，并可为聊天、记忆和定时任务分别配置模型。
- **固定工具目录**：内置文件操作、目录检索、命令执行、Web 搜索、Web 获取和定时任务等十项 Tool，通过统一的 Tool Gateway 执行校验与授权。
- **三层记忆系统**：由 Short-term Memory、Conversation Summary 和 Long-term Memory 组成。
- **Skill 发现与渐进使用**：从 Agent Home 捕获可原子重载的冻结 Skill Snapshot，支持手动斜杠调用、模型自主读取和可选的 System Prompt 投影。
- **任务连续性**：在普通前台输入之前执行 Task Framing，用隐藏 Blackboard 维护当前目标和完成边界。
- **Workspace 隔离**：每个启动目录拥有独立的 Session、Memory、Schedule、Artifact 和 Session Log。

## 环境要求与平台支持

- Python 3.12 或更高版本。
- 默认命令需要交互式 `stdin`、`stdout` 和 `stderr` TTY。
- Exec Tool 启动一个直接的 Bash 子进程，能否使用取决于宿主机是否具备可用的 Bash。
- 项目没有运行前的平台拦截（no platform gate）。

发行包是同时包含 Windows 与 POSIX 宿主适配器的 `py3-none-any` Wheel。Windows x64是目前经过验证的平台（currently validated）；macOS Intel 与 Apple Silicon 是预期兼容目标，但尚未完成原生验证（unverified）。Linux 和其他 POSIX 宿主会尝试使用POSIX 适配器，但当前版本不作正式支持承诺。

## Agent 安装与使用

### 从源码安装

在仓库根目录创建虚拟环境并安装 MyClaw。

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install .
```

macOS 或其他 POSIX Shell：

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install .
```

### 从 Wheel 安装

先构建或取得发行 Wheel，再将路径传给 `pip`：

```powershell
.\.venv\Scripts\python.exe -m pip install .\dist\myclaw-0.1.0-py3-none-any.whl
```

安装完成后，激活虚拟环境即可使用 `myclaw`。如果不激活，也可以直接运行虚拟环境中的可执行文件，例如 `.\.venv\Scripts\myclaw.exe`。

### 首次启动

先进入希望 Agent 操作的目录，再启动 MyClaw：

```powershell
Set-Location D:\path\to\your-workspace
myclaw
```

首次启动会创建当前操作系统账户的 Agent Home 和默认配置：

```text
~/.myclaw/config.toml
```

此时程序输出 `config_missing` 并以状态码 2 退出。这是预期行为：编辑新生成的配置，填入可用的 Provider 和 Model Route 后，再次执行 `myclaw`。

> `~` 表示当前用户主目录。在 Windows 上通常对应
> `C:\Users\<用户名>\.myclaw\`。

### 配置模型

下面是可工作的最小结构。Provider 地址、API Key、模型 ID 和模型限制必须替换为实际值：

```toml
[models.providers.openai-local]
protocol = "openai-compatible"
base_url = "https://provider.example/v1"
api_key = "replace-with-a-dedicated-key"
models = ["replace-with-a-model-id"]

[models.routes.default]
provider_id = "openai-local"
model = "replace-with-a-model-id"
context_window = 200000
max_output = 8192
temperature = 0.2
reasoning_effort = "medium"
timeout = 120
```

MyClaw 支持两种 Provider 协议：

- `openai-compatible`
- `anthropic`

Model Route 按用途选择模型：

| Route | 用途 |
| --- | --- |
| `default` | 其他 Route 不存在或允许回退时使用 |
| `chat` | 前台对话、Session 标题和 Task Framing |
| `memory` | Conversation Summary 与 Dream |
| `schedule` | Schedule Job |

只配置 `default` 即可启动。删除某个用途专用的 Route 后，该用途会回退到 `default`。每个 Route 的 `model` 必须同时出现在对应 Provider 的 `models` 数组中。

`api_key` 以明文保存在 `config.toml`。配置查看和面向用户的错误会隐藏 Key，但当前版本不支持环境变量引用或操作系统 Keychain。建议使用权限最小化的专用 Key，并保护 AgentHome 的文件权限。

查看脱敏后的当前配置：

```powershell
myclaw config
```

即使 TOML 无效，此命令也会尽量显示脱敏内容和错误位置。

### 开始对话

配置有效后，在 Workspace 目录中执行：

```powershell
myclaw
```

当前目录的规范化绝对路径就是 Workspace 边界，也是 Session 与其他非全局状态的归属。MyClaw 不会自动查找 Git 根目录或父目录。

普通输入会进入 Agent Loop。精确匹配的管理命令不经过模型：

| 命令 | 功能 |
| --- | --- |
| `/config` | 查看脱敏后的 User Configuration |
| `/status` | 查看 Runtime、模型、Token 和 Session 状态 |
| `/effort` | 选择当前 Runtime Lifetime 的 chat Reasoning Effort |
| `/resume` | 列出并恢复当前 Workspace 的 Conversation Session |
| `/memory` | 查看当前 Long-term Memory |
| `/dream` | 立即处理尚未消费的 Conversation Summary |
| `/reload_skill` | 原子重载当前 Agent Loop 的 Skill |

精确提交 `/effort` 会用五档横向 selector 替换输入框；确认后，当前 Runtime Lifetime 的
`chat` 与 `default` 请求立即使用所选值，显式 `memory` 与 `schedule` route 保持独立。
切换 Conversation Session 不会重置该值。

`/reload_skill` 不进入 Message Bus 或 Conversation Session。成功后，后续 Agent Run、手动 Skill 调用和终端补全共同使用新状态；已经开始的 Agent Run 继续使用其已构造的消息。失败时显示稳定错误，并完整保留先前状态。

其他常用操作：

- 输入 `exit` 或 `quit`：正常关闭 MyClaw。
- 按 `Ctrl+C`：取消当前前台 Agent Run，终端对话仍保持可用。
- Tool 请求需要一次性确认时：在终端确认或拒绝该次具体调用。

## Skill 安装与使用

Skill 是指导 Agent 使用现有能力的指令包。它不会注册新 Tool，也不会扩大文件、命令或
网络权限。

### 安装 Skill

在 Agent Home 的 `skills` 下创建一个直接子目录，并在其中放置 UTF-8 编码的
`SKILL.md`：

```text
~/.myclaw/
  skills/
    planner/
      SKILL.md
```

最小 Skill 示例：

```markdown
---
name: planner
description: 将复杂需求整理为清晰、可执行的计划
---

# 工作方式

1. 明确目标和完成边界。
2. 找出约束、依赖与风险。
3. 输出可验证的执行步骤。
```

元数据规则：

- `name` 和 `description` 为必填字符串。
- `name` 长度为 1～64 个字符，首字符只能是小写字母、下划线或连字符，后续还可使用数字。
- `description` 去除首尾空白后长度为 1～1024 个字符。
- Skill 名不能与 `/config`、`/status`、`/resume`、`/memory`、`/dream`、`/reload_skill` 等管理命令冲突。
- MyClaw 只扫描 `~/.myclaw/skills` 的直接子目录；初始启动、任意 `/resume` 或成功的 `/reload_skill` 会重新扫描。磁盘修改在下一次成功加载前不会改变当前冻结状态。

### 手动调用

在终端中使用精确的 Skill 名称：

```text
/planner 为下周的发布工作制定计划
```

MyClaw 使用当前 Runtime Generation 创建时已经完整读取、校验并冻结的 `SKILL.md`，把 Skill 文档和 `/planner` 后面的请求一起提供给当前前台 Agent Run；手动调用不会再次访问磁盘。Conversation Session 只持久化用户输入的原始斜杠命令。

未知的斜杠输入、大小写不匹配的名称或不完整名称不会触发 Skill，而是作为普通输入处理。

### 模型自主选择

前台 System Prompt 会获得有效 Skill 的名称、描述和绝对路径。模型可以根据描述选择Skill，再通过现有 `read_file` Tool 渐进读取 `SKILL.md`。Skill 根目录内的规范路径允许免确认读取；通过链接逃逸到目录外的路径仍遵循 Workspace 外部路径的确认规则。

Skill 的绝对路径和被读取的内容可能发送给已配置的 Model Provider，请勿在 `SKILL.md` 中保存秘密。

### 启动时加载

需要每次前台模型调用都包含某个 Skill 时，在 Skill frontmatter 中设置布尔值：

```yaml
always: true
```

同时在 `~/.myclaw/config.toml` 中启用：

```toml
[runtime]
enable_skill_always_load = true
```

每次成功加载的完整内容保持冻结，直到成功执行 `/reload_skill` 或创建新的 Agent Loop。`/reload_skill` 会先扫描、校验并检查输入预算，再一次性发布新状态；失败不会替换当前状态。此模式没有固定的 Skill 文件大小上限，但内容仍受聊天模型输入预算约束；初始启动或 `/resume` 的同步 preflight 超出预算时会以 `skill_context_too_large` 终止 Terminal Conversation。

## 项目架构

MyClaw 使用宿主无关的组合根，将终端呈现、Agent 编排、模型调用、Tool 授权和本地持久化分开：

```text
myclaw/
├── terminal/       全屏 Terminal Conversation、键盘适配与 CLI 入口
├── management/     管理命令分发及只读/受控管理视图
├── agent/          Message Bus、Agent Loop、Agent Runner 与上下文构建
├── provider/       Model Router、Provider 工厂及协议适配器
├── tools/          Tool Gateway、权限策略和固定 Tool 实现
├── session/        Conversation Session 及模型消息投影
├── memory/         Conversation Summary、Long-term Memory、Memory Manager 与 Dream
├── schedule/       Schedule Job、Schedule Service 与 Workspace 存储
├── skills/         Skill Catalog、校验和渐进加载
├── config/         Agent Home 与 User Configuration
├── logging/        进程诊断和 Workspace Session Log
├── templates/      System Prompt 与 Runtime Prompt 模板
└── utils/          宿主文件系统、时间、校验和异步任务支持
```

### 核心组件

| 组件 | 职责 |
| --- | --- |
| Terminal Conversation | 接收用户输入，渲染回复、推理、Tool 活动和确认对话框 |
| CLI composition root | 拥有 Runtime Lifetime 级组件、当前 Agent Loop 引用、Session 替换和关闭顺序 |
| Message Bus | 在整个 Runtime Lifetime 内复用，在 Terminal Conversation 与当前 Agent Loop 间传递临时 Inbound/Outbound Message |
| Agent Loop | 串行处理前台输入，管理 Session、Task Framing、Tool 和结果持久化 |
| Agent Runner | 执行一次有迭代上限的 ReAct 模型与 Tool 循环 |
| Model Router | 按逻辑 Route 解析 Provider 和模型，并处理限定重试与回退 |
| Tool Gateway | Tool 调用的唯一公共入口，负责解析、校验、授权、执行和结果归一化 |
| Memory Manager 与 Dream | 管理 Summary/Cursor/Long-term Memory 状态，并通过独立 Dream Runner 处理长期记忆 |
| Schedule Service | 保存、触发和取消 Schedule Job；User Job 调用当前 Agent Loop，Dream System Job 直接调用 Dream |
| Skill Snapshot | Skill Loader 在每次成功加载时完整读取并冻结有效 Skill 文档，对模型按用途投影元数据或正文 |

### 固定 Tool Catalog

Tool Catalog 不能通过配置增删或替换，固定包含：

1. Read File
2. Write File
3. Edit File
4. List Dir
5. Glob
6. Grep
7. Exec
8. Web Search
9. Web Fetch
10. Schedule

过大的成功 Tool Result 会被外部化到当前 Workspace 的 `.myclaw/artifacts/<session_id>/`，模型收到指向该 Artifact 的归一化结果。

## 核心数据流

### 启动

```text
CLI
  → 读取 ~/.myclaw/config.toml
  → 以当前目录建立 Workspace
  → 初始化 <workspace>/.myclaw/
  → 组合 Runtime Lifetime 级 Message Bus、Model Router、Memory Manager、Dream 与 Schedule Service
  → 创建并 preflight 初始 Agent Loop，同时捕获初始 Skill Snapshot
  → 注册或校正 Dream System Job，并创建或校正 schedule.json
  → 启动 Terminal Conversation
```

### 前台 Agent Run

```text
用户输入
  → Management Command 精确匹配，或进入 Message Bus
  → Task Framing 更新当前 Blackboard
  → 拼装 System Prompt、Runtime Context、Memory、Skill 与短期历史
  → Agent Loop 调用 Agent Runner
  → Model Router 调用 chat Route
  → Tool Call 经 Tool Gateway 校验、授权并执行
  → Tool Result 返回模型，直至生成最终回复或达到迭代上限
  → 更新 Conversation Session、Token 用量和持久化请求
  → Outbound Message 交给 Terminal Conversation 渲染
```

每个普通前台输入会额外触发一次无 Tool 的 Task Framing 调用。Blackboard 只包含当前
`goal` 和 `completion_boundary`，用于帮助模型理解任务连续性；它不能授权执行、
绕过 Tool Confirmation 或控制工作流。

### Memory

```text
较早的 Session 消息
  → Conversation Summary
  → <workspace>/.myclaw/memory/summary.jsonl
  → Dream System Job 或 /dream 触发 Dream
  → memory Route 判断并更新 memory.md
  → 推进 Summary Cursor
```

Short-term Memory 是 Session 中尚未被摘要覆盖的后缀；Conversation Summary 是按序
保存的摘要流；Long-term Memory 是跨 Conversation Session 生效的稳定信息。

### Schedule

Schedule Service 从 Workspace 的 `schedule.json` 读取任务，在 Runtime Lifetime 内
等待触发时间。User Schedule Job 使用独立 Schedule Session 和 `schedule` Route，并通过
当前 Agent Loop 共享该 Runtime Generation 的 Tool Gateway 与 Agent Runner；Dream System
Job 则直接调用 `Dream.run()`，不创建 Schedule Session 或进入 Agent Loop。两条路径都不会
向前台 Outbound Message 流发布执行过程，也没有交互式确认通道。

## 本地数据与目录

### Agent Home

Agent Home 固定为当前账户的 `~/.myclaw/`，不能通过配置切换：

```text
~/.myclaw/
├── config.toml
└── skills/
    └── <skill-directory>/
        └── SKILL.md
```

它只保存全局 User Configuration 和用户编写的 Skill。

### Workspace State

每个启动目录都拥有独立的 `.myclaw`：

```text
<workspace>/.myclaw/
├── .gitignore
├── schedule.json
├── memory/
│   ├── memory.md
│   ├── summary.jsonl
│   └── .cursor
├── sessions/
│   └── <session_id>.jsonl
├── schedule-sessions/
│   └── schedule_<job_id>.jsonl
├── artifacts/
│   └── <session_id>/
│       └── <tool_call_id_or_uuid4>.txt
└── logs/
    └── <session_id>.log
```

启动时创建 Workspace State 根目录、内部 `.gitignore`、`memory/`、`sessions/`、缺失的
`memory.md`，并在注册 Dream System Job 时创建或校正 `schedule.json`；其余文件和目录由
对应功能按需创建。

建议将 Workspace 与其中的 `.myclaw` 一起备份。不要在 MyClaw 运行期间手动编辑Session、Summary、Summary Cursor 或 Schedule 状态文件。

## 权限与安全边界

- Workspace 内的文件操作仍受操作系统账户权限限制。
- Workspace 外部文件路径，以及未通过具体安全检查的 Exec/Web 目标，会请求绑定到该次调用的一次性 Tool Confirmation。
- Schedule Agent Run 没有交互式确认能力，因此拒绝所有需要确认的操作。
- Exec 不是操作系统沙箱。命令继承当前用户权限，可能影响 Workspace 之外的系统资源。
- Web Tool 会执行 URL、DNS、重定向及目标地址检查，但这不等同于完整网络隔离。
- Tool、Skill 和 Blackboard 都不能扩大 Permission Policy 允许的权限。
- Artifact 没有自动清理策略，Long-term Memory 也没有自动大小上限。

## 运行限制

- 同一 Conversation Session 不支持并发写入（same-session concurrency is unsupported）。
- 多个 MyClaw 进程不会协调 Session、Session Log 或后台 Schedule。
- Session Log 使用无界队列（unbounded queue）；正常上下文退出会无限等待队列排空（infinite drain）。
- Session Log 不对每条记录执行 `fsync`（no per-record fsync），异常退出、断电或强制终止可能丢失最近记录。
- Session Log 不主动脱敏（no active redaction），也不转义控制字符（no control escaping）。传给日志调用的凭据、换行或异常文本可能原样落盘。
- 日志保留按 Session 独立计算（per-session retention），Workspace 的日志总量没有全局上限。
- 旧版 Agent Home Runtime Log 文件保持原样（legacy Agent Home Runtime Log files remain untouched）；升级不会读取、移动、删除、截断或更新它们。
- 普通后台 Session 保存失败没有用户确认或失败日志；崩溃后 Conversation Summary 与`last_consolidated` 可能暂时不一致。
- 当前版本没有 daemon、HTTP/IPC 服务、MCP、subagent runtime、profiles、跨进程状态协调、Keychain 集成或环境变量 API Key。

## License

MyClaw 使用 Apache License 2.0，完整条款见 [LICENSE](LICENSE)。
