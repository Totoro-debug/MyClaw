# MyClaw

## 项目简介

MyClaw 是面向单用户、本地优先的个人 Agent 运行时。通过全屏终端对话调用模型与工具，支持会话恢复、三层记忆、Skill、MCP 和定时任务，运行状态以文件形式保存在本地。

## 项目安装

需要 Python 3.12+ 和 Git；使用命令执行工具还需安装 Bash。

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

MCP Server 可在 `[mcp.servers.<name>.tool_keywords]` 下按远端 Tool 原名配置英文关键词；缺失项会在启动时通过现有 `chat` Model Route 准备并尽力保存。关键词生成或保存失败不会阻止 Agent 启动，失败项仅在当前进程内使用远端 Tool 原名。

## 项目启动

在已激活虚拟环境的交互式终端中，进入希望 Agent 操作的目录后启动（将 `<workspace>` 替换为实际路径）：

```bash
cd <workspace>
myclaw
```

新开终端后需重新激活安装目录中的虚拟环境，或使用其中 `myclaw` 可执行文件的绝对路径。启动需要交互式输入、输出，不能通过管道运行。

启动目录即 Workspace，不会自动切换到 Git 根目录。运行状态保存在该目录的 `.myclaw/` 中；定时任务仅在 MyClaw 进程运行期间执行。

`Enter` 提交输入，`Ctrl+J` 换行；`Ctrl+C` 取消当前回复，输入 `exit` 或 `quit` 退出。

以下管理命令需单独输入，不附带参数：

| 命令 | 用途 |
| --- | --- |
| `/resume` | 从当前 Workspace 的会话列表选择并恢复历史会话 |
| `/status` | 查看运行状态与上下文用量 |
| `/config` | 查看脱敏后的配置 |
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
| Agent Runner | 执行有迭代上限的模型与工具循环，供前台、定时任务和 Dream 复用 |
| Model Router | 按用途选择模型，适配 OpenAI 兼容协议与 Anthropic，处理重试和回退 |
| Tool Gateway / MCP | 统一内置与 MCP 工具的调用入口；MCP 连接由 CLI 管理，每个 Agent Loop 使用固定工具快照，单次 Agent Run 通过 `tool_search` 按需暴露延迟 Tool schema |
| Memory / Dream | 管理短期记忆、会话摘要和长期记忆；Dream 使用独立 Runner 与受限工具整理长期记忆 |
| Schedule Service | 持久化并调度任务；用户任务调用当前 Agent Loop，记忆整理任务直接调用 Dream |
| Skill Loader / Context Builder | 加载 Skill 快照，按需提供指令，统一构建 Agent Loop 的模型请求上下文 |

全局配置与 Skill 位于 `~/.myclaw/`；会话、记忆、定时任务、工具产物和日志归各 Workspace 的 `.myclaw/` 所有。

内置工具通过权限检查决定是否请求一次性确认；Exec 以当前用户权限执行，不提供操作系统沙箱。MCP 支持 stdio 和 Streamable HTTP，已启用的 Server 视为可信能力，其工具调用不再逐次确认。

架构决策见[现行 ADR](docs/adr/)，领域术语见[CONTEXT.md](CONTEXT.md)。
