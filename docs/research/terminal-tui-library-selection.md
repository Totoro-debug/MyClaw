# MyClaw 全屏终端 UI 库选型：Enter 与 Shift+Enter

## 文档状态

- 状态：已完成
- 日期：2026-08-11
- 范围：Python 全屏 TUI；Enter 发送、Shift+Enter 换行；Windows Terminal/ConPTY 与 macOS 兼容性
- 来源：仅采用官方文档、官方源码与终端协议

## 问题

MyClaw 需要一个占满终端的两区聊天界面：上方对话区，下方输入区。输入必须支持 Enter 发送、Shift+Enter 换行，同时还要覆盖 Markdown 流式更新、滚动和鼠标、确认弹窗、`asyncio` 集成与自动化测试。

核心限制不在按键绑定 API，而在终端输入协议：应用只有在终端发送了可区分的 Shift+Enter 序列后，库才能绑定它。传统终端输入把 Enter 编码为 `CR`（`0x0d`），没有 Shift 状态；此时 Shift+Enter 与 Enter 对应用完全相同，任何 Python TUI 库都无法可靠区分。Kitty Keyboard Protocol 对传统编码和增强标志的定义见[官方协议](https://sw.kovidgoyal.net/kitty/keyboard-protocol/)。

## 结论

选择 **Textual 8.2.8 或更高版本**作为 MyClaw 的全屏 TUI 框架，但精确的 Shift+Enter 契约必须附带 **Kitty Keyboard Protocol 的 bit 8（report all keys as escape codes）与 capability gate**：

1. 启动时探测终端是否支持 Kitty Keyboard Protocol，并确认已启用 bit 8；为保留普通文本输入，应同时评估/启用 bit 16（associated text）。
2. 只有探测成功时才声明 Shift+Enter 可用，并将 `CSI 13;2u` 绑定为换行；Enter 继续发送。
3. 探测失败时必须提供明确的备用换行键（建议 Ctrl+J）或要求用户配置终端映射。不能假定旧终端能区分 Shift+Enter，也不能把收到的普通 `CR` 猜测为 Shift+Enter。

这不是 Textual 单独能消除的限制。Textual 当前 Windows 驱动主动发送的是 `CSI > 1 u`，即只启用 disambiguate escape codes；[驱动源码](https://github.com/Textualize/textual/blob/06dbeef4bb70fb718236aa418ed658ef4667a126/src/textual/drivers/windows_driver.py#L95-L100)可见该序列。Kitty 协议明确规定 bit 1 仍让 Enter、Tab 和 Backspace 使用传统字节，必须使用 bit 8 才能让 Enter 携带 Shift 修饰。因此，采用 Textual 后仍需一个小型协议适配层、上游支持，或受控的终端配置。

`Alt+Enter` 也不能作为无需探测的通用后备键。Windows Terminal 默认将它绑定为 `Terminal.ToggleFullscreen`，宿主绑定会优先于下层应用；用户必须在 Terminal 设置中解绑该动作或改用 `sendInput`，MyClaw 才能收到按键。参见[Windows Terminal Actions](https://learn.microsoft.com/en-us/windows/terminal/customize-settings/actions)。因此应用可以在收到独立的 `alt+enter` 事件时将其作为换行别名，但不能用它证明旧终端具备可靠的多行输入能力。

## 候选比较

| 维度 | Textual | prompt-toolkit 3.0.52 | Urwid |
| --- | --- | --- | --- |
| Shift+Enter | 能解析 `CSI 13;2u`，但默认协商不足；需 bit 8 + capability gate | 官方 ANSI 映射把 Shift+Enter 序列折叠为普通 `ControlM` | 官方文档明确按键可用性依赖终端；无 Kitty 协商能力 |
| 全屏两区 | 声明式布局、`VerticalScroll`、`TextArea`，适合直接实现 | `Application`/layout 可实现，但需要自行组合与渲染 | `Frame`/`Pile`/`ListBox` 可实现 |
| Markdown 流式更新 | 内置 `Markdown` 和流式更新接口 | 无内置 Markdown；需自行桥接 Rich/格式化缓存 | 无内置 Markdown |
| 滚动/鼠标/modal | 内置滚动容器、鼠标事件、`ModalScreen` 与按钮 | 可用，但弹窗、焦点和鼠标行为需更多手工编排 | Overlay、鼠标与 ListBox 可用 |
| asyncio | `App.run_async()` | `Application.run_async()` | 提供 `AsyncioEventLoop`，Windows 有额外事件循环约束 |
| 测试 | `run_test()`、Pilot 按键/点击/尺寸测试 | pipe input + dummy output | 可测，但缺少同等级的应用级 Pilot API |
| 结论 | **推荐**，UI 能力最完整；键盘协议仍需补齐 | 保留现状依赖最少，但 UI 工作量高且按键解析直接违背需求 | 没有解决关键按键问题，整体收益低 |

## Textual

Textual 与本需求最匹配：官方聊天界面示例使用 `VerticalScroll` 加输入框构成上下布局，并通过锚定保持最新消息可见；布局会随终端尺寸变化。参见[官方界面剖析](https://textual.textualize.io/blog/2024/09/15/anatomy-of-a-textual-user-interface/)和[教程中的滚动行为](https://textual.textualize.io/tutorial/)。

`Markdown` 提供 `get_stream()`，用于合并频繁的增量更新，适合模型 token 流式输出；官方也提醒直接高频 append 会形成更新积压。参见[Markdown widget 文档](https://textual.textualize.io/widgets/markdown/)。`ModalScreen` 会阻止下层界面交互，适合工具确认按钮，参见[Screen 文档](https://textual.textualize.io/guide/screens/)。`run_async()` 与无头 `run_test()`/Pilot 分别覆盖运行时和测试，参见[App API](https://textual.textualize.io/api/app/)与[测试指南](https://textual.textualize.io/guide/testing/)。

键盘方面，Textual 官方输入文档允许使用 `shift+` 前缀绑定非打印键，同时警告组合键是否可见取决于终端，参见[Input 指南](https://textual.textualize.io/guide/input/)。8.2.8 修复了 Kitty 扩展按键解析，参见[官方发布说明](https://github.com/Textualize/textual/releases/tag/v8.2.8)；当前解析器能够把 `CSI 13;2u` 解释为 `shift+enter`，参见[`_xterm_parser.py`](https://github.com/Textualize/textual/blob/06dbeef4bb70fb718236aa418ed658ef4667a126/src/textual/_xterm_parser.py#L366-L409)。

但是，`TextArea` 默认只处理普通 `enter` 插入换行，并没有把 Shift+Enter 定义为换行，参见[`TextArea._on_key`](https://github.com/Textualize/textual/blob/06dbeef4bb70fb718236aa418ed658ef4667a126/src/textual/widgets/_text_area.py#L1818-L1849)。MyClaw 应在自己的输入组件中拦截 Enter 发送，并在确认收到 `shift+enter` 时显式插入换行。

依赖方面，Textual 8.2.8 要求 Rich 14.2 或更高，并增加 `markdown-it-py`、`mdit-py-plugins`、`platformdirs`、`typing-extensions` 等运行时依赖；MyClaw 当前 Rich 上界 `<15` 仍可保留，但下界需要提升。官方元数据见[Textual 8.2.8 发布页](https://pypi.org/project/textual/8.2.8/)。

## prompt-toolkit 基线

prompt-toolkit 已经具备全屏 `Application`、布局、键绑定、对话框、鼠标与异步运行能力，参见[全屏应用文档](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/full_screen_apps.html)。它也能通过 pipe input 和 dummy output 做输入测试，参见[单元测试文档](https://python-prompt-toolkit.readthedocs.io/en/3.0.41/pages/advanced_topics/unit_testing.html)。

但它不能直接满足本次键盘契约。3.0.52 的官方 ANSI 映射将 xterm 的 Shift+Enter、Ctrl+Enter、Shift+Ctrl+Enter 序列都映射为普通 `Keys.ControlM`，丢失修饰状态，参见[`ansi_escape_sequences.py`](https://github.com/prompt-toolkit/python-prompt-toolkit/blob/3.0.52/src/prompt_toolkit/input/ansi_escape_sequences.py#L125-L131)。Windows 输入实现虽然读取 `ControlKeyState`，其 Shift 映射没有 Shift+Enter，参见[`win32.py`](https://github.com/prompt-toolkit/python-prompt-toolkit/blob/3.0.52/src/prompt_toolkit/input/win32.py)。官方特殊键列表同样没有 Shift+Enter，参见[键绑定文档](https://python-prompt-toolkit.readthedocs.io/en/3.0.52/pages/advanced_topics/key_bindings.html)。

继续使用 prompt-toolkit 意味着 MyClaw 需要自行实现 Kitty 协议启用、解析和恢复，还要自行构建 Markdown 增量渲染、左右消息布局、滚动缓存与 modal 焦点管理。它的依赖优势不足以抵消这些定制成本。

## Urwid

Urwid 可通过 `Frame`、`Pile`、`ListBox` 和 `Overlay` 组成全屏聊天布局及弹窗，参见[Widget 手册](https://urwid.org/manual/widgets.html)。它提供鼠标支持和 asyncio 事件循环适配，参见[主循环 API](https://urwid.org/reference/main_loop.html)。

其官方输入文档明确说明并非所有按键都会被终端发送，且不同终端行为不同；文档列出了 Enter 和部分带修饰键，但没有 Shift+Enter 或 Kitty 协商，参见[用户输入手册](https://urwid.org/manual/userinput.html)。Urwid 也没有内置 Markdown 流式组件，因此没有比 Textual 或 prompt-toolkit 更好的关键能力，予以排除。

## 终端协议与兼容矩阵

Kitty Keyboard Protocol 使用 `CSI ? u` 查询能力，并用 enhancement flags 控制输入。bit 1 只消除部分歧义；bit 8 才要求所有按键以转义序列报告；Shift 修饰值为基础值 1 加 Shift 位 1，因此 Shift+Enter 的规范序列为 `CSI 13;2u`。完整定义见[Kitty Keyboard Protocol](https://sw.kovidgoyal.net/kitty/keyboard-protocol/)。

| 环境 | 官方事实 | 对 MyClaw 的结论 |
| --- | --- | --- |
| Windows Terminal stable 1.24 | 现有稳定版早于 Kitty Keyboard Protocol 支持 | 不能可靠区分；必须 fallback 或用户映射 |
| Windows Terminal Preview 1.25+ | 1.25.622.0 官方发布说明新增 Kitty Keyboard Protocol 与 modifier state 支持 | 终端具备前提，但仍需 MyClaw/Textual 启用 bit 8 并确认能力 |
| ConPTY | 双向文本通道；终端把输入编码后送入伪控制台 | ConPTY 不会替应用恢复已丢失的 Shift 状态；关键在终端编码和库的输入路径 |
| macOS 的 Kitty/iTerm2 等兼容终端 | Kitty 官方实现列表包含多个现代终端 | capability probe 成功且 bit 8 生效时可支持 |
| Apple Terminal | Apple 仅官方提供自定义键映射；没有文档承诺 Kitty/CSI-u | 视为旧终端；使用 Ctrl+J fallback 或让用户映射 Shift+Enter |

Windows Terminal 1.25 Preview 的新增能力见[官方 1.25.622.0 发布说明](https://github.com/microsoft/terminal/releases/tag/v1.25.622.0)；稳定版本状态见[官方 releases](https://github.com/microsoft/terminal/releases)。Microsoft 将伪控制台描述为通过管道承载的文本通道，参见[控制台定义](https://learn.microsoft.com/en-us/windows/console/definitions)和[CreatePseudoConsole](https://learn.microsoft.com/en-us/windows/console/createpseudoconsole)。传统 VT 输入只规定 Alt/Control 等有限修饰编码，没有通用 Shift+Enter，参见[控制台虚拟终端序列](https://learn.microsoft.com/en-us/windows/console/console-virtual-terminal-sequences)。Apple Terminal 的可用保底路径是[自定义键映射](https://support.apple.com/guide/terminal/trml108/mac)。

## 推荐实现

1. 引入 `textual>=8.2.8,<9`，并把 Rich 下界提升到 Textual 所需版本；首期可暂时保留 prompt-toolkit，直到旧 REPL seam 被替换。
2. 实现两区 `App`：对话区使用可滚动容器，消息内容使用 Markdown；输入区使用自定义 `TextArea`，普通 Enter 触发发送，真正的 `shift+enter` 插入换行。
3. 在终端驱动生命周期内加入 Kitty capability adapter：查询支持，启用至少 bit 8；为正常文本输入启用/验证 associated text；退出时按协议恢复原状态。该适配应封装在单一驱动边界，避免与 Textual 自己的 push/pop 次序冲突。
4. capability gate 未通过时，界面必须使用可靠的备用换行键（建议 Ctrl+J），或明确提示当前终端不支持 Shift+Enter。产品若坚持 Shift+Enter 是唯一换行方式，就必须把支持范围限制到已验证的现代终端版本。
5. 测试分三层：原始 `CSI 13;2u` 解析单测；Textual `run_test()` 下的发送、换行、modal、滚动和 resize 测试；Windows Terminal stable/Preview、至少一个 Kitty 兼容 macOS 终端及 Apple Terminal 的真实终端验收。无头测试只能证明绑定逻辑，不能证明宿主终端会发出可区分序列。

最终决策是：**Textual 是 UI 框架首选；“Textual + Kitty bit 8 + capability gate + fallback”才是完整方案。旧终端无法满足零配置、可靠的 Shift+Enter。**
