# Agent Tool Calling 参数定义与校验方案调研

## 文档状态

- 状态：Completed research note
- 调研日期：2026-07-23
- 范围：Tool Calling 的输入参数定义、模型侧约束生成、宿主运行时校验、语义/业务校验，以及跨语言协议契约
- 一手资料范围：JSON Schema、Ajv、python-jsonschema、Pydantic v2、PydanticAI、Zod、Vercel AI SDK、OpenAI API、Anthropic API、Gemini API、MCP 2025-11-25

## 1. 结论先行

这些方案不是同一层的替代品。成熟实现通常把它们组合为如下链路：

```text
Schema 定义
    -> 转换为 Provider/MCP 可接收的 JSON Schema
    -> Provider 尽可能约束模型生成
    -> 宿主在执行前再次做结构校验
    -> 宿主做权限、状态和业务不变量校验
    -> 执行 Tool
    -> 可选：按 output schema 校验结构化结果
```

最重要的判断是：

1. **模型或 Provider 的 `strict` 只降低错误参数的产生概率，不是执行安全边界。** 宿主仍需在每次执行前校验输入，并独立做权限和业务校验。
2. **JSON Schema 2020-12 是当前最适合做跨语言、跨进程 canonical contract 的方案。** JSON Schema 官方当前发布版本是 2020-12；Ajv 和 python-jsonschema 都有成熟实现。[JSON Schema specification](https://json-schema.org/specification)、[JSON Schema validation specification](https://json-schema.org/draft/2020-12/json-schema-validation)
3. **Pydantic/Zod 适合做单语言内的 schema authoring 和类型推导。** 它们可以导出 JSON Schema，但自定义 validator、transform、运行时上下文检查并不能完整跨语言迁移。
4. **Provider 原生 strict/validated tool use 是生成层增强。** OpenAI、Anthropic 和 Gemini 都有 schema-constrained tool calling，但入口、schema 子集和 optional 语义不同；不能直接等同于完整 JSON Schema runtime validator。[OpenAI function calling strict mode](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)、[Anthropic strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)、[Gemini function calling modes](https://ai.google.dev/gemini-api/docs/generate-content/function-calling#function-calling-modes)
5. **MCP 是发现和传输 Tool contract 的协议，不是 validator。** 最新稳定版 2025-11-25 规定默认 dialect 为 JSON Schema 2020-12，客户端和服务端必须至少支持该 dialect；服务端仍必须校验所有输入。[MCP JSON Schema usage](https://modelcontextprotocol.io/specification/2025-11-25/basic#json-schema-usage)、[MCP Tools](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

对 MyClaw 的建议是：**继续以 JSON Schema 2020-12 + python-jsonschema 为 canonical contract 和强制执行边界；为不同 Provider 生成受控的 schema projection；在明确支持的 Provider route 上再开启原生 strict/validated 模式；权限和业务校验继续留在 Tool/Gateway 代码中。** 当前没有足够收益为此整体迁移到 PydanticAI 或 LangChain。

## 2. 必须分开的五层职责

| 层 | 负责什么 | 不负责什么 | 典型方案 |
| --- | --- | --- | --- |
| 1. Schema 定义 | 字段、JSON 类型、必填项、边界、枚举、额外字段策略 | 当前用户权限、资源是否存在、跨系统状态 | JSON Schema、Pydantic、Zod、TypeBox |
| 2. 模型/Provider 约束生成 | 让模型在解码时尽量只生成符合 schema 的参数 | 授权、安全、资源状态、Tool 自身副作用 | OpenAI/Anthropic `strict`、Gemini `VALIDATED`/`ANY` |
| 3. 宿主运行时校验 | 在 Tool 执行前拒绝缺字段、错类型、越界、未知字段 | 需要 I/O 或领域状态的判断 | python-jsonschema、Ajv、Pydantic、Zod |
| 4. 语义/业务校验 | 权限、Workspace 边界、记录存在性、跨字段约束、当前时间/余额等 | 通用数据形状的重复定义 | Tool service、policy、PydanticAI `args_validator`、普通代码 |
| 5. 跨语言可移植性 | 同一 wire contract 能否被 Python、TypeScript、MCP、Provider 共同理解 | 本地语言的任意函数或 transform | 标准 JSON Schema 最强；Pydantic/Zod 导出的可表达子集次之 |

这五层应各自有测试。把 Provider 返回过一次“合法参数”当作宿主可以跳过校验，会把模型输出和网络边界错误地视为可信输入。

## 3. 方案一：JSON Schema 2020-12 + Ajv / python-jsonschema

### 3.1 定义方式

直接维护 JSON Schema，把 `type`、`properties`、`required`、`enum`、数值和字符串边界等作为 Tool 的 wire contract。JSON Schema 的验证词汇用于断言 JSON instance 的结构约束；`properties` 中出现的字段默认并不等于必填，必须单独放入 `required`。[JSON Schema validation specification](https://json-schema.org/draft/2020-12/json-schema-validation)、[JSON Schema object reference](https://json-schema.org/understanding-json-schema/reference/object)

推荐的 Tool 输入基线：

- 根为 `type: object`。
- 明确 `required`。
- 通常设置 `additionalProperties: false`，避免模型拼写错误或未来字段被静默接受。
- 显式固定 dialect，例如 `$schema: https://json-schema.org/draft/2020-12/schema`；发往只支持子集的 Provider 时由 adapter 生成兼容投影。
- 启动或注册 Tool 时先校验 schema 本身，不要等到首个 Tool call 才发现 schema 写错。

### 3.2 宿主运行时校验

Ajv 把 schema 编译成高效 JavaScript 校验函数，并缓存编译结果；官方建议 schema 编译一次并复用，因为编译比执行校验慢。[Ajv getting started](https://ajv.js.org/guide/getting-started)、[Ajv managing schemas](https://ajv.js.org/guide/managing-schemas.html)

python-jsonschema 当前支持 Draft 2020-12 等多个 draft。`validate(instance, schema)` 会先检查 schema 本身；已知 schema 会重复使用时，官方建议直接复用明确版本的 validator，例如 `Draft202012Validator`。`iter_errors` 可以返回全部详细错误。[python-jsonschema documentation](https://python-jsonschema.readthedocs.io/en/stable/)、[python-jsonschema Validator API](https://python-jsonschema.readthedocs.io/en/stable/api/jsonschema/validators/)

两个容易遗漏的配置点：

- JSON Schema 的 `format` 在规范中通常是 annotation。python-jsonschema 不会仅因 schema 写了 `format` 就默认强制检查，必须显式传 `FormatChecker`。[python-jsonschema format validation](https://python-jsonschema.readthedocs.io/en/stable/faq/#my-schema-specifies-format-validation-why-do-invalid-instances-seem-valid)
- Ajv 的 `strict` 主要用于发现 schema 中被忽略或含糊的写法，不是 OpenAI 的 constrained decoding。Ajv 还提供 `removeAdditional`、`useDefaults`、`coerceTypes` 等会修改输入的选项；边界校验若要求 fail closed，应保持这些修改选项关闭，除非产品明确需要规范化行为。[Ajv options](https://ajv.js.org/options)

Ajv 使用 Draft 2020-12 时需要专用的 `Ajv2020` class，而且不能在同一 Ajv instance 中混用 2020-12 和旧 draft。这是多 dialect MCP host 需要提前设计的边界。[Ajv JSON Schema versions](https://ajv.js.org/json-schema.html)

### 3.3 语义/业务校验

JSON Schema 很适合声明局部、确定性约束；它不适合承担“该路径是否仍在 Workspace”“当前用户是否可退款”“记录在数据库中是否仍存在”这类依赖环境状态的检查。可以通过 Ajv custom keywords 扩展，但官方明确指出自定义关键字会让 schema 失去可移植性，因此更适合把这类检查留在普通业务代码中。[Ajv user-defined keywords](https://ajv.js.org/guide/user-keywords.html)

### 3.4 优劣

**优点**

- 标准化程度和跨语言可移植性最高。
- 与 OpenAI function parameters、MCP `inputSchema`/`outputSchema` 直接对接。
- schema 与执行语言解耦，适合插件、远程 Tool 和持久化 contract。
- Ajv 和 python-jsonschema 都能给出成熟、可测试的运行时校验。

**缺点**

- 手写较冗长，容易与函数签名或领域类型发生漂移。
- dialect、`format` 和 Provider 支持子集必须显式管理。
- 不自动生成 Python/TypeScript 领域对象。
- 把自定义业务逻辑塞入 validator extension 会损害可移植性。

**适合**

- Provider-neutral agent runtime。
- MCP server/client。
- Python 与 TypeScript 共用 Tool catalog。
- Tool schema 是公开或长期兼容 contract 的系统。

## 4. 方案二：Pydantic v2 + PydanticAI

### 4.1 Schema 定义

Pydantic 以 Python type annotation、`BaseModel`/`TypeAdapter` 和 `Field` 为定义入口，`model_json_schema()` 或 `TypeAdapter.json_schema()` 可生成符合 JSON Schema Draft 2020-12 和 OpenAPI 3.1 的 schema。[Pydantic JSON Schema](https://docs.pydantic.dev/latest/concepts/json_schema/)

这使 Python 项目可以从同一份类型定义获得：

- IDE/mypy 可见的 Python 类型；
- 宿主 runtime parsing/validation；
- 发往 Provider 或 MCP 的 JSON Schema 表示。

### 4.2 宿主运行时校验

Pydantic 默认会做类型 coercion，例如把字符串 `"123"` 转成整数 `123`。如果 Agent Tool 边界要精确区分模型生成的 JSON 类型，应在 model、field 或每次 validation call 上启用 strict mode；即使 strict mode 下，从 JSON 输入解析某些类型仍有特定宽松规则，必须按类型测试。[Pydantic strict mode](https://docs.pydantic.dev/latest/concepts/strict_mode/)

额外字段策略也必须显式决定。Pydantic `extra` 默认是 `ignore`，可配置为 `forbid`；Tool call 通常更适合 `forbid`，以免模型生成的未知字段被无声丢弃。[Pydantic ConfigDict.extra](https://docs.pydantic.dev/latest/api/config/#pydantic.config.ConfigDict.extra)

### 4.3 PydanticAI 的 Tool inference 与修复循环

PydanticAI 会从函数签名提取参数，并从 docstring 提取函数和参数说明来构造 Tool schema。普通 typed function Tool 在执行前用 Pydantic 按签名校验模型参数；失败时框架生成包含错误细节的 retry prompt，让模型修正参数。[PydanticAI function tools](https://ai.pydantic.dev/tools/)、[PydanticAI tool execution and retries](https://ai.pydantic.dev/tools-advanced/#tool-execution-retries-and-failures)

PydanticAI 还提供 `args_validator`，它在 Pydantic schema validation 之后、Tool 执行之前运行，官方定位就是跨字段、业务逻辑或基于 runtime context 的参数校验。这很好地体现了结构校验和语义校验应分层。[PydanticAI custom args validator](https://ai.pydantic.dev/tools-advanced/#custom-args-validator)

一个重要陷阱是：使用 `Tool.from_schema` 直接提供手写 JSON Schema 时，PydanticAI 官方明确说明不会自动执行参数校验，而是把参数直接作为 keyword arguments 传入。采用这一入口时必须另外配置 validator 或在宿主边界校验。[PydanticAI custom tool schema](https://ai.pydantic.dev/tools-advanced/#custom-tool-schema)

### 4.4 跨语言可移植性

Pydantic 生成出的标准 JSON Schema 部分可以跨语言；Python validator、任意函数、依赖注入 context、预处理和 coercion 逻辑不能随 JSON Schema 一起传输。若 schema 的实际接受范围依赖这些代码，其他语言只拿到 JSON Schema 时会出现行为差异。

### 4.5 优劣

**优点**

- Python 类型、runtime validation 和 JSON Schema generation 合一，减少重复定义。
- 错误信息和复杂领域对象构造能力强。
- field/model validator 适合本地复杂约束。
- PydanticAI 已内建参数错误反馈与受限重试流程。

**缺点**

- Python-centric；完整行为无法跨语言复现。
- 默认 coercion 和默认忽略额外字段不适合作为 Agent 边界的隐含策略。
- 生成的完整 JSON Schema 不保证落在每个 Provider 的 constrained decoding 子集内。
- 引入 PydanticAI 意味着同时采用其 Agent loop、retry 和 Tool abstraction，迁移面大于单独解决 schema validation。

**适合**

- Python-only Agent 或 API。
- Tool 参数本身就是复杂 Python domain model。
- 希望框架自动完成函数 schema inference、参数解析和模型修复重试。

## 5. 方案三：Zod v4 + Vercel AI SDK

### 5.1 Schema 定义与宿主校验

Zod 用 TypeScript API 定义 runtime schema，`.parse()`/`.safeParse()` 校验不可信输入，同时用 `z.infer<>` 推导静态 TypeScript 类型。[Zod basic usage](https://zod.dev/basics)

Zod v4 可以通过 `z.toJSONSchema()` 导出 JSON Schema，默认 target 为 Draft 2020-12。不过 `Date`、`Map`、`Set`、transform、`z.custom()` 等类型或行为没有可靠的 JSON Schema 对应物；官方默认对这些不可表达类型抛错，而不是假装可移植。[Zod JSON Schema conversion](https://zod.dev/json-schema)

### 5.2 Vercel AI SDK Tool Calling

Vercel AI SDK 的 Tool `inputSchema` 接受 Zod schema 或 JSON Schema。该 schema 一方面发给模型生成参数，另一方面用于校验模型返回的 Tool call；`tool()` helper 把 schema 的类型连接到 `execute` 参数，获得 TypeScript inference。[AI SDK tool reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool)、[AI SDK tool calling](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling)

每个 Tool 可设置 `strict: true`。AI SDK 只在 Provider 支持时使用它；不支持的 Provider 会忽略，且不同 Provider 支持的 schema 子集不同。因此 AI SDK 自己的 host validation 与 Provider strict 仍然是两个独立层。[AI SDK strict mode](https://ai-sdk.dev/docs/ai-sdk-core/tools-and-tool-calling#strict-mode)

AI SDK 的 `zodSchema()` 会把 Zod 转为 SDK 可用的 JSON Schema 并保留验证能力。递归 schema 可使用 references，但官方提醒并非所有模型/Provider 都支持 `$ref`。[AI SDK zodSchema](https://ai-sdk.dev/docs/reference/ai-sdk-core/zod-schema)

需要注意通用 Tool 的 `outputSchema` 文档只承诺类型推导，不应由此假设所有 `execute` 返回值都会自动做 runtime output validation；MCP adapter 在提供 `outputSchema` 时则明确会校验 `structuredContent`。[AI SDK tool reference](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool)、[AI SDK MCP tools](https://ai-sdk.dev/docs/ai-sdk-core/mcp-tools#typed-tool-outputs)

### 5.3 优劣

**优点**

- TypeScript runtime schema 与静态类型来自同一来源。
- Zod refinement 和错误处理体验成熟。
- AI SDK 提供 Provider abstraction、Tool input validation 和类型安全的 execute。
- 可以按需导出 JSON Schema 对接 MCP 或其他语言。

**缺点**

- Zod native validator/transform 是 JS/TS 行为，不能完整跨语言。
- Zod schema 转 JSON Schema 不是无损过程。
- Provider strict 可能被忽略，不能作为统一行为保证。
- input/output transform 可能让“模型看到的 JSON shape”和“execute 得到的对象”不一致，需要额外 contract test。

**适合**

- TypeScript-first Agent、Next.js/Vercel AI 应用。
- 强调本地开发体验和前后端共享类型。
- Provider abstraction 比跨语言 canonical schema 更重要的项目。

### 5.4 TypeBox + Ajv 作为 TypeScript 备选

TypeBox 直接构造内存中的 JSON Schema，同时推导 TypeScript 类型；再用 Ajv 编译校验。相较 Zod，它更接近“JSON Schema 是 source of truth”，适合 REST/RPC/MCP 等 wire contract，但定义更显式，validator 和 format 插件也要单独配置。[TypeBox official repository](https://github.com/sinclairzx81/typebox)、[Ajv getting started](https://ajv.js.org/guide/getting-started)

选择标准很简单：优先 TypeScript 本地 refinement/transform 体验时用 Zod；优先 schema 原样跨语言和跨进程传输时用 TypeBox + Ajv。

## 6. 方案四：Provider 原生 strict / validated Tool Calling

### 6.1 OpenAI strict Structured Outputs / Function Calling

OpenAI function Tool 通过 JSON Schema 定义 `parameters`。设置 `strict: true` 后，function call 会可靠遵循 schema，而不是 best effort；OpenAI 当前文档建议开启 strict。[OpenAI function calling](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)

strict schema 有额外要求：

- 每个 object 都必须设置 `additionalProperties: false`。
- `properties` 中的每个字段都必须列入 `required`。
- “可选”通常通过类型联合 `null` 表达。
- 不兼容的 strict schema 会在请求时被拒绝。
- Structured Outputs 只支持 JSON Schema 子集；例如当前不支持 `allOf`、`not`、`if/then/else`、`dependentRequired` 和 `dependentSchemas`。[OpenAI supported schemas](https://developers.openai.com/api/docs/guides/structured-outputs#supported-schemas)

当前 API 默认行为也不同：Responses 请求省略 `strict` 时会尝试把 schema 规范化为 strict，无法兼容时回退并在返回的 Tool 上显示 `strict: false`；Chat Completions 省略时仍默认 non-strict。为了避免依赖隐式行为，adapter 应显式设置并记录最终能力状态。[OpenAI function calling strict mode](https://developers.openai.com/api/docs/guides/function-calling#strict-mode)

因此 canonical JSON Schema 若允许字段省略，不能机械地加一个 `strict: true`。通常需要 Provider projection：

- 有稳定默认值的字段，在 Provider schema 中设为 required，让模型显式给出值。
- 语义上确实可空的字段，在 Provider schema 中设为 required + nullable。
- 保留原 canonical schema 作为宿主接受和校验的真实 contract。
- 对 projection 做 fixture/contract tests，避免 Provider schema 与宿主语义漂移。

### 6.2 边界和失败形态

Structured Outputs 能保证受支持 schema 的结构一致性，但安全拒绝可能不符合业务 schema，API 会通过独立的 `refusal` 字段表示；不完整响应也仍需处理。[OpenAI refusals](https://developers.openai.com/api/docs/guides/structured-outputs#refusals-with-structured-outputs)

即使结构完全合法，以下问题仍只能由宿主处理：

- 用户是否授权此次调用。
- 路径、账户、订单或资源是否存在。
- 参数组合在当前业务状态下是否允许。
- Tool 是否幂等、是否需要确认、是否越权。
- Provider、SDK、adapter 或未来版本是否出现实现错误。

### 6.3 优劣

**优点**

- 显著减少缺字段、错类型、非法 enum 和额外字段。
- 降低 schema error 后让模型修复重试的延迟与 token 成本。
- OpenAI SDK 可从 Pydantic/Zod 辅助生成 schema。

**缺点**

- OpenAI/model-specific，不是通用 Tool contract。
- 只支持 JSON Schema 子集，并对 required/nullable 有额外规则。
- refusal、截断和 Provider failure 仍需处理。
- 对权限、事实正确性、状态和业务不变量没有保证。
- generic OpenAI-compatible endpoint 不一定实现相同 strict 语义。

**适合**

- 作为支持该能力的 OpenAI Provider adapter 的增强层。
- 不能取代宿主 validator，也不应渗透成 Tool domain model 的唯一表示。

### 6.4 Anthropic 与 Gemini 的对应能力

Anthropic Messages API 同样允许在 Tool definition 顶层设置 `strict: true`，通过 grammar-constrained sampling 保证 Tool name 和 input 符合其支持的 JSON Schema 子集。它的请求字段是 `input_schema`，而且 strict schema 的 optional 规则并不等同于 OpenAI 的“所有 properties 都 required”，因此不能复用未经测试的 OpenAI projection。[Anthropic strict tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use)

Gemini Generate Content API 的 function calling 使用 OpenAPI schema 子集。`VALIDATED` 模式允许自然语言或 schema-compliant function call，`ANY` 模式强制 function call 并保证 schema adherence；官方同时提示过大或过深的 schema 可能被拒绝。[Gemini function calling modes](https://ai.google.dev/gemini-api/docs/generate-content/function-calling#function-calling-modes)

| Provider | 开启方式 | Contract 方言/子集 | 主要差异 |
| --- | --- | --- | --- |
| OpenAI | Tool `strict: true` | JSON Schema 子集 | 每层 object 关闭额外字段；所有 properties required；optional 用 nullable 表达 |
| Anthropic | Tool `strict: true` | JSON Schema 子集 | `input_schema`；optional 语义与 OpenAI 不同；同样不能替代宿主校验 |
| Gemini Generate Content | `VALIDATED` 或 `ANY` mode | OpenAPI schema 子集 | `VALIDATED` 可返回文本；`ANY` 强制调用；复杂 schema 可能被拒绝 |

这三个方案进一步证明：Provider projection 和 capability negotiation 应是 adapter 职责，canonical Tool contract 不应为了某一家 Provider 的 strict 子集而改写。

## 7. 方案五：MCP `inputSchema` / `outputSchema`

### 7.1 2025-11-25 稳定规范

MCP Tool definition 包含必需的 `inputSchema` 和可选的 `outputSchema`。两者都是 JSON Schema；在 2025-11-25 schema reference 中根类型限制为 object。`outputSchema` 描述 `CallToolResult.structuredContent`。[MCP 2025-11-25 schema reference](https://modelcontextprotocol.io/specification/2025-11-25/schema#tool)

MCP 的 dialect 规则是：

- schema 未写 `$schema` 时默认 JSON Schema 2020-12。
- schema 可以显式声明其他 dialect。
- client 和 server **必须至少支持 2020-12**。
- client 和 server 必须按声明或默认 dialect 校验 schema；不支持的 dialect 要以适当错误优雅失败。[MCP JSON Schema usage](https://modelcontextprotocol.io/specification/2025-11-25/basic#json-schema-usage)

不要把 draft 页面将来的提案误认为已进入 2025-11-25 稳定规范。例如放宽 Tool schema 根类型的 SEP-2106 当前仍是 Draft。[MCP SEP-2106](https://modelcontextprotocol.io/seps/2106-json-schema-2020-12)

### 7.2 谁负责校验

MCP 规范明确规定：

- Server 必须校验所有 Tool inputs。
- 若声明 `outputSchema`，Server 必须返回符合 schema 的 structured result。
- Client 应校验 structured result。
- Client 在把 Tool result 交给 LLM 前也应校验结果。[MCP Tools output schema and security considerations](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#output-schema)

输入校验错误和业务逻辑错误属于 Tool execution error，放在 `isError: true` 的 Tool result 中，使模型有机会根据可操作反馈自我修正；找不到 Tool、协议请求本身畸形等才是 JSON-RPC protocol error。[MCP Tool error handling](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#error-handling)

MCP 还明确说明，`structuredContent` 是 Server 产生的结果，与 LLM Structured Outputs/constrained generation 无关。MCP 不会替下游 Provider 自动开启 OpenAI strict。[MCP structured content](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#structured-content)

### 7.3 优劣

**优点**

- 标准化 Tool discovery、调用和输入/输出 wire contract。
- Provider-neutral、跨进程、跨语言。
- 明确 server/client 的输入输出校验责任和错误语义。
- `outputSchema` 能让 Tool result 也成为可验证 contract。

**缺点**

- MCP 只规定协议，不提供具体 validator 或业务规则实现。
- `outputSchema` 可选，生态中未必都提供。
- Provider constrained decoding 取决于 host 如何把 MCP schema 转交给具体模型。
- 多 dialect、Provider subset 和 MCP 稳定版本之间仍需 adapter/capability 管理。

**适合**

- 插件化 Tool ecosystem。
- 跨进程、本地或远程 Tool server。
- 多 Agent host 共用同一 Tool catalog。

## 8. 横向比较

| 方案 | Schema source of truth | Provider 约束生成 | 宿主 runtime 校验 | 业务校验 | 跨语言可移植性 | 主要代价 |
| --- | --- | --- | --- | --- | --- | --- |
| JSON Schema + Ajv/jsonschema | JSON Schema | 取决于 Provider | 强 | 普通代码/custom keyword | **最高** | 手写冗长、类型可能漂移 |
| Pydantic v2 + PydanticAI | Python 类型/模型 | 框架转 JSON Schema，再由 Provider 决定 | 强；注意 coercion/extra | 强，支持 context/retry | 中；仅导出部分可移植 | Python-centric、框架迁移面大 |
| Zod + Vercel AI SDK | Zod runtime schema | SDK 按 Provider 转发 `strict` | 强 | 强，refinement/普通代码 | 中；导出非无损 | JS/TS-centric、transform 不可移植 |
| TypeBox + Ajv | JSON Schema builder | 取决于 Provider | 强 | 普通代码/custom keyword | 高 | API 更显式、需配置 Ajv |
| Provider 原生 strict/validated | Provider 接收的 JSON Schema/OpenAPI 子集 | **强，限受支持模型/子集** | 无法替代宿主校验 | 无 | 低到中，Provider-specific | 子集和 optional 语义不统一 |
| MCP | Wire-level JSON Schema | 本身不约束模型 | 规定责任，不指定实现 | Server 实现 | **高** | 仍需 validator、adapter 和业务代码 |

## 9. 对 MyClaw 的具体建议

### 9.1 当前状态

MyClaw 已经接近推荐分层：

- `ToolDefinition` 只有 `name`、`description`、`input_schema`，尚无 `strict` 或 `output_schema`；`input_schema` 是 Provider-neutral JSON object。
- [`ToolGateway`](../../myclaw/tools/tool_gateway.py) 和 [Memory Task](../../myclaw/memory/memory_task.py) 使用 `Draft202012Validator(..., format_checker=FormatChecker())` 在 permission/执行前校验，但目前每次调用都会新建 Validator 和 FormatChecker。
- 具体 Tool 继续检查路径边界、资源类型、权限、cron 等语义条件。
- [`openai_compatible`](../../myclaw/provider/openai_compatible.py) 和 [`anthropic`](../../myclaw/provider/anthropic.py) adapter 当前都直传 canonical schema，且都没有设置 Provider 原生 `strict`。
- [`read_file`](../../myclaw/tools/core/read_file.py)、[`write_file`](../../myclaw/tools/core/write_file.py) 和 [`edit_file`](../../myclaw/tools/core/edit_file.py) 等 Tool 的可选字段带 `default` 且未全部列入 `required`，不能原样满足 OpenAI strict 的 schema 要求。
- [`pyproject.toml`](../../pyproject.toml) 已经依赖 `jsonschema>=4.23,<5`，没有 Pydantic/PydanticAI/Zod 依赖。

这说明当前主要缺口不是换 schema library，而是完善 schema 生命周期和 Provider capability 层。

### 9.2 推荐目标架构

1. **保留 JSON Schema 2020-12 为 canonical Tool contract。**
   - 继续由 `ToolDefinition` 所有。
   - schema 注册时用 `Draft202012Validator.check_schema()` fail fast。
   - 每个 Tool/Gateway 预构造并复用 validator，不在每次 call 重新构造。
   - 继续显式启用 `FormatChecker`，并固定项目实际依赖的 formats。

2. **新增 Provider schema projection，不修改 canonical contract 迎合某一个 Provider。**
   - Anthropic、generic OpenAI-compatible、OpenAI strict 分别由 adapter 投影。
   - projection 明确处理 optional/default/nullable、`$schema`、unsupported keyword 和 `$ref`。
   - projection 失败要在请求前返回稳定的配置/能力错误，而不是静默降级。

3. **只对明确声明支持的 Provider route 开启原生 strict/validated 模式。**
   - 当前 `openai-compatible` 可能连接本地或第三方实现，不能假设都完整支持 OpenAI strict。
   - 现有 Tool schema 普遍有可省略的默认字段，不能直接加 `strict: true`；先建立并测试 strict projection。
   - Anthropic strict 与 OpenAI strict 的 schema 规则分别测试，不共享未经验证的 projection；未来接入 Gemini 时也按其 OpenAPI 子集单独处理。
   - 即使 strict 开启，Gateway 的 runtime validation 仍必须保留。

4. **继续把语义、权限和安全校验放在宿主。**
   - Schema 只处理纯数据形状。
   - Permission Policy 必须在执行前读取已经结构校验过的参数。
   - 路径解析、Workspace/Agent Home 边界、文件身份、Schedule state 等仍由领域代码负责。

5. **为未来 MCP 扩展 `output_schema`，但不要提前引入完整 MCP runtime。**
   - 如果 Tool 结果以后需要跨进程复用，再让 `ToolDefinition` 可选携带 `output_schema`。
   - Server 侧验证输出，Host/Client 侧再次验证不可信远端结果。
   - 采用 MCP 2025-11-25 时固定默认 2020-12 和 root object 限制。

6. **Pydantic 可作为未来 authoring adapter，而非现在替换 canonical schema。**
   - 若 Tool 参数模型显著复杂、手写 schema 与 Python 类型漂移成为实际问题，可允许部分 Tool 从 Pydantic 生成 canonical JSON Schema。
   - 生成结果必须经过 schema meta-validation、Provider projection tests 和固定 fixture review。
   - 不建议仅为 Tool 参数校验迁移整个 Agent loop 到 PydanticAI 或 LangChain。

### 9.3 最小验证矩阵

每个 Tool 至少应覆盖：

| 测试面 | 必测案例 |
| --- | --- |
| Schema 自身 | dialect 正确、schema 能通过 meta-validation、未知 keyword 被发现 |
| 输入结构 | 缺 required、错 JSON type、额外字段、边界值、空字符串、null |
| Provider projection | canonical fixture 到每个 Provider payload 的 exact shape；unsupported keyword fail closed |
| OpenAI strict | 所有 object 关闭额外字段；所有 properties required；nullable/default 语义与宿主兼容 |
| 语义/权限 | 合法 shape 但越权路径、资源不存在、状态冲突、需确认操作 |
| 错误反馈 | schema error 与 business error 使用稳定、无敏感信息的 Tool result；不得执行副作用 |
| 输出 contract | 声明 output schema 时，Tool 和远端 MCP result 的成功/失败路径均校验 |

## 10. 决策速查

- **像 MyClaw 这样的 Python、Provider-neutral runtime**：JSON Schema 2020-12 + python-jsonschema；Provider adapter 单独投影；宿主业务校验。
- **Python-only 且参数模型复杂**：Pydantic v2；若接受框架 Agent loop，再考虑 PydanticAI。
- **TypeScript-first 产品**：Zod + Vercel AI SDK。
- **TypeScript 但 wire contract 优先**：TypeBox + Ajv。
- **支持原生约束的 Provider route**：在各自兼容的 schema projection 后启用 strict/validated 模式，但绝不删除宿主校验。
- **跨进程 Tool ecosystem**：MCP 2025-11-25 + JSON Schema 2020-12，并同时验证 input 和 structured output。

最终推荐不是在上述方案中五选一，而是：

```text
JSON Schema 2020-12 canonical contract
  + python-jsonschema/Ajv 宿主强制校验
  + Pydantic/Zod/TypeBox 可选 authoring adapter
  + OpenAI/Anthropic strict、Gemini validated 等 Provider 约束生成
  + 普通领域代码完成语义、权限与安全校验
  + MCP 负责跨进程发现与传输
```
