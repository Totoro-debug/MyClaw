<!-- CODEGRAPH_START -->
## CodeGraph

CodeGraph indexes are local to each Git worktree. Before searching or reading code,
ensure that the current worktree has an index:

```powershell
pwsh -NoLogo -NoProfile -ExecutionPolicy Bypass -File scripts/ensure-codegraph.ps1
```

After the command succeeds, use `codegraph explore` or `codegraph node` before
`rg`, `find`, or direct file reads. Do not reuse another worktree's `.codegraph`
directory. If initialization fails, report the failure and then continue with the
available repository tools.
<!-- CODEGRAPH_END -->

## Agent 技能

### 问题跟踪

问题和产品需求文档（PRD）通过 GitHub Issues 进行跟踪，仓库为 `Totoro-debug/myclaw`。详见 `docs/agents/issue-tracker.md`。

### 领域文档

这是一个单一上下文仓库。详见 `docs/agents/domain.md`。

## 编程规范

### 1. 方案先行原则

严禁在方案评审完成前直接编写代码。开发前必须输出详尽的实施方案，内容至少涵盖技术选型、架构影响分析、数据流向及接口定义等；方案经评审确认后方可实施。

### 2. 任务原子化拆分

实施方案必须按业务逻辑拆分为多个独立、边界清晰的子 Task。每个子 Task 必须满足独立开发、独立测试、独立合并的要求，避免任务之间形成强耦合阻塞。

### 3. 量化验收标准

每个子 Task 必须明确定义可量化的完成评价指标，并以此作为开发完成与交付测试的唯一标准。

### 4. 最小化变更控制

严格遵循“按需修改”原则，控制代码改动范围。仅允许修改实现当前需求所必需的代码及配置文件；严禁夹带私货，不得顺手重构无关代码、修改非相关代码格式或调整无关逻辑。

### 5. 变更影响性评估

修改既有逻辑前，必须全面评估该改动对上下游调用链路、全局状态及周边模块的潜在影响，并记录评估结论。

### 6. 零回归保障

禁止引入任何新的缺陷或异常，且必须确保原有正常功能的行为与输出完全一致，实现零功能回归。

### 7. 事实依据原则

排查问题或扫描代码时，必须严格基于代码事实。对于不确定的逻辑，必须详细分析代码上下文及调用链路；严禁脱离代码实际进行主观臆断，避免因误判导致错误修复或非必要变更。
