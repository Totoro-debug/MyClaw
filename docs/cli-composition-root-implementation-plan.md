# CLI Composition Root 与 Session-scoped Agent Loop 实施计划

## 文档状态

- 架构状态：已确认
- Implementation status: T1-T8 complete after final verification
- 适用版本：MyClaw `v0.1`
- 决策来源：[ADR-0017](adr/0017-use-cli-composition-root-and-session-scoped-agent-loop.md)
- 产品行为来源：[PRD](myclaw-personal-agent-prd.md)
- 运行时契约来源：[Runtime Contracts](myclaw-runtime-contracts.md)
- 领域语言来源：[CONTEXT.md](../CONTEXT.md)

本文是 ADR-0017 的可执行实施方案。它不改变已经确认的产品行为，而是把代码层 Runtime 聚合层迁移为
CLI composition root 与 Session-scoped Agent Loop，并给出可独立开发、测试和合并的原子 Task。

在本计划通过实施评审前不得修改实现代码。每个 Task 合并时必须保持仓库可运行，不得把临时兼容层暴露为
新的产品或 package contract；所有迁移期私有兼容代码必须在 T7 删除。

## 1. 目标与非目标

### 1.1 目标

1. 彻底删除代码层 Runtime 聚合层，包括 `RuntimeHost`、`PreparedRuntime`、`RuntimeBindings`、
   `prepare_runtime` 和 `_prepare_runtime`。
2. CLI private async root 成为唯一 composition root，并拥有全部 Runtime-Lifetime 组件的生命周期。
3. 一个 Agent Loop 对应一个 Runtime Generation，并创建、拥有全部 Session-scoped 组件。
4. CLI 创建并终身复用一个 Message Bus；任意 `/resume` 原子清空其 Inbound 和 Outbound。
5. `/resume` 即使选择当前 Session，也创建新的 Agent Loop、刷新 Skill Snapshot 并执行完整切换流程。
6. Memory Manager 只负责持久化与内存快照；Dream 负责长期记忆的语义生成。
7. Schedule Service 创建并持有 Schedule Store，同时调度 User Job 与内置 Dream System Job。
8. 删除 `Workspace` 包装类和 `MemoryTaskScheduler`，分别改为直接注入绝对 `Path` 和使用
   Schedule Service 调度 Dream。
9. 保持现有 Session、Summary、Cursor、Long-term Memory、Schedule 和 Tool Artifact 持久化 schema。

### 1.2 非目标

- 不新增 daemon、HTTP/IPC、跨进程协调或多 Workspace Runtime。
- 不改变固定十 Tool Catalog、Tool Confirmation、安全边界或 Provider retry/fallback 契约。
- 不改变 User Schedule Job 的 UUID4、公开 list/remove 或 Schedule Session 行为。
- 不给 Dream 创建 Conversation Session、Schedule Session、Tool Artifact 或 foreground Outbound。
- 不引入依赖注入容器、Service Locator、全局单例或改名后的 Runtime Host。
- 不迁移、重写或删除 legacy scheduled-work state 和 legacy Runtime Log 文件。

## 2. 最终所有权模型

| 生命周期 | 所有者 | 被拥有组件 |
| --- | --- | --- |
| Runtime Lifetime | CLI private async root | normalized Workspace `Path`、`WorkspaceState`、Message Bus、Model Router、Memory Manager、Dream、Schedule Service、Management Service/Dispatcher、Terminal application、current Agent Loop reference |
| Runtime Generation | Agent Loop | Session、Skill Loader、immutable full Skill Snapshot、Context Builder、Conversation Summary Manager、Task Framer、Tool Gateway、Agent Runner、foreground consumer、active run、title work、confirmation state |
| Schedule Lifetime | Schedule Service | Schedule Store、dispatcher、active User/System Job tasks、terminal commit tasks、cron/every cursors |
| Dream Lifetime | Dream | dedicated Agent Runner、restricted Tool Gateway、run mutex、active Dream task |
| Session Lifetime | Session | conversation history、metadata、`last_consolidated`、ordered snapshot persistence |

所有权遵守以下规则：

- CLI 只创建跨 Session 可复用的对象，不创建 Session 内部协作者。
- Agent Loop 只创建当前 Session/Runtime Generation 内的对象，不关闭或重建 CLI-owned 对象。
- Service 持有自己的 Store；CLI 持有 Service 生命周期，不直接持有 Store。
- Terminal 只持有交互和展示 seam，不启动或关闭业务组件。
- Management 与 Schedule 通过 CLI callable 获取当前 Agent Loop，不缓存 generation reference。

## 3. 技术选择

### 3.1 CLI async composition root

同步 Typer callback 保持现有命令边界，只负责：

1. 加载配置。
2. 校验 TTY。
3. 使用 `asyncio.run()` 进入 private async root。
4. 把稳定的安全错误转换为 CLI exit code。

private async root 直接构造顶层组件，调用 `TerminalConversationApp.run_async()`，并在 `finally` 中按所有权
逆序 awaited close。不得使用一个 aggregate/dataclass 把全部组件重新包装成新的 Runtime 对象。

### 3.2 直接 Path 注入

删除只封装一个 `Path` 的 `Workspace`。CLI 使用与当前 `Workspace.from_path()` 相同的 lexical normalization
语义生成绝对 `Path`，不额外解析 filesystem alias。`WorkspaceState` 直接存储该 `Path`；Tool、Session、Memory、
Schedule 和日志组件继续使用现有 `HOST_FILESYSTEM` canonical containment 检查。

### 3.3 稳定对象加动态 callable

CLI 保存唯一的 `current_agent_loop: AgentLoop | None`，并提供两个闭包：

```python
def current_agent_loop() -> AgentLoop: ...

async def replace_agent_loop(session_id: str, force: bool) -> None: ...
```

Management Service 获得两个闭包；Schedule Service 的 User Job executor 只获得
`current_agent_loop()`。每次调用都重新解引用，保证切换后不会使用旧 Agent Loop。

### 3.4 同步 prepare/preflight，异步 lifecycle

Agent Loop constructor 创建所有 Session-scoped 对象，但不创建 asyncio task。CLI 随后同步调用
`AgentLoop.preflight()`。只有 preflight 成功后才允许 `start()` 创建 task。

- `start()`：启动 foreground consumer。
- `close()`：正常 awaited shutdown，保留 Session final-save 语义。
- `abort()`：异步取消并等待 Agent Loop 内部 task，然后执行 `Session.abandon()`，不做 final save。

首次启动或 `/resume` 的 constructor/preflight failure 都是 Terminal Conversation fatal error。失败时 CLI 输出
简短稳定 code/message，不输出 traceback、底层异常、Skill 正文或敏感路径。

### 3.5 不可变 generation Skill Snapshot

每个 Agent Loop 创建自己的 Skill Loader。Loader 对每个候选只完整读取一次 UTF-8 document，并用同一份
内容完成 canonical path、frontmatter、metadata、`always` 和正文校验。有效候选形成 frozen dataclass/tuple
组成的 immutable Skill Snapshot；无效候选被丢弃并记录不含正文的安全 warning。

当前 generation 的 manual invocation 与 always-loaded projection 只读取 Snapshot 中的 frozen document，
不再次访问文件系统。模型主动调用普通 `read_file` 仍保持实时 Tool 语义。首次启动和任意 `/resume` 都创建
新的 Loader/Snapshot，因此可以看到最新 Skill 状态。

### 3.6 Memory Manager 与 Dream 分离

Memory Manager 接收 `WorkspaceState` 并创建/持有 Summary Store、Summary Cursor Store、Long-term Memory Store
和当前 memory snapshot。它不得 import Model Router、Agent Runner、Tool Gateway、prompt 或 scheduling module。

Dream 接收 Memory Manager、Model Router 和 memory route 配置，创建 dedicated Agent Runner 与 restricted
Tool Gateway。Dream 只生成长期记忆的语义内容；所有 Summary、Cursor 和 `memory.md` 写入都通过 Memory Manager。

保留现有 Summary Cursor 语义：Dream 在模型工作前领取 batch 并推进 cursor；no update、edit success、
model failure 和 Tool/edit failure 都保留已推进的 cursor，不自动重试该 batch。

### 3.7 Schedule Service 与 Dream System Job

Schedule Service constructor 接收 `WorkspaceState` 并创建自己的 Store。User Job 继续使用 UUID4；仅内置
System Job 可以使用保留 symbolic ID，当前唯一值为 `dream`。

CLI 在 Schedule Service 初始化后、dispatcher 启动前注册：

```text
job_id = "dream"
source = "system"
schedule = [memory].schedule + startup local IANA timezone
```

Dream Job identity、注册、校正和 dispatch 只检查 `job_id` 与 `source`，不检查 `message`。缺失 Job 创建时使用
稳定的内部非空 placeholder message；已有 Job 的 message 原样保留，且永不作为 Dream 模型输入。

注册规则：

1. ID 不存在：创建并持久化 Dream Job。
2. ID/source 匹配且 cron/timezone 一致：跳过写入。
3. ID/source 匹配但 cron/timezone 变化：更新 schedule，保持 Job identity 和 state。
4. 相同 ID 但 source 不是 `system`：以 `schedule_state_error` 阻止启动。

dispatch 规则：

```python
if job.job_id == "dream" and job.source == "system":
    await dream.run()
elif job.source == "user":
    await current_agent_loop().run_schedule_job(job)
else:
    raise ScheduleStateError(...)
```

Dream 无 pending summary 或处理成功映射为 `ok`；安全失败映射为 `error`；`memory_task_running` 表示当前
occurrence skipped，不修改 Job state。Dream Job 不创建 Schedule Session。

### 3.8 Message Bus 原子 reset

Message Bus 在一个共享的 async coordination boundary 下维护 Inbound 与 Outbound FIFO。九个公开 async 操作是：

```python
async def inbound_snapshot() -> tuple[InboundMessage, ...]: ...
async def put_inbound(message: InboundMessage) -> None: ...
async def get_inbound() -> InboundMessage: ...
async def pause_inbound_delivery() -> None: ...
async def resume_inbound_delivery() -> None: ...
async def drain_inbound() -> tuple[InboundMessage, ...]: ...
async def put_outbound(message: OutboundMessage) -> None: ...
async def get_outbound() -> OutboundMessage: ...
async def reset() -> None: ...

def set_inbound_changed_callback(callback: Callable[[tuple[InboundMessage, ...]], None] | None) -> None: ...
def unbind_inbound_changed_callback(callback: Callable[[tuple[InboundMessage, ...]], None]) -> None: ...
```

`reset()` 在同一临界区内清空两个 FIFO，再在释放 coordination 后用空 Inbound snapshot 调用现有 callback。
Message Bus identity 在整个 Runtime Lifetime 内保持不变；Agent Loop 不 detach、close 或替换它。

## 4. 目标接口

以下签名限定职责和依赖方向；具体 private helper 可以调整，但不得扩大 ownership。

```python
@dataclass(frozen=True, slots=True)
class LoadedSkill:
    metadata: SkillMetadata
    document: str
    always: bool


@dataclass(frozen=True, slots=True)
class SkillSnapshot:
    root: Path
    skills: tuple[LoadedSkill, ...]


class SkillLoader:
    def __init__(
        self,
        *,
        root: Path,
        reserved_names: Collection[str],
        enable_always_load: bool,
    ) -> None: ...

    def load(self) -> SkillSnapshot: ...


@dataclass(frozen=True, slots=True)
class SummaryClaim:
    previous_cursor: int
    cursor: int
    entries: tuple[SummaryEntry, ...]


class MemoryManager:
    def __init__(self, workspace_state: WorkspaceState) -> None: ...

    @property
    def long_term_path(self) -> Path: ...
    async def append_summary(self, content: str, timestamp: datetime) -> SummaryEntry: ...
    async def claim_summaries(self, limit: int) -> SummaryClaim: ...
    async def read_long_term(self) -> str: ...
    async def edit_long_term(
        self, *, old: str, new: str, replace_all: bool = False
    ) -> str: ...
    def memory_snapshot(self) -> str: ...


class Dream:
    def __init__(
        self,
        *,
        memory_manager: MemoryManager,
        model_router: DreamModelRouter,
        batch_size: int,
        max_iterations: int,
    ) -> None: ...

    async def run(self) -> DreamResult: ...
    async def close(self) -> None: ...
    async def wait_until_idle(self) -> None: ...
    def abort(self) -> None: ...
    async def abort_and_wait(self) -> None: ...


class ScheduleService:
    def __init__(
        self,
        *,
        workspace_state: WorkspaceState,
        clock: ScheduleClock,
        execute_user_job: Callable[[ScheduleJob], Awaitable[None]],
        execute_dream: Callable[[], Awaitable[object]],
        timezone_name: str | None = None,
    ) -> None: ...

    async def register_dream_job(self, *, schedule: JobSchedule) -> ScheduleJob: ...
    def start(self) -> None: ...
    async def pause_and_drain(self) -> None: ...
    def resume(self) -> None: ...
    async def close(self) -> None: ...
    def abort(self) -> None: ...
    async def abort_and_wait(self) -> None: ...


class AgentLoop:
    def __init__(
        self,
        *,
        workspace_path: Path,
        workspace_state: WorkspaceState,
        agent_home: AgentHome,
        configuration: UserConfiguration,
        bus: MessageBus,
        schedule_service: ScheduleService,
        model_router: ModelRouter,
        memory_manager: MemoryManager,
        session_id: str | None,
        now: Callable[[], datetime],
        new_uuid: Callable[[], UUID],
        monotonic_now: Callable[[], float],
    ) -> None: ...

    @property
    def session(self) -> Session: ...

    @property
    def skill_metadata(self) -> tuple[SkillMetadata, ...]: ...

    @property
    def control(self) -> TerminalAgentLoopControl: ...

    def preflight(self) -> None: ...
    async def start(self) -> None: ...
    async def abort(self) -> None: ...
    async def close(self) -> None: ...
    async def run_schedule_job(self, job: ScheduleJob) -> None: ...


class TerminalConversationApp:
    def __init__(
        self,
        *,
        bus: MessageBus,
        control: TerminalAgentLoopControl,
        management_dispatcher: ManagementCommandDispatcher,
        skill_metadata: tuple[SkillMetadata, ...],
    ) -> None: ...

    async def rebind_agent_loop(
        self,
        *,
        control: TerminalAgentLoopControl,
        skill_metadata: tuple[SkillMetadata, ...],
        session_projection: ForegroundConversationProjection,
    ) -> None: ...
```

Management Service 接口保持现有命令契约，但其依赖改为：

```python
class ManagementViewService:
    def __init__(
        self,
        agent_home: AgentHome,
        *,
        current_agent_loop: Callable[[], AgentLoop],
        workspace_state: WorkspaceState,
        replace_agent_loop: Callable[[str, bool], Awaitable[None]],
        prepare_session_resume: Callable[[str], Awaitable[None]],
        memory_manager: MemoryManager,
        dream: Dream,
        schedule_status: Callable[[], dict[str, object]],
        now: Callable[[], datetime],
        monotonic: Callable[[], float],
    ) -> None: ...
```

`/status` 每次从 current Agent Loop 读取 Session、context projection 和 generation start monotonic time，因此
`uptime_seconds` 在任意 `/resume` 后重置；Schedule health 继续来自同一个 lifetime Schedule Service。

## 5. 数据流

### 5.1 首次启动

```text
Typer callback
  -> load configuration / validate TTY
  -> asyncio.run(private CLI root)
      -> normalize Workspace Path
      -> initialize WorkspaceState
      -> create MessageBus
      -> create ModelRouter
      -> create MemoryManager
      -> create Dream
      -> create ScheduleService (creates Store)
      -> create AgentLoop(session_id=None)
      -> AgentLoop.preflight()
      -> prepare ScheduleService start
      -> register/reconcile Dream System Job
      -> set current AgentLoop reference
      -> create Management Service/Dispatcher
      -> create TerminalConversationApp
      -> start AgentLoop and ScheduleService
      -> await TerminalConversationApp.run_async()
      -> finally: Management deactivate
      -> Schedule pause_and_drain(), then close()
      -> pending/active Agent Loop abort() or close()
      -> Dream close()
      -> Model Router close()
```

正常情况下所有 constructor/preflight 在创建业务 task 前完成。任何 constructor/preflight failure 都直接退出，
不会进入可交互 Terminal。

### 5.2 Foreground Agent Run

```text
Terminal -> shared MessageBus.inbound
  -> current AgentLoop consumer
  -> Task Framer
  -> Conversation Summary Manager
       -> MemoryManager.append_summary()
       -> Session.last_consolidated update after successful append
  -> Context Builder
  -> AgentRunner + ToolGateway
  -> Session append/persist
  -> shared MessageBus.outbound
  -> Terminal
```

### 5.3 Dream

```text
/dream or Dream System Job
  -> Dream.run()
  -> MemoryManager.claim_summaries(batch_size)
       -> read cursor
       -> select pending summaries
       -> advance cursor before model work
  -> Dream dedicated AgentRunner / restricted ToolGateway
  -> MemoryManager.read_long_term() / edit_long_term()
  -> refresh MemoryManager snapshot after successful edit
  -> DreamResult
```

### 5.4 User Schedule Job

```text
ScheduleService dispatcher
  -> source=user
  -> CLI execute_user_job closure
  -> current_agent_loop().run_schedule_job(job)
  -> isolated Schedule Session
  -> current generation AgentRunner / ToolGateway
  -> no confirmation and no foreground Outbound
  -> ScheduleService commits terminal state
```

### 5.5 Dream System Job

```text
ScheduleService dispatcher
  -> job_id=dream and source=system
  -> Dream.run()
  -> no AgentLoop
  -> no Schedule Session
  -> no Job message model input
  -> no foreground Outbound
  -> map DreamResult to Job terminal state
```

### 5.6 `/resume`

The contract is explicit: target preparation is a precondition and completes while the old generation is still available:
construct the target Agent Loop and run synchronous `preflight()`. A target constructor or
preflight failure is fatal, ends Terminal Conversation, and leaves the CLI `finally` cleanup
for the still-owned components. The final linearization refinement formed during later
implementation review is not a claim about the original parent issue wording. After target
preparation succeeds, the successful cutover is exactly:

`quiesce_for_rebind -> pause_and_drain -> current unavailable -> old abort/drain -> bus.reset() -> rebind_agent_loop -> target.start() -> publish current -> schedule_service.resume()`

Here `current unavailable` is the CLI current reference being cleared before old-loop
abort/drain. Target `start()` is successful activation; only then is the target published as
current, and Schedule resumes after publication. An active foreground run still requires the
existing `force` confirmation; a rejected target is aborted without entering cutover. The old
Session is abandoned without final save. Choosing the current Session runs the same flow.
Schedule Service, Dream, Memory Manager, Model Router and Message Bus are not closed or
recreated during replacement.

After Terminal `run_async()` returns, the actual CLI shutdown order is `Management deactivate -> Schedule pause_and_drain + close -> Loop close/abort -> Dream close -> Model Router close`; Terminal exit/unmount cleanup is already complete at that boundary.

## 6. 原子 Task 与量化验收

### T0：固化文档契约

状态：已完成。

范围：

- 更新 `CONTEXT.md`。
- 新增 ADR-0017。
- 在 ADR-0014、ADR-0016 标记被取代的决定。
- 更新 PRD 与 Runtime Contracts。

验收：

- `git diff --check` exit code 为 `0`。
- 实现代码和测试文件变更数为 `0`。
- ADR、PRD、Runtime Contracts 对 ownership、Dream、Skill Snapshot、Schedule Job 和 `/resume` 的描述一致。

### T1：移除 Workspace 包装类

主要范围：

- 删除 `myclaw/agent/workspace.py`。
- 修改 `myclaw/agent/workspace_state.py` 直接保存 `Path`。
- 修改 Agent、Tool、Session、Memory、Schedule、Management、Session Log 的 Workspace 参数。
- 迁移 `tests/fixtures/paths.py` 与所有 Workspace fixture。
- 删除或改写 `tests/test_workspace.py`。

实现要求：

- 抽取一个小型 normalization function 可以接受 `Path`/`PurePath`，但不得重新创建 Workspace value object。
- 继续使用当前 lexical absolute normalization，不自行增加 `resolve()` 改变 symlink 语义。
- canonical containment、外部路径确认和 Workspace State 安全检查保持原样。

量化验收：

- `rg "from myclaw\.agent\.workspace|class Workspace\b|\bWorkspace\(" myclaw tests` 返回 `0` 个匹配。
- `myclaw/agent/workspace.py` 和旧 Workspace 专用测试文件不存在。
- Workspace State、Session IO security、file Tool、directory Tool、grep、Exec、Schedule Store 测试通过率 `100%`。
- Workspace 内/外路径的既有 confirmation/refusal snapshot 无变化。

独立合并条件：该 Task 不依赖其他 Task；完成后现有 Runtime 层仍能基于 `Path` 正常工作。

### T2：引入 Skill Loader 与完整 generation Snapshot

主要范围：

- 重构 `myclaw/skills/catalog.py`，必要时新增 `myclaw/skills/loader.py`。
- 增加 `LoadedSkill`、`SkillSnapshot`、`SkillLoader`。
- 删除 Runtime-lifetime Snapshot 和 manual `read_body` seam。
- 在迁移期间让现有 generation preparation 每次调用 Loader；T5 再把调用位置收进 Agent Loop。
- 更新 Skill、Context、Agent Loop、Terminal completion 与 CLI error 测试。

实现要求：

- 每个候选只打开并完整读取一次。
- metadata/body/always/path validation 使用同一份 document。
- Snapshot 只含 immutable tuple/frozen dataclass，不暴露 mutable name mapping。
- manual invocation 在 title work 之前只解析 Snapshot，不执行文件 IO。
- invalid candidate 只记录 candidate path 和 reason，不记录正文。

量化验收：

- fake filesystem counter 证明每个候选在一次 Loader execution 中完整读取次数恰好为 `1`。
- 当前 generation 中修改/删除 Skill 后，manual invocation 仍得到 frozen document；构造新 generation 后得到最新状态。
- 同 Session `/resume` 也刷新 Snapshot。
- `rg "RuntimeSkillSnapshot|build_runtime_skill_snapshot|SkillUnavailableError|\.read_body\(" myclaw tests`
  返回 `0` 个匹配。
- invalid Skill 日志中 document 内容匹配数为 `0`。
- Skill、Context、Agent Loop、CLI、Terminal completion 测试通过率 `100%`。

独立合并条件：Task 内允许保留一个只由旧 Runtime preparation 调用的 private Loader adapter；不得增加新的
公开 Runtime abstraction，该 adapter 必须在 T7 删除。

### T3：拆分 Memory Manager 与 Dream

主要范围：

- 将当前 `myclaw/memory/memory_task.py` 拆为 persistence-oriented Memory Manager 与 Dream execution。
- 建议新增 `myclaw/memory/manager.py`、`myclaw/memory/dream.py`，保留 Store/record module 的既有边界。
- 将 Memory Tool、prompt、model loop、运行互斥和结果类型迁入 Dream。
- Conversation Summary Manager 通过 Memory Manager append。
- `/memory` 使用 Memory Manager 磁盘读取；`/dream` 调用 `Dream.run()`。
- 迁移期允许 MemoryTaskScheduler 临时改为调用 Dream；T4 删除 Scheduler。

实现要求：

- Memory Manager 创建并拥有 Summary、Cursor、Long-term Memory Store 与 runtime memory snapshot。
- `append_summary()` 成功后 Conversation Summary Manager 才更新 Session `last_consolidated`。
- `claim_summaries()` 维持当前 cursor 预推进和 no-retry 语义。
- Dream 创建独立 Agent Runner 和 restricted Tool Gateway，只注册 Long-term Memory read/edit Tool。
- 保留稳定 error code `memory_task_running` 和现有 `/dream` 用户可见结果字段。

量化验收：

- `myclaw/memory/manager.py` 对 Model Router、Agent Runner、Tool Gateway、prompt、cron/scheduler 的 import 数为 `0`。
- Dream Runner/Gateway construction 次数在一个 Runtime Lifetime 内各为 `1`。
- no pending summary 时 provider call 数为 `0`。
- no update、edit success、model failure、Tool failure、edit failure 五条路径的 cursor 值与当前契约一致。
- Summary append failure 时 `last_consolidated` 变化量为 `0`。
- `/memory`、`/dream`、Summary、Cursor、Memory Tool 测试通过率 `100%`。

独立合并条件：旧 MemoryTaskScheduler 可以作为 private compatibility caller 暂时存在，但不得继续包含 memory
生成逻辑。

### T4：Schedule Service 持有 Store 并调度 Dream

主要范围：

- 修改 `myclaw/schedule/model.py` 支持 source-aware Job ID validation。
- 扩展 `myclaw/schedule/store.py` 的 System Job 注册/校正与内部 terminal commit。
- 修改 `myclaw/schedule/service.py` 自行创建 Store，并支持 User/Dream 两条 executor。
- 增加 pause/drain/resume lifecycle。
- 删除 `myclaw/memory/memory_scheduler.py` 及其专用测试。
- 将 Scheduler 测试中仍有效的 cron/timezone/DST cases 迁入 Schedule Service tests。

实现要求：

- `source="user"` 必须始终使用 canonical UUID4。
- `source="system"` 当前只接受保留 ID `dream`。
- public list/remove 仍只处理 User Job；System Job 不暴露给 Schedule Tool。
- Store public mutation 与 internal System mutation 使用不同方法，不能让 Tool 添加 symbolic ID。
- `commit_terminal()`、internal remove/update 必须同时支持 UUID4 User ID 与保留 System ID。
- pause 先阻止新 occurrence，再取消并等待 dispatcher-owned run/commit task。

量化验收：

- 以下注册场景各至少一个 contract test：missing、exact match、cron changed、timezone changed、wrong source。
- exact match 的 Schedule Store write 次数为 `0`。
- User add 产生的 Job ID 通过 UUID4 校验率为 `100%`。
- public snapshot 中 System Job 数为 `0`。
- Dream Job 创建的 Schedule Session 数为 `0`，Agent Loop invocation 数为 `0`。
- User Job 继续创建/加载 `schedule_<uuid4>` Session。
- `pause_and_drain()` 返回时 active Job、run task、terminal commit task 数均为 `0`。
- `rg "MemoryTaskScheduler|memory_scheduler" myclaw tests` 返回 `0` 个匹配。
- Schedule model/store/service/tool 与迁移后的 cron/DST tests 通过率 `100%`。

依赖：T3。

### T5：Agent Loop 接管 Session-scoped 组件

主要范围：

- 重构 `myclaw/agent/loop.py` constructor 与 lifecycle。
- Agent Loop 内创建 Session、Skill Loader/Snapshot、Context Builder、Conversation Summary Manager、Task Framer、
  Tool Gateway 和 Agent Runner。
- Message Bus 改为 constructor injection。
- 保留 User Schedule Job execution，但只允许 `source="user"`。
- `abort()` 改为 async 并等待内部 task。
- 迁移 Runtime preparation、Agent Loop、Schedule Loop、Session title、shutdown tests。

实现要求：

- `session_id=None` 创建新的 foreground Session；非空时严格 load 当前 Workspace 的目标 Session。
- constructor 不启动 task，不持久化空 Session。
- `preflight()` 验证 Tool schemas、Skill budget、context projection、clock 和其他同步 invariant。
- Agent Loop 不调用 Message Bus reset/close/detach。
- abort 取消 confirmation、active run、consumer、title work，并等待它们结束后调用 `Session.abandon()`。
- normal close 保留现有 bounded Session final save。

量化验收：

- Agent Loop constructor 不再接受已构造的 Session、Context Builder、Task Framer、Tool Gateway 或 Agent Runner。
- 每个 Agent Loop 创建上述 Session-scoped component 各恰好 `1` 个。
- CLI-owned Message Bus、Model Router、Memory Manager、Schedule Service 均未被 Agent Loop close。
- abort 返回后 consumer、execution、title task 未完成数量为 `0`。
- abort 返回后旧 generation 新增 Outbound 数量为 `0`，Session mutation 数量为 `0`。
- normal close、forced abort、title、foreground、User Schedule Job tests 通过率 `100%`。

依赖：T1、T2、T4。

### T6：Message Bus reset 与 Terminal presentation rebind

主要范围：

- 修改 `myclaw/agent/message_bus.py` 使用共享 coordination 并增加 `reset()`。
- 修改 `myclaw/terminal/conversation.py` 删除业务 lifecycle ownership。
- 增加 generation presentation rebind 接口。
- 更新 Message Bus、Terminal、REPL bus、confirmation 和 resume UI tests。

实现要求：

- Inbound 与 Outbound 必须在同一个 reset 临界区内清空。
- waiting getter 在 reset 后继续等待下一条新消息，不接收被清除的旧消息。
- reset 释放 coordination 后调用一次空 Inbound snapshot callback。
- Terminal constructor 不再接收 `start_runtime`、`close_runtime`、`runtime_host` 或 RuntimeBindings。
- Terminal `on_mount` 只绑定 bus/control/UI worker；`on_unmount` 只关闭 UI-owned state。
- rebind 时 Bus 和 Management Dispatcher identity 不变，只替换 control、Skill metadata 和 Session projection。

量化验收：

- 并发 barrier test 覆盖 inbound put/get、outbound put/get 与 reset，观察到的状态只能是 reset 前或 reset 后，
  半清空状态数量为 `0`。
- reset 后 Inbound/Outbound 长度均为 `0`，Bus object identity 未变化。
- reset callback 调用次数恰好为 `1`，参数为 `()`。
- Terminal constructor 中 Runtime lifecycle callback 参数数为 `0`。
- Terminal mount/unmount/rebind、Message Bus、REPL bus、confirmation tests 通过率 `100%`。

独立合并条件：可在 T7 前保留旧 Runtime 到新 Terminal API 的 private adapter，但 Terminal 本身不得重新取得
业务 lifecycle ownership。

### T7：CLI 成为唯一 composition root 并删除 Runtime 层

主要范围：

- 重写 `myclaw/terminal/cli.py` 的默认入口，新增 private async root。
- `TerminalConversationApp.run()` 改为由 CLI `await run_async()`。
- CLI 直接创建所有 Runtime-Lifetime 组件、current Agent Loop reference 和 replacement transaction。
- Management Service/Dispatcher 改为 lifetime object，使用 current/replace callables。
- Runtime Status 改为从当前 Agent Loop 动态读取 Session/generation 状态。
- 删除 `myclaw/agent/runtime.py`。
- 删除迁移期 private adapters，并迁移 `tests/runtime_bus.py` 与 runtime generation/shutdown/resume fixtures。

实现要求：

- CLI startup 顺序必须与 5.1 一致。
- replacement 顺序必须与 5.6 一致。
- current Agent Loop reference 只能在 bus reset、Terminal rebind 和 target successful start 后发布。
- target constructor/preflight failure 必须触发 Terminal fatal exit；不得继续旧 Session。
- initial/resume fatal error 都只由 CLI error boundary 输出一次。
- CLI `finally` 按以下顺序 shutdown：Management deactivate，Schedule pause/drain + close，pending/current
  Agent Loop abort/close，Dream close，最后 Model Router close；每个 owned component close 最多调用一次。
- 不引入新的 aggregate object 保存全部组件。

量化验收：

- `myclaw/agent/runtime.py` 不存在。
- `rg "RuntimeHost|PreparedRuntime|RuntimeBindings|prepare_runtime|_prepare_runtime" myclaw tests`
  返回 `0` 个匹配。
- 单次 CLI invocation 中 Message Bus、Model Router、Memory Manager、Dream、Schedule Service 创建次数各为 `1`。
- 每次 `/resume` Agent Loop 创建次数增加恰好 `1`；选择当前 Session 也一样。
- replacement 前后 Message Bus、Model Router、Memory Manager、Dream、Schedule Service object identity 全部相同。
- initial constructor/preflight failure 与 resume constructor/preflight failure 均产生 exit code `1`，安全错误输出
  恰好 `1` 次，traceback/Skill body/API key 输出次数为 `0`。
- CLI、runtime generation、active Session、resume、shutdown、Terminal integration tests 通过率 `100%`。

依赖：T5、T6。

### T8：集成、零回归与发布证据

主要范围：

- 删除已经没有引用的 test fixtures、compatibility aliases 和 stale exports。
- 更新 `docs/release-readiness.md`，使其只描述已经落地并验证的实现。
- 对 ADR/PRD/Runtime Contracts/本计划执行最终一致性检查。
- 执行全量 test、lint、type-check 和 package build。

量化验收：

```powershell
python -m pytest
python -m ruff check .
python -m mypy
python -m build
git diff --check
```

以上命令退出码全部为 `0`，并满足：

- 全量测试失败数为 `0`。
- Ruff violation 数为 `0`。
- Mypy error 数为 `0`。
- wheel/sdist build failure 数为 `0`。
- Runtime/Workspace/MemoryTaskScheduler stale symbol 的 AST structural findings 为 `0`；扫描覆盖 `myclaw` 与 `tests` 的每个 Python 文件，不整文件排除。
- Session、Summary、Cursor、Long-term Memory、Schedule、Artifact fixture 的非预期 schema diff 数为 `0`。
- release contract 固定映射 14 个 owner pytest node，并用一个 collect-only 和一个定向 execution subprocess 覆盖完整映射；不映射 release meta-test。
- `release-readiness.md` 中每个更新后的 architecture claim 都有对应 test path，架构接口事实另由 production source/AST 直接检查。

依赖：T7。

## 7. Task 依赖与合并顺序

可并行开始：

- T1：Path migration
- T2：Skill Loader/Snapshot
- T3：Memory Manager/Dream
- T6：Message Bus/Terminal presentation seam

依赖链：

```text
T0 complete

T1 ----+
T2 ----+--> T5 --+
T3 --> T4 --> T5 +--> T7 --> T8
T6 -------------+
```

每个 Task 独立提交、独立测试、独立合并。T2、T3、T6 允许最小 private compatibility adapter 维持旧 Runtime
入口，但不得形成新 public API；T7 必须删除全部 adapter。

## 8. 变更影响评估

### 8.1 上游入口

受影响：

- `myclaw.terminal.process_entry:run`
- Typer default callback
- Textual application startup/shutdown

风险：sync/async event-loop ownership、Terminal mount failure 与 CLI exit code。

控制：保留同步 Typer 边界；只在一个位置调用 `asyncio.run()`；Terminal 改用 `run_async()`；为 initial/mount/resume
failure 分别建立 integration test。

### 8.2 Foreground Agent Run

受影响：Agent Loop constructor、Message Bus ownership、Context/Summary/Tool/Runner 创建位置。

不应变化：Inbound FIFO、streaming Outbound、Task Framing、Session commit、Tool loop、title、cancellation 和
confirmation 行为。

控制：迁移构造位置，不修改 Agent Run 核心顺序；保留现有 scripted provider transcripts 和 Session snapshots。

### 8.3 Session 与 `/resume`

受影响：同 Session resume 从 no-op 改为完整 generation replacement；old abort 从 detached cancellation 改为 awaited。

风险：迟到 Outbound、旧 task 修改新 UI、切换期间 Schedule 回调使用旧 Loop、target failure 后状态悬空。

控制：固定 replacement 顺序；Schedule pause/drain；old Loop awaited abort；bus reset 在 old work 完全停止后；
current reference 单点更新；target failure 直接终止 Conversation。

### 8.4 Schedule

受影响：Store ownership、System Job ID validation、Dream dispatch、replacement pause/resume。

持久化变化：字段集合不变；有效 Terminal Conversation 启动会创建或校正 `schedule.json`，并可能增加
`job_id="dream", source="system"` 记录。

不应变化：User Job UUID4、Tool add/list/remove、User Schedule Session、at/every/cron、terminal state、legacy state。

控制：source-aware validator；public/internal Store API 分离；strict round-trip fixtures；Dream hidden-list tests。

### 8.5 Memory

受影响：Memory Manager API、Dream execution、Summary append dependency、`/dream` dispatcher。

不应变化：Summary schema、Cursor preadvance、Memory partitions、manual result、stable error code 和 memory route。

控制：把现有 Memory Task behavior tests 迁移为 Dream tests，先锁定结果与 cursor，再移动实现。

### 8.6 Skills

行为变化：manual Skill 不再每次读取磁盘；Snapshot 从 Runtime Lifetime 改为 Runtime Generation；任意 `/resume`
刷新最新磁盘状态。

不应变化：discovery order、reserved name、duplicate handling、metadata prompt、always prompt encoding、ordinary
`read_file` 权限与实时读取。

控制：single-read fake filesystem、frozen/current-generation tests、resume refresh tests、prompt byte-for-byte fixtures。

### 8.7 Workspace 与安全

受影响：大量 constructor/type annotation 与 test fixture。

风险：误把 lexical Workspace normalization 改为 filesystem resolution，导致 symlink/confirmation 行为变化。

控制：只删除 wrapper，不改变 normalization 或 containment helper；优先迁移测试，再机械替换参数。

### 8.8 Management 与 `/status`

受影响：Service 从 generation-owned 改为 lifetime-owned；Session/status dependency 改为 current-loop callable。

不应变化：命令集合、redaction、picker、memory view、Schedule health 和返回 shape。

明确变化：`uptime_seconds` 始终表示当前 Agent Loop/Session uptime，任意 `/resume` 后重置。

### 8.9 Shutdown 与副作用

正常 shutdown 必须 awaited。forced replacement 不回滚已接受的 Tool、Artifact、Memory 或 Schedule side effect；
old Session 仍不做 final save。相比当前实现，Agent Loop 与 Schedule task 会在切换前被取消并等待，减少迟到输出，
但不引入跨文件 transaction 或 side-effect rollback。

## 9. 最小变更控制

- 不顺手重构 Provider、Tool Schema、Session persistence 或 Terminal visual layout。
- 不重命名与本需求无关的 domain type。
- 不格式化无关文件。
- 只在 constructor/lifecycle 迁移需要时调整测试 fixture。
- 每个 Task 提交中列出删除、增加和临时 compatibility file；T7 检查临时代码归零。
- 发现与本计划无关的缺陷时单独记录，不在当前变更中修复。

## 10. 零回归策略

1. 先以现有 contract tests 锁定行为，再移动 ownership。
2. 每个 Task 先运行定向 tests，再运行全部受影响 package tests。
3. T7 完成后执行完整 pytest；T8 执行 pytest、Ruff、Mypy、build。
4. 对持久化文件使用 exact-key 和 strict round-trip assertion，不仅比较业务字段。
5. 对 concurrency 使用 controlled clock、barrier、event 和 fake executor，不使用真实 sleep 判断时序。
6. 对 constructor/lifecycle 使用 call counter 和 identity assertion，证明对象只创建/关闭一次。
7. 对错误输出断言 code/message、exit code 和敏感内容 absence，不锁死无关终端排版。
8. 对 `/resume` 覆盖 idle、active-run no-force、active-run force、current Session、target load failure、target preflight
   failure、Schedule User Job active、Dream Job active 八类路径。

## 11. 预计文件范围

核心实现：

- `myclaw/terminal/cli.py`
- `myclaw/terminal/conversation.py`
- `myclaw/agent/loop.py`
- `myclaw/agent/message_bus.py`
- `myclaw/agent/workspace_state.py`
- `myclaw/agent/workspace.py`（删除）
- `myclaw/agent/runtime.py`（删除）
- `myclaw/skills/catalog.py`
- `myclaw/skills/loader.py`（如采用独立 module）
- `myclaw/memory/manager.py`（新增）
- `myclaw/memory/dream.py`（新增）
- `myclaw/memory/memory_task.py`（拆分完成后删除；不保留兼容 export）
- `myclaw/memory/memory_scheduler.py`（删除）
- `myclaw/memory/conversation_summary.py`
- `myclaw/schedule/model.py`
- `myclaw/schedule/store.py`
- `myclaw/schedule/service.py`
- `myclaw/management/service.py`
- `myclaw/management/commands.py`
- `myclaw/tools/tool_gateway.py`
- Workspace-dependent file/directory/Exec Tool modules
- `myclaw/session/session.py`
- `myclaw/logging/session.py`

主要测试：

- `tests/test_cli.py`
- `tests/test_runtime.py`
- `tests/test_runtime_generation.py`
- `tests/test_runtime_active_session.py`
- `tests/test_runtime_shutdown.py`
- `tests/runtime_bus.py`
- `tests/agent/test_loop.py`
- `tests/agent/test_message_bus.py`
- `tests/agent/test_schedule_loop.py`
- `tests/agent/test_context.py`
- `tests/skills/test_catalog.py`
- `tests/memory/test_memory_task.py`（有效 cases 迁入 Dream/Manager tests 后删除）
- `tests/memory/test_memory_scheduler.py`（有效 cases 迁入 Schedule tests 后删除）
- `tests/memory/test_conversation_summary.py`
- `tests/scheduling/test_schedule_model.py`
- `tests/scheduling/test_schedule_store.py`
- `tests/scheduling/test_schedule_service.py`
- `tests/scheduling/test_schedule_service_boundary.py`
- `tests/sessions/test_session_resume.py`
- `tests/terminal/test_conversation.py`
- Workspace、Tool、Session IO security 相关 tests

## 12. 完成定义

只有同时满足以下条件，本计划才算实施完成：

1. T1–T8 全部达到各自量化验收标准。
2. 代码中不存在 Runtime Host/Prepared Runtime/Runtime Bindings 或职责等价替代物。
3. CLI 直接拥有全部 Runtime-Lifetime component lifecycle。
4. Agent Loop 直接拥有全部 Session-scoped component lifecycle。
5. 同 Session `/resume` 会创建新 Agent Loop、刷新 Skill、清空同一个 Message Bus。
6. Schedule Service lifetime 跨 generation 保持，切换时 pause/drain/rebind/resume。
7. Dream 由 `/dream` 和 Dream System Job 共同调用，且不经过 Agent Loop/Schedule Session。
8. Memory Manager 不含 model/tool/scheduling logic。
9. `Workspace`、`MemoryTaskScheduler`、代码层 Runtime 文件全部删除。
10. 全量 test、lint、type-check、build 和 `git diff --check` 全部通过。

## 13. T8 Completion Evidence

- Implementation status: T1-T8 complete after final verification.
- Verification base: clean `d60b96d1beed98b4325d2913b674be32d669adb3` plus the staged prospective documentation patch.
- `python -m pytest -q`: 1,438 passed and 10 conditionally skipped on Windows.
- `python -m ruff check .`: passed with zero violations.
- `python -m mypy`: passed with 170 source files checked.
- `python -m build --no-isolation`: passed; one source distribution and one `myclaw-0.1.0-py3-none-any.whl` were built.
- `git diff --check`: exit code `0`; the staged prospective patch also passed `git diff --cached --check`.
- The fixed mapped owner tuple collected 14 nodes and targeted execution passed all 14 nodes in one execution subprocess.
- `python -m pytest -q tests/test_release_contract.py`: 24 passed with 42 tracked Markdown files and no active stale-symbol findings.
- Claim-to-test mappings and the exact six-schema/Dream record boundary remain in `docs/release-readiness.md`.
