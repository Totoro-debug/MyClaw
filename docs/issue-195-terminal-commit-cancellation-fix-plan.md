# Issue #195 Terminal Commit Cancellation Fix Plan

状态：已完成。修复已由 commit `7c84fcb8a05e015f8ffcbb530b7c801de19a4a70` 交付；本文保留经评审的实施方案与修复前事实基线，当前行为以 Runtime Contracts 和实现为准。

## 1. 问题与事实依据

本方案修复代码审查 finding：`ScheduleService.pause_and_drain()` 没有取消
terminal-commit task，可能令 `/resume` 长时间或无限等待。

规范依据：

- Parent Issue #188 要求 Runtime Generation replacement 暂停 Schedule dispatch，并取消、等待
  所有 active User/System Job 和 terminal-commit task。
- Child Issue #195 的 What to build 明确要求 “cancel and await every active User/System Job
  and terminal-commit task”；验收要求 `pause_and_drain()` 返回时 active Job、run task、
  unfinished terminal-commit task 均为 `0`。
- ADR-0017 将 `pause_and_drain` 固定在 destructive cutover 之前，因此该方法不返回，CLI 就不会
  进入 old Agent Loop abort、Message Bus reset、Terminal rebind 或 target start。

修复前实现事实（`7c84fcb` 之前）：

1. `ScheduleService._run_job()` 在执行完成后调用 `_commit_terminal()`。
2. `_commit_terminal()` 创建独立 `asyncio.Task`，登记到 `_terminal_commit_tasks`，并通过
   `asyncio.shield()` 使 run task 的取消不会自动取消 Store operation。
3. `_pause_owned_tasks()` 调用
   `_cancel_and_drain_job_tasks(cancel_terminal=False)`：run task 会被取消，terminal-commit task
   只会被等待。
4. 因此只要 Store operation 停在一个可等待但不自行完成的点，`pause_and_drain()` 就无法返回，
   `/resume` 的 replacement transaction 也无法继续。
5. 现有测试 `test_pause_and_drain_waits_for_a_completed_*_terminal_commit` 明确断言 commit 没有被
   取消，实际固化了与 #188/#195 相反的行为。

### 1.1 根因

根因不是 CLI 超时不足，而是 Schedule Service 的任务所有权语义不完整：它拥有
terminal-commit task，却在 pause Interface 中只等待、不取消该 task。取消 run task 也不能补救，
因为 `_commit_terminal()` 刻意 shield 了 Store operation。

### 1.2 隐含回归风险

不能只把 `cancel_terminal=False` 改成 `True` 后结束：

- `at` Job：terminal removal 被取消后，Job 必须保持 persisted pending，并在 `resume()` 后只重试
  一次。当前 `_retry_at_jobs_after_resume` / `_consumed_at_jobs` 可以承担该语义。
- `every` Job：`_commit_terminal()` 当前在 Store commit 成功前就把 `_every_deadlines` 改为
  “完成时刻 + interval”。如果 Store commit 随 pause 被取消，内存 deadline 与未更新的 Store
  anchor 会不一致；恢复时 `_sync_every_deadlines()` 可能按旧 anchor 重新计算并立即重复执行。
- `cron` Job：Cron cursor 在 reservation 时已经推进。取消 terminal commit 后必须保留该 cursor，
  只允许下一个正常 Cron occurrence 执行。

因此，“取消 terminal commit”和“取消后保持 Schedule cursor 一致”是一个不可拆开的业务原子变更。

## 2. 目标与非目标

### 2.1 目标

- `pause_and_drain()` 线性化 paused state 后，取消并等待 dispatcher、所有 run task，以及所有
  terminal-commit task。
- drain 循环必须覆盖 run task 在取消清理期间新建的 terminal-commit task。
- 返回时不存在 active reservation、unfinished run task 或 unfinished terminal-commit task。
- `at` Job 在 terminal removal 未提交时保持 pending，恢复后只执行一次。
- `every`/`cron` Job 不因 terminal commit 被取消而立即重复已经执行过的 occurrence；只执行下一个
  有效 occurrence。
- caller 取消等待 `pause_and_drain()` 时，内部 pause barrier 仍完成，再向 caller 传播
  `CancelledError`。
- 保持 Schedule Service 与 Store identity、持久化 schema、公开 Schedule Tool 行为不变。

### 2.2 非目标

- 不重开 ADR-0017 与后续实施评审已经收口的 current Agent Loop publication 顺序；本修复沿用该替换序列。
- 不修改 Dream 的领域模型或长期记忆行为。
- 不增加自动重试队列、持久化 retry 字段、schema version 或跨进程锁。
- 不给 pause 增加 wall-clock timeout。timeout 会让 Service 在 owned task 尚未退出时向 CLI 假报
  drain 完成，破坏所有权和零晚到写入保证。
- 不改变普通 Job execution、Provider、Agent Runner、Tool、Session 或 Message Bus 语义。

## 3. 技术选择

选择“Schedule Service 内部协作式取消并完整 drain”，而不是让 CLI 做补偿。

`ScheduleService` 是一个应保持 deep 的 **Module**：CLI 只需要学习
`pause_and_drain()` / `resume()` 的 **Interface**，不应知道 run task、terminal task、one-shot rearm
或 recurrence cursor。任务集合和 Store operation 继续留在该 Module 的 private **Seam** 内。

不新增 Store port、TerminalCommit adapter 或公开测试 hook。Schedule Service 仍创建并拥有唯一 Store；
受控并发测试沿现有 private Seam 替换单个 Store method，并通过公开生命周期 Interface 驱动行为。

### 3.1 Public Interface

公开类型签名保持不变：

```python
async def pause_and_drain(self) -> None: ...
def resume(self) -> None: ...
```

`pause_and_drain()` 的完整 Interface 契约收敛为：

1. 返回前阻止新 reservation。
2. 取消并等待 dispatcher、run task、terminal-commit task。
3. caller cancellation 不得取消共享的内部 pause task；内部 barrier 完成后再传播。
4. 返回时所有 generation-sensitive Schedule work 为零。
5. 不关闭或替换 Schedule Service/Store；后续可调用 `resume()`。

### 3.2 Private implementation 调整

#### A. 明确三条生命周期策略

把 `_cancel_and_drain_job_tasks()` 的 `cancel_terminal` keyword 改为必填，三个 call site 显式选择：

| 调用场景 | `cancel_terminal` | 理由 |
| --- | ---: | --- |
| `pause_and_drain()` | `True` | #188/#195 明确要求 replacement pause 取消 terminal commit |
| `abort_and_wait()` | `True` | abort 不允许 owned task 留存 |
| direct `close()` cleanup | `False` | 保留现有 direct-close 等待已开始 terminal persistence 的语义 |

CLI 正常 shutdown 当前先调用 `pause_and_drain()` 再 `close()`，因此 CLI shutdown 也会使用 pause 的取消
语义；未提交的 `at` Job 会留在 Store，供下次启动执行。这是改变 pause Interface 后的明确影响，需由
回归测试和运行时契约记录，不能隐式忽略。

#### B. 取消并 drain 动态 terminal task

`_pause_owned_tasks()` 继续按以下顺序执行：

```text
reservation gate: publish paused
  -> cancel/await dispatcher
  -> snapshot active one-shot reservations
  -> cancel run tasks and currently registered terminal tasks
  -> await snapshot
  -> repeat until both owned task sets are empty
  -> re-arm uncommitted one-shot Jobs
  -> clear active reservations
  -> return
```

保留 `while self._run_tasks or self._terminal_commit_tasks`。这是必要的：一个 run task 可能在第一次
snapshot 之后进入 `finally` 并创建新的 terminal-commit task；下一轮必须发现、取消并等待它。

#### C. 让 `every` deadline 与 terminal commit 成功线性化

`_commit_terminal()` 可先计算候选 `_EveryDeadline`，但不得在 Store operation 完成前写入
`self._every_deadlines`：

- operation 正常完成：发布候选 deadline，保持“完成时刻 + interval”的既有成功语义。
- run task 被取消，但 Store operation 已在取消前正常完成：仍发布候选 deadline，然后传播 caller
  cancellation；持久化事实优先。
- Store operation 被 pause 取消：不发布候选 deadline，保留 reservation 时已经推进的 deadline。
- Store operation 真实失败：维持现有 fault latch 和安全日志；不把失败误报为正常取消。

Store `_publish()` 在 Condition lock 内执行 canonical serialization 和 atomic replace，期间没有 async
yield；因此可观察结果保持二态：取消在线性化点前生效则 Store 不变，Store 已线性化则 commit 完整可见，
不引入部分 Schedule record。

### 3.3 Job 类型恢复矩阵

| pause 观察到的状态 | Store 结果 | pause 后内存状态 | `resume()` 行为 |
| --- | --- | --- | --- |
| `at` execution 尚未完成 | 无 terminal write | 移除 consumed marker，Job 保持 persisted | 立即重试且只执行一次 |
| `at` terminal removal 被取消 | Job 仍存在 | 加入 rearm 集合并移除 consumed marker | 立即重试且成功后删除 |
| `at` removal 已在线性化 | Job 已删除 | 即使 run 收到取消也不存在可调度 Job | 不重放 |
| `every` terminal commit 被取消 | terminal state 未更新 | 保留 reservation 已推进的 deadline | 到下一个 interval 执行一次，不立即重复 |
| `every` commit 已在线性化 | terminal state 已更新 | 发布完成时刻对应的新 deadline | 按既有完成时刻 cadence 执行 |
| `cron` terminal commit 被取消 | terminal state 未更新 | 保留 reservation 已推进的 Cron cursor | 只在下一个 Cron occurrence 执行 |

## 4. 数据流与上下游影响

### 4.1 Runtime Generation replacement

```text
Management /resume
  -> CLI replace_agent_loop()
  -> Terminal quiesce
  -> ScheduleService.pause_and_drain()
       -> no new reservation
       -> cancel/await User Job, Dream Job and terminal commit
       -> restore resumable scheduling cursors
  -> old Agent Loop abort/drain
  -> Message Bus reset
  -> target rebind/start/publish
  -> ScheduleService.resume()
```

修复集中在 Schedule Service；CLI replacement ordering、Management Interface 和 Terminal presentation
均不修改。

### 4.2 影响评估

| 模块/状态 | 影响结论 |
| --- | --- |
| CLI | 代码无需修改；`await pause_and_drain()` 不再依赖 terminal Store operation 自行释放 |
| Schedule Service | 唯一生产修改点；pause 取消语义与 recurrence deadline publication 收敛于此 |
| Schedule Store | Interface、schema、serialization 与 atomic replace 不变 |
| User Schedule Job | active run/commit 在 replacement 时取消；`at` 可能在新 generation 重试 |
| Dream Schedule Job | active Dream run/commit 在 replacement 时取消；Summary Cursor 已接受的副作用不回滚 |
| Agent Loop | 仍只在 Schedule drain 完成后 abort；不会新增旧 generation late work |
| Management `/status` | pause 返回后 active Job count 必须为 `0`；字段集合不变 |
| 正常 CLI shutdown | 因调用 pause 后 close，未提交 terminal commit 会被取消；pending `at` Job 可在下次启动重试 |
| direct `ScheduleService.close()` | 保持等待已开始 terminal commit 的现有行为 |
| 安全/日志 | intentional cancellation 不写 error terminal、不 latch fault、不新增原始异常或路径输出 |
| 持久化 | 字段集合和 canonical encoding 零变化；不做 rollback |

## 5. 原子 Task

### T0：评审并冻结修复契约

范围：仅本文档。

独立交付：

- 评审确认 pause 必须取消 terminal commit。
- 评审确认 `at` 重试与 `every`/`cron` 下一正常 occurrence 的语义。
- 评审确认 CLI shutdown 复用 pause 时的取消影响可接受。

量化验收：

- 生产代码改动文件数 `0`。
- 未决行为语义数 `0` 后，方可进入 T1。
- 文档覆盖技术选型、架构影响、数据流、Interface、测试矩阵和回滚策略各 `1` 节以上。

独立合并条件：文档评审通过；不依赖 T1 代码。

### T1：实现 Schedule pause 原子取消与状态一致性

范围：

- `myclaw/schedule/service.py`
- `tests/scheduling/test_schedule_dream.py`

一个 Task 同时包含 terminal cancellation 和 `every` deadline publication；拆开会产生“pause 已取消
commit、但恢复后重复 recurring occurrence”的不可接受中间版本。

实现内容：

- private drain policy 在 pause/abort/close 三个 call site 显式化。
- pause 取消并等待当前及动态新建的 terminal-commit task。
- `every` 的完成后 deadline 只在 Store operation 正常完成时发布。
- 替换当前“等待 blocked terminal commit”的相反测试，保留 direct-close 等待行为测试。

量化验收：

- blocked terminal operation 收到 `CancelledError` 次数恰为 `1`。
- 测试不设置 release event 时，`pause_and_drain()` 仍完成。
- 返回时 `_run_tasks`、`_terminal_commit_tasks`、`_active_job_ids` 大小均为 `0`。
- paused state 发布后新增 reservation 数为 `0`。
- 未提交 `at` Job 在 `resume()` 后 callback 增量恰为 `1`，最终 Store 中该 Job 数为 `0`。
- `every` Job 在恢复后 interval 前 callback 增量为 `0`，下一个 interval 时增量恰为 `1`。
- `cron` Job 在恢复后下一个 Cron occurrence 前 callback 增量为 `0`，到点增量恰为 `1`。
- 已提交 terminal operation 的 persisted state 与基线完全一致。
- intentional cancellation 导致 Schedule fault 数为 `0`、error terminal write 数为 `0`。
- caller cancellation case 中内部 task 集合先归零，随后 caller 收到 `CancelledError` 恰 `1` 次。
- 所有新增并发测试依赖 real sleep 的数量为 `0`，只使用 Event/barrier 和受控 clock。
- Schedule focused tests 通过率 `100%`。

独立合并条件：一个提交内同时包含实现和上述 focused contracts；不修改 CLI、Store schema 或无关格式。

### T2：固化跨模块回归与发布证据

范围：只增加或更新契约/发布证据；若 T1 的 Schedule public Interface 已充分覆盖，不新增生产 Seam。

候选文件：

- `tests/test_cli.py` 或既有 replacement contract test（仅在能使用真实 Schedule public Interface 时增加）
- `docs/myclaw-runtime-contracts.md`
- `docs/release-readiness.md`

原则：不要用一个只会记录调用的 FakeScheduleService 宣称证明了 terminal cancellation；CLI 轴只需证明它在
Schedule barrier 返回前不进入 old-loop abort，并在返回后继续 replacement。terminal task 的取消与状态
恢复由 T1 在 Schedule Interface 上证明。

量化验收：

- replacement event trace 中 `schedule drain completed` 早于 `old loop abort` 的违例数为 `0`。
- Schedule Service/Store identity across pause/resume 变化次数为 `0`。
- Runtime Generation、CLI replacement、Schedule、Dream 相关测试通过率 `100%`。
- 完整 `pytest` 失败数为 `0`；平台能力 skip 必须逐项可解释。
- `ruff check .`、strict `mypy myclaw tests`、`git diff --check` 返回码均为 `0`。
- sdist/wheel build 返回码为 `0`，Schedule persisted field-set 变化数为 `0`。
- release-readiness 只记录本次真实执行得到的命令、节点数和结果，不沿用旧计数。

独立合并条件：T1 已合入且全量验证通过；本 Task 不混入新的产品行为。

## 6. 测试设计

受控测试 adapter 需要区分三个事件：`started`、`cancelled`、`release`。关键断言是 pause 触发
`cancelled` 并在 `release` 未触发时返回；不能用短 timeout 或真实 sleep 代替因果 barrier。

建议替换/增加以下 contract：

1. `test_pause_and_drain_cancels_one_shot_terminal_commit_and_retries_once`
2. `test_pause_and_drain_cancels_recurring_terminal_commit_without_immediate_replay`
3. `test_pause_and_drain_catches_terminal_commit_created_during_run_cancellation`
4. `test_cancelled_pause_waiter_observes_terminal_drain_before_cancellation_propagates`
5. `test_direct_close_waits_for_terminal_commit_without_cancelling_it`
6. `test_completed_terminal_commit_wins_the_pause_race_without_replay`

现有 active User Job、active Dream Job、concurrent pause waiter、reservation gate、Store identity 与 next
occurrence tests 保留，作为零回归集合。

## 7. 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| run 取消后在 `finally` 新建 terminal task | drain 使用 while 固定点循环，直到两个 owned set 同时为空 |
| `at` 外部动作已发生但 removal 被取消，恢复后重复外部动作 | 这是 #195 接受的 one-shot retry 语义；测试明确记录，文档不伪装为 rollback |
| `every` commit 取消导致立即重复 | deadline 只在 Store operation 正常完成后发布；取消时保留 reservation cursor |
| Store commit 与 cancel 同时发生 | 以 Store atomic publication 为线性化点；已提交则保留，未提交则按取消恢复 |
| caller 取消破坏共享 pause barrier | 继续使用 `await_task_preserving_cancellation()` |
| 正常 shutdown 行为被 pause 改变 | 增加 shutdown/next-start contract 并更新 runtime contracts |
| 为测试暴露新 Interface | 禁止新增公开 hook/port；使用现有 private test Seam 与公开生命周期驱动 |
| 协程吞掉 `CancelledError` | 生产 Store operation 不吞取消；不以 timeout 脱离 owned task。第三方/未来 adapter 必须遵守取消契约 |
| 已进入同步 filesystem atomic replace 后操作系统阻塞 | asyncio cancellation 无法抢占不 yield 的同步 I/O；本修复保证 cooperative async wait 可取消，不承诺 hard real-time deadline。若需解决 OS-level hang，应另立 I/O execution/timeout 方案，不扩入本 finding |

## 8. 回滚策略

T1 只修改 Schedule Service implementation 与 focused tests，不修改数据 schema，因此可以通过反向提交完整
回滚。若回滚，必须同时回滚 terminal cancellation 与 deferred `every` deadline publication，不能只恢复
其中一半。已存在的 Schedule 文件无需迁移或恢复。

T2 为契约与证据更新，可独立反向提交；不得通过把失败测试改回“commit 不应取消”来规避产品回滚。

## 9. 完成定义

只有同时满足以下条件才可关闭 finding：

- T0 方案已评审确认。
- T1 所有量化验收通过，且 `/resume` 不再等待一个 cooperative blocked terminal operation 自行释放。
- T2 全量验证与发布证据通过。
- 改动范围仅包含实现当前 finding 必需的 Schedule implementation、tests 和契约文档。
- Issue #188/#195、ADR-0017、runtime contracts 与实际 `pause_and_drain()` 行为之间不存在未记录冲突。

## 10. 完成证据

- T0–T2 已随 `7c84fcb` 完成，GitHub Issue #195 已关闭。
- `python -m pytest tests/scheduling -q`：`186 passed`。
- Runtime Generation、CLI replacement、Dream 与 Schedule 定向验证：`254 passed, 1 skipped`；skip 来自 Windows Python 缺少 `termios/pty` harness。
- `python -m pytest -q`：`1412 passed, 10 skipped`，失败数为 `0`。
- Ruff、Mypy、sdist/wheel build 与 `git diff --check` 均通过；Schedule persisted field set 变化数为 `0`。
