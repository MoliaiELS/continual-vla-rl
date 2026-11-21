# Hard 全任务原生动作预算评测

最新状态：用户要求先补完原 Easy 剩余 791 回合，并确认算法短测暂时结束后正常提交。接续作业 **20975** 已释放并在 `4090node3` 运行，先执行 Easy 第三批收尾，严格核验原计划累计 7,000 后才进入 Hard。原生预算 Hard 仍为 0/7,000，计划和输出目录保留。旧占位 `20326`、旧 Hard-only `20965` 均已取消；旧 CPU 接续控制器保持停止。当前入口及回执见 [Easy→Hard 接续说明](hard-after-ppo.md)。

2026-11-21 最新恢复：Easy 已完成，20975 的 Hard 阶段保留 311 条正式结果后交接给 **21038**。原生预算 worker 和 profile 不变，专家预筛改用独立分块进程，控制器自动监控基础设施续跑。应按 [预筛恢复与扩测口径](2026-11-21-prescreen-recovery.md) 使用固定恢复提交，以下旧接续命令仅供历史追溯。

此前曾按用户要求提前提交 Hard-only，原 Easy 791 回合暂缓；现已按最新指令恢复 Easy 优先。Easy 原生预算重评 `20730` 已于 22:20 完成全部 900 回合。下方旧控制器恢复命令保留为历史，不用于当前已替换占位作业的接续流程。

2026-11-20 用户确认：Hard 全部 50 项使用各自 RoboTwin 原生动作上限。首批每项 100 回合，共 5,000；预定十个重点任务再各扩测 200，累计 7,000。任务选择和样本数量保持原计划。

新协议为 `robotwin-native-b0-hard-native-budget-v1`。各任务上限从固定 RoboTwin 的 `task_config/_eval_step_limit.yml` 读取并与[任务目录](task-catalog.md)核对，范围 400–1,700 个有效动作锚点。最大模型查询次数为 `ceil(step_lim / 30)`；最后一个 chunk 的已发送命令和真正推进的动作分别记录。权重、观测、动作转换、指令、随机化配置和原生成功判定继续沿用固定版本。作者循环仅扩展外层查询次数。

已部署执行提交为 `bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460`，协议指纹为 `bdb780be0ade42b680b5f81f7dc5533b1cb8e6a8699697ca3ec5b7b7b507583c`。服务器真实资产和原 smoke 初态准备通过，7,000 回合计划、四份独立 pilot 清单与正式 0/7,000 汇总已核验，相关 CPU 单元检查通过。2026-11-20 16:48 接管成功，暂停前阶段为 `waiting_native_easy`；Hard GPU 短测尚未运行。

CPU 准备阶段冻结完整 7,000 回合的计划。每项任务在 GPU 作业内完成原生专家预筛后，再冻结该项的完整初态与 policy seed；未冻结的计划数明确列为未完成。专家预筛异常和候选排除不计作策略失败。原评测面板作为预筛证据来源，新协议的 manifest、逐回合结果和视频保存在独立目录，不向旧 300 动作结果追加。

执行顺序为：Easy 五任务 900 回合重评 → 原 Easy 累计 7,000 回合收尾 → Hard 双进程短测 → Hard 首批 5,000 → Hard 重点任务新增 2,000。CPU 接续控制器接管仍在运行的 Easy 作业 `20730`，使用它原来的 `15728f76ba267fef3bd75c36c7d825c4ab2a4360` 固定代码和输出；不会重启该 GPU 作业。只有两套 Easy 都通过完整性门槛后，才取消旧 300 动作 Hard 占位作业 `20326`，提交原生预算 Hard 作业。

Hard 短测使用 `move_stapler_pad`、`turn_switch` 各 5 个原 smoke 初态，分别串行和单卡双进程重放，共 20 个执行回合，均不计入正式 7,000 分母。双进程要求 10 对种子的成功判定及终止原因一致、无技术异常、吞吐提高且采样显存保留至少 1 GiB 余量。若只是吞吐、结果一致性或显存余量不满足，则正式任务使用单进程；短测执行本身出现技术错误则停止并保留非零退出码。

正式任务申请一张 RTX 4090、16 个 CPU。专家预筛串行；双进程通过短测后，每次最多评测两个不同任务，启动前再检查获配 GPU 的可用显存。两个评测进程退出后才继续专家预筛。各任务不同时运行自己的两个批次。Slurm 超时、节点故障或抢占后从固定种子和终局日志有限续跑。

服务器输出位于 `<DataRoot>/xvla/robotwin-b0/hard-native-budget/<HardCommit>/`：

- `profile.json`：50 项计划、动作／查询上限、协议指纹、模型和源码身份。
- `adoption.json`、`state.json`、`dispatch.log`：Easy 接管凭据和后续 Slurm 状态。
- `pilot/qualification.json`：独立短测、吞吐、一致性和显存证据。
- `formal/<task>/hard/`：冻结初态、逐回合尝试、运行清单及成功／失败视频。
- `reports/summary.json`、`reports/report.md`：首批、扩测及累计统计；未冻结、未运行、待重试和未解决分列。仅完整任务参与已完成子集均值，完整 50 项前不报告全任务均值。

本机查询使用实际已部署的 `<HardCommit>`。后续文档提交不改变作业代码；须显式指定原作业提交。

```powershell
$hard = 'bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460'
./tools/xvla/remote.ps1 robotwin-status
./tools/xvla/remote.ps1 robotwin-report formal -HardNativeBudget -HardNativeBudgetCommit $hard
# 仅在用户明确恢复评测、另一边短测已结束，并核对暂停记录后恢复。
./tools/xvla/remote.ps1 robotwin-run 20298 -HardNativeBudget -HardNativeBudgetCommit $hard -RobotHours 48
```

本轮 Hard 使用扩展动作预算，报告时须标注该口径。原 300 动作预算和本轮原生预算的成功率分别展示，不直接混合成一组结果。
