# 五任务原生动作预算重评

2026-11-19 用户确认按扩展动作预算重评此前核查的五项任务，并记录新的正式成功率。此前 11 个诊断回放的结果见[预算与接触诊断](2026-11-19-budget-audit.md)。本次对完整冻结初态清单重新运行。

| 任务 | 原生有效动作锚点上限 | 最大查询次数 | 首批回合 | 原定扩测后累计回合 |
| --- | ---: | ---: | ---: | ---: |
| `open_microwave` | 1,500 | 50 | 100 | 300 |
| `stack_bowls_three` | 1,200 | 40 | 100 | 100 |
| `blocks_ranking_rgb` | 1,200 | 40 | 100 | 100 |
| `blocks_ranking_size` | 1,200 | 40 | 100 | 100 |
| `click_alarmclock` | 400 | 14 | 100 | 300 |

Easy 首批共 500 回合，微波炉和闹钟按原定重点任务计划再各加两批 100 回合，合计 900。先完成五项首批，再进入扩测。首批、各扩测批次和累计结果分别报告。

冻结模型及其 revision、三路 RGB、EEF 动作转换、原生成功判定、场景和 policy seed 均沿用原始评测。作者循环仅将外层查询次数扩展至覆盖原生 `step_lim`。闹钟最后一个 chunk 会发送最多 420 条命令，有效推进仍止于 400 个动作锚点；日志分别统计发送数、有效动作和物理 tick。

评测读取 `9baa17360abe242cb36e8f9c78f0aadbe64b3bee` 的五项原始 seed manifest，冻结新协议 `robotwin-native-b0-native-budget-v1`。新协议指纹为 `220a455a49b823a98c7794e2cc7e3c214848697275a7b7262d0c91a288fe1779`，复制全部 900 个已确定初态及对应 policy seed，并建立新的 episode ID。新旧结果按各自预算报告，只有对应计划队列均完成后才计算成功率差值。

有效执行提交为 `15728f76ba267fef3bd75c36c7d825c4ab2a4360`。作业 `20730` 申请单张 RTX 4090 和 16 个 CPU，按获配 UUID 上的可用显存启动最多两个独立任务进程。每个任务的批次顺序执行。写入逐回合日志，每类结果保存前两段视频，支持按既有终局记录断点续跑。

服务器输出根为 `<DataRoot>/xvla/robotwin-b0/native-budget/15728f76ba267fef3bd75c36c7d825c4ab2a4360/`。`profile.json` 与各任务 `seed_manifest.json` 定义初态和预算，`reports/summary.json`、`reports/report.md` 报告完成数、成功率、Wilson 区间、技术异常及原 300 锚点结果。每完成批次或约 120 秒更新汇总，排队期间显示 0/900 且成功率为待完成。

原 Easy 续跑 `20728` 在等待期间获得 GPU，切换到本次重评时已保留累计 6,209 个旧预算正式回合。旧派发暂缓，五任务 900 回合重评通过核验后，控制器会接续原 Easy 剩余回合，再恢复原派发。Hard `20326` 继续等待 Easy 完成。队列交接记录位于 `<DataRoot>/xvla/robotwin-b0/evidence/native-budget-handoff-20261119.json`。

2026-11-20 用户确认 [Hard 全部 50 项采用原生动作预算](hard-native-budget.md)。新的 CPU 接续控制器继续使用本轮 Easy 的固定执行代码与数据，但负责在两套 Easy 完成后替换旧 Hard 占位作业；接管后不再启动会释放旧 300 动作 Hard 的普通派发器。当前流程恢复应使用 Hard 文档的入口。

首次启动 `20729` 在执行前的 Git 路径检查失败，原因是 Slurm 复制了直接提交的批处理脚本。已改为从固定工作树用 `bash` 启动，失败作业和 0/900 初始清单保留，原 profile 标注由当前 profile 替代。

在本机查询实时汇总。作业会自动保存 `reports` 目录中的文件。

```powershell
./tools/xvla/remote.ps1 robotwin-status
./tools/xvla/remote.ps1 robotwin-report formal -NativeBudget -NativeBudgetCommit 15728f76ba267fef3bd75c36c7d825c4ab2a4360
```

已准备的 profile 可用同一固定执行提交恢复控制器。

```powershell
./tools/xvla/remote.ps1 robotwin-run 20298 -NativeBudget -NativeBudgetCommit 15728f76ba267fef3bd75c36c7d825c4ab2a4360 -RobotHours 48
```
