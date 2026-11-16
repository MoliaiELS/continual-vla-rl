# FYP 当前进度

后续模型开发在 [RLinf-VLA 的 `codex/xvla` 分支](https://github.com/MoliaiELS/RLinf-VLA/tree/codex/xvla)。本仓库的登记继续跟踪已经启动、使用旧固定 worktree 的 RoboTwin 评测。

当前工作：X‑VLA 在 RoboTwin 2.0 Easy／Hard 的冻结基座评测。按 [协议](docs/robotwin/README.md)推进；[执行登记](docs/robotwin/status.md)保存最新证据、作业 ID 和恢复入口。已有 X-VLA-Pt 与 OpenVLA-OFT 旧实验继续保留封存状态。

旧实验归档时，北京时间 2026-11-16 19:56 核实 Slurm 运行和排队作业为 0。

| 范围 | 归档状态 |
| --- | --- |
| E3 正式训练 | 完成 9/12 组，其中 SFT-cont 完成 3/3，PPO 完成 6/9。 |
| PPO seed 303 | B1 作业 `20137` 在首次优化前退出，Slurm 状态为 `FAILED 1:0`。B2 和 M 未提交。 |
| 独立源任务 A 和迁移面板 H | 未运行。 |
| E4 顺序训练、E5 原始 Pt 直接 PPO | 未运行。 |

已完成结果和失败证据见[实验归档](docs/archive/2026-11-16-experiment-closeout.md)，文档与产物位置见[目录索引](docs/README.md)。

## 只读查询

```powershell
./tools/xvla/remote.ps1 status
./tools/xvla/remote.ps1 e3-status
./tools/xvla/remote.ps1 logs 20137
```
