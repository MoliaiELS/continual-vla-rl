# 持续强化后训练启动记录

代码运行版本为 `5b08e8c4c16264eb8943451599a54356e0a3e761`，已推送到 `fyp/deploy-base`。完整运行参数见 [执行协议](experiment-execution.md) 与 [十任务协议](long-sequence-execution.md)。服务器端 43 项测试全部通过，包含真实 Torch reentrant checkpoint 的梯度累积和随机状态一致性测试；绘图数据检查及实际 Matplotlib 渲染通过。

## 当前有效作业链

| 顺序 | 服务器 suite | 训练范围 |
| --- | --- | --- |
| 1 | `20261104-qualification-v3` | 两任务，四方法，每任务两周期，种子 1234 |
| 2 | `20261104-core-v2` | 五任务，四方法，三个种子，每任务十周期 |
| 3 | `20261104-stress10-v1` | 十任务，Seq/Seq+ER，三个种子，每任务十周期 |

三个 plan 均已生成并冻结，使用同一代码与模型 snapshot。资格验收成功后自动启动五任务；五任务全部成功后启动十任务。每个当前作业最多提交一个后继，始终遵守实际 Slurm 提交数量配额及至多两张 GPU 的资源限制。任意失败都会阻断后继，保留原退出状态与日志。

主对照与长序列共 18 次正式训练、4,915,200 个名义策略控制步。评估、settling/reset、资格验证和中断排查分别记账。十任务训练作业的 walltime 为 24 小时，五任务为 12 小时；当前未实现完整断点恢复，超时或失败不能伪装成完整结果。

根目录为 `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/experiments/`。每个 suite 内有 `plan.json`、`submission.json`、逐任务输出、Slurm 日志与最终 analysis；代码位于 homes 中的固定 commit worktree。本机关机不影响已提交作业和服务器上的逐级衔接。

## 已核实的证据

- 原始回放文件固定 revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，首批 Spatial 0–4 文件大小与 LFS SHA256 全部通过。十任务准备阶段只补齐缺失的固定版本文件。
- OOD 仿真作业 19026 与 19028 完成；实际 camera pose、真实渲染差异及物理状态一致性均通过。Object 0–4 的资产、40–49 初态和 7 维动作接口可用；这尚不证明策略在 Object 上有成功率。
- 作业 19029 完成两个任务各 4 条 SFT 轨迹，总策略控制步 4,096、settling 160。Task 0 的 ever 为 2/4、final 为 1/4；task 1 为 0/4。它们用于工程检查，样本量不足以据此选择研究任务或判断学习能力为零。
- 上述数据与图像的小型副本位于本机 `.local/experiments/20261104-qualification-v1/` 和 `...-v2/`；缺失的训练结果保持为空，不以零成功补齐。

## 保留的排查历史

初版批量提交在两个作业后被集群 `AssocMaxSubmitJobLimit` 拒绝，已改为单后继滚动提交；成功的数据与仿真作业保留在原 suite。

训练作业 19030 在更新开始前被主动取消：代码审查发现 `autograd.grad` 诊断与模型管理器启用的 reentrant gradient checkpoint 不兼容。现在诊断复用一次真实 `.backward()`，保留此前累积梯度，不重复反向；CPU Torch 测试已验证梯度和 RNG 一致。

19030 已持久化 3,392 个策略控制步、160 个 settling 步及 2,370 个 post-done 执行步，未形成完整 episode 或更新。计数按动作段落盘，因此突发中断时末端仍可能有未落盘调用；3,392 是已核实的下界。Slurm 主作业为 CANCELLED，batch 状态含信号 15，不能将主行 `0:0` 解释为训练成功。该成本不并入正式训练预算，原日志保留。

旧 `20261104-core-v1` 尚未启动，其资格前驱已经停止；后续以本页列出的 v3/v2/stress10 链为准。

## 查询与结果导出

```powershell
./tools/remote.ps1 experiment-status 20261104-qualification-v3
./tools/remote.ps1 experiment-logs 20261104-qualification-v3
./tools/remote.ps1 experiment-status 20261104-core-v2
./tools/remote.ps1 experiment-status 20261104-stress10-v1
# 可在运行过程中汇总部分证据；不完整指标会保持缺失
./tools/remote.ps1 experiment-analyze 20261104-qualification-v3
./tools/remote.ps1 experiment-fetch 20261104-qualification-v3
```

最终输出逐任务 ever/final 矩阵、有效组与预算记录、终局 NBT、FWT、固定面板曲线和分层 OOD 基座差值。十任务的稀疏评估不支持完整时间平均 NBT/AUC，将明确保留该限制。
