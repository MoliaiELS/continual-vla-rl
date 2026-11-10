# 持续强化后训练启动记录

当前 X-VLA 工作见 [X-VLA 部署与实验进度](#x-vla-部署与实验进度)。下文保留 11 月 4 日 OpenVLA-OFT 基线的启动记录。

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

- 新版资格训练 19034 已完成首个采集—更新周期：task 0、初态 9，真实控制步 4,096，8 条轨迹中 ever 为 7/8、final 为 2/8；post-done 2,986、有效训练步 1,110。组内回报有差异，更新前 logprob 均值/最大绝对差均为 0，完成 8 次有限非零梯度优化，梯度范数 3.421875–4.34375，Torch 峰值 16,797,029,376 字节。`experiment.jsonl` 已记录完成并开始第二周期。资格运行整体及正式对照仍在进行，这些短测数字不作为性能结论。
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

## X-VLA 部署与实验进度

更新于北京时间 2026-11-10 08:56。此时 Slurm 中没有本项目正在运行或排队的作业。**X-VLA 已部署，PPO 和 GRPO 已各完成一次真实在线更新验收。Pt 少示教路径已接到在线采样，当前卡在成功奖励。**

### 主要目标

- 将固定版本 X-VLA 接入 RLinf，统一使用原生推理、示教转换、SFT、rollout、actor 更新、权重同步及 checkpoint 恢复流程。
- 建立可复用的后训练入口，让 SFT、PPO 和 GRPO 可以从同一份权重起跑，并在固定任务和初态上比较。后续研究可扩展到不同参数更新范围、任务提升、技能保持和 OOD 泛化。
- 保留两条独立的研究起点。官方 `X-VLA-Libero` 用作已适配 LIBERO 的参考和在线算法验收起点。`X-VLA-Pt` 用作本项目少示教 SFT 和数据效率对照的起点。

### 已部署的能力

| 能力 | 当前验收 |
| --- | --- |
| 原生推理与环境 | 固定输入和噪声下，processor、raw action 与最终环境动作对齐（`19344`）。真实 LIBERO rollout 已通过（`19336`）。 |
| SFT | 通过真实更新、冻结组检查、原生恢复和独立进程严格重载。已实测 `prompt_heads` 的 domain 3 参数。 |
| PPO | 从官方 `X-VLA-Libero` 起点完成真实 reward、GAE、actor/critic 更新、同步和新 rollout（`19339`）。 |
| GRPO | 从同一官方起点，在压力探索配置下采到有方差的真实回报组，完成 actor 更新、同步、新 rollout 和严格重载（`19408`）。 |
| SAC / IQL | 尚未接入 X-VLA。入口对这些配置会明确报错。 |

代码保存在 `codex/xvla-integration`。固定 RLinf 运行时为 `third_party/rlinf@1bf78980d9339cbc67497eddceb9a1e3a74e2acc`，X-VLA 参考源码为 `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`。固定权重为 `2toINF/X-VLA-Libero@129e71460678b7236cee6fc9707f09d9fa0c3590` 和 `2toINF/X-VLA-Pt@c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`。独立 Conda 环境为 `xvla_rlinf`。

### 已完成的实验

全任务配对评估覆盖 LIBERO-Spatial 的 10 项任务，每项 5 个相同初态。各权重的初态和种子已逐项核对。

| 权重 | 作业 | 成功率 |
| --- | --- | ---: |
| 官方 `X-VLA-Libero` | `19443` | **46/50（92%）** |
| 未微调 `X-VLA-Pt` | `19444` | **0/50** |
| `X-VLA-Pt`，1 条示教、1 次 SFT 更新 | SFT `19407`，评估 `19469` | **0/50** |

随后以同一 Pt 起点做了任务 0 的五示教实验。转换作业 `19505` 使用 seed 1234，得到五条独立成功示教、421 个动作帧及 371 个有效训练窗口。数据清单位于 `datasets/fewshot-spatial-task0-5/manifest.json`。

| 实验 | 作业 | 实测结果 |
| --- | --- | --- |
| 五示教，256 步 SFT | `19506` / `19507` | 冻结检查和严格重载通过。任务 0 的 5 个固定初态成功 **0/5**。 |
| 256 步 SFT → PPO | `19508` | actor 和 rollout 加载的策略核心哈希一致，首次同步通过。8 个在线回合奖励均为 0，未进行优化更新。 |
| 256 步 SFT → GRPO | `19509` | 从同一 SFT 权重加载并同步。两组各 4 个回合奖励均为 0，组内方差为 0，未进行优化更新。 |
| 五示教，1,024 步 SFT | `19510` / `19511` | 保持数据、学习率和参数范围一致，严格重载通过。与 256 步版配对的初态仍是 **0/5**。最后 32 步平均训练损失由 1.56 降到 1.13。该权重尚未运行在线 RL。 |

PPO 和 GRPO 的有效更新证据来自官方 Libero 权重小测。Pt 少示教权重的实测证据覆盖 SFT、重载、同步和真实采样，后续仍需取得有效奖励并完成更新。多卡、长期训练、`full`／LoRA 实训和 OOD 研究结果仍待验收。

### 新 agent 接手

1. 先读 [AGENTS.md](../AGENTS.md) 和 [CODEX_HANDOFF.md](../CODEX_HANDOFF.md)，再查 [接入审计](xvla_integration_audit.md)、[操作手册](xvla_runbook.md) 和 [RL 采样器](xvla_rl_sampler.md)。
2. 优先用一条已知成功示教验证转换后的绝对动作能否复现成功。随后测固定观测和噪声下的位置、旋转和夹爪误差，区分目标转换问题、拟合不足和闭环漂移。现有转换验收验证了相对动作回放以及绝对目标记录。
3. 找到能稳定产生成功轨迹的 Pt-SFT 起点后，用同一个通过严格重载的 checkpoint 启动 PPO 和 GRPO。PPO 需要实测正奖励，GRPO 需要实测组内奖励方差。比较算法时固定任务、初态、采样和交互预算。

服务器结果根目录为 `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/xvla`，原始作业日志在 `logs/<job>.out/.err`，逐回合结果、权重与审计记录在 `runs/<job>/`。SFT 权重入口分别为 `runs/19506/sft_fewshot-smoke/checkpoints/global_step_256/actor` 和 `runs/19510/sft_fewshot-smoke/checkpoints/global_step_1024/actor`。使用 `tools/xvla/remote.ps1`、固定 Git worktree 和 Slurm，遵守两张卡的合计上限。提交、配置和查询命令见操作手册。
