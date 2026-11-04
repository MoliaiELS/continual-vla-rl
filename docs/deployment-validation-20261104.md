# 单任务 GRPO 保存与重载验收（2026-11-04）

已通过 SFT rollout、真实 GRPO 更新、adapter 保存和独立重载。运行于 4090node2，使用作者 Spatial SFT snapshot `39e5240e879c80b6cda6b3a83763dad717f5c05d`。

| 阶段 | Slurm 作业 | 退出状态 | 耗时 |
| --- | --- | --- | --- |
| SFT 两条完整轨迹 | 18977 | COMPLETED / 0:0 | 7 分 40 秒 |
| GRPO 一个外层周期 | 19011 | COMPLETED / 0:0 | 11 分 4 秒 |
| 独立 adapter 重载与两条轨迹 | 19012 | COMPLETED / 0:0 | 4 分 41 秒 |

训练与重载使用 commit `9c8ac1e2c57976c5d8c5fd3402ab58d2701b3a93`。SFT 使用此前的 `217ac5ef1705d60dd04cac19e37323102f6bc653`。

## 真实更新证据

- Spatial task 0，组内 8 条轨迹均为 init state 18，policy seed 1234，rank-32 LoRA，H512、chunk8、temperature1.6、micro1/global64。
- 组内有效回报为 `[5,5,5,5,0,5,0,0]`，1/1 组通过原奖励过滤器。
- 实际策略控制步 4,096，settling 160，首次成功后的执行步 1,928，有效训练步 2,168。settling 包含 worker 初始化和 rollout 开始时各一次 reset。
- 完成 8 次 optimizer.step，梯度范数范围 3.109375–4.375，全部有限且非零。actor 的 Torch 峰值分配为 16,797,029,376 bytes。
- 首次优化批次的 2,135 个有效 action token，新旧 logprob 的平均和最大绝对差均为 0。
- 保存 878 个 adapter 张量，其中 426 个 LoRA B 张量由零初始化变为非零。该变化结合实际 RL 梯度证明了有效更新。

checkpoint 位于 `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/runs/train_19011/checkpoints/global_step_1/actor/`。包含 `adapter_model.bin`、`adapter_config.json`、单 rank AdamW 格式的 `optim.pt`。当前加载入口恢复 adapter，完整优化器、RNG 和交互预算恢复属于后续实现。

## 独立重载

新 Slurm 作业重新加载同一 SFT base 和保存的 adapter，按实际推理 dtype 逐项核对 878 个张量，全部匹配。随后在 task 0 的初态 42、31 上各执行 512 步，两个 episode 均曾成功，末帧成功为 1/2。SFT 连通性检查同样为 ever 2/2、final 1/2。首尾视频帧已检查。

同模型、同观察、同 token 的对照显示，batch 2 的采样路径与 actor forward 路径逐项相等，改为 batch 1 后平均绝对 logprob 差为 0.0598358，最大为 0.635746。最终训练将采样与更新的前向批大小统一为 1，并获得上述更新前零差异结果。后续协议须固定前向批大小。

这些是单任务部署短测，后续性能比较需预定轨迹数与统一协议。两任务 Seq/Joint 将从固定 SFT 起点重新计数，并控制实际策略控制步及任务分配。

## 部署修复与资源选择

训练申请同节点 2 张 4090 和 8 CPU，actor 占一张，rollout 与 EGL 占另一张。每组仍有 8 条轨迹，采样逐条前向后按原序合组。冻结 base 由两端分别从同一 snapshot 加载，每次同步全部可训练参数。

单 rank actor 使用直接模型与 AdamW 状态，沿用原 GRPO 损失、奖励、LoRA 和梯度累积。该执行路径由 `FYP_SPLIT_TRAIN=1` 显式启用。LIBERO 动作转换在 CPU 完成，随后由仿真消费。

失败作业及其非零退出码已保留。19006 为单卡模型共存显存不足，19007 为环境动作 CUDA 拷贝失败，19008 为 batch 8 推理显存不足，19009 为 FSDP 参数形状写回错误，19010 完成梯度更新后在 FSDP 优化器状态导出时报错。失败运行的交互属于部署成本。

## 证据与入口

远端结果根目录为 `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/runs/`，对应 `eval_18977/`、`train_19011/`、`reload_19012/`。Slurm 原始日志位于同一数据根目录的 `logs/`。本机小型证据副本为 `.local/experiment-20261104/summary.json`，视频为同目录的 `sft-18977.mp4` 和 `reload-19012.mp4`。

```powershell
./tools/remote.ps1 status 19011
./tools/remote.ps1 status 19012
./tools/remote.ps1 logs 19011
# 需要再次验证保存的 adapter 时
./tools/remote.ps1 sync
./tools/remote.ps1 reload-smoke 19011
```
