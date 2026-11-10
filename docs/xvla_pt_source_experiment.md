# X-VLA-Pt 源任务 SFT 与在线更新实验

本轮先建立可用于毕设主实验的共同起点。开发源任务 A 固定为 LIBERO-Spatial 任务 0。模型起点固定为 `2toINF/X-VLA-Pt@c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`，RLinf 和 X-VLA 源码沿用接入分支的固定版本。SFT 只使用 A 的示教。目标组合的示教、在线轨迹和评估结果不进入训练或 checkpoint 选择，预训练曝光无法审计的条目标记为未知。

## 参考配方与本地预算

[X-VLA 论文 §4.2.1、附录 D.3 和 H](https://arxiv.org/html/2510.10274) 报告先适配 prompt/action heads，再联合适配。附录 H 的冻结阶段为 1,000 次迭代，随后用 1,000 次迭代逐步恢复联合更新的学习率。论文给出的少示教结果采用 PEFT。固定源码的 [PEFT 入口](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/peft_train.py) 和论文的调度描述分别记录。本项目将公开配方用于选择数据量级与更新范围，保留单卡资源和独立重载验收。逐阶段记录 batch、样本呈现次数、实际优化步、参数范围、精度、学习率和 GPU 小时。

## 开发阶段与停止条件

| 阶段 | 固定设置与产出 | 根据结果推进 |
| --- | --- | --- |
| D0 控制回放 | 从固定原始 HDF5 的示教初态开始。相对控制和转换后的绝对控制各做同样的 settling，记录成功、步数及首次偏离。 | 绝对回放能成功才用于监督拟合诊断。失败保留原日志并检查控制器模式、动作转换和初态。 |
| D1 拟合诊断 | 固定示教观测和模型初始噪声。对 Pt 和已有 256／1,024 步 checkpoint 分别比较位置、旋转和夹爪，再定位接近、抓取、搬运、放置中的首个闭环偏离。 | 如果连训练示教都难拟合，检查标签与输入对齐、动作损失和参数范围。如果已拟合但闭环失败，优先增加 A 的独立示教与初态覆盖。 |
| D2 源任务 SFT | 从 Pt 训练。先扩展至约 10 条经验证的 A 示教，在相同 `prompt_heads` 范围和有界优化预算下与既有五示教结果比较。若仍无可靠闭环成功，再用同一数据尝试论文启发的 LoRA 加 heads，单独验收该参数布局和严格重载。 | 只用 A 的固定开发面板比较成功数、首次成功步、逐回合结果。现有五初态是工程对照，增加预先选定的 A 初态验证重复性。 |
| D3 在线资格 | 同一合格 Pt-SFT checkpoint。固定训练 horizon、探索噪声、奖励、控制频率和预算。 | PPO 需出现正奖励和有效 actor 更新。GRPO 需出现组内回报差异和有效 actor 更新。两者均检查权重同步、更新后 rollout 和重载。 |

首轮 D0 复用已校验的五示教清单。D2 的原生评估沿用 800 步上限和原生历史，在线资格沿用 300 步上限、实测历史与终端探索噪声。分别记录这两套评估口径。单个成功轨迹用于连通性诊断，多个独立开发初态的成功才用作选择共同起点的证据。

源任务十示教对照保留原五条示教身份，追加顺序由 seed 1234 固定。新增候选须在相对和绝对控制下均保持终局成功。筛选拒绝及其交互成本单独记录，按固定顺序递补。第一轮 1,024 步对照保持原五示教版的 FP32、batch 1、AdamW 学习率 `1e-4` 和预处理，以隔离数据变化。原始论文的 LIBERO 配置是 BF16、batch 128、60,000 步和无图像增强，固定公开训练代码的通用图像管线包含 ColorJitter。本地按公开管线保留增强，累计样本呈现次数远小于论文配置。

### 已完成的开发诊断

- Slurm 19513 完成 `demo_46` 的绝对回放验收。相对和绝对回放均在第 74 步首次成功且第 85 步仍成功，每条另有 10 步 settling，夹爪命令一致。
- Slurm 19514 在 24 个固定观测、固定噪声的示教窗口比较 Pt 和五示教 1,024 步版。首动作平均位置误差从 64.6 cm 降到 6.5 cm，旋转误差从 119.7° 降到 7.2°，夹爪开合命令不一致率从 54.2% 降到 45.8%。这些是使用示教 proprio 的离线拟合结果。
- Slurm 19515 在相对回放 `demo_17` 时终局失败，以 1:0 退出，未发布十示教清单。保留该次日志和部分转换产物，后续按固定候选顺序递补并保留严格终局门槛。
- Slurm 19517 按固定顺序递补，排除 `demo_17` 和 `demo_42`。十条示教均通过相对和绝对终局回放，共 885 个动作帧。原五条示教的 ID 与 HDF5 SHA256 逐项相同。发布清单为 `datasets/source-spatial-task0-10/manifest.json`，SHA256 为 `0a3d6d871939e3365cd8719e94f2c6946cf40a594643c579cc82f733cb245bd77`。Slurm 19518 的加载器确认 785 个有效训练窗口。
- Slurm 19518 在 3090 完成 1,024 次 prompt/action heads 更新，五个训练张量改变，冻结组未变化。同进程恢复和独立严格重载均通过，退出 0:0，Slurm 耗时 649 秒。闭环评估使用作业 19521。
- Slurm 19521 在原生 benchmark 完成 5 回合，成功 0/5，均达到 800 个策略步。与五示教版 19511 的任务、回合、环境种子、策略种子和初态哈希逐项匹配。
- Slurm 19519 的单测在旧 actor 测试 fixture 中出现两次配置缺失。补齐 fixture 后，19522 在专用环境通过全部 153 项测试，包含真实 Torch 和 PEFT 的 domain 行热启动测试。
- Slurm 19523 从 19518 的 1,024 步 checkpoint 热启动同一十示教清单。256 次联合优化后有 532 个 LoRA 张量和五个 domain 3 prompt/action-heads 张量改变，冻结组未变化。同进程恢复和独立严格重载通过，退出 0:0。Slurm 耗时 395 秒，5 秒采样的作业计算进程峰值为 5,990 MiB。闭环评估使用作业 19524。
- Slurm 19524 完成相同五个初态的原生评估，成功 0/5，均达到 800 个策略步，退出 0:0。

## 直接 PPO 与毕设主对照

原始 Pt 的直接 PPO 是可选的零本地示教对照。先按预定上限用同一 PPO 探索配置采样，记录正奖励和首次成功控制步。只有采到有效奖励才安排训练，并与 Pt-SFT→PPO 对齐在线控制步预算，示教成本另列。保留现有 Pt-SFT 在线入口的 checkpoint 来源和奖励验证。

共同起点与普通 A-only RL 通过后，正式对照包含无本轮更新、默认 reset 的 A-only RL、扩展池均匀采样、同池分层采样。三条 RL 条件共享参数范围、奖励、算法和实际策略控制步预算，额外记录 settling、post-done、有效训练步和 GRPO 组保留率。按预先冻结的目标组合报告相对共同起点的逐任务成功率变化，分列本地 SFT＋RL 留出和仅 RL 留出。 [毕设研究主线](https://app.notion.com/p/40fd26c2767d8104926bcd6c07d7f25c)

## 执行与证据

使用 `tools/xvla/remote.ps1`、独立 `xvla_rlinf` 环境及固定 commit worktree。仿真和训练经 Slurm，排除登录节点，保留获配 GPU 可见性。每个作业单卡，本用户合计最多两张 GPU。失败保留非零退出码，结果与作业日志分别保存在 X-VLA 数据根的 `runs/<job>/` 和 `logs/<job>.out/.err`。

```powershell
./tools/xvla/remote.ps1 submit convert_source10 -GpuModel 4090
./tools/xvla/remote.ps1 submit sft_source10 -SftSteps 1024 -GpuModel 3090
./tools/xvla/remote.ps1 submit benchmark_sft_source10 -TaskCount 1 -Trials 5 -SftSteps 1024 -CheckpointJobId $promptJob -GpuModel 3090
```

若 prompt/action heads 闭环仍不合格，阶段二用上一步审计通过的 1,024 步权重和同一十示教清单。继承五个已训练的 domain 3 参数张量，单独初始化 all-linear LoRA（rank 8、alpha 16）和优化器。首轮 256 步、FP32、batch 1，按固定公开 PEFT 代码的默认联合更新阶段使用恒定学习率。本地两阶段是可审计的小预算试验，论文的联合 warm-up 与计算规模另行区分。

若 256 步仍无成功，下一条训练从同一个 prompt/action-heads checkpoint 重新开始阶段二，将联合预算扩大到 1,024 步、batch 4，共 4,096 次窗口呈现。它用于检验训练预算扩大的效果，步数和 batch 的贡献无法单独归因。先复用同一开发面板确认是否出现闭环成功，再决定是否继续增加示教。

```powershell
./tools/xvla/remote.ps1 submit sft_source10_lora -SftSteps 256 -CheckpointJobId $promptJob -GpuModel 4090
./tools/xvla/remote.ps1 submit benchmark_sft_source10 -TaskCount 1 -Trials 5 -SftSteps 256 -CheckpointJobId $loraJob -GpuModel 3090
```
