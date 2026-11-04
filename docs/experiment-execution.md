# 持续强化后训练运行协议

冻结日期：2026-11-04。实现对应 [研究实验方案](research-protocol.md)。先执行两任务资格验证，通过后运行五任务完整对照；10/20/30 任务的共同起点与数据接口另经预检查后冻结。

| 项目 | 资格验证 | 五任务主对照 |
| --- | --- | --- |
| 训练任务 | Spatial 0、1 | Spatial 0–4 |
| 方法 | Seq、Joint、Seq+ER、Joint+ER | 同左 |
| 训练种子 | 1234 | 1234、2345、3456 |
| 每任务采集周期 | 1 | 10 |
| 每组与时长 | K8、H512、chunk8 | 同左 |
| 每运行策略控制预算 | 8,192 | 204,800 |
| 单次每任务评估轨迹数 | 4 | 20 |
| 训练初态 | 0–39，按任务独立固定排列 | 同左 |
| 评估初态 | 40–43 | 40–49，遍历两次 |

每个方法每次从相同 SFT snapshot 与新 LoRA 起步。Seq 在同一进程切换任务并保留优化器，Joint 按组轮转。阶段末按实际采集预算保存 checkpoint；默认不逐周期反复保存。评估所有阶段的固定训练任务面板，生成 ever/final 矩阵、NBT、FWT、AUC 与区间。

主对照累计 12 个训练运行、2,457,600 个新增策略控制步，额外评估与仿真检查单列。回放固定总容量 256 个动作 chunk，BC 权重 0.03，来源为每个允许训练任务的原始 `demo_0`。原始专家图像与作者 regenerated SFT 图像不同，SFT demo 重叠未知；这些条件写入数据 manifest。

全部方法在无有效混合成功组时同时跳过优化与 BC；已执行环境步仍计入预算。资格验证要求整个运行存在隔离的非零 RL 梯度；ER 还要求隔离的非零 BC 梯度。

资格验证只使用开发任务评估相机变体与 Object 0、1 的基座能力。主对照的末尾才执行冻结的最终评估：Spatial 7–9 的 RL 留出、训练任务上的固定相机偏移，以及 Object 3、4 的跨 suite 迁移。Object 的全预训练历史曝光为 unknown，不能据此称全训练历史未见。所有方法和参数在这些终评之前已经固定。

每批 `plan.json` 记录任务、方法、预算、源代码 commit、模型/数据 revision、曝光类别、路径和完整作业列表。单批作业通过 Slurm `afterok` 串行衔接，最多同时使用 2 张 4090；任何失败保留非零退出码并阻断后续作业。代码/Conda 在 homes，数据、输出和 Ray 溢写在个人存储目录，不依赖本机常开。

入口：

```powershell
./tools/remote.ps1 replay-download
# 代码修改经审查提交后
./tools/remote.ps1 sync
./tools/remote.ps1 experiment-tests
./tools/remote.ps1 experiment-submit qualification -SuiteName 20261104-qualification-v1
./tools/remote.ps1 experiment-status 20261104-qualification-v1
# 资格验证通过后提交；也可显式依赖已核实的资格 gate 作业
./tools/remote.ps1 experiment-submit core -SuiteName 20261104-core-v1
./tools/remote.ps1 experiment-status 20261104-core-v1
```

当前支持阶段边界保存与独立重载，尚不支持优化器/RNG/预算的完整断点恢复。失败重跑需使用新批次/明确新尝试，保留原日志与已消耗交互；不能覆盖原证据或把失败成本抹去。
