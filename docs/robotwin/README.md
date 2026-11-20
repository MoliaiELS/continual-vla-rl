# X‑VLA RoboTwin 2.0 Easy／Hard 基座能力评测

本轮用于 **baseline calibration／能力审计**。使用已经按 RoboTwin 示教数据适配的 [官方 RoboTwin 权重](https://huggingface.co/2toINF/X-VLA-RoboTwin2/tree/a157c580cfe6f9f445614490f3bec1b2f9ef9f18)，冻结推理。后续 RL 的未参与任务不等于 SFT 未见任务，最终对照需另冻结独立的评测初态。

[任务目录](task-catalog.md)按官方 50 项清单列出任务内容、分类、成功判定和覆盖状态。[执行登记](status.md)记录真实作业和进度。[X‑VLA 作者的公开参考](https://github.com/2toinf/X-VLA/blob/6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2/evaluation/robotwin-2.0/README.md)是 Easy 70.0%、Hard 39.0%，逐项参考见[论文附录 M 表 16](https://arxiv.org/html/2510.10274v1#A13.T16)。这些不是本地实测。

2026-11-20 用户确认 Hard 全部 50 项按各自原生动作上限评测，详见 [Hard 原生预算与自动接续](hard-native-budget.md)。当前 Easy 作业继续运行；Easy 五任务重评和原定扩测完成后，接续新的 Hard 协议。下文 10 次查询／300 动作的定义保留为原评测记录，恢复当前流程应使用新文档的 `-HardNativeBudget` 入口。

## 固定身份和实验定义

| 项目 | 版本或规则 |
| --- | --- |
| 权重 | `2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18`；`ee6d`、proprio 开启、30 anchors；FP32、冻结、10 次 denoising；权重及 processor 哈希见已校验下载回执 |
| 项目和上游 | 项目以 Slurm 作业的固定 Git worktree 为准；X‑VLA `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`；RLinf submodule `1bf78980d9339cbc67497eddceb9a1e3a74e2acc` |
| 仿真和规划 | 原生 RoboTwin `6e75b8b80ee32d2f495eb756cf5687e4e16f0eea`，权重发布前最近的 `main` 提交；保留作者 client 所需的 `task_config`、`script` 和逐 anchor `take_action(action_type='ee')`。CuRobo v0.7.8 `d64c4b005459db10c5dd867d8b30a87d5bda9bdb`。预定 Python 3.11、Torch 2.6 / CUDA 12.4、SAPIEN 3.0.0b1、MPLib 0.2.1；是否可运行以 Slurm 验收为准。 |
| 资产 | `TianxingChen/RoboTwin2.0@25ed6fa76343d7b900e2b44d39f32a0eae1950b1` 的三个原始 ZIP，逐文件 SHA256 验证。Aloha AgileX，按原生脚本展开放置路径。 |
| Easy／Hard | `task_config/demo_clean.yml` / `demo_randomized.yml`，保留随机化参数和原生物体、纹理、相机。使用原生 API 避开 RLinf 训练包装层的关节执行方式及相机 near plane 默认值。 |
| 观测 | head、left、right 顺序 RGB，原生 D435 320×240、near 0.1 m、far 100 m，原生 ray tracing 32 spp，无额外 crop；缩放交给原生 processor。domain 6。 |
| 动作与历史 | 双臂 20D `xyz + rot6D + grip`，转换成两个 8D `xyz + quat + grip` 命令。保留作者 SciPy 参数和 RoboTwin planner 使用的原始四元数分量顺序。夹爪阈值 `>0.7`，命令在 `{-1,+1}`。后续 proprio 的 pose 部分来自最后命令，夹爪状态仍为实测。 |
| 指令 | `task_id.replace('_', ' ')`，完整文本写入每个 episode。 |
| 终止 | 每回合最多 10 次模型查询，每次 30 anchors；每个 anchor 调用原生成功判定，成功即停。原生 `step_lim` 之后的 no-op 仍计为已发送命令，单独记实际推进的 anchors、仿真 tick、settling ticks、模拟秒数和规划失败。默认仿真步长 1/250 s。 |

三个任务的原生成功条件依赖专家示教脚本赋值：`open_laptop`、`place_object_scale`、`put_object_cabinet`。评测 adapter 在 reset 后只用原始对象位姿按专家脚本的规则初始化相关字段，不执行示教动作、不改成功谓词。`place_fan` 无此依赖。两个 Hard 任务复用时可能受原生可变默认参数影响，adapter 每次传入独立列表。SAPIEN 绑定到 Slurm 获配设备，对齐 GPU 参数而不改画面参数。

## 初态、统计与产物

- Smoke：`move_stapler_pad`、`turn_switch` × Easy／Hard × 5，使用独立种子。`robotwin-native-b0-v2` 的 smoke 与正式评测共用经 SHA256 校验的作者 `_rollout`，保留 v1 的 policy seed。Smoke 在同一真实观测和同一噪声下比较本地推理与作者 `/act`，逐条核对作者循环发出的命令。在另一份同种子、经核对的初态用同一作者循环重放相同动作，并分开记录 TCP、夹爪、tick 和成功判定。作者若需要超出记录范围的查询，记录已共同执行的轨迹前缀。四个包装层缺口另做 `coverage`，与正式数据分开。
- 正式：50 个任务每条件首批 100，共 10,000 回合。预先指定 10 个重点任务每条件再各加两批各 100，累计 14,000。批次是 evaluation batches。
- 固定条件内按候选 seed 升序预筛，正式从 2000 开始，smoke／coverage 使用独立命名空间。通过原生专家成功判定并去重后冻结完整清单及 policy seed，新旧回合不按结果重新抽样。Easy／Hard 的同整数 seed 不作为配对证据。
- 正常失败只记一次。基础设施异常按相同 seed 有限重试，耗尽记 `unresolved`。保留候选排除记录、每次尝试和回合日志；每个 episode ID 只进入一次分母。计划数、完成数、Wilson 区间和未完成范围通过汇总命令输出。总体与类别以任务等权计算，首批与扩测累计分列。
- 视频以固定顺序保存每个任务／条件前两次成功和前两次失败。未出现对应结果或编码失败时在日志中体现。

服务器目录根为 `<DataRoot>/xvla/robotwin-b0/`，包含 `archives/`、`models/`、`jobs/<jobid>/runtime.json`、`logs/`、`panels/<项目提交>/<阶段>/<任务>/<条件>/` 和 `reports/`。项目代码与独立 `xvla_robotwin_b0` Conda 环境在个人 homes。`runtime.json` 核对 GPU、驱动、依赖、代码和配置哈希；shard 记录模型/初态 manifest、协议指纹、原始 episode、对齐证据和视频。

## 执行与恢复

在本机仓库根目录运行。日常 `sync -RootOnly` 推送提交，并快进服务器的 `RLinf` 编辑目录到 X‑VLA 分支；准备作业时用 `sync` 另外生成固定 worktree。若直接在服务器编辑，请先保留该改动并与本机分支同步；脚本会拒绝覆盖有改动的远端目录。选择作业使用的已同步提交；后续编辑和提交不影响已有作业。

```powershell
./tools/xvla/remote.ps1 sync
./tools/xvla/remote.ps1 download -Model robotwin
./tools/xvla/remote.ps1 robotwin-prepare env
./tools/xvla/remote.ps1 robotwin-prepare assets
# 安装和资产下载可并行；看到 ROBOTWIN_ENV_READY 后再编译。
./tools/xvla/remote.ps1 robotwin-submit build
# 看到 ROBOTWIN_ASSETS_READY 后生成原生模型路径配置。
./tools/xvla/remote.ps1 robotwin-prepare link
# build 成功后可以先验收 CUDA/CuRobo/Vulkan，无需等待任务资产。
./tools/xvla/remote.ps1 robotwin-submit probe
./tools/xvla/remote.ps1 robotwin-status
./tools/xvla/remote.ps1 robotwin-logs 12345
```

权重、资产、probe 就绪后，分别提交两个单卡 lane，比对串行与双 lane 吞吐、显存峰值和 reset／规划耗时。达到 smoke 的证据门槛后，才为相同执行提交运行正式或扩测。实际时间按 smoke 统计的有效回合/小时及专家预筛接受率外推；未实测前不给 10,000／14,000 回合估算。

```powershell
$c = git rev-parse HEAD # 或从 jobs.tsv 取既有评测提交
./tools/xvla/remote.ps1 robotwin-submit smoke -RobotCommit $c -RobotLane easy
./tools/xvla/remote.ps1 robotwin-submit smoke -RobotCommit $c -RobotLane hard
./tools/xvla/remote.ps1 robotwin-report smoke -RobotCommit $c
./tools/xvla/remote.ps1 robotwin-submit coverage -RobotCommit $c
# 以上回收验收证据后再提交正式任务，默认优先十个重点任务。
./tools/xvla/remote.ps1 robotwin-submit formal -RobotCommit $c -RobotLane easy -RobotHours 48
./tools/xvla/remote.ps1 robotwin-submit formal -RobotCommit $c -RobotLane hard -RobotHours 48
./tools/xvla/remote.ps1 robotwin-report formal -RobotCommit $c
# 首批结果与缺口记录齐备后，按照预定计划新增两批。
./tools/xvla/remote.ps1 robotwin-submit extension -RobotCommit $c -RobotLane easy -RobotHours 48
./tools/xvla/remote.ps1 robotwin-submit extension -RobotCommit $c -RobotLane hard -RobotHours 48
```

Slurm 超时或中断后，用同一 `-RobotCommit`、阶段和 lane 重新提交。runner 保留已接受初态、尝试次数和正式终局结果；`-RobotTask <ID>` 可只恢复某个任务。观测、动作、指令、预算或成功判定若有变化，须新建协议并独立记录。超过自动重试上限的回合继续列为 unresolved。Slurm 最多两张己方卡，排除登录节点，保留本次作业的 GPU 可见性。作业不声明无法调度的内存申请，物理内存只作为测量值。

先用 `./tools/xvla/remote.ps1 sync` 生成当前派发器的固定工作树，再运行 `./tools/xvla/remote.ps1 robotwin-run <CuRobo编译JobID> -RobotCommit $c -RobotHours 48`。`$c` 选择已同步的评测提交，可沿用该版本的初态和回合产物。派发仅提交 Slurm 作业，锁和 `dispatch/$c/state.json` 避免重复提交。等校验、GPU 探针与正式环境 smoke 门槛通过后才递交首批。按批汇总并在 Slurm 超时、抢占或节点故障后有限次自动续跑；断点和任务缺口遵守上述分母规则。使用 `robotwin-status` 查当前状态。

当前在跑的正式评测固定为 `9baa17360abe242cb36e8f9c78f0aadbe64b3bee`。若服务器派发器退出，先确认其进程已结束，再用 `./tools/xvla/remote.ps1 robotwin-run 20298 -RobotCommit 9baa17360abe242cb36e8f9c78f0aadbe64b3bee -RobotHours 48 -RecoverPrescreen -EasyFirst` 接管现有作业。`-RecoverPrescreen` 使后续作业按原日志续筛。空抓取位姿，以及倒垃圾任务在专家规划失败后读取空动作列表的情况，均需经固定版调用栈核实，才会记为专家预筛排除。续跑作业的评测与编排提交记录在 `evidence/prescreen-recovery-jobs.tsv`。`-EasyFirst` 要求 Easy 首批 5,000 和累计 7,000 回合均通过日志核对，再释放 Hard 的 Slurm 用户挂起。

`-PilotRoot` 可指定含并发短测入口的固定 `RLinf-VLA` 工作树。首次 Easy 正式续跑前，派发器等待上一作业释放名额，再在同一冻结评测环境中提交一次独立短测。短测若排队 15 分钟仍未开始，派发器撤回该短测并接续 Easy。短测结果保存在 `robotwin-native/jobs/<jobid>/concurrency-pilot/`，正式 Easy 续跑维持串行执行。

通过同一固定提交的并发短测后，可用 `-ParallelEasy` 在续跑时启用单卡双进程。专家预筛保持串行，每对就绪任务分别使用独立评测进程和日志。每个作业仍使用获配的一张 RTX 4090，申请 16 个 CPU，并记录评测提交、编排提交和任务实际并发数。若按计划在任务边界停止当前 Easy，应通过 `-PausedEasyJobId` 指定该作业编号，派发器会从原始日志接续。
