# RoboTwin B0 执行登记

2026-11-16 开始。任务定义审计完成 50/50，仿真与正式覆盖待验收。统计分母：smoke 20；首批 10,000；扩测新增 4,000。当前可核实完成 0，成功率留空。

| 先决条件或阶段 | 当前证据 |
| --- | --- |
| 固定权重 | `2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18` 的 6 个文件均已核验；`config.json` 确认 `ee6d`、proprio 开启、30 anchors。权重 SHA256 `3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b`。路径 `<DataRoot>/xvla/robotwin-b0/models/X-VLA-RoboTwin2--a157c580cfe6`。 |
| 软件环境 | CuRobo 编译作业 `20298` 在 `3090node1` 完成，`0:0`，7 分 20 秒，`pip check` 通过。GPU 探针 `20299` 在 `4090node3` 完成，`0:0`，1 分 37 秒。获配 GPU 上的 Vulkan 帧有效，冻结权重经完整校验和推理输出 `30×20`，转换得 `30×16`；进程每 5 秒采样显存峰值 4,204 MiB。这是探针证据，未执行 RoboTwin 任务。 |
| 资产 | 三个冻结 ZIP 共 14,927,288,034 字节，全部通过 SHA256。CPU 作业 `20310` 在 `4090node3` 本地存储用 5 分 49 秒完成解压，随后写入完整资产标记并链接运行目录。 |
| 任务定义 | [50 项中文任务目录](task-catalog.md)与官方目录、固定作者 client 和原生步数表逐项吻合。 |
| Smoke／对齐／缺口覆盖 | 完整资产探针 `20309` 通过。旧版 Easy `20311` 和 Hard `20312` 因物理步数对齐未通过而退出 `1:0`。新版评测固定在 `9baa17360abe242cb36e8f9c78f0aadbe64b3bee`。新版探针 `20317` 通过，Easy `20319` 完成 10/10 回合，Hard `20320` 已完成 5/10 并继续运行。截至 2026-11-17 00:39，smoke 共 15/20 回合。 |
| 正式 | 0/14,000。服务器派发器从固定工作树运行，完整 smoke 通过后先提交缺口覆盖，再提交首批 10,000 回合和预定扩测。 |

旧版 smoke 的固定提交为 `7a9141c80e0e57e93ac960f8a226eecee1b14fc8`。Slurm 账号 `MaxJobs=1`、`MaxSubmitJobs=2`，此次两路单卡作业串行。11 月 17 日 00:21 的双卡调度试算显示，同节点两张 RTX 4090 最早约 11:00 才可用。状态在 `dispatch/<执行提交>/state.json`，日志在同目录的 `dispatch.log`，可通过 `./tools/xvla/remote.ps1 robotwin-status` 查询。

固定物体清单列出了 `assets/objects/objaverse/pencil/001/model.urdf`，冻结压缩包缺少该文件。Hard 专家筛选记录因该确切路径触发缺文件错误的初态，并继续筛选后续种子。`20312` 已在 `move_stapler_pad` 的种子 `200000004` 记录该排除原因。

两种难度的四次对照使用真实场景，固定观测和噪声下模型输出完全一致，动作锚点数量和成功判定也一致。下表是旧评测入口、作者入口分别独立重放的物理步数。该轮在物理步门槛提前退出，未计算执行命令或 TCP 轨迹差。`20313` 增加了逐锚点步数、命令、TCP 轨迹和作者循环自重放的测量。

| 任务／条件 | 锚点 | 判定 | 物理步数（旧入口／作者） |
| --- | ---: | --- | --- |
| `move_stapler_pad` Easy | 219 | 成功 | 11,710／11,687 |
| `turn_switch` Easy | 300 | 失败 | 24,370／24,388 |
| `move_stapler_pad` Hard | 300 | 失败 | 18,175／18,167 |
| `turn_switch` Hard | 300 | 失败 | 17,137／17,130 |

定向作业 `20313` 在 `move_stapler_pad` Easy 的同一冻结种子上核对了 133 条共同执行命令，最大绝对差为 0，记录的初态差为 0。两个入口都判定成功，分别在第 133 和 134 个锚点停止，物理步数为 8,588 和 8,608。该次旧诊断受锚点计数条件限制，未进入作者自重放。后续对照使用同一个经 SHA256 校验的作者循环，直接核对评测适配器的模型输入、输出和实际执行命令，并独立记录同一串动作在重置场景上的轨迹及物理步数。

`20314` 的五次模型查询与作者 `/act` 一致，140 条实际送达的命令与独立转换一致。作者循环原运行和自重放在同一冻结种子上分别执行 140 和 139 个锚点，均判定成功。139 条共同执行命令完全一致，物理步数 9,010／8,991，TCP 位置差最大 0.339 mm、旋转角差最大 0.001455 rad，夹爪归一化值差最大 0.138889。计数从首个锚点起已经出现差异。`robotwin-native-b0-v2` 使用相同的作者循环进行验收和正式执行，独立重放的轨迹及终止差单列为仿真复现性证据。

`20315` 的十次模型查询与作者 `/act` 一致，300 条实际送达的命令与独立转换一致。作者循环自重放也执行 300 个锚点，命令、调用顺序和失败判定一致，物理步数为 19,135／19,145，TCP 位置差最大 0.444 mm。两路均完成五个 smoke 回合，Easy 4 成功、1 失败，Hard 1 成功、4 失败。每回合平均耗时 Easy 34.1 秒、Hard 55.4 秒，均来自同一任务，涵盖回合 reset 和 rollout。两路各占用 1 张 RTX 4090 和 8 个 CPU，实测进程峰值显存分别为 9,727 和 10,239 MiB。新版派发器已以 `9baa1736` 恢复全流程，沿用已完成回合并补齐 `turn_switch`。

后续开发入口已迁至 [RLinf-VLA 的 `codex/xvla`](https://github.com/MoliaiELS/RLinf-VLA/tree/codex/xvla)。独立的迁移验证作业 `20301` 在 `4090node3` 通过 Vulkan 和 X-VLA GPU 推理探针，退出码 `0:0`。

`20303` 在 `4090node3` 用固定合成相机数组和末端位姿比较本地推理与作者 `/act`，`30×20` 动作张量的最大绝对差为 `0`。`20304` 使用核验过的物体和机器人资产，各跑一个 Easy 回合。`move_stapler_pad` 和 `turn_switch` 都通过官方判定，分别执行 124 和 122 个动作锚点。两个任务的专家筛选、回合记录和视频位于 `<DataRoot>/xvla/robotwin-native/easy-preflight/126016d84aa009895fcee054f0ca1ecbb5555f7c/coverage/`。这两个回合用于运行诊断，正式分母仍为 0/14,000。

## 服务器工作树清理与 OpenVLA 盘点

已通过 `git worktree remove` 移除 14 个无本地文件、无作业和环境引用、由远端分支可恢复的旧 detached 工作树，清理前目录占用合计 7,391,424 KiB（约 7.05 GiB）。剩余 70 个。删除清单位于 `<DataRoot>/xvla/robotwin-b0/evidence/clean-worktrees-20261116.tsv`。活跃工作树、固定归档引用、带本地文件的目录和 editable 安装引用的 `3dff0bddf7f1421fda0072f83ea9b59175ecc5aa` 均保留。

2026-11-16 按用户要求清理服务器的旧 OpenVLA 权重、部署缓存和实验产物。下表的删除量为清理前 `du -sk` 实测，合计约 48.69 GiB。路径和占用回执位于 `<DataRoot>/xvla/robotwin-b0/evidence/cleanup-openvla-20261116.tsv`。

| 内容 | 服务器位置 | 状态 |
| --- | --- | --- |
| 公开 OpenVLA-OFT SFT 权重 | `<DataRoot>/models/Openvla-oft-SFT-libero-spatial-traj1` | 已删除，14.05 GiB。可按旧部署手册的固定 snapshot 重下。 |
| 旧部署下载与构建缓存 | `<DataRoot>/cache` | 已删除，5.66 GiB。 |
| 旧 OpenVLA 实验和部署产物 | `<DataRoot>/experiments/20261104-*` 的 9 个目录，及 `<DataRoot>/{runs,logs,reports}` | 已删除，28.98 GiB，含原始训练检查点、逐回合记录和部署报告。 |
| 收尾归档和旧 X-VLA 产物 | `<DataRoot>/xvla/` | 收尾归档及旧 X-VLA 产物保留。 |
| LIBERO 原始数据 | `<DataRoot>/datasets/libero_raw` | 保留，旧 X-VLA 也引用。 |
| 旧 Conda 环境、个人环境包缓存和源码 | `<CondaRoot>/envs/vla_budget_rl`、`<CondaRoot>/pkgs-vla-budget`、服务器 `RLinf/openvla-oft` | 保留。当前 RoboTwin 资产下载和派发使用此环境的 Python。 |

恢复命令、证据目录与数值口径见[协议与运行手册](README.md)。
