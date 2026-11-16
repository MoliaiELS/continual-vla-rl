# RoboTwin B0 执行登记

2026-11-16 开始。任务定义审计完成 50/50，仿真与正式覆盖待验收。统计分母：smoke 20；首批 10,000；扩测新增 4,000。当前可核实完成 0，成功率留空。

| 先决条件或阶段 | 当前证据 |
| --- | --- |
| 固定权重 | `2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18` 的 6 个文件均已核验；`config.json` 确认 `ee6d`、proprio 开启、30 anchors。权重 SHA256 `3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b`。路径 `<DataRoot>/xvla/robotwin-b0/models/X-VLA-RoboTwin2--a157c580cfe6`。 |
| 软件环境 | CuRobo 编译作业 `20298` 在 `3090node1` 完成，`0:0`，7 分 20 秒，`pip check` 通过。GPU 探针 `20299` 在 `4090node3` 完成，`0:0`，1 分 37 秒。获配 GPU 上的 Vulkan 帧有效，冻结权重经完整校验和推理输出 `30×20`，转换得 `30×16`；进程每 5 秒采样显存峰值 4,204 MiB。这是探针证据，未执行 RoboTwin 任务。 |
| 资产 | 三个冻结 ZIP 共 14,927,288,034 字节。HF 主站连接重置，镜像 8 MiB 定长 Range 断点下载运行中；`embodiments.zip` 已通过 SHA256，其余两项尚未完成。 |
| 任务定义 | [50 项中文任务目录](task-catalog.md)与官方目录、固定作者 client 和原生步数表逐项吻合。 |
| Smoke／对齐／缺口覆盖 | 尚无闭环证据。 |
| 正式 | 0/14,000，未提交。 |

已启动评测派发器，执行提交 `009f05cfe4de2de9fe4d911b6fd665776add7abc`。状态在 `dispatch/<执行提交>/state.json`，日志在同目录的 `dispatch.log`。探针、资产校验、固定噪声对齐与两条件 smoke 是正式评测门槛。`./tools/xvla/remote.ps1 robotwin-status` 可查询。

## 服务器工作树清理与 OpenVLA 盘点

已通过 `git worktree remove` 移除 14 个无本地文件、无作业和环境引用、由远端分支可恢复的旧 detached 工作树，清理前目录占用合计 7,391,424 KiB（约 7.05 GiB）。剩余 70 个。删除清单位于 `<DataRoot>/xvla/robotwin-b0/evidence/clean-worktrees-20261116.tsv`。活跃工作树、固定归档引用、带本地文件的目录和 editable 安装引用的 `3dff0bddf7f1421fda0072f83ea9b59175ecc5aa` 均保留。

以下为 `du -sh` 的占用估算。旧 OpenVLA 归档、权重和环境尚未删除。

| 内容 | 服务器位置 | 占用与依赖 |
| --- | --- | --- |
| 公开 OpenVLA-OFT SFT 权重 | `<DataRoot>/models/Openvla-oft-SFT-libero-spatial-traj1` | 15 GiB。RoboTwin 不使用，可按旧 [部署手册](../deployment.md) 的固定 snapshot 重下。 |
| 旧 OpenVLA 实验 | `<DataRoot>/experiments/20261104-*` | 29 GiB；`core-v3` 24 GiB、`qualification-v3` 3.7 GiB、`qualification-v4` 1.3 GiB。包含未打进收尾包的原始训练产物与检查点。 |
| 旧部署结果和日志 | `<DataRoot>/runs`、`logs`、`reports` | 632 MiB、276 KiB、5.6 MiB。保留可追溯证据。 |
| LIBERO 原始数据 | `<DataRoot>/datasets/libero_raw` | 2.9 GiB，与旧 LIBERO 研究相关；不归为 OpenVLA 专用权重。 |
| 旧 Conda 环境 | `<CondaRoot>/envs/vla_budget_rl` | 9.2 GiB。当前 RoboTwin 资产下载与派发器仍使用此环境的 Python，报告与权重下载入口也有引用。不能在本轮运行中移除。 |
| 旧 Conda 缓存、源码 | `<CondaRoot>/pkgs-vla-budget`、服务器 `RLinf/openvla-oft` | 279 MiB、1.4 MiB。Conda 可有硬链接，删除缓存的实际释放量与目录占用不同。代码保留在 Git。 |

恢复命令、证据目录与数值口径见[协议与运行手册](README.md)。
