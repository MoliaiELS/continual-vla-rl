# RoboTwin B0 执行登记

2026-11-16 开始。任务定义审计完成 50/50，仿真与正式覆盖待验收。统计分母：smoke 20；首批 10,000；扩测新增 4,000。当前可核实完成 0，成功率留空。

| 先决条件或阶段 | 当前证据 |
| --- | --- |
| 固定权重 | `2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18` 的 6 个文件均已核验；`config.json` 确认 `ee6d`、proprio 开启、30 anchors。权重 SHA256 `3f16a4b67a1d2675fc1cb0350c6d5617522e452f47359e729d74d76b8aa4835b`。路径 `<DataRoot>/xvla/robotwin-b0/models/X-VLA-RoboTwin2--a157c580cfe6`。 |
| 软件环境 | CuRobo 编译作业 `20298` 在 `3090node1` 完成，`0:0`，7 分 20 秒，`pip check` 通过。GPU 探针 `20299` 在 `4090node3` 完成，`0:0`，1 分 37 秒。获配 GPU 上的 Vulkan 帧有效，冻结权重经完整校验和推理输出 `30×20`，转换得 `30×16`；进程每 5 秒采样显存峰值 4,204 MiB。这是探针证据，未执行 RoboTwin 任务。 |
| 资产 | 三个冻结 ZIP 共 14,927,288,034 字节。`embodiments.zip` 和 `objects.zip` 已通过 SHA256。`background_texture.zip` 通过镜像继续 8 MiB 分段下载。 |
| 任务定义 | [50 项中文任务目录](task-catalog.md)与官方目录、固定作者 client 和原生步数表逐项吻合。 |
| Smoke／对齐／缺口覆盖 | 固定输入接口作业 `20303` 对齐了模型输出。Easy 诊断 `20304` 完成两个成功回合；20 回合 smoke 及其场景对齐仍待完整资产。 |
| 正式 | 0/14,000，未提交。 |

ALOHA AgileX 的冻结资产包含左右臂 CuRobo 模板。准备和提交入口校验生成的 `curobo_left.yml` 与 `curobo_right.yml`。修正 Slurm `gres/gpu` 资源计数后，当前派发器固定在 `e194f8269aeb1fb78a05887fdf56d014860e0530`，提交探针 `20305` 后等候完整资产。状态在 `dispatch/<执行提交>/state.json`，日志在同目录的 `dispatch.log`，可通过 `./tools/xvla/remote.ps1 robotwin-status` 查询。

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
