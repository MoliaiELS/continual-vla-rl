# 2026-11-16 实验归档

研究方向调整，本轮 OpenVLA-OFT 和 X-VLA 实验归档。北京时间 2026-11-16 19:56:49 核对 Slurm 队列和旧派发进程，活动作业为 0，无需取消作业。服务器在 `<DataRoot>/xvla/protocol_20261110/archive.json` 登记归档，当前客户端的 `e2-start`、`e3-start`、`final-submit` 读取该标记并拒绝继续派发。

## X-VLA-Pt

旧主线以 LIBERO-Spatial 任务 0 的 10 条示教建立共同起点 B0。两个 SFT 阶段各训练 1,024 次更新，第二阶段使用 rank 8 LoRA。B0 权重来自作业 `19525`，在任务 0 的 30 个开发初态成功 25/30。三个 E2 PPO pilot 均跑完 25k 控制步，B1、B2、M 的开发终点为 24、25、25/30。配置和权重来源见[协议执行登记](../xvla_protocol_20261110_execution.md)。

| E3 条件 | seed 101 | seed 202 | seed 303 |
| --- | --- | --- | --- |
| SFT-cont，原 10 条示教追加 2,048 更新 | `19735` 27/30 | `19736` 28/30 | `19740` 29/30 |
| B1，5 个初态均匀采样，200k 控制步 | `19746` 22/30 | `19894` 26/30 | `20137` 失败 |
| B2，10 个初态均匀采样，200k 控制步 | `19788` 25/30 | `19990` 22/30 | 未提交 |
| M，与 B2 同一支持集，增加两个边缘初态权重，200k 控制步 | `19834` 25/30 | `20063` 27/30 | 未提交 |

已核验 9/12 组和 27 个检查点。SFT-cont 三个终点在共同开发集平均 93.3%，B0 为 83.3%。PPO 只有 seed 101 和 202 成套完成，以上分数对应同一源任务开发集。原定 H 上的 M−B2 迁移比较缺少最终评估数据。

`20137` 在 4090node3 完成首次 rollout（287 个控制步、40 个静置步），随后 actor 重算最大 logprob 绝对差 `0.0011177063` 超过阈值 `0.001`。作业于第一次优化前以 `FAILED 1:0` 结束，无检查点和正式预算汇总。原始派发表保留 `RUNNING / SUBMITTED` 供审计，停跑状态以归档标记及 Slurm 记录为准。

独立源任务 A 和迁移面板 H 均未启动，冻结初态保留。E4 已确认任务 0/1/2 原始示教、下载凭证和合并清单工具，三任务 Cseq、顺序训练和 NBT/FWT 结果均未生成。E5 原始 Pt 直接 PPO 未启动。[实验注册表](../../configs/xvla/protocol_20261110/registry.json) · [冻结初态](../../configs/xvla/protocol_20261110/split.json)。

## OpenVLA-OFT 基线

下表的数量是 Slurm 作业数，包含训练以外的准备和评估。`qualification-v2/v3/v4` 的分析文件随归档保存。训练、逐回合记录和权重在服务器原目录。

| 服务器 suite | 已提交 / 计划 | 停止时核验 |
| --- | ---: | --- |
| `20261104-qualification-v1` | 2 / 18 | 2 已完成 |
| `20261104-qualification-v2` | 4 / 18 | 3 已完成，`19030` 已取消 |
| `20261104-qualification-v3` | 10 / 18 | 9 已完成，`19040` 失败 |
| `20261104-qualification-v4` | 8 / 8 | 8 已完成 |
| `20261104-core-v3` | 45 / 114 | 44 已完成，末个 `19316` 已取消 |
| `20261104-core-v1/v2`、`20261104-stress10-v1/v2` | 0 | 四个目录仅有冻结计划 |

## 归档文件

| 内容 | 位置 |
| --- | --- |
| 关闭前的仓库状态 | `0823f6bccb5e6cd342ba6413b8bbad935bff7e91`，包含完整旧进度。E3 派发版本 `7ddb110d183d54a2436a9b95e4919bd624befb01`。 |
| 本机收尾包 | `docs/reports/2026-11-16-experiment-closeout/`。`closeout-evidence.zip` 含原始派发表、Slurm 记账、E2/E3 预算与开发评估摘要、`20137` 日志、OpenVLA 计划和现有分析。`closeout.json` 附 ZIP 的大小和 SHA256。`protocol.md` 的 SHA256 与冻结登记一致。 |
| 历史图表和视频 | `docs/reports/2026-11-04/`、`docs/reports/2026-11-11-xvla/`，保留 HTML、相关资产和各自 ZIP。[离线报告索引](../reports/README.md) |
| 服务器归档 | `<DataRoot>/xvla/protocol_20261110/archive/2026-11-16/`，含 `closeout-evidence.zip` 和 `closeout.json`。 |
| 服务器原始产物 | `<DataRoot>/xvla/{runs,logs,datasets,models,evidence}/` 和 `<DataRoot>/experiments/20261104-*/`。完整训练日志、逐回合文件、视频和权重可按作业号查找。 |

本机收尾包的 ZIP SHA256 为 `de4626b92fd9553c3ab98a8ae1cc2c7546d23187d5ec52f86422fe1cd2c985e6`。[目录索引](../README.md) · [当前进度](../../PROGRESS.md)。
