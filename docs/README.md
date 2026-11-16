# FYP 目录索引

当天的作业、结果和后续安排从根目录的 [PROGRESS.md](../PROGRESS.md) 开始。下表区分实验说明、机器可读登记和运行产物，便于取证和交接。

| 查找内容 | 位置 | 管理方式 |
| --- | --- | --- |
| X-VLA 实验配置、冻结初态、作业 ID | [协议执行登记](xvla_protocol_20261110_execution.md)、[注册表](../configs/xvla/protocol_20261110/registry.json)、[任务划分](../configs/xvla/protocol_20261110/split.json) | Git；运行前锁定协议 |
| X-VLA 实现、环境、命令 | [接入入口](xvla/README.md)、[运行手册](xvla_runbook.md)、[接入审计](xvla_integration_audit.md)、[调度工具](../tools/xvla/) | Git；上游参考放在 `docs/xvla/reference/`，运行时在 `third_party/rlinf/` |
| X-VLA-Pt 源任务的早期训练和诊断 | [源任务实验](xvla_pt_source_experiment.md)、[实现交接](../CODEX_HANDOFF.md) | Git；保留逐阶段证据 |
| OpenVLA-OFT 基线 | [旧基线启动记录](experiment-progress-20261104.md)、[部署](deployment.md)、[基线执行协议](experiment-execution.md)、[长序列协议](long-sequence-execution.md) | Git；与 X-VLA 的命令、环境和产物分开 |
| 离线报告、图片和仿真视频 | [本地报告目录索引](reports/README.md) | 保留各日期的快照目录及 ZIP，产物不入 Git |
| 本机证据副本、缓存和构建中间产物 | `.local/`；X-VLA 小型证据在 `.local/xvla-evidence/`（11 月 16 日调度快照按日存放）；旧报告构建证据在 `.local/report-build/` | 本机工作区，Git 忽略；原始结果以服务器为准 |

服务器路径由 [`tools/idac/settings.psd1`](../tools/idac/settings.psd1) 提供。X-VLA 产物统一在 `<DataRoot>/xvla/`，下列路径均相对于该目录。

| 用途 | 路径 |
| --- | --- |
| Slurm 日志 | `logs/<job>.out`、`logs/<job>.err` |
| 提交回执、权重、控制步、逐回合评估和资源记录 | `runs/submission-<job>.json`、`runs/<job>/` |
| E2 / E3 派发表、最终评估清单 | `protocol_20261110/` |
| 原始示教、转换产物与审计 | `datasets/`、`evidence/` |
| 权重快照 | `models/` |

服务器代码与 Conda 分别在个人 homes 的 `<RunCodeRoot>/<commit>/` 和 `<CondaRoot>/envs/xvla_rlinf/`。保留既有作业目录名，便于通过登记中的 Job ID 找到原始文件。
