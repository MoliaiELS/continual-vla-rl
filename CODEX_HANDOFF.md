# X-VLA 实现交接

2026-11-08 开始实际接入。此前仓库没有 CODEX_HANDOFF.md，本文件新增以保存实施状态。

- 当前分支 `codex/xvla-integration`，启动时 HEAD `bdaa3568`，dirty status 为空，无既有 submodule 或 X-VLA adapter。用户原有提交全部保留。
- 根目录仍是 Simple Recipe 基线。实际新接口运行时位于 `third_party/rlinf`，git submodule 固定 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc`。
- X-VLA 源码固定 `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`。已下载审计源到本机 `.local/xvla-source`，vendor 与 adapter 使用 `xvla_rlinf` 命名空间。
- 原生评估权重：`2toINF/X-VLA-Libero@129e71460678b7236cee6fc9707f09d9fa0c3590`；后续受控 SFT 起点：`2toINF/X-VLA-Pt@c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`。
- 服务器配置来自 `tools/idac/settings.psd1`，入口 `tools/xvla/remote.ps1`，新独立 prefix 为其 CondaRoot 下 `envs/xvla_rlinf`；新结果位于其 DataRoot 下 `xvla`。
- 已审计旧环境 Python3.11.10 / Torch2.6.0 / transformers4.40.1 / PEFT0.11.1 / timm0.9.10。与 X-VLA 的 transformers<=4.51.3、PEFT0.17.1 不一致，因此只在独立克隆环境安装 pin，原环境不变。
- 2026-11-08 核实旧作业19316为 OpenVLA-OFT 五任务 Joint+ER 大规模对照，用户条件授权后已取消；checkpoint、日志保留，取消凭据在个人DataRoot/xvla/evidence。新 smoke 仍遵守最多2卡上限，不能停止他人作业或超配。所有仿真/训练通过 Slurm。
- 已实际实现adapter/native/SFT、终端clean-action条件高斯策略和默认关闭的上游补丁。native_eval保留官方循环与历史。
- 验收明细最终写 `docs/xvla_integration_audit.md`，采样器定义 `docs/xvla_rl_sampler.md`，操作手册 `docs/xvla_runbook.md`。未执行真实项标 NOT RUN 或具体 BLOCKED，不能把 mock 当真机/仿真训练。
- 独立环境安装、pip check、固定 LIBERO 权重验证完成。19330为57项真实依赖单元PASS；19331固定噪声官方循环对齐PASS（raw最大误差0）；19333扩大为77项单元/配置/进程注册测试PASS。原生仿真首次19332配置继承失败，19334暴露上游NVML忽略Slurm可见范围的枚举问题，均保留非零退出码，正以默认关闭补丁修复。不能把这两次失败算真实rollout通过。

## 2026-11-09 恢复连接后的交接

- 本轮SFT和GRPO使用已同步的固定提交 `fedcdf2d980db9e4a880ed375c468f4e6550a107`，submodule仍固定1bf78980。测试修复 `b2fa4b9b` 只改测试。
- 两个固定HF snapshot均已完整哈希校验；Pt下载的Xet停滞已改成HTTP Range断点续传，未更换revision或执行HF Python。
- 19344独立官方processor/raw20/sigmoid/env7对照误差均0，FP32/H30/steps10。19336原生真实2/2成功，150策略步+20settling；19338同history加噪2/2成功，151+20。仅同一任务/初态的连通性证据。
- 19337成功转换demo_48（1独立demo、96动作帧、86官方窗口）。环境20Hz与官方SFT插值30Hz分别记录，不混用。
- 19339真实PPO1次更新及post-update rollout成功；actor128020参数、value267777参数，全部5个actor参数组非零梯度和实际变化；初始有效ratio0.995446–1.001728；更新后全state hash/固定noise probe相同；原生DCP扰动后恢复PASS。完整Slurm331秒=.091944GPUh，actor峰值14.562GiB、5秒采样己方compute总21492MiB（不含全部EGL）。本地`.local/xvla-evidence/ppo19339_summary.json`和三个视频已保留。
- 修复版 `b2fa4b9b` 的 Slurm CPU 回归作业 19438 通过 **126/126**，无跳过。旧结果 19352 在 `2862a8c6` 上通过125项，19405的旧patch文件数断言失败已修复。
- 19343/19345全胜组方差0，按设计拒绝伪造更新。19348压力策略σpos=.04/rot=.12/grip=.6得到两组[0,0,1,1]、[0,1,1,1]，但第二次FSDP前向view写回失败，不能称GRPO有效更新；采用单根FSDP配置重跑。
- 19353（SFT）和19354（GRPO stress）均以1:0退出。SFT的完整state导出缺冻结视觉投影，GRPO真实回报组 [0,1,0,0] 和 [1,1,1,1] 中一组有方差，随后actor backward OOM。单根 FSDP 排除冻结参数和 GRPO 的非重入激活重算为本轮修复。
- 19407（SFT）和19408（GRPO stress）在单张4090上依次完成，退出0:0。19407完成Pt的1条示教、1步SFT，5个训练张量变化、冻结组不变。19408有一组混合回报并完成1步actor更新、同步和新rollout。两项均通过同worker原生恢复与新进程严格权重加载。
- 下一阶段可做LIBERO-Spatial的Pt/SFT配对评估。19407产物可验证评估加载，数据效率对照需另定独立示教和训练步数。完整新worker恢复、Pt→SFT→RL串联、多GPU/full/LoRA真实训练和研究规模评估仍为NOT RUN。
- 恢复连接优先使用`tools/xvla/remote.ps1 session`的单连接串行诊断；从settings派生FYP变量。所有新训练依旧提交Git固定worktree并经sbatch，最多2卡（含pending预约的保守预算），不连接他人的Ray。

## 2026-11-09 LIBERO-Spatial benchmark

- 固定提交 `6a0e7fcbd44ff79bb9487cf2a0aacf3456d21fd1` 新增单卡配对评估入口。3090 验证作业 19441 通过，任务0初态0成功，73个策略步、10个settling步、0个终止后步。Slurm耗时384秒，获配设备5秒采样峰值4556MiB。
- 官方 Libero 基线 19443、Pt 基线 19444 均在 3090node2 完成，退出码0:0，成功分别为46/50和0/50。已逐项核对50组任务、初态哈希、环境种子和策略噪声种子一致。逐回合及资源证据位于各自 `runs/<job>/`。
- 作业 19469 在 3090node2 完成，退出码0:0，用19407产物评估同一50回合。该产物来自任务0的1条独立示教和1次优化器更新。50个回合全部达到800步，成功0/50，与配对Pt基线的二元结果逐项相同。证据位于 `runs/19469/`。
