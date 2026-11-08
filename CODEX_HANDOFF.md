# X-VLA 实现交接

2026-11-08 开始实际接入。此前仓库没有 CODEX_HANDOFF.md，本文件新增以保存实施状态。

- 当前分支 `codex/xvla-integration`，启动时 HEAD `bdaa3568`，dirty status 为空，无既有 submodule 或 X-VLA adapter。用户原有提交全部保留。
- 根目录仍是 Simple Recipe 基线。实际新接口运行时位于 `third_party/rlinf`，git submodule 固定 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc`。
- X-VLA 源码固定 `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`。已下载审计源到本机 `.local/xvla-source`，vendor 与 adapter 使用 `xvla_rlinf` 命名空间。
- 原生评估权重：`2toINF/X-VLA-Libero@129e71460678b7236cee6fc9707f09d9fa0c3590`；后续受控 SFT 起点：`2toINF/X-VLA-Pt@c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`。
- 服务器配置来自 `tools/idac/settings.psd1`，入口 `tools/xvla/remote.ps1`，新独立 prefix 为其 CondaRoot 下 `envs/xvla_rlinf`；新结果位于其 DataRoot 下 `xvla`。
- 已审计旧环境 Python3.11.10 / Torch2.6.0 / transformers4.40.1 / PEFT0.11.1 / timm0.9.10。与 X-VLA 的 transformers<=4.51.3、PEFT0.17.1 不一致，因此只在独立克隆环境安装 pin，原环境不变。
- 2026-11-08 核实旧作业19316为 OpenVLA-OFT 五任务 Joint+ER 大规模对照，用户条件授权后已取消；checkpoint、日志保留，取消凭据在个人DataRoot/xvla/evidence。新 smoke 仍遵守最多2卡上限，不能停止他人作业或超配。所有仿真/训练通过 Slurm。
- 已实际实现adapter/native/SFT、终端clean-action条件高斯策略、默认关闭的7处上游补丁。新RL sampler不冒充native概率；native_eval保留官方循环与历史。
- 验收明细最终写 `docs/xvla_integration_audit.md`，采样器定义 `docs/xvla_rl_sampler.md`，操作手册 `docs/xvla_runbook.md`。未执行真实项标 NOT RUN 或具体 BLOCKED，不能把 mock 当真机/仿真训练。
- 独立环境安装、pip check、固定 LIBERO 权重验证完成。19330为57项真实依赖单元PASS；19331固定噪声官方循环对齐PASS（raw最大误差0）；19333扩大为77项单元/配置/进程注册测试PASS。原生仿真首次19332配置继承失败，19334暴露上游NVML忽略Slurm可见范围的枚举问题，均保留非零退出码，正以默认关闭补丁修复。不能把这两次失败算真实rollout通过。

## 2026-11-08 网络中断时的实际交接

- 已确认远端同步的最后代码SHA：`ff440880da2ad9b8c0a4f92d6e3a925842f6de17`；上游submodule仍固定1bf78980，没有升级main/latest。后续本地提交为文档/查询工具，须网络恢复后再sync。
- 两个固定HF snapshot均已完整哈希校验；Pt下载的Xet停滞已改成HTTP Range断点续传，未更换revision或执行HF Python。
- 19344独立官方processor/raw20/sigmoid/env7对照误差均0，FP32/H30/steps10。19336原生真实2/2成功，150策略步+20settling；19338同history加噪2/2成功，151+20。仅同一任务/初态的连通性证据。
- 19337成功转换demo_48（1独立demo、96动作帧、86官方窗口）。环境20Hz与官方SFT插值30Hz分别记录，不混用。
- 19339真实PPO1次更新及post-update rollout成功；actor128020参数、value267777参数，全部5个actor参数组非零梯度和实际变化；初始有效ratio0.995446–1.001728；更新后全state hash/固定noise probe相同；原生DCP扰动后恢复PASS。完整Slurm331秒=.091944GPUh，actor峰值14.562GiB、5秒采样己方compute总21492MiB（不含全部EGL）。本地`.local/xvla-evidence/ppo19339_summary.json`和三个视频已保留。
- 最新已确认的真实依赖测试：19352，提交2862a8c6，**125/125 PASS，无SKIP**。覆盖真实TorchData later-epoch余序列恢复、稳定state审计、概率解析梯度/ratio、默认路径和预算。ff440之后FSDP配置变化的真实运行待19354。
- 19343/19345全胜组方差0，按设计拒绝伪造更新。19348压力策略σpos=.04/rot=.12/grip=.6得到两组[0,0,1,1]、[0,1,1,1]，但第二次FSDP前向view写回失败，不能称GRPO有效更新；采用单根FSDP配置重跑。
- **先读19353（SFT）和19354（GRPO stress）的最终out/err、sacct及checkpoint_check.json，不重复提交。** 最后确认19354 RUNNING，19353已不在队列；结果未知。网络再次中断后用户确认暂时无法恢复。没有凭空登记PASS或停止未知作业，每个作业时限1小时。
- 19346 SFT已到真实backward/optimizer，但原named_parameters审计KeyError而失败；已改canonical完整state+trainable别名，19353为重跑。19347唯一恢复测试使用错误TorchData私有字段assert，已修，19352实际通过。
- 未完成真实验收：C完整SFT/冻结/重载、E独立进程模型恢复及完整新worker恢复、F GRPO完整更新、G新布局完整显存。完整新worker/多GPU/full/LoRA真实训练和研究实验NOT RUN。不得把已充分LIBERO SFT权重称few-shot或把RL已见条件称zero-shot OOD。
- 恢复连接优先使用`tools/xvla/remote.ps1 session`的单连接串行诊断；从settings派生FYP变量。所有新训练依旧提交Git固定worktree并经sbatch，最多2卡（含pending预约的保守预算），不连接他人的Ray。
