# X-VLA 实现交接

2026-11-08 开始实际接入。此前仓库没有 CODEX_HANDOFF.md，本文件新增以保存实施状态。

- 当前分支 `codex/xvla-integration`，启动时 HEAD `bdaa3568`，dirty status 为空，无既有 submodule 或 X-VLA adapter。用户原有提交全部保留。
- 根目录仍是 Simple Recipe 基线。实际新接口运行时位于 `third_party/rlinf`，git submodule 固定 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc`。
- X-VLA 源码固定 `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`。已下载审计源到本机 `.local/xvla-source`，vendor 与 adapter 使用 `xvla_rlinf` 命名空间。
- 原生评估权重：`2toINF/X-VLA-Libero@129e71460678b7236cee6fc9707f09d9fa0c3590`；后续受控 SFT 起点：`2toINF/X-VLA-Pt@c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`。
- 服务器配置来自 `tools/idac/settings.psd1`，入口 `tools/xvla/remote.ps1`，新独立 prefix 为其 CondaRoot 下 `envs/xvla_rlinf`；新结果位于其 DataRoot 下 `xvla`。
- 已审计旧环境 Python3.11.10 / Torch2.6.0 / transformers4.40.1 / PEFT0.11.1 / timm0.9.10。与 X-VLA 的 transformers<=4.51.3、PEFT0.17.1 不一致，因此只在独立克隆环境安装 pin，原环境不变。
- 2026-11-08 核实旧作业19316为 OpenVLA-OFT 五任务 Joint+ER 大规模对照，用户条件授权后已取消；checkpoint、日志保留，取消凭据在个人DataRoot/xvla/evidence。新 smoke 仍遵守最多2卡上限，不能停止他人作业或超配。所有仿真/训练通过 Slurm。
- 实施中：adapter/native/SFT、终端 clean-action 条件高斯新策略、RLinf 默认关闭的最小环境/模式/概率维度补丁。新 RL sampler 不冒充 native 概率；native_eval 保留官方循环与历史。
- 验收明细最终写 `docs/xvla_integration_audit.md`，采样器定义 `docs/xvla_rl_sampler.md`，操作手册 `docs/xvla_runbook.md`。未执行真实项标 NOT RUN 或具体 BLOCKED，不能把 mock 当真机/仿真训练。
- 独立环境安装、pip check、固定 LIBERO 权重验证完成。19330为57项真实依赖单元PASS；19331固定噪声官方循环对齐PASS（raw最大误差0）；19333扩大为77项单元/配置/进程注册测试PASS。原生仿真首次19332配置继承失败，19334暴露上游NVML忽略Slurm可见范围的枚举问题，均保留非零退出码，正以默认关闭补丁修复。不能把这两次失败算真实rollout通过。
