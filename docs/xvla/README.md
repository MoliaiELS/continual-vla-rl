# X-VLA 接入

更新：2026-11-08。分支：`codex/xvla-integration`。

本分支已实现独立的 `xvla_rlinf` adapter、固定版本 vendor、SFT/PPO/GRPO 入口及原生计算对照。根目录保留 Simple Recipe；X-VLA 使用 `third_party/rlinf` 的固定运行时。实际验收状态和日志见 [接入审计](../xvla_integration_audit.md)，命令见 [运行手册](../xvla_runbook.md)，概率口径见 [sampler 定义](../xvla_rl_sampler.md)。本页其余接口梳理是接入前的参考记录，不代表实际实验已通过。

| 内容 | 入口 |
| --- | --- |
| 新上游通用模型接口、字段、实际调用链与兼容性缺口 | [upstream-model-interface.md](upstream-model-interface.md) |
| 旧新接口映射、X-VLA 决策与阶段验收 | [integration-plan.md](integration-plan.md) |
| 上游文件清单、固定 commit、SHA256 与永久链接 | [upstream-reference.json](upstream-reference.json) |
| 上游 BasePolicy 的原样参考副本及使用边界 | [reference/README.md](reference/README.md) |

## Git 身份

| 身份 | 仓库 / commit |
| --- | --- |
| 原基线 | `MoliaiELS/continual-vla-rl:fyp/deploy-base`，创建分支时为 `f2965db7a1b0c6313ed1ded2575a5ea515e0aea4` |
| 本分支 | `MoliaiELS/continual-vla-rl:codex/xvla-integration`，从原基线派生 |
| 原 `upstream` | `UT-Austin-RobIn/continual-vla-rl`，保持原含义 |
| 新 `rlinf-upstream` | `RLinf/RLinf`，参考固定为 [`1bf78980d9339cbc67497eddceb9a1e3a74e2acc`](https://github.com/RLinf/RLinf/commit/1bf78980d9339cbc67497eddceb9a1e3a74e2acc)，上游提交时间 2026-11-06 |

可以本地查看参考源码，不切换工作树：

```powershell
git show 1bf78980d9339cbc67497eddceb9a1e3a74e2acc:rlinf/models/embodiment/base_policy.py
git show 1bf78980d9339cbc67497eddceb9a1e3a74e2acc:rlinf/models/embodiment/openpi/tasks/rl.py
```

新远端的名称保存在本机 Git 配置中，其他机器需要单独添加：

```powershell
git remote add rlinf-upstream https://github.com/RLinf/RLinf.git
git fetch --filter=blob:none --no-tags rlinf-upstream 1bf78980d9339cbc67497eddceb9a1e3a74e2acc
git cat-file -e '1bf78980d9339cbc67497eddceb9a1e3a74e2acc^{commit}'
```

X-VLA 的入口是 `tools/xvla/remote.ps1`，读取同一份项目服务器 settings，并使用独立 `xvla_rlinf` 环境和数据子目录。原 OpenVLA 的模型目录、Conda 和实验命令仍不是 X-VLA 的部署方案。

本次仅开展接入及小规模验证。大规模 RL、经验回放及持续学习实验在相应验收之后开展。
