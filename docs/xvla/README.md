# X-VLA 接入准备

更新：2026-11-08。分支：`codex/xvla-integration`。

本阶段已经建立独立分支、固定新上游参考版本并梳理模型契约。当前运行时代码仍为 Simple Recipe；模型加载、X-VLA rollout 和 RL 更新均尚未接入。文档中的模板是后续实现设计，不是可执行训练配置。

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
git show rlinf-upstream/main:rlinf/models/embodiment/base_policy.py
# 为重复核查使用固定 SHA，而不是会移动的 main：
git show 1bf78980d9339cbc67497eddceb9a1e3a74e2acc:rlinf/models/embodiment/openpi/tasks/rl.py
```

新远端的名称保存在本机 Git 配置中，其他机器需要单独添加：

```powershell
git remote add rlinf-upstream https://github.com/RLinf/RLinf.git
git fetch --filter=blob:none --no-tags rlinf-upstream main
git cat-file -e '1bf78980d9339cbc67497eddceb9a1e3a74e2acc^{commit}'
```

`tools/remote.ps1 sync` 在本分支指向新的同步分支；本阶段没有同步服务器或改变运行作业。原 OpenVLA 的模型目录、Conda 和实验命令仍不是 X-VLA 的部署方案。

后续先确定上游运行时采用方式和 X-VLA checkpoint，再实现官方推理对齐。大规模 RL、经验回放及持续学习实验在相应验收之后开展。
