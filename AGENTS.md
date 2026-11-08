# FYP：共享 VLA 强化后训练

## 当前分支：X-VLA 接入准备

- `codex/xvla-integration` 从基线 `f2965db7a1b0c6313ed1ded2575a5ea515e0aea4` 建立；当前阶段实际实现并小规模验证 X-VLA。入口为 `docs/xvla/README.md` 和 `CODEX_HANDOFF.md`。
- 新上游为独立远端 `rlinf-upstream`（`RLinf/RLinf`），参考 commit 固定为 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc`。原 `upstream` 仍指 Simple Recipe 作者仓库，不改其含义。
- 根目录运行时保持 Simple Recipe；新增 `third_party/rlinf` submodule 固定为上述上游 commit，X-VLA 使用该运行时及显式最小补丁，不升级 main/latest。`docs/xvla/reference` 仍是只读参考。每项接入验收以实际日志为准。
- 本分支通过 Git 推送到 `origin/codex/xvla-integration`；`tools/idac/settings.psd1` 的同步分支已单独设置。现有 `experiment-submit`、模型与 Conda 配置仍属于 OpenVLA 基线，不能用于启动 X-VLA。
- 原 `fyp/deploy-base` 分支与已提交的远端作业继续作为基线。后续 X-VLA 实现需要独立配置、固定上游版本和模型 revision、专用环境及新的验收记录。
- 本分支的 X-VLA 准备优先于下述基线部署顺序；服务器使用规范、Git 工作流、真实更新验收和研究计数要求继续适用。

## 已有基线约束

- 当前优先部署 Simple Recipe 的 `UT-Austin-RobIn/continual-vla-rl`，基准 commit `dc1b1c8a7fb630c8d9aaf349376ae5a49b575b4e`。本目录名字 RLinf 不代表使用最新上游 RLinf。
- 本机修改代码，经 Git 推送到 `MoliaiELS/continual-vla-rl` 的 `fyp/deploy-base`；远端仅运行已提交版本。不要混入 DSRL 的实验方案或修改其环境。
- SSH 使用 `idac_sever`。先读 `docs/deployment.md` 和 `tools/idac/settings.psd1`，通过 `tools/remote.ps1` 操作。
- 必须遵守 `C:/Users/moliai/Desktop/fyp/IADC server for USER.docx`：代码、独立 Conda 在个人 homes；权重、pip 缓存和结果在个人存储目录；长评估、仿真和训练经 sbatch，排除登录节点；不安装代理，不改系统或公共环境，不停止其他人的 Ray/作业。Conda 不放 NFS 存储盘，避免小文件解包与导入瓶颈。
- Slurm 的 GPU 可见性必须保留。当前节点 RealMemory=1MB 是集群配置问题，不能据此申请 64G；读取实际 RAM，使用有界 Ray object store，记录内存未被调度器可靠隔离。
- 每个作业使用固定 commit worktree，显式初始化 Conda 和本项目环境，核实 transformers/prismatic/libero/rlinf 的实际导入位置。
- 第一阶段 LIBERO-Spatial、同一个作者 SFT snapshot、rank32 LoRA、embodied_grpo。不擅改奖励、模型、chunk 或 rank 来掩盖故障。先环境与 SFT，再真实更新、保存和重载，之后才两任务 Seq/Joint。
- 配置合成、随机动作仿真、模型加载和真实有效 RL 更新是不同验收级别。全胜/全败组不能证明有效 RL 更新。失败必须保留非零退出码。
- 联动研究时记录真实控制步、settling、post-done、有效训练步、组保留率、逐任务 ever/final；外层 global_step 不等于环境步。未参与本轮 RL 不代表 SFT 从未见过。
- 不提前启用回放、MoE、新奖励或最终留出评估。短测仅用于部署，不报告为性能复现。
