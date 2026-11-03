# IADC 部署入口（2026-11-01）

## 固定实现与路径

本项目使用 [Simple Recipe 作者仓库](https://github.com/UT-Austin-RobIn/continual-vla-rl/tree/dc1b1c8a7fb630c8d9aaf349376ae5a49b575b4e)，原始 commit 为 `dc1b1c8a7fb630c8d9aaf349376ae5a49b575b4e`。本地源码核对后建立 `fyp/deploy-base`，origin 为 `MoliaiELS/continual-vla-rl`。Fork 的 main 就是这个 commit，第一阶段不需要其他分支；bundled LIBERO / OpenVLA-OFT / Transformers 是普通受控目录，不是 submodule。

| 用途 | 路径 |
| --- | --- |
| Windows 开发 | `C:/Users/moliai/Desktop/fyp/RLinf` |
| SSH | `idac_sever` |
| 远端 Git 管理目录 | `/mnt/slurmfs-4090node1/homes/yguo704/RLinf` |
| 固定运行代码 | `/mnt/slurmfs-4090node1/homes/yguo704/RLinf-runs/<commit>` |
| 大文件根目录 | `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf` |
| 独立 Conda prefix | `/mnt/slurmfs-4090node1/homes/yguo704/miniforge3/envs/vla_budget_rl`，Python 3.11.10 |
| 模型 | `<大文件根>/models/Openvla-oft-SFT-libero-spatial-traj1` |
| 作业输出 | `<大文件根>/logs/<job>.out`、`.err` |
| 运行证据 | `<大文件根>/runs/<mode>_<job>/` |

集中配置见 `tools/idac/settings.psd1`。大文件不进入 Git。远端每个作业使用不可变 commit worktree，并通过 PYTHONPATH 显式指向该 worktree 的 bundled 包；doctor 会检查真实导入路径。

## 使用

在本地仓库根目录的 PowerShell 中：

```powershell
./tools/remote.ps1 doctor
# 修改完成后按实际文件 git add / git commit，再同步；拒绝未提交修改。
./tools/remote.ps1 sync
./tools/remote.ps1 bootstrap
./tools/remote.ps1 bootstrap-status
./tools/remote.ps1 model-download # 后台下载固定 SFT snapshot 到个人存储盘，不占 GPU
./tools/remote.ps1 model-status   # 查看下载进度与大小/SHA256 校验结果
./tools/remote.ps1 smoke          # CUDA + 作业独立 Ray + LIBERO/EGL，非策略评估
./tools/remote.ps1 config         # 额外导出训练/评估的 resolved config
./tools/remote.ps1 eval-smoke     # 模型下载完整后，2 条 SFT 回合
./tools/remote.ps1 train-smoke    # SFT 连通性验收后才运行真实 GRPO
./tools/remote.ps1 status 12345
./tools/remote.ps1 logs 12345
```

bootstrap 为安装工作，日志位于 `reports/bootstrap.log`，使用 flock 防止并发安装。环境创建命令：

```bash
export CONDA_PKGS_DIRS=/mnt/slurmfs-4090node1/homes/yguo704/miniforge3/pkgs-vla-budget
export TMPDIR=/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/tmp
/mnt/slurmfs-4090node1/homes/yguo704/miniforge3/bin/conda create \
  --prefix /mnt/slurmfs-4090node1/homes/yguo704/miniforge3/envs/vla_budget_rl \
  --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main \
  python=3.11.10 pip -y
```

安装固定 Linux PyPI Torch 2.6.0，并断言 `torch.version.cuda == 12.4`，避免浮动依赖引入新 CUDA / Transformers API。官方 CUDA wheel CDN 在本次下载中明显较慢，使用已测速的清华 PyPI 镜像；临时解包在个人 `/tmp/fyp_install_<user>`，下载缓存仍在存储根。OpenVLA-OFT 的浮动 Git Transformers 不执行，明确安装仓库自带版本；其余依赖与固定 dlimp 统一解析。`pip check`、完整 freeze 和包来源报告为验收项；未安装 FlashAttention 不应宣称使用 FA2，需实测 backend。原始 requirements 保留，部署子集及版本见 `tools/idac/requirements.txt`。

依赖审查还发现两项实际约束：固定 dlimp 要求 TensorFlow **2.15.0**；TensorFlow Addons 0.23 要求 typeguard `<3`，而 Tyro 0.9.31 要求 `>=4`。本 LIBERO/GRPO 路径未引用 Tyro，因此部署子集不安装它，保留 TF Graphics/Addons 所需的 typeguard 2.13.3。不要把原始浮动 requirements 直接当成可复现 lockfile。

## 服务器规范对方案的实际修订

依据用户指定的 `IADC server for USER.docx`。代码只能放个人 homes；大文件转到存储节点。登录节点用于安装与调试；超过 30 分钟的 GPU 任务会被清理，所有本项目 GPU 检查也统一走 sbatch。只管理自己的文件和明确 job ID，不安装代理、不改驱动、不用全节点 `ray stop`。

- 2026-11-01 登录盘约剩 47GB，存储节点约剩 4.3TB。初试 Conda 放存储盘，但 NFS 小文件解包约 8 分钟仍未完成 Python 基础包；因此独立 Conda 改放个人 homes 的本地盘，控制安装体积，pip 下载缓存、模型和结果仍放存储根。Conda 小缓存为 `miniforge3/pkgs-vla-budget`，已下载的基础包直接复用。
- 可见 Slurm 分区为 `GPU`，3090/4090；A100 是单独申请的渲染资源，不在本账号 Slurm 分区。默认请求 1 张 4090、排除登录节点，让 Slurm 选择空闲的计算节点；需要时可在 settings.psd1 固定 Node。本次在 4090node3 完成环境验证，单卡 GRPO 显存可行性仍须 profile。
- 节点 `RealMemory=1`、`SelectTypeParameters=CR_CORE`：`--mem=64G` 会导致不可调度，因此暂不声明内存请求。作业记录物理 RAM 与可用 RAM；这不构成 64GB 内存预留或 cgroup 限额，需管理员修正后才能可靠预约。不得使用 `--mem=0` 独占节点。
- 保留 Slurm 的 CUDA_VISIBLE_DEVICES，Ray 只登记 1 GPU、8 CPU、2GiB object store，溢写到本次结果目录；独立 `/tmp/fyp_<user>_<job>` 与 `address=local`，不附着已有共享 Ray。2GiB 是工程起点，不是训练已验证最小值。
- HF 主站当前连接被 reset；先核对合法镜像，不安装代理。模型下载必须固定 snapshot revision 并核对文件，不能把空目录算下载完成。

## 验收与研究接口

作者 Spatial SFT 固定 snapshot 为 `39e5240e879c80b6cda6b3a83763dad717f5c05d`。HF 主站仍连接重置，下载经 `https://hf-mirror.com`，不安装代理。`model.snapshot.json` 记录仓库、revision、端点、文件大小和 LFS SHA256；全部文件大小与 LFS 哈希验证通过后才标记 verified。评估/训练作业会检查该标记及文件大小。权重约 15.08 GB，位于个人存储盘。

按 [Notion 部署说明](https://app.notion.com/p/409d26c2767d81a49702cd9763c738dd) 分阶段验收：环境身份 → SFT 闭环 → 有效 GRPO 更新 → 保存与独立重载 → 两任务等预算 Seq/Joint。

本轮小配置保留 K8/G1/R1/H512/chunk8、rank32、温度1.6、micro1/global64；日志 TensorBoard、单周期保存、不自动评估。低层 launcher 使用数组传参和 pipefail；评估尊重外部 LOG_DIR 与项目 LIBERO_CONFIG_PATH。

基础环境通过并不证明 GRPO 已有效更新。必须检查 mixed-success group、非零 RL 梯度、LoRA 张量改变、old/new logprob 自一致，以及相同 base 的 adapter 重载。当前 adapter 加载不是完整 optimizer/RNG/预算恢复。

研究联动先作为待验证问题：GRPO 全败/全胜组都消耗交互，但不提供组内相对优势。任务调度的效率指标应考虑 `groups_kept / policy_env_steps`，同时记录各任务成功率和组内奖励方差；不能据此直接推断新方法有效。默认环境成功后仍可能继续模拟，实际控制步、有效训练步、settling 和 post-done 必须分开统计；正式等预算基线前实现计数器。

第一批不启用回放和最终留出。Spatial SFT 对 suite 有既往曝光，任务 7–9 即使不参与本轮 RL，也只能称本轮 RL 留出。任务采样与预算器在真实更新通过后接入，避免同时改部署和算法。
