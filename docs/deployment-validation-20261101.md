# 基本环境部署验收（2026-11-01）

**基本环境已部署并验证通过。** 当前边界是依赖、调度、CUDA、Ray/RLinf worker、LIBERO/EGL 和配置入口。模型权重下载、SFT 闭环、GRPO 更新和 adapter 重载属于后续阶段。

## 运行证据

| 项目 | 实际结果 |
| --- | --- |
| 上游固定起点 | `UT-Austin-RobIn/continual-vla-rl@dc1b1c8a7fb630c8d9aaf349376ae5a49b575b4e` |
| 开发分支 | `MoliaiELS/continual-vla-rl:fyp/deploy-base`；main 保持原样 |
| GPU 验证 commit | `d5982a21662e93262cb000568329a0b2b261e529` |
| Slurm 作业 | **18901 — COMPLETED，ExitCode 0:0，耗时 00:02:10** |
| 计算节点 | `4090node3`，1 张 RTX 4090，8 CPU；未预约独立 RAM 配额 |
| GPU | CUDA_VISIBLE_DEVICES=`0`；24564 MiB；driver `580.178.04` |
| Torch | `2.6.0+cu124`，CUDA runtime `12.4`；真实 GPU 张量运算通过 |
| Ray | `2.49.0`，独立本地集群，仅登记 1 GPU / 8 CPU，object store 2 GiB |
| RLinf worker | 实际 worker 可用同一获配 GPU，逻辑 global_ids=`[0]` |
| LIBERO | Spatial task 0 / init state 0；初态加载、显式 10 步零动作 settling、EGL 图像通过 |
| 图像 | 256×256×3，像素标准差 50.9568；人工检查场景正常 |
| 实际 launcher | 训练、评估均以 `--cfg job --resolve` 完成；没有进入训练 main |

日志末尾包含 `FYP_JOB_OK mode=config job=18901`。另有受控失败测试：伪 Python 返回 23，训练与评估 launcher 均保留 23，带空格的参数及外部 LIBERO_CONFIG_PATH 保持完整。单独验证过物理 GPU 2 到逻辑 GPU 0 的编号映射；本次真实作业获配的是物理 GPU 0。

GPU 作业之后完善了干净 YAML 输出与版本约束文件。纯配置合成生成的 YAML 与同次作业真实 launcher 输出的配置逐项一致，原始含启动提示的输出保存在 `.launcher.txt`。这些整理没有改变算法或已安装依赖版本。

## 环境与目录

- 独立 Conda：`/mnt/slurmfs-4090node1/homes/yguo704/miniforge3/envs/vla_budget_rl`，Python **3.11.10**，安装体积约 **9.2 GiB**。
- 代码管理目录：`/mnt/slurmfs-4090node1/homes/yguo704/RLinf`。
- 固定运行目录：`/mnt/slurmfs-4090node1/homes/yguo704/RLinf-runs/<commit>`。
- 模型、数据、pip 下载缓存、日志与结果根目录：`/mnt/slurmfs-4090node3/user_data/yguo704/RLinf`。
- 已核实 Transformers、prismatic、LIBERO、RLinf 的实际导入位置属于本次指定 worktree。editable 安装登记的路径另记在原始 pip freeze；运行使用显式 PYTHONPATH 并检查实际来源。

主要版本：Transformers **4.40.1（bundled fork）**，PEFT **0.11.1**，robosuite **1.4.1**，MuJoCo **3.3.7**，NumPy **1.26.4**，TensorFlow **2.15.0**。TensorFlow、TFDS、dlimp、TF Graphics 等数据接口均导入通过。`pip check` 返回 `No broken requirements found.`

全部第三方版本见 `tools/idac/constraints.resolved.txt`，bootstrap 会使用这些约束。bundled 包从当前 Git checkout 安装；原始完整 freeze 保留在远端，包含其 editable 位置。版本约束没有 wheel hash，不宣称逐字节构建一致。

## 方案的实际调整

1. Fork 的 main 已是指定上游 commit，不需要补 fork 其他分支。源码原先缺少 `.git`，已保留文件并建立对应历史与独立开发分支。
2. A100 不在本账号可见的 Slurm 分区；改用 4090 计算节点。节点内存登记为 1 MB，因此不提交无法满足的 `--mem=64G`，并明确物理可用 RAM 不是已预约额度。
3. NFS 存储盘解包 Python 基础包约 8 分钟仍未完成；Conda 改放个人 homes，本地盘完成安装。大文件及 pip 下载缓存仍放存储盘，临时 wheel 解包使用个人 `/tmp/fyp_install_yguo704`。
4. 浮动依赖不能直接作为安装锁：固定 bundled Transformers；[dlimp 的固定源码](https://github.com/moojink/dlimp_openvla/blob/040105d256bd28866cc6620621a3d5f7b6b91b46/setup.py) 要求 TF 2.15.0；排除与 TF Addons 的 typeguard 要求冲突且当前路径未引用的 Tyro；补齐 BDDL 未声明的 `future`。
5. 修复 launcher 失败传播和参数引用、Slurm 复制脚本后的路径定位、作业专属 Ray、GPU 编号映射、项目专属 LIBERO config。小配置独立保存，避免 Hydra 继承 primary searchpath 的限制。

本轮没有修改 GRPO 损失、奖励、LoRA rank、模型结构或动作 chunk。

## 证据位置与复查

完整结果：`/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/runs/config_18901/`

- `commit.txt`、`identity.json`、`requirements.resolved.txt`、`gpu_inventory.csv`
- `libero_task0.png`
- `train.resolved.yaml`、`eval.resolved.yaml`（干净 YAML）
- `train.launcher.txt`、`eval.launcher.txt`（保留真实入口输出）

Slurm 日志：`/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/logs/18901.out` 和 `18901.err`。本机保留的小型证据副本在 `.local/deployment-20261101/`，不进入 Git。

本地 PowerShell 复查：

```powershell
./tools/remote.ps1 bootstrap-status
./tools/remote.ps1 status 18901
./tools/remote.ps1 logs 18901
# 需要重新运行环境 + 配置验收时，先同步已提交版本：
./tools/remote.ps1 sync
./tools/remote.ps1 config
```

下一阶段可在此环境上开始作者模型 snapshot 下载与 SFT 连通性检查。单卡 4090 的模型显存、实际 attention backend、有效 GRPO 更新及保存重载仍需该阶段单独验收。部署流程见 [deployment.md](deployment.md)；研究与部署背景见 [Notion 双主线](https://app.notion.com/p/409d26c2767d81fa8758efc492635447) 和 [部署方案](https://app.notion.com/p/409d26c2767d81a49702cd9763c738dd)。
