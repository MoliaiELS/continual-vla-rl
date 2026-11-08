# X-VLA 小规模接入操作手册

本手册对应固定 RLinf `1bf78980d9339cbc67497eddceb9a1e3a74e2acc` 和固定 X-VLA 源码 `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`。实际完成度以 [接入审计](xvla_integration_audit.md) 的逐项状态和日志为准。本次只运行接入 smoke；不要把原 OpenVLA 的 `tools/remote.ps1 experiment-submit` 用于 X-VLA。

## 工作流与路径

在本机仓库编辑、检查并提交，经 Git 推送 `origin/codex/xvla-integration`；远端仅运行固定 commit worktree。当前环境不能直接从根目录导入旧 `rlinf`。作业脚本将 PYTHONPATH 顺序设为 pinned `third_party/rlinf`、本仓库、bundled LIBERO，实际导入路径会打印并验证。

所有远端位置从 `tools/idac/settings.psd1` 读取：

| 用途 | 来源 |
| --- | --- |
| SSH | `SshHost`，当前 `idac_sever` |
| 固定代码 | `RunCodeRoot/<commit>`，submodule 固定上游 commit |
| 新 Conda | `CondaRoot/envs/xvla_rlinf`，不覆盖旧 `vla_budget_rl` |
| 新大文件根 | `DataRoot/xvla` |
| 模型 | `<新大文件根>/models/X-VLA-Libero--129e71460678`、`X-VLA-Pt--c1c4a64a7e03` |
| 安装证据 | `<新大文件根>/evidence/requirements.freeze.txt`、`pip-check.txt`、`bootstrap.complete` |
| 提交收据 | `<新大文件根>/runs/submission-<job>.json` |
| 作业日志 | `<新大文件根>/logs/<job>.out`、`<job>.err` |
| 运行证据 | `<新大文件根>/runs/<job>/` |

模型/数据/结果不进入 Git。作业不 source 其他项目 bash.sh，不使用 dino_wm/dsrl_pi0 的环境或数据。

## 1. 审计与同步

在仓库根目录 PowerShell 中执行；显式 `powershell.exe -ExecutionPolicy Bypass -File` 仅影响本次进程，不修改系统执行策略。

```powershell
git branch --show-current
git status --short
git submodule status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 audit

# 提交本次明确选择的文件；保留用户其他未提交内容。
git add <本次需要同步的文件>
git commit -m "Implement pinned X-VLA integration"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 sync
```

`sync` 拒绝 tracked/staged 未提交修改，推送当前 HEAD 到 X-VLA 分支，在远端建立 detached worktree 并初始化固定 submodule。它不 reset 用户工作树、不 force-push、不切换 main/latest。新增运行时补丁在作业 worktree 内应用，不改变 submodule 的基线 commit。

## 2. 独立依赖与固定权重

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 bootstrap
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 download -Model libero
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 download -Model pt
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 status
```

bootstrap 只在新 prefix 安装 `tools/xvla/requirements.txt`，缓存和证据写个人存储。它运行 `pip check`，失败不创建 complete 标记。完整安装版本以 freeze 为准。下载固定 model repo/revision，只取权重与处理器资产，核对 LFS SHA256 并写 `verified.json`，不执行 HF custom Python。不把空模型目录视为下载完成。

安装与下载是 CPU/I/O 操作，可以使用登录节点；所有仿真、长评估、训练统一走下面的 sbatch 入口。不要在登录节点直接运行 alignment、convert 或模型生成。

## 3. 按阶段提交有界 smoke

```powershell
# CPU 单元测试；真实Torch来自专用环境，Ray Cluster在配置测试中被逻辑替身替代。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit unit -Gpus 0

# 输出实际python/rlinf/transformers/peft身份。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit imports -Gpus 0

# 固定输入/权重/fp32/噪声：processor、raw动作、最终7维转换。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit alignment -Gpus 1

# 同原生协议的真实仿真；前项通过后再运行。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit native_eval -Gpus 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit stochastic_eval -Gpus 1

# controller验证的1条独立demo转换，随后Pt起点真实SFT一步。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit convert -Gpus 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit sft -Gpus 1

# 先PPO通过真实闭环，再GRPO；不得同时把未通过项登记PASS。
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit ppo -Gpus 1
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 submit grpo -Gpus 1
```

`submit.py` 固定1小时、8 CPU、排除登录节点、申请明确4090卡，并在提交前核查本用户当前 GPU 占用，超过同时2张预算则拒绝。模型/仿真/SFT/PPO/GRPO smoke 要求恰好1GPU，`-Gpus 2` 会显式失败：当前 `actor,env,rollout:all` 是单GPU验收配置，增加GPU将改变worker数和GRPO组分布，不能直接扩展。提交前仍需检查现有 pending/running 作业与个人资源；资源不足应等待或减少独立作业，不停止其他人的作业。

convert 默认只抽样1条有效独立演示，自动定位**本项目** raw dataset 下的 task0 HDF5，验证 Hugging Face 下载 receipt 的固定 revision/SHA256，拒绝覆盖已有 converted manifest。manifest 存于 `<新大文件根>/datasets/limited-spatial-task0/manifest.json`。需要其他输出目录/2demo/seed时使用 `submit.py` 明确参数并保留提交收据，不能把帧数当作demo数。转换需真实 delta controller 回放并验证最终成功。

当前 SFT 配置使用 Pt 初始化，而 PPO/GRPO 接入 smoke 默认使用 Libero 初始化；它们是两个独立验收入口。尚未验证的 Pt→该次SFT checkpoint→RL 链不能据此宣称已完成数据效率实验。

## 4. 不打断查询与证据

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 status
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\xvla\remote.ps1 logs 19300 # 替换为提交返回的真实job ID
```

查询只读，不向作业发送中断。`logs` 查看 out/err 尾部；完整日志及视频留在运行证据目录。`XVLA_JOB_PASS` 仅在命令成功返回后打印。不能只看到模型加载或shape正确就把真实仿真/训练登记通过。

重要验收证据：

- `alignment`：processor逐tensor严格一致、第三视角 image_mask=false、raw action和环境动作误差、真实 horizon 与 dtype。
- `unit.log`：解析概率/梯度、history/reset、参数范围、补丁幂等/拒绝覆盖、注册hook和真实配置validate；SKIP逐项列为NOT RUN。
- `XVLA_TRAINABLE`：真实名称、数量、shape、dtype及domain rows。
- `XVLA_RECOMPUTE`：更新前缓存与重算同z/q、max difference阈值、归约后的ratio≈1。
- `XVLA_AUDIT` 与 `audit-<pid>.jsonl`：真实reward/advantage、actor-transformer梯度、参数变化、完整statehash及固定probe同步、实际checkpoint扰动后恢复。
- `XVLA_SFT_AUDIT` 与 `sft_worker_rank*.jsonl`：真实SFT backward、冻结组hash不变、trainable变化与原生save/load验证。
- 资源证据：全作业 wall time、GPU-hours、峰值显存、采样/更新/同步时间、实际policy/settling步、episodes；以作业日志与资源文件为准，未记录不填估计值冒充实测。

少量全胜/全败的 GRPO 组会产生零相对优势；必须报告组内方差和保留比例，不能伪造 advantage 或改奖励掩盖。smoke未提升成功率不等于接入失败；smoke成功也不代表方法有效。

## 5. 运行时补丁与隔离

```bash
# 只在对应固定worktree执行；--check不写文件。
python tools/xvla/apply_runtime_patch.py --runtime third_party/rlinf --check
python tools/xvla/apply_runtime_patch.py --runtime third_party/rlinf
```

helper验证精确HEAD、patch哈希和每个受影响文件before/after哈希；已应用则幂等返回，遇到用户修改或部分应用时拒绝，不自动reset。6个上游文件改动均默认关闭。训练 mode开关必须同时位于 `actor.model.pass_predict_mode` 和独立评估的 `rollout.model.pass_predict_mode`，因为上游worker根据 only_eval选择model_cfg。

`job.sh` 设置 `RLINF_XVLA_ISOLATED_RAY=1`，只启动本作业 local Ray；object store有界2GiB，CPU来自Slurm，GPU来自未被修改的 `CUDA_VISIBLE_DEVICES`，`RAY_TMPDIR` 和 spill 路径均包含job ID。禁止全局 `ray stop`。本集群 RealMemory调度配置异常；不申请错误的64G/`--mem=0`独占，不声称内存已被可靠隔离。

## 6. 继续研究前的门槛

训练使用 measured proprio，原生评估使用官方 autoregressive history。绝对控制、图像、Rotate6D、夹爪转换和完整chunk不可悄悄替换；所有配置变更先记录协议差异。当前训练300步、评估800步、settling10步；transport810只是容纳最后不完整chunk，不代表执行810步。固定上游GAE将termination/truncation都作为结束，不在当前有限时域后bootstrap，chunk reward求和与gamma时间尺度见审计。

只有A–H真实证据齐备后，才为更长预算、多任务、更多seed、SFT/PPO/GRPO与参数范围比较另行冻结协议。两个研究起点分别保留Pt有限示教SFT和已LIBERO-SFT的Libero泛化路径，明确RL留出、SFT未见和扰动OOD的差异。本次不开展完整规模实验、不声称SAC/IQL已支持、不用充分SFT权重冒充few-shot。
