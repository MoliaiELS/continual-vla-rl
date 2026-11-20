# PPO／GRPO 短测之后启动 Hard

2026-11-20 用户要求先完成另一对话的 PPO／GRPO 测试，再启动原生动作预算 Hard。新 Hard 作业以 `--hold` 提交，保持 `JobHeldUser`，不会因测试间隙 GPU 空闲而启动。负责算法测试的 agent 在全部短测完成后释放该作业即可。

已提交作业 **20965**，提交时间 2026-11-20 22:51，当前 `JobHeldUser`，尚未占用 GPU。调度入口固定在 `0dd3d9c66824490ce8334134bc6b3a8f2670b2e9`；同一入口再次调用核验为同一个编号，未重复提交。旧占位 `20326` 已按所有权和挂起状态核对后取消。独立接续回执记录原 Easy 已完成 6,209/7,000，剩余 791 回合暂缓；没有将其标为已完成。

Hard 使用固定 worker 提交 `bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460` 和已准备的同名输出目录。调度入口为 `tools/xvla/robotwin/submit_hard_native.py`；它只提交原生预算 Hard，并记录单独的 `direct-dispatch.json`。原 Easy 的旧预算扩测暂缓，不将其未完成回合标为已完成。原 `hard_native_dispatch.py` 的接续路径仍要求先补齐旧 Easy，不能用于这次直接启动。

启动配置为一张 RTX 4090、16 个 CPU、每次分配 48 小时，排除登录节点。全部 50 项按各自原生上限 400–1,700 动作运行，首批 5,000、重点任务扩测 2,000，累计 7,000。开始后先运行 Hard 独立串行／双进程短测：两任务各五个初态，串行和并发分别重放，共 20 次执行，不计入正式分母。通过结果一致性、吞吐与显存余量检查后才启用双进程；每次启动两个不同任务前还会检查获配卡上的可用显存。专家预筛与策略评测不重叠。

本机 PowerShell 或 SSH 客户端使用以下流程；作业编号以本页的最新提交回执为准。

```powershell
# 仅在 PPO 和 GRPO 全部短测完成后释放新的原生预算 Hard 作业。
ssh idac_sever "scontrol release 20965"
ssh idac_sever "squeue -j 20965 -o '%i|%j|%T|%M|%b|%R'"
```

不要恢复旧作业 `20326`，它使用 300 动作预算，现已取消。新作业的调度环境显式带入 RoboTwin 专用 Conda、代码和结果路径，不继承算法短测对 Python 环境的选择。

新协议输出根为 `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/xvla/robotwin-b0/hard-native-budget/bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460/`。`reports/summary.json` 提供正式进度，`pilot/qualification.json` 提供实际单／双进程资格结果，逐任务日志及采样资源在 `robotwin-b0/jobs/<HardJob>/`。运行前只能确认已配置双进程及门槛，不能将尚未执行的短测报告为通过。

48 小时分配超时、被抢占或节点故障后，使用同一个 worker 和输出目录，通过调度入口的 `--retry` 提交新的挂起作业，再释放新编号。已完成回合会由冻结 runner 跳过，初态和正式分母保持原样。技术失败须先排查，不能作为普通策略失败计入分母。
