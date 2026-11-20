# PPO／GRPO 短测之后先完成 Easy，再启动 Hard

2026-11-20 用户追加要求：将原 Easy 剩余 791 回合放在 Hard 前；随后确认 PPO／GRPO 暂时结束，可以正常提交执行。最新入口为 `submit_easy_then_hard.py` 与 `easy_then_hard_job.sh`，同一次单卡分配中依次运行 Easy 收尾、真实 7,000 回合完成核验和原生预算 Hard。旧 Hard-only 挂起作业 `20965` 已取消，不能再使用其释放命令。

原 Easy 当前为 6,209/7,000，未完成的 791 回合沿用原 300 动作预算和冻结初态，已完成回合跳过。五任务原生预算重评的 900 回合已经单独完成，不与旧计划合并。

两阶段沿用固定 worker `bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460`，原 Easy 的评测核心仍固定在 `9baa17360abe242cb36e8f9c78f0aadbe64b3bee`。新接续信息写入 Hard 输出目录中的 `easy-first-dispatch.json`、`easy-first-jobs.tsv` 和 `easy_first_handoff.json`，原 `direct-dispatch.json` 及暂停回执保留为历史。不会提前把旧 Easy 标为 7,000 完成。

启动配置为一张 RTX 4090、16 个 CPU、每次分配 48 小时，排除登录节点。全部 50 项按各自原生上限 400–1,700 动作运行，首批 5,000、重点任务扩测 2,000，累计 7,000。开始后先运行 Hard 独立串行／双进程短测：两任务各五个初态，串行和并发分别重放，共 20 次执行，不计入正式分母。通过结果一致性、吞吐与显存余量检查后才启用双进程；每次启动两个不同任务前还会检查获配卡上的可用显存。专家预筛与策略评测不重叠。

提交入口先建立挂起作业，经核对后按用户本次指令正常释放。实际编号以最新提交回执为准：

```powershell
ssh idac_sever "squeue -u yguo704 -o '%i|%j|%T|%M|%b|%R'"
```

不要恢复旧作业 `20326` 或 `20965`。新作业的调度环境显式带入 RoboTwin 专用 Conda、代码和结果路径。两个阶段的全局运行身份和资源采样分别归档到 `jobs/<JobID>/stages/easy/` 与 `stages/hard/`；中断重排队遗留文件先归档到 `stages/interrupted/`，避免覆盖已有证据。

新协议输出根为 `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/xvla/robotwin-b0/hard-native-budget/bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460/`。`reports/summary.json` 提供正式进度，`pilot/qualification.json` 提供实际单／双进程资格结果，逐任务日志及采样资源在 `robotwin-b0/jobs/<HardJob>/`。运行前只能确认已配置双进程及门槛，不能将尚未执行的短测报告为通过。

48 小时分配超时、被抢占或节点故障后，使用同一个接续入口固定提交、worker 和输出目录，通过 `submit_easy_then_hard.py --retry` 提交后续作业。若 Easy 已真实完成 7,000 回合，就跳过该阶段并继续 Hard；否则先补 Easy。已完成回合、初态和正式分母保持原样，技术失败须先排查。
