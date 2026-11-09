# X-VLA 随机策略：terminal_clean_gaussian_v1

版本身份：X-VLA 源码 `6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2`；RLinf 基线 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc`。

本文描述本项目实际实现的独立 RL sampler。它不是原生 X-VLA 动作边缘密度，不是 π₀ 的 velocity sampler，也不声称复现 ReinFlow。原生 `native_eval` 保留官方算法和 autoregressive proprio/history，另行做计算一致性对照。

## 数学定义与采样变量

设观测为 o，机器人域为 d，模型预测长度 H，执行前缀长度 C≤H。全部内部维度为 20，单臂有效输出为 10（位置3、Rotate6D6、夹爪logit1），环境动作为7。

每次决策先独立抽取 `z ~ Normal(0,I)`，形状 `[B,H,20]`。`z` 是与参数 θ 无关的辅助变量。完整张量必须保存；即使最终只执行前 C 步、只使用第一只手，其余时间/维度仍可能经 self-attention 影响执行动作。

使用固定源码的 clean-action 递推得到 `fθ(o,d,z)`：

```text
y = 0
for i = K,...,1:
    t = i/K
    x = t*z + (1-t)*y
    y = predictorθ(preprocess(x), observation, proprio, domain, t)
mu = y[:C, :10]
```

没有把 clean prediction 改作 velocity，没有 Euler 速度积分。RL 实现保留 autograd，不调用带 `no_grad` 的官方 `generate_actions()`；rollout 与训练重算均关闭 dropout/DropPath，并采用相同权重存储、autocast 和输入预处理。

`actor.model.xvla.checkpoint_transformer:true` 对 actor 重算的每次 transformer 调用使用非重入激活重算。反向时重新计算中间激活，保留完整 clean-action 递推的梯度路径、缓存的 `z` 和实际采样的 `q`。固定噪声单元测试逐项比较启用前后的 logprob 和梯度。

接着对实际执行前缀的10维 raw output 加独立终端高斯噪声：

`q[c,j] = mu[c,j] + sigma[j] * epsilon[c,j]`，`epsilon ~ Normal(0,I)`。

这是一条有明确密度的**新探索策略**，并非事后用 Gaussian 拟合已经生成的动作。sigma 是配置固定的正数，初始建议位置 `.003`、旋转6维 `.01`、夹爪logit `.15`，单位随官方 absolute EEF 表示；它们不是测得的最优参数。当前没有可训练 noise head。

完整扩展策略为 `p(z,q|o)=p(z) pθ(q|o,z)`。实现保存并重算 `log pθ(q|o,z)`；`p(z)` 与 θ 无关，在 old/new likelihood ratio 中严格抵消。不得把仅初始噪声的概率当作 actor 的可训练概率。

每个动作行返回：

`l[c] = sum(j=0..9) [-0.5*((q[c,j]-mu[c,j])/sigma[j])² - log sigma[j] - 0.5 log(2π)]`。

返回 tensor 为 `[B,C,1]`，float32。最后一维是10个 raw 坐标的**求和**，不是环境动作前7维的概率，不是平均值。

一个执行 chunk 的严格联合条件概率是 `sum_c executed_mask[c]*l[c]`，先 mask 再 sum；PPO/GRPO 比率为 `exp(sum(new_l-old_l))`。`strict_joint_logprob()` 提供独立可测的归约函数。RLinf 对此模型必须配置 `logprob_action_dim=1`，防止按环境维度7错误 reshape；chunk-level 归约必须是 sum。只有实际调用路径与该粒度一致后才验收 RL。

## 末端后处理与夹爪

q 的第9坐标是夹爪 logit，经过 sigmoid，再经官方阈值/符号转换成为离散环境夹爪控制。Rotate6D 的正交化与转 axis-angle 也属于确定性映射。动作变换可以非一一映射，因此不能把 q 的密度称为最终7维动作密度。

本策略是在扩展采样变量空间上进行 policy gradient，不对这些确定性推送映射虚构 Jacobian，也不对夹爪阈值后的 ±1 假设一个连续 Gaussian 密度。原生动作密度与这里的扩展策略密度需分别记录。

latent 不 clip；任何环境边界限制是固定且记录的执行映射，不能通过 clip 后的动作反推 q 来重算概率。actor 使用缓存的原始 q。

## 预测后缀、padding、执行 mask 与历史

RL **训练**强制 `history_mode=measured_state`，独立于 native_eval 官方 autoregressive history。下一次 RL 观测只能取执行后的实测状态，不得使用未执行 q 后缀更新 history。单独的 `stochastic_eval_smoke` 使用 `eval_history_mode=native_history` 和完整 chunk，在同原生历史协议下观察终端噪声的影响；此评估模式不用于 actor 概率重算或策略梯度。

在给定完整 z 和观测的条件下，终端 q 的各行/各维独立；不会执行且不进入任何后续状态/历史的 q 后缀可以被精确边缘化。保留完整 z，删去无因果影响的终端 q 后缀，与删去影响前缀的 z 后缀不同。

环境提前终止后未执行的 q 行通过 executed_mask 排除；它们不得在后续隐状态继续影响策略。环境内部20维 padding不作为终端随机动作，但初始 z 的20维始终保存。若后续改为 q 跨时间耦合噪声或基于完整 q 更新 history，必须重新推导 mask 与概率口径，现实现不支持这种配置。

entropy 是终端条件 Gaussian 的解析 entropy（10维求和），不是 sigmoid/离散环境动作的 entropy。当前 sigma 固定，entropy 不提供 actor 均值梯度；不要把它作为已实现的自适应探索噪声训练。

## 数值、概率与梯度验收

`tests/xvla/test_sampler.py` 包含：独立标量 Gaussian 解析例子；解析均值 score 梯度；固定噪声 rollout/recompute ratio=1；独立 clean-action 递推对照；未执行时间和 inactive-arm z 对前缀的耦合检查；mask 后 sum；reward/advantage 符号翻转导致 actor 梯度翻转；拒绝7维概率、非法std/history等输入。

这些 CPU 测试只证明采样器的数学与单元逻辑，不证明模型权重、LIBERO 仿真、在线 PPO/GRPO 更新或算法有效。最终 PASS/FAIL/NOT RUN/BLOCKED 与对应日志由 `docs/xvla_integration_audit.md` 汇总。

## 在线运行必须额外记录

- native_eval 与该随机策略在同一开发任务/初态分布上的成功数、失败数及噪声敏感性；小 smoke 不作为性能结论。
- ratio 的归约粒度、执行 mask、有效控制步、bootstrap 与 GAE 时间尺度。
- reward-dependent actor 梯度，不能只有 value head 或 AdamW weight decay 变化。
- 全循环显存峰值、采样/更新/同步时间、GPU-hours、episodes 和实际 env steps。
