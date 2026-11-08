# 新上游 RLinf 的通用模型接口

审核日期：2026-11-08。源码基准：`RLinf/RLinf@1bf78980d9339cbc67497eddceb9a1e3a74e2acc`。这是接入设计文档；当前 Simple Recipe 运行时尚未迁移。

## 1. 先区分类接口与 worker 契约

新上游的 [BasePolicy][base] 只将 `default_forward()` 和 `predict_action_batch()` 声明为抽象方法；ABC 没有规定所有参数、返回字段或张量形状。真正可运行的契约来自 [rollout worker][rollout]、[actor worker][actor] 和 [数据 schema][schema]。

模型既需要 PyTorch 模块能力（参数、设备、state_dict、forward hooks），也需要满足这些调用契约。继承 BasePolicy 是推荐形式，不是注册成功的唯一条件。不能仅通过两个空方法就宣称完成接入。

| 入口 | 职责 | 在本方案中的地位 |
| --- | --- | --- |
| `register_model(model_type, builder, category="embodied", force=False)` | 注册 builder、模型名校验和模型类别 | 新上游扩展入口 |
| `get_model(cfg)` | 按 `cfg.model_type` 查 builder，传入 precision 对应 dtype；可移动设备并应用 LoRA | 框架 factory；不再是旧的 `get_model(path, cfg)` |
| `predict_action_batch(env_obs, ..., **kwargs)` | rollout 采样，返回 `(actions, result)` | worker 必须能实际调用 |
| `forward(forward_type=..., **kwargs)` | 训练统一入口；保持 FSDP / PyTorch forward hooks | actor 使用 `self.model(...)` |
| `default_forward(forward_inputs, **kwargs)` | 在旧轨迹上重算当前策略的 logprob，按需计算 value / entropy | PPO / GRPO 的训练分支 |
| `set_global_step(step)` | 可选地更新采样噪声调度等运行状态 | worker 用 `hasattr` 检查 |
| SFT / SAC / IQL / NFT 等分支 | 对应不同学习目标 | 只实现实际选择的分支 |

`ForwardType` 的枚举包含 DEFAULT、SFT、SAC 等，但 BasePolicy 默认 `forward()` **只分发 DEFAULT**；其他类型抛出 `NotImplementedError`。实现 SFT 或其他算法时要显式补充分发。BasePolicy 也不实现 compile / CUDA graph；初期应关闭这些功能。

如果类写成 `class XVLAPolicy(torch.nn.Module, BasePolicy)`，须显式定义 `forward()`，避免方法解析先命中 `nn.Module` 的未实现 forward。实际运算应经 `model(...)` 进入训练，不能为了调用方便直接绕过 FSDP wrapper 调内部方法。

## 2. 注册、配置与独立进程

[注册实现][registry] 的 builder 接收 `(cfg, torch_dtype)` 并返回模型。新模型使用 `category="embodied"`；动作头采用 flow matching 不意味着应该注册为视频/图像生成的 `diffusion` 类别。`force=True` 会覆盖已有实现，初始接入使用独立名称 `xvla`，避免冒充 OpenPI 或 OpenVLA。

建议后续将适配器放进独立扩展包，例如 `xvla_rlinf_ext`。以下只是接口结构示意，尚无对应运行模块：

```python
def build_xvla(cfg, torch_dtype):
    # 在真正的上游运行时中加载固定 X-VLA checkpoint，并构造 policy。
    raise NotImplementedError("X-VLA builder is not implemented yet")

def register():
    from rlinf.models import register_model
    register_model("xvla", build_xvla, category="embodied")
```

driver 的注册不会自动出现在 Ray worker 独立进程。新上游提供 `RLINF_EXT_MODULE=xvla_rlinf_ext`，在 worker 初始化时导入扩展并执行 `register()`；driver 配置校验/建模之前也必须完成注册。扩展包必须存在于每个固定 worktree 的 import 路径中，不能只在本机安装。

新配置的模型根键为 `model_type`、`model_path`、`precision`、`is_lora`；具体模型的额外配置由 builder 解释。`model_name`、旧 tokenizer 路径和旧 LoRA targets 不能逐字沿用。

| 当前基线配置 | 新上游示例配置 | 迁移时的含义 |
| --- | --- | --- |
| `algorithm.rollout_epoch` | `env.train.rollout_epoch` | rollout 预算移到环境配置，重新核对总交互量 |
| `env.train.num_envs` | `env.train.total_num_envs` | 重新核对全局环境数、分组和每个 worker 的 batch |
| `algorithm.sampling_params` | `rollout.sampling_params` | rollout 与 actor 重算必须读取一致的采样语义 |
| `adv_type/loss_type: embodied_grpo` | 示例为 `adv_type: grpo`、`loss_type: actor` | 名称及实现路径均改变，不宣称算法行为自动等价 |

上游 OpenPI 示例还使用其自身的 horizon、SFT 起点和概率粒度。参考配置用于理解接口，不能替代本项目冻结的奖励、预算、数据曝光与评估协议。

## 3. 观测入口

`env_obs` 是 batched 字典，首维是 batch。`main_images`、`wrist_images`、`extra_view_images`、`states`、`task_descriptions` 是环境与模型接入中常见字段，**不是 BasePolicy 强制的统一输入 schema**。应固定环境输出及所选模型的转换函数后，列明必需字段、相机顺序、图像方向、缩放/裁剪、state 定义和语言处理。

预处理应由 policy 的输入转换层负责，训练重算必须复用相同变换。rollout 只缓存图像却丢弃 domain ID、proprio 或 normalization 身份，会使 actor 无法重现行为策略概率。

## 4. rollout 返回值与轨迹缓存

当前 pinned HF worker 在采集 RL 旧信息时直接访问 `result["prev_logprobs"]` 和 `result["prev_values"]`，并将 `forward_inputs` 放入 PolicyOutput。文档中所谓“字段名可以不同”不意味着能直接给这个 worker 换字段；换字段需要配套修改数据调用方。

| 字段 | 含义与约束 |
| --- | --- |
| `actions` | 已完成模型输出反变换的环境动作；常见 `[B,C,A]`，某些路径为 `[B,C*A]`；dtype / device 必须符合环境 |
| `result["prev_logprobs"]` | 实际采样策略对本次轨迹的旧 logprob；训练所需，数值通常保留 float32 |
| `result["prev_values"]` | 旧 value。GRPO 不要求学习 critic，但通用收集器可能仍要求字段；可参考 OpenPI 的 `[B,1]` 零占位 |
| `result["forward_inputs"]` | 重算所需的 batched tensor 字典；应可切分、合批、传输及移动设备 |
| 其他字段 | 如 OpenPI 的 `model_actions`；按实际 worker/算法消费情况提供 |

令 `B` 为一次请求的环境 batch，`C` 为执行 chunk，`A` 为环境动作维度，`H` 为模型预测 horizon，`D` 为模型内部动作维度。不要把 `C=H` 或 `A=D` 当作通用前提。

[Pi0RL][pi0rl] 是具体示例：缓存 `chains`、`denoise_inds`、tokenized prompt、原始观测、`action` 与 `model_action`；内部预测维度与环境动作维度分开。其旧 logprob 常为 `[B,C,A]`，去噪链维度的选择/归约由模型实现完成。这些字段和形状是 OpenPI 的实现选择，X-VLA 需要自己的定义。

环境侧的 `executed_mask` 标记真正执行的 action slots，与终止、截断及训练有效 mask 不同。动作空间 padding 和未执行的 chunk 尾部不应贡献概率比或损失。计数仍以真实控制步为准，不以采样器迭代数或外层 step 代替。

## 5. actor 训练重算

pinned actor 在 on-policy 分支主要这样调用模型：

```python
output = model(
    forward_inputs=forward_inputs,
    compute_logprobs=True,
    compute_entropy=compute_entropy,
    compute_values=(adv_type == "gae"),
    use_cache=False,
    # 其他按模型类别添加的 kwargs
)
```

actor 读取 `output["logprobs"]`；`values` 和 `entropy` 按算法及配置消费。GRPO 不应为了满足示例输出而额外训练一个无必要的 critic。需要 critic 或 entropy 正则时，必须实现相应语义，不能用占位值掩盖未实现功能。

概率比的基本要求是新旧 logprob 对应同一批观测和已采样动作/潜在路径，粒度及 mask 一致。模型参数不变时，actor 重算应与 rollout 的旧概率吻合。温度、噪声调度、normalization 和 preprocessing 身份也属于一致性检查。

X-VLA 的官方 flow-matching SFT loss 不提供可直接放入 PPO/GRPO 比率的 token logprob。必须先确定 flow-RL 的随机采样与概率定义；若选择去噪链概率路线，保存所需链、噪声步和索引。训练时重新抽一个随机监督时间步并计算 MSE，不能当作对旧采样动作的概率重算。

## 6. 动作进入环境之前

模型 adapter 负责位置/旋转/夹爪、normalization、模型 padding 和内部表示转换。[上游 action_utils][actions] 对 LIBERO 只给若干已知模型执行额外夹爪翻转；已注册的自定义类型通常原样通过。因此 X-VLA 应返回最终环境格式，明确记录是否还有环境侧后处理，防止重复翻转。

绝对位姿和增量位姿即使同为 7 维也不能互换。设置输出格式时须同时核对 LIBERO 控制器配置，保留环境任务和奖励语义。

## 7. 固定版本中的兼容性缺口

注册成功只证明 factory 能找到 builder。pinned HF worker 的 `predict()` 对传入 `mode="train"/"eval"` 使用了模型类型白名单；动态注册的 `xvla` 不会自动进入该名单。

后续必须让 worker 对 X-VLA 显式传递 mode，或引入经过验证的通用 capability/dispatch 机制。不能仅设置 policy 默认 mode 为 eval，否则训练 rollout 可能静默使用评估采样路径。driver、rollout worker、actor worker 都要验证构造与 train/eval 分支。

其他按模型类型分支包括特殊 sampling kwargs、FSDP 包裹策略、PEFT 和动作后处理，也须逐项核对。动态注册不能替代这些运行时检查。

## 8. 参数训练、同步、FSDP 与保存

- 明确可训练参数集合：soft prompts、投影层、action transformer 或 LoRA。`is_lora=False` 本身不保证基座冻结。
- 通用 factory 会在 builder 之后应用配置指定的 LoRA；使用 soft prompts 时避免重复套 PEFT，显式设置 requires_grad。
- FSDP wrap 必须匹配新模型的模块边界；视图、action head 和可训练参数的 dtype、activation checkpointing 单独核实。不要沿用 Prismatic/Llama 的包裹层。
- actor 与 rollout 使用同一 checkpoint/config，权重同步覆盖所有允许训练的参数，接收端核对名称、形状与版本；不能只同步 LoRA-B。
- 保存应覆盖所选适配参数、normalization/processor 身份和模型配置。完整恢复另需 optimizer、scheduler、RNG、任务阶段、交互预算和 replay 状态。
- 新上游的 checkpoint 能力不能自动证明我们的持续学习任务调度已恢复。必须单独测试进程退出再续训。

## 9. 参考实现的目录已发生变化

在本次 pinned commit：

| 职责 | 实际文件 |
| --- | --- |
| registry / factory | `rlinf/models/__init__.py` |
| 基础接口 | `rlinf/models/embodiment/base_policy.py` |
| OpenPI 构造与 task 选择 | `rlinf/models/embodiment/openpi/__init__.py` |
| OpenPI 基础 SFT 模型 | `rlinf/models/embodiment/openpi/pi0.py` |
| OpenPI on-policy RL 采样与概率重算 | `rlinf/models/embodiment/openpi/tasks/rl.py` |
| 评估路径 | `rlinf/models/embodiment/openpi/tasks/eval.py` |
| actor | `rlinf/workers/actor/embodied_fsdp_actor_worker.py` |
| rollout | `rlinf/workers/rollout/hf/huggingface_worker.py` |
| 轨迹数据 | `rlinf/data/schema/embodied_types.py`、`embodied_trajectory.py` |
| 示例模型配置 | `examples/embodiment/config/model/pi0_5.yaml` |

部分公开教程仍写 `openpi_action_model.py`，在本次 snapshot 中不要按该旧路径查找。示例配置的 `openpi.task=rl` 也说明，调用纯 SFT 基础模型不等于获得 RL 接口。

所有参考文件的永久链接与内容哈希见 [upstream-reference.json](upstream-reference.json)。后续工作按 [integration-plan.md](integration-plan.md) 的阶段验收推进。

[base]: https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/embodiment/base_policy.py
[registry]: https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/__init__.py
[rollout]: https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/workers/rollout/hf/huggingface_worker.py
[actor]: https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/workers/actor/embodied_fsdp_actor_worker.py
[schema]: https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/data/schema/embodied_types.py
[pi0rl]: https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/embodiment/openpi/tasks/rl.py
[actions]: https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/envs/action_utils.py
