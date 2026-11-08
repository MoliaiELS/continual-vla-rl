# X-VLA 接入准备与迁移计划

## 当前边界

- 工作分支：`codex/xvla-integration`，从已验证的 Simple Recipe 部署版本分出。
- 现有基线分支：`fyp/deploy-base`。它的实验与环境继续作为独立基线。
- 本文是接口审查和待实现清单，不表示已经迁移新上游、加载 X-VLA 或完成 RL 更新。
- 新上游参考仓库：`RLinf/RLinf`，固定 commit 为 `1bf78980d9339cbc67497eddceb9a1e3a74e2acc`。
- 下文的“上游文件”属于该固定 commit；当前工作目录仍是 Simple Recipe 源码，不能直接导入上游新接口。
- 参考：[上游模型注册与构建](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/__init__.py)、[上游 BasePolicy](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/embodiment/base_policy.py)。

## 旧接口与新接口映射

| 层次 | 当前 Simple Recipe 实现 | 固定新上游实现 | 接入决策 |
| --- | --- | --- | --- |
| 模型选择 | `actor.model.model_name`，工厂 `get_model(model_path, cfg)` | `actor.model.model_type`，工厂 `get_model(cfg)`，模型 builder 读取 `cfg.model_path` | 新配置使用 `model_type: xvla`；不得只给旧配置改键后直接运行 |
| 注册 | [`rlinf/models/__init__.py`](../../rlinf/models/__init__.py) 中条件分支 | `register_model(type, builder, category="embodied")` 动态登记 builder 和 `SupportedModel` | X-VLA 作为 embodied policy，不因使用 flow matching 就登记为视频 diffusion 类别 |
| 模型抽象 | 工作者直接依赖 OpenVLA 的 processor、logits、token 字段 | `BasePolicy` 的 `predict_action_batch`、`default_forward`，经 `forward` 分派 | 新模型提供独立适配层；不能伪造 OpenVLA token/logits 来满足旧接口 |
| 观测 | [`model_utils.py`](../../rlinf/models/embodiment/model_utils.py) 中 `images_and_states` 与固定语言模板 | 环境输出统一为 `main_images`、可选 wrist/extra views、`states`、`task_descriptions` | 在模型边界调用所选 X-VLA 路线的 processor，保留视角、状态与 domain ID |
| 推理结果 | [`huggingface_worker.py`](../../rlinf/workers/rollout/hf/huggingface_worker.py) 返回动作、token、logprob、value | 返回 `(actions, result)`；result 含 `forward_inputs`，按算法记录旧策略统计 | `forward_inputs` 必须足以重建训练所需计算，不只保存最终动作 |
| 轨迹运输 | 平铺的 `action_tokens`、图像与旧概率缓存 | `rlinf/data/schema/embodied_types.py` 中 `PolicyOutput`、`PolicyPart`、`EnvPart`、`TrajectoryStep` | 保留策略版本、执行 mask、终止与截断；新模型自行约定可运输的缓存字段 |
| 训练重算 | [`fsdp_actor_worker.py`](../../rlinf/workers/actor/fsdp_actor_worker.py) 以词表 logits 重算 token 概率 | actor 调模型 `forward_inputs=...`，按配置指定 `compute_logprobs/entropy/values` | GRPO 所需概率必须定义正确；PPO 另需匹配 value 接口 |
| 环境动作 | [`action_utils.py`](../../rlinf/envs/action_utils.py) 对 LIBERO 统一翻转夹爪 | 上游按 `model_type` 只处理指定内置模型；新类型通常原样通过 | X-VLA 适配层返回环境可执行动作，核对没有二次夹爪翻转或重复归一化 |
| 保存与同步 | 当前 FYP 分卡路径强制 LoRA，并检查 LoRA-B | 上游 actor 使用模型 state dict 与 weight syncer；FSDP checkpoint 独立管理训练状态 | soft prompts、投影层等所有允许训练参数都必须保存、同步、重载和验收 |

上游并不要求所有模型使用同一个内部动作表示。
环境动作 `actions` 与训练缓存中的模型动作可以不同，但两者的映射必须明确且可复核。
`PolicyOutput` 的概率粒度由模型与损失约定，不能把 token、环境步和去噪步混为一维。
若启用 `executed_mask`，上游要求对应概率的前两维与 `[batch, executed_chunk]` 对齐，模型重算时也要应用相同 mask。

## X-VLA 开始实现前必须冻结的选择

| 决策 | 当前状态与要求 |
| --- | --- |
| 模型与输入路线 | 待选择具体 checkpoint/revision。官方原版与 LeRobot 集成应分别核查，不能混用 processor、proprio 格式和权重配置 |
| 任务适配起点 | 先选择与 LIBERO 对应的 checkpoint，记录它的 SFT/预训练曝光；未参与本轮 RL 不等于模型从未见过 |
| 动作语义 | 原版 LIBERO 路线涉及绝对 EEF 目标、6D 旋转及内部 padding。转换成 7 维后仍不能当作 OpenVLA 的增量动作 |
| 预测与执行长度 | 分别固定模型预测 horizon、每次执行步数、环境总 horizon、剩余步处理及 mask。当前 FYP runner 的整除检查需要新方案验证 |
| Flow-RL 目标 | 待确定能重算行为概率的随机策略/去噪链方案，或明确采用另一种有依据的 RL 目标。官方监督 loss 不能直接充当动作 logprob |
| 可训练参数 | 待选择 soft prompts、LoRA 或其他明确子集；打印完整参数名称、数量与冻结情况，不默认套用 OpenVLA LoRA targets |
| 回放监督 | 待 RL 闭环通过后再接。原始 LIBERO 示教可转换，OpenVLA token/logit 缓存不能直接复用；采用 X-VLA 对应监督目标 |

对同一条 rollout，旧策略概率和新策略重算必须对应同一套随机变量和概率定义。
若所选方法需要初始噪声、时间步、去噪中间状态或执行 mask，这些信息必须进入训练缓存。
不得在训练时重新采样一组无关噪声，然后把得到的分数当成原动作的概率比。
原始动作密度与扩展去噪链的密度不是可互换的概念，文档和指标需要注明采用哪一种。

## 上游模块的参考位置

- 模型加载：[openpi/__init__.py](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/embodiment/openpi/__init__.py)。它区分 `sft/eval/rl` 等任务子类；用于理解结构，不代表 X-VLA 可直接复用其概率实现。
- Flow-RL 参考：[openpi/tasks/rl.py](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/models/embodiment/openpi/tasks/rl.py)。
- 模型配置示例：[model/pi0_5.yaml](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/examples/embodiment/config/model/pi0_5.yaml)。配置挂载到 `actor.model`；模型路径、精度和任务子配置需核对 worker 的实际合成结果。
- 轨迹与观测：[embodied_types.py](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/data/schema/embodied_types.py)。
- 训练消费者：[embodied_fsdp_actor_worker.py](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/workers/actor/embodied_fsdp_actor_worker.py)。
- 动作转换：[action_utils.py](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/envs/action_utils.py)。新 LIBERO wrapper 位于 `rlinf/envs/sim/libero/`，与本分支目录不同。
- FSDP 包裹：[fsdp/utils.py](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/hybrid_engines/fsdp/utils.py)。可配置 transformer class、module class 和 no-split 目标，不能假设 Florence/动作 Transformer 与 Prismatic 一致。
- 训练状态：[strategy/checkpoint.py](https://github.com/RLinf/RLinf/blob/1bf78980d9339cbc67497eddceb9a1e3a74e2acc/rlinf/hybrid_engines/fsdp/strategy/checkpoint.py)。上游保存模型、优化器、调度器与 rank RNG；任务调度、预算和回放状态仍须另行检查。

## 分阶段验收

### 阶段 0：固定设计与独立运行身份

1. 固定新上游 commit、X-VLA checkpoint/revision、processor 来源与依赖版本。
2. 固定动作控制器、采样器、预测/执行 chunk、训练参数集合和概率定义。
3. 为新路线创建独立配置与环境身份记录；不要导入当前 bundled Prismatic 后误认为正在运行 X-VLA。
4. 当前状态仅为本阶段的准备，下面各阶段均待实施。

### 阶段 1：官方模型基线

1. 先用所选官方实现，在开发任务上核实 checkpoint 能正常执行。
2. 保存输入图像、状态、domain ID、控制器与动作转换记录。
3. 记录成功率及初态分布；低于预期时先排查输入/控制器，不用 RL 掩盖适配故障。
4. 此阶段只证明原生策略可用，不证明 RLinf 接入或真实更新成立。

### 阶段 2：冻结权重的 RLinf rollout

1. 用相同观测和固定随机输入，对照官方与适配层的预处理和动作结果。
2. 验证 batch、dtype、相机顺序、proprio、动作单位、旋转和夹爪。
3. 验证每个执行步的 done、truncation、remaining chunk mask 与预算计数。
4. 对照开发任务的 rollout 表现；保留失败轨迹和非零退出码。

### 阶段 3：真实 RL 更新

1. 参数未更新时，rollout 记录与 actor 重算的 logprob 应在声明的数值误差内一致。
2. 构造/采集真实 mixed-success group，记录组保留率与有效训练步；全胜/全败不证明有效更新。
3. 单独验证 RL loss 的非零梯度，证明允许训练的参数确实改变。
4. 检查概率粒度、mask、归一化及 clip ratio；不能仅凭有限的 loss 数值宣布接入成功。

### 阶段 4：保存、同步与独立重载

1. actor 与 rollout 的可训练参数键、数值和策略版本一致。
2. 保存所有选择训练的 prompts/adapter/projection/head 及对应配置。
3. 新进程独立加载同一 base 和适配参数，对照固定输入的动作及概率。
4. 此验收仍不等于优化器和任务进度的完整恢复。

### 阶段 5：长训练与持续学习

1. 验证中断前后优化器、调度器、RNG、任务顺序、预算和回放状态。
2. 先做单任务递增预算，测有效样本/小时、显存、CPU buffer 与存储增长。
3. 再接 Seq/Joint；RL 更新通过后才接 ER，并验证独立的 BC 梯度。
4. 最后冻结多任务、多种子与 OOD 协议；不得把接入短测写成论文性能复现。

## 配置、环境与服务器工作流

- Git 开发仍在本机完成，服务端只运行已提交版本的固定 worktree。
- 新分支的推送/部署目标应显式隔离，不能通过现有默认入口将接入变更推送到 `fyp/deploy-base`。
- X-VLA 依赖建立独立 Conda，位于个人 homes；不替换现有 VLA 环境，不修改 DSRL 环境。
- 权重、数据、缓存和结果保留在个人存储目录；不复制到 Git 或登录盘。
- 仿真、评估和训练继续经 sbatch，排除登录节点，保留 Slurm GPU 可见性。
- 同时两张 4090 的既有上限继续有效；新模型接口准备不构成扩卡或提交新实验的授权。
- Ray/CPU buffer 继续有界，记录 Slurm 内存未可靠隔离的现实限制；不根据节点展示的 1MB 申请虚假内存配额。
- 当前 core/stress 的 12/24 小时配置属于旧路线，迁移后按实测设置时限；长训练前先验收恢复，不能仅延长 walltime。

## 当前完成与待办

- 已完成：新分支、固定上游来源的接口审查、旧新映射与验收清单。
- 待完成：运行时迁移/适配、具体 X-VLA 配置与环境、概率目标、原生对照、真实更新和重载。
- 后续实现应逐项附证据，不以“类已注册”或“checkpoint 已加载”替代策略与算法验收。
