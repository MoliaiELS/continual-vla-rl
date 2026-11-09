# X-VLA 接入审计与验收

更新于 2026-11-09。固定权重原生计算对齐、真实 LIBERO rollout、SFT、PPO 和 GRPO 压力配置均通过各自的单卡小测。各项 `PASS` 只覆盖表中列出的验收范围。

## 固定身份与工作边界

| 项目 | 固定身份 / 入口 |
| --- | --- |
| 开发分支 | `codex/xvla-integration`；保留已有 Simple Recipe 历史和用户提交 |
| 新 RLinf 运行时 | `third_party/rlinf` git submodule，`RLinf/RLinf@1bf78980d9339cbc67497eddceb9a1e3a74e2acc` |
| 原目录运行时 | 根目录 `rlinf/` 仍为原 Simple Recipe 代码，不能作为此次新接口的导入来源 |
| X-VLA 源码 | `2toinf/X-VLA@6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2` |
| 原生评估权重 | `2toINF/X-VLA-Libero@129e71460678b7236cee6fc9707f09d9fa0c3590` |
| 受控 SFT 起点 | `2toINF/X-VLA-Pt@c1c4a64a7e03ac5b95c468bf1578f3d03651b53b` |
| 模型扩展 | `xvla_rlinf/`，`register_model('xvla', ..., category='embodied')` |
| 最小上游补丁 | `patches/rlinf-xvla.patch` 与 `patches/rlinf-xvla.json`，基于固定 commit 的逐文件 SHA256 |
| 执行入口 | `tools/xvla/remote.ps1` → 固定 commit worktree → `tools/xvla/job.sh` → 原 RLinf worker/runner |

代码与独立 Conda 放个人 homes；模型、数据、缓存和结果放本项目个人存储根下 `xvla/`。SSH、根目录和分区读取 `tools/idac/settings.psd1`，不使用历史项目路径。本次仅可信接入与小规模验证，未授权扩大为完整研究实验。所有仿真、GPU 检查、训练通过 Slurm，排除登录节点；不 sudo、不修改公共环境、不停止他人的 Ray 或作业。

## 源码与依赖审计

已按固定版本核对 BasePolicy、模型注册、OpenPI RL 接口、HF rollout worker、FSDP actor、环境动作准备、LIBERO 子进程、loss/advantage 归约、runner、权重同步和 checkpoint 调用。实现不是从 `docs/xvla/reference` 导入运行，也没有冒用 `openpi` 或 `openvla_oft` builder。

X-VLA 原 requirements 的 `transformers<=4.51.3`、`peft==0.17.1` 与原 OpenVLA 环境不兼容。部署在新 `xvla_rlinf` prefix 中克隆已有本项目环境，再仅在新 prefix 移除旧 editable `openvla-oft` 并安装 `tools/xvla/requirements.txt`。主要 pin：Torch 2.6.0+cu124、torchvision 0.21.0、transformers 4.51.3、PEFT 0.17.1、timm 1.0.12、NumPy 1.26.3。独立环境 pip check 已通过；完整观察版本在 `tools/xvla/environment.lock.txt` 及远端 `evidence/requirements.freeze.txt`。lock 用于已有 clone 的约束，不是完整 Conda 二进制环境描述。克隆中旧 LIBERO 的 editable 元数据未作为 lock 安装源，实际运行通过显式 PYTHONPATH 指向本提交 bundled LIBERO；prismatic 未导入。

vendor 位于明确命名空间 `xvla_rlinf.vendor.xvla`，保留 Apache-2.0 许可证与 `UPSTREAM.json`。本项目必要修订应由该 provenance 文件登记。Hugging Face 只下载指定 revision 的权重/processor 资产，不下载执行 checkpoint 附带的 Python custom code；模型与 processor 明确 `trust_remote_code=False`、`local_files_only=True`。`verified.json` 记录文件 SHA256/LFS 哈希。

独立原生对照位于 `xvla_rlinf/vendor/reference_libero.py`：从固定版本官方 LIBERO client 摘录图像翻转与旋转/夹爪转换，记录原文件和摘录成员哈希。计算对照调用固定 vendor 的官方 `generate_actions`，捕获其 raw action，再与带 autograd 的 adapter 循环比较。这能检测转换和递推差异，但仍需要真实仿真验证控制器、reset 和时序。

## 实际数据流与最小补丁

| 调用位置 | X-VLA 契约 / 默认旧行为 |
| --- | --- |
| driver / worker / checkpoint | `RLINF_EXT_MODULE=xvla_rlinf` 使用固定上游的扩展加载 hook；重复注册不覆盖其他 builder |
| HF rollout mode | 训练读取 `actor.model`，独立评估读取 `rollout.model`；两处均显式 `pass_predict_mode:true`，原模型默认不增加 mode 参数 |
| predict | 返回物理动作 `[B,C,7]`、逐行10维求和的 `prev_logprobs[B,C,1]`、观测条件 `prev_values[B,1]`、真实使用的 `forward_inputs` |
| actor 重算 | 保存完整 `z[B,H,20]`、实际 raw `q[B,C,10]`、processor 输入、当时 proprio；不重采样替代 |
| loss | `action_dim:7` 只表示控制维度；`logprob_action_dim:1` 与 `logprob_type:chunk_level` 将 `[B,C,1]` 严格求和一次 |
| LIBERO 图像 | `xvla_raw_observations:true` 提供 raw main/wrist；adapter `image_orientation:native_raw` 只翻转 main180°。原 helper 默认对两视角翻转，其他模型路径不变 |
| proprio | 从真实 controller 的 `ee_pos` 和 `ee_ori_mat` 前两列构建20维：xyz3、Rotate6D6、零夹爪、零 padding |
| 控制器 | reset/recreate/auto-reset 后 settling 用 delta 零指令，settling 完成设置 `use_delta=False`；缺失所需 controller 属性时显式失败 |
| history/reset | 每环境独立 reset mask。`native_eval` 保留官方首条实测、后续上一完整预测最后一条前9维；RL 训练始终使用实测闭环 |
| 终止/后缀 | 严格模式只执行 active 环境，不执行 post-done 或超过 horizon 的动作；未执行槽位 reward0，发布 `executed_mask[B,C]`，旧 logprob 和重算 logprob 一致 mask 后再 sum |
| Ray | 仅 `RLINF_XVLA_ISOLATED_RAY=1` 时新建本作业 local Ray，object store ≤2GiB、CPU ≤Slurm配额、GPU来自保留的 CUDA mask，临时目录和 spill 独立；不自动连接别人的 Ray |
| GPU 枚举 | 新增第7处 opt-in 补丁，将 NVML 物理枚举限制到 Slurm 原 mask；placement 使用 allocation-local ordinal，CUDA 保留原数字/UUID token；EGL 必须查询实际映射，不猜测未获配卡 |
| 单卡 FSDP | `xvla_ignore_frozen_parameters:true` 仅用于一 rank、FP32、NO_SHARD、use_orig_params、无 FSDP CPU offload 且关闭子模块自动包装的 X-VLA。完整模型先迁到当前 rank 设备，再将冻结参数排除在 FSDP flat parameter 之外。冻结权重仍注册在模型中并纳入全状态校验 |

参数范围为 `full`、`lora_heads`、`prompt_heads`。三类 prompt/action 子树显式纳入，包括 DomainAwareLinear 的 Embedding；RLinf factory 的二次 LoRA 注入被拒绝。只训练domain rows时使用独立selected-row Parameter与冻结完整buffer，其他行不进入optimizer，因此不受AdamW weight decay或历史动量影响。真实已验证的训练范围见下文；不能把小模型scope单元测试外推为所有大模型范围的训练通过。

本次真实 PPO 的 domain3 prompt/action 可训练参数为 **128,020**；另有观测条件 value head **267,777**，合计 **395,797**。五个 action/prompt 参数张量均有非零 actor 梯度、最大元素改变量约 `1e-6`，weight decay=0、entropy bonus=0。DomainRows 将未选行移为 buffer，因此日志的 `total_parameters` 不包含全部冻结 embedding buffer，不能将它直接当作原模型架构参数总量。`full`/LoRA 的真实大模型训练仍未验收。

SFT 调用官方 clean-action 监督及权重，不以 velocity loss 代替。数据转换通过原 LIBERO delta controller 回放，记录 controller 的 absolute goal，再转 Rotate6D；manifest 固定源数据 revision/SHA256、任务、独立 demo ID、抽样 seed、成功/回放校验和有效帧数。训练帧或窗口数不能记为独立 demonstrations。

## 小规模协议与已知限制

| 项目 | 当前 smoke 配置 |
| --- | --- |
| 模型 / dtype | 固定权重、float32，关闭 compile 和 CUDA graph。GRPO 压力验证可对 actor 重算的每次 clean-action transformer 调用启用非重入激活重算 |
| 推理 | domain3、clean-action steps10；chunk30，加载时必须与 checkpoint horizon 核对 |
| 原生 / 噪声对照 | `native_eval_smoke` / `stochastic_eval_smoke`，同初态与800步 horizon，rollout transport810步，严格 mask 最后未执行10槽 |
| PPO | 2个环境、300真实步上限、1次 update、prompt/action domain3 rows，观测条件 value head |
| GRPO | 4个同任务兼容初态环境/group4、独立随机采样、1次 update，group无奖励方差时不得声称有效 RL 更新 |
| PPO/GRPO 更新后 | `val_check_interval:1`，原 runner 同步新权重并执行新的原生 eval rollout |
| 资源 | 每个模型/仿真/更新 smoke 固定1张4090，最多同时2张；`all` placement 不可简单扩为2GPU，launcher拒绝此用法 |

`specific_reset_id:42` 是 suite 中展平的初态索引；任务/初态名称应由真实 resolved config 与仿真日志确认。它不是整个 Spatial benchmark 的性能评估。

PPO 的归约单位是 chunk：奖励在 chunk 内求和，配置 gamma 为 `0.99^30 ≈ 0.7397003734`，lambda 为0.95。它不等价于逐控制步折扣每个奖励。固定上游 GAE 使用 `dones=termination|truncation`；当前300步训练 horizon 被视为有限时域结束，不在 time limit 之后 bootstrap。采集的最终 value 并不改变这个 mask。短 smoke 与800/900步原生协议不能直接比较成功率。

RL sampler 为本项目明确命名的 `terminal_clean_gaussian_v1`，在完整原生 clean prediction 周围对执行单臂10维 raw action 加固定终端 Gaussian 噪声；它不是原生动作边缘密度，也不是 π₀ velocity sampler/ReinFlow 复现。概率、梯度与 mask 定义见 [采样器说明](xvla_rl_sampler.md)。噪声评估保留同一 native history，仅用于同协议性能影响估计；训练使用 measured state。

原生与随机评估即使 seed 相同，终端噪声会消耗额外 RNG，因此不是所有后续 chunk 都严格配对相同 z 的实验。固定输入、固定噪声的计算一致性由 alignment 测试单独验证；少量真实回合只能给出噪声敏感性观察。

## 实际作业与证据

结果根目录为 `/mnt/slurmfs-4090node3/user_data/yguo704/RLinf/xvla`；各作业保存 `logs/<job>.out/.err`、`runs/<job>/` 和提交收据。每行对应其提交 SHA 的不可变 worktree。

| 作业 | 状态 | 实际结果 |
| --- | --- | --- |
| 19330 / 19333 | PASS | 57 / 77 项真实依赖单元测试通过 |
| 19332 / 19334 / 19335 | FAIL，已修复 | Hydra继承、NVML整机枚举、LIBERO首次导入交互提示；没有冒充真实 rollout 成功 |
| 19336 | PASS | 原生评估2/2成功；策略控制150步、settling20步；平均75步；800步上限、成功即停止执行 |
| 19337 | PASS | 固定原始数据revision `f13aa24a3da8c43c7225569f28c562979fa0e35a`，seed1234选中demo_48；96步回放、10步settling，ever/final成功；1条独立demo |
| 19338 | PASS | 同初态、同native history的终端噪声评估2/2成功；151策略步、20 settling；平均75.5步。样本不足以估计性能差异 |
| 19339 | PASS | PPO：真实采样→reward→GAE→actor/critic backward→1次optimizer→同步→新eval→原生checkpoint恢复；训练147策略步、评估149步，均2/2成功 |
| 19340 / 19341 | FAIL，已修复 | 新模型config被旧model/忽略规则漏提交；已加入Git并重跑，不影响19339的已完成证据 |
| 19342 | PASS | 106项测试全部通过、无SKIP；包含解析概率梯度、独立进程注册、默认旧路径、补丁、GPU映射、恢复检查器边界 |
| 19343 | FAIL（有效更新门槛） | GRPO回报[1,1,1,1]，方差0，全胜组1，保留率0；拒绝伪造advantage或声称actor更新 |
| 19344 | PASS | 独立官方参考对照：processor逐tensor严格一致；raw20、sigmoid后、最终7D动作的最大误差均0；FP32/H30/steps10 |
| 19345 | FAIL（有效更新门槛） | 更强独立探索配置σpos=.015/σrot=.05/σgrip=.3，仍[1,1,1,1]；不是有效GRPO更新 |
| 19346 | FAIL，已修复审计 | Pt真实SFT已到backward/optimizer；瞬态named_parameters造成image_projection键缺失；改用原生完整state稳定键，不算本次已通过 |
| 19347 | FAIL，已修复测试 | 115项中1项误断言TorchData私有字段非空；改为核验全部剩余batch与epoch |
| 19348 | FAIL，封装修复待验 | 压力配置σpos=.04/σrot=.12/σgrip=.6，8回合两组真实回报[0,0,1,1]、[0,1,1,1]，方差.25/.1875；第一次backward后第二次FSDP前向view写回失败，未完成optimizer |
| 19352 | PASS | 125项全部通过、无SKIP，含真实TorchData跨epoch余序列、稳定完整state审计、算法白名单及GPU预算 |
| 19353 | FAIL | 88秒，Pt SFT 到达优化步骤后原生完整 state 导出因缺少冻结的 `core.vlm.image_projection` 抛出 AssertionError |
| 19354 | FAIL | 150秒，真实组回报 [0,1,0,0] 和 [1,1,1,1]，方差 .1875 和 0，有信息组比例 50%。单根 FSDP 在 actor backward 触发 CUDA OOM，未完成 optimizer |
| 19405 | FAIL，已修复测试 | GPU 映射测试把 patch manifest 文件数写死为 7，新增 FSDP 补丁后 setUpClass 失败。新增的激活重算 logprob 和梯度等价测试通过 |
| 19406 | NOT RUN | 排队时发现固定版本 wrap policy 对空列表仍可返回包装函数，已取消尚未启动的 SFT，改由 19407 运行显式禁用子模块自动包装的配置 |
| 19407 | PASS | X-VLA-Pt，1条独立demo、86个可用窗口、batch1、1次SFT更新。5个可训练张量均有有限非零梯度和权重变化，冻结状态逐项哈希不变。原生同worker恢复和独立进程严格全状态恢复通过 |
| 19408 | PASS | X-VLA-Libero GRPO 压力配置，真实组回报 [1,0,1,1] 和 [1,1,1,1]，方差 .1875 和 0，保留率50%。5个actor张量均有非零梯度和实际变化。同步后原生评估2/2成功，同worker原生恢复、独立进程严格加载及固定噪声探针通过 |
| 19438 | PASS | `b2fa4b9b` 的Slurm CPU回归，126项全通过、无跳过，包含激活重算等价性和GPU分配映射 |

原生/PPO视频已复制到本机 `.local/xvla-evidence/`；原生视频 `native-19336.mp4`。PPO的机器可读摘要为 `.local/xvla-evidence/ppo19339_summary.json`，保留逐行日志及TensorBoard来源。

## 验收矩阵

| 条目 | 当前状态 | 已有证据 / 尚需证据 |
| --- | --- | --- |
| A 注册、预处理、旋转/夹爪、batch/mask/history/reset | PASS（已测范围） | 19438真实依赖126项全通过，19336验证真实控制器和回合。自动reset、重建分支主要为替身测试 |
| B 固定噪声原生一致性、真实短 rollout | PASS | 19344独立原生processor/raw/7D全部误差0；19336真实2回合成功与视频 |
| C 真实小 batch SFT、冻结组、保存重载 | PASS（19407，单卡prompt/action） | 5个可训练张量有实际变化，冻结组和缓冲区不变。原生同worker恢复与独立进程全状态digest匹配 |
| D 更新前 old/recompute、ratio≈1、解析概率梯度 | PASS（近似数值一致） | PPO 19339有效chunk ratio 0.995446–1.001728，最大chunk logdiff .0045643。GRPO 19408全部80个micro的ratio 0.998784–1.000944。解析Gaussian/score梯度独立例子PASS |
| E 更新后 actor→rollout、checkpoint | PASS（已测范围） | PPO 19339、GRPO 19408完整同步哈希及同噪声探针匹配，主动扰动后同worker原生DCP恢复。19407/19408独立进程严格全状态加载通过，GRPO固定噪声探针误差为0。完整新worker恢复 NOT RUN |
| F 真实在线 reward→advantage→actor梯度→step→sync→新rollout | PPO PASS，GRPO压力配置 PASS | 19339和19408均有五个action/prompt张量的非零actor梯度与实际变化，并执行了同步后的新rollout |
| G 全循环资源与计数 | PASS（已测作业与采样口径） | PPO 19339 Slurm331秒、0.091944 GPUh，actor allocator峰值14.562GiB。SFT 19407 Slurm300秒，获配设备5秒采样峰值4312MiB。GRPO 19408 Slurm391秒，获配设备5秒采样峰值14900MiB，actor allocator峰值约3.55GiB。详见下文 |
| H 相关旧模型/配置回归 | PASS（静态/单元）；旧GPU任务NOT RUN | 19438覆盖默认kwargs、图像、概率维度及注册保留检查。根旧rlinf/examples/legacy工具相对启动SHA无diff |

125项通过对应 `2862a8c647f0ac112adcc9183ba8fcce29c862e1`。`ff440880` 将 GRPO 改用单根 FSDP，19354 越过此前嵌套包装的 view 写回失败点，但反向显存仍不足。`fedcdf2d` 在单根 FSDP 中排除冻结参数，GRPO 另启用 transformer 激活重算。两项重跑沿用原任务、数据、动作长度、奖励和噪声。

19339第一个有效micro逐action误差.0008545，满足当时first-only阈值.001；后续最大.0014477。当前新增第一步optimizer前**每个micro**的chunk ratio ±1%数值门槛，有专门测试。通用actor/ratio=.2998176是20个micro含14个全mask零指标的平均，不能读成有效ratio≈.3；有效6个chunk的比率范围见D。

控制频率在真实demo回放中测为20Hz；官方SFT窗口沿用其30Hz时间轴假设，分别记录，没有将官方插值改成20Hz。视频编码fps不等于控制频率。

19407 和 19408 使用 `fedcdf2d980db9e4a880ed375c468f4e6550a107`，已于 2026-11-09 在 `4090node2` 依次完成。按 Slurm 整作业计时分别为 0.08333 和 0.10861 GPUh。GRPO 的训练采样为8回合、865策略步、80 settling步，更新后评估为2回合、151策略步、20 settling步。actor更新耗时34.71秒，更新前/后同步及探针分别耗时16.55和12.32秒。指标取自 `logs/19408.out`、`runs/19408/audit-*.jsonl` 和 `runs/19408/resources.json`。SFT 审计见 `runs/19407/sft_worker_rank0.jsonl`，两个独立恢复结果均位于各自 `checkpoints/global_step_1/actor/checkpoint_check.json`。

2026-11-08 本地重跑命令：

```powershell
# 本地仅为临时 Hydra/NumPy 审计依赖；未把缺 Torch 的项目记为 PASS。
python -m unittest discover -s tests/xvla -p test_runtime_patch.py -v
python -m unittest discover -s tests/xvla -p test_registration.py -v
```

本地日志：`.local/xvla-runtime-patch-local.log`（22项：20通过、2跳过）；`.local/xvla-registration-local.log`（10项：3通过、7跳过）。完整真实 Torch 测试必须使用 pinned runtime PYTHONPATH 在本项目 Slurm 作业内重新执行。远端作业以 `logs/<job>.out/.err` 和 `runs/<job>/` 为准，失败须保留非零退出码；配置修正后需新提交固定 commit 重跑，旧失败日志保留。

## 后续实验入口与未完成项

1. `X-VLA-Pt → 有限独立示教 SFT → RL`：用于数据效率比较，须固定示教身份/seed/有效数量和相同起点。
2. `X-VLA-Libero → 目标任务或扰动 RL → held-out PRO/Plus`：用于任务提升、技能保持和泛化，须记录 SFT/RL/replay 暴露矩阵。

已充分 LIBERO SFT 的 Libero 权重不能称为 few-shot 起点；RL 已见过的条件不能称为零样本 OOD。本次不运行完整规模、不添加 SAC/IQL，不声称已获得统计显著提升。多GPU FSDP、LoRA范围真实在线更新、独立进程完整恢复、更广任务/OOD协议与长期预算仍须单独验收；未支持的模型/算法分支应明确失败。

两个初始化路径目前以分离的Pt-SFT和Libero-RL配置保留，尚未完成Pt→本次SFT产物→RL串联验收。尤其SFT无value head，而PPO新增value head，直接把SFT full_weights.pt用于严格整policy加载会有缺失键；跨full/LoRA/domain-row表示还需明确的core导出/初始化转换，不能跳过strict检查称为同一起点。这些列为后续未完成项，不把现有独立smoke当作数据效率实验入口已全通。

可跟进的研究观察：同一强SFT基座、简单初态和较小噪声下GRPO组可能全部成功，实际有效RL数据为零；4cm压力噪声才在本次产生混合组。后续应记录组内方差、保留率及探索导致的性能变化，统一协议再比较PPO/GRPO。这不构成技能遗忘、few-shot优势或OOD结论。

## LIBERO-Spatial 配对基线

固定代码 `6a0e7fcbd44ff79bb9487cf2a0aacf3456d21fd1`。每份权重运行任务0–9、各自初态0–4，共50回合。两组逐项匹配任务ID、初态ID、初态SHA256、环境种子和策略噪声种子。使用单张RTX 3090、FP32、原生采样、domain3、30步动作块、10次预测迭代、10步settling和800个实际策略步上限。

| 权重 | 作业 | 成功 | 策略步 | Slurm用时 | 获配设备5秒采样峰值 |
| --- | --- | ---: | ---: | ---: | ---: |
| 官方 X-VLA-Libero | 19443 | 46/50（92%） | 7,860 | 8分16秒 | 4558MiB |
| X-VLA-Pt | 19444 | 0/50（0%） | 40,000 | 30分40秒 | 4558MiB |
| X-VLA-Pt 单步 SFT | 19469 | 0/50（0%） | 40,000 | 33分49秒 | 4558MiB |

两项作业均退出0:0，各完成50回合和500个settling步。官方权重的任务4（木柜顶层抽屉）3/5，任务5（小烤碗上）和任务8（盘子旁）各4/5，其余7项各5/5。Pt每项0/5，全部回合达到800步上限。此结果覆盖预先固定的50个初态。

单步SFT评估19469也退出0:0，加载了通过严格恢复校验的19407产物。训练使用任务0的一条独立示教和一次优化器更新。该任务0/5，其余任务也各0/5。全部50回合达到800步上限，与未微调Pt的任务、初态哈希、种子和二元结果逐项匹配。

证据在 `runs/19443/`、`runs/19444/` 和 `runs/19469/` 的 `benchmark_episodes.jsonl`、`benchmark_summary.json` 和 `resources.json`。
