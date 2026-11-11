#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?sbatch required}" "${FYP_XVLA_ENV:?}" "${FYP_XVLA_DATA:?}"
[[ "$(hostname)" != 4090node1 ]]
repo="$(realpath "${1:?repo}")"; mode="${2:?mode}"; gpus="${3:?gpus}"
runtime="$repo/third_party/rlinf"
test "$(git -C "$runtime" rev-parse HEAD)" = 1bf78980d9339cbc67497eddceb9a1e3a74e2acc
source "$FYP_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$FYP_XVLA_ENV"
export PYTHONPATH="$runtime:$repo:$repo/LIBERO"
export RLINF_EXT_MODULE=xvla_rlinf RLINF_RUNTIME="$runtime"
export HF_HOME="$FYP_XVLA_DATA/cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export RLINF_XVLA_ISOLATED_RAY=1
export RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_TMPDIR="/tmp/xvla_${USER}_${SLURM_JOB_ID}"
export FYP_XVLA_RAY_SPILL="$FYP_XVLA_DATA/runs/$SLURM_JOB_ID/ray-spill"
export LIBERO_CONFIG_PATH="$FYP_XVLA_DATA/config/libero/$SLURM_JOB_ID"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export XVLA_RUN_DIR="$FYP_XVLA_DATA/runs/$SLURM_JOB_ID"
export XVLA_ALLOCATED_GPUS="$gpus"
export XVLA_MODEL_PATH="$FYP_XVLA_DATA/models/X-VLA-Libero--129e71460678"
export XVLA_PT_MODEL_PATH="$FYP_XVLA_DATA/models/X-VLA-Pt--c1c4a64a7e03"
mkdir -p "$XVLA_RUN_DIR" "$FYP_XVLA_RAY_SPILL" "$LIBERO_CONFIG_PATH"
python - "$repo" <<'PY'
import json,os,sys
from pathlib import Path
base=Path(sys.argv[1])/'LIBERO/libero/libero'
paths={'benchmark_root':str(base),'bddl_files':str(base/'bddl_files'),
       'init_states':str(base/'init_files'),'assets':str(base/'assets'),
       'datasets':str(Path(os.environ['FYP_DATA_ROOT'])/'datasets/libero_raw')}
for value in paths.values():
    if not Path(value).is_dir():raise FileNotFoundError(value)
(Path(os.environ['LIBERO_CONFIG_PATH'])/'config.yaml').write_text(json.dumps(paths,indent=2))
PY
if [[ "$gpus" != 0 ]]; then : "${CUDA_VISIBLE_DEVICES:?Slurm mask missing}"; fi
cd "$runtime"
python "$repo/tools/xvla/apply_runtime_patch.py" --runtime "$runtime"
python "$repo/tools/xvla/resource_monitor.py" &
resource_pid=$!
trap 'result=$?; kill -TERM "$resource_pid" 2>/dev/null || true; wait "$resource_pid" || true; exit "$result"' EXIT
python - <<'PY'
import importlib.util,json,os,sys,rlinf,torch,transformers,peft
from pathlib import Path
specs={name:(getattr(importlib.util.find_spec(name),'origin',None)) for name in ['rlinf','transformers','prismatic','libero.libero','xvla_rlinf']}
print(json.dumps({'python':sys.executable,'imports':specs,'torch':torch.__version__,'transformers':transformers.__version__,'peft':peft.__version__}),flush=True)
uuids=[str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())] if int(os.environ["XVLA_ALLOCATED_GPUS"]) else []
(Path(os.environ['XVLA_RUN_DIR'])/'allocated_gpu_uuids.json').write_text(json.dumps(uuids))
PY
case "$mode" in
 unit) python -m unittest discover -s "$repo/tests/xvla" -v 2>&1 | tee "$XVLA_RUN_DIR/unit.log" ;;
 imports) python -m xvla_rlinf.smoke imports ;;
 source_reset_audit)
   python -m xvla_rlinf.source_reset_audit --start-trial 30 --trials 10 --along-offset-m 0.01 ;;
 e1_source_audit)
   python -m xvla_rlinf.e1_source_audit --split "$repo/configs/xvla/protocol_20261110/split.json" \
     --manifest "$FYP_XVLA_DATA/datasets/source-spatial-task0-10/manifest.json" ;;
 source_reset_solvability)
   python -m xvla_rlinf.source_reset_solvability --audit "${4:?physics audit report}" ;;
 source_reset_sft)
   python -m xvla_rlinf.source_reset_solvability --audit "${4:?physics audit report}" \
     --checkpoint "${5:?audited Pt-SFT checkpoint}" ;;
 benchmark_e1_deployed|benchmark_e1_online)
   python -m xvla_rlinf.benchmark --model sft-fewshot --policy-path "${mode#benchmark_e1_}" \
     --start-task "${4:?source task}" --task-count "${5:?task count}" --trials "${6:?trial count}" \
     --start-trial "${7:?start trial}" --checkpoint "${8:?audited B0 checkpoint}" ;;
 benchmark_protocol_dev)
   python "$repo/tools/xvla/protocol_dev.py" --plan "${4:?audited paired source development plan}" ;;
 alignment) python -m xvla_rlinf.smoke alignment ;;
 native_eval) python -m xvla_rlinf.entry --config-name native_eval_smoke ;;
 stochastic_eval) python -m xvla_rlinf.entry --config-name stochastic_eval_smoke ;;
 replay_absolute)
   python -m xvla_rlinf.replay_absolute "${4:?converted manifest}" \
     --output "$XVLA_RUN_DIR/absolute_replay.json" ;;
 sft_action_check)
   python -m xvla_rlinf.sft_action_check --manifest "${4:?converted manifest}" \
     --checkpoint "${5:?SFT actor checkpoint directory}" ;;
 benchmark_libero|benchmark_pt|benchmark_sft_smoke|benchmark_sft_fewshot|benchmark_sft_source10|benchmark_rl_source10|benchmark_sft_sequence_dev|benchmark_rl_sequence_dev)
   benchmark_model="${mode#benchmark_}"
   benchmark_args=(--start-task "${4:?start task}" --task-count "${5:?task count}" --trials "${6:?trials}"
                   --start-trial "${7:?start trial}")
   if [[ "$mode" == benchmark_sft_source10 || "$mode" == benchmark_sft_sequence_dev ]]; then
     benchmark_model=sft_fewshot
   elif [[ "$mode" == benchmark_rl_sequence_dev ]]; then
     benchmark_model=rl_source10
   fi
   if [[ "$mode" == benchmark_sft_* || "$mode" == benchmark_rl_* ]]; then
     benchmark_model="${benchmark_model//_/-}"
     benchmark_args+=(--checkpoint "${8:?SFT actor checkpoint directory}")
   fi
   python -m xvla_rlinf.benchmark --model "$benchmark_model" "${benchmark_args[@]}" ;;
 convert|convert_fewshot|convert_source10)
   source_hdf5="${4:?source HDF5}"; converted_dir="${5:?converted output}"
   data_revision="${6:?data revision}"; source_hash="${7:?source SHA256}"
   selection_seed="${8:?selection seed}"; demonstrations="${9:?demonstration count}"
   convert_args=()
   if [[ "$mode" != convert ]]; then
     [[ "$selection_seed" == 1234 ]]
     if [[ "$mode" == convert_fewshot ]]; then
       [[ "$demonstrations" == 5 ]]
     else
       [[ "$demonstrations" == 10 ]]
       convert_args+=(--include-manifest "$FYP_XVLA_DATA/datasets/fewshot-spatial-task0-5/manifest.json"
                      --verify-absolute)
     fi
     publish_dir="$converted_dir"
     if [[ -e "$publish_dir" || -L "$publish_dir" ]]; then
       echo "Few-shot destination already exists: $publish_dir" >&2
       exit 1
     fi
     converted_dir="$XVLA_RUN_DIR/fewshot-conversion"
   fi
   python -m xvla_rlinf.sft convert --source-hdf5 "$source_hdf5" --output-dir "$converted_dir" \
     --task-suite libero_spatial --task-id 0 --data-revision "$data_revision" \
     --source-sha256 "$source_hash" --seed "$selection_seed" --demonstrations "$demonstrations" "${convert_args[@]}" \
     2>&1 | tee "$XVLA_RUN_DIR/convert.log"
   python -m xvla_rlinf.sft inspect "$converted_dir/manifest.json"
   if [[ "$mode" != convert ]]; then
     python - "$converted_dir" "$publish_dir" <<'PY'
import fcntl
import os
from pathlib import Path
import sys

source, target = map(Path, sys.argv[1:])
target.parent.mkdir(parents=True, exist_ok=True)
with (target.parent / f".{target.name}.publish.lock").open("a") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Few-shot destination already exists: {target}")
    os.rename(source, target)
print(f"XVLA_FEWSHOT_MANIFEST {target / 'manifest.json'}")
PY
   fi ;;
 ppo|grpo|grpo_exploration|grpo_stress) python -m xvla_rlinf.entry --config-name "${mode}_smoke" ;;
 ppo_pt|grpo_pt)
   export XVLA_INITIAL_SFT_CHECKPOINT="${4:?audited few-shot SFT actor checkpoint directory}"
   python -m xvla_rlinf.entry --config-name "${mode}_smoke" ;;
 ppo_source10|grpo_source10|ppo_source10_pool|ppo_e2|ppo_e3)
   export XVLA_INITIAL_SFT_CHECKPOINT="${4:?audited source-10 LoRA checkpoint}"
   rl_mode="${mode%%_source10*}"
   if [[ "$mode" == ppo_e2 || "$mode" == ppo_e3 ]]; then rl_mode=ppo; fi
   env_count=2
   if [[ "$rl_mode" == grpo ]]; then env_count=4; fi
   pool_args=()
   if [[ "$mode" == ppo_source10_pool || "$mode" == ppo_e2 || "$mode" == ppo_e3 ]]; then
     env_count=4
     pool_args=(env.train.specific_reset_id=null '+env.train.task_id_filter=[0]'
                "+env.train.xvla_source_reset_pool.path=${5:?frozen reset pool}"
                "+env.train.xvla_source_reset_pool.sha256=${6:?reset pool SHA256}"
                "+env.train.xvla_source_reset_pool.mode=${7:?reset sampling mode}")
     if [[ "$mode" == ppo_e3 ]]; then
       pool_args+=(runner.max_epochs=2048 runner.max_steps=2048 runner.save_interval=-1
                   runner.val_check_interval=-1 +runner.xvla_control_step_budget=200000
                   '+runner.xvla_control_step_milestones=[50000,100000,200000]'
                   "+runner.xvla_training_seed=${8:?registered training seed}"
                   env.train.video_cfg.save_video=false)
     elif [[ "$mode" == ppo_e2 ]]; then
       pool_args+=(runner.max_epochs=256 runner.max_steps=256 runner.save_interval=-1
                   runner.val_check_interval=-1 +runner.xvla_control_step_budget=25000
                   env.train.video_cfg.save_video=false)
     else
       pool_args+=(runner.max_epochs=2 runner.max_steps=2 runner.save_interval=2 runner.val_check_interval=2)
     fi
   fi
   python -m xvla_rlinf.entry --config-name "${rl_mode}_pt_smoke" \
     "runner.logger.experiment_name=$mode-smoke" actor.model.xvla.update_scope=lora_heads \
     +actor.model.xvla.lora_rank=8 actor.model.xvla.position_std=0.003 \
     actor.model.xvla.rotation_std=0.01 actor.model.xvla.gripper_logit_std=0.15 \
     "env.train.total_num_envs=$env_count" "actor.global_batch_size=$((env_count * 10))" "${pool_args[@]}" ;;
 sft) python -m xvla_rlinf.sft_entry --config-path "$repo/configs/xvla" --config-name sft_smoke ;;
 sft_cont)
   export XVLA_SFT_CONT_CHECKPOINT="${4:?audited B0 checkpoint}"
   train_seed="${5:?registered SFT-cont training seed}"
   [[ "$XVLA_SFT_CONT_CHECKPOINT" == /* && "$train_seed" =~ ^(0|[1-9][0-9]*)$ ]]
   python -m xvla_rlinf.sft_entry --config-path "$repo/configs/xvla" --config-name sft_cont \
     "actor.seed=$train_seed" ;;
 sft_fewshot|sft_source10|sft_source10_lora)
   sft_steps="${4:?bounded SFT step budget}"
   [[ "$sft_steps" == 256 || "$sft_steps" == 1024 ]]
   sft_args=()
   if [[ "$mode" != sft_fewshot ]]; then
     sft_args+=("data.train_data_paths=${5:?converted manifest}" data.manifest_requirements.demonstrations=10
                runner.logger.experiment_name=sft_source10)
   fi
   if [[ "$mode" == sft_source10_lora ]]; then
     sft_args+=("+runner.initial_prompt_checkpoint=${6:?audited prompt-head SFT checkpoint}"
                actor.model.xvla.update_scope=lora_heads +actor.model.xvla.lora_rank=8)
     if [[ "$sft_steps" == 1024 ]]; then
       sft_args+=(actor.micro_batch_size=4 actor.global_batch_size=4)
     fi
   fi
   python -m xvla_rlinf.sft_entry --config-path "$repo/configs/xvla" --config-name sft_fewshot \
     "runner.max_steps=$sft_steps" "runner.save_interval=$sft_steps" "actor.optim.total_training_steps=$sft_steps" \
     "${sft_args[@]}" ;;
 *) exit 2 ;;
esac
case "$mode" in
 ppo|grpo|grpo_exploration|grpo_stress|sft|ppo_pt|grpo_pt|ppo_source10|grpo_source10)
   python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$XVLA_RUN_DIR/${mode}-smoke/checkpoints/global_step_1/actor" ;;
 ppo_source10_pool)
   python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$XVLA_RUN_DIR/${mode}-smoke/checkpoints/global_step_2/actor" ;;
 ppo_e3)
   for steps in 50000 100000 200000; do
     checkpoint_info="$(python - "$XVLA_RUN_DIR" "$steps" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
budget = json.loads((root / "control_step_budget.json").read_text())
if budget["state"] not in ("COMPLETE_VALID", "COMPLETE_NO_SIGNAL"):
    raise RuntimeError("Formal PPO did not reach its control-step budget")
row, = [row for row in budget["milestones"] if row["requested_policy_env_steps"] == int(sys.argv[2])]
checkpoint = Path(row["checkpoint_path"]).resolve()
if not checkpoint.is_relative_to(root):
    raise RuntimeError("Formal PPO checkpoint escaped its Slurm output directory")
print(checkpoint, row["checkpoint_sha256"])
PY
     )"
     read -r checkpoint digest <<< "$checkpoint_info"
     python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$checkpoint"
     eval_dir="$XVLA_RUN_DIR/source_dev/control_step_$steps"
     mkdir -p "$eval_dir"
     XVLA_RUN_DIR="$eval_dir" python -m xvla_rlinf.benchmark --model rl-source10 \
       --policy-path deployed --start-task 0 --task-count 1 --start-trial 0 --trials 30 \
       --checkpoint "$checkpoint" --expected-checkpoint-sha256 "$digest"
   done ;;
 ppo_e2)
   checkpoint="$(python - "$XVLA_RUN_DIR" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1]).resolve()
data = json.loads((root / "control_step_budget.json").read_text())
if data["state"] not in ("COMPLETE_VALID", "COMPLETE_NO_SIGNAL"):
    raise RuntimeError("E2 did not reach its registered budget endpoint")
checkpoint = Path(data["checkpoint_path"]).resolve()
if not checkpoint.is_relative_to(root):
    raise RuntimeError("E2 checkpoint escaped its Slurm output directory")
print(checkpoint)
PY
   )"
   python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$checkpoint"
   dispatch_status=0
   python "$repo/tools/xvla/e2_dispatch.py" advance --repo "$repo" --data "$FYP_XVLA_DATA" \
     --partition "${9:?E2 Slurm partition}" --job-id "$SLURM_JOB_ID" --pool "${8:?E2 condition}" || dispatch_status=$?
   if ((dispatch_status == 75)); then
     echo 'E2 endpoint verified. Successor awaits available Slurm resources.'
   elif ((dispatch_status)); then
     exit "$dispatch_status"
   fi ;;
 sft_cont)
   sft_cont_milestones=(512 1024 2048)
   sft_cont_checkpoints="$XVLA_RUN_DIR/sft_cont/checkpoints"
   for step in "${sft_cont_milestones[@]}"; do
     python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$sft_cont_checkpoints/global_step_${step}/actor"
   done
   for step in "${sft_cont_milestones[@]}"; do
     eval_dir="$XVLA_RUN_DIR/source_dev/global_step_${step}"
     mkdir -p "$eval_dir"
     XVLA_RUN_DIR="$eval_dir" python -m xvla_rlinf.benchmark --model sft-fewshot --policy-path deployed \
       --start-task 0 --task-count 1 --start-trial 0 --trials 30 \
       --checkpoint "$sft_cont_checkpoints/global_step_${step}/actor"
   done ;;
 sft_fewshot)
   python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$XVLA_RUN_DIR/sft_fewshot-smoke/checkpoints/global_step_${sft_steps}/actor" ;;
 sft_source10|sft_source10_lora)
   python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$XVLA_RUN_DIR/sft_source10/checkpoints/global_step_${sft_steps}/actor" ;;
esac
case "$mode" in
 sft_cont|ppo_e3)
   dispatch_status=0
   python "$repo/tools/xvla/e3_dispatch.py" advance --repo "$repo" --data "$FYP_XVLA_DATA" \
     --job-id "$SLURM_JOB_ID" || dispatch_status=$?
   if ((dispatch_status == 75)); then
     echo 'E3 endpoint verified. Successor awaits available Slurm resources.'
   elif ((dispatch_status)); then
     exit "$dispatch_status"
   fi ;;
esac
echo XVLA_JOB_PASS
