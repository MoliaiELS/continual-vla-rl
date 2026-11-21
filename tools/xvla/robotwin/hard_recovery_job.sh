#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch}" "${FYP_CONDA_ROOT:?}" "${FYP_ROBOTWIN_DATA:?}" "${FYP_ROBOTWIN_ENV:?}" "${FYP_ROBOTWIN_CODE:?}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && $(hostname -s) != 4090node1 ]]
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
evaluation_root="$(cd "${1:?pinned evaluation root}" && pwd -P)"
worker_root="$(cd "${2:?frozen bcb worker root}" && pwd -P)"
profile="$(cd "${3:?existing Hard output profile}" && pwd -P)"
data="$(cd "$FYP_ROBOTWIN_DATA" && pwd -P)"
[[ $(git -C "$evaluation_root" rev-parse HEAD) == 9baa17360abe242cb36e8f9c78f0aadbe64b3bee ]]
[[ $(git -C "$worker_root" rev-parse HEAD) == bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460 ]]
[[ "$profile" == "$data/hard-native-budget/bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460" ]]
git -C "$evaluation_root" diff --quiet HEAD -- tools/xvla
git -C "$worker_root" diff --quiet HEAD -- tools/xvla/robotwin
git -C "$tool_dir" ls-files --error-unmatch hard_recovery_panel.py hard_recovery_job.sh hard_screen_recovery.py >/dev/null
git -C "$tool_dir" diff --quiet HEAD -- hard_recovery_panel.py hard_recovery_job.sh hard_screen_recovery.py
export FYP_ROBOTWIN_ORCHESTRATION_COMMIT="$(git -C "$tool_dir" rev-parse HEAD)"
source "$evaluation_root/tools/xvla/robotwin/runtime.env"
export FYP_ROBOTWIN_ROOT="$FYP_ROBOTWIN_CODE/RoboTwin-$ROBOTWIN_REV"
export FYP_ROBOTWIN_XVLA="$FYP_ROBOTWIN_CODE/X-VLA-$XVLA_REV"
export FYP_ROBOTWIN_CUROBO="$FYP_ROBOTWIN_CODE/curobo-$CUROBO_REV"
export FYP_ROBOTWIN_MODEL="$FYP_ROBOTWIN_DATA/models/X-VLA-RoboTwin2--${MODEL_REV:0:12}"
export FYP_ROBOTWIN_JOB="$data/jobs/$SLURM_JOB_ID"
export XVLA_RUN_DIR="$FYP_ROBOTWIN_JOB" XVLA_ALLOCATED_GPUS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
export WARP_CACHE_PATH="$FYP_ROBOTWIN_JOB/warp-cache"
unset PYTHONPATH
source "$FYP_CONDA_ROOT/etc/profile.d/conda.sh"
set +u
conda activate "$FYP_ROBOTWIN_ENV"
set -u
[[ "$CONDA_PREFIX" == "$FYP_ROBOTWIN_ENV" ]]
mkdir -p "$FYP_ROBOTWIN_JOB" "$WARP_CACHE_PATH"
cd "$evaluation_root"
export PYTHONPATH="$evaluation_root"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not assign a visible CUDA device}"
python -u tools/xvla/robotwin/job_runtime.py hard-native-budget-recovery
export FYP_ROBOTWIN_VULKAN_DEVICE="$(cat "$FYP_ROBOTWIN_JOB/vulkan-device.txt")"
python -u tools/xvla/resource_monitor.py &
monitor_pid=$!
trap 'kill -TERM "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" || true' EXIT
args=(--evaluation-root "$evaluation_root" --worker-root "$worker_root" --output-root "$profile")
if [[ -n "${4:-}" ]]; then args+=(--boundary-file "$4"); fi
python -u "$tool_dir/hard_recovery_panel.py" "${args[@]}"
