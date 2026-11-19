#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch}" "${FYP_CONDA_ROOT:?}" "${FYP_ROBOTWIN_DATA:?}" "${FYP_ROBOTWIN_ENV:?}" "${FYP_ROBOTWIN_CODE:?}"
[[ $(hostname -s) != 4090node1 ]]
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
evaluation_root="$(cd "${1:?pinned evaluation root}" && pwd -P)"
[[ $(git -C "$evaluation_root" rev-parse HEAD) == 9baa17360abe242cb36e8f9c78f0aadbe64b3bee ]]
git -C "$tool_dir" diff --quiet HEAD -- audit_job.sh audit_rollout.py
source "$evaluation_root/tools/xvla/robotwin/runtime.env"
export FYP_ROBOTWIN_ROOT="$FYP_ROBOTWIN_CODE/RoboTwin-$ROBOTWIN_REV"
export FYP_ROBOTWIN_XVLA="$FYP_ROBOTWIN_CODE/X-VLA-$XVLA_REV"
export FYP_ROBOTWIN_CUROBO="$FYP_ROBOTWIN_CODE/curobo-$CUROBO_REV"
export FYP_ROBOTWIN_MODEL="$FYP_ROBOTWIN_DATA/models/X-VLA-RoboTwin2--${MODEL_REV:0:12}"
export FYP_ROBOTWIN_JOB="$FYP_ROBOTWIN_DATA/jobs/$SLURM_JOB_ID"
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
python -u tools/xvla/robotwin/job_runtime.py audit
export FYP_ROBOTWIN_VULKAN_DEVICE="$(cat "$FYP_ROBOTWIN_JOB/vulkan-device.txt")"
python -u tools/xvla/resource_monitor.py &
monitor_pid=$!
trap 'kill -TERM "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" || true' EXIT
python -u "$tool_dir/audit_rollout.py" \
    --pinned-evaluation-root "$evaluation_root" \
    --input-panel "$FYP_ROBOTWIN_DATA/panels/$(git rev-parse HEAD)" \
    --output-root "$FYP_ROBOTWIN_DATA/investigations/cutoff-height-$SLURM_JOB_ID" \
    --robotwin-root "$FYP_ROBOTWIN_ROOT" --xvla-root "$FYP_ROBOTWIN_XVLA" \
    --checkpoint "$FYP_ROBOTWIN_MODEL" --episodes-per-task 2 --record-wrist
