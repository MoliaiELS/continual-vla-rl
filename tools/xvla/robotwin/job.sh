#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch}" "${FYP_CONDA_ROOT:?}" "${FYP_ROBOTWIN_DATA:?}" "${FYP_ROBOTWIN_ENV:?}" "${FYP_ROBOTWIN_CODE:?}"
[[ $(hostname -s) != 4090node1 ]]
mode="${1:?mode}"
shift
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$tool_dir/../../.." && pwd)"
source "$tool_dir/runtime.env"
export FYP_ROBOTWIN_ROOT="$FYP_ROBOTWIN_CODE/RoboTwin-$ROBOTWIN_REV"
export FYP_ROBOTWIN_XVLA="$FYP_ROBOTWIN_CODE/X-VLA-$XVLA_REV"
export FYP_ROBOTWIN_CUROBO="$FYP_ROBOTWIN_CODE/curobo-$CUROBO_REV"
export FYP_ROBOTWIN_MODEL="$FYP_ROBOTWIN_DATA/models/X-VLA-RoboTwin2--${MODEL_REV:0:12}"
export FYP_ROBOTWIN_JOB="$FYP_ROBOTWIN_DATA/jobs/$SLURM_JOB_ID"
export XVLA_RUN_DIR="$FYP_ROBOTWIN_JOB" XVLA_ALLOCATED_GPUS=1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PIP_CACHE_DIR="$FYP_ROBOTWIN_DATA/cache/pip"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
export NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility
export WARP_CACHE_PATH="$FYP_ROBOTWIN_DATA/cache/warp"
unset PYTHONPATH
source "$FYP_CONDA_ROOT/etc/profile.d/conda.sh"
# Some third-party Conda activation scripts read optional undefined variables.
set +u
conda activate "$FYP_ROBOTWIN_ENV"
set -u
test "$CONDA_PREFIX" = "$FYP_ROBOTWIN_ENV"
mkdir -p "$FYP_ROBOTWIN_JOB" "$WARP_CACHE_PATH"
cd "$repo"
if [[ "$mode" == build ]]; then
    exec flock -n "$FYP_ROBOTWIN_DATA/build.lock" bash "$tool_dir/build.sh"
fi

export PYTHONPATH="$repo"
: "${CUDA_VISIBLE_DEVICES:?Slurm did not assign a visible CUDA device}"
test -s "$FYP_ROBOTWIN_DATA/evidence/build.complete"
test "$(git -C third_party/rlinf rev-parse HEAD)" = "$RLINF_REV"
rm -f "$FYP_ROBOTWIN_JOB/vulkan-device.txt"
python -u "$tool_dir/job_runtime.py" "$mode"
export FYP_ROBOTWIN_VULKAN_DEVICE="$(cat "$FYP_ROBOTWIN_JOB/vulkan-device.txt")"
python -u "$repo/tools/xvla/resource_monitor.py" &
monitor_pid=$!
trap 'kill -TERM "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" || true' EXIT
python -u "$tool_dir/panel.py" "$mode" "$@"
