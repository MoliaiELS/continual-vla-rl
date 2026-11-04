#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Use tools/remote.ps1 experiment-submit}"
[[ "$(hostname)" != 4090node1 ]] || { echo 'Login node refused' >&2; exit 1; }
source "${FYP_JOB_REPO:?Missing immutable code checkout}/tools/idac/env.sh"
cd "$FYP_REPO"
git diff --quiet && git diff --cached --quiet
python tools/idac/prepare.py
suite="${1:?Missing suite directory}"
key="${2:?Missing task key}"
export RAY_TMPDIR="/tmp/fyp_${USER}_${SLURM_JOB_ID}"
export FYP_RAY_SPILL_DIR="$suite/ray-spill/$SLURM_JOB_ID"
export FYP_RAY_ISOLATED=1 FYP_RAY_OBJECT_STORE_BYTES=2147483648
mkdir -p "$RAY_TMPDIR" "$FYP_RAY_SPILL_DIR"
unset RAY_ADDRESS
case "$key" in
    train_*) export FYP_NUM_GPUS=2 FYP_SPLIT_TRAIN=1 ;;
    eval_*|oodcheck) export FYP_NUM_GPUS=1; unset FYP_SPLIT_TRAIN ;;
    *) export FYP_NUM_GPUS=0; unset FYP_SPLIT_TRAIN ;;
esac
if [[ "$FYP_NUM_GPUS" != 0 ]]; then
    export FYP_ALLOCATED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm GPU allocation required}"
    IFS=, read -r -a allocated_devices <<< "$FYP_ALLOCATED_CUDA_DEVICES"
    [[ "${#allocated_devices[@]}" == "$FYP_NUM_GPUS" ]]
    render_device="${allocated_devices[-1]}"
    if [[ "$render_device" =~ ^[0-9]+$ ]]; then export MUJOCO_EGL_DEVICE_ID="$render_device"; fi
fi
python tools/idac/experiment.py run-task "$suite" "$key"
echo "FYP_EXPERIMENT_TASK_OK $key $SLURM_JOB_ID"
