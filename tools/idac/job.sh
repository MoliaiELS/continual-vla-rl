#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through tools/remote.ps1}"
[[ "$(hostname)" != 4090node1 ]] || { echo 'Login-node execution refused'; exit 1; }
# Slurm copies this script into its spool directory, so $0 is not the checkout.
source "${FYP_JOB_REPO:?Missing submitted commit directory}/tools/idac/env.sh"
mode="${1:-env}"
export LOG_DIR="$FYP_DATA_ROOT/runs/${mode}_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
cd "$FYP_REPO"
git diff --quiet && git diff --cached --quiet
python tools/idac/prepare.py
export FYP_MODEL_PATH="${FYP_MODEL_PATH:-$FYP_DATA_ROOT/models/Openvla-oft-SFT-libero-spatial-traj1}"
if [[ "$mode" == eval || "$mode" == train || "$mode" == reload ]]; then
    python - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ["FYP_MODEL_PATH"])
manifest = json.loads((Path(os.environ["FYP_DATA_ROOT"]) / "reports/model.snapshot.json").read_text())
assert manifest["verified"], "Download and verify the fixed SFT snapshot first"
assert manifest["revision"] == "39e5240e879c80b6cda6b3a83763dad717f5c05d"
for entry in manifest["files"]:
    path = root / entry["name"]
    assert path.is_file() and path.stat().st_size == entry["size"], str(path)
print("SFT_SNAPSHOT", manifest["repository"], manifest["revision"])
PY
fi
export RAY_TMPDIR="/tmp/fyp_${USER}_${SLURM_JOB_ID}"
export FYP_RAY_SPILL_DIR="$LOG_DIR/ray-spill"
export FYP_RAY_ISOLATED=1 FYP_NUM_GPUS=1
export FYP_ALLOCATED_CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:?Slurm must assign GPU visibility}"
IFS=, read -r -a allocated_devices <<< "$FYP_ALLOCATED_CUDA_DEVICES"
if [[ "$mode" == train ]]; then
    export FYP_NUM_GPUS=2 FYP_SPLIT_TRAIN=1
fi
[[ "${#allocated_devices[@]}" == "$FYP_NUM_GPUS" ]]
render_device="${allocated_devices[-1]}"
if [[ "$render_device" =~ ^[0-9]+$ ]]; then
    export MUJOCO_EGL_DEVICE_ID="$render_device"
fi
export FYP_RAY_OBJECT_STORE_BYTES=2147483648
mkdir -p "$RAY_TMPDIR" "$FYP_RAY_SPILL_DIR"
unset RAY_ADDRESS
git rev-parse HEAD > "$LOG_DIR/commit.txt"
python -m pip freeze > "$LOG_DIR/requirements.resolved.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv > "$LOG_DIR/gpu_inventory.csv"
python tools/idac/doctor.py > "$LOG_DIR/identity.json"
python tools/idac/gpu_smoke.py "$mode"
case "$mode" in
    env) ;;
    config)
        bash examples/embodiment/run_embodiment.sh crl_experiment/fyp_spatial_smoke --cfg job --resolve > "$LOG_DIR/train.launcher.txt"
        bash examples/embodiment/eval_embodiment.sh crl_experiment/fyp_spatial_eval --cfg job --resolve > "$LOG_DIR/eval.launcher.txt"
        python tools/idac/check_config.py
        ;;
    eval|reload)
        export FYP_DEPLOYMENT_EVIDENCE="$LOG_DIR/evidence"
        overrides=()
        if [[ "$mode" == reload ]]; then
            overrides+=(actor.model.is_lora=True "+actor.model.lora_path=${FYP_ADAPTER_PATH:?Missing training adapter}")
        fi
        bash examples/embodiment/eval_embodiment.sh crl_experiment/fyp_spatial_eval "${overrides[@]}" --cfg job --resolve > "$LOG_DIR/eval.launcher.txt"
        python tools/idac/check_config.py "${overrides[@]}"
        bash examples/embodiment/eval_embodiment.sh crl_experiment/fyp_spatial_eval "${overrides[@]}"
        python tools/idac/verify_sft_rollout.py
        ;;
    train)
        export FYP_DEPLOYMENT_EVIDENCE="$LOG_DIR/evidence"
        bash examples/embodiment/run_embodiment.sh crl_experiment/fyp_spatial_smoke --cfg job --resolve > "$LOG_DIR/train.launcher.txt"
        python tools/idac/check_config.py
        bash examples/embodiment/run_embodiment.sh crl_experiment/fyp_spatial_smoke
        ;;
    *) echo "Unknown mode $mode" >&2; exit 2 ;;
esac
echo "FYP_JOB_OK mode=$mode job=$SLURM_JOB_ID logs=$LOG_DIR"
