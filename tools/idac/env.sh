#!/usr/bin/env bash
# Source this file in each process; never source the DSRL environment.
export FYP_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export FYP_DATA_ROOT="${FYP_DATA_ROOT:-/mnt/slurmfs-4090node3/user_data/yguo704/RLinf}"
export FYP_CONDA_ROOT="${FYP_CONDA_ROOT:-/mnt/slurmfs-4090node1/homes/yguo704/miniforge3}"
export FYP_ENV_PREFIX="${FYP_ENV_PREFIX:-$FYP_CONDA_ROOT/envs/vla_budget_rl}"
export CONDA_PKGS_DIRS="$FYP_CONDA_ROOT/pkgs-vla-budget"
export PIP_CACHE_DIR="$FYP_DATA_ROOT/cache/pip"
export HF_HOME="$FYP_DATA_ROOT/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=30
export FYP_MODEL_PATH="${FYP_MODEL_PATH:-$FYP_DATA_ROOT/models/Openvla-oft-SFT-libero-spatial-traj1}"
export PYTHONUNBUFFERED=1
export TMPDIR="$FYP_DATA_ROOT/tmp"
export XDG_CACHE_HOME="$FYP_DATA_ROOT/cache/xdg"
export NUMBA_CACHE_DIR="$FYP_DATA_ROOT/cache/numba"
export MPLCONFIGDIR="$FYP_DATA_ROOT/cache/matplotlib"
export LIBERO_REPO_PATH="$FYP_REPO/LIBERO"
# Preserve this outside the source tree; launchers honor the explicit value.
export LIBERO_CONFIG_PATH="$FYP_DATA_ROOT/config/libero/$(git -C "$FYP_REPO" rev-parse HEAD)"
export REPO_PATH="$FYP_REPO"
export EMBODIED_PATH="$FYP_REPO/examples/embodiment"
export PYTHONPATH="$FYP_REPO:$FYP_REPO/transformers-openvla-oft/src:$FYP_REPO/openvla-oft:$FYP_REPO/LIBERO${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS" MKL_NUM_THREADS="$OMP_NUM_THREADS"
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$LIBERO_CONFIG_PATH" "$XDG_CACHE_HOME" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
source "$FYP_CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$FYP_ENV_PREFIX"
