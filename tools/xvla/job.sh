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
export RAY_TMPDIR="/tmp/xvla_${USER}_${SLURM_JOB_ID}"
export FYP_XVLA_RAY_SPILL="$FYP_XVLA_DATA/runs/$SLURM_JOB_ID/ray-spill"
export LIBERO_CONFIG_PATH="$FYP_XVLA_DATA/config/libero"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TF_NUM_INTEROP_THREADS=1 TF_NUM_INTRAOP_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export XVLA_RUN_DIR="$FYP_XVLA_DATA/runs/$SLURM_JOB_ID"
export XVLA_MODEL_PATH="$FYP_XVLA_DATA/models/X-VLA-Libero--129e71460678"
export XVLA_PT_MODEL_PATH="$FYP_XVLA_DATA/models/X-VLA-Pt--c1c4a64a7e03"
mkdir -p "$XVLA_RUN_DIR" "$FYP_XVLA_RAY_SPILL" "$LIBERO_CONFIG_PATH"
if [[ "$gpus" != 0 ]]; then : "${CUDA_VISIBLE_DEVICES:?Slurm mask missing}"; fi
cd "$runtime"
python "$repo/tools/xvla/apply_runtime_patch.py" --runtime "$runtime"
python -c 'import sys,rlinf,torch,transformers,peft;print({"python":sys.executable,"rlinf":rlinf.__file__,"torch":torch.__version__,"transformers":transformers.__version__,"peft":peft.__version__})'
case "$mode" in
 unit) python -m unittest discover -s "$repo/tests/xvla" -v 2>&1 | tee "$XVLA_RUN_DIR/unit.log" ;;
 imports) python -m xvla_rlinf.smoke imports ;;
 alignment) python -m xvla_rlinf.smoke alignment ;;
 native_eval) python -m xvla_rlinf.entry --config-name native_eval_smoke ;;
 ppo|grpo) python -m xvla_rlinf.entry --config-name "${mode}_smoke" ;;
 sft) python -m xvla_rlinf.sft_entry --config-name sft_smoke ;;
 *) exit 2 ;;
esac
echo XVLA_JOB_PASS
