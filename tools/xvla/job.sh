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
uuids=[str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())]
(Path(os.environ['XVLA_RUN_DIR'])/'allocated_gpu_uuids.json').write_text(json.dumps(uuids))
PY
case "$mode" in
 unit) python -m unittest discover -s "$repo/tests/xvla" -v 2>&1 | tee "$XVLA_RUN_DIR/unit.log" ;;
 imports) python -m xvla_rlinf.smoke imports ;;
 alignment) python -m xvla_rlinf.smoke alignment ;;
 native_eval) python -m xvla_rlinf.entry --config-name native_eval_smoke ;;
 stochastic_eval) python -m xvla_rlinf.entry --config-name stochastic_eval_smoke ;;
 convert)
   source_hdf5="${4:?source HDF5}"; converted_dir="${5:?converted output}"
   data_revision="${6:?data revision}"; source_hash="${7:?source SHA256}"
   selection_seed="${8:?selection seed}"; demonstrations="${9:?demonstration count}"
   python -m xvla_rlinf.sft convert --source-hdf5 "$source_hdf5" --output-dir "$converted_dir" \
     --task-suite libero_spatial --task-id 0 --data-revision "$data_revision" \
     --source-sha256 "$source_hash" --seed "$selection_seed" --demonstrations "$demonstrations" \
     2>&1 | tee "$XVLA_RUN_DIR/convert.log"
   python -m xvla_rlinf.sft inspect "$converted_dir/manifest.json" ;;
 ppo|grpo|grpo_exploration) python -m xvla_rlinf.entry --config-name "${mode}_smoke" ;;
 sft) python -m xvla_rlinf.sft_entry --config-path "$repo/configs/xvla" --config-name sft_smoke ;;
 *) exit 2 ;;
esac
if [[ "$mode" == ppo || "$mode" == grpo || "$mode" == grpo_exploration || "$mode" == sft ]]; then
  python -m xvla_rlinf.checkpoint_check --checkpoint-dir "$XVLA_RUN_DIR/${mode}-smoke/checkpoints/global_step_1/actor"
fi
echo XVLA_JOB_PASS
