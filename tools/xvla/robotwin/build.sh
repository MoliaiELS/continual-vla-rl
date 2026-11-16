#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tool_dir/runtime.env"
test "$(git -C "$FYP_ROBOTWIN_CUROBO" rev-parse HEAD)" = "$CUROBO_REV"
git -C "$FYP_ROBOTWIN_CUROBO" diff --quiet HEAD
export CUDA_HOME="$CONDA_PREFIX" TORCH_CUDA_ARCH_LIST='8.6;8.9' MAX_JOBS=2
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.7.8
test -x "$CUDA_HOME/bin/nvcc"
nvcc --version
# Refresh the requirements for this exact checkout if the environment was
# initially prepared from an earlier installation-only commit.
python -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r "$tool_dir/requirements.txt"
wheel_dir="$FYP_ROBOTWIN_DATA/cache/wheels/$CUROBO_REV"
mkdir -p "$wheel_dir"
python -m pip wheel --no-deps --no-build-isolation --wheel-dir "$wheel_dir" "$FYP_ROBOTWIN_CUROBO"
python -m pip install --no-deps --force-reinstall "$wheel_dir"/nvidia_curobo-0.7.8-*.whl
python -m pip check | tee "$FYP_ROBOTWIN_DATA/evidence/pip-check.txt"
python -m pip freeze > "$FYP_ROBOTWIN_DATA/evidence/requirements.freeze.txt"
python - <<'PY'
import torch
import sapien, mplib, curobo
from curobo.wrap.reacher.motion_gen import MotionGen
from curobo.curobolib import geom_cu
print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'sapien': sapien.__file__,
       'mplib': mplib.__file__, 'curobo': curobo.__file__, 'geom_cu': geom_cu.__file__})
PY
printf '%s\n' "$SLURM_JOB_ID" > "$FYP_ROBOTWIN_DATA/evidence/build.complete"
