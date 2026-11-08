#!/usr/bin/env bash
set -euo pipefail
: "${FYP_CONDA_ROOT:?}" "${FYP_XVLA_ENV:?}" "${FYP_XVLA_DATA:?}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CONDA_PKGS_DIRS="$FYP_CONDA_ROOT/pkgs-xvla"
export PIP_CACHE_DIR="$FYP_XVLA_DATA/cache/pip"
export TMPDIR="$FYP_XVLA_DATA/tmp"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$FYP_XVLA_DATA/evidence"
# Clone a known project environment without modifying it; new wheels replace only
# files in the new prefix. GPU computation is never performed by bootstrap.
if [[ ! -x "$FYP_XVLA_ENV/bin/python" ]]; then
  "$FYP_CONDA_ROOT/bin/conda" create -y --prefix "$FYP_XVLA_ENV" --clone "$FYP_CONDA_ROOT/envs/vla_budget_rl"
fi
"$FYP_XVLA_ENV/bin/python" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r "$repo/tools/xvla/requirements.txt"
"$FYP_XVLA_ENV/bin/python" -m pip freeze > "$FYP_XVLA_DATA/evidence/requirements.freeze.txt"
"$FYP_XVLA_ENV/bin/python" -m pip check > "$FYP_XVLA_DATA/evidence/pip-check.txt"
touch "$FYP_XVLA_DATA/evidence/bootstrap.complete"
echo XVLA_BOOTSTRAP_COMPLETE
