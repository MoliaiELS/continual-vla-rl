#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/env.sh"
cd "$FYP_REPO"
exec 9>"$FYP_DATA_ROOT/bootstrap.lock"
flock -n 9 || { echo 'Another bootstrap is running'; exit 1; }
export PIP_INDEX_URL=https://pypi.org/simple
python -m pip install 'pip==25.2' 'setuptools==75.8.0' wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r tools/idac/requirements.txt
python -m pip install 'dlimp @ git+https://github.com/moojink/dlimp_openvla.git@040105d256bd28866cc6620621a3d5f7b6b91b46'
# Avoid openvla-oft's floating Git Transformers requirement replacing the vendor copy.
python -m pip install --no-deps -e ./transformers-openvla-oft -e ./openvla-oft -e ./LIBERO
python tools/idac/prepare.py
python -m pip check
python -m pip freeze > "$FYP_DATA_ROOT/reports/requirements.resolved.txt"
python tools/idac/doctor.py
touch "$FYP_DATA_ROOT/reports/bootstrap.complete"
