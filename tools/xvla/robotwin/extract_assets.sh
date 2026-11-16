#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?Run asset extraction through Slurm}" "${FYP_ROBOTWIN_DATA:?}" "${FYP_CONDA_ROOT:?}"
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tool_dir/runtime.env"
data="$FYP_ROBOTWIN_DATA"
archives="$data/archives/$ASSETS_REV"
tree="$data/sim-assets/$ASSETS_REV"

echo "ROBOTWIN_EXTRACT_NODE $(hostname -s)"
findmnt -n -T "$data" -o SOURCE,FSTYPE
exec 9>"$data/prepare-assets.lock"
flock -w 240 9
if [[ -f "$tree/.extracted" ]]; then
    echo ROBOTWIN_ASSETS_READY
    exit 0
fi

"$FYP_CONDA_ROOT/envs/vla_budget_rl/bin/python" - "$tool_dir" "$archives" "$ASSETS_REV" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from download_assets import ARCHIVES, REPO

root, revision = Path(sys.argv[2]), sys.argv[3]
receipt = json.loads((root / "verified.json").read_text())
files = {row["file"]: (row["bytes"], row["sha256"]) for row in receipt["files"]}
if receipt["repo_id"] != REPO or receipt["revision"] != revision or files != ARCHIVES:
    raise SystemExit("Unexpected RoboTwin archive verification receipt")
for name, (size, _) in ARCHIVES.items():
    if (root / name).stat().st_size != size:
        raise SystemExit(f"Verified archive size has changed: {name}")
PY

mkdir -p "$tree"
for asset in embodiments objects background_texture; do
    echo "ROBOTWIN_EXTRACT_START $asset"
    # Overwrite entries in case the previous extractor stopped mid-write.
    unzip -q -o "$archives/$asset.zip" "$asset/*" -d "$tree"
    test -d "$tree/$asset"
    echo "ROBOTWIN_EXTRACT_COMPLETE $asset"
done
touch "$tree/.extracted"
echo ROBOTWIN_ASSETS_READY
