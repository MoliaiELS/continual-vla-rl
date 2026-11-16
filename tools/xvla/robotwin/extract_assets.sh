#!/usr/bin/env bash
set -euo pipefail
: "${FYP_ROBOTWIN_DATA:?}" "${FYP_CONDA_ROOT:?}"
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tool_dir/runtime.env"
data="$FYP_ROBOTWIN_DATA"
archives="$data/archives/$ASSETS_REV"
tree="$data/sim-assets/$ASSETS_REV"

# The login node mounts this disk over NFS while the worker mounts it locally.
# Hold the existing preparation lock on the login node and signal the worker.
if [[ ${1:-} == lease ]]; then
    job="${2:?Extraction job ID required}"
    [[ "$job" =~ ^[0-9]+$ ]]
    ticket="$data/evidence/extract-$job.lock-ready"
    exec 9>"$data/prepare-assets.lock"
    flock -w 240 9
    trap 'rm -f -- "$ticket"' EXIT
    touch "$ticket"
    while [[ ! -f "$tree/.extracted" ]]; do
        state=$(sacct -X -nP -j "$job" --format=State | awk -F '|' 'NF {print $1; exit}')
        case "$state" in
            FAILED*|CANCELLED*|TIMEOUT*|NODE_FAIL*|OUT_OF_MEMORY*|BOOT_FAIL*) exit 1 ;;
        esac
        sleep 5
    done
    exit 0
fi

: "${SLURM_JOB_ID:?Run asset extraction through Slurm}"
echo "ROBOTWIN_EXTRACT_NODE $(hostname -s)"
findmnt -n -T "$data" -o SOURCE,FSTYPE
ticket="$data/evidence/extract-$SLURM_JOB_ID.lock-ready"
deadline=$((SECONDS + 240))
while [[ ! -f "$ticket" && ! -f "$tree/.extracted" ]]; do
    if (( SECONDS >= deadline )); then
        echo 'No login-node asset lock for this extraction job' >&2
        exit 1
    fi
    sleep 2
done
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
