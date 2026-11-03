#!/usr/bin/env bash
# Wait without holding a GPU, then submit the committed SFT rollout once verified.
set -euo pipefail
source "${1:?Missing fixed checkout}/tools/idac/env.sh"
cd "$FYP_REPO"
exec 9>"$FYP_DATA_ROOT/rollout-launch.lock"
flock -n 9 || { echo 'Another rollout launcher is active'; exit 1; }
report="$FYP_DATA_ROOT/reports/rollout-launch"
trap 'rc=$?; echo "$rc" > "$report.exit-code"' EXIT
rm -f "$report.job-id"
printf 'waiting_for_verified_snapshot\n' > "$report.state"
echo "Waiting for verified SFT snapshot; no GPU reserved. Commit=$(git rev-parse HEAD)"
deadline=$((SECONDS + 10800))
while ! python - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ["FYP_DATA_ROOT"]) / "reports/model.snapshot.json"
try:
    m = json.loads(p.read_text())
except (OSError, ValueError):
    raise SystemExit(1)
raise SystemExit(0 if m.get("verified") and m.get("revision") == "39e5240e879c80b6cda6b3a83763dad717f5c05d" else 1)
PY
do
    if flock -n "$FYP_DATA_ROOT/model-download.lock" true; then
        echo 'Model download has stopped without a verified snapshot' >&2
        echo failed_model_download > "$report.state"
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo 'Snapshot wait exceeded three hours' >&2
        echo failed_timeout > "$report.state"
        exit 1
    fi
    sleep 30
done
git diff --quiet && git diff --cached --quiet
export FYP_JOB_REPO="$FYP_REPO"
mkdir -p "$FYP_DATA_ROOT/logs"
job_id=$(sbatch --parsable --chdir="$FYP_REPO" --partition="${FYP_PARTITION:?}" \
    --exclude=4090node1 --nodes=1 --ntasks=1 --gres="${FYP_GRES:?}" \
    --cpus-per-task="${FYP_CPUS:?}" --time=00:30:00 --job-name=fyp-sft-rollout \
    --output="$FYP_DATA_ROOT/logs/%j.out" --error="$FYP_DATA_ROOT/logs/%j.err" \
    "$FYP_REPO/tools/idac/job.sh" eval)
echo "$job_id" > "$report.job-id"
echo submitted > "$report.state"
echo "SFT_ROLLOUT_SUBMITTED job=$job_id commit=$(git rev-parse HEAD)"
