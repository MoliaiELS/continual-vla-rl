#!/usr/bin/env bash
# Submit a held job without reserving a GPU; release only after snapshot verification.
set -euo pipefail
source "${1:?Missing fixed checkout}/tools/idac/env.sh"
cd "$FYP_REPO"
exec 9>"$FYP_DATA_ROOT/rollout-launch.lock"
flock -n 9 || { echo 'Another rollout launcher is active'; exit 1; }
report="$FYP_DATA_ROOT/reports/rollout-launch"
job_id=""
on_exit() {
    rc=$?
    echo "$rc" > "$report.exit-code"
    if (( rc != 0 )) && [[ "$job_id" =~ ^[0-9]+$ ]]; then
        scancel "$job_id"
    fi
}
trap on_exit EXIT
rm -f "$report.job-id" "$report.exit-code"
git diff --quiet && git diff --cached --quiet
export FYP_JOB_REPO="$FYP_REPO"
mkdir -p "$FYP_DATA_ROOT/logs"
job_id=$(sbatch --hold --parsable --chdir="$FYP_REPO" --partition="${FYP_PARTITION:?}" \
    --exclude=4090node1 --nodes=1 --ntasks=1 --gres="${FYP_GRES:?}" \
    --cpus-per-task="${FYP_CPUS:?}" --time=00:30:00 --job-name=fyp-sft-rollout \
    --output="$FYP_DATA_ROOT/logs/%j.out" --error="$FYP_DATA_ROOT/logs/%j.err" \
    "$FYP_REPO/tools/idac/job.sh" eval)
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Invalid Slurm job ID: $job_id" >&2; exit 1; }
echo "$job_id" > "$report.job-id"
printf 'held_waiting_for_verified_snapshot\n' > "$report.state"
echo "SFT_ROLLOUT_HELD job=$job_id commit=$(git rev-parse HEAD); no GPU reserved"
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
scontrol release "$job_id"
echo released_to_scheduler > "$report.state"
echo "SFT_ROLLOUT_RELEASED job=$job_id commit=$(git rev-parse HEAD)"
