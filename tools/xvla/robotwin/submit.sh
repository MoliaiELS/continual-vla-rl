#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$tool_dir/runtime.env"
mode="${1:?mode}" partition="${2:?partition}" gpu="${3:?gpu}" lane="${4:?lane}" hours="${5:?hours}" task="${6:-}"
data="$FYP_ROBOTWIN_DATA"
mkdir -p "$data/logs" "$data/evidence"
exec 9> "$data/submit.lock"
flock 9
if [[ "$mode" == build ]]; then
    test -s "$data/evidence/install-patches.json"
    test -x "$FYP_ROBOTWIN_ENV/bin/nvcc"
    gres=()
else
    test -s "$data/evidence/build.complete"
    if [[ "$mode" != probe ]]; then
        test -s "$data/models/X-VLA-RoboTwin2--${MODEL_REV:0:12}/verified.json"
        for arm in left right; do
            test -s "$FYP_ROBOTWIN_CODE/RoboTwin-$ROBOTWIN_REV/assets/embodiments/aloha-agilex/curobo_${arm}.yml"
        done
    fi
    # Count every own allocated or pending GPU, including other FYP runs, and
    # refuse a third. Jobs are submitted one at a time under this suite's lock.
    own="$(squeue -h -u "$(id -un)" -o '%b')"
    own_gpus="$(awk 'BEGIN {n=0} {for (i=1;i<=NF;i++) if ($i ~ /^gres:gpu(:[[:alnum:]_-]+)?:[0-9]+$/) {split($i,a,":"); n+=a[length(a)]}} END{print n}' <<< "${own//,/ }")"
    if (( own_gpus >= 2 )); then echo 'Two own GPUs are already allocated or requested' >&2; exit 1; fi
    gres=("--gres=gpu:$gpu:1")
fi
job_args=("$mode")
if [[ "$mode" != build ]]; then
    job_args+=(--lane "$lane")
    if [[ -n "$task" ]]; then job_args+=(--task "$task"); fi
fi
printf -v launch '%q ' bash "$tool_dir/job.sh" "${job_args[@]}"
jobid="$(sbatch --parsable --partition="$partition" --exclude=4090node1 --nodes=1 --ntasks=1 \
    --cpus-per-task=8 --time="$((10#$hours * 60))" "${gres[@]}" \
    --job-name="rw-b0-$mode-$lane" --output="$data/logs/%j.out" --error="$data/logs/%j.err" \
    --export=ALL --wrap="$launch")"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$jobid" "$mode" "$lane" "$gpu" "$(git -C "$tool_dir" rev-parse HEAD)" "$task" >> "$data/evidence/jobs.tsv"
echo "ROBOTWIN_JOB_ID=$jobid"
