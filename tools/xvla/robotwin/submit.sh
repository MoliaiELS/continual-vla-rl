#!/usr/bin/env bash
set -euo pipefail
tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:?mode}" partition="${2:?partition}" gpu="${3:?gpu}" lane="${4:?lane}" hours="${5:?hours}" task="${6:-}"
evaluation_root="${7:-}"
eval_workers="${8:-1}"
cpus=8
case "$eval_workers" in
    1) ;;
    2)
        if [[ -z "$evaluation_root" || "$lane" != easy || "$gpu" != 4090 ]]; then
            echo 'Two eval workers require the frozen Easy recovery and one RTX 4090' >&2
            exit 1
        fi
        cpus=16
        ;;
    *) echo 'Expected one or two eval workers' >&2; exit 1 ;;
esac
orchestration_commit="$(git -C "$tool_dir" rev-parse HEAD)"
evaluation_commit="$orchestration_commit"
evaluation_tools="$tool_dir"
if [[ -n "$evaluation_root" ]]; then
    if [[ "$mode" != formal && "$mode" != extension ]]; then
        echo 'Prescreen recovery requires formal or extension' >&2
        exit 1
    fi
    evaluation_root="$(cd "$evaluation_root" && pwd -P)"
    evaluation_commit="$(git -C "$evaluation_root" rev-parse HEAD)"
    if [[ "$evaluation_commit" != 9baa17360abe242cb36e8f9c78f0aadbe64b3bee ]]; then
        echo 'Prescreen recovery requires the original RoboTwin evaluation commit' >&2
        exit 1
    fi
    evaluation_tools="$evaluation_root/tools/xvla/robotwin"
    git -C "$tool_dir" ls-files --error-unmatch prescreen_recovery.py prescreen_failures.py >/dev/null
    git -C "$tool_dir" diff --quiet HEAD -- submit.sh job.sh panel.py prescreen_recovery.py prescreen_failures.py
fi
source "$evaluation_tools/runtime.env"
comment=()
submission_token="${FYP_ROBOTWIN_SUBMISSION_TOKEN:-}"
if [[ -n "$submission_token" ]]; then
    if [[ ! "$submission_token" =~ ^rw-hard-easy-[0-9a-f]{24}$ || "$mode" != extension || "$lane" != easy ||
          "$gpu" != 4090 || "$eval_workers" != 2 || "$evaluation_commit" != 9baa17360abe242cb36e8f9c78f0aadbe64b3bee ]]; then
        echo 'Submission token requires the fixed Hard-controller Easy extension' >&2
        exit 1
    fi
    git -C "$tool_dir" ls-files --error-unmatch hard_native_dispatch.py >/dev/null
    git -C "$tool_dir" diff --quiet HEAD -- hard_native_dispatch.py submit.sh
    comment=("--comment=$submission_token")
fi
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
    own_gpus="$(awk 'BEGIN {n=0} {for (i=1;i<=NF;i++) if ($i ~ /^gres[/:]gpu(:[[:alnum:]_-]+)?:[0-9]+$/) {split($i,a,":"); n+=a[length(a)]}} END{print n}' <<< "${own//,/ }")"
    if (( own_gpus >= 2 )); then echo 'Two own GPUs are already allocated or requested' >&2; exit 1; fi
    gres=("--gres=gpu:$gpu:1")
fi
job_args=("$mode")
if [[ "$mode" != build ]]; then
    if [[ -n "$evaluation_root" ]]; then job_args+=(--pinned-evaluation-root "$evaluation_root"); fi
    job_args+=(--lane "$lane")
    if [[ -n "$task" ]]; then job_args+=(--task "$task"); fi
    if [[ "$eval_workers" == 2 ]]; then job_args+=(--eval-workers 2); fi
fi
printf -v launch '%q ' bash "$tool_dir/job.sh" "${job_args[@]}"
jobid="$(sbatch --parsable --partition="$partition" --exclude=4090node1 --nodes=1 --ntasks=1 \
    --cpus-per-task="$cpus" --time="$((10#$hours * 60))" "${gres[@]}" \
    --job-name="rw-b0-$mode-$lane" --output="$data/logs/%j.out" --error="$data/logs/%j.err" \
    --export=ALL "${comment[@]}" --wrap="$launch")"
if [[ -n "$evaluation_root" ]]; then
    printf '%s\t%s\t%s\n' "${jobid%%;*}" "$evaluation_commit" "$orchestration_commit" >> "$data/evidence/prescreen-recovery-jobs.tsv"
fi
if [[ "$eval_workers" == 2 ]]; then
    printf '%s\t%s\t%s\t%s\n' "${jobid%%;*}" "$evaluation_commit" "$orchestration_commit" "$eval_workers" >> "$data/evidence/parallel-eval-jobs.tsv"
fi
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$jobid" "$mode" "$lane" "$gpu" "$evaluation_commit" "$task" >> "$data/evidence/jobs.tsv"
echo "ROBOTWIN_JOB_ID=$jobid"
