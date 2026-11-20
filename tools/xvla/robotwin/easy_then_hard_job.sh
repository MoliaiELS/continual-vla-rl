#!/usr/bin/env bash
set -euo pipefail
: "${SLURM_JOB_ID:?use sbatch}" "${FYP_ROBOTWIN_DATA:?}" "${FYP_ROBOTWIN_ENV:?}"
[[ "$SLURM_JOB_ID" =~ ^[0-9]+$ && $(hostname -s) != 4090node1 ]]
evaluation="$(cd "${1:?evaluation root}" && pwd -P)"
worker="$(cd "${2:?bcb worker root}" && pwd -P)"
profile="$(cd "${3:?Hard profile}" && pwd -P)"
tools="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
data="$(cd "$FYP_ROBOTWIN_DATA" && pwd -P)"
evaluation_commit=9baa17360abe242cb36e8f9c78f0aadbe64b3bee
worker_commit=bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460
[[ $(git -C "$evaluation" rev-parse HEAD) == "$evaluation_commit" ]]
[[ $(git -C "$worker" rev-parse HEAD) == "$worker_commit" ]]
[[ "$profile" == "$data/hard-native-budget/$worker_commit" ]]
git -C "$evaluation" diff --quiet HEAD -- tools/xvla
git -C "$worker" diff --quiet HEAD -- tools/xvla/robotwin
git -C "$tools" ls-files --error-unmatch easy_then_hard_job.sh >/dev/null
git -C "$tools" diff --quiet HEAD -- easy_then_hard_job.sh
helper_commit="$(git -C "$tools" rev-parse HEAD)"
python="$FYP_ROBOTWIN_ENV/bin/python"
[[ -x "$python" ]]
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export FYP_ROBOTWIN_JOB="$data/jobs/$SLURM_JOB_ID"
job="$FYP_ROBOTWIN_JOB"
mkdir -p "$job"
sequence="$(date +%s%N)-$$"
stage=none
child_running=0
telemetry=(runtime.json resources.json allocated_gpu_uuids.json vulkan-device.txt gpu_memory_samples.jsonl)

record() {
    "$python" - "$job/easy-then-hard.jsonl" "$sequence" "$stage" "$helper_commit" "$worker_commit" "$evaluation_commit" "$@" <<'PY'
import json, os, sys, time
path, sequence, stage, helper, worker, evaluation, event, code, *command = sys.argv[1:]
row = dict(time=time.time(), job_id=os.environ["SLURM_JOB_ID"], sequence=sequence, stage=stage,
           helper_commit=helper, worker_commit=worker, evaluation_commit=evaluation,
           event=event, returncode=int(code), command=command)
with open(path, "a") as stream:
    stream.write(json.dumps(row) + "\n"); stream.flush(); os.fsync(stream.fileno())
PY
}

archive() {
    [[ "$stage" != none && "$child_running" == 0 ]] || return 0
    local destination="$job/stages/$stage/$sequence" name
    mkdir -p "$destination" || return "$?"
    for name in "${telemetry[@]}"; do
        if [[ -f "$job/$name" ]]; then
            [[ ! -e "$destination/$name" ]] || return 1
            mv -- "$job/$name" "$destination/$name" || return "$?"
        fi
    done
    if [[ "$stage" == hard && -f "$profile/reports/summary.json" ]]; then
        cp -- "$profile/reports/summary.json" "$destination/hard-summary.json" || return "$?"
    fi
    record archived 0 "$destination" || return "$?"
    stage=none
}
finish() {
    local code=$? archive_code=0
    trap - EXIT
    archive || archive_code=$?
    [[ "$code" != 0 ]] || code=$archive_code
    record wrapper_exit "$code" || true
    exit "$code"
}
trap finish EXIT

gate() {
    local mode="$1" target="$job/stages/easy/$sequence/$2.json"
    mkdir -p "$(dirname "$target")"
    "$python" - "$evaluation" "$data/panels/$evaluation_commit" "$profile" "$mode" "$target" <<'PY'
import json, os, pathlib, sys
evaluation, source, profile, mode, target = map(pathlib.Path, sys.argv[1:])
sys.path.insert(0, str(evaluation))
from tools.xvla.robotwin import summarize
assert pathlib.Path(summarize.__file__).resolve() == evaluation / "tools/xvla/robotwin/summarize.py"
hard = json.loads((profile / "profile.json").read_bytes())
assert hard["implementation_commit"] == "bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460"
assert hard["source_evaluation_commit"] == "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
report = summarize.summarize(source)
first, total = (report["overall"]["easy"][key] for key in ("first_batch", "cumulative"))
assert first["planned"] == first["finished"] == 5000 and first["unresolved"] == first["retry_pending"] == 0
assert total["planned"] == 7000 and total["unresolved"] == 0
if str(mode) == "complete":
    assert total["finished"] == 7000 and total["retry_pending"] == 0
with target.open("w") as stream:
    json.dump(dict(first_batch=first, cumulative=total), stream, indent=2)
    stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
print(total["finished"])
PY
}

run_stage() {
    stage="$1"; shift
    record started 0 "$@"
    child_running=1
    local code=0
    bash "$@" || code=$?
    child_running=0
    record finished "$code" "$@"
    archive
    [[ "$code" == 0 ]] || exit "$code"
}

record wrapper_started 0 "${BASH_SOURCE[0]}" "$evaluation" "$worker" "$profile"
for name in "${telemetry[@]}"; do
    if [[ -f "$job/$name" ]]; then stage=interrupted; archive; break; fi
done
finished="$(gate before before)"
if [[ "$finished" != 7000 ]]; then
    run_stage easy "$worker/tools/xvla/robotwin/job.sh" extension --pinned-evaluation-root "$evaluation" --lane easy --eval-workers 2
else
    record easy_already_complete 0
fi
gate complete after >/dev/null
record easy_7000_verified 0
run_stage hard "$worker/tools/xvla/robotwin/hard_native_job.sh" "$evaluation" "$profile"
