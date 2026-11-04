param(
    [Parameter(Position=0)][ValidateSet('doctor','sync','bootstrap','bootstrap-status','model-download','model-status','rollout','rollout-status','smoke','config','eval-smoke','train-smoke','reload-smoke','status','logs','cancel','experiment-inventory','experiment-submit','experiment-status','experiment-logs','experiment-tests','experiment-fetch','replay-download')][string]$Command = 'doctor',
    [Parameter(Position=1)][string]$JobId,
    [string]$SuiteName,
    [string]$Dependency,
    [string]$NextSuiteName,
    [string]$StressSuiteName,
    [string]$PairedSuiteName
)
$ErrorActionPreference = 'Stop'
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$root = Split-Path $PSScriptRoot -Parent
$cfg = Import-PowerShellDataFile "$PSScriptRoot/idac/settings.psd1"
function Invoke-Remote([string]$Script) {
    $Script = "export FYP_DATA_ROOT='$($cfg.DataRoot)' FYP_CONDA_ROOT='$($cfg.CondaRoot)'`n" + $Script
    $Script.Replace("`r",'') | & ssh -T -o BatchMode=yes -o ConnectTimeout=15 $cfg.SshHost "tr -d '\r' | bash -se"
    if ($LASTEXITCODE -ne 0) { throw "Remote command failed: $LASTEXITCODE" }
}
function Get-Commit {
    $value = (& git -C $root rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $value -notmatch '^[a-f0-9]{40}$') { throw 'Invalid Git checkout' }
    return $value
}
$data = $cfg.DataRoot
$code = $cfg.CodeRoot
switch ($Command) {
    'cancel' {
        if ($JobId -notmatch '^\d+$') { throw 'Specify an explicit numeric job ID' }
        Invoke-Remote @"
owner=`$(squeue -h -j '$JobId' -o '%u')
test `"`$owner`" = `"`$(id -un)`"
scancel '$JobId'
echo OWN_JOB_CANCELLED_$JobId
"@
    }
    'experiment-logs' {
        if ($JobId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Specify the suite name' }
        Invoke-Remote @"
'$($cfg.CondaRoot)/envs/vla_budget_rl/bin/python' - '$data/experiments/$JobId' <<'PY'
import json
from pathlib import Path
import sys
root=Path(sys.argv[1])
receipt=json.loads((root/'submission.json').read_text())
latest=receipt['jobs'][-1]
print(json.dumps(latest))
for suffix in ['out','err']:
    path=root/'slurm'/(latest['job_id']+'.'+suffix)
    if path.is_file(): print(str(path)+'\n'+path.read_text(errors='replace')[-9000:])
PY
"@
    }
    'experiment-fetch' {
        if ($JobId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Specify the suite name' }
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        Invoke-Remote @"
source '$snapshot/tools/idac/env.sh'
python '$snapshot/tools/idac/experiment.py' export '$data/experiments/$JobId'
"@
        $localEvidence = Join-Path $root ".local/experiments/$JobId"
        New-Item -ItemType Directory -Force -Path $localEvidence | Out-Null
        & scp -q "$($cfg.SshHost):$data/experiments/$JobId/evidence.tar.gz" "$localEvidence/evidence.tar.gz"
        if ($LASTEXITCODE -ne 0) { throw 'Evidence download failed' }
        & tar -xf "$localEvidence/evidence.tar.gz" -C $localEvidence
        if ($LASTEXITCODE -ne 0) { throw 'Evidence extraction failed' }
        Write-Output "LOCAL_EVIDENCE $localEvidence"
    }
    'experiment-tests' {
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        Invoke-Remote @"
test -f '$snapshot/tools/idac/test_protocol.py'
source '$snapshot/tools/idac/env.sh'
cd '$snapshot'
python tools/idac/test_protocol.py
python tools/idac/test_analysis.py
python tools/idac/test_replay.py
python tools/idac/test_rollout_evidence.py
python tools/idac/test_ood_validation.py
python tools/idac/test_experiment.py
python tools/idac/plot_experiment.py --self-test
python -m py_compile tools/idac/experiment.py rlinf/runners/embodied_runner.py rlinf/runners/embodied_eval_runner.py rlinf/workers/actor/fsdp_actor_worker.py rlinf/workers/env/env_worker.py rlinf/envs/libero/libero_env.py
bash -n tools/idac/experiment_job.sh
"@
    }
    'experiment-submit' {
        if ($JobId -notin @('qualification','core','stress10')) { throw 'Specify qualification, core, or stress10' }
        if (-not $SuiteName) { $SuiteName = (Get-Date -Format 'yyyyMMdd') + '-' + $JobId + '-v1' }
        if ($SuiteName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid suite name' }
        if ($Dependency -and $Dependency -notmatch '^\d+$') { throw 'Invalid dependency job ID' }
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        $dependencyArg = if ($Dependency) { "--dependency '$Dependency'" } else { '' }
        $continuationSetup = ''
        $continuationArg = ''
        $pairedArg = ''
        if ($JobId -eq 'stress10') {
            if ($PairedSuiteName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'stress10 requires a valid -PairedSuiteName' }
            $pairedArg = "--paired-baseline-suite '$data/experiments/$PairedSuiteName'"
        }
        if ($NextSuiteName) {
            if ($JobId -ne 'qualification' -or $NextSuiteName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Only qualification can create a named core continuation' }
            $stressArg = ''
            if ($StressSuiteName) {
                if ($StressSuiteName -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Invalid stress suite name' }
                $stressArg = "--next-suite '$data/experiments/$StressSuiteName'"
            }
            $continuationSetup = "python tools/idac/experiment.py create '$data/experiments/$NextSuiteName' --profile core --commit '$sha' $stressArg"
            if ($StressSuiteName) {
                $continuationSetup += "`npython tools/idac/experiment.py create '$data/experiments/$StressSuiteName' --profile stress10 --commit '$sha' --paired-baseline-suite '$data/experiments/$NextSuiteName'"
            }
            $continuationArg = "--next-suite '$data/experiments/$NextSuiteName'"
        }
        Invoke-Remote @"
test -f '$snapshot/tools/idac/experiment.py'
source '$snapshot/tools/idac/env.sh'
cd '$snapshot'
export FYP_JOB_REPO='$snapshot' FYP_PARTITION='$($cfg.Partition)'
$continuationSetup
python tools/idac/experiment.py create '$data/experiments/$SuiteName' --profile '$JobId' --commit '$sha' $continuationArg $pairedArg
python tools/idac/experiment.py submit '$data/experiments/$SuiteName' $dependencyArg
"@
    }
    'experiment-status' {
        if ($JobId -notmatch '^[a-zA-Z0-9_-]+$') { throw 'Specify the suite name' }
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        Invoke-Remote @"
source '$snapshot/tools/idac/env.sh'
python '$snapshot/tools/idac/experiment.py' status '$data/experiments/$JobId'
"@
    }
    'replay-download' {
        Invoke-Remote @"
mkdir -p '$data/datasets/libero_raw' '$data/reports' '$data/cache/huggingface'
nohup flock -n '$data/replay-download.lock' bash -c '
set -euo pipefail
source "`$1/etc/profile.d/conda.sh"
conda activate "`$1/envs/vla_budget_rl"
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1 HF_HOME="`$2/cache/huggingface"
exec nice -n 19 python -m huggingface_hub.commands.huggingface_cli download yifengzhu-hf/LIBERO-datasets \
libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
libero_spatial/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate_demo.hdf5 \
libero_spatial/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate_demo.hdf5 \
libero_spatial/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate_demo.hdf5 \
libero_spatial/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo.hdf5 \
--repo-type dataset --revision f13aa24a3da8c43c7225569f28c562979fa0e35a --local-dir "`$2/datasets/libero_raw"
' _ '$($cfg.CondaRoot)' '$data' > '$data/reports/replay-download.log' 2>&1 < /dev/null &
echo REPLAY_DOWNLOAD_PID=`$!
"@
    }
    'experiment-inventory' {
        Invoke-Remote @"
scontrol show partition '$($cfg.Partition)'
find '$data/datasets' -maxdepth 3 -type f -name '*.hdf5' -print 2>/dev/null | head -n 15 || true
find /mnt/slurmfs-4090node3/public_data -maxdepth 2 -iname '*libero*' -print 2>/dev/null | head -n 15 || true
'$($cfg.CondaRoot)/envs/vla_budget_rl/bin/python' - <<'PY'
import json
from pathlib import Path
root=Path('$data/models/Openvla-oft-SFT-libero-spatial-traj1')
c=json.loads((root/'config.json').read_text())
print('NORM_STATS_KEYS', list(c.get('norm_stats', {})))
PY
tail -c 1200 '$data/reports/replay-download.log' 2>/dev/null || true
find '$data/datasets/libero_raw' -maxdepth 7 -type f -name '*.incomplete' -printf '%s %f\n' 2>/dev/null | head -n 8 || true
"@
    }
    'doctor' {
        Invoke-Remote @'
hostname
id
sinfo -N -o '%N %P %t %m %G %C'
squeue -u "$(id -un)"
df -h /mnt/slurmfs-4090node1 /mnt/slurmfs-4090node3
'@
    }
    'sync' {
        if (& git -C $root status --porcelain) { throw 'Commit local changes before sync' }
        $sha = Get-Commit
        & git -C $root push origin "HEAD:refs/heads/$($cfg.Branch)"
        if ($LASTEXITCODE -ne 0) { throw 'Git push failed' }
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        Invoke-Remote @"
if [ ! -d '$code/.git' ]; then
  test ! -e '$code'
  git clone --filter=blob:none --single-branch --branch '$($cfg.Branch)' '$($cfg.Repository)' '$code'
fi
git -C '$code' diff --quiet
git -C '$code' diff --cached --quiet
git -C '$code' fetch origin '$($cfg.Branch)'
git -C '$code' cat-file -e '$sha^{commit}'
if [ ! -d '$snapshot' ]; then
  mkdir -p '$($cfg.RunCodeRoot)'
  git -C '$code' worktree add --quiet --detach '$snapshot' '$sha'
fi
test "`$(git -C '$snapshot' rev-parse HEAD)" = '$sha'
echo 'SYNC_OK $sha $snapshot'
"@
    }
    'bootstrap' {
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        Invoke-Remote "test -f '$snapshot/tools/idac/bootstrap.sh'; nohup bash '$snapshot/tools/idac/bootstrap.sh' > '$data/reports/bootstrap.log' 2>&1 < /dev/null & echo BOOTSTRAP_PID=`$!"
    }
    'bootstrap-status' {
        Invoke-Remote "if test -f '$data/reports/bootstrap.complete'; then echo BOOTSTRAP_COMPLETE; echo 'Identity: $data/reports/environment.identity.json'; echo 'Dependencies: $data/reports/requirements.resolved.txt'; else echo BOOTSTRAP_NOT_COMPLETE; tail -n 30 '$data/reports/bootstrap.log'; fi"
    }
    'model-download' {
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        Invoke-Remote @"
test -f '$snapshot/tools/idac/download_model.py'
mkdir -p '$data/reports'
export HF_ENDPOINT='$($cfg.ModelEndpoint)' FYP_MODEL_REPOSITORY='$($cfg.ModelRepository)' FYP_MODEL_REVISION='$($cfg.ModelRevision)'
nohup flock -n '$data/model-download.lock' bash -c 'set -euo pipefail; source "`$1/tools/idac/env.sh"; cd "`$FYP_REPO"; exec nice -n 19 python -u tools/idac/download_model.py' _ '$snapshot' > '$data/reports/model-download.log' 2>&1 < /dev/null &
echo MODEL_DOWNLOAD_PID=`$!
"@
    }
    'model-status' {
        Invoke-Remote @"
if test -f '$data/reports/model.snapshot.json'; then cat '$data/reports/model.snapshot.json'; fi
if test -f '$data/reports/model-download.log'; then tail -c 4000 '$data/reports/model-download.log'; fi
"@
    }
    'rollout' {
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        Invoke-Remote @"
test -f '$snapshot/tools/idac/rollout_when_ready.sh'
export FYP_PARTITION='$($cfg.Partition)' FYP_GRES='$($cfg.Gres)' FYP_CPUS='$($cfg.Cpus)'
nohup bash '$snapshot/tools/idac/rollout_when_ready.sh' '$snapshot' > '$data/reports/rollout-launch.log' 2>&1 < /dev/null &
echo ROLLOUT_LAUNCHER_PID=`$!
"@
    }
    'rollout-status' {
        Invoke-Remote @"
for suffix in state job-id exit-code; do
  if test -f '$data/reports/rollout-launch.'`"`$suffix`"; then printf '%s: ' `"`$suffix`"; cat '$data/reports/rollout-launch.'`"`$suffix`"; fi
done
tail -n 12 '$data/reports/rollout-launch.log'
if test -s '$data/reports/rollout-launch.job-id'; then
  job_id=`$(cat '$data/reports/rollout-launch.job-id')
  squeue -u `$(id -un) -h -o '%i %j %T %R'
  sacct -j `"`$job_id`" --format=JobID,State,ExitCode,Elapsed -P
fi
"@
    }
    { $_ -in 'smoke','config','eval-smoke','train-smoke','reload-smoke' } {
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        $mode = @{'smoke'='env';'config'='config';'eval-smoke'='eval';'train-smoke'='train';'reload-smoke'='reload'}[$Command]
        $adapterSetup = 'unset FYP_ADAPTER_PATH'
        if ($Command -eq 'reload-smoke') {
            if ($JobId -notmatch '^\d+$') { throw 'An explicit training job ID is required' }
            $adapterSetup = "export FYP_ADAPTER_PATH='$data/runs/train_$JobId/checkpoints/global_step_1/actor'`ntest -s `"`$FYP_ADAPTER_PATH/adapter_model.bin`"`ntest -s `"`$FYP_ADAPTER_PATH/adapter_config.json`""
        }
        $nodeArg = if ($cfg.Node) { "--nodelist='$($cfg.Node)'" } else { '' }
        $gres = if ($mode -eq 'train') { $cfg.Gres -replace ':1$', ':2' } else { $cfg.Gres }
        Invoke-Remote @"
test -f '$snapshot/tools/idac/job.sh'
mkdir -p '$data/logs'
export FYP_JOB_REPO='$snapshot'
$adapterSetup
sbatch --parsable --chdir='$snapshot' --partition='$($cfg.Partition)' $nodeArg --exclude=4090node1 --nodes=1 --ntasks=1 --gres='$gres' --cpus-per-task=$($cfg.Cpus) --time=00:30:00 --job-name='fyp-$mode' --output='$data/logs/%j.out' --error='$data/logs/%j.err' '$snapshot/tools/idac/job.sh' '$mode'
"@
    }
    { $_ -in 'status','logs' } {
        if ($JobId -notmatch '^\d+$') { throw 'An explicit numeric job ID is required' }
        if ($Command -eq 'status') {
            Invoke-Remote "squeue -u `$(id -un) -h -o '%i %j %T %R' | awk '`$1 == $JobId'; sacct -j '$JobId' --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P"
        } else {
            Invoke-Remote "tail -n 80 '$data/logs/$JobId.out' '$data/logs/$JobId.err'"
        }
    }
}
