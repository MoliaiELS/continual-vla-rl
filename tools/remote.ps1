param(
    [Parameter(Position=0)][ValidateSet('doctor','sync','bootstrap','bootstrap-status','model-download','model-status','smoke','config','eval-smoke','train-smoke','status','logs')][string]$Command = 'doctor',
    [Parameter(Position=1)][string]$JobId
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
    { $_ -in 'smoke','config','eval-smoke','train-smoke' } {
        $sha = Get-Commit
        $snapshot = "$($cfg.RunCodeRoot)/$sha"
        $mode = @{'smoke'='env';'config'='config';'eval-smoke'='eval';'train-smoke'='train'}[$Command]
        $nodeArg = if ($cfg.Node) { "--nodelist='$($cfg.Node)'" } else { '' }
        Invoke-Remote @"
test -f '$snapshot/tools/idac/job.sh'
mkdir -p '$data/logs'
export FYP_JOB_REPO='$snapshot'
sbatch --parsable --chdir='$snapshot' --partition='$($cfg.Partition)' $nodeArg --exclude=4090node1 --nodes=1 --ntasks=1 --gres='$($cfg.Gres)' --cpus-per-task=$($cfg.Cpus) --time=00:30:00 --job-name='fyp-$mode' --output='$data/logs/%j.out' --error='$data/logs/%j.err' '$snapshot/tools/idac/job.sh' '$mode'
"@
    }
    { $_ -in 'status','logs' } {
        if ($JobId -notmatch '^\d+$') { throw 'An explicit numeric job ID is required' }
        if ($Command -eq 'status') {
            Invoke-Remote "squeue -j '$JobId'; sacct -j '$JobId' --format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES -P"
        } else {
            Invoke-Remote "tail -n 80 '$data/logs/$JobId.out' '$data/logs/$JobId.err'"
        }
    }
}
