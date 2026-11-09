param(
    [Parameter(Position=0)][ValidateSet('audit','sync','bootstrap','download','submit','status','logs','session')][string]$Command='audit',
    [Parameter(Position=1)][string]$Value='unit',
    [ValidateSet('libero','pt')][string]$Model='libero',
    [int]$Gpus=1,
    [ValidateSet('3090','4090')][string]$GpuModel='4090',
    [ValidateRange(0,9)][int]$StartTask=0,
    [ValidateRange(1,10)][int]$TaskCount=10,
    [ValidateRange(1,50)][int]$Trials=5,
    [ValidatePattern('^[0-9]+$')][string]$CheckpointJobId
)
$ErrorActionPreference='Stop'
$OutputEncoding=[Text.UTF8Encoding]::new($false)
$repoRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$settings=Import-PowerShellDataFile "$repoRoot/tools/idac/settings.psd1"
$data="$($settings.DataRoot)/xvla"
$prefix="$($settings.CondaRoot)/envs/xvla_rlinf"
function Invoke-XRemote([string]$Script) {
    $preamble="export FYP_DATA_ROOT='$($settings.DataRoot)' FYP_CONDA_ROOT='$($settings.CondaRoot)' FYP_XVLA_DATA='$data' FYP_XVLA_ENV='$prefix'`n"
    ($preamble+$Script).Replace("`r",'') | & ssh -T -o BatchMode=yes -o ConnectTimeout=15 $settings.SshHost "tr -d '\r' | bash -se"
    if ($LASTEXITCODE -ne 0) { throw "X-VLA remote failed: $LASTEXITCODE" }
}
$sha=(& git -C $repoRoot rev-parse HEAD).Trim()
$snapshot="$($settings.RunCodeRoot)/$sha"
switch ($Command) {
 'session' {
  # One persistent connection for serial diagnostics during unstable VPN links.
  $initialize=@"
export FYP_DATA_ROOT='$($settings.DataRoot)' FYP_CONDA_ROOT='$($settings.CondaRoot)' FYP_XVLA_DATA='$data' FYP_XVLA_ENV='$prefix' FYP_XVLA_SNAPSHOT='$snapshot'
exec bash --noprofile --norc
"@
  & ssh -tt -o BatchMode=yes -o ConnectTimeout=5 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 $settings.SshHost ($initialize.Replace("`r",''))
  if ($LASTEXITCODE -ne 0) { throw "X-VLA session failed: $LASTEXITCODE" }
 }
 'audit' {
  Invoke-XRemote @'
hostname
id
squeue -u "$(id -un)" -o '%i|%j|%T|%M|%b|%N'
df -h "$FYP_CONDA_ROOT" "$FYP_DATA_ROOT"
"$FYP_CONDA_ROOT/envs/vla_budget_rl/bin/python" - <<'PY'
import importlib.metadata as m,json,os,sys
names=['torch','torchvision','transformers','peft','ray','timm','numpy','scipy','hydra-core','accelerate','tensordict','torchdata','mujoco','robosuite','huggingface-hub']
versions={}
for n in names:
    try: versions[n]=m.version(n)
    except m.PackageNotFoundError: versions[n]=None
print(json.dumps({'python':sys.version,'prefix':sys.prefix,'versions':versions}))
PY
'@
 }
 'sync' {
  & git -C $repoRoot diff --quiet
  if ($LASTEXITCODE -ne 0) { throw 'Commit tracked changes before sync' }
  & git -C $repoRoot diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { throw 'Commit staged changes before sync' }
  & git -C $repoRoot push origin HEAD:refs/heads/codex/xvla-integration
  if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
  Invoke-XRemote @"
git -C '$($settings.CodeRoot)' fetch origin codex/xvla-integration
test -e '$snapshot' || git -C '$($settings.CodeRoot)' worktree add --quiet --detach '$snapshot' '$sha'
git -C '$snapshot' submodule update --quiet --init third_party/rlinf
test "`$(git -C '$snapshot/third_party/rlinf' rev-parse HEAD)" = 1bf78980d9339cbc67497eddceb9a1e3a74e2acc
echo 'XVLA_SYNC_OK $sha'
"@
 }
 'bootstrap' {
  Invoke-XRemote @"
test -f '$snapshot/tools/xvla/bootstrap.sh'
mkdir -p '$data/logs'
nohup flock -n '$data/bootstrap.lock' bash '$snapshot/tools/xvla/bootstrap.sh' > '$data/logs/bootstrap.log' 2>&1 < /dev/null &
echo XVLA_BOOTSTRAP_PID=`$!
"@
 }
 'download' {
  Invoke-XRemote @"
test -f '$snapshot/tools/xvla/download.py'
mkdir -p '$data/logs'
nohup flock -n '$data/download-$Model.lock' '$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' '$snapshot/tools/xvla/download.py' '$data/models' '$Model' > '$data/logs/download-$Model.log' 2>&1 < /dev/null &
echo XVLA_DOWNLOAD_PID=`$!
"@
 }
 'submit' {
  if ($Value -notmatch '^[a-z0-9_-]+$' -or $Gpus -notin @(0,1,2)) { throw 'Invalid X-VLA mode or GPU count' }
  $benchmarkOptions=''
  if ($Value -in @('benchmark_libero','benchmark_pt','benchmark_sft_smoke')) {
   $benchmarkOptions=" --gpu-model $GpuModel --start-task $StartTask --task-count $TaskCount --trials $Trials"
   if ($CheckpointJobId) { $benchmarkOptions += " --checkpoint-job-id $CheckpointJobId" }
  } elseif ($PSBoundParameters.ContainsKey('GpuModel') -or $CheckpointJobId -or $StartTask -ne 0 -or $TaskCount -ne 10 -or $Trials -ne 5) {
   throw 'GPU model and benchmark options require a benchmark mode'
  }
  Invoke-XRemote @"
test -f '$snapshot/tools/xvla/job.sh'
mkdir -p '$data/logs'
'$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' '$snapshot/tools/xvla/submit.py' --repo '$snapshot' --data '$data' --mode '$Value' --gpus $Gpus --partition '$($settings.Partition)'$benchmarkOptions
"@
 }
 'status' {
  Invoke-XRemote @"
tail -c 2500 '$data/logs/bootstrap.log' 2>/dev/null || true
tail -c 1000 '$data/logs/download-libero.log' 2>/dev/null || true
tail -c 1000 '$data/logs/download-pt.log' 2>/dev/null || true
squeue -u "`$(id -un)" -o '%i|%j|%T|%M|%b|%N'
"@
 }
 'logs' {
  if ($Value -notmatch '^\d+$') { throw 'logs requires job ID' }
  Invoke-XRemote "if test -f '$data/logs/$Value.out'; then tail -c 9000 '$data/logs/$Value.out'; tail -c 4000 '$data/logs/$Value.err' 2>/dev/null || true; else squeue -j '$Value' -o '%i|%j|%T|%M|%R'; fi"
 }
}
