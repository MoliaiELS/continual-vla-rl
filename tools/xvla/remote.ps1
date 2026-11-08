param(
    [Parameter(Position=0)][ValidateSet('audit','sync','bootstrap','download','submit','status','logs')][string]$Command='audit',
    [Parameter(Position=1)][string]$Value='unit',
    [ValidateSet('libero','pt')][string]$Model='libero',
    [int]$Gpus=1
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
test -e '$snapshot' || git -C '$($settings.CodeRoot)' worktree add --detach '$snapshot' '$sha'
git -C '$snapshot' submodule update --init third_party/rlinf
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
  if ($Value -notmatch '^[a-z0-9_-]+$' -or $Gpus -notin @(0,1,2)) { throw 'Invalid smoke mode or GPU count' }
  Invoke-XRemote @"
test -f '$snapshot/tools/xvla/job.sh'
mkdir -p '$data/logs'
'$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' '$snapshot/tools/xvla/submit.py' --repo '$snapshot' --data '$data' --mode '$Value' --gpus $Gpus --partition '$($settings.Partition)'
"@
 }
 'status' {
  Invoke-XRemote @"
tail -c 2500 '$data/logs/bootstrap.log' 2>/dev/null || true
tail -c 1000 '$data/logs/download-libero.log' 2>/dev/null || true
squeue -u "`$(id -un)" -o '%i|%j|%T|%M|%b|%N'
"@
 }
 'logs' {
  if ($Value -notmatch '^\d+$') { throw 'logs requires job ID' }
  Invoke-XRemote "tail -c 9000 '$data/logs/$Value.out'; tail -c 2500 '$data/logs/$Value.err'"
 }
}
