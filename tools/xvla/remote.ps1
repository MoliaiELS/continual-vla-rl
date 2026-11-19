param(
    [Parameter(Position=0)][ValidateSet('audit','sync','bootstrap','download','submit','status','logs','session','e2-start','e2-status','e3-start','e3-status','final-submit','robotwin-prepare','robotwin-extract','robotwin-status','robotwin-logs','robotwin-submit','robotwin-report','robotwin-run')][string]$Command='audit',
    [Parameter(Position=1)][string]$Value='unit',
    [ValidateSet('libero','pt','robotwin')][string]$Model='libero',
    [int]$Gpus=1,
    [ValidateSet('3090','4090')][string]$GpuModel='4090',
    [ValidateRange(0,9)][int]$StartTask=0,
    [ValidateRange(1,10)][int]$TaskCount=10,
    [ValidateRange(0,499)][int]$StartTrial=0,
    [ValidateRange(1,50)][int]$Trials=5,
    [ValidatePattern('^[0-9]+$')][string]$CheckpointJobId,
    [ValidatePattern('^[0-9]+$')][string]$AuditJobId,
    [ValidatePattern('^[0-9]+$')][string]$DeployedJobId,
    [ValidatePattern('^[0-9]+$')][string]$OnlineJobId,
    [ValidatePattern('^[0-9]+$')][string]$DevJobId,
    [ValidatePattern('^[0-9]+$')][string]$FailedJobId,
    [ValidatePattern('^[0-9]+$')][string]$FallbackJobId,
    [ValidateSet('256','1024')][string]$SftSteps='256',
    [ValidatePattern('^[0-9a-f]{40}$')][string]$RobotCommit,
    [ValidateSet('easy','hard','both')][string]$RobotLane='both',
    [ValidatePattern('^[a-z0-9_]+$')][string]$RobotTask,
    [ValidateRange(1,72)][int]$RobotHours=4,
    [switch]$RecoverPrescreen,
    [switch]$EasyFirst,
    [ValidatePattern('^(/[A-Za-z0-9_.-]+)+$')][string]$PilotRoot,
    [ValidatePattern('^[0-9]+$')][string]$PausedEasyJobId,
    [switch]$ParallelEasy,
    [switch]$NativeBudget,
    [ValidatePattern('^[0-9a-f]{40}$')][string]$NativeBudgetCommit,
    [switch]$RootOnly,
    [ValidateSet('reference','expanded','stratified')][string]$ResetPool
)
$ErrorActionPreference='Stop'
$OutputEncoding=[Text.UTF8Encoding]::new($false)
$repoRoot=Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$settings=Import-PowerShellDataFile "$repoRoot/tools/idac/settings.psd1"
$branch=$settings.Branch
$data="$($settings.DataRoot)/xvla"
$prefix="$($settings.CondaRoot)/envs/xvla_rlinf"
$robotData="$data/robotwin-b0"
$robotPrefix="$($settings.CondaRoot)/envs/xvla_robotwin_b0"
$robotCode="$($settings.CodeRoot)-RobotwinB0"
function Invoke-XRemote([string]$Script) {
    $preamble="export FYP_DATA_ROOT='$($settings.DataRoot)' FYP_CONDA_ROOT='$($settings.CondaRoot)' FYP_XVLA_DATA='$data' FYP_XVLA_ENV='$prefix' FYP_ROBOTWIN_DATA='$robotData' FYP_ROBOTWIN_ENV='$robotPrefix' FYP_ROBOTWIN_CODE='$robotCode'`n"
    if ($Command -in @('e2-start','e3-start','final-submit')) {
        $preamble += @'
if [[ -f "$FYP_XVLA_DATA/protocol_20261110/archive.json" ]]; then
    echo 'X-VLA protocol 2026-11-10 is archived. See docs/archive/2026-11-16-experiment-closeout.md.' >&2
    exit 1
fi
'@ + "`n"
    }
    ($preamble+$Script).Replace("`r",'') | & ssh -T -o BatchMode=yes -o ConnectTimeout=15 $settings.SshHost "tr -d '\r' | bash -se"
    if ($LASTEXITCODE -ne 0) { throw "X-VLA remote failed: $LASTEXITCODE" }
}
$sha=(& git -C $repoRoot rev-parse HEAD).Trim()
$snapshot="$($settings.RunCodeRoot)/$sha"
$robotSha=$sha
if ($RobotCommit) { $robotSha=$RobotCommit }
if ($NativeBudget -and -not $RobotCommit) { $robotSha='9baa17360abe242cb36e8f9c78f0aadbe64b3bee' }
$robotSnapshot="$($settings.RunCodeRoot)/$robotSha"
$nativeSha=if ($NativeBudgetCommit) { $NativeBudgetCommit } else { $sha }
$nativeSnapshot="$($settings.RunCodeRoot)/$nativeSha"
$nativeOutput="$robotData/native-budget/$nativeSha"
switch ($Command) {
 'robotwin-run' {
  if ($Value -notmatch '^[0-9]+$') { throw 'Requires the CuRobo Slurm build job ID' }
  if ($NativeBudget) {
   Invoke-XRemote @"
test -f '$nativeSnapshot/tools/xvla/robotwin/native_budget_dispatch.py'
source '$robotSnapshot/tools/xvla/robotwin/runtime.env'
'$robotPrefix/bin/python' '$nativeSnapshot/tools/xvla/robotwin/native_budget.py' --prepare-only --pinned-evaluation-root '$robotSnapshot' --input-panel '$robotData/panels/$robotSha' --output-root '$nativeOutput' --robotwin-root '$robotCode/RoboTwin-'"`$ROBOTWIN_REV" --xvla-root '$robotCode/X-VLA-'"`$XVLA_REV" --checkpoint '$robotData/models/X-VLA-RoboTwin2--'"`${MODEL_REV:0:12}"
'$robotPrefix/bin/python' '$nativeSnapshot/tools/xvla/robotwin/native_budget_report.py' --input '$nativeOutput' --source-panel '$robotData/panels/$robotSha' --output '$nativeOutput/reports'
nohup '$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' -u '$nativeSnapshot/tools/xvla/robotwin/native_budget_dispatch.py' --evaluation-root '$robotSnapshot' --output-root '$nativeOutput' --build-job '$Value' --partition '$($settings.Partition)' --gpu '$GpuModel' --hours '$RobotHours' >> '$nativeOutput/dispatch.log' 2>&1 < /dev/null &
echo ROBOTWIN_NATIVE_BUDGET_DISPATCH_PID=`$!
"@
  } else {
  $recoveryArg=if ($RecoverPrescreen) { '--recover-prescreen' } else { '' }
  $priorityArg=if ($EasyFirst) { '--easy-first' } else { '' }
  if ($PilotRoot -match '/\.\.?(/|$)') { throw 'Use a resolved fixed worktree for -PilotRoot' }
  $pilotArg=if ($PilotRoot) { "--pilot-root '$PilotRoot'" } else { '' }
  $pauseArg=if ($PausedEasyJobId) { "--paused-easy-job '$PausedEasyJobId'" } else { '' }
  $parallelArg=if ($ParallelEasy) { '--parallel-easy' } else { '' }
  Invoke-XRemote @"
test -f '$snapshot/tools/xvla/robotwin/dispatch.py'
test "`$(git -C '$robotSnapshot' rev-parse HEAD)" = '$robotSha'
mkdir -p '$robotData/dispatch/$robotSha'
nohup flock -n '$robotData/dispatch.lock' '$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' -u '$snapshot/tools/xvla/robotwin/dispatch.py' --evaluation-root '$robotSnapshot' --build-job '$Value' --partition '$($settings.Partition)' --gpu '$GpuModel' --hours '$RobotHours' $recoveryArg $priorityArg $pilotArg $pauseArg $parallelArg >> '$robotData/dispatch/$robotSha/dispatch.log' 2>&1 < /dev/null &
echo ROBOTWIN_DISPATCH_PID=`$!
"@
  }
 }
 'robotwin-submit' {
  if ($Value -notin @('build','probe','smoke','coverage','formal','extension')) { throw 'Requires build, probe, smoke, coverage, formal, or extension' }
  Invoke-XRemote @"
test -f '$robotSnapshot/tools/xvla/robotwin/submit.sh'
bash '$robotSnapshot/tools/xvla/robotwin/submit.sh' '$Value' '$($settings.Partition)' '$GpuModel' '$RobotLane' '$RobotHours' '$RobotTask'
"@
 }
 'robotwin-report' {
  if ($Value -notin @('smoke','formal')) { throw 'Requires smoke or formal' }
  if ($NativeBudget) {
   if ($Value -ne 'formal') { throw 'Native-budget reevaluation uses the formal cohort' }
   Invoke-XRemote "'$robotPrefix/bin/python' '$nativeSnapshot/tools/xvla/robotwin/native_budget_report.py' --input '$nativeOutput' --source-panel '$robotData/panels/$robotSha' --output '$nativeOutput/reports'"
  } else {
  Invoke-XRemote @"
'$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' '$robotSnapshot/tools/xvla/robotwin/summarize.py' --input '$robotData/panels/$robotSha' --output '$robotData/reports/$robotSha/$Value' --phase '$Value'
"@
  }
 }
 'robotwin-prepare' {
  if ($Value -notin @('env','assets','link')) { throw 'Requires env, assets, or link' }
  Invoke-XRemote @"
test -f '$robotSnapshot/tools/xvla/robotwin/prepare.sh'
mkdir -p '$robotData/logs'
nohup flock -n '$robotData/prepare-$Value.lock' bash '$robotSnapshot/tools/xvla/robotwin/prepare.sh' '$Value' >> '$robotData/logs/prepare-$Value.log' 2>&1 < /dev/null &
echo ROBOTWIN_PREPARE_PID=`$!
"@
 }
 'robotwin-extract' {
  Invoke-XRemote @"
tool='$robotSnapshot/tools/xvla/robotwin'
test -f "`$tool/extract_assets.sh"
source "`$tool/runtime.env"
test -s '$robotData/archives'/"`$ASSETS_REV/verified.json"
mkdir -p '$robotData/logs' '$robotData/evidence'
printf -v command '%q ' bash "`$tool/extract_assets.sh"
jobid=`$(sbatch --parsable --partition='$($settings.Partition)' --nodelist=4090node3 --exclude=4090node1 --ntasks=1 --cpus-per-task=2 --time=01:00:00 --job-name=rw-b0-extract --output='$robotData/logs/%j.out' --error='$robotData/logs/%j.err' --export=ALL --wrap="`$command")
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "`$jobid" extract both cpu '$robotSha' '' >> '$robotData/evidence/jobs.tsv'
nohup bash "`$tool/extract_assets.sh" lease "`${jobid%%;*}" >> '$robotData/logs/prepare-assets-handoff.log' 2>&1 < /dev/null &
echo ROBOTWIN_JOB_ID="`$jobid"
"@
 }
 'robotwin-status' {
  Invoke-XRemote @'
squeue -u "$(id -un)" -o '%i|%j|%T|%M|%b|%R'
if [[ -f "$FYP_ROBOTWIN_DATA/evidence/jobs.tsv" ]]; then echo '--- submitted jobs'; tail -n 10 "$FYP_ROBOTWIN_DATA/evidence/jobs.tsv"; fi
for record in "$FYP_ROBOTWIN_DATA"/dispatch/*/dispatch.log; do
  if [[ -f "$record" ]]; then echo "--- $(dirname "$record")"; tail -n 4 "$record"; fi
done
python3 - "$FYP_ROBOTWIN_DATA" <<'PY'
import json,sys
from pathlib import Path
for profile in sorted((Path(sys.argv[1])/'native-budget').glob('*/profile.json'),key=lambda path:path.stat().st_mtime):
    root=profile.parent
    state=json.loads((root/'state.json').read_text()) if (root/'state.json').is_file() else {}
    print('--- native-budget',root.name,'stage',state.get('stage','prepared'))
    if state.get('superseded_by'):
        print('superseded by',state['superseded_by'])
        continue
    report=root/'reports/summary.json'
    if report.is_file():
        summary=json.loads(report.read_text())
        total=summary['overall']['cumulative']
        print('native-budget Easy',str(total['finished'])+'/'+str(total['planned']),'completed,',total['unresolved'],'unresolved')
        for task in summary['tasks']:
            first=task['batches']['1']
            rate=first['success_rate']
            print(task['task_id'],str(first['finished'])+'/100,',first['successes'],'successes,',str(round(rate*100,2))+'%' if rate is not None else 'pending')
    log=root/'dispatch.log'
    if log.is_file():
        print('\n'.join(log.read_text().splitlines()[-2:]))
PY
for log in "$FYP_ROBOTWIN_DATA"/logs/{prepare-*,download-robotwin}.log; do
  if [[ -f "$log" ]]; then echo "--- ${log##*/}"; tail -n 4 "$log"; fi
done
'@
 }
 'robotwin-logs' {
  if ($Value -notmatch '^(\d+|prepare-(env|assets|link)|download-robotwin)$') { throw 'Requires a job ID or preparation status name' }
  Invoke-XRemote "for ext in log out err; do if test -f '$robotData/logs/$Value'.`$ext; then tail -n 45 '$robotData/logs/$Value'.`$ext; fi; done"
 }
 'e2-start' {
  if (-not ($DeployedJobId -and $OnlineJobId -and $AuditJobId)) {
   throw 'E2 start requires the deployed, online and physical E1 job IDs'
  }
  Invoke-XRemote @"
python='$($settings.CondaRoot)/envs/vla_budget_rl/bin/python'
repo='$snapshot'
state='$data/protocol_20261110/e2_dispatch.json'
if [[ -f "`$state" ]]; then
  commit=`$("`$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "`$state")
  [[ "`$commit" =~ ^[a-f0-9]{40}`$ ]]
  repo='$($settings.RunCodeRoot)'/"`$commit"
fi
"`$python" "`$repo/tools/xvla/e2_dispatch.py" start --repo "`$repo" --data '$data' --partition '$($settings.Partition)' --deployed-job-id '$DeployedJobId' --online-job-id '$OnlineJobId' --physical-job-id '$AuditJobId'
"@
 }
 'e2-status' {
  Invoke-XRemote "cat '$data/protocol_20261110/e2_dispatch.json'"
 }
 'e3-start' {
  if (-not $DevJobId) { throw 'E3 start requires the paired development benchmark job ID' }
  if ($FailedJobId -and $FallbackJobId) { throw 'Choose a failed retry or a pending GPU fallback' }
  $retryOptions=''
  if ($FailedJobId) { $retryOptions=" --failed-job-id $FailedJobId" }
  if ($FallbackJobId) { $retryOptions=" --fallback-job-id $FallbackJobId" }
  Invoke-XRemote @"
python='$($settings.CondaRoot)/envs/vla_budget_rl/bin/python'
repo='$snapshot'
state='$data/protocol_20261110/e3_dispatch.json'
if [[ -z '$retryOptions' && -f "`$state" ]]; then
  commit=`$("`$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["commit"])' "`$state")
  [[ "`$commit" =~ ^[a-f0-9]{40}`$ ]]
  repo='$($settings.RunCodeRoot)'/"`$commit"
fi
"`$python" "`$repo/tools/xvla/e3_dispatch.py" start --repo "`$repo" --data '$data' --partition '$($settings.Partition)' --dev-job-id '$DevJobId'$retryOptions
"@
 }
 'e3-status' {
  Invoke-XRemote "cat '$data/protocol_20261110/e3_dispatch.json'"
 }
 'final-submit' {
  Invoke-XRemote @"
python='$($settings.CondaRoot)/envs/vla_budget_rl/bin/python'
repo='$snapshot'
plan='$data/protocol_20261110/final_h_plan.json'
if [[ -f "`$plan" ]]; then
  commit=`$("`$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["evaluation_commit"])' "`$plan")
  [[ "`$commit" =~ ^[a-f0-9]{40}`$ ]]
  repo='$($settings.RunCodeRoot)'/"`$commit"
fi
"`$python" "`$repo/tools/xvla/final_panel.py" submit --repo "`$repo" --data '$data' --partition '$($settings.Partition)'
"@
 }
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
  $makeSnapshot=if ($RootOnly) { 'no' } else { 'yes' }
  & git -C $repoRoot diff --quiet
  if ($LASTEXITCODE -ne 0) { throw 'Commit tracked changes before sync' }
  & git -C $repoRoot diff --cached --quiet
  if ($LASTEXITCODE -ne 0) { throw 'Commit staged changes before sync' }
  Invoke-XRemote @"
if [[ -n `$(git -C '$($settings.CodeRoot)' status --porcelain) ]]; then
  echo 'Server RLinf checkout has changes. Commit or stash them explicitly before syncing.' >&2
  exit 1
fi
"@
  & git -C $repoRoot push origin "HEAD:refs/heads/$branch"
  if ($LASTEXITCODE -ne 0) { throw 'Push failed' }
  Invoke-XRemote @"
code='$($settings.CodeRoot)'
branch='$branch'
exec 9>"`$code/.git/fyp-xvla-sync.lock"
flock 9
if [[ -n `$(git -C "`$code" status --porcelain) ]]; then
  echo 'Server RLinf checkout changed during sync. Review it before updating.' >&2
  exit 1
fi
refspec="+refs/heads/`$branch`:refs/remotes/origin/`$branch"
if ! git -C "`$code" config --get-all remote.origin.fetch | grep -Fxq -e "`$refspec" -e '+refs/heads/*:refs/remotes/origin/*'; then
  git -C "`$code" config --add remote.origin.fetch "`$refspec"
fi
git -C "`$code" fetch origin "refs/heads/`$branch`:refs/remotes/origin/`$branch"
test "`$(git -C "`$code" rev-parse "origin/`$branch")" = '$sha'
current=`$(git -C "`$code" symbolic-ref --quiet --short HEAD || true)
if [[ -z "`$current" ]] && ! git -C "`$code" merge-base --is-ancestor HEAD '$sha'; then
  echo 'Detached server HEAD is not part of the incoming branch; preserve it before switching.' >&2
  exit 1
fi
if [[ "`$current" != "`$branch" ]]; then
  if git -C "`$code" show-ref --verify --quiet "refs/heads/`$branch"; then
    git -C "`$code" switch "`$branch"
  else
    git -C "`$code" switch --create "`$branch" --track "origin/`$branch"
  fi
fi
git -C "`$code" merge --ff-only "origin/`$branch"
test "`$(git -C "`$code" rev-parse HEAD)" = '$sha'
git -C "`$code" branch --set-upstream-to="origin/`$branch" "`$branch"
git -C "`$code" submodule update --quiet --init third_party/rlinf
if [[ '$makeSnapshot' == yes ]]; then
  test -e '$snapshot' || git -C "`$code" worktree add --quiet --detach '$snapshot' '$sha'
  test "`$(git -C '$snapshot' rev-parse HEAD)" = '$sha'
  git -C '$snapshot' submodule update --quiet --init third_party/rlinf
  test "`$(git -C '$snapshot/third_party/rlinf' rev-parse HEAD)" = 1bf78980d9339cbc67497eddceb9a1e3a74e2acc
  echo 'XVLA_JOB_SNAPSHOT $snapshot'
fi
echo 'XVLA_SYNC_OK branch=$branch root=$($settings.CodeRoot) commit=$sha'
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
  $downloadData=$data
  if ($Model -eq 'robotwin') { $downloadData="$data/robotwin-b0" }
  Invoke-XRemote @"
test -f '$snapshot/tools/xvla/download.py'
mkdir -p '$downloadData/logs'
nohup flock -n '$downloadData/download-$Model.lock' '$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' '$snapshot/tools/xvla/download.py' '$downloadData/models' '$Model' > '$downloadData/logs/download-$Model.log' 2>&1 < /dev/null &
echo XVLA_DOWNLOAD_PID=`$!
"@
 }
 'submit' {
  if ($Value -notmatch '^[a-z0-9_-]+$' -or $Gpus -notin @(0,1,2)) { throw 'Invalid X-VLA mode or GPU count' }
  $submitOptions=''
  $isSequenceDev=$Value -in @('benchmark_sft_sequence_dev','benchmark_rl_sequence_dev')
  $isE1Panel=$Value -in @('benchmark_e1_deployed','benchmark_e1_online')
  $isProtocolDev=$Value -eq 'benchmark_protocol_dev'
  $isBenchmark=$isE1Panel -or $isSequenceDev -or $Value -in @('benchmark_libero','benchmark_pt','benchmark_sft_smoke','benchmark_sft_fewshot','benchmark_sft_source10','benchmark_rl_source10')
  $isPtRl=$Value -in @('ppo_pt','grpo_pt','ppo_source10','grpo_source10','ppo_source10_pool','ppo_e2')
  if ($Value -in @('ppo_source10_pool','ppo_e2')) {
   if (-not $ResetPool) { throw 'Source pool pilot requires -ResetPool' }
   $submitOptions += " --reset-pool $ResetPool"
  } elseif ($ResetPool) {
   throw '-ResetPool applies only to the source pool pilot'
  }
  $isSourceResetEval=$Value -in @('source_reset_solvability','source_reset_sft')
  if ($isSourceResetEval) {
   if (-not $AuditJobId) { throw 'Source reset evaluation requires -AuditJobId' }
   $submitOptions += " --audit-job-id $AuditJobId"
  } elseif ($AuditJobId) {
   throw '-AuditJobId applies only to source reset evaluation'
  }
  if ($isBenchmark) {
   $submitOptions += " --start-task $StartTask --task-count $TaskCount --start-trial $StartTrial --trials $Trials"
  } elseif ($StartTask -ne 0 -or $TaskCount -ne 10 -or $StartTrial -ne 0 -or $Trials -ne 5) {
   throw 'Benchmark shard options require a benchmark mode'
  }
  $isSourceSft=$Value -in @('sft_source10','sft_source10_lora')
  if ($isBenchmark -or $isProtocolDev -or $isSourceSft -or $isPtRl -or $isSourceResetEval -or $Value -in @('convert_fewshot','sft_fewshot','replay_absolute','sft_action_check','convert_source10')) {
   $submitOptions += " --gpu-model $GpuModel"
  } elseif ($PSBoundParameters.ContainsKey('GpuModel')) {
   throw 'GPU model override applies only to benchmark and Pt few-shot pipeline modes'
  }
  if ($isPtRl -or $isSequenceDev -or $isE1Panel -or $Value -in @('benchmark_sft_smoke','benchmark_sft_fewshot','benchmark_sft_source10','benchmark_rl_source10','sft_source10_lora','sft_action_check','source_reset_sft')) {
   if (-not $CheckpointJobId) { throw "$Value requires -CheckpointJobId" }
   $submitOptions += " --checkpoint-job-id $CheckpointJobId"
  } elseif ($CheckpointJobId) {
   throw 'Checkpoint job ID requires an SFT benchmark or Pt RL mode'
  }
  if ($isSourceSft -or $isPtRl -or $isE1Panel -or $Value -in @('sft_fewshot','benchmark_sft_fewshot','sft_action_check','benchmark_sft_source10','benchmark_sft_sequence_dev','source_reset_sft')) {
   $submitOptions += " --sft-steps $SftSteps"
  } elseif ($PSBoundParameters.ContainsKey('SftSteps')) {
   throw 'SFT steps apply only to few-shot SFT and its consumers'
  }
  Invoke-XRemote @"
test -f '$snapshot/tools/xvla/job.sh'
mkdir -p '$data/logs'
'$($settings.CondaRoot)/envs/vla_budget_rl/bin/python' '$snapshot/tools/xvla/submit.py' --repo '$snapshot' --data '$data' --mode '$Value' --gpus $Gpus --partition '$($settings.Partition)'$submitOptions
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
