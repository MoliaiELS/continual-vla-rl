"""Submit bounded X-VLA jobs within the existing two-GPU user budget."""
import argparse,json,os,re,subprocess
import hashlib
from pathlib import Path

TASK0_HDF5='pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5'
BENCHMARK_MODELS = {
    'benchmark_libero': 'libero',
    'benchmark_pt': 'pt',
    'benchmark_sft_smoke': 'sft-smoke',
    'benchmark_sft_fewshot': 'sft-fewshot',
    'benchmark_sft_source10': 'sft-fewshot',
    'benchmark_rl_source10': 'rl-source10',
    'benchmark_sft_sequence_dev': 'sft-fewshot',
    'benchmark_rl_sequence_dev': 'rl-source10',
    'benchmark_e1_deployed': 'sft-fewshot',
    'benchmark_e1_online': 'sft-fewshot',
}
SEQUENCE_DEV_MODES = ('benchmark_sft_sequence_dev', 'benchmark_rl_sequence_dev')
E1_PANELS = {'benchmark_e1_deployed': (10, 20), 'benchmark_e1_online': (30, 10)}
PROTOCOL_DEV_MODE = 'benchmark_protocol_dev'
CPU_RESET_AUDITS = ('source_reset_audit', 'e1_source_audit')
SOURCE_RL_MODES = ('ppo_source10', 'grpo_source10', 'ppo_source10_pool', 'ppo_e2')
PT_RL_MODES = ('ppo_pt', 'grpo_pt', *SOURCE_RL_MODES)
SOURCE_DATASET = Path('datasets/source-spatial-task0-10')
CONVERSION_MODES = ('convert', 'convert_fewshot', 'convert_source10')
GPU_MODEL_MODES = (*BENCHMARK_MODELS, *PT_RL_MODES, PROTOCOL_DEV_MODE, 'convert_fewshot', 'sft_fewshot',
                   'replay_absolute', 'sft_action_check', 'convert_source10', 'sft_source10', 'sft_source10_lora',
                   'source_reset_solvability', 'source_reset_sft')
FEWSHOT_DATASET = Path('datasets/fewshot-spatial-task0-5')
FEWSHOT_SFT_RUN = 'sft_fewshot-smoke'
FEWSHOT_SFT_STEPS = (256, 1024)
FEWSHOT_SFT_MODES = ('sft_fewshot', 'benchmark_sft_fewshot', 'sft_action_check', 'sft_source10',
                     'sft_source10_lora', 'benchmark_sft_source10', 'benchmark_sft_sequence_dev',
                     'source_reset_sft', *E1_PANELS, *PT_RL_MODES)

def discover_task0_source(data_root: Path):
    """Read only our project's raw-data tree and pinned HF download receipt."""
    raw_root=data_root.resolve()/'datasets'/'libero_raw'
    matches=sorted({path.resolve() for path in raw_root.rglob(TASK0_HDF5) if path.is_file()})
    if len(matches)!=1:
        raise RuntimeError(f'Expected exactly one task0 source under {raw_root}; found {matches}')
    source=matches[0]
    relative=source.relative_to(raw_root)
    metadata=raw_root/'.cache/huggingface/download'/Path(str(relative)+'.metadata')
    lines=metadata.read_text().splitlines()
    if len(lines)<2 or not re.fullmatch('[0-9a-f]{40}',lines[0]) or not re.fullmatch('[0-9a-f]{64}',lines[1]):
        raise ValueError(f'Raw dataset lacks a pinned revision/SHA256 download receipt: {metadata}')
    return {'source_hdf5':str(source),'data_revision':lines[0],'source_sha256':lines[1],
            'download_receipt':str(metadata)}


def checked_sft_checkpoint(args, run_name, step):
    if not re.fullmatch('[0-9]+', args.checkpoint_job_id or ''):
        raise ValueError(f'{args.mode} requires a numeric --checkpoint-job-id')
    checkpoint = (args.data.resolve() / 'runs' / args.checkpoint_job_id / run_name
                  / 'checkpoints' / f'global_step_{step}' / 'actor')
    report = checkpoint / 'checkpoint_check.json'
    if not report.is_file() or json.loads(report.read_text(encoding='utf-8')).get('status') != 'PASS':
        raise ValueError(f'SFT checkpoint requires an independent load PASS: {report}')
    return str(checkpoint)


def benchmark_settings(args):
    if args.gpu_model is not None and args.mode not in GPU_MODEL_MODES:
        raise ValueError('--gpu-model applies only to benchmark and Pt few-shot pipeline modes')
    model = BENCHMARK_MODELS.get(args.mode)
    if model is None:
        if (args.start_task, args.task_count, args.start_trial, args.trials) != (0, 10, 0, 5):
            raise ValueError('Benchmark shard options require a benchmark mode')
        if args.checkpoint_job_id is not None and args.mode not in (
            *PT_RL_MODES, 'sft_action_check', 'sft_source10_lora', 'source_reset_sft'
        ):
            raise ValueError('--checkpoint-job-id requires an SFT benchmark, warm-start, action check or Pt RL mode')
        return None

    if not 0 <= args.start_task < args.start_task + args.task_count <= 10:
        raise ValueError('Benchmark shard must be a nonempty subset of Spatial task IDs 0–9')
    if args.start_trial < 0 or args.trials < 1 or args.task_count * args.trials > 50:
        raise ValueError('Benchmark shard must contain between 1 and 50 episodes')
    if args.mode in SEQUENCE_DEV_MODES and (args.start_task, args.task_count) != (1, 1):
        raise ValueError('Sequence development benchmark is restricted to Spatial task 1')
    if args.mode in E1_PANELS and (args.start_task, args.task_count, args.start_trial, args.trials) != (
        0, 1, *E1_PANELS[args.mode]
    ):
        raise ValueError('E1 benchmark must use its frozen task-0 panel')
    if args.mode in E1_PANELS and (args.checkpoint_job_id, args.sft_steps) != ('19525', 1024):
        raise ValueError('E1 benchmark must use the frozen B0 checkpoint')

    checkpoint = None
    if model == 'sft-smoke':
        checkpoint = checked_sft_checkpoint(args, 'sft-smoke', 1)
    elif model == 'rl-source10':
        if args.mode == 'benchmark_rl_source10' and (args.start_task, args.task_count) != (0, 1):
            raise ValueError('Source-10 RL development evaluation is restricted to Spatial task 0')
        checkpoint = checked_sft_checkpoint(args, 'ppo_source10-smoke', 1)
    elif model == 'sft-fewshot':
        run = ('sft_source10' if args.mode in ('benchmark_sft_source10', 'benchmark_sft_sequence_dev', *E1_PANELS)
               else FEWSHOT_SFT_RUN)
        if args.mode == 'benchmark_sft_source10' and (args.start_task, args.task_count) != (0, 1):
            raise ValueError('Source-10 development evaluation is restricted to Spatial task 0')
        checkpoint = checked_sft_checkpoint(args, run, args.sft_steps)
    elif args.checkpoint_job_id is not None:
        raise ValueError('--checkpoint-job-id applies only to SFT benchmarks and Pt RL')

    return {'model': model, 'start_task': args.start_task, 'task_count': args.task_count, 'start_trial': args.start_trial,
            'trials': args.trials, 'checkpoint_job_id': args.checkpoint_job_id,
            'checkpoint': checkpoint}


def conversion_settings(args):
    fewshot = args.mode in ('convert_fewshot', 'convert_source10')
    if fewshot and (args.convert_output is not None or args.convert_demonstrations is not None or args.seed != 1234):
        raise ValueError('Controlled conversion fixes the destination, demonstration count and seed 1234')
    if not os.environ.get('FYP_DATA_ROOT'):
        raise RuntimeError('FYP_DATA_ROOT must be supplied from current project settings')
    if args.mode == 'convert_source10':
        require_fewshot_manifest(args.data)
        output = args.data.resolve() / SOURCE_DATASET
        demonstrations = 10
    elif fewshot:
        output = args.data.resolve() / FEWSHOT_DATASET
        demonstrations = 5
    else:
        output = (args.convert_output or args.data / 'datasets/limited-spatial-task0').resolve()
        demonstrations = args.convert_demonstrations or 1
    if fewshot:
        if output.exists() or output.is_symlink():
            raise FileExistsError(f'Few-shot destination already exists: {output}; inspect/reuse it')
    else:
        if (output / 'manifest.json').exists():
            raise FileExistsError(f'Converted manifest already exists: {output}/manifest.json; inspect/reuse it')
    conversion = discover_task0_source(Path(os.environ['FYP_DATA_ROOT']))
    conversion.update(output_dir=str(output), selection_seed=args.seed,
                      requested_demonstrations=demonstrations)
    return conversion


def require_fewshot_manifest(data, dataset=FEWSHOT_DATASET, demonstrations=5):
    path = data.resolve() / dataset / 'manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    episodes = manifest.get('episodes', [])
    if (manifest.get('requested_demonstrations') != demonstrations or manifest.get('demonstrations') != demonstrations
            or manifest.get('selection_seed') != 1234 or len(episodes) != demonstrations
            or len({episode['demo_id'] for episode in episodes}) != demonstrations
            or any(episode['task_suite'] != 'libero_spatial' or episode['task_id'] != 0
                   or not episode.get('success_final') or not episode.get('controller_replay_verified')
                   for episode in episodes)):
        label = 'five' if demonstrations == 5 else str(demonstrations)
        raise ValueError(f'Expected {label} verified task-0 demonstrations selected with seed 1234: {path}')
    return path


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--mode', choices=[
        'unit', 'imports', *CPU_RESET_AUDITS, 'source_reset_solvability', 'source_reset_sft',
        'alignment', 'native_eval', 'stochastic_eval', *CONVERSION_MODES,
        'sft', 'ppo', 'grpo', 'grpo_exploration', 'grpo_stress',
        'sft_fewshot', 'sft_source10', 'sft_source10_lora', 'replay_absolute', 'sft_action_check',
        *PT_RL_MODES, *BENCHMARK_MODELS, PROTOCOL_DEV_MODE,
    ], required=True)
    p.add_argument('--gpus',type=int,choices=[0,1,2],required=True);p.add_argument('--partition',required=True)
    p.add_argument('--gpu-model', choices=['3090', '4090'])
    p.add_argument('--start-task', type=int, default=0)
    p.add_argument('--task-count', type=int, default=10)
    p.add_argument('--start-trial', type=int, default=0)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--checkpoint-job-id')
    p.add_argument('--audit-job-id')
    p.add_argument('--reset-pool', choices=('reference', 'expanded', 'stratified'))
    p.add_argument('--sft-steps', type=int, choices=FEWSHOT_SFT_STEPS, default=FEWSHOT_SFT_STEPS[0])
    p.add_argument('--convert-output',type=Path)
    p.add_argument('--convert-demonstrations',type=int,choices=[1,2])
    p.add_argument('--seed',type=int,default=1234)
    a=p.parse_args()
    if (a.audit_job_id is not None) != (a.mode in ('source_reset_solvability', 'source_reset_sft')):
        raise ValueError('--audit-job-id is required only for source reset evaluation')
    if (a.reset_pool is not None) != (a.mode in ('ppo_source10_pool', 'ppo_e2')):
        raise ValueError('--reset-pool is required only for the source pool pilot')
    if a.mode == 'ppo_e2' and (a.checkpoint_job_id, a.sft_steps, a.gpu_model) != ('19525', 1024, '4090'):
        raise ValueError('E2 requires frozen B0 checkpoint 19525/1024 and a single 4090')
    if a.mode in CPU_RESET_AUDITS and a.gpus != 0:
        raise ValueError('Unrendered source reset audit requires --gpus 0')
    if a.mode not in ('unit', 'imports', *CPU_RESET_AUDITS) and a.gpus!=1:
        raise ValueError('Model, simulation and online audit modes require exactly one GPU')
    if a.sft_steps != FEWSHOT_SFT_STEPS[0] and a.mode not in FEWSHOT_SFT_MODES:
        raise ValueError('--sft-steps applies only to few-shot SFT and its consumers')
    benchmark = benchmark_settings(a)
    conversion=None
    initial_sft = None
    extra=[]
    if benchmark is not None:
        extra = [str(a.start_task), str(a.task_count), str(a.trials), str(a.start_trial)]
        if benchmark['checkpoint'] is not None:
            extra.append(benchmark['checkpoint'])
    if a.mode == PROTOCOL_DEV_MODE:
        from protocol_dev import development_plan

        benchmark = development_plan(a.repo, a.data, a.partition)
        extra = [json.dumps(benchmark, separators=(',', ':'))]
    if a.mode in CONVERSION_MODES:
        conversion = conversion_settings(a)
        extra = [str(conversion[key]) for key in (
            'source_hdf5', 'output_dir', 'data_revision', 'source_sha256',
            'selection_seed', 'requested_demonstrations',
        )]
    if a.mode == 'sft_fewshot':
        require_fewshot_manifest(a.data)
        extra = [str(a.sft_steps)]
    if a.mode in ('sft_source10', 'sft_source10_lora'):
        manifest = require_fewshot_manifest(a.data, SOURCE_DATASET, 10)
        source = json.loads(manifest.read_text(encoding='utf-8'))
        if (source.get('absolute_replay_required') != 'success_final'
                or any(not episode.get('absolute_replay', {}).get('success_final')
                       for episode in source['episodes'])):
            raise ValueError('Source-10 SFT requires successful relative and absolute replay of every demonstration')
        extra = [str(a.sft_steps), str(manifest)]
        if a.mode == 'sft_source10_lora':
            extra.append(checked_sft_checkpoint(a, 'sft_source10', 1024))
    if a.mode == 'replay_absolute':
        extra = [str(require_fewshot_manifest(a.data))]
    if a.mode == 'sft_action_check':
        manifest = require_fewshot_manifest(a.data)
        checkpoint = checked_sft_checkpoint(a, FEWSHOT_SFT_RUN, a.sft_steps)
        extra = [str(manifest), checkpoint]
    if a.mode in ('source_reset_solvability', 'source_reset_sft'):
        if not re.fullmatch('[0-9]+', a.audit_job_id):
            raise ValueError('Source reset solvability requires a numeric audit job ID')
        audit = a.data.resolve() / 'runs' / a.audit_job_id / 'source_reset_audit.json'
        if not audit.is_file():
            raise FileNotFoundError(audit)
        extra = [str(audit)]
        if a.mode == 'source_reset_sft':
            extra.append(checked_sft_checkpoint(a, 'sft_source10', a.sft_steps))
    if a.mode in PT_RL_MODES:
        run = 'sft_source10' if a.mode in SOURCE_RL_MODES else FEWSHOT_SFT_RUN
        checkpoint = checked_sft_checkpoint(a, run, a.sft_steps)
        initial_sft = {'checkpoint_job_id': a.checkpoint_job_id, 'checkpoint': checkpoint, 'sft_steps': a.sft_steps}
        extra = [checkpoint]
        if a.reset_pool is not None:
            sampling = 'stratified' if a.reset_pool == 'stratified' else 'uniform'
            plan_name = 'expanded' if a.reset_pool == 'stratified' else a.reset_pool
            plan = a.repo.resolve() / 'configs/xvla/source_pools' / f'{plan_name}.json'
            digest = hashlib.sha256(plan.read_bytes()).hexdigest()
            extra.extend((str(plan), digest, sampling))
            if a.mode == 'ppo_e2':
                extra.extend((a.reset_pool, a.partition))
    import fcntl
    a.data.mkdir(parents=True,exist_ok=True)
    with (a.data/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        submit_locked(a,conversion,extra,benchmark,initial_sft)


def gpu_commitment(info):
    # Use the largest accounting description; generic/type-specific TRES
    # duplicate the same GPU and must not be summed across descriptions.
    totals=[int(value) for value in re.findall(r'\bgres/gpu=(\d+)',info)]
    nodes=re.search(r'NumNodes=(\d+)(?:-(\d+))?',info)
    node_count=int((nodes[2] or nodes[1])) if nodes else 1
    for field,multiplier in [('TresPerNode',node_count),('TresPerJob',1)]:
        for value in re.findall(field+r'=([^ ]+)',info):
            total=sum(int(gpu[1]) for gpu in re.finditer(r'(?:gres/)?gpu(?::[^,:=]+)?[:=](\d+)',value))
            totals.append(total*multiplier)
    return max(totals,default=0)


def submit_locked(a,conversion,extra,benchmark=None,initial_sft=None,*,walltime=None,hold=False):
    allocated=0
    # squeue's active set includes pending reservations as well as running jobs.
    ids=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i'],text=True).split()
    for job in ids:
        info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        allocated+=gpu_commitment(info)
    if a.gpus+allocated>2:
        raise RuntimeError(f'BLOCKED_GPU_BUDGET: requested={a.gpus}, own active/pending commitment={allocated}, cap=2; no job submitted')
    if a.mode not in ('unit', 'imports', *CPU_RESET_AUDITS) and a.gpus==0:raise ValueError('Real model/rollout smoke requires explicit GPU allocation')
    walltime = walltime or {
        'ppo_e2': '04:00:00',
        'benchmark_protocol_dev': '02:00:00',
    }.get(a.mode, '01:00:00')
    cmd=['sbatch','--parsable','--partition='+a.partition,'--exclude=4090node1','--nodes=1','--ntasks=1',
         '--cpus-per-task=8','--time='+walltime,'--job-name=xvla-'+a.mode,
         '--output='+str(a.data/'logs/%j.out'),'--error='+str(a.data/'logs/%j.err')]
    gpu_model = a.gpu_model or '4090'
    if a.gpus:
        cmd.append(f'--gres=gpu:{gpu_model}:{a.gpus}')
    if hold:
        cmd.append('--hold')
    cmd+=[str(a.repo/'tools/xvla/job.sh'),str(a.repo),a.mode,str(a.gpus),*extra]
    try:
        result=subprocess.check_output(cmd,text=True,stderr=subprocess.PIPE).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f'Slurm rejected submission: {error.stderr.strip()}') from error
    record={'job_id':result,'mode':a.mode,'requested_gpus':a.gpus,'own_preexisting_gpus':allocated,
            'command':cmd,'conversion':conversion,'benchmark':benchmark,
            'requested_gpu_model':gpu_model if a.gpus else None}
    if initial_sft is not None:
        record['initial_sft'] = initial_sft
    receipt=a.data/'runs'/('submission-'+result.split(';')[0]+'.json')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
    return record
if __name__=='__main__':main()
