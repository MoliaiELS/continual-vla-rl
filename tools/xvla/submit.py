"""Submit bounded X-VLA jobs within the existing two-GPU user budget."""
import argparse,json,os,re,subprocess
from pathlib import Path

TASK0_HDF5='pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5'
BENCHMARK_MODELS = {
    'benchmark_libero': 'libero',
    'benchmark_pt': 'pt',
    'benchmark_sft_smoke': 'sft-smoke',
}

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


def benchmark_settings(args):
    model = BENCHMARK_MODELS.get(args.mode)
    if model is None:
        if (args.gpu_model is not None or args.checkpoint_job_id is not None
                or (args.start_task, args.task_count, args.trials) != (0, 10, 5)):
            raise ValueError('GPU model and benchmark options require a benchmark mode')
        return None

    if not 0 <= args.start_task < args.start_task + args.task_count <= 10:
        raise ValueError('Benchmark shard must be a nonempty subset of Spatial task IDs 0–9')
    if args.trials < 1 or args.task_count * args.trials > 50:
        raise ValueError('Benchmark shard must contain between 1 and 50 episodes')

    checkpoint = None
    if model == 'sft-smoke':
        if not re.fullmatch('[0-9]+', args.checkpoint_job_id or ''):
            raise ValueError('benchmark_sft_smoke requires a numeric --checkpoint-job-id')
        checkpoint = (args.data.resolve() / 'runs' / args.checkpoint_job_id / 'sft-smoke'
                      / 'checkpoints' / 'global_step_1' / 'actor')
        report = checkpoint / 'checkpoint_check.json'
        if not report.is_file() or json.loads(report.read_text(encoding='utf-8')).get('status') != 'PASS':
            raise ValueError(f'SFT checkpoint requires an independent load PASS: {report}')
    elif args.checkpoint_job_id is not None:
        raise ValueError('--checkpoint-job-id applies only to benchmark_sft_smoke')

    return {'model': model, 'start_task': args.start_task, 'task_count': args.task_count,
            'trials': args.trials, 'checkpoint_job_id': args.checkpoint_job_id,
            'checkpoint': str(checkpoint) if checkpoint is not None else None}


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--mode',choices=['unit','imports','alignment','native_eval','stochastic_eval','convert','sft','ppo','grpo','grpo_exploration','grpo_stress',*BENCHMARK_MODELS],required=True)
    p.add_argument('--gpus',type=int,choices=[0,1,2],required=True);p.add_argument('--partition',required=True)
    p.add_argument('--gpu-model', choices=['3090', '4090'])
    p.add_argument('--start-task', type=int, default=0)
    p.add_argument('--task-count', type=int, default=10)
    p.add_argument('--trials', type=int, default=5)
    p.add_argument('--checkpoint-job-id')
    p.add_argument('--convert-output',type=Path)
    p.add_argument('--convert-demonstrations',type=int,choices=[1,2],default=1)
    p.add_argument('--seed',type=int,default=1234)
    a=p.parse_args()
    if a.mode not in ('unit','imports') and a.gpus!=1:
        raise ValueError('Model, simulation and online audit modes require exactly one GPU')
    benchmark = benchmark_settings(a)
    conversion=None
    extra=[]
    if benchmark is not None:
        extra = [str(a.start_task), str(a.task_count), str(a.trials)]
        if benchmark['checkpoint'] is not None:
            extra.append(benchmark['checkpoint'])
    if a.mode=='convert':
        if a.gpus!=1:raise ValueError('Controller replay smoke requests exactly one GPU')
        if not os.environ.get('FYP_DATA_ROOT'):
            raise RuntimeError('FYP_DATA_ROOT must be supplied from current project settings')
        conversion=discover_task0_source(Path(os.environ['FYP_DATA_ROOT']))
        output=(a.convert_output or a.data/'datasets/limited-spatial-task0').resolve()
        if (output/'manifest.json').exists():
            raise FileExistsError(f'Converted manifest already exists: {output}/manifest.json; inspect/reuse it')
        conversion.update(output_dir=str(output),selection_seed=a.seed,
                          requested_demonstrations=a.convert_demonstrations)
        extra=[conversion['source_hdf5'],str(output),conversion['data_revision'],
               conversion['source_sha256'],str(a.seed),str(a.convert_demonstrations)]
    import fcntl
    a.data.mkdir(parents=True,exist_ok=True)
    with (a.data/'submit.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        submit_locked(a,conversion,extra,benchmark)


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


def submit_locked(a,conversion,extra,benchmark=None):
    allocated=0
    # squeue's active set includes pending reservations as well as running jobs.
    ids=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-o','%i'],text=True).split()
    for job in ids:
        info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        allocated+=gpu_commitment(info)
    if a.gpus+allocated>2:
        raise RuntimeError(f'BLOCKED_GPU_BUDGET: requested={a.gpus}, own active/pending commitment={allocated}, cap=2; no job submitted')
    if a.mode not in ('unit','imports') and a.gpus==0:raise ValueError('Real model/rollout smoke requires explicit GPU allocation')
    cmd=['sbatch','--parsable','--partition='+a.partition,'--exclude=4090node1','--nodes=1','--ntasks=1',
         '--cpus-per-task=8','--time=01:00:00','--job-name=xvla-'+a.mode,
         '--output='+str(a.data/'logs/%j.out'),'--error='+str(a.data/'logs/%j.err')]
    gpu_model = a.gpu_model or '4090'
    if a.gpus:
        cmd.append(f'--gres=gpu:{gpu_model}:{a.gpus}')
    cmd+=[str(a.repo/'tools/xvla/job.sh'),str(a.repo),a.mode,str(a.gpus),*extra]
    result=subprocess.check_output(cmd,text=True).strip()
    record={'job_id':result,'mode':a.mode,'requested_gpus':a.gpus,'own_preexisting_gpus':allocated,
            'command':cmd,'conversion':conversion,'benchmark':benchmark,
            'requested_gpu_model':gpu_model if a.gpus else None}
    receipt=a.data/'runs'/('submission-'+result.split(';')[0]+'.json')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
if __name__=='__main__':main()
