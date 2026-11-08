"""Submit only bounded smoke jobs; check the existing two-GPU user budget."""
import argparse,json,os,re,subprocess
from pathlib import Path

TASK0_HDF5='pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate_demo.hdf5'

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

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--mode',choices=['unit','imports','alignment','native_eval','stochastic_eval','convert','sft','ppo','grpo'],required=True)
    p.add_argument('--gpus',type=int,choices=[0,1,2],required=True);p.add_argument('--partition',required=True)
    p.add_argument('--convert-output',type=Path)
    p.add_argument('--convert-demonstrations',type=int,choices=[1,2],default=1)
    p.add_argument('--seed',type=int,default=1234)
    a=p.parse_args()
    if a.mode in ('alignment','native_eval','stochastic_eval','convert','sft','ppo','grpo') and a.gpus!=1:
        raise ValueError('Model, simulation and online audit smoke modes require exactly one GPU')
    conversion=None
    extra=[]
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
    allocated=0
    ids=subprocess.check_output(['squeue','-h','-u',os.environ['USER'],'-t','RUNNING,COMPLETING','-o','%i'],text=True).split()
    for job in ids:
        info=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        matches=re.findall(r'TresPerNode=([^ ]+)',info)
        for value in matches:
            for gpu in re.finditer(r'(?:gres/)?gpu(?::[^,:=]+)?[:=](\d+)',value):allocated+=int(gpu[1])
    if a.gpus+allocated>2:
        raise RuntimeError(f'BLOCKED_GPU_BUDGET: requested={a.gpus}, own allocated={allocated}, cap=2; no job submitted')
    if a.mode not in ('unit','imports') and a.gpus==0:raise ValueError('Real model/rollout smoke requires explicit GPU allocation')
    cmd=['sbatch','--parsable','--partition='+a.partition,'--exclude=4090node1','--nodes=1','--ntasks=1',
         '--cpus-per-task=8','--time=01:00:00','--job-name=xvla-'+a.mode,
         '--output='+str(a.data/'logs/%j.out'),'--error='+str(a.data/'logs/%j.err')]
    if a.gpus:cmd+=['--gres=gpu:4090:'+str(a.gpus)]
    cmd+=[str(a.repo/'tools/xvla/job.sh'),str(a.repo),a.mode,str(a.gpus),*extra]
    result=subprocess.check_output(cmd,text=True).strip()
    record={'job_id':result,'mode':a.mode,'requested_gpus':a.gpus,'own_preexisting_gpus':allocated,
            'command':cmd,'conversion':conversion}
    receipt=a.data/'runs'/('submission-'+result.split(';')[0]+'.json')
    receipt.parent.mkdir(parents=True,exist_ok=True)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))
if __name__=='__main__':main()
