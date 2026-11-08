"""Submit only bounded smoke jobs; check the existing two-GPU user budget."""
import argparse,json,os,re,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--data',type=Path,required=True)
    p.add_argument('--mode',choices=['unit','imports','alignment','native_eval','sft','ppo','grpo'],required=True)
    p.add_argument('--gpus',type=int,choices=[0,1,2],required=True);p.add_argument('--partition',required=True);a=p.parse_args()
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
    cmd+=[str(a.repo/'tools/xvla/job.sh'),str(a.repo),a.mode,str(a.gpus)]
    result=subprocess.check_output(cmd,text=True).strip()
    print(json.dumps({'job_id':result,'mode':a.mode,'requested_gpus':a.gpus,'own_preexisting_gpus':allocated,'command':cmd}))
if __name__=='__main__':main()
