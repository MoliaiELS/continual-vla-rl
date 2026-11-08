"""Low frequency, job-scoped GPU memory evidence; no model imports."""
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def main():
    job = os.environ['SLURM_JOB_ID']
    directory = Path(os.environ['XVLA_RUN_DIR'])
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    started = time.time()
    peak = {}
    device_peak = {}
    errors = 0
    samples = 0
    def belongs(pid):
        try:
            return (b'SLURM_JOB_ID='+job.encode()) in Path('/proc',pid,'environ').read_bytes().split(b'\0')
        except (OSError,PermissionError):
            return False
    with (directory/'gpu_memory_samples.jsonl').open('a') as stream:
        while not stop.is_set():
            try:
                query = subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'], check=True,capture_output=True,text=True,timeout=4)
                by_gpu = {}
                for row in csv.reader(io.StringIO(query.stdout)):
                    if len(row)!=3: continue
                    uuid,pid,memory = (part.strip() for part in row)
                    if not belongs(pid): continue
                    try: memory = int(memory)
                    except ValueError: continue
                    by_gpu[uuid] = by_gpu.get(uuid,0)+memory
                for uuid,memory in by_gpu.items(): peak[uuid] = max(peak.get(uuid,0),memory)
                allocated_file=directory/'allocated_gpu_uuids.json'
                # PyTorch exposes the bare UUID; nvidia-smi prefixes it GPU-.
                allocated={value.removeprefix('GPU-').lower() for value in
                           (json.loads(allocated_file.read_text()) if allocated_file.is_file() else [])}
                totals={}
                if allocated:
                    total_query=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used',
                        '--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=4)
                    for row in csv.reader(io.StringIO(total_query.stdout)):
                        if len(row)==2 and row[0].strip().removeprefix('GPU-').lower() in allocated:
                            uuid=row[0].strip(); totals[uuid]=int(row[1].strip())
                            device_peak[uuid]=max(device_peak.get(uuid,0),totals[uuid])
                    if not totals: errors+=1
                stream.write(json.dumps(dict(time=time.time(),job_compute_memory_mib=by_gpu,
                                             allocated_device_total_memory_mib=totals))+'\n')
                stream.flush()
                samples += 1
            except (OSError,subprocess.SubprocessError):
                errors += 1
            stop.wait(5)
    physical_memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemTotal:','MemAvailable:')):
            key,value = line.split(':',1); physical_memory[key] = value.strip()
    result = dict(job_id=job,hostname=os.uname().nodename,started=started,finished=time.time(),
        elapsed_seconds=time.time()-started,allocated_gpus=int(os.environ.get('XVLA_ALLOCATED_GPUS','0')),
        gpu_hours=int(os.environ.get('XVLA_ALLOCATED_GPUS','0'))*(time.time()-started)/3600,
        peak_job_compute_memory_mib_by_gpu=peak,samples=samples,query_errors=errors,
        peak_allocated_device_total_memory_mib=device_peak,
        physical_memory=physical_memory,slurm_cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        definition='5-second sampled sum of nvidia-smi compute-process memory for exact SLURM_JOB_ID; not unsampled allocator high-water',
        device_total_definition='5-second total device use on CUDA-queried allocated UUIDs, includes EGL and any external process on those devices',
        ram_limit='Cluster RealMemory is misconfigured; physical RAM is observed, not reliably reserved/isolated')
    (directory/'resources.json').write_text(json.dumps(result,indent=2)+'\n')
    print('XVLA_RESOURCES '+json.dumps(result),flush=True)


if __name__=='__main__': main()
