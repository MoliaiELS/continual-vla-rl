"""Read-only import and provenance report; no Ray/model/GPU initialization."""
import importlib
import importlib.metadata
import contextlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import psutil

report = {
    "python": platform.python_version(),
    "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "repo": os.environ["FYP_REPO"],
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "ram_available_gib": psutil.virtual_memory().available / 2**30,
    "shm_gib": psutil.disk_usage("/dev/shm").total / 2**30,
    "packages": {},
}
with contextlib.redirect_stdout(sys.stderr):
    for name in ("torch", "torchvision", "transformers", "peft", "ray", "prismatic", "libero", "rlinf", "robosuite", "mujoco"):
        module = importlib.import_module(name)
        report["packages"][name] = {"version": getattr(module, "__version__", None), "path": str(module.__file__)}
for name, fragment in {"transformers": "transformers-openvla-oft", "prismatic": "openvla-oft", "libero": "LIBERO", "rlinf": "rlinf"}.items():
    source = Path(report["packages"][name]["path"]).resolve()
    assert source.is_relative_to(Path(os.environ["FYP_REPO"]) / fragment), (name, source)
with contextlib.redirect_stdout(sys.stderr):
    from libero.libero import benchmark
    suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    report["tasks"] = [{"id": i, "name": suite.get_task(i).name, "language": suite.get_task(i).language} for i in range(suite.get_num_tasks())]
report["torch_cuda_runtime"] = importlib.import_module("torch").version.cuda
print(json.dumps(report, indent=2))
