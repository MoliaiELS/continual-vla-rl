"""Validate assigned CUDA, local Ray GPU mapping and real LIBERO EGL rendering."""
import json
import os
from pathlib import Path

import numpy as np
import ray
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, "Expected exactly the allocated GPU"
x = torch.ones(16, device="cuda")
assert x.sum().item() == 16
from rlinf.scheduler import Cluster
cluster = Cluster(num_nodes=1)
assert cluster.num_accelerators_in_cluster == 1

@ray.remote(num_gpus=1)
def check_gpu():
    import os
    import torch
    assert torch.cuda.device_count() == 1
    return {"mask": os.environ.get("CUDA_VISIBLE_DEVICES"), "name": torch.cuda.get_device_name(0)}

print("RAY_GPU", ray.get(check_gpu.remote()))
ray.shutdown()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
task = suite.get_task(0)
env = OffScreenRenderEnv(
    bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
    camera_heights=256,
    camera_widths=256,
)
try:
    env.seed(1234)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, reward, done, info = env.step(np.zeros(7))
    image = obs["agentview_image"]
    assert image.shape == (256, 256, 3) and image.std() > 1
    from PIL import Image
    Image.fromarray(image[::-1]).save(Path(os.environ["LOG_DIR"]) / "libero_task0.png")
    print(json.dumps({"task_id": 0, "language": task.language, "init_state_id": 0, "settling_steps": 10, "image_shape": image.shape, "image_std": float(image.std()), "goal": bool(env.check_success())}))
finally:
    env.close()
