"""Run independently recoverable task/condition shards inside one Slurm allocation."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PRIORITY = ("move_stapler_pad", "turn_switch", "click_alarmclock", "move_pillbottle_pad", "move_can_pot",
            "open_microwave", "pick_dual_bottles", "lift_pot", "place_bread_basket", "press_stapler")
MISSING_IN_RLINF = ("place_fan", "open_laptop", "place_object_scale", "put_object_cabinet")


def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(args):
    if args.mode == "probe":
        from bind_renderer import probe

        probe()
        model_root = Path(os.environ["FYP_ROBOTWIN_MODEL"])
        if (model_root / "verified.json").is_file():
            probe_model(model_root)
        return 0
    root = Path(os.environ["FYP_ROBOTWIN_DATA"])
    job = Path(os.environ["FYP_ROBOTWIN_JOB"])
    code = json.loads((job / "runtime.json").read_text())["code"]["repository"]["commit"]
    output = root / "panels" / code
    output.mkdir(parents=True, exist_ok=True)
    with (output / f"lane-{args.lane}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return run_lane(args, job, output, code)


def run_lane(args, job, output, code):
    conditions = ("easy", "hard") if args.lane == "both" else (args.lane,)
    smoke = args.mode in ("smoke", "coverage")
    if args.mode == "smoke":
        tasks = PRIORITY[:2]
    elif args.mode == "coverage":
        tasks = MISSING_IN_RLINF
    elif args.mode == "extension":
        tasks = PRIORITY
    else:
        tasks_json = json.loads((HERE / "tasks.json").read_text(encoding="utf-8"))
        tasks = tuple(item["task_id"] for item in tasks_json["tasks"])
        assert len(set(tasks)) == 50 and set(PRIORITY) <= set(tasks)
        tasks = PRIORITY + tuple(t for t in tasks if t not in PRIORITY)
    if not smoke:
        for condition in ("easy", "hard"):
            if not ((output / "gate" / f"smoke-{condition}.json").is_file()
                    or (output / "gate" / "smoke-both.json").is_file()):
                raise ValueError("Both conditions must pass the same commit's parity and smoke jobs")
    phases = [2, 3] if args.mode == "extension" else [1]
    states = []
    for batch in phases:
        for task in tasks:
            for condition in conditions:
                if args.task and task != args.task:
                    continue
                base = ["--robotwin-root", os.environ["FYP_ROBOTWIN_ROOT"], "--task", task, "--condition", condition,
                        "--phase", args.mode if smoke else "formal", "--output", str(output),
                        "--limit", str(1 if args.mode == "coverage" else 5 if smoke else 300 if task in PRIORITY else 100)]
                model = ["--checkpoint", os.environ["FYP_ROBOTWIN_MODEL"],
                         "--xvla-root", os.environ["FYP_ROBOTWIN_XVLA"]]
                steps = [("screen", []), ("eval", model + ["--batch", str(batch)])]
                if args.mode == "smoke":
                    steps.insert(1, ("parity", model))
                ok = True
                for step, opts in steps:
                    started = time.monotonic()
                    proc = subprocess.run([sys.executable, "-u", "-m", "tools.xvla.robotwin.runner", step] + base + opts)
                    state = dict(task_id=task, condition=condition, mode=args.mode, step=step, batch=batch,
                                 returncode=proc.returncode, seconds=round(time.monotonic() - started, 3))
                    append(job / "shards.jsonl", state)
                    states.append(state)
                    if proc.returncode:
                        ok = False
                        break
                print(f"ROBOTWIN_SHARD {task} {condition} {batch}: {'PASS' if ok else 'INCOMPLETE'}", flush=True)
    if not states or any(row["returncode"] for row in states):
        return 1
    if args.mode == "smoke" and not args.task:
        gate = output / "gate"
        gate.mkdir(exist_ok=True)
        tmp = gate / f"smoke-{args.lane}.json.tmp"
        tmp.write_text(json.dumps({"commit": code, "job_id": os.environ["SLURM_JOB_ID"], "shards": states}, indent=2) + "\n")
        tmp.replace(gate / f"smoke-{args.lane}.json")
    return 0


def probe_model(model_root):
    # Model-only forward check while simulator assets download. This does not
    # serve as a parity, observation, action execution, or success check.
    import numpy as np
    import torch
    from tools.xvla.robotwin import protocol
    from tools.xvla.robotwin.runner import checkpoint

    policy, _ = checkpoint(model_root, "cuda:0", True)
    rgb = np.zeros(protocol.RGB_SHAPE, dtype=np.uint8)
    obs = {"observation": {camera: {"rgb": rgb} for camera in protocol.CAMERAS},
           "endpose": {f"{side}_{key}": value for side in ("left", "right")
                       for key, value in (("endpose", [0., 0., 0., 1., 0., 0., 0.]), ("gripper", 0.5))}}
    policy.begin("move stapler pad", 0)
    with torch.inference_mode():
        raw = policy.step(obs)
    env = protocol.actions_to_env(raw)
    assert np.isfinite(raw).all() and env.shape == (30, 16)
    print(f"ROBOTWIN_MODEL_PROBE raw={raw.shape} ee={env.shape} peak_allocated={torch.cuda.max_memory_allocated()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["probe", "smoke", "coverage", "formal", "extension"])
    parser.add_argument("--lane", choices=["easy", "hard", "both"], default="both")
    parser.add_argument("--task")
    raise SystemExit(run(parser.parse_args()))
