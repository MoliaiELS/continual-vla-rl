"""Run independently recoverable task/condition shards inside one Slurm allocation."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Two measured serial device peaks plus 2 GiB of headroom.
MIN_PAIR_FREE_MIB = 2 * 10_275 + 2 * 1024


def append(path, record):
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run(args):
    if args.pinned_evaluation_root is not None and args.mode not in {"formal", "extension"}:
        raise ValueError("Prescreen recovery requires formal or extension")
    if args.eval_workers == 2 and (args.pinned_evaluation_root is None or args.lane != "easy"
                                  or args.mode not in {"formal", "extension"}):
        raise ValueError("Two evaluation workers require Easy formal or extension with pinned prescreen recovery")
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
    evaluation_tools = HERE
    runner = [sys.executable, "-u", "-m", "tools.xvla.robotwin.runner"]
    screen_runner = runner
    if args.pinned_evaluation_root is not None:
        evaluation_tools = args.pinned_evaluation_root / "tools/xvla/robotwin"
        screen_runner = [sys.executable, "-u", str(HERE / "prescreen_recovery.py"),
                         "--pinned-evaluation-root", str(args.pinned_evaluation_root)]
    conditions = ("easy", "hard") if args.lane == "both" else (args.lane,)
    smoke = args.mode in ("smoke", "coverage")
    if args.mode == "smoke":
        tasks = PRIORITY[:2]
    elif args.mode == "coverage":
        tasks = MISSING_IN_RLINF
    elif args.mode == "extension":
        tasks = PRIORITY
    else:
        tasks_json = json.loads((evaluation_tools / "tasks.json").read_text(encoding="utf-8"))
        tasks = tuple(item["task_id"] for item in tasks_json["tasks"])
        assert len(set(tasks)) == 50 and set(PRIORITY) <= set(tasks)
        tasks = PRIORITY + tuple(t for t in tasks if t not in PRIORITY)
    if not smoke:
        for condition in ("easy", "hard"):
            if not ((output / "gate" / f"smoke-{condition}.json").is_file()
                    or (output / "gate" / "smoke-both.json").is_file()):
                raise ValueError("Both conditions must pass the same commit's parity and smoke jobs")
    phases = [2, 3] if args.mode == "extension" else [1]
    if args.eval_workers == 2:
        return run_parallel_easy(args, job, output, evaluation_tools, runner, screen_runner, tasks, phases)
    states = []
    for batch in phases:
        for task in tasks:
            for condition in conditions:
                if args.task and task != args.task:
                    continue
                base = shard_arguments(args.mode, output, task, condition)
                model = model_arguments()
                steps = [("screen", []), ("eval", model + ["--batch", str(batch)])]
                if args.mode == "smoke":
                    steps.insert(1, ("parity", model))
                ok = True
                for step, opts in steps:
                    started = time.monotonic()
                    command = screen_runner if step == "screen" else runner
                    proc = subprocess.run(command + [step] + base + opts)
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


def shard_arguments(mode, output, task, condition):
    limit = 1 if mode == "coverage" else 5 if mode == "smoke" else 300 if task in PRIORITY else 100
    phase = mode if mode in ("coverage", "smoke") else "formal"
    return ["--robotwin-root", os.environ["FYP_ROBOTWIN_ROOT"], "--task", task, "--condition", condition,
            "--phase", phase, "--output", str(output), "--limit", str(limit)]


def model_arguments():
    return ["--checkpoint", os.environ["FYP_ROBOTWIN_MODEL"], "--xvla-root", os.environ["FYP_ROBOTWIN_XVLA"]]


def verified_completed_batches(output, evaluation_tools):
    from tools.xvla.robotwin import summarize

    if Path(summarize.__file__).resolve() != (evaluation_tools / "summarize.py").resolve():
        raise ImportError("Easy evaluation resume checks must use the pinned summarizer")
    try:
        evidence = summarize.summarize(output, tasks_path=evaluation_tools / "tasks.json")
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"RoboTwin resume verification deferred to pinned runner: {error}", flush=True)
        return set()
    return {(task["task_id"], int(batch)) for task in evidence["tasks"]
            for batch, stats in task["conditions"]["easy"]["batches"].items()
            if stats["finished"] == stats["planned"] == 100 and stats["unresolved"] == 0}


def free_gpu_memory_mib(job):
    try:
        uuids = json.loads((job / "allocated_gpu_uuids.json").read_text())
        if len(uuids) != 1 or not isinstance(uuids[0], str) or not uuids[0]:
            return None
        uuid = f"GPU-{uuids[0].removeprefix('GPU-')}"
        value = subprocess.check_output(["nvidia-smi", f"--id={uuid}", "--query-gpu=memory.free",
                                         "--format=csv,noheader,nounits"], text=True, timeout=5)
        return int(value.strip())
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
        return None


def run_parallel_easy(args, job, output, evaluation_tools, runner, screen_runner, tasks, phases):
    complete = verified_completed_batches(output, evaluation_tools)
    model = model_arguments()
    states = []
    log_dir = job / "eval-logs"
    log_dir.mkdir(exist_ok=True)

    def record(state):
        # Only the lane owner writes the shared journal, even when both subprocesses finish together.
        append(job / "shards.jsonl", state)
        states.append(state)

    def shard_result(state):
        result = "PASS" if state["returncode"] == 0 else "INCOMPLETE"
        print(f"ROBOTWIN_SHARD {state['task_id']} easy {state['batch']}: {result}", flush=True)

    def evaluate(task, batch, base, workers, free_mib):
        started = time.monotonic()
        log = log_dir / f"{task}-batch{batch}.log"
        cache = job / "eval-cache" / task
        state = dict(task_id=task, condition="easy", mode=args.mode, step="eval", batch=batch,
                     eval_workers=workers, free_gpu_memory_mib=free_mib, log=log.relative_to(job).as_posix())
        try:
            cache.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, WARP_CACHE_PATH=str(cache))
            with log.open("ab") as stream:
                command = runner + ["eval"] + base + model + ["--batch", str(batch)]
                state["returncode"] = subprocess.run(command, env=env, stdout=stream,
                                                     stderr=subprocess.STDOUT).returncode
        except OSError as error:
            state.update(returncode=1, error=str(error))
        state["seconds"] = round(time.monotonic() - started, 3)
        return state

    def evaluate_ready(pool, ready):
        # Keep prescreen off the GPU while either evaluation subprocess is active.
        free_mib = free_gpu_memory_mib(job) if len(ready) == 2 else None
        if free_mib is not None and free_mib >= MIN_PAIR_FREE_MIB:
            futures = [pool.submit(evaluate, *work, 2, free_mib) for work in ready]
            results = (future.result() for future in as_completed(futures))
        else:
            if len(ready) == 2:
                print(f"RoboTwin pair falls back to serial with {free_mib} MiB free on the assigned GPU", flush=True)
            results = (evaluate(*work, 1, free_mib) for work in ready)
        for state in results:
            record(state)
            shard_result(state)
        ready.clear()

    with ThreadPoolExecutor(max_workers=2) as pool:
        for batch in phases:
            ready = []
            for task in tasks:
                if args.task and task != args.task:
                    continue
                identity = dict(task_id=task, condition="easy", mode=args.mode, batch=batch)
                if (task, batch) in complete:
                    state = dict(**identity, step="eval", returncode=0, seconds=0, eval_workers=0, reused=True)
                    record(state)
                    shard_result(state)
                    continue
                base = shard_arguments(args.mode, output, task, "easy")
                started = time.monotonic()
                proc = subprocess.run(screen_runner + ["screen"] + base)
                state = dict(**identity, step="screen", returncode=proc.returncode,
                             seconds=round(time.monotonic() - started, 3))
                record(state)
                if proc.returncode:
                    shard_result(state)
                    continue
                ready.append((task, batch, base))
                if len(ready) == 2:
                    evaluate_ready(pool, ready)
            evaluate_ready(pool, ready)
    return int(not states or any(row["returncode"] for row in states))


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
    parser.add_argument("--pinned-evaluation-root", type=Path)
    parser.add_argument("--eval-workers", type=int, choices=(1, 2), default=1)
    raise SystemExit(run(parser.parse_args()))
