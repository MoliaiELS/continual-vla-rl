"""Run the five-task native-budget reevaluation in independent task processes."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import os
from pathlib import Path
import subprocess
import sys
import time

from panel import MIN_PAIR_FREE_MIB, append, free_gpu_memory_mib


HERE = Path(__file__).resolve().parent
TASKS = ("open_microwave", "stack_bowls_three", "blocks_ranking_rgb", "blocks_ranking_size", "click_alarmclock")
SINGLE_WORKER_FREE_MIB = MIN_PAIR_FREE_MIB // 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Run through the GPU Slurm job entrypoint")
    job = Path(os.environ["FYP_ROBOTWIN_JOB"])
    profile = args.output_root.resolve(strict=True)
    source_panel = Path(os.environ["FYP_ROBOTWIN_DATA"]) / "panels" / args.evaluation_root.name
    log_dir = job / "native-budget-logs"
    log_dir.mkdir(exist_ok=True)
    shared = ["--pinned-evaluation-root", str(args.evaluation_root), "--input-panel", str(source_panel),
              "--output-root", str(profile), "--robotwin-root", os.environ["FYP_ROBOTWIN_ROOT"],
              "--xvla-root", os.environ["FYP_ROBOTWIN_XVLA"], "--checkpoint", os.environ["FYP_ROBOTWIN_MODEL"]]

    def report():
        subprocess.run([sys.executable, str(HERE / "native_budget_report.py"), "--input", str(profile),
                        "--source-panel", str(source_panel), "--output", str(profile / "reports")], check=True)

    def evaluate(task, batch, concurrency):
        started = time.time()
        log = log_dir / f"{task}-batch{batch}.log"
        cache = job / "native-budget-cache" / task
        cache.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, WARP_CACHE_PATH=str(cache))
        command = [sys.executable, "-u", str(HERE / "native_budget.py"), *shared,
                   "--task", task, "--batch", str(batch)]
        with log.open("ab") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        return dict(task_id=task, batch=batch, returncode=result.returncode, started=started,
                    finished=time.time(), seconds=round(time.time() - started, 3),
                    active_processes_at_launch=concurrency, log=log.relative_to(job).as_posix())

    completed = set()

    def phase(pool, work):
        pending = list(work)
        active = {}
        failed = False
        free = free_gpu_memory_mib(job)
        worker_limit = 2 if free is not None and free >= MIN_PAIR_FREE_MIB else 1
        append(job / "native-budget-shards.jsonl", {"event": "phase_started", "worker_limit": worker_limit,
                                                  "free_gpu_memory_mib": free, "work": pending})
        last_report = time.monotonic()
        while pending or active:
            active_tasks = {task for task, _ in active.values()}
            ready = [item for item in pending if item[0] not in active_tasks
                     and (item[1] == 1 or (item[0], item[1] - 1) in completed)]
            for task, batch in ready:
                if len(active) >= worker_limit:
                    break
                if active:
                    free = free_gpu_memory_mib(job)
                    if free is None or free < SINGLE_WORKER_FREE_MIB:
                        break
                future = pool.submit(evaluate, task, batch, len(active) + 1)
                active[future] = (task, batch)
                pending.remove((task, batch))
                active_tasks.add(task)
            if not active:
                return 1
            finished, _ = wait(active, timeout=40, return_when=FIRST_COMPLETED)
            for future in finished:
                task, batch = active.pop(future)
                state = future.result()
                append(job / "native-budget-shards.jsonl", state)
                if state["returncode"] == 0:
                    completed.add((task, batch))
                else:
                    failed = True
            if finished or time.monotonic() - last_report >= 120:
                report()
                last_report = time.monotonic()
        return int(failed)

    report()
    with ThreadPoolExecutor(max_workers=2) as pool:
        if phase(pool, [(task, 1) for task in TASKS]):
            return 1
        if phase(pool, [(task, batch) for batch in (2, 3)
                        for task in ("open_microwave", "click_alarmclock")]):
            return 1
    report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
