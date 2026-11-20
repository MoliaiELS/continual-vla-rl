"""Run the frozen 50-task Hard plan with qualified, bounded task concurrency."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import hard_native_report
else:
    import hard_native_report


HERE = Path(__file__).resolve().parent


def lane_helpers():
    # Keep Linux locking and GPU inspection out of CPU-only accounting imports.
    if __package__:
        from . import panel
    else:
        import panel
    return panel


def qualify(args):
    if __package__:
        from . import hard_native_pilot
    else:
        import hard_native_pilot
    return hard_native_pilot.qualify(args)


def complete_gate(report):
    total = report["overall"]["cumulative"]
    return (report["status"] == "complete" and total["planned"] == total["finished"] == 7000
            and total["unresolved"] == total["retry_pending"] == 0)


class HardPanel:
    def __init__(self, args):
        self.args = args
        self.job = Path(os.environ["FYP_ROBOTWIN_JOB"]).resolve(strict=True)
        self.root = args.output_root.resolve(strict=True)
        self.source = Path(os.environ["FYP_ROBOTWIN_DATA"]) / "panels" / args.evaluation_root.name
        self.helpers = lane_helpers()
        self.log_dir = self.job / "hard-native-logs"
        self.log_dir.mkdir(exist_ok=True)
        self.journal = self.job / "hard-native-shards.jsonl"
        self.shared = ["--pinned-evaluation-root", str(args.evaluation_root), "--source-panel", str(self.source),
                       "--output-root", str(self.root), "--robotwin-root", os.environ["FYP_ROBOTWIN_ROOT"],
                       "--xvla-root", os.environ["FYP_ROBOTWIN_XVLA"], "--checkpoint", os.environ["FYP_ROBOTWIN_MODEL"]]
        self.failed = False
        self.workers = 1
        self.report = None

    def record(self, state):
        # Subprocess workers return states; only the lane owner appends here.
        self.helpers.append(self.journal, {"job_id": os.environ["SLURM_JOB_ID"], **state})

    def refresh(self):
        self.report = hard_native_report.summarize(self.root, self.source)
        target = self.root / "reports"
        target.mkdir(exist_ok=True)
        text = io.StringIO()
        hard_native_report.emit_markdown(self.report, text)
        hard_native_report.atomic_text(target / "summary.json", json.dumps(self.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        hard_native_report.atomic_text(target / "report.md", text.getvalue())
        self.tasks = {row["task_id"]: row for row in self.report["tasks"]}
        return self.report

    def completed(self, task, batch):
        row = self.tasks[task]["batches"][str(batch)]
        return (row["status"] == "complete" and row["planned"] == row["finished"] == 100
                and row["unresolved"] == row["retry_pending"] == 0)

    def frozen(self, task):
        row = self.tasks[task]["cumulative"]
        return row["frozen"] == row["planned"]

    def child(self, task, step, batch=None, workers=1, free_mib=None):
        started = time.time()
        suffix = f"batch{batch}" if step == "eval" else "screen"
        log = self.log_dir / f"{task}-{suffix}.log"
        state = dict(event="shard_finished", task_id=task, condition="hard", phase="formal", step=step,
                     batch=batch, started=started, eval_workers=workers if step == "eval" else 0,
                     free_gpu_memory_mib=free_mib, log=log.relative_to(self.job).as_posix())
        try:
            cache = self.job / "hard-native-cache" / task
            cache.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, "-u", str(HERE / "hard_native.py"), *self.shared,
                       "--eval" if step == "eval" else "--screen-only", "--task", task]
            if batch is not None:
                command.extend(["--batch", str(batch)])
            env = dict(os.environ, WARP_CACHE_PATH=str(cache))
            with log.open("ab") as stream:
                state["returncode"] = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT).returncode
        except OSError as error:
            state.update(returncode=1, error=f"{type(error).__name__}: {error}")
        state.update(finished=time.time(), seconds=round(time.time() - started, 3))
        return state

    def retain(self, state):
        self.record(state)
        self.failed |= state["returncode"] != 0
        print(f"ROBOTWIN_HARD {state['task_id']} {state['step']} {state.get('batch')}: "
              f"{'PASS' if state['returncode'] == 0 else 'INCOMPLETE'}", flush=True)

    def wait_children(self, futures):
        active = set(futures)
        while active:
            finished, active = wait(active, timeout=120, return_when=FIRST_COMPLETED)
            for future in finished:
                self.retain(future.result())
            # Journals are append-only; the reporter consumes committed lines.
            self.refresh()

    def evaluate_ready(self, pool, ready):
        if not ready:
            return
        assert len(ready) <= 2 and len({task for task, _ in ready}) == len(ready)
        free = self.helpers.free_gpu_memory_mib(self.job) if len(ready) == 2 else None
        workers = 2 if self.workers == 2 and len(ready) == 2 and free is not None and free >= self.helpers.MIN_PAIR_FREE_MIB else 1
        self.record(dict(event="group_started", work=ready, eval_workers=workers, free_gpu_memory_mib=free,
                         parallel_qualified=self.workers == 2))
        if workers == 2:
            futures = [pool.submit(self.child, task, "eval", batch, workers, free) for task, batch in ready]
            self.wait_children(futures)
        else:
            for task, batch in ready:
                self.wait_children([pool.submit(self.child, task, "eval", batch, workers, free)])
        # Both policy processes have exited before any following expert screen.
        self.refresh()
        for task, batch in ready:
            if not self.completed(task, batch):
                self.failed = True
                self.record(dict(event="batch_incomplete", task_id=task, batch=batch, returncode=1))
        ready.clear()

    def phase(self, pool, tasks, batch):
        ready = []
        for task in tasks:
            if self.completed(task, batch):
                self.record(dict(event="shard_reused", task_id=task, condition="hard", phase="formal",
                                 step="eval", batch=batch, returncode=0, eval_workers=0, reused=True))
                continue
            if batch > 1 and not self.completed(task, batch - 1):
                self.failed = True
                self.record(dict(event="dependency_incomplete", task_id=task, batch=batch, returncode=1))
                continue
            if not self.frozen(task):
                # Nothing has been submitted to the pool while ready is staged.
                state = self.child(task, "screen")
                self.retain(state)
                self.refresh()
                if state["returncode"] != 0:
                    continue
                if not self.frozen(task):
                    self.failed = True
                    self.record(dict(event="screen_incomplete", task_id=task, returncode=1))
                    continue
            ready.append((task, batch))
            if len(ready) == self.workers:
                self.evaluate_ready(pool, ready)
        self.evaluate_ready(pool, ready)

    def run(self):
        self.refresh()
        if complete_gate(self.report):
            self.record(dict(event="formal_complete", reused=True, counts=self.report["overall"]["cumulative"]))
            return 0
        try:
            qualification = qualify(self.args)
        except Exception as error:
            self.record(dict(event="qualification_failed", returncode=1, error=f"{type(error).__name__}: {error}"))
            raise
        if qualification.get("profile_fingerprint") != self.report["protocol_fingerprint"]:
            raise ValueError("Hard concurrency qualification belongs to another protocol fingerprint")
        self.workers = 2 if qualification.get("qualified") is True and qualification.get("eval_workers") == 2 else 1
        self.record(dict(event="qualification", eval_workers=self.workers, qualified=qualification.get("qualified"),
                         reason=qualification.get("reason"), returncode=0))
        profile = self.report["profile"]
        priority = profile["priority_tasks"]
        ordered = priority + [task for task in profile["task_ids"] if task not in priority]
        with ThreadPoolExecutor(max_workers=2) as pool:
            self.phase(pool, ordered, 1)
            first = self.report["overall"]["first_batch"]
            if (first["planned"] == first["finished"] == 5000
                    and first["unresolved"] == first["retry_pending"] == 0):
                self.phase(pool, priority, 2)
                self.phase(pool, priority, 3)
            else:
                self.failed = True
                self.record(dict(event="first_batch_incomplete", returncode=1, counts=first))
        self.refresh()
        complete = complete_gate(self.report)
        self.record(dict(event="formal_finished", returncode=int(self.failed or not complete),
                         counts=self.report["overall"]["cumulative"]))
        return int(self.failed or not complete)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Run through the GPU Slurm job entrypoint")
    args.evaluation_root = args.evaluation_root.resolve(strict=True)
    args.output_root = args.output_root.resolve(strict=True)
    if args.evaluation_root.name != hard_native_report.EVALUATION_COMMIT:
        parser.error("Use the original fixed 9baa evaluation worktree")
    import fcntl

    with (args.output_root / "lane-hard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return HardPanel(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
