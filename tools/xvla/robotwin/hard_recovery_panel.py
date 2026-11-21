"""Use fresh expert-screen processes while retaining the frozen Hard evaluator."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types


HERE = Path(__file__).resolve().parent
WORKER = "bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460"
EVALUATION = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
BOUNDARY_EXIT = 75


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=50).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_panel(worker):
    """Relative imports, pilot code and evaluator HERE all stay in bcb."""
    tools = worker / "tools/xvla/robotwin"
    package = "_robotwin_frozen_hard_recovery"
    parent = types.ModuleType(package)
    parent.__path__ = [str(tools)]
    sys.modules[package] = parent
    spec = importlib.util.spec_from_file_location(f"{package}.hard_native_panel", tools / "hard_native_panel.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BoundaryRequested(Exception):
    pass


def recovery_panel_class(frozen):
    class RecoveryPanel(frozen.HardPanel):
        def boundary(self):
            if self.args.boundary_file.exists():
                raise BoundaryRequested(str(self.args.boundary_file))

        def child(self, task, step, batch=None, workers=1, free_mib=None):
            if step != "screen":
                # The policy child, observations, seed handling and counters
                # execute the original method with its original bcb globals.
                return super().child(task, step, batch, workers, free_mib)
            self.boundary()
            started = time.time()
            log = self.log_dir / f"{task}-screen-recovery.log"
            state = dict(event="shard_finished", task_id=task, condition="hard", phase="formal", step="screen",
                         batch=batch, started=started, eval_workers=0, free_gpu_memory_mib=free_mib,
                         log=log.relative_to(self.job).as_posix(), screening="fresh_process_recovery")
            try:
                cache = self.job / "hard-native-cache" / task
                cache.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, "-u", str(HERE / "hard_screen_recovery.py"),
                           "--worker-root", str(self.args.worker_root), "--evaluation-root", str(self.args.evaluation_root),
                           "--output-root", str(self.root), "--task", task]
                with log.open("ab") as stream:
                    state["returncode"] = subprocess.run(command, env=dict(os.environ, WARP_CACHE_PATH=str(cache)),
                                                        cwd=self.args.evaluation_root,
                                                        stdout=stream, stderr=subprocess.STDOUT).returncode
            except OSError as error:
                state.update(returncode=1, error=f"{type(error).__name__}: {error}")
            state.update(finished=time.time(), seconds=round(time.time() - started, 3))
            return state

        def evaluate_ready(self, pool, ready):
            self.boundary()
            result = super().evaluate_ready(pool, ready)
            self.boundary()  # Frozen evaluate_ready has joined every policy child.
            return result

        def phase(self, pool, tasks, batch):
            self.boundary()
            result = super().phase(pool, tasks, batch)
            self.boundary()
            return result

    return RecoveryPanel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker-root", "evaluation-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--boundary-file", type=Path)
    args = parser.parse_args()
    for name in ("worker_root", "evaluation_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    require(os.environ.get("SLURM_JOB_ID", "").isdigit() and os.environ.get("CUDA_VISIBLE_DEVICES"),
            "Run in a Slurm GPU allocation")
    data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
    job = Path(os.environ["FYP_ROBOTWIN_JOB"]).resolve(strict=True)
    require(job == data / "jobs" / os.environ["SLURM_JOB_ID"], "Wrong Slurm evidence directory")
    require(args.output_root == data / "hard-native-budget" / WORKER, "Use the unchanged bcb Hard profile")
    require(Path(sys.prefix).resolve() == Path(os.environ["FYP_ROBOTWIN_ENV"]).resolve(strict=True),
            "Activate the qualified RoboTwin environment")
    args.boundary_file = (args.boundary_file or args.output_root / "recovery-pause.request").resolve()
    for root, commit in ((args.worker_root, WORKER), (args.evaluation_root, EVALUATION)):
        require(git(root, "rev-parse", "HEAD") == commit, "Wrong frozen source checkout")
        git(root, "diff", "--exit-code", "HEAD", "--", "tools/xvla/robotwin")
    files = ("hard_recovery_panel.py", "hard_recovery_job.sh", "hard_screen_recovery.py")
    git(HERE, "ls-files", "--error-unmatch", *files)
    git(HERE, "diff", "--exit-code", "HEAD", "--", *files)
    commit = git(HERE, "rev-parse", "HEAD")
    profile_path = args.output_root / "profile.json"
    profile = json.loads(profile_path.read_bytes())
    require(profile.get("implementation_commit") == WORKER and profile.get("source_evaluation_commit") == EVALUATION,
            "Hard profile implementation changed")
    for name, expected in profile["source_sha256"].items():
        require(sha256(args.worker_root / "tools/xvla/robotwin" / name) == expected, "Frozen Hard source changed: " + name)
    frozen = frozen_panel(args.worker_root)
    import fcntl

    with (args.output_root / "lane-hard.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        panel = recovery_panel_class(frozen)(args)
        evidence = dict(orchestration_commit=commit, worker_commit=WORKER, evaluation_commit=EVALUATION,
                        job_id=os.environ["SLURM_JOB_ID"], boundary_file=str(args.boundary_file),
                        protocol_fingerprint=profile["protocol_fingerprint"], profile_sha256=sha256(profile_path),
                        sources={name: sha256(HERE / name) for name in files},
                        frozen_panel_path=str(frozen.__file__))
        frozen.hard_native_report.atomic_text(job / "hard-recovery-orchestration.json", json.dumps(evidence, indent=2) + "\n")
        panel.record(dict(event="recovery_orchestration", **evidence))
        try:
            panel.boundary()
            return panel.run()
        except BoundaryRequested as error:
            panel.refresh()
            panel.record(dict(event="boundary_pause", returncode=BOUNDARY_EXIT, request_file=str(error),
                              counts=panel.report["overall"]["cumulative"]))
            return BOUNDARY_EXIT


if __name__ == "__main__":
    raise SystemExit(main())
