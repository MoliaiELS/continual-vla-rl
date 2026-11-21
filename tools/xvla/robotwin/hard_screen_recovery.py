"""Complete frozen Hard expert screening in bounded, fresh GPU processes."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


WORKER = "bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460"
EVALUATION = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
SOURCE_PROTOCOL = "robotwin-native-b0-v2"
SOURCE_FINGERPRINT = "78fa1f315e850c366ae0e36250f3cb633c6667747ebdfbb0608cee0656320f46"
CHUNK_CANDIDATES = 20
MAX_FRESH_RESTARTS = 3
CANDIDATE_START = 2000
CANDIDATE_STOP = 102000
RENDERER_ERROR = 'RuntimeError: Failed to find a supported physical device "cuda:0"'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=50).strip()


def verify_checkout(root, expected):
    require(Path(git(root, "rev-parse", "--show-toplevel")).resolve() == root,
            f"Expected a checkout root: {root}")
    require(git(root, "rev-parse", "HEAD") == expected, f"Wrong fixed checkout: {root}")
    require(not git(root, "status", "--porcelain", "--untracked-files=no"),
            f"Tracked source differs from {expected}: {root}")


def renderer_failure(row):
    trace = row.get("traceback", "")
    return (row.get("status") == "technical_error" and row.get("reason") == "RuntimeError"
            and trace.rstrip().splitlines()[-1:] == [RENDERER_ERROR]
            and 'in setup_scene\n    self.renderer = sapien.SapienRenderer()' in trace
            and '/envs/_base_task.py",' in trace
            and '/sapien/wrapper/renderer.py",' in trace)


@dataclass
class Snapshot:
    complete_prefix: bytes
    rows: int
    accepted: int
    excluded: int
    technical_attempts: int
    next_seed: int
    pending: dict | None

    @property
    def finalized(self):
        return self.accepted + self.excluded

    def evidence(self):
        return {"journal_complete_bytes": len(self.complete_prefix),
                "journal_complete_sha256": hashlib.sha256(self.complete_prefix).hexdigest(),
                "rows": self.rows, "accepted": self.accepted, "excluded": self.excluded,
                "technical_attempts": self.technical_attempts, "next_seed": self.next_seed,
                "pending_seed": self.pending["requested_seed"] if self.pending else None,
                "pending_attempt": self.pending["attempt"] if self.pending else None}


def snapshot(path, task):
    raw = path.read_bytes() if path.exists() else b""
    complete = raw[:raw.rfind(b"\n") + 1]
    accepted = excluded = technical = count = 0
    seed, attempt, pending = CANDIDATE_START, 0, None
    for line in complete.splitlines():
        row = json.loads(line)
        require(all(row.get(key) == value for key, value in {
            "task": task, "condition": "hard", "phase": "formal", "protocol": SOURCE_PROTOCOL,
            "protocol_fingerprint": SOURCE_FINGERPRINT,
        }.items()), f"Unexpected source journal identity in {path}")
        require(CANDIDATE_START <= row.get("requested_seed", -1) < CANDIDATE_STOP
                and row["requested_seed"] == row.get("actual_seed") == seed
                and row.get("attempt") == attempt + 1,
                f"Candidate or attempt order differs from the fixed runner in {path}")
        status = row.get("status")
        require(status in {"accepted", "excluded", "technical_error"}, f"Unknown prescreen status in {path}")
        count += 1
        attempt = row["attempt"]
        if status == "technical_error":
            technical += 1
            pending = row
        else:
            accepted += int(status == "accepted")
            excluded += int(status == "excluded")
            seed += 1
            attempt, pending = 0, None
    return Snapshot(complete, count, accepted, excluded, technical, seed, pending)


class Recovery:
    def __init__(self, args, environment, identity):
        self.args, self.environment, self.identity = args, environment, identity
        self.worker_tools = args.worker_root / "tools/xvla/robotwin"
        self.source = Path(environment["FYP_ROBOTWIN_DATA"]) / "panels" / EVALUATION
        self.shard = self.source / "formal" / args.task / "hard"
        self.journal = self.shard / "prescreen.jsonl"
        self.manifest = self.shard / "seed_manifest.json"
        self.job = Path(environment["FYP_ROBOTWIN_JOB"])
        self.events = self.job / "hard-screen-recovery.jsonl"
        self.logs = self.job / "hard-screen-recovery-logs"
        self.limit = identity["limit"]

    def record(self, event, **fields):
        row = {"event": event, "time": time.time(), "job_id": self.environment["SLURM_JOB_ID"],
               "task": self.args.task, **self.identity, **fields}
        with self.events.open("ab") as stream:
            stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def manifest_complete(self, state):
        if not self.manifest.exists():
            return False
        manifest = read_json(self.manifest)
        expected = {"task": self.args.task, "condition": "hard", "phase": "formal",
                    "protocol": SOURCE_PROTOCOL, "protocol_fingerprint": SOURCE_FINGERPRINT,
                    "limit": self.limit, "candidate_start": CANDIDATE_START}
        require(all(manifest.get(key) == value for key, value in expected.items()),
                "Existing source manifest differs from the frozen Hard candidate plan")
        require(state.pending is None and state.accepted == self.limit
                and len(manifest.get("episodes", [])) == self.limit,
                "Source manifest and accepted candidate journal differ")
        return True

    def run_process(self, command, stage, number):
        self.logs.mkdir(exist_ok=True)
        log = self.logs / f"{self.args.task}-{stage}-{number}-{time.time_ns()}.log"
        started = time.perf_counter()
        with log.open("xb") as stream:
            process = subprocess.run(command, cwd=self.args.evaluation_root, env=self.environment,
                                     stdout=stream, stderr=subprocess.STDOUT)
        return process.returncode, time.perf_counter() - started, str(log.relative_to(self.job))

    def screen_command(self):
        return [sys.executable, "-u", str(self.worker_tools / "prescreen_recovery.py"),
                "--pinned-evaluation-root", str(self.args.evaluation_root), "screen",
                "--robotwin-root", self.environment["FYP_ROBOTWIN_ROOT"], "--task", self.args.task,
                "--condition", "hard", "--phase", "formal", "--output", str(self.source),
                "--limit", str(self.limit), "--screen-candidates", str(CHUNK_CANDIDATES)]

    def freeze_command(self):
        return [sys.executable, "-u", str(self.worker_tools / "hard_native.py"),
                "--pinned-evaluation-root", str(self.args.evaluation_root),
                "--source-panel", str(self.source), "--output-root", str(self.args.output_root),
                "--robotwin-root", self.environment["FYP_ROBOTWIN_ROOT"],
                "--xvla-root", self.environment["FYP_ROBOTWIN_XVLA"],
                "--checkpoint", self.environment["FYP_ROBOTWIN_MODEL"], "--screen-only", "--task", self.args.task]

    def run(self):
        state = snapshot(self.journal, self.args.task)
        restarts = {}
        number = 0
        # A process either finalizes candidates or consumes one of three fresh
        # retries for its pending candidate, bounded by the original seed range.
        process_limit = (CANDIDATE_STOP - state.next_seed + 1) * (MAX_FRESH_RESTARTS + 1)
        self.record("recovery_started", before=state.evidence(), chunk_candidates=CHUNK_CANDIDATES,
                    max_fresh_restarts_per_seed=MAX_FRESH_RESTARTS, process_limit=process_limit)
        try:
            while not self.manifest_complete(state):
                require(state.accepted < self.limit or (state.accepted == self.limit and state.pending is None),
                        "Accepted candidate journal exceeds the fixed plan or has an unexpected pending seed")
                require(number < process_limit and (state.next_seed < CANDIDATE_STOP or state.accepted == self.limit),
                        "Hard recovery exhausted the bounded process or candidate limit")
                restart = 0
                if state.pending is not None:
                    require(renderer_failure(state.pending), "Pending seed has an unknown technical error")
                    restart = restarts.get(state.next_seed, 0) + 1
                    require(restart <= MAX_FRESH_RESTARTS,
                            f"Renderer recovery exhausted at the unchanged seed {state.next_seed}")
                    restarts[state.next_seed] = restart
                before = state
                number += 1
                self.record("chunk_started", chunk=number, fresh_restart=restart, before=before.evidence())
                code, seconds, log = self.run_process(self.screen_command(), "screen", number)
                state = snapshot(self.journal, self.args.task)
                self.record("chunk_finished", chunk=number, fresh_restart=restart, before=before.evidence(),
                            after=state.evidence(), returncode=code, seconds=seconds, log=log)
                require(state.complete_prefix.startswith(before.complete_prefix), "Existing candidate history changed")
                require(state.rows > before.rows or self.manifest_complete(state),
                        "Screen process returned without journal progress or a complete manifest")
                require(0 <= state.finalized - before.finalized <= CHUNK_CANDIDATES,
                        "Screen process exceeded the fixed candidate chunk")
                if code:
                    require(state.pending is not None and renderer_failure(state.pending),
                            "Screen process failed without the known renderer error at its pending seed")
                    continue
                require(state.pending is None and (state.finalized > before.finalized
                                                  or self.manifest_complete(state)),
                        "Successful screen process did not advance the candidate plan")
            checksum = sha256(self.manifest)
            code, seconds, log = self.run_process(self.freeze_command(), "freeze", number)
            self.record("native_manifest_finished", returncode=code, seconds=seconds, log=log,
                        source_manifest_sha256=checksum, after=state.evidence())
            require(code == 0, "Frozen Hard manifest preparation failed")
            require(sha256(self.manifest) == checksum, "Manifest preparation changed the frozen source cohort")
            require(snapshot(self.journal, self.args.task).complete_prefix == state.complete_prefix,
                    "Manifest preparation unexpectedly performed expert screening")
        except Exception as error:
            self.record("recovery_failed", error=f"{type(error).__name__}: {error}")
            raise
        self.record("recovery_completed", chunks=number, after=state.evidence(),
                    source_manifest_sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker-root", "evaluation-root", "output-root"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--task", required=True)
    args = parser.parse_args()
    for name in ("worker_root", "evaluation_root", "output_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    environment = dict(os.environ)
    require(environment.get("SLURM_JOB_ID", "").isdigit() and environment.get("CUDA_VISIBLE_DEVICES"),
            "Hard screening requires a GPU Slurm allocation")
    for name in ("DATA", "ROOT", "XVLA", "MODEL", "JOB", "ENV"):
        require(environment.get(f"FYP_ROBOTWIN_{name}"), f"Missing FYP_ROBOTWIN_{name}")
    data = Path(environment["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
    job = Path(environment["FYP_ROBOTWIN_JOB"]).resolve(strict=True)
    require(job == data / "jobs" / environment["SLURM_JOB_ID"], "Expected this Slurm job's evidence directory")
    require(Path(sys.prefix).resolve() == Path(environment["FYP_ROBOTWIN_ENV"]).resolve(strict=True),
            "Activate the existing RoboTwin evaluation environment")
    require(args.output_root == data / "hard-native-budget" / WORKER, "Use the existing frozen Hard output profile")
    verify_checkout(args.worker_root, WORKER)
    verify_checkout(args.evaluation_root, EVALUATION)
    source = Path(__file__).resolve()
    orchestrator = Path(git(source.parent, "rev-parse", "--show-toplevel")).resolve()
    relative = source.relative_to(orchestrator)
    git(orchestrator, "ls-files", "--error-unmatch", str(relative))
    require(not git(orchestrator, "diff", "HEAD", "--", str(relative)), "Commit the recovery orchestrator before running")
    profile = read_json(args.output_root / "profile.json")
    require(profile.get("implementation_commit") == WORKER
            and profile.get("source_evaluation_commit") == EVALUATION
            and profile.get("source_protocol_fingerprint") == SOURCE_FINGERPRINT
            and profile.get("source_panel") == str(data / "panels" / EVALUATION),
            "Hard profile differs from the fixed worker or evaluation")
    profile_basis = {key: value for key, value in profile.items() if key not in {
        "protocol_fingerprint", "planned_episodes", "planned_first_batch", "planned_extension",
        "source_panel", "implementation_commit"}}
    require(profile.get("protocol_fingerprint") == hashlib.sha256(json.dumps(profile_basis, sort_keys=True).encode()).hexdigest(),
            "Hard profile fingerprint failed validation")
    worker_tools = args.worker_root / "tools/xvla/robotwin"
    for name, expected in profile["source_sha256"].items():
        require(sha256(worker_tools / name) == expected, f"Frozen Hard source differs: {name}")
    catalog = read_json(args.evaluation_root / "tools/xvla/robotwin/tasks.json")
    tasks = {row["task_id"] for row in catalog["tasks"]}
    require(args.task in tasks, "Task is outside the original 50-task evaluation")
    limit = 300 if args.task in catalog["priority_tasks"] else 100
    require(profile["per_task_episodes"].get(args.task) == limit, "Task episode count differs from the fixed plan")
    environment["PYTHONPATH"] = str(args.evaluation_root)
    identity = {"worker_commit": WORKER, "evaluation_commit": EVALUATION,
                "orchestrator_commit": git(orchestrator, "rev-parse", "HEAD"), "orchestrator_sha256": sha256(source),
                "worker_source_sha256": {name: sha256(worker_tools / name) for name in
                                         ("prescreen_recovery.py", "prescreen_failures.py", "hard_native.py")},
                "evaluation_runner_sha256": sha256(args.evaluation_root / "tools/xvla/robotwin/runner.py"),
                "hard_profile_sha256": sha256(args.output_root / "profile.json"), "limit": limit}
    import fcntl
    with (job / f"hard-screen-recovery-{args.task}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Recovery(args, environment, identity).run()


if __name__ == "__main__":
    main()
