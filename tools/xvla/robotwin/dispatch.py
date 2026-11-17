"""Advance the fixed experiment through Slurm gates; persist every job and decision."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


HERE = Path(__file__).resolve().parent
INTERVAL = 40
DONE = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED",
        "BOOT_FAIL", "DEADLINE", "REVOKED"}
RESUMABLE = {"TIMEOUT", "NODE_FAIL", "PREEMPTED"}
RECOVERY_COMMIT = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"


def output(*cmd):
    return subprocess.check_output(cmd, text=True, timeout=50).strip()


class Dispatch:
    def __init__(self, config):
        self.cfg = config
        experiment_config = vars(config).copy()
        evaluation_root = experiment_config.pop("evaluation_root", None) or HERE.parents[2]
        evaluation_root = evaluation_root.resolve(strict=True)
        self.recover_prescreen = experiment_config.pop("recover_prescreen", False)
        self.evaluation_root = evaluation_root
        self.evaluation_tools = evaluation_root / "tools/xvla/robotwin"
        self.data = Path(os.environ["FYP_ROBOTWIN_DATA"])
        self.code = output("git", "-C", str(evaluation_root), "rev-parse", "HEAD")
        if self.recover_prescreen and self.code != RECOVERY_COMMIT:
            raise ValueError("Prescreen recovery is scoped to the original frozen evaluation")
        self.rev = dict(line.split("=", 1) for line in (self.evaluation_tools / "runtime.env").read_text().splitlines())
        self.panel = self.data / "panels" / self.code
        self.report = self.data / "reports" / self.code
        self.path = self.data / "dispatch" / self.code / "state.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "commit": self.code, "jobs": {}, "configuration": experiment_config,
        }
        if self.state["configuration"] != experiment_config:
            raise ValueError("Dispatch arguments differ from existing experiment")
        if self.recover_prescreen:
            self.state["prescreen_recovery_commit"] = output("git", "-C", str(HERE), "rev-parse", "HEAD")

    def save(self, stage, **extra):
        self.state.update(stage=stage, at_unix=time.time(), **extra)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state, indent=2) + "\n")
        tmp.replace(self.path)
        print(json.dumps({"stage": stage, **extra}), flush=True)

    def accounting(self, job):
        rows = output("sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode")
        for row in rows.splitlines():
            parts = row.split("|")
            if len(parts) >= 3 and parts[0] == job:
                state = parts[1].split()[0].rstrip("+")
                return state, parts[2]
        return "UNKNOWN", None

    def journal_submission(self, stage, lane, known):
        # submit.sh writes under a cluster-side lock; recover a job if this
        # process stopped after sbatch but before its own atomic state write.
        journal = self.data / "evidence/jobs.tsv"
        for line in reversed(journal.read_text().splitlines() if journal.exists() else []):
            job, mode, job_lane, _, code, task = line.split("\t")
            job = job.partition(";")[0]
            if (mode, job_lane, code, task) == (stage, lane, self.code, "") and job not in known:
                return job
        return None

    def start(self, stage, lane, hours, retry=False):
        key = f"{stage}:{lane}"
        jobs = self.state["jobs"].setdefault(key, [])
        if jobs and not retry:
            return jobs[-1]
        job = self.journal_submission(stage, lane, jobs)
        if job is None:
            submit = self.evaluation_tools / "submit.sh"
            extra = ()
            if self.recover_prescreen and stage in {"formal", "extension"}:
                submit = HERE / "submit.sh"
                extra = ("", str(self.evaluation_root))
            msg = output("bash", str(submit), stage, self.cfg.partition, self.cfg.gpu, lane, str(hours), *extra)
            job = re.search(r"ROBOTWIN_JOB_ID=(\d+)", msg)[1]
        jobs.append(job)
        self.save(stage, submitting={"lane": lane, "job": job})
        return job

    def old_prescreen_failed(self, job):
        if not self.recover_prescreen:
            return False
        launches = self.data / "evidence/prescreen-recovery-jobs.tsv"
        if launches.exists() and any(row.split("\t", 1)[0] == job for row in launches.read_text().splitlines()):
            return False
        shards = self.data / "jobs" / job / "shards.jsonl"
        if not shards.is_file():
            return False
        for line in shards.read_bytes().splitlines(keepends=True):
            if line.endswith(b"\n"):
                row = json.loads(line)
                if row["step"] == "screen" and row["returncode"]:
                    return True
        return False

    def wait_for(self, stage, lanes, hours):
        jobs = {lane: self.start(stage, lane, hours) for lane in lanes}
        retries = {lane: 0 for lane in lanes}
        while True:
            states = {lane: self.accounting(job) for lane, job in jobs.items()}
            for lane, (state, _) in states.items():
                if stage not in {"formal", "extension"} or retries[lane] >= 8:
                    continue
                if state in RESUMABLE or (state == "FAILED" and self.old_prescreen_failed(jobs[lane])):
                    retries[lane] += 1
                    jobs[lane] = self.start(stage, lane, hours, retry=True)
                    states[lane] = ("RESUBMITTED", None)
            if all(state in DONE for state, _ in states.values()):
                self.save(stage, last_job_states=states, submitting=None)
                return states
            time.sleep(INTERVAL)

    def wait_download(self, path, lock):
        while not path.is_file():
            if subprocess.run(["flock", "-n", str(lock), "true"]).returncode == 0 and not path.is_file():
                raise RuntimeError(f"Download is not running and verification is missing: {path}")
            time.sleep(INTERVAL)

    def aggregate(self, phase):
        subprocess.run([os.sys.executable, str(self.evaluation_tools / "summarize.py"), "--input", str(self.panel),
                        "--phase", phase, "--output", str(self.report / phase)], check=True)
        return json.loads((self.report / phase / "summary.json").read_text(encoding="utf-8"))

    def estimate(self, lanes, evidence):
        info = {}
        for lane in lanes:
            job = self.state["jobs"][f"smoke:{lane}"][-1]
            lane_summary = evidence["overall"][lane]
            completed = lane_summary["cumulative"]["finished"]
            screen_seconds = lane_summary["prescreen"]["attempt_wall_seconds"]
            eval_seconds = sum(
                task["conditions"][lane]["cumulative"]["attempt_wall_seconds"]
                for task in evidence["tasks"]
            )
            screen_per_episode = screen_seconds / completed
            eval_per_episode = eval_seconds / completed
            first_batch_gpu_hours = (7000 * screen_per_episode + 5000 * eval_per_episode) / 3600
            full_gpu_hours = 7000 * (screen_per_episode + eval_per_episode) / 3600
            info[lane] = dict(job=job, measured_episodes=completed,
                              stage_seconds={"screen": screen_seconds, "eval": eval_seconds},
                              seconds_per_episode=screen_per_episode + eval_per_episode,
                              first_batch_gpu_hours=first_batch_gpu_hours, full_gpu_hours=full_gpu_hours)
        info["limits"] = (
            "Two smoke tasks per lane; journal times include candidate screening and policy attempts, "
            "including attempts retained from earlier targeted jobs. The first batch freezes all "
            "7,000 initial states per condition before or during its evaluation of the first 5,000. "
            "Estimates exclude process startup, parity, coverage, queueing and transfer. "
            "Acceptance rates and per-episode times can differ across tasks."
        )
        (self.report / "throughput.json").write_text(json.dumps(info, indent=2) + "\n")

    def preflight(self, lanes):
        if self.accounting(self.cfg.build_job) != ("COMPLETED", "0:0"):
            self.save("waiting_for_build")
            while self.accounting(self.cfg.build_job)[0] not in DONE:
                time.sleep(INTERVAL)
            if self.accounting(self.cfg.build_job) != ("COMPLETED", "0:0"):
                raise RuntimeError(f"CuRobo build {self.cfg.build_job} failed")
        if self.wait_for("probe", ("both",), 1)["both"] != ("COMPLETED", "0:0"):
            raise RuntimeError("Allocated GPU and native Vulkan probe did not pass")
        self.save("waiting_for_verified_downloads")
        model = self.data / "models" / f"X-VLA-RoboTwin2--{self.rev['MODEL_REV'][:12]}"
        assets = self.data / "sim-assets" / self.rev["ASSETS_REV"]
        self.wait_download(model / "verified.json", self.data / "download-robotwin.lock")
        self.wait_download(assets / ".extracted", self.data / "prepare-assets.lock")
        subprocess.run(["flock", str(self.data / "prepare-link.lock"), "bash", str(self.evaluation_tools / "prepare.sh"), "link"], check=True)
        smoke = self.wait_for("smoke", lanes, 4)
        evidence = self.aggregate("smoke")
        for lane in lanes:
            totals = evidence["overall"][lane]["cumulative"]
            if smoke[lane] != ("COMPLETED", "0:0") or totals["finished"] != 10 or totals["unresolved"]:
                raise RuntimeError(f"Parity or smoke did not qualify {lane}; inspect the recorded jobs")
        self.estimate(lanes, evidence)
        self.wait_for("coverage", lanes, 4)  # A task-specific gap is not a global gate.

    def run(self):
        lanes = ("easy", "hard")
        if not any(self.state["jobs"].get(f"formal:{lane}") for lane in lanes):
            self.preflight(lanes)
        elif self.recover_prescreen:
            self.save("formal")
        self.wait_for("formal", lanes, self.cfg.hours)
        self.aggregate("formal")
        self.wait_for("extension", lanes, self.cfg.hours)
        evidence = self.aggregate("formal")
        totals = {lane: evidence["overall"][lane]["cumulative"] for lane in lanes}
        done = all(t["finished"] == 7000 and not t["unresolved"] for t in totals.values())
        self.save("completed" if done else "incomplete", counters=totals)
        return 0 if done else 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--build-job", required=True)
    cli.add_argument("--partition", required=True)
    cli.add_argument("--gpu", choices=("4090", "3090"), default="4090")
    cli.add_argument("--hours", type=int, default=48)
    cli.add_argument("--evaluation-root", type=Path, help="pinned evaluation worktree (defaults to this worktree)")
    cli.add_argument("--recover-prescreen", action="store_true", help="recover expert no-pose exclusions on the frozen run")
    dispatch = Dispatch(cli.parse_args())
    try:
        raise SystemExit(dispatch.run())
    except Exception as error:
        dispatch.save("blocked", error=str(error))
        raise
