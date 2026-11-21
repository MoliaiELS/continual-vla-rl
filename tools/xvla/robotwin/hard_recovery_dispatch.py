"""Monitor an adopted Hard recovery job without changing frozen experiment metadata."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
EVALUATION = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
WORKER = "bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460"
TOKEN_PREFIX = "rw-hard-recovery-"
RESUMABLE = {"TIMEOUT", "NODE_FAIL", "PREEMPTED"}
DONE = RESUMABLE | {"COMPLETED", "FAILED", "CANCELLED", "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "REVOKED"}
POLL_SECONDS = 40
MAX_RETRIES = 8


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def output(*command):
    return subprocess.check_output(command, text=True, stderr=subprocess.PIPE, timeout=50).strip()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def slurm_fields(raw):
    return dict(re.findall(r"(?:^|\s)([A-Za-z][A-Za-z0-9_/]*)=([^\s]+)", raw))


def duration_seconds(value):
    days, separator, clock = value.partition("-")
    if not separator:
        clock, days = days, "0"
    hours, minutes, seconds = map(int, clock.split(":"))
    return int(days) * 86400 + hours * 3600 + minutes * 60 + seconds


class RecoveryDispatch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
        self.evaluation = cfg.evaluation_root.resolve(strict=True)
        self.worker = cfg.worker_root.resolve(strict=True)
        self.profile = cfg.output_root.resolve(strict=True)
        require(cfg.adopt_job.isdigit(), "The adopted Slurm job ID must be numeric")
        require(self.profile == self.data / "hard-native-budget" / WORKER,
                "Use the original bcb Hard profile without copying or rewriting it")
        for path, commit in ((self.evaluation, EVALUATION), (self.worker, WORKER)):
            require(output("git", "-C", str(path), "rev-parse", "HEAD") == commit,
                    f"Unexpected frozen worktree: {path}")
            require(Path(output("git", "-C", str(path), "rev-parse", "--show-toplevel")).resolve() == path,
                    "Evaluation and worker roots must be complete Git worktrees")
            require(not output("git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"),
                    f"Frozen worktree has tracked changes: {path}")
        self.code = output("git", "-C", str(HERE), "rev-parse", "HEAD")
        output("git", "-C", str(HERE), "ls-files", "--error-unmatch", "hard_recovery_dispatch.py",
               "hard_recovery_job.sh", "hard_recovery_panel.py", "hard_screen_recovery.py")
        output("git", "-C", str(HERE), "diff", "--exit-code", "HEAD", "--", ".")
        self.root = self.data / "recovery-dispatch" / self.code
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "jobs").mkdir(exist_ok=True)
        self.path = self.root / "state.json"
        self.journal = self.root / "jobs.tsv"
        raw = (self.profile / "profile.json").read_bytes()
        profile = json.loads(raw)
        require(profile.get("implementation_commit") == WORKER and profile.get("condition") == "hard"
                and profile.get("planned_episodes") == 7000 and profile.get("planned_first_batch") == 5000
                and profile.get("planned_extension") == 2000
                and re.fullmatch(r"[0-9a-f]{64}", profile.get("protocol_fingerprint", "")),
                "The frozen Hard profile identity or 7,000-episode plan differs")
        environment = {name: os.environ[name] for name in
                       ("FYP_CONDA_ROOT", "FYP_ROBOTWIN_DATA", "FYP_ROBOTWIN_ENV", "FYP_ROBOTWIN_CODE")}
        configuration = dict(evaluation_root=str(self.evaluation), worker_root=str(self.worker),
                             output_root=str(self.profile), adopt_job=cfg.adopt_job, partition=cfg.partition,
                             hours=cfg.hours, environment=environment)
        self.state = json.loads(self.path.read_bytes()) if self.path.is_file() else {
            "configuration": configuration, "orchestration_commit": self.code,
            "profile_fingerprint": profile["protocol_fingerprint"], "profile_sha256": sha256(raw),
            "created_at": time.time(), "stage": "prepared", "jobs": [], "submission": None,
        }
        require(self.state["configuration"] == configuration and self.state["orchestration_commit"] == self.code
                and self.state["profile_sha256"] == sha256(raw)
                and self.state["profile_fingerprint"] == profile["protocol_fingerprint"],
                "Recovery controller arguments, code or frozen profile changed")

    def save(self, stage, **values):
        self.state.update(stage=stage, at_unix=time.time(), **values)
        atomic_json(self.path, self.state)
        print(json.dumps({"stage": stage, **values}, allow_nan=False), flush=True)

    def accounting(self, job):
        raw = output("sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode")
        matches = [parts for line in raw.splitlines() if len(parts := line.split("|")) >= 3 and parts[0] == job]
        require(len(matches) <= 1, f"Ambiguous Slurm accounting for {job}")
        return (matches[0][1].split()[0].rstrip("+"), matches[0][2]) if matches else ("UNKNOWN", None)

    def command(self):
        return ["bash", str(HERE / "hard_recovery_job.sh"), str(self.evaluation), str(self.worker), str(self.profile)]

    def validate_job(self, job, fields, batch, token=None):
        require(fields.get("JobId") == job and fields.get("UserId", "").endswith(f"({os.getuid()})"),
                f"Job {job} is not the identified user's job")
        comment = fields.get("Comment", "")
        require(comment.startswith(TOKEN_PREFIX) and re.fullmatch(r"[A-Za-z0-9_-]+", comment)
                and (token is None or comment == token), f"Job {job} has another submission token")
        require(fields.get("Partition") == self.cfg.partition and fields.get("NumNodes") in {"1", "1-1"}
                and fields.get("NumCPUs") == "16" and fields.get("NumTasks") == "1"
                and fields.get("CPUs/Task") == "16" and fields.get("ExcNodeList") == "4090node1"
                and duration_seconds(fields["TimeLimit"]) == self.cfg.hours * 3600,
                f"Job {job} resources differ from the recovery allocation")
        requested = fields.get("ReqTRES", "").split(",")
        node_gres = fields.get("TresPerNode", "")
        require(("gres/gpu=1" in requested and "gres/gpu:4090=1" in requested)
                or node_gres == "gres/gpu:4090:1", f"Job {job} is not a typed single RTX 4090 allocation")
        require(all(item in {"gres/gpu=1", "gres/gpu:4090=1"} for item in requested if "gpu" in item),
                f"Job {job} has an additional or different GPU request")
        commands = [shlex.split(line) for line in batch.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        require(commands == [self.command()], f"Job {job} does not execute the exact fixed recovery wrapper")

    def inspect_job(self, job, token=None):
        path = self.root / "jobs" / f"{job}.json"
        previous = json.loads(path.read_bytes()) if path.is_file() else None
        try:
            fields = slurm_fields(output("scontrol", "show", "job", job, "-o"))
            batch = output("scontrol", "write", "batch_script", job, "-")
        except subprocess.CalledProcessError:
            require(previous is not None, f"Job {job} left scontrol without a preserved batch receipt")
            fields, batch = previous["fields"], previous["batch_script"]
        self.validate_job(job, fields, batch, token)
        receipt = dict(job=job, orchestration_commit=self.code, evaluation_commit=EVALUATION, worker_commit=WORKER,
                       profile=str(self.profile), profile_sha256=self.state["profile_sha256"],
                       token=fields["Comment"], batch_script=batch, batch_sha256=sha256(batch.encode()), fields=fields)
        if previous:
            for key in ("job", "orchestration_commit", "evaluation_commit", "worker_commit", "profile",
                        "profile_sha256", "token", "batch_sha256"):
                require(previous.get(key) == receipt[key], f"Job {job} preserved receipt identity changed: {key}")
        else:
            atomic_json(path, receipt)
        return receipt

    def append_job(self, job, token):
        expected = [job, token, EVALUATION, WORKER, self.code, str(self.profile)]
        rows = [line.split("\t") for line in self.journal.read_text().splitlines()] if self.journal.is_file() else []
        matches = [row for row in rows if row[0] == job or row[1] == token]
        require(not matches or matches == [expected], "Conflicting recovery submission journal")
        if not matches:
            with self.journal.open("a", encoding="utf-8") as stream:
                stream.write("\t".join(expected) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def queue(self):
        raw = output("squeue", "--me", "-h", "-o", "%i|%b|%k")
        rows = [line.split("|", 2) for line in raw.splitlines() if line.strip()]
        require(all(len(row) == 3 and row[0].isdigit() for row in rows), "Unexpected own Slurm queue format")
        return rows

    def check_slot(self):
        rows = self.queue()
        require(len(rows) < 2, "Two own jobs are already running or pending, including held jobs")
        count = 0
        for _, resources, _ in rows:
            for item in resources.split(","):
                if "gpu" not in item:
                    continue
                match = re.fullmatch(r"gres[/:]gpu(?::[A-Za-z0-9_-]+)?:(\d+)(?:\([^)]*\))?", item)
                require(match is not None, f"Unknown own GPU request: {item}")
                count += int(match[1])
        require(count < 2, "Two own GPUs are already requested, including held jobs")

    def tagged_job(self, token):
        found = {job for job, _, comment in self.queue() if comment == token}
        since = datetime.fromtimestamp(self.state["created_at"], timezone.utc).strftime("%Y-%m-%d")
        raw = output("sacct", "-X", "-nP", "-u", str(os.getuid()), "-S", since,
                     "--format=JobIDRaw,Comment%120")
        found.update(parts[0] for line in raw.splitlines()
                     if len(parts := line.split("|")) >= 2 and parts[1] == token)
        if self.journal.is_file():
            found.update(row[0] for line in self.journal.read_text().splitlines()
                         if len(row := line.split("\t")) == 6 and row[1] == token)
        require(len(found) <= 1 and all(job.isdigit() for job in found), f"Ambiguous recovery token {token}: {found}")
        return next(iter(found), None)

    def adopt(self):
        job = self.cfg.adopt_job
        receipt = self.inspect_job(job)
        state, code = self.accounting(job)
        live = receipt["fields"].get("JobState")
        require(state in {"PENDING", "RUNNING"} or (state == "UNKNOWN" and live in {"PENDING", "RUNNING"})
                or (state, code) == ("COMPLETED", "0:0"),
                f"Initial adopted job {job} is not pending, running or cleanly completed: {state} {code}")
        self.append_job(job, receipt["token"])
        self.state["jobs"].append(job)
        self.save("running", job=job, last_slurm_state=state)
        return job

    def start(self):
        import fcntl

        intent = self.state.get("submission")
        if intent:
            job = self.tagged_job(intent["token"])
            require(job is not None, "Submission delivery is uncertain; inspect the durable token before retrying")
        else:
            token = TOKEN_PREFIX + sha256(f"{self.root}:{len(self.state['jobs'])}".encode())[:24]
            with (self.data / "submit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.check_slot()
                require(sha256((self.profile / "profile.json").read_bytes()) == self.state["profile_sha256"],
                        "Frozen Hard profile changed before resubmission")
                intent = dict(token=token, at_unix=time.time(), attempt=len(self.state["jobs"]))
                self.save("submitting", submission=intent)
                logs = self.data / "logs"
                logs.mkdir(exist_ok=True)
                launched = output("sbatch", "--parsable", f"--partition={self.cfg.partition}", "--exclude=4090node1",
                                  "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--gres=gpu:4090:1",
                                  f"--time={self.cfg.hours * 60}", "--job-name=rw-hard-recovery", f"--comment={token}",
                                  f"--output={logs}/%j.out", f"--error={logs}/%j.err", "--export=ALL",
                                  f"--wrap={shlex.join(self.command())}")
                job = launched.partition(";")[0]
                require(job.isdigit(), f"Unexpected recovery sbatch response: {launched}")
        self.inspect_job(job, intent["token"])
        self.append_job(job, intent["token"])
        require(job not in self.state["jobs"], "Recovery job already appears in controller history")
        self.state["jobs"].append(job)
        self.save("running", submission=None, job=job, last_slurm_state=None)
        return job

    def wait_slot_release(self, job):
        deadline = time.monotonic() + 300
        while job in {row[0] for row in self.queue()}:
            require(time.monotonic() < deadline, f"Slurm has not released job {job}'s queue slot")
            time.sleep(5)

    def verify_report(self):
        require(sha256((self.profile / "profile.json").read_bytes()) == self.state["profile_sha256"],
                "Frozen Hard profile changed before completion verification")
        target = self.root / "reports"
        target.mkdir(exist_ok=True)
        command = [sys.executable, str(self.worker / "tools/xvla/robotwin/hard_native_report.py"),
                   "--input", str(self.profile), "--source-panel", str(self.data / "panels" / EVALUATION),
                   "--output", str(target)]
        with (target / "stdout.log").open("ab") as stdout, (target / "stderr.log").open("ab") as stderr:
            code = subprocess.run(command, cwd=self.evaluation, env=dict(os.environ, PYTHONPATH=str(self.evaluation)),
                                  stdout=stdout, stderr=stderr).returncode
        require(code == 0, f"Frozen Hard reporter failed ({code}); inspect {target / 'stderr.log'}")
        report = json.loads((target / "summary.json").read_bytes())
        total = report["overall"]["cumulative"]
        require(report.get("protocol_fingerprint") == self.state["profile_fingerprint"]
                and report.get("status") == "complete" and total["planned"] == total["finished"] == 7000
                and total["unresolved"] == total["retry_pending"] == 0,
                f"Hard recovery finished without a complete verified 7,000-episode report: {total}")
        self.save("complete", report=str(target / "summary.json"), counts=total)

    def run(self):
        if self.state["stage"] == "complete":
            return
        job = self.start() if self.state.get("submission") else (self.state["jobs"][-1] if self.state["jobs"] else self.adopt())
        self.inspect_job(job)
        unknown_since = None
        while True:
            state, code = self.accounting(job)
            if state != self.state.get("last_slurm_state"):
                self.save("running", job=job, last_slurm_state=state, last_exit_code=code)
            if state in DONE:
                self.wait_slot_release(job)
                if state in RESUMABLE and len(self.state["jobs"]) <= MAX_RETRIES:
                    job = self.start()
                    unknown_since = None
                    continue
                require((state, code) == ("COMPLETED", "0:0"),
                        f"Recovery job {job} ended {state} {code}; inspect recorded errors or an explicit boundary request")
                self.verify_report()
                return
            if state == "UNKNOWN":
                unknown_since = unknown_since or time.monotonic()
                require(time.monotonic() - unknown_since < 600, f"No Slurm accounting record for recovery job {job}")
            else:
                unknown_since = None
            time.sleep(POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-root", "worker-root", "output-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adopt-job", required=True)
    parser.add_argument("--hours", type=int, choices=range(1, 73), default=48)
    parser.add_argument("--partition", default="GPU")
    cfg = parser.parse_args()
    import fcntl

    data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
    with ExitStack() as stack:
        for name in ("hard-recovery-dispatch.lock", "native-budget-dispatch.lock", "dispatch.lock"):
            lock = stack.enter_context((data / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch = RecoveryDispatch(cfg)
        try:
            dispatch.run()
        except Exception as error:
            dispatch.save("attention", error=f"{type(error).__name__}: {error}",
                          stderr=getattr(error, "stderr", None))
            raise


if __name__ == "__main__":
    main()
