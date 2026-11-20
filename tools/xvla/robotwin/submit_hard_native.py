"""Queue the frozen Hard panel on user hold after the PPO/GRPO handoff."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import types

WORKER = "bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460"
EVALUATION = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
EASY = "15728f76ba267fef3bd75c36c7d825c4ab2a4360"
OLD_HARD = "20326"
RESUMABLE = {"TIMEOUT", "NODE_FAIL", "PREEMPTED"}
HERE = Path(__file__).resolve()


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def output(*args):
    return subprocess.check_output(args, text=True, timeout=50).strip()


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def load(root, name, package):
    tools = root / "tools/xvla/robotwin"
    parent = types.ModuleType(package); parent.__path__ = [str(tools)]; sys.modules[package] = parent
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", tools / f"{name}.py")
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def accounting(job):
    for row in output("sacct", "-X", "-nP", "-j", job, "--format=JobIDRaw,State,ExitCode").splitlines():
        fields = row.split("|")
        if len(fields) >= 3 and fields[0] == job:
            return fields[1].split()[0].rstrip("+"), fields[2]
    return "UNKNOWN", None


def job_fields(job):
    text = output("scontrol", "show", "job", job, "-o")
    return dict(re.findall(r"(?:^| )([A-Za-z]+)=([^ ]+)", text))


def held(job, name):
    fields = job_fields(job)
    require(fields.get("JobId") == job and fields.get("JobName") == name
            and fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser"
            and fields.get("UserId", "").endswith(f"({os.getuid()})"), "Expected an owned user-held job: " + job)
    return fields


def queue():
    return [line.split("|", 2) for line in output("squeue", "--me", "-h", "-o", "%i|%b|%k").splitlines() if line.strip()]


def gpu_count(exclude=()):
    count = 0
    for job, gres, _ in queue():
        if job in exclude: continue
        for item in gres.split(","):
            if "gpu" not in item: continue
            match = re.fullmatch(r"gres[/:]gpu(?::[A-Za-z0-9_-]+)?:(\d+)(?:\([^)]*\))?", item)
            require(match is not None, "Unknown GPU request: " + item); count += int(match[1])
    return count


class Submit:
    def __init__(self, args):
        self.args = args
        self.data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
        self.root = args.output_root.resolve(strict=True)
        self.worker = args.worker_root.resolve(strict=True); self.evaluation = args.evaluation_root.resolve(strict=True)
        require(self.root == self.data / "hard-native-budget" / WORKER, "Use the original separate bcb Hard profile")
        for root, commit in ((self.worker, WORKER), (self.evaluation, EVALUATION)):
            require(output("git", "-C", str(root), "rev-parse", "HEAD") == commit, "Wrong frozen execution source")
            output("git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "tools/xvla/robotwin")
        self.code = output("git", "-C", str(HERE.parent), "rev-parse", "HEAD")
        output("git", "-C", str(HERE.parent), "ls-files", "--error-unmatch", HERE.name)
        output("git", "-C", str(HERE.parent), "diff", "--exit-code", "HEAD", "--", HERE.name)
        for key in ("FYP_CONDA_ROOT", "FYP_ROBOTWIN_ENV", "FYP_ROBOTWIN_CODE"):
            require(Path(os.environ[key]).is_dir(), "Missing qualified environment path: " + key)
        require((Path(os.environ["FYP_ROBOTWIN_ENV"]) / "bin/python").is_file(), "Qualified Python is missing")
        self.path = self.root / "direct-dispatch.json"; self.journal = self.root / "direct-jobs.tsv"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "worker_commit": WORKER, "evaluation_commit": EVALUATION, "controller_commit": self.code,
            "output_root": str(self.root), "created_at": time.time(), "jobs": [], "stage": "prepared"}
        require(all(self.state.get(key) == value for key, value in
                    {"worker_commit": WORKER, "evaluation_commit": EVALUATION,
                     "controller_commit": self.code, "output_root": str(self.root)}.items()), "Direct submission identity changed")

    def preflight(self):
        for proc in Path("/proc").iterdir():
            try:
                if not proc.name.isdigit() or proc.stat().st_uid != os.getuid(): continue
                argv = (proc / "cmdline").read_bytes().decode().split("\0")
            except (FileNotFoundError, PermissionError, ProcessLookupError): continue
            require(not (re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name)
                         and any(v.endswith(("/robotwin/dispatch.py", "/hard_native_dispatch.py", "/native_budget_dispatch.py"))
                                 for v in argv)), "A RoboTwin CPU dispatcher is still running")
        source = self.data / "panels" / EVALUATION
        hard = load(self.worker, "hard_native_report", "_hard_direct_worker").summarize(self.root, source)
        self.hard = hard
        require(hard["profile"]["implementation_commit"] == WORKER, "Hard profile implementation changed")
        easy_code = self.evaluation.parent / EASY
        require(output("git", "-C", str(easy_code), "rev-parse", "HEAD") == EASY, "Wrong native Easy reporter checkout")
        output("git", "-C", str(easy_code), "diff", "--exit-code", "HEAD", "--", "tools/xvla/robotwin")
        easy = load(easy_code, "native_budget_report", "_hard_direct_easy").summarize(self.data / "native-budget" / EASY, source)
        total = easy["overall"]["cumulative"]
        require(accounting("20730") == ("COMPLETED", "0:0") and easy["status"] == "complete"
                and total["planned"] == total["finished"] == 900 and total["unresolved"] == total["retry_pending"] == 0,
                "The current native Easy execution has not completed cleanly")
        original = load(self.evaluation, "summarize", "_hard_direct_original").summarize(source)["overall"]["easy"]["cumulative"]
        require(original["planned"] == 7000 and original["unresolved"] == 0, "Original Easy evidence is unresolved")
        self.state["original_easy_deferred"] = {"finished": original["finished"], "planned": 7000,
                                                 "remaining": 7000 - original["finished"]}

    def recover(self, token):
        jobs = {job for job, _, comment in queue() if comment == token}
        since = datetime.fromtimestamp(self.state["created_at"], timezone.utc).strftime("%Y-%m-%d")
        for line in output("sacct", "-X", "-nP", "-u", str(os.getuid()), "-S", since, "--format=JobIDRaw,Comment%100").splitlines():
            row = line.split("|")
            if len(row) >= 2 and row[1] == token: jobs.add(row[0])
        for line in self.journal.read_text().splitlines() if self.journal.exists() else []:
            job, recorded = line.split("\t")
            if recorded == token: jobs.add(job)
        require(len(jobs) <= 1 and all(job.isdigit() for job in jobs), "Ambiguous direct submission token")
        return next(iter(jobs), None)

    def verify_new_job(self, job, token, must_hold=False):
        fields = held(job, "rw-hard-native-budget") if must_hold else job_fields(job)
        require(fields.get("JobId") == job and fields.get("JobName") == "rw-hard-native-budget"
                and fields.get("UserId", "").endswith(f"({os.getuid()})") and fields.get("Comment") == token
                and fields.get("NumCPUs") == "16" and fields.get("NumNodes") in {"1", "1-1"}
                and fields.get("TimeLimit") == "2-00:00:00"
                and (re.fullmatch(r"gres[/:]gpu:4090:1(?:\([^)]*\))?", fields.get("TresPerNode", ""))
                     or ({item for item in fields.get("ReqTRES", "").split(",") if item.startswith("gres/gpu")}
                         == {"gres/gpu=1", "gres/gpu:4090=1"})), "Unexpected native Hard job resources or token")
        batch = output("scontrol", "write", "batch_script", job, "-")
        require(str(self.worker / "tools/xvla/robotwin/hard_native_job.sh") in batch
                and str(self.evaluation) in batch and str(self.root) in batch and "bash " in batch,
                "Native Hard job did not use the frozen wrapper")
        self.state["batch_sha256"] = hashlib.sha256(batch.encode()).hexdigest()
        return fields

    def run(self):
        self.preflight()
        jobs = self.state["jobs"]
        intent = self.state.get("submission")
        job = self.recover(intent["token"]) if intent else None
        if job:
            self.verify_new_job(job, intent["token"], must_hold=True)
        require(gpu_count([OLD_HARD, *jobs, *([job] if job else [])]) <= 1,
                "Another experiment already requests more than the remaining GPU slot")
        if jobs and not self.args.retry:
            status, code = accounting(jobs[-1])
            if status == "COMPLETED":
                total = self.hard["overall"]["cumulative"]
                require(code == "0:0" and self.hard["status"] == "complete" and total["finished"] == 7000
                        and total["unresolved"] == total["retry_pending"] == 0, "Existing Hard job completed with incomplete results")
                self.state["stage"] = "hard_direct_complete"
            elif status in {"PENDING", "RUNNING"}:
                token = self.state["job_fields"]["Comment"]
                fields = self.verify_new_job(jobs[-1], token)
                self.state["stage"] = "hard_queued_user_hold" if fields.get("Reason") == "JobHeldUser" else "hard_direct_" + status.lower()
            if not self.args.dry_run:
                atomic(self.path, self.state); self.mirror()
            return jobs[-1]
        if self.args.retry:
            require(jobs and accounting(jobs[-1])[0] in RESUMABLE and len(jobs) <= 8,
                    "Retry requires TIMEOUT/NODE_FAIL/PREEMPTED and at most eight infrastructure retries")
        if intent and intent["stage"] == "submitting":
            require(job is not None, "Submission delivery is uncertain; recover the durable token before repeating")
        if self.args.dry_run:
            if accounting(OLD_HARD)[0] != "CANCELLED": held(OLD_HARD, "rw-b0-formal-hard")
            print(json.dumps({"dry_run": True, "hold": True, "worker_commit": WORKER,
                              "original_easy_deferred": self.state["original_easy_deferred"], "recovered_job": job}))
            return job
        if not intent:
            token = "rw-hard-direct-" + hashlib.sha256(f"{self.root}:{len(jobs)}".encode()).hexdigest()[:24]
            intent = {"token": token, "stage": "ready", "at_unix": time.time()}
            self.state["submission"] = intent
            atomic(self.path, self.state)
        if not self.state.get("placeholder_cancel_intent"):
            fields = held(OLD_HARD, "rw-b0-formal-hard")
            self.state["placeholder_cancel_intent"] = fields; atomic(self.path, self.state)
        if accounting(OLD_HARD)[0] != "CANCELLED":
            held(OLD_HARD, "rw-b0-formal-hard"); output("scancel", "--state=PENDING", OLD_HARD)
        deadline = time.monotonic() + 120
        while accounting(OLD_HARD)[0] != "CANCELLED" or OLD_HARD in {row[0] for row in queue()}:
            require(time.monotonic() < deadline, "Old held Hard cancellation has not settled"); time.sleep(3)
        if job is None:
            require(gpu_count() < 2, "Two own GPUs remain requested; do not submit a third")
            intent["stage"] = "submitting"; atomic(self.path, self.state)
            command = shlex.join(["bash", str(self.worker / "tools/xvla/robotwin/hard_native_job.sh"), str(self.evaluation), str(self.root)])
            reply = output("sbatch", "--parsable", "--hold", "--partition=GPU", "--exclude=4090node1", "--nodes=1",
                           "--ntasks=1", "--cpus-per-task=16", "--gres=gpu:4090:1", "--time=2880",
                           "--job-name=rw-hard-native-budget", f"--comment={intent['token']}", "--export=ALL",
                           f"--output={self.data}/logs/%j.out", f"--error={self.data}/logs/%j.err", f"--wrap={command}")
            job = reply.partition(";")[0]; require(job.isdigit(), "Invalid sbatch job ID: " + reply)
        fields = self.verify_new_job(job, intent["token"], must_hold=True)
        row = f"{job}\t{intent['token']}"
        if row not in (self.journal.read_text().splitlines() if self.journal.exists() else []):
            with self.journal.open("a") as stream:
                stream.write(row + "\n"); stream.flush(); os.fsync(stream.fileno())
        if job not in jobs: jobs.append(job)
        self.state.update(stage="hard_queued_user_hold", submission=None, held_job=job, job_fields=fields)
        atomic(self.path, self.state)
        self.mirror()
        return job

    def mirror(self):
        jobs = self.state["jobs"]
        path = self.root / "state.json"; previous = json.loads(path.read_text())
        if "previous_state_backup" not in self.state:
            backup = self.root / f"state.before-direct-handoff-{time.time_ns()}.json"; atomic(backup, previous)
            self.state["previous_state_backup"] = str(backup); atomic(self.path, self.state)
        previous.update(stage=self.state["stage"], native_jobs=list(jobs), execution_mode="hard_after_ppo_direct",
                        controller_commit=self.code, original_easy_deferred=self.state["original_easy_deferred"], direct_dispatch=str(self.path))
        atomic(path, previous); atomic(self.root / "direct_handoff.json", self.state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker-root", "evaluation-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--retry", action="store_true"); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
    with ExitStack() as stack:
        for name in ("hard-native-dispatch.lock", "native-budget-dispatch.lock", "dispatch.lock", "submit.lock"):
            lock = stack.enter_context((data / name).open("a")); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        job = Submit(args).run()
        if job:
            print(f"ROBOTWIN_HARD_JOB_ID={job}")
            fields = job_fields(job) if accounting(job)[0] in {"PENDING", "RUNNING"} else {}
            print("ROBOTWIN_HARD_RELEASE_REQUIRED=" + str(int(fields.get("Reason") == "JobHeldUser")))


if __name__ == "__main__":
    main()
