"""Run the five-task native-budget panel, then restore the original Easy queue."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

if __package__:
    from .dispatch import Dispatch, RESUMABLE, DONE, output
else:
    from dispatch import Dispatch, RESUMABLE, DONE, output


HERE = Path(__file__).resolve().parent
EVALUATION = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
PAUSED_EASY = "20728"
HELD_HARD = "20326"
POLL_SECONDS = 40
MAX_RETRIES = 8


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def own_dispatchers(evaluation_root):
    matches = []
    for process in Path("/proc").iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
            argv = (process / "cmdline").read_bytes().decode().split("\0")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if (re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name)
                and any(value.endswith("/tools/xvla/robotwin/dispatch.py") for value in argv)
                and "--evaluation-root" in argv and str(evaluation_root) in argv):
            matches.append(int(process.name))
    return matches


class NativeBudgetDispatch:
    def __init__(self, cfg):
        self.cfg = cfg
        self.data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
        self.root = cfg.output_root.resolve()
        self.evaluation = cfg.evaluation_root.resolve(strict=True)
        protected = ("panels", "archives", "models", "sim-assets", "jobs", "logs", "dispatch", "evidence", "investigations")
        require(self.root != self.data and self.root.is_relative_to(self.data)
                and not any(self.root.is_relative_to(self.data / name) for name in protected),
                "Native-budget output must be a separate directory under the experiment data root")
        require(output("git", "-C", str(self.evaluation), "rev-parse", "HEAD") == EVALUATION,
                "Use the original fixed 300-anchor evaluation worktree as the source")
        self.code = output("git", "-C", str(HERE), "rev-parse", "HEAD")
        output("git", "-C", str(HERE), "ls-files", "--error-unmatch", "native_budget_dispatch.py",
               "native_budget.py", "native_budget_job.sh", "native_budget_panel.py", "native_budget_report.py")
        output("git", "-C", str(HERE), "diff", "--exit-code", "HEAD", "--", ".")
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "state.json"
        configuration = dict(evaluation_root=str(self.evaluation), output_root=str(self.root),
                             build_job=cfg.build_job, partition=cfg.partition, gpu=cfg.gpu, hours=cfg.hours)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "configuration": configuration, "orchestration_commit": self.code,
            "stage": "prepared", "native_jobs": [], "created_at": time.time(),
        }
        require(self.state["configuration"] == configuration and self.state["orchestration_commit"] == self.code,
                "Native-budget dispatcher arguments or fixed source commit changed")
        self.handoff_path = self.data / "evidence/native-budget-handoff-20261119.json"
        self.journal_path = self.data / "evidence/native-budget-jobs.tsv"

    def save(self, stage, **values):
        self.state.update(stage=stage, at_unix=time.time(), **values)
        temporary = self.path.with_suffix(".json.tmp")
        with temporary.open("w") as stream:
            json.dump(self.state, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.path)
        print(json.dumps({"stage": stage, **values}), flush=True)

    def check_handoff(self):
        handoff_bytes = self.handoff_path.read_bytes()
        handoff = json.loads(handoff_bytes)
        require(handoff.get("paused_easy_job") == PAUSED_EASY and handoff.get("held_hard_job") == HELD_HARD
                and handoff.get("evaluation_commit") == EVALUATION, "Unexpected native-budget queue handoff")
        require(Dispatch.accounting(self, PAUSED_EASY)[0] == "CANCELLED",
                "The old pending Easy extension has not been canceled")
        self.check_hard_hold()
        require(not own_dispatchers(self.evaluation), "The original dispatcher is still running")
        checksum = hashlib.sha256(handoff_bytes).hexdigest()
        require(self.state.get("handoff_sha256", checksum) == checksum, "Recorded queue handoff changed")
        self.state["handoff_sha256"] = checksum

    def check_hard_hold(self):
        fields = Dispatch.hard_job_fields(HELD_HARD)
        require(fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser"
                and fields.get("UserId", "").endswith(f"({os.getuid()})"),
                "The original Hard job must retain its existing user hold")

    def queue(self):
        text = output("squeue", "--me", "-h", "-o", "%i|%b|%k")
        return [line.split("|", 2) for line in text.splitlines() if line.strip()]

    def check_gpu_slot(self):
        count = 0
        for _, gres, _ in self.queue():
            for item in gres.split(","):
                if "gpu" not in item:
                    continue
                match = re.fullmatch(r"gres[/:]gpu(?::[A-Za-z0-9_-]+)?:(\d+)(?:\([^)]*\))?", item)
                require(match is not None, f"Unknown own GPU request: {item}")
                count += int(match[1])
        require(count < 2, "Two own GPUs are already running or pending, including held Hard")

    def tagged_jobs(self, token):
        found = {job for job, _, comment in self.queue() if comment == token}
        since = datetime.fromtimestamp(self.state["created_at"], timezone.utc).strftime("%Y-%m-%d")
        text = output("sacct", "-X", "-nP", "-u", str(os.getuid()), "-S", since,
                      "--format=JobIDRaw,Comment%120")
        found.update(parts[0] for line in text.splitlines()
                     if len(parts := line.split("|")) >= 2 and parts[1] == token)
        require(all(job.isdigit() for job in found) and len(found) <= 1,
                f"Ambiguous jobs for durable submission token {token}: {found}")
        return next(iter(found), None)

    def append_native_job(self, job, token):
        row = [job, EVALUATION, self.code, str(self.root), token]
        rows = [line.split("\t") for line in self.journal_path.read_text().splitlines()] if self.journal_path.exists() else []
        same_job = [value for value in rows if value[0] == job]
        require(not same_job or same_job == [row], "Native-budget job journal identity changed")
        if not same_job:
            with self.journal_path.open("a") as stream:
                stream.write("\t".join(row) + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def start_native(self):
        intent = self.state.get("native_submission")
        if intent:
            job = self.tagged_jobs(intent["token"])
            if job is None:
                for line in self.journal_path.read_text().splitlines() if self.journal_path.exists() else []:
                    values = line.split("\t")
                    if values[1:] == [EVALUATION, self.code, str(self.root), intent["token"]]:
                        require(job in (None, values[0]), "Duplicate native-budget journal token")
                        job = values[0]
            require(job is not None, "Submission may have reached Slurm; recover its token before restarting")
        else:
            self.check_hard_hold()
            attempt = len(self.state["native_jobs"])
            token = "rw-native-" + hashlib.sha256(f"{self.root}:{self.code}:{attempt}".encode()).hexdigest()[:24]
            with (self.data / "submit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.check_gpu_slot()
                intent = {"token": token, "attempt": attempt, "at_unix": time.time()}
                self.save("submitting", native_submission=intent)
                logs = self.data / "logs"
                logs.mkdir(exist_ok=True)
                launch = output("sbatch", "--parsable", f"--partition={self.cfg.partition}", "--exclude=4090node1",
                                "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--gres=gpu:4090:1",
                                f"--time={self.cfg.hours * 60}", "--job-name=rw-native-budget-easy",
                                f"--comment={token}", f"--output={logs}/%j.out", f"--error={logs}/%j.err",
                                "--export=ALL", str(HERE / "native_budget_job.sh"), str(self.evaluation), str(self.root))
                job = launch.partition(";")[0]
                require(job.isdigit(), f"Unexpected sbatch response: {launch}")
        self.append_native_job(job, intent["token"])
        require(job not in self.state["native_jobs"], "Native-budget job was already recorded")
        self.state["native_jobs"].append(job)
        self.save("native_running", native_submission=None, pending_since=time.time(), native_job=job)
        return job

    def run_native(self):
        job = self.start_native() if self.state.get("native_submission") or not self.state["native_jobs"] else self.state["native_jobs"][-1]
        unknown_since = None
        while True:
            state, code = Dispatch.accounting(self, job)
            if state in DONE:
                if state in RESUMABLE and len(self.state["native_jobs"]) <= MAX_RETRIES:
                    self.wait_slot_release(job)
                    job = self.start_native()
                    continue
                require((state, code) == ("COMPLETED", "0:0"),
                        f"Native-budget job {job} ended {state} {code}; inspect its recorded errors")
                self.wait_slot_release(job)
                self.save("native_complete", native_job=job)
                return
            if state == "UNKNOWN":
                unknown_since = unknown_since or time.time()
                require(time.time() - unknown_since < 600, f"No Slurm accounting record for native-budget job {job}")
            else:
                unknown_since = None
            if state == "PENDING":
                if self.state.get("pending_since") is None:
                    self.save("native_running", pending_since=time.time())
            elif state == "RUNNING" and self.state.get("pending_since"):
                self.save("native_running", pending_since=None)
            time.sleep(POLL_SECONDS)

    def wait_slot_release(self, job):
        deadline = time.monotonic() + 300
        while job in {row[0] for row in self.queue()}:
            require(time.monotonic() < deadline, f"Slurm has not released job {job}'s submission slot")
            time.sleep(5)

    def verify_report(self):
        target = self.root / "reports"
        subprocess.run([sys.executable, str(HERE / "native_budget_report.py"), "--input", str(self.root),
                        "--source-panel", str(self.data / "panels" / EVALUATION), "--output", str(target)], check=True)
        report = json.loads((target / "summary.json").read_text())
        total = report["overall"]["cumulative"]
        require(report["status"] == "complete" and total["planned"] == total["finished"] == 900
                and total["unresolved"] == total["retry_pending"] == 0,
                f"Native-budget panel is incomplete: {total}")
        self.save("native_complete", report=str(target / "summary.json"), counts=total)

    def restore_original(self):
        self.check_hard_hold()
        self.save("resuming_original")
        with (self.data / "dispatch.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            cfg = argparse.Namespace(build_job=self.cfg.build_job, partition=self.cfg.partition, gpu="4090",
                                     hours=self.cfg.hours, evaluation_root=self.evaluation, recover_prescreen=True,
                                     easy_first=True, pilot_root=None, paused_easy_job=None, parallel_easy=True)
            original = Dispatch(cfg)
            history = original.state["jobs"].get("extension:easy", [])
            require("20688" in history and PAUSED_EASY in history, "Original Easy extension history changed")
            if history[-1] == PAUSED_EASY:
                require(Dispatch.accounting(self, PAUSED_EASY)[0] == "CANCELLED", "Original Easy replacement is still queued")
                intent = self.state.get("original_submission")
                if intent is None:
                    token = "rw-restore-" + hashlib.sha256(str(self.root).encode()).hexdigest()[:24]
                    intent = {"token": token, "at_unix": time.time()}
                    self.save("resuming_original", original_submission=intent)
                    fresh = True
                else:
                    fresh = False
                tagged = self.tagged_jobs(intent["token"])
                # The existing submit helper journals its job before Dispatch
                # saves state. A durable tag also covers a journal-write crash.
                journaled = original.journal_submission("extension", "easy", history)
                require(not tagged or not journaled or tagged == journaled, "Original resume jobs disagree")
                if tagged and not journaled:
                    self.repair_original_journals(tagged)
                    journaled = tagged
                require(fresh or journaled is not None,
                        "Original resume delivery is uncertain; recover its durable Slurm token first")
                previous = os.environ.get("SBATCH_COMMENT")
                os.environ["SBATCH_COMMENT"] = intent["token"]
                try:
                    job = original.start("extension", "easy", self.cfg.hours, retry=True)
                finally:
                    if previous is None:
                        os.environ.pop("SBATCH_COMMENT", None)
                    else:
                        os.environ["SBATCH_COMMENT"] = previous
            else:
                job = history[-1]
                require(job == self.state.get("original_resume_job") or job == self.tagged_jobs(
                    self.state.get("original_submission", {}).get("token", "__not_submitted__")),
                    "Another original extension has appeared; review before restoring its dispatcher")
            original.state.pop("error", None)
            original.save("extension", native_budget_priority={"output_root": str(self.root), "completed": True,
                                                               "replacement_job": job, "paused_job": PAUSED_EASY})
            self.save("resuming_original", original_resume_job=job)
        self.start_original_dispatcher()

    def repair_original_journals(self, job):
        require(Dispatch.accounting(self, job)[0] in {"PENDING", "RUNNING", "COMPLETED"}, "Original replacement has failed")
        batch = output("scontrol", "write", "batch_script", job, "-")
        require(str(HERE / "job.sh") in batch and str(self.evaluation) in batch
                and "--eval-workers 2" in batch and "extension" in batch, "Unexpected original replacement batch script")
        with (self.data / "submit.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            for file, row in (
                ("prescreen-recovery-jobs.tsv", [job, EVALUATION, self.code]),
                ("parallel-eval-jobs.tsv", [job, EVALUATION, self.code, "2"]),
                ("jobs.tsv", [job, "extension", "easy", "4090", EVALUATION, ""]),
            ):
                path = self.data / "evidence" / file
                rows = [line.split("\t") for line in path.read_text().splitlines()] if path.exists() else []
                matches = [value for value in rows if value[0].partition(";")[0] == job]
                require(not matches or matches == [row], f"Original job journal conflict: {path}")
                if not matches:
                    with path.open("a") as stream:
                        stream.write("\t".join(row) + "\n")
                        stream.flush()
                        os.fsync(stream.fileno())

    def start_original_dispatcher(self):
        existing = own_dispatchers(self.evaluation)
        if not existing:
            log = self.data / "dispatch" / EVALUATION / "dispatch.log"
            argv = ["flock", "-n", str(self.data / "dispatch.lock"), sys.executable, "-u", str(HERE / "dispatch.py"),
                    "--evaluation-root", str(self.evaluation), "--build-job", self.cfg.build_job,
                    "--partition", self.cfg.partition, "--gpu", "4090", "--hours", str(self.cfg.hours),
                    "--recover-prescreen", "--easy-first", "--parallel-easy"]
            with log.open("ab") as stream:
                process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            for _ in range(10):
                existing = own_dispatchers(self.evaluation)
                if existing:
                    break
                require(process.poll() is None, "Original dispatcher did not acquire its lock; inspect its log")
                time.sleep(0.5)
        require(len(existing) == 1, f"Expected exactly one original dispatcher, found {existing}")
        self.save("complete", original_dispatcher_pid=existing[0])

    def run(self):
        if self.state.get("stage") == "complete":
            return
        if self.state.get("original_resume_job"):
            self.start_original_dispatcher()
            return
        self.check_handoff()
        self.run_native()
        self.verify_report()
        self.restore_original()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-job", default="20298")
    parser.add_argument("--partition", default="GPU")
    parser.add_argument("--gpu", choices=("4090",), default="4090")
    parser.add_argument("--hours", type=int, default=48, choices=range(1, 73))
    cfg = parser.parse_args()
    data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
    with (data / "native-budget-dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch = NativeBudgetDispatch(cfg)
        try:
            dispatch.run()
        except Exception as error:
            dispatch.save("attention", error=str(error))
            raise


if __name__ == "__main__":
    main()
