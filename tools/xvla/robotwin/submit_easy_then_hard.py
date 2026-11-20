"""Queue a held allocation that finishes original Easy before native-budget Hard."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import time

if __package__:
    from . import submit_hard_native as base
else:
    import submit_hard_native as base

output, atomic, load = base.output, base.atomic, base.load
held, job_fields, accounting = base.held, base.job_fields, base.accounting
queue, gpu_count, require = base.queue, base.gpu_count, base.require
WORKER, EVALUATION = base.WORKER, base.EVALUATION
REPLACE_JOB = "20965"
HERE = Path(__file__).resolve().parent


class Submit(base.Submit):
    def __init__(self, args):
        self.args = args
        self.data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
        self.root = args.output_root.resolve(strict=True)
        self.worker = args.worker_root.resolve(strict=True)
        self.evaluation = args.evaluation_root.resolve(strict=True)
        require(self.root == self.data / "hard-native-budget" / WORKER, "Use the original bcb Hard profile")
        for root, commit in ((self.worker, WORKER), (self.evaluation, EVALUATION)):
            require(output("git", "-C", str(root), "rev-parse", "HEAD") == commit, "Wrong frozen execution source")
            output("git", "-C", str(root), "diff", "--exit-code", "HEAD", "--", "tools/xvla/robotwin")
        self.code = output("git", "-C", str(HERE), "rev-parse", "HEAD")
        files = ("submit_easy_then_hard.py", "easy_then_hard_job.sh", "submit_hard_native.py")
        output("git", "-C", str(HERE), "ls-files", "--error-unmatch", *files)
        output("git", "-C", str(HERE), "diff", "--exit-code", "HEAD", "--", *files)
        for key in ("FYP_CONDA_ROOT", "FYP_ROBOTWIN_ENV", "FYP_ROBOTWIN_CODE"):
            require(Path(os.environ[key]).is_dir(), "Missing qualified environment: " + key)
        require((Path(os.environ["FYP_ROBOTWIN_ENV"]) / "bin/python").is_file(), "Qualified Python is missing")
        self.path = self.root / "easy-first-dispatch.json"
        self.journal = self.root / "easy-first-jobs.tsv"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "worker_commit": WORKER, "evaluation_commit": EVALUATION, "controller_commit": self.code,
            "output_root": str(self.root), "created_at": time.time(), "jobs": [], "stage": "prepared"}
        require(all(self.state.get(key) == value for key, value in {
            "worker_commit": WORKER, "evaluation_commit": EVALUATION,
            "controller_commit": self.code, "output_root": str(self.root)}.items()), "Combined submission identity changed")

    def preflight(self):
        # Reuse the frozen-source, native-Easy900, environment and CPU-controller gates.
        super().preflight()
        self.state["original_easy_before_hard"] = self.state.pop("original_easy_deferred")
        summary = load(self.evaluation, "summarize", "_easy_first_original").summarize(self.data / "panels" / EVALUATION)
        first = summary["overall"]["easy"]["first_batch"]
        require(first["planned"] == first["finished"] == 5000 and first["unresolved"] == first["retry_pending"] == 0
                and 6209 <= self.state["original_easy_before_hard"]["finished"] <= 7000,
                "Original Easy must retain its complete first batch and current6209-or-later progress")
        path = self.root / "direct-dispatch.json"
        raw = path.read_bytes(); previous = json.loads(raw)
        require(previous.get("worker_commit") == WORKER and previous.get("evaluation_commit") == EVALUATION
                and previous.get("output_root") == str(self.root) and previous.get("jobs", [None])[-1] == REPLACE_JOB,
                "The held native-Hard submission history changed")
        digest = hashlib.sha256(raw).hexdigest()
        require(self.state.get("previous_dispatch_sha256", digest) == digest, "Original direct dispatch evidence changed")
        self.state["previous_dispatch_sha256"] = digest
        self.previous = previous
        if not self.state.get("replacement_cancel_intent"):
            if accounting(REPLACE_JOB)[0] == "CANCELLED":
                receipt_path = self.root / "easy-first-reorder.json"
                raw_receipt = receipt_path.read_bytes(); receipt = json.loads(raw_receipt)
                fields = receipt.get("prior_fields", {})
                require(receipt.get("job_id") == REPLACE_JOB and receipt.get("worker_commit") == WORKER
                        and receipt.get("action") == "cancel_owned_held_predecessor"
                        and receipt.get("direct_controller_commit") == previous.get("controller_commit")
                        and receipt.get("batch_sha256") == previous.get("batch_sha256")
                        and fields.get("JobId") == REPLACE_JOB and fields.get("JobName") == "rw-hard-native-budget"
                        and fields.get("JobState") == "PENDING" and fields.get("Reason") == "JobHeldUser"
                        and fields.get("UserId", "").endswith(f"({os.getuid()})")
                        and fields.get("Comment") == previous["job_fields"]["Comment"],
                        "Canceled predecessor lacks matching owned-hold reorder evidence")
                self.state["replacement_cancel_intent"] = fields
                self.state["reorder_receipt_sha256"] = hashlib.sha256(raw_receipt).hexdigest()
                return
            fields = held(REPLACE_JOB, "rw-hard-native-budget")
            require(fields.get("Comment") == previous["job_fields"]["Comment"], "Held Hard token differs from its ledger")
            batch = output("scontrol", "write", "batch_script", REPLACE_JOB, "-")
            require(hashlib.sha256(batch.encode()).hexdigest() == previous["batch_sha256"]
                    and str(self.worker / "tools/xvla/robotwin/hard_native_job.sh") in batch
                    and str(self.evaluation) in batch and str(self.root) in batch,
                    "Held Hard batch differs from its original frozen submission")
            self.state["replacement_verified"] = fields

    def verify_new_job(self, job, token, must_hold=False):
        fields = held(job, "rw-easy-then-hard") if must_hold else job_fields(job)
        require(fields.get("JobId") == job and fields.get("JobName") == "rw-easy-then-hard"
                and fields.get("UserId", "").endswith(f"({os.getuid()})") and fields.get("Comment") == token
                and fields.get("NumCPUs") == "16" and fields.get("NumNodes") in {"1", "1-1"}
                and fields.get("TimeLimit") == "2-00:00:00"
                and (re.fullmatch(r"gres[/:]gpu:4090:1(?:\([^)]*\))?", fields.get("TresPerNode", ""))
                     or {item for item in fields.get("ReqTRES", "").split(",") if item.startswith("gres/gpu")}
                     == {"gres/gpu=1", "gres/gpu:4090=1"}), "Unexpected combined job resources or token")
        batch = output("scontrol", "write", "batch_script", job, "-")
        require(str(HERE / "easy_then_hard_job.sh") in batch and str(self.evaluation) in batch
                and str(self.worker) in batch and str(self.root) in batch and "bash " in batch,
                "Combined job did not use the fixed wrapper and inputs")
        self.state["batch_sha256"] = hashlib.sha256(batch.encode()).hexdigest()
        self.state["wrapper_sha256"] = hashlib.sha256((HERE / "easy_then_hard_job.sh").read_bytes()).hexdigest()
        return fields

    def run(self):
        self.preflight()
        jobs = self.state["jobs"]; intent = self.state.get("submission")
        job = self.recover(intent["token"]) if intent else None
        if job: self.verify_new_job(job, intent["token"], must_hold=True)
        require(gpu_count([REPLACE_JOB, *jobs, *([job] if job else [])]) <= 1,
                "Another experiment already requests the remaining GPU slots")
        if jobs and not self.args.retry:
            status, code = accounting(jobs[-1])
            if status == "COMPLETED":
                total = self.hard["overall"]["cumulative"]
                require(code == "0:0" and self.hard["status"] == "complete" and total["finished"] == 7000
                        and total["unresolved"] == total["retry_pending"] == 0
                        and self.state["original_easy_before_hard"]["finished"] == 7000,
                        "Existing combined job completed with incomplete results")
                self.state["stage"] = "easy_then_hard_complete"
            elif status in {"PENDING", "RUNNING"}:
                fields = self.verify_new_job(jobs[-1], self.state["job_fields"]["Comment"])
                self.state["stage"] = "easy_first_queued_hold" if fields.get("Reason") == "JobHeldUser" else "easy_then_hard_" + status.lower()
            if not self.args.dry_run: atomic(self.path, self.state); self.mirror()
            return jobs[-1]
        if self.args.retry:
            require(jobs and accounting(jobs[-1])[0] in base.RESUMABLE and len(jobs) <= 8,
                    "Retry requires TIMEOUT/NODE_FAIL/PREEMPTED and at most eight infrastructure retries")
        if intent and intent["stage"] == "submitting":
            require(job is not None, "Submission delivery is uncertain; recover its durable token before repeating")
        if self.args.dry_run:
            print(json.dumps({"dry_run": True, "hold": True, "replace_job": REPLACE_JOB,
                              "original_easy_before_hard": self.state["original_easy_before_hard"], "recovered_job": job}))
            return job
        if not intent:
            token = "rw-easy-hard-" + hashlib.sha256(f"{self.root}:{len(jobs)}".encode()).hexdigest()[:24]
            intent = {"token": token, "stage": "ready", "at_unix": time.time()}
            self.state["submission"] = intent; atomic(self.path, self.state)
        if not self.state.get("replacement_cancel_intent"):
            held(REPLACE_JOB, "rw-hard-native-budget")
            self.state["replacement_cancel_intent"] = self.state["replacement_verified"]; atomic(self.path, self.state)
        if accounting(REPLACE_JOB)[0] != "CANCELLED":
            held(REPLACE_JOB, "rw-hard-native-budget"); output("scancel", "--state=PENDING", REPLACE_JOB)
        deadline = time.monotonic() + 120
        while accounting(REPLACE_JOB)[0] != "CANCELLED" or REPLACE_JOB in {row[0] for row in queue()}:
            require(time.monotonic() < deadline, "Held Hard cancellation has not settled"); time.sleep(3)
        if job is None:
            require(gpu_count() < 2, "Two own GPUs remain requested; do not submit a third")
            intent["stage"] = "submitting"; atomic(self.path, self.state)
            command = shlex.join(["bash", str(HERE / "easy_then_hard_job.sh"), str(self.evaluation), str(self.worker), str(self.root)])
            reply = output("sbatch", "--parsable", "--hold", "--partition=GPU", "--exclude=4090node1", "--nodes=1",
                           "--ntasks=1", "--cpus-per-task=16", "--gres=gpu:4090:1", "--time=2880",
                           "--job-name=rw-easy-then-hard", f"--comment={intent['token']}", "--export=ALL",
                           f"--output={self.data}/logs/%j.out", f"--error={self.data}/logs/%j.err", f"--wrap={command}")
            job = reply.partition(";")[0]; require(job.isdigit(), "Invalid sbatch job ID: " + reply)
        fields = self.verify_new_job(job, intent["token"], must_hold=True)
        row = f"{job}\t{intent['token']}"
        if row not in (self.journal.read_text().splitlines() if self.journal.exists() else []):
            with self.journal.open("a") as stream:
                stream.write(row + "\n"); stream.flush(); os.fsync(stream.fileno())
        if job not in jobs: jobs.append(job)
        self.state.update(stage="easy_first_queued_hold", submission=None, held_job=job, job_fields=fields)
        atomic(self.path, self.state); self.mirror()
        return job

    def mirror(self):
        path = self.root / "state.json"; previous = json.loads(path.read_text())
        if "previous_state_backup" not in self.state:
            backup = self.root / f"state.before-easy-first-{time.time_ns()}.json"; atomic(backup, previous)
            self.state["previous_state_backup"] = str(backup); atomic(self.path, self.state)
        previous.pop("original_easy_deferred", None)
        previous.update(stage=self.state["stage"], native_jobs=list(self.state["jobs"]), execution_mode="easy_then_hard",
                        controller_commit=self.code, original_easy_before_hard=self.state["original_easy_before_hard"],
                        easy_first_dispatch=str(self.path))
        atomic(path, previous); atomic(self.root / "easy_first_handoff.json", self.state)


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
            print(f"ROBOTWIN_EASY_THEN_HARD_JOB_ID={job}")
            fields = job_fields(job) if accounting(job)[0] in {"PENDING", "RUNNING"} else {}
            print("ROBOTWIN_RELEASE_REQUIRED=" + str(int(fields.get("Reason") == "JobHeldUser")))


if __name__ == "__main__":
    main()
