"""Finish both Easy cohorts, then replace the held Hard job with native budgets."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import time
import types

if __package__:
    from .dispatch import Dispatch, DONE, RESUMABLE, output
    from .native_budget_dispatch import NativeBudgetDispatch, EVALUATION, HELD_HARD, PAUSED_EASY, own_dispatchers, require
else:
    from dispatch import Dispatch, DONE, RESUMABLE, output
    from native_budget_dispatch import NativeBudgetDispatch, EVALUATION, HELD_HARD, PAUSED_EASY, own_dispatchers, require


HERE = Path(__file__).resolve().parent
EASY_COMMIT = "15728f76ba267fef3bd75c36c7d825c4ab2a4360"
EASY_JOB = "20730"


def fixed_easy_controller(evaluation):
    """Import both controller modules from their original immutable worktree."""
    tools = evaluation.parent / EASY_COMMIT / "tools/xvla/robotwin"
    require(output("git", "-C", str(tools), "rev-parse", "HEAD") == EASY_COMMIT,
            "The original native Easy controller worktree is missing")
    output("git", "-C", str(tools), "diff", "--exit-code", "HEAD", "--", ".")
    package = "_robotwin_easy_native_157"
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(tools)]
    sys.modules[package] = namespace
    for name in ("dispatch", "native_budget_dispatch"):
        qualified = f"{package}.{name}"
        spec = importlib.util.spec_from_file_location(qualified, tools / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module
        spec.loader.exec_module(module)
    return module.NativeBudgetDispatch, tools


class HardNativeDispatch(NativeBudgetDispatch):
    def __init__(self, cfg):
        super().__init__(cfg)
        output("git", "-C", str(HERE), "ls-files", "--error-unmatch", "hard_native_dispatch.py",
               "hard_native.py", "hard_native_job.sh", "hard_native_panel.py", "hard_native_pilot.py", "hard_native_report.py")
        require(self.root == self.data / "hard-native-budget" / self.code,
                "Use the separate Hard profile directory named by its fixed implementation commit")
        self.easy_root = cfg.easy_native_root.resolve(strict=True)
        require(self.easy_root == self.data / "native-budget" / EASY_COMMIT,
                "Use the original 157 native Easy output without copying or rewriting it")
        previous = self.state.get("easy_native_root", str(self.easy_root))
        require(previous == str(self.easy_root), "The adopted native Easy output changed")
        self.state["easy_native_root"] = previous
        self.receipt_path = self.root / "adoption.json"
        self.journal_path = self.data / "evidence/hard-native-jobs.tsv"
        profile = json.loads((self.root / "profile.json").read_text())
        require(profile.get("implementation_commit") == self.code
                and profile.get("condition") == "hard" and profile.get("planned_episodes") == 7000
                and profile.get("planned_first_batch") == 5000 and profile.get("planned_extension") == 2000
                and len(profile.get("task_ids", [])) == len(set(profile.get("task_ids", []))) == 50
                and len(profile.get("priority_tasks", [])) == len(set(profile.get("priority_tasks", []))) == 10
                and set(profile["priority_tasks"]) <= set(profile["task_ids"])
                and profile.get("per_task_episodes") == {
                    task: 300 if task in profile["priority_tasks"] else 100 for task in profile["task_ids"]},
                "Prepare the complete 50-task Hard native-budget profile first")
        self.unknown_since = {}

    def accounting_checked(self, job):
        state, code = Dispatch.accounting(self, job)
        if state == "UNKNOWN":
            since = self.unknown_since.setdefault(job, time.monotonic())
            require(time.monotonic() - since < 600, f"No Slurm accounting record for adopted job {job}")
        else:
            self.unknown_since.pop(job, None)
        return state, code

    def adopt_easy_native(self):
        raw = self.receipt_path.read_bytes()
        receipt = json.loads(raw)
        require(receipt.get("evaluation_commit") == EVALUATION
                and receipt.get("easy_native_commit") == EASY_COMMIT
                and receipt.get("easy_native_root") == str(self.easy_root)
                and receipt.get("easy_native_job") == EASY_JOB
                and receipt.get("held_hard_job") == HELD_HARD, "Unexpected Hard controller adoption receipt")
        checksum = hashlib.sha256(raw).hexdigest()
        require(self.state.get("adoption_sha256", checksum) == checksum, "The controller adoption receipt changed")
        require(not own_dispatchers(self.evaluation), "Stop the verified original CPU dispatcher before adoption")
        easy_state = json.loads((self.easy_root / "state.json").read_text())
        jobs = easy_state.get("native_jobs", [])
        require(easy_state.get("orchestration_commit") == EASY_COMMIT
                and jobs and jobs[0] == EASY_JOB and all(job.isdigit() for job in jobs)
                and easy_state.get("configuration", {}).get("evaluation_root") == str(self.evaluation)
                and easy_state["configuration"].get("output_root") == str(self.easy_root)
                and all(easy_state["configuration"].get(key) == value for key, value in
                        {"build_job": "20298", "partition": "GPU", "gpu": "4090", "hours": 48}.items()),
                "The native Easy execution identity changed")
        controller, tools = fixed_easy_controller(self.evaluation)
        if "easy_native_batch_sha256" not in self.state:
            # A completed job may have left slurmctld before the controller
            # handoff. The operator can preserve its exact batch text in the
            # adoption receipt while it is still available.
            try:
                batch = output("scontrol", "write", "batch_script", EASY_JOB, "-")
            except Exception:
                require(self.accounting_checked(EASY_JOB) == ("COMPLETED", "0:0"),
                        "The original Easy batch script is unavailable before completion")
                batch = receipt.get("easy_native_batch_script", "")
            require(str(tools / "native_budget_job.sh") in batch and str(self.evaluation) in batch
                    and str(self.easy_root) in batch and "bash " in batch,
                    "The adopted native Easy job did not execute its original fixed source")
            self.save("waiting_native_easy", adoption_sha256=checksum,
                      easy_native_batch_sha256=hashlib.sha256(batch.encode()).hexdigest())
        cfg = argparse.Namespace(**easy_state["configuration"])
        cfg.evaluation_root, cfg.output_root = self.evaluation, self.easy_root
        easy = controller(cfg)
        self.check_hard_hold()
        easy.run_native()  # Infrastructure retries retain the exact 157 source.
        easy.verify_report()
        total = json.loads((self.easy_root / "reports/summary.json").read_text())["overall"]["cumulative"]
        self.save("native_easy_complete", easy_native_jobs=easy.state["native_jobs"], easy_native_counts=total)
        self.receipt = receipt

    def deliver_original_extension(self, original, start):
        history = original.state["jobs"]["extension:easy"]
        intent = self.state.get("original_submission")
        if intent is not None and intent.get("known_history") != history:
            intent = None  # The previous delivery is already in the old state.
        fresh = intent is None
        if fresh:
            token = "rw-hard-easy-" + hashlib.sha256(f"{self.root}:{len(history)}".encode()).hexdigest()[:24]
            intent = {"token": token, "at_unix": time.time(), "known_history": list(history)}
            self.save("resuming_original_easy", original_submission=intent)
        tagged = self.tagged_jobs(intent["token"])
        journaled = original.journal_submission("extension", "easy", history)
        require(not tagged or not journaled or tagged == journaled, "Original Easy resume jobs disagree")
        if tagged and not journaled:
            self.repair_original_journals(tagged)
            journaled = tagged
        require(fresh or journaled is not None, "Original Easy resume delivery is uncertain; recover its token")
        previous = os.environ.get("FYP_ROBOTWIN_SUBMISSION_TOKEN")
        os.environ["FYP_ROBOTWIN_SUBMISSION_TOKEN"] = intent["token"]
        try:
            job = start("extension", "easy", original.cfg.hours, retry=True)
        finally:
            if previous is None:
                os.environ.pop("FYP_ROBOTWIN_SUBMISSION_TOKEN", None)
            else:
                os.environ["FYP_ROBOTWIN_SUBMISSION_TOKEN"] = previous
        self.save("waiting_original_easy", original_resume_job=job)
        return job

    def finish_original_easy(self):
        """The caller holds dispatch.lock until the replacement Hard run ends."""
        self.check_hard_hold()
        original_path = self.data / "dispatch" / EVALUATION / "state.json"
        original_before = json.loads(original_path.read_text())
        configuration = original_before["configuration"]
        cfg = argparse.Namespace(**configuration, evaluation_root=self.evaluation, recover_prescreen=True,
                                 easy_first=True, pilot_root=None, paused_easy_job=None, parallel_easy=True)
        original = Dispatch(cfg)
        original.accounting = self.accounting_checked
        if "original_state_at_adoption" not in self.state:
            self.save("resuming_original_easy", original_state_at_adoption=original_before)
        history = original.state["jobs"].get("extension:easy", [])
        require("20688" in history and PAUSED_EASY in history, "The original Easy extension history changed")
        original.require_easy("first_batch", 5000)
        if history[-1] == PAUSED_EASY:
            require(self.accounting_checked(PAUSED_EASY)[0] == "CANCELLED", "The paused Easy extension is still queued")
            job = self.deliver_original_extension(original, original.start)
        else:
            job = history[-1]
            accepted = {self.state.get("original_resume_job"), self.receipt.get("adopted_original_extension_job")}
            token = self.state.get("original_submission", {}).get("token")
            if token:
                accepted.add(self.tagged_jobs(token))
            require(job in accepted, "Receipt must identify the already restored original Easy extension")
        state = self.accounting_checked(job)
        require(state[0] not in DONE or state[0] in RESUMABLE or state == ("COMPLETED", "0:0"),
                f"Original Easy extension {job} failed before adoption")
        original.state.pop("error", None)
        original.save("extension", hard_native_controller=str(self.root))
        self.save("waiting_original_easy", original_resume_job=job)
        original_start = original.start

        def start(stage, lane, hours, retry=False):
            require((stage, lane) == ("extension", "easy"), "The adopted dispatcher may only finish original Easy")
            return (self.deliver_original_extension(original, original_start) if retry
                    else original_start(stage, lane, hours))

        original.start = start
        result = original.wait_for("extension", ("easy",), cfg.hours)
        require(result["easy"] == ("COMPLETED", "0:0"), f"Original Easy did not complete cleanly: {result}")
        original.require_easy("cumulative", 7000)
        job = original.state["jobs"]["extension:easy"][-1]
        self.wait_slot_release(job)
        original.save("easy_complete_hard_replaced", easy_confirmed_complete=True, hard_native_controller=str(self.root))
        self.save("original_easy_complete", original_easy_job=job, original_easy_finished=7000)

    def cancel_placeholder(self):
        require(self.state.get("original_easy_finished") == 7000, "Verify all original Easy episodes before canceling Hard")
        if self.state.get("placeholder_cancelled"):
            require(self.accounting_checked(HELD_HARD)[0] == "CANCELLED", "The archived Hard placeholder changed")
            return
        if not self.state.get("placeholder_cancel_intent"):
            self.check_hard_hold()
            self.save("canceling_old_hard", placeholder_cancel_intent={"job": HELD_HARD, "at_unix": time.time(),
                                                                      "fields": Dispatch.hard_job_fields(HELD_HARD)})
        if self.accounting_checked(HELD_HARD)[0] != "CANCELLED":
            self.check_hard_hold()
            output("scancel", "--state=PENDING", HELD_HARD)
        deadline = time.monotonic() + 300
        while self.accounting_checked(HELD_HARD)[0] != "CANCELLED":
            require(time.monotonic() < deadline, "The held old Hard job did not confirm cancellation")
            time.sleep(5)
        self.wait_slot_release(HELD_HARD)
        self.save("old_hard_replaced", placeholder_cancelled=HELD_HARD)

    def start_native(self):
        require(self.state.get("placeholder_cancelled") == HELD_HARD
                and self.state.get("original_easy_finished") == 7000, "Finish Easy and replace its held Hard placeholder first")
        intent = self.state.get("native_submission")
        if intent:
            job = self.tagged_jobs(intent["token"])
            if job is None:
                rows = self.journal_path.read_text().splitlines() if self.journal_path.exists() else []
                for line in rows:
                    values = line.split("\t")
                    if values[1:] == [EVALUATION, self.code, str(self.root), intent["token"]]:
                        require(job in (None, values[0]), "Duplicate Hard submission journal token")
                        job = values[0]
            require(job is not None, "Hard submission may have reached Slurm; recover its token before restarting")
        else:
            token = "rw-hard-native-" + hashlib.sha256(
                f"{self.root}:{self.code}:{len(self.state['native_jobs'])}".encode()).hexdigest()[:24]
            with (self.data / "submit.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                self.check_gpu_slot()
                intent = {"token": token, "attempt": len(self.state["native_jobs"]), "at_unix": time.time()}
                self.save("submitting", native_submission=intent)
                logs = self.data / "logs"
                logs.mkdir(exist_ok=True)
                command = shlex.join(["bash", str(HERE / "hard_native_job.sh"), str(self.evaluation), str(self.root)])
                launch = output("sbatch", "--parsable", f"--partition={self.cfg.partition}", "--exclude=4090node1",
                                "--nodes=1", "--ntasks=1", "--cpus-per-task=16", "--gres=gpu:4090:1",
                                f"--time={self.cfg.hours * 60}", "--job-name=rw-hard-native-budget",
                                f"--comment={token}", f"--output={logs}/%j.out", f"--error={logs}/%j.err",
                                "--export=ALL", f"--wrap={command}")
                job = launch.partition(";")[0]
                require(job.isdigit(), f"Unexpected Hard sbatch response: {launch}")
        self.append_native_job(job, intent["token"])
        require(job not in self.state["native_jobs"], "Hard job was already recorded")
        self.state["native_jobs"].append(job)
        self.save("native_running", native_submission=None, pending_since=time.time(), native_job=job)
        return job

    def verify_report(self):
        import subprocess

        target = self.root / "reports"
        subprocess.run([sys.executable, str(HERE / "hard_native_report.py"), "--input", str(self.root),
                        "--source-panel", str(self.data / "panels" / EVALUATION), "--output", str(target)], check=True)
        report = json.loads((target / "summary.json").read_text())
        total = report["overall"]["cumulative"]
        require(report["status"] == "complete" and total["planned"] == total["finished"] == 7000
                and total["unresolved"] == total["retry_pending"] == 0, f"Hard native-budget panel is incomplete: {total}")
        self.save("complete", report=str(target / "summary.json"), counts=total)

    def run(self):
        if self.state.get("stage") == "complete":
            return
        if not self.state.get("original_easy_finished"):
            self.adopt_easy_native()
            self.finish_original_easy()
        self.cancel_placeholder()
        self.run_native()
        self.verify_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--easy-native-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-job", default="20298")
    parser.add_argument("--partition", default="GPU")
    parser.add_argument("--gpu", choices=("4090",), default="4090")
    parser.add_argument("--hours", type=int, choices=range(1, 73), default=48)
    cfg = parser.parse_args()
    data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
    with ExitStack() as stack:
        for name in ("hard-native-dispatch.lock", "native-budget-dispatch.lock", "dispatch.lock"):
            lock = stack.enter_context((data / name).open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        dispatch = HardNativeDispatch(cfg)
        try:
            dispatch.run()
        except Exception as error:
            dispatch.save("attention", error=str(error))
            raise


if __name__ == "__main__":
    main()
