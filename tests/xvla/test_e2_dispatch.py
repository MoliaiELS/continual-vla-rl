import hashlib
import json
import sys
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from tools.xvla import e2_dispatch


class E2DispatchTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        self.repo, self.data = root / "repo", root / "data"
        self.args = SimpleNamespace(repo=self.repo, data=self.data, partition="GPU")
        self.runs = self.data / "runs"
        self.state_path = self.data / "protocol_20261110/e2_dispatch.json"
        self.start = ["start", "--repo", str(self.repo), "--data", str(self.data), "--partition", "GPU",
                      "--deployed-job-id", "101", "--online-job-id", "102", "--physical-job-id", "103"]

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def record_e1(self, online_successes=1):
        for job, mode, first, count, horizon, successes in (
            ("101", "deployed", 10, 20, 800, 16),
            ("102", "online", 30, 10, 300, online_successes),
        ):
            self.write(self.runs / job / "benchmark_summary.json", {
                "status": "PASS", "complete": True, "job_id": job, "suite": "libero_spatial",
                "completed_episodes": count, "successes": successes,
                "protocol": {"policy_path": mode, "horizon": horizon},
                "checkpoint": {"checkpoint_state_sha256": e2_dispatch.B0_DIGEST},
                "trial_plan": [{"task_id": 0, "trial_id": i} for i in range(first, first + count)],
            })
        split, manifest = "split.json", "manifest.json"
        split_digest = self.write(self.repo / split, {})
        manifest_digest = self.write(self.data / manifest, {})
        self.write(self.repo / "configs/xvla/protocol_20261110/registry.json", {
            "split_path": split, "split_sha256": split_digest,
            "common_start": {"manifest": manifest, "manifest_sha256": manifest_digest},
        })
        self.write(self.runs / "103/e1_source_audit.json", {
            "status": "PASS", "job_id": "103", "audit_scope": "libero_spatial_task0_source",
            "split_sha256": split_digest, "demo_manifest_sha256": manifest_digest,
        })

    def submission(self, pool, job):
        plan_name = "expanded" if pool == "stratified" else pool
        plan = self.repo / "configs/xvla/source_pools" / f"{plan_name}.json"
        digest = self.write(plan, {"name": plan_name})
        checkpoint = self.data / "runs/19525/sft_source10/checkpoints/global_step_1024/actor"
        receipt = {
            "mode": "ppo_e2", "requested_gpu_model": "4090", "requested_gpus": 1, "job_id": job,
            "initial_sft": {"checkpoint_job_id": "19525", "sft_steps": 1024},
            "command": ["sbatch", str(self.repo / "tools/xvla/job.sh"), str(self.repo),
                        "ppo_e2", "1", str(checkpoint), str(plan), digest,
                        "stratified" if pool == "stratified" else "uniform", pool, "GPU"],
        }
        return receipt

    def run_dispatch(self, argv):
        # flock is Linux-only. All jobs and their output paths remain local fixtures.
        lock = SimpleNamespace(flock=Mock(), LOCK_EX=1)
        with (patch.dict(sys.modules, {"fcntl": lock}),
              patch.object(e2_dispatch, "fixed_commit", return_value="fixed"),
              patch("builtins.print")):
            e2_dispatch.main(argv)

    def test_e1_requires_successful_accounting_and_frozen_physical_input(self):
        self.record_e1()
        with patch.object(e2_dispatch.subprocess, "check_output", return_value=(
            "101|COMPLETED|0:0\n102|COMPLETED|0:0\n103|FAILED|1:0\n"
        )):
            with self.assertRaisesRegex(RuntimeError, "not all completed"):
                e2_dispatch.e1_evidence(self.repo, self.data, "101", "102", "103")

        with patch.object(e2_dispatch.subprocess, "check_output", return_value=(
            "101|COMPLETED|0:0\n102|COMPLETED|0:0\n103|COMPLETED|0:0\n"
        )):
            evidence = e2_dispatch.e1_evidence(self.repo, self.data, "101", "102", "103")
            self.assertEqual(evidence["online"]["successes"], 1)
            self.assertEqual(evidence["physical"]["job_id"], "103")
            self.write(self.data / "manifest.json", {"changed": True})
            with self.assertRaisesRegex(ValueError, "frozen split and B0 manifest"):
                e2_dispatch.e1_evidence(self.repo, self.data, "101", "102", "103")

    def test_zero_e1_signal_stops_without_submitting(self):
        with (patch.object(e2_dispatch, "e1_evidence", return_value={"online": {"successes": 0}}),
              patch.object(e2_dispatch.subprocess, "run") as submit):
            self.run_dispatch(self.start)
            self.run_dispatch(self.start)
        submit.assert_not_called()
        self.assertEqual(e2_dispatch.read(self.state_path)["state"], "COMPLETE_NO_SIGNAL")

    def test_no_signal_endpoint_can_advance_and_retry_safe_resource_block(self):
        first, second = self.submission("reference", "201"), self.submission("expanded", "202")
        results = [
            CompletedProcess([], 0, json.dumps(first), ""),
            CompletedProcess([], 1, "", "BLOCKED_GPU_BUDGET: no job submitted"),
            CompletedProcess([], 0, json.dumps(second), ""),
        ]
        advance = ["advance", *self.start[1:7], "--job-id", "201", "--pool", "reference"]
        endpoint = {"job_id": "201", "state": "COMPLETE_NO_SIGNAL", "optimizer_updates": 0}
        with (patch.object(e2_dispatch, "e1_evidence", return_value={"online": {"successes": 1}}),
              patch.object(e2_dispatch, "endpoint", return_value=endpoint),
              patch.object(e2_dispatch.subprocess, "run", side_effect=results) as submit):
            self.run_dispatch(self.start)
            self.run_dispatch(self.start)
            self.assertEqual(submit.call_count, 1)
            with self.assertRaises(e2_dispatch.BlockedResource):
                self.run_dispatch(advance)
            state = e2_dispatch.read(self.state_path)
            self.assertEqual(state["pools"]["reference"], endpoint)
            self.assertEqual(state["pools"]["expanded"]["state"], "BLOCKED_RESOURCE")
            self.run_dispatch(self.start)
            self.run_dispatch(advance)
            self.assertEqual(submit.call_count, 3)
        state = e2_dispatch.read(self.state_path)
        self.assertEqual(state["pools"]["expanded"], {"job_id": "202", "state": "SUBMITTED"})
        self.assertNotIn("stratified", state["pools"])

    def test_submission_receipt_recovers_interrupted_recording(self):
        receipt = self.submission("reference", "201")
        state = {"pools": {"reference": {"state": "SUBMITTING", "prior_receipts": []}}}
        self.write(self.state_path, state)
        with patch.object(e2_dispatch.subprocess, "run") as submit:
            with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
                e2_dispatch.dispatch_next(self.args, state, self.state_path)
            self.assertEqual(state["pools"]["reference"]["state"], "SUBMISSION_UNCERTAIN")
            self.write(self.runs / "submission-201.json", receipt)
            e2_dispatch.dispatch_next(self.args, state, self.state_path)
        submit.assert_not_called()
        self.assertEqual(state["pools"]["reference"], {"state": "SUBMITTED", "job_id": "201"})

    def test_endpoint_requires_budget_and_independent_checkpoint(self):
        receipt = self.submission("reference", "201")
        self.write(self.runs / "submission-201.json", receipt)
        output = self.runs / "201"
        budget_path = output / "control_step_budget.json"
        checkpoint = output / "checkpoints/global_step_74/actor"
        budget = {
            "state": "COMPLETE_NO_SIGNAL", "requested_policy_env_steps": 25000,
            "policy_env_steps": 25300, "optimizer_updates": 0, "max_rollout_batch_control_steps": 1200,
            "overshoot_policy_env_steps": 300, "partial_rollout": False,
            "reset_plan_sha256": receipt["command"][-4], "sampling_mode": "uniform",
            "checkpoint_path": str(checkpoint),
        }
        self.write(budget_path, budget)
        check_path = checkpoint / "checkpoint_check.json"
        check = {
            "status": "PASS", "strict_state_load": True, "identity": {"slurm_job_id": "201"},
            "checkpoint_dir": str(checkpoint), "digest": "digest", "exported_digest": "digest",
            "fixed_probe": {"status": "PASS"},
        }
        self.write(check_path, check)
        self.args.job_id, self.args.pool = "201", "reference"
        self.assertEqual(e2_dispatch.endpoint(self.args)["state"], "COMPLETE_NO_SIGNAL")
        self.write(check_path, check | {"fixed_probe": {"status": "FAIL"}})
        with self.assertRaisesRegex(RuntimeError, "independent checkpoint"):
            e2_dispatch.endpoint(self.args)
        self.write(budget_path, budget | {"policy_env_steps": 26200, "overshoot_policy_env_steps": 1200})
        with self.assertRaisesRegex(RuntimeError, "control-step budget"):
            e2_dispatch.endpoint(self.args)
