"""Check the paired deployment evaluator without running simulation."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.xvla import e3_dispatch, protocol_dev


class ProtocolDevelopmentTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        self.repo = root / "worktrees" / ("a" * 40)
        self.data = root / "data"
        self.runs = self.data / "runs"
        self.run_dir = self.runs / "300"
        self.run_dir.mkdir(parents=True)
        self.panel = [f"{trial:064x}" for trial in range(30)]
        self.b0 = protocol_dev.e2_dispatch.B0_DIGEST

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return protocol_dev._sha256(path)

    def shard(self, folder, job, digest, first, count):
        plan = [{"task_id": 0, "trial_id": trial, "environment_seed": 142 + trial,
                 "policy_seed": 10000 + trial} for trial in range(first, first + count)]
        rows = [
            {**row, "policy_path": "deployed", "initial_state_sha256": self.panel[row["trial_id"]],
             "success": row["trial_id"] % 2 == 0, "policy_env_steps": 100, "settling_env_steps": 10}
            for row in plan
        ]
        digest = self.write(folder / "benchmark_summary.json", {
            "job_id": job, "suite": "libero_spatial", "status": "PASS", "complete": True,
            "completed_episodes": count, "successes": sum(row["success"] for row in rows),
            "protocol": {"policy_path": "deployed", "history": "measured_state", "horizon": 800,
                         "terminal_noise": "off"},
            "checkpoint": {"checkpoint_state_sha256": digest}, "trial_plan": plan,
        })
        (folder / "benchmark_episodes.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return digest

    def register(self):
        split_name = "split.json"
        split_digest = self.write(self.repo / split_name, {
            "protocol_id": "xvla-20261110", "suite": "libero_spatial",
            "common_evaluation": {"source_dev_task_id": 0, "source_dev_trial_ids": list(range(30))},
            "tasks": [{"task_id": 0, "state_sha256_by_trial_id": self.panel}],
        })
        checkpoint = self.runs / "19525/actor"
        self.write(checkpoint / "checkpoint_check.json", {
            "status": "PASS", "strict_state_load": True, "identity": {"slurm_job_id": "19525"},
            "checkpoint_dir": str(checkpoint), "digest": self.b0, "exported_digest": self.b0,
        })
        self.write(self.repo / "configs/xvla/protocol_20261110/registry.json", {
            "split_path": split_name, "split_sha256": split_digest,
            "common_start": {"job_id": 19525, "actor_checkpoint": "runs/19525/actor", "actor_state_sha256": self.b0},
        })
        self.endpoints = {
            pool: {"job_id": str(201 + i), "state": "COMPLETE_VALID", "policy_env_steps": 25100,
                   "optimizer_updates": 80, "checkpoint_sha256": f"{i + 1:064x}",
                   "checkpoint_check_path": str(self.runs / str(201 + i) / "actor/checkpoint_check.json")}
            for i, pool in enumerate(protocol_dev.e2_dispatch.POOLS)
        }
        self.dispatch_path = self.data / "protocol_20261110/e2_dispatch.json"
        self.state = {
            "state": "COMPLETE", "partition": "GPU", "commit": "b" * 40, "pools": self.endpoints,
            "e1": {"deployed": {"job_id": "19569",
                                "summary_sha256": self.shard(self.runs / "19569", "19569", self.b0, 10, 20)}},
        }
        self.write(self.dispatch_path, self.state)

    def plan(self):
        with (patch.object(protocol_dev.e2_dispatch, "endpoint", side_effect=lambda args: self.endpoints[args.pool]),
              patch.object(protocol_dev, "_check_completed_jobs") as completed):
            plan = protocol_dev.development_plan(self.repo, self.data, "GPU")
            completed.assert_called_once_with(["19525", "201", "202", "203", "19569"])
            return plan

    def test_runs_four_deployed_shards_and_combines_b0(self):
        self.register()
        plan = self.plan()

        def launch(command, *, env, check):
            folder = Path(env["XVLA_RUN_DIR"])
            self.assertEqual(command[command.index("--policy-path") + 1], "deployed")
            count = int(command[command.index("--trials") + 1])
            self.assertEqual(count, 10 if folder.name == "B0" else 30)
            digest = command[command.index("--expected-checkpoint-sha256") + 1]
            self.assertEqual(digest, plan["conditions"][folder.name]["checkpoint_state_sha256"])
            self.shard(folder, "300", digest, 0, count)

        with patch.object(protocol_dev.subprocess, "run", side_effect=launch) as mocked, patch("builtins.print"):
            protocol_dev.evaluate(plan, self.run_dir, "300")
        self.assertEqual(mocked.call_count, 4)
        result = protocol_dev.e2_dispatch.read(self.run_dir / "protocol_dev_summary.json")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["conditions"]["B0"]["eval_job_ids"], ["300", "19569"])
        self.assertEqual(result["conditions"]["M"]["checkpoint_job_id"], "203")
        for condition in result["conditions"].values():
            self.assertEqual((condition["successes"], condition["episodes"]), (15, 30))

    def test_rejects_changed_legacy_shard_and_incomplete_dispatch(self):
        self.register()
        plan = self.plan()
        path = self.runs / "19569/benchmark_episodes.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with patch.object(protocol_dev.subprocess, "run") as mocked:
            with self.assertRaisesRegex(ValueError, "shard has changed"):
                protocol_dev.evaluate(plan, self.run_dir, "300")
        mocked.assert_not_called()
        self.state["state"] = "RUNNING"
        self.write(self.dispatch_path, self.state)
        with self.assertRaisesRegex(ValueError, "complete registered E2 chain"):
            protocol_dev.development_plan(self.repo, self.data, "GPU")

    def test_e3_accepts_complete_paired_evidence_and_rejects_a_swapped_endpoint(self):
        self.register()
        plan = self.plan()
        registry_path = self.repo / "configs/xvla/protocol_20261110/registry.json"
        registry = e3_dispatch.read(registry_path)
        registry["protocol_id"] = plan["protocol_id"]
        registry["common_start"].update(
            manifest="manifest.json",
            manifest_sha256=self.write(self.data / "manifest.json", {}),
            strict_reload_evidence="runs/19525/actor/checkpoint_check.json",
        )
        self.write(registry_path, registry)
        conditions = {
            key: {**row, "episodes": 30, "successes": 20} for key, row in plan["conditions"].items()
        }
        report = {**plan, "job_id": "300", "status": "PASS", "complete": True, "conditions": conditions}
        report_path = self.run_dir / "protocol_dev_summary.json"
        summary_sha256 = self.write(report_path, report)
        with patch.object(e3_dispatch.subprocess, "check_output", return_value="300|COMPLETED|0:0\n"):
            self.assertEqual(e3_dispatch.prerequisites(self.repo, self.data, "300")["summary_sha256"],
                             summary_sha256)
            conditions["M"]["checkpoint_state_sha256"] = conditions["B2"]["checkpoint_state_sha256"]
            self.write(report_path, report)
            with self.assertRaisesRegex(ValueError, "frozen M checkpoint"):
                e3_dispatch.prerequisites(self.repo, self.data, "300")

    def test_e3_schedules_all_conditions_with_paired_seeds_and_distinct_sampling(self):
        schedule = e3_dispatch.schedule()
        for condition in ("SFT-cont", "B1", "B2", "M"):
            self.assertEqual([row["seed"] for row in schedule if row["condition"] == condition],
                             [101, 202, 303])
        pool_dir = self.repo / "configs/xvla/source_pools"
        self.write(pool_dir / "expanded.json", {"pool": "expanded"})
        b2, mixture = (next(row for row in schedule if row["condition"] == condition)
                       for condition in ("B2", "M"))
        _, b2_args, walltime = e3_dispatch.submission(self.repo, self.data, "GPU", b2)
        _, m_args, _ = e3_dispatch.submission(self.repo, self.data, "GPU", mixture)
        self.assertEqual(b2_args[:3], m_args[:3])
        self.assertEqual((b2_args[3], m_args[3], walltime), ("uniform", "stratified", "24:00:00"))

    def test_e3_sft_strict_reload_and_paired_rollouts_cover_checkpoint_without_ppo_probe(self):
        checkpoint = self.run_dir / "actor"
        self.write(checkpoint / "checkpoint_check.json", {
            "status": "PASS", "strict_state_load": True, "identity": {"slurm_job_id": "300"},
            "digest": self.b0, "exported_digest": self.b0, "fixed_probe": {"status": "NOT RUN"},
        })
        self.shard(self.run_dir / "source_dev/global_step_512", "300", self.b0, 0, 30)
        arguments = self.run_dir, "300", "global_step_512", checkpoint, self.b0, self.panel
        self.assertEqual(e3_dispatch.verify_benchmark(*arguments, require_fixed_probe=False)["successes"], 15)
        with self.assertRaisesRegex(ValueError, "Independent checkpoint check failed"):
            e3_dispatch.verify_benchmark(*arguments, require_fixed_probe=True)
