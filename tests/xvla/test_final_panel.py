import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from tools.xvla import final_panel
from xvla_rlinf import benchmark


class FinalPanelTests(unittest.TestCase):
    def test_frozen_panel_repeats_each_state_with_two_distinct_seed_pairs(self):
        repo = Path(__file__).resolve().parents[2]
        split = final_panel.read(repo / final_panel.PROTOCOL / "split.json")
        shards = final_panel.trial_panels(split)
        self.assertEqual([len(shard) for shard in shards], [50] * 5)
        self.assertEqual([shard[0]["task_id"] for shard in shards], [5, 6, 7, 8, 9])
        for shard in shards:
            self.assertEqual([row["trial_id"] for row in shard[::2]], list(range(5, 30)))
            for first, second in zip(shard[::2], shard[1::2], strict=True):
                self.assertEqual(first["initial_state_sha256"], second["initial_state_sha256"])
                self.assertEqual((first["repetition_index"], second["repetition_index"]), (0, 1))
                self.assertEqual(second["environment_seed"] - first["environment_seed"], 1000)
                self.assertEqual(second["policy_seed"] - first["policy_seed"], 1000)

    def test_source_a_uses_audited_candidates_and_fixed_seed_repetitions(self):
        repo = Path(__file__).resolve().parents[2]
        split_path = repo / final_panel.PROTOCOL / "split.json"
        split = final_panel.read(split_path)
        split_sha = final_panel.digest(split_path)
        audit = {
            "status": "PASS", "audit_scope": "libero_spatial_task0_source",
            "split_sha256": split_sha, "demo_manifest_sha256": "manifest",
            "role_sizes": {"source_final_test_sealed": 10},
            "state_hashes_match_split": True, "policy_env_steps": 0, "sealed_role_overlaps": [],
            "success_during_reset_ids": {"source_final_test_sealed": []},
        }
        panel = final_panel.source_a_panel(split, audit, split_sha, "manifest")
        self.assertEqual(len(panel), 50)
        self.assertEqual({row["task_id"] for row in panel}, {0})
        self.assertEqual([row["trial_id"] for row in panel[::5]], list(range(40, 50)))
        self.assertEqual([row["environment_seed"] for row in panel[:5]], [182, 1182, 2182, 3182, 4182])
        self.assertEqual([row["policy_seed"] for row in panel[:5]], [10040, 11040, 12040, 13040, 14040])
        audit["success_during_reset_ids"]["source_final_test_sealed"] = [40]
        with self.assertRaisesRegex(ValueError, "successful E1 layout audit"):
            final_panel.source_a_panel(split, audit, split_sha, "manifest")

    def test_refuses_release_before_formal_matrix_completes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / final_panel.RUN_PROTOCOL
            folder.mkdir()
            final_panel.write(folder / "e3_dispatch.json", {
                "state": "RUNNING", "partition": "GPU", "runs": final_panel.schedule(),
            })
            with self.assertRaisesRegex(ValueError, "H remains sealed"):
                final_panel.build_plan(root, root, "GPU", "commit")

    def test_evaluate_keeps_source_a_out_of_h_journal(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan = {
                "evaluation_commit": "commit",
                "policies": [{"model": "sft-fewshot", "checkpoint": "/actor", "checkpoint_sha256": "digest"}],
                "source_a_shard": [{"task_id": 0}],
                "shards": [[{"task_id": task}] for task in final_panel.H_TASKS],
            }
            final_panel.write(plan_path, plan)

            def fake_benchmark(argv, *, trial_plan):
                output = Path(os.environ["XVLA_RUN_DIR"])
                final_panel.write(output / "benchmark_summary.json", {
                    "status": "PASS", "completed_episodes": 1, "successes": 1,
                    "trial_plan": trial_plan, "job_id": "42",
                })
                (output / "benchmark_episodes.jsonl").write_text(
                    str(trial_plan[0]["task_id"]) + "\n", encoding="utf-8",
                )

            with (patch.dict(os.environ, {"XVLA_RUN_DIR": str(root), "SLURM_JOB_ID": "42"}),
                  patch.object(final_panel, "fixed_commit", return_value="commit"),
                  patch.object(benchmark, "main", side_effect=fake_benchmark)):
                final_panel.evaluate(plan_path, final_panel.digest(plan_path), 0)
                self.assertEqual(os.environ["XVLA_RUN_DIR"], str(root))
            self.assertEqual((root / "final_h_episodes.jsonl").read_text().splitlines(), list("56789"))
            self.assertEqual((root / "task_0/benchmark_episodes.jsonl").read_text().splitlines(), ["0"])
            receipt = final_panel.read(root / "final_h_summary.json")
            self.assertEqual(receipt["source_a"]["episodes"], 1)
            self.assertEqual(receipt["source_a"]["episodes_sha256"],
                             final_panel.digest(root / "task_0/benchmark_episodes.jsonl"))
