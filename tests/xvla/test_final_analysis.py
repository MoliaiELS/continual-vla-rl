"""Synthetic coverage for the frozen paired H analysis."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.xvla import final_analysis as analysis


def _rows(condition, seed):
    rows = []
    for key, (digest, repetition) in analysis._expected_panel(analysis.DEFAULT_SPLIT).items():
        task, trial, _, _ = key
        success = trial < 15 and repetition == 0
        if condition == "M" and task == 5:
            if seed == 101 and 15 <= trial < 20:
                success = True
            if seed == 202:
                success = False
        rows.append({
            **dict(zip(analysis.IDENTITY, key, strict=True)), "initial_state_sha256": digest,
            "repetition_index": repetition, "success": success,
            "policy_env_steps": 60 if success else 800, "policy_path": "deployed",
        })
    return rows


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class FinalAnalysisTests(unittest.TestCase):
    def test_shared_state_resampling_and_training_seed_spread(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            policies = []
            for condition in analysis.CONDITIONS:
                seeds = [None] if condition == "B0" else analysis.TRAINING_SEEDS
                for seed in seeds:
                    rows = _rows(condition, seed)
                    journal = root / f"{condition}-{seed}.jsonl"
                    _write_rows(journal, rows)
                    policies.append({"condition": condition, "seed": seed, "episodes": journal.name})
            b0 = policies[0]
            b0_rows = _rows("B0", None)
            second = root / "B0-2.jsonl"
            _write_rows(root / b0["episodes"], b0_rows[:100])
            _write_rows(second, list(reversed(b0_rows[100:])))
            b0["episodes"] = [b0["episodes"], str(second)]
            manifest = root / "input.json"
            manifest.write_text(json.dumps({"policies": policies}), encoding="utf-8")
            result = analysis.analyze(manifest, replicates=300)

        self.assertEqual(len(result["policies"]), 13)
        self.assertEqual(result["policies"][0]["episodes"], 250)
        self.assertEqual(result["policies"][0]["unique_task_initial_state_pairs"], 125)
        self.assertEqual(result["policies"][0]["mean_first_success_steps"], 60)
        self.assertEqual(result["policies"][0]["mean_penalized_first_success_steps"], 652)
        self.assertIsNone(result["conditions"]["B0"]["training_seed_variability"])
        self.assertEqual(result["contrasts"]["B1-B0"]["paired_state_cluster_ci95"], [0.0, 0.0])
        contrast = result["contrasts"]["M-B2"]
        self.assertEqual(contrast["role"], "primary")
        self.assertAlmostEqual(contrast["macro_success_rate_difference"], 0)
        self.assertAlmostEqual(contrast["training_seed_variability"]["sample_sd"], 0.04)
        self.assertEqual(contrast["by_training_seed"][0]["per_task"]["5"]["paired_new_successes"], 10)
        self.assertEqual(contrast["by_training_seed"][1]["per_task"]["5"]["paired_new_failures"], 10)
        lower, upper = contrast["paired_state_cluster_ci95"]
        self.assertLess(lower, 0)
        self.assertGreater(upper, 0)
        json.dumps(result, allow_nan=False)

    def test_rejects_wrong_hash_missing_repetition_and_duplicate(self):
        expected = analysis._expected_panel(analysis.DEFAULT_SPLIT)
        rows = _rows("B0", None)
        invalid = (
            ([{**rows[0], "initial_state_sha256": "f" * 64}, *rows[1:]], "frozen deployment panel"),
            (rows[1:], "249 of 250"),
            ([rows[0], *rows], "repeated episode identity"),
            ([{**rows[0], "first_success_step": 59}, *rows[1:]], "first-success time"),
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            journal = root / "episodes.jsonl"
            for records, error in invalid:
                with self.subTest(error=error):
                    _write_rows(journal, records)
                    with self.assertRaisesRegex(ValueError, error):
                        analysis._read_policy({"episodes": journal.name}, root, expected)
