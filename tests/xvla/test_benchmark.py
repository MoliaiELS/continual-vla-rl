"""Paired benchmark identities and scoring, without running simulation."""

import unittest
from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
from unittest.mock import patch

from xvla_rlinf.benchmark import _checkpoint_layout, _locked_trial_plan, _validate_reset_states, build_trials, main, summarize


def completed(trial, *, success, steps):
    return {
        **trial,
        "success": success,
        "policy_env_steps": steps,
        "settling_env_steps": 10,
    }


class BenchmarkAccountingTests(unittest.TestCase):
    def test_full_panel_has_stable_trial_ids_and_seeds_across_shards(self):
        full = build_trials(0, 10, 5)
        self.assertEqual(len(full), 50)
        self.assertEqual(full, build_trials(0, 4, 5) + build_trials(4, 6, 5))
        self.assertEqual(
            {(row["task_id"], row["trial_id"]) for row in full},
            {(task, trial) for task in range(10) for trial in range(5)},
        )
        self.assertEqual(len({row["policy_seed"] for row in full}), 50)

    def test_trial_offset_preserves_the_initial_state_ids_and_seeds(self):
        complete = build_trials(0, 1, 30)
        extra = build_trials(0, 1, 20, start_trial=10)
        self.assertEqual(extra, complete[10:])

    def test_reports_per_task_successes_and_executed_steps(self):
        plan = build_trials(2, 2, 2)
        episodes = [
            completed(row, success=i == 0, steps=73 if i == 0 else 800)
            for i, row in enumerate(plan)
        ]
        summary = summarize(episodes, plan)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["planned_episodes"], 4)
        self.assertEqual(summary["completed_episodes"], 4)
        self.assertEqual(summary["per_task"]["2"]["successes"], 1)
        self.assertEqual(summary["per_task"]["3"]["successes"], 0)
        self.assertEqual(summary["macro_success_rate"], 0.25)
        self.assertEqual(summary["policy_env_steps"], 2473)
        self.assertEqual(summary["settling_env_steps"], 40)

    def test_rejects_incomplete_duplicate_and_mismatched_trials(self):
        plan = build_trials(0, 1, 2)
        rows = [completed(row, success=False, steps=800) for row in plan]
        wrong_seed = {**rows[1], "policy_seed": rows[1]["policy_seed"] + 1}
        for episodes in (rows[:1], [rows[0], rows[0]], [rows[0], wrong_seed]):
            with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                summarize(episodes, plan)

    def test_rejects_out_of_bounds_plans(self):
        for args in ((-1, 1, 1), (9, 2, 1), (0, 10, 6), (0, 1, 0)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                build_trials(*args)
        for start_trial in (-1, True, 1.5):
            with self.subTest(start_trial=start_trial), self.assertRaises(ValueError):
                build_trials(0, 1, 20, start_trial=start_trial)


class LockedTrialPlanTests(unittest.TestCase):
    def test_repeats_the_frozen_state_with_distinct_episode_seeds(self):
        first = {"task_id": 0, "trial_id": 5, "environment_seed": 147, "policy_seed": 10005,
                 "initial_state_sha256": "b" * 64, "repetition_index": 0}
        second = {**first, "environment_seed": 1147, "policy_seed": 11005, "repetition_index": 1}
        plan = _locked_trial_plan([first, second])
        self.assertEqual(plan, [first, second])
        episodes = [completed(row, success=True, steps=120) for row in plan]
        self.assertEqual(summarize(episodes, plan)["successes"], 2)
        for bad_plan in ([first, first], [{**first, "initial_state_sha256": ""}], [first] * 51):
            with self.subTest(bad_plan=bad_plan), self.assertRaises(ValueError):
                _locked_trial_plan(bad_plan)

    def test_detects_mismatch_in_runtime_reset_archive(self):
        row = {"task_id": 0, "trial_id": 0, "initial_state_sha256": "a" * 64}
        with patch("xvla_rlinf.benchmark._initial_state_sha256", return_value="b" * 64) as digest:
            with self.assertRaisesRegex(ValueError, "Runtime initial-state SHA256 differs"):
                _validate_reset_states([row], [["reset-state"]])
            digest.assert_called_once_with("reset-state")

    def test_rejects_shard_flags_with_locked_plan(self):
        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaises(SystemExit):
            main(["--model", "pt", "--start-trial", "0"], trial_plan=[])
        self.assertIn("cannot be combined with benchmark shard flags", stderr.getvalue())


class BenchmarkCheckpointLayoutTests(unittest.TestCase):
    def setUp(self):
        self.pt_path = (Path.cwd() / "X-VLA-Pt").resolve()
        self.config = {
            "model_path": str(self.pt_path), "add_value_head": False, "num_action_chunks": 30,
            "xvla": {"update_scope": "prompt_heads", "domain_rows": [3], "domain_id": 3, "steps": 10},
        }

    def test_preserves_prompt_heads(self):
        self.assertEqual(_checkpoint_layout(self.config, self.pt_path, is_rl=False),
                         {"update_scope": "prompt_heads"})

    def test_uses_saved_lora_rank_without_copying_training_settings(self):
        self.config["xvla"].update(update_scope="lora_heads", lora_rank=16,
                                   eval_sampler="terminal_clean_gaussian_v1")
        self.assertEqual(_checkpoint_layout(self.config, self.pt_path, is_rl=False),
                         {"update_scope": "lora_heads", "lora_rank": 16})

    def test_rejects_invalid_lora_rank_and_nonmatching_domain(self):
        self.config["xvla"]["update_scope"] = "lora_heads"
        for rank in (0, True, "8"):
            self.config["xvla"]["lora_rank"] = rank
            with self.subTest(rank=rank), self.assertRaises(ValueError):
                _checkpoint_layout(self.config, self.pt_path, is_rl=False)
        self.config["xvla"].update(lora_rank=8, domain_rows=[2, 3])
        with self.assertRaises(ValueError):
            _checkpoint_layout(self.config, self.pt_path, is_rl=False)

    def test_accepts_rank8_rl_layout_with_or_without_value_head(self):
        self.config["action_dim"] = 7
        self.config["xvla"].update(update_scope="lora_heads", lora_rank=8,
                                   initial_sft_checkpoint="/runs/source-sft/actor")
        for add_value_head in (True, False):
            with self.subTest(add_value_head=add_value_head):
                self.config["add_value_head"] = add_value_head
                self.assertEqual(_checkpoint_layout(self.config, self.pt_path, is_rl=True),
                                 {"update_scope": "lora_heads", "lora_rank": 8})
        self.config["add_value_head"] = True
        with self.assertRaises(ValueError):
            _checkpoint_layout(self.config, self.pt_path, is_rl=False)

    def test_rl_layout_rejects_uninitialized_or_wrong_rank_export(self):
        self.config["action_dim"] = 7
        self.config["xvla"].update(update_scope="lora_heads", lora_rank=8)
        with self.assertRaises(ValueError):
            _checkpoint_layout(self.config, self.pt_path, is_rl=True)
        self.config["xvla"].update(lora_rank=16, initial_sft_checkpoint="/runs/source-sft/actor")
        with self.assertRaises(ValueError):
            _checkpoint_layout(self.config, self.pt_path, is_rl=True)
