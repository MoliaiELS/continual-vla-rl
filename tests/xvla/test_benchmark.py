"""Paired benchmark identities and scoring, without running simulation."""

import unittest

from xvla_rlinf.benchmark import build_trials, summarize


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
