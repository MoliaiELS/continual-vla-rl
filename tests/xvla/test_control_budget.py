import json
import math
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from xvla_rlinf.control_budget import (
    FORMAL_MILESTONES,
    control_budget_actor,
    control_budget_rollout,
    control_budget_runner,
    read_budget_ledger,
    validate_control_budget,
)


PLAN = "a" * 64


def config(path, budget=10, cap=4, seed=None, milestones=None, horizon=5):
    runner = NS(
        xvla_control_step_budget=budget, max_steps=cap, max_epochs=cap,
        xvla_training_seed=seed, xvla_control_step_milestones=milestones,
        save_interval=-1, val_check_interval=-1, logger=NS(log_path=path, experiment_name="pilot"),
    )
    runner.get = lambda key: getattr(runner, key, None)
    return NS(
        runner=runner,
        env=NS(train=NS(
            total_num_envs=2, max_episode_steps=horizon, rollout_epoch=1,
            xvla_source_reset_pool=NS(sha256=PLAN, mode="uniform"),
            get=lambda key: {} if key == "xvla_source_reset_pool" else None,
        )),
        rollout=NS(pipeline_stage_num=1),
        algorithm=NS(adv_type="gae", loss_type="actor_critic"),
    )


class FakeRunner:
    def __init__(self, cfg, batches):
        self.cfg = cfg
        self.batches = batches
        self.max_steps = cfg.runner.max_steps
        self.global_step = 0
        self.syncs = 0
        self.saves = []
        self.finished = 0

    def run(self):
        for index in range(self.max_steps):
            steps, success = self.batches[index]
            row = {"rollout_index": index, "plan_sha256": PLAN, "mode": "uniform",
                   "task_id": 0, "seed_offset": 0, "trial_ids": [30, 32]}
            if self.training_seed is not None:
                row["training_seed"] = self.training_seed
            with self.ledger_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**row, "event": "source_rollout_started"}) + "\n")
                stream.write(json.dumps({
                    **row, "event": "source_rollout_completed",
                    "policy_env_steps": steps, "settling_env_steps": [10, 10], "success": success,
                }) + "\n")
            if any(success):
                with (self.run_dir / "audit-123.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write('{"event":"optimizer"}\n')
            self.global_step += 1
            self._log_step_metrics()
        self._finish_run()

    def _log_step_metrics(self):
        pass

    def update_rollout_weights(self):
        self.syncs += 1

    def _save_checkpoint(self):
        self.saves.append(self.global_step)
        if self.training_seed is not None:
            checkpoint = self.run_dir / "pilot/checkpoints" / f"global_step_{self.global_step}" / "actor"
            checkpoint.mkdir(parents=True)
            (checkpoint / "xvla_evidence.json").write_text(json.dumps({"digest": f"{self.global_step:064x}"}),
                                                           encoding="utf-8")

    def _finish_run(self):
        self.finished += 1


class FakeRewards(tuple):
    def __gt__(self, threshold):
        return NS(any=lambda: any(value > threshold for value in self))


class FakeActor:
    def __init__(self):
        self.rollout_batch = {}
        self.optimizer_steps = 0
        self._rank = 0

    def run_training(self):
        self.optimizer_steps += 1
        return {"updated": True}


class ControlBudgetTests(unittest.TestCase):
    def setUp(self):
        self.folder = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"XVLA_RUN_DIR": self.folder}))
        self.root = Path(self.folder)

    def test_stops_at_first_crossing_and_keeps_action_counts_separate(self):
        runner = control_budget_runner(FakeRunner)(
            config(self.root), [([3, 3], [True, False]), ([3, 3], [True, True])]
        )
        runner.run()
        summary = json.loads(runner.summary_path.read_text())
        self.assertEqual(runner.saves, [2])
        self.assertEqual((runner.syncs, runner.finished), (1, 1))
        self.assertEqual(summary["state"], "COMPLETE_VALID")
        self.assertEqual(summary["policy_env_steps"], 12)
        self.assertEqual(summary["overshoot_policy_env_steps"], 2)
        self.assertEqual(summary["settling_env_steps"], 40)
        self.assertEqual(summary["optimizer_updates"], 2)
        self.assertIn("global_step_2", summary["checkpoint_path"])

    def test_no_signal_is_explicit_at_full_budget(self):
        runner = control_budget_runner(FakeRunner)(
            config(self.root), [([5, 5], [False, False])]
        )
        runner.run()
        summary = json.loads(runner.summary_path.read_text())
        self.assertEqual(summary["state"], "COMPLETE_NO_SIGNAL")
        self.assertEqual(summary["optimizer_updates"], 0)

    def test_finite_cap_does_not_report_budget_completion(self):
        runner = control_budget_runner(FakeRunner)(
            config(self.root, cap=1), [([3, 3], [True, False])]
        )
        with self.assertRaisesRegex(RuntimeError, "cycle cap"):
            runner.run()
        summary = json.loads(runner.summary_path.read_text())
        self.assertEqual(summary["state"], "BATCH_CAP_REACHED")
        self.assertEqual(summary["policy_env_steps"], 6)
        self.assertEqual(runner.saves, [])

    def test_rejects_stale_or_malformed_rollout_evidence(self):
        path = self.root / "source_resets_0.jsonl"
        path.write_text(json.dumps({
            "event": "source_rollout_started", "rollout_index": 1, "plan_sha256": PLAN,
            "mode": "uniform", "task_id": 0, "seed_offset": 0, "trial_ids": [30, 32],
        }) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "mismatched rollout"):
            read_budget_ledger(path, environments=2, horizon=5, plan_sha256=PLAN, mode="uniform")

    def test_cfg_rejects_unsupported_run_paths(self):
        cfg = config(self.root)
        self.assertTrue(validate_control_budget(cfg))
        cfg.env.train.rollout_epoch = 2
        with self.assertRaisesRegex(ValueError, "one source-pool rollout"):
            validate_control_budget(cfg)

    def test_formal_requires_registered_milestones_and_seed(self):
        cfg = config(self.root, budget=200_000, seed=101, milestones=FORMAL_MILESTONES)
        self.assertTrue(validate_control_budget(cfg))
        for milestones, seed in (([50_000, 200_000], 101), (FORMAL_MILESTONES, 42), (50_000, 101)):
            with self.subTest(milestones=milestones, seed=seed):
                cfg.runner.xvla_control_step_milestones = milestones
                cfg.runner.xvla_training_seed = seed
                with self.assertRaisesRegex(ValueError, "registered"):
                    validate_control_budget(cfg)

    def test_formal_saves_three_first_crossings_with_cumulative_evidence(self):
        cfg = config(self.root, budget=200_000, cap=6, seed=101, milestones=FORMAL_MILESTONES, horizon=25_000)
        batches = [
            ([25_000, 24_998], [False, False]), ([4, 4], [False, False]),
            ([24_995, 24_995], [True, True]), ([4, 4], [False, False]),
            ([25_000, 24_996], [True, False]), ([25_000, 25_000], [True, True]),
        ]
        runner = control_budget_runner(FakeRunner)(cfg, batches)
        runner.run()
        summary = json.loads(runner.summary_path.read_text())
        self.assertEqual(runner.saves, [2, 4, 6])
        self.assertEqual((runner.syncs, runner.finished), (1, 1))
        self.assertEqual((summary["state"], summary["training_seed"]), ("COMPLETE_VALID", 101))
        milestones = summary["milestones"]
        self.assertEqual([receipt["requested_policy_env_steps"] for receipt in milestones], list(FORMAL_MILESTONES))
        self.assertEqual([receipt["policy_env_steps"] for receipt in milestones], [50_006, 100_004, 200_000])
        self.assertEqual([receipt["overshoot_policy_env_steps"] for receipt in milestones], [6, 4, 0])
        self.assertEqual([receipt["optimizer_updates"] for receipt in milestones], [0, 1, 3])
        self.assertEqual([receipt["settling_env_steps"] for receipt in milestones], [40, 80, 120])
        self.assertEqual([receipt["state"] for receipt in milestones],
                         ["COMPLETE_NO_SIGNAL", "COMPLETE_VALID", "COMPLETE_VALID"])
        self.assertEqual([receipt["checkpoint_sha256"] for receipt in milestones],
                         [f"{step:064x}" for step in runner.saves])

    def test_formal_cap_failure_keeps_reached_checkpoint_receipt(self):
        cfg = config(self.root, budget=200_000, cap=2, seed=202, milestones=FORMAL_MILESTONES, horizon=25_000)
        runner = control_budget_runner(FakeRunner)(cfg, [
            ([25_000, 25_000], [True, False]), ([1, 1], [False, False]),
        ])
        with self.assertRaisesRegex(RuntimeError, "cycle cap"):
            runner.run()
        summary = json.loads(runner.summary_path.read_text())
        self.assertEqual(summary["state"], "BATCH_CAP_REACHED")
        self.assertEqual(summary["policy_env_steps"], 50_002)
        self.assertEqual([receipt["requested_policy_env_steps"] for receipt in summary["milestones"]], [50_000])
        with self.assertRaisesRegex(RuntimeError, "mismatched rollout"):
            read_budget_ledger(runner.ledger_path, environments=2, horizon=25_000,
                               plan_sha256=PLAN, mode="uniform", training_seed=303)

    def test_formal_worker_seed_applies_before_model_init_and_restarts_exploration(self):
        calls = []

        class Worker:
            cfg = config(self.root, seed=303)

            def init_worker(self):
                calls.append("init")

        with patch("xvla_rlinf.control_budget.seed_training_worker", side_effect=calls.append):
            control_budget_actor(Worker)().init_worker()
            control_budget_rollout(Worker)().init_worker()
        self.assertEqual(calls, [303, "init", 303, "init", 303])

    def test_zero_reward_does_not_call_optimizer_but_positive_reward_does(self):
        events = []
        modules = {
            "torch": NS(isfinite=lambda values: NS(all=lambda: all(map(math.isfinite, values)))),
            "xvla_rlinf.audit_workers": NS(record=lambda event, **info: events.append((event, info))),
        }
        with patch.dict(sys.modules, modules):
            actor = control_budget_actor(FakeActor)()
            actor.rollout_batch["rewards"] = FakeRewards((0.0, 0.0))
            self.assertEqual(actor.run_training(), {})
            self.assertEqual(actor.optimizer_steps, 0)
            self.assertEqual(events[0][0], "training_skipped")
            actor.rollout_batch["rewards"] = FakeRewards((0.0, 1.0))
            self.assertEqual(actor.run_training(), {"updated": True})
            self.assertEqual(actor.optimizer_steps, 1)
            actor.rollout_batch["rewards"] = FakeRewards((math.nan,))
            with self.assertRaisesRegex(RuntimeError, "Nonfinite rewards"):
                actor.run_training()
