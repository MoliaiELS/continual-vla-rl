"""Check reset-plan boundaries and rollout evidence with a stand-in simulator."""
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf

from xvla_rlinf.source_reset_pool import (
    load_source_reset_plan, source_reset_worker, validate_source_reset_pool,
)


class FakeLibero:
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.num_envs, self.group_size, self.seed_offset = num_envs, cfg.group_size, seed_offset
        self.trial_id_bins = [50]
        self._compute_total_num_group_envs()
        self.update_reset_state_ids()
        self.is_start = True

    def _compute_total_num_group_envs(self):
        pass

    def update_reset_state_ids(self):
        self.reset_state_ids = self._get_random_reset_state_ids(self.num_envs // self.group_size).repeat(self.group_size)

    def reset(self, env_idx=None, reset_state_ids=None):
        self.is_start = False
        self.trial_ids = self.reset_state_ids.copy()
        self.task_ids = np.zeros_like(self.trial_ids)
        return "observation", {}

    def chunk_step(self, actions):
        return "observation", None, None, None, [{"executed_mask": actions}]


class FakeWorker:
    def _setup_env_and_wrappers(self, env_cls, env_cfg, num_envs_per_stage):
        return env_cls(env_cfg, num_envs_per_stage, 0, 1, None)


class SourceResetPoolTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(TemporaryDirectory())
        self.enterContext(patch.dict(os.environ, {"XVLA_RUN_DIR": folder, "LIBERO_TYPE": "standard"}))
        self.root = Path(folder)
        self.path = self.root / "plan.json"
        self.document = {"schema_version": 1, "task_suite": "libero_spatial", "task_id": 0, "seed": 123,
                         "train_trial_ids": [0, 1, 2, 3], "source_dev_trial_ids": [10, 11],
                         "strata": {"rare": [0], "common": [1, 2, 3]}}
        self.write_plan()

    def write_plan(self):
        contents = json.dumps(self.document).encode()
        self.path.write_bytes(contents)
        self.digest = hashlib.sha256(contents).hexdigest()

    def config(self):
        return OmegaConf.create({
            "env_type": "libero", "task_suite_name": "libero_spatial", "task_id_filter": [0],
            "group_size": 2, "use_fixed_reset_state_ids": True, "xvla_strict_episodes": True,
            "specific_reset_id": None, "auto_reset": False, "use_ordered_reset_state_ids": False,
            "xvla_source_reset_pool": {"path": str(self.path), "sha256": self.digest, "mode": "stratified"},
        })

    def test_uniform_and_stratified_differ_on_the_same_unequal_pool(self):
        uniform = load_source_reset_plan(self.path, self.digest, "uniform")
        stratified = load_source_reset_plan(self.path, self.digest, "stratified")
        self.assertEqual(uniform.train_trial_ids, stratified.train_trial_ids)
        for mode, plan, expected in (("uniform", uniform, 0.25), ("stratified", stratified, 0.5)):
            with self.subTest(mode=mode):
                samples = plan.sample(np.random.default_rng(plan.seed), 2000)
                np.testing.assert_array_equal(samples, plan.sample(np.random.default_rng(plan.seed), 2000))
                self.assertEqual(set(samples), set(plan.train_trial_ids))
                self.assertAlmostEqual(float(np.mean(samples == 0)), expected, delta=.04)

    def test_rejects_changed_bytes_invalid_partitions_and_dev_leakage(self):
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_source_reset_plan(self.path, "0" * 64, "uniform")
        for key, value, message in (
            ("source_dev_trial_ids", [0, 10], "disjoint from development"),
            ("train_trial_ids", [0, 0], "duplicate trial IDs"),
            ("train_trial_ids", [0, True], "nonnegative integer"),
            ("strata", {"rare": [0, 1], "common": [1, 2]}, "disjoint, nonempty partition"),
            ("strata", {}, "requires reset strata"),
        ):
            with self.subTest(key=key, value=value), patch.dict(self.document, {key: value}):
                self.write_plan()
                with self.assertRaisesRegex(ValueError, message):
                    load_source_reset_plan(self.path, self.digest, "stratified")

    def test_driver_guards_training_and_forbids_eval_or_resume(self):
        cfg = SimpleNamespace(env=SimpleNamespace(train=self.config(), eval={}), runner={})
        self.assertTrue(validate_source_reset_pool(cfg))
        cfg.runner["resume_dir"] = "checkpoint"
        with self.assertRaisesRegex(ValueError, "fresh training"):
            validate_source_reset_pool(cfg)
        cfg.runner.clear()
        cfg.env.eval["xvla_source_reset_pool"] = {}
        with self.assertRaisesRegex(ValueError, "cannot modify evaluation"):
            validate_source_reset_pool(cfg)
        cfg.env.eval.clear()
        cfg.env.train.specific_reset_id = 0
        with self.assertRaisesRegex(ValueError, "strict non-auto-reset"):
            validate_source_reset_pool(cfg)

    def test_worker_logs_actual_grouped_resets_once_per_first_chunk(self):
        worker = source_reset_worker(FakeWorker)()
        env = worker._setup_env_and_wrappers(FakeLibero, self.config(), 4)
        log = self.root / "source_resets_0.jsonl"
        self.assertFalse(log.exists())
        env.reset()
        self.assertFalse(log.exists())
        first_chunk = np.array([[True, True], [True, False], [True, False], [True, True]])
        env.chunk_step(first_chunk)
        env.chunk_step(first_chunk)
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["rollout_index"], 0)
        self.assertEqual(row["trial_ids"], env.trial_ids.tolist())
        self.assertEqual(row["trial_ids"], [trial for trial in row["group_trial_ids"] for _ in range(2)])
        self.assertEqual(row["first_chunk_policy_env_steps"], [2, 1, 1, 2])
        env.update_reset_state_ids()
        env.is_start = True
        env.reset()
        env.chunk_step(first_chunk)
        self.assertEqual(json.loads(log.read_text().splitlines()[-1])["rollout_index"], 1)

    def test_environment_rejects_missing_trial_and_keeps_unconfigured_env(self):
        worker = source_reset_worker(FakeWorker)()
        with patch.dict(self.document, {"source_dev_trial_ids": [50]}):
            self.write_plan()
            with self.assertRaisesRegex(ValueError, "outside task 0"):
                worker._setup_env_and_wrappers(FakeLibero, self.config(), 4)
        cfg = self.config()
        del cfg["xvla_source_reset_pool"]
        with patch.object(FakeLibero, "_get_random_reset_state_ids", return_value=np.array([0, 1]), create=True):
            self.assertIs(type(worker._setup_env_and_wrappers(FakeLibero, cfg, 4)), FakeLibero)


if __name__ == "__main__":
    unittest.main()
