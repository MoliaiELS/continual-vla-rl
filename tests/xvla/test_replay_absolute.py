"""Check absolute replay orchestration without claiming simulator evidence."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

from test_sft_data import fixture
from xvla_rlinf.replay_absolute import replay_manifest
from xvla_rlinf.sft import sha256_file


class ReplayEnv:
    def __init__(self, successful_steps):
        self.successful_steps = successful_steps
        self.controller = SimpleNamespace(use_delta=True)
        self.env = SimpleNamespace(robots=[SimpleNamespace(controller=self.controller)], control_freq=20)
        self.calls = []
        self.closed = False

    def seed(self, seed):
        self.seed_value = seed

    def reset(self):
        self.calls.clear()

    def set_init_state(self, state):
        self.initial_state = state.copy()

    def step(self, action):
        self.calls.append((self.controller.use_delta, action.copy()))
        return None, 0, len(self.calls) - 10 in self.successful_steps, {}

    def close(self):
        self.closed = True


class ReplayAbsoluteTests(unittest.TestCase):
    def test_replays_both_tracks_and_preserves_ever_final_difference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest_path, manifest = fixture(root, steps=15)
            source = root / "source.hdf5"
            initial_state = np.array([1., 2., 3.])
            relative = np.zeros((15, 7), dtype=np.float32)
            relative[:, 6] = -1
            with h5py.File(source, "w") as data:
                data.create_dataset("data/demo_1/states", data=initial_state[None])
                data.create_dataset("data/demo_1/actions", data=relative)
            manifest.update(source_hdf5=str(source), source_sha256=sha256_file(source), data_revision="pinned")
            manifest["episodes"][0]["control_freq"] = 20
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            relative_env, absolute_env = ReplayEnv({12, 15}), ReplayEnv({13})
            environments = iter((relative_env, absolute_env))

            def env_factory(task_suite, task_id, camera_size):
                self.assertEqual((task_suite, task_id, camera_size), ("libero_spatial", 0, 256))
                return next(environments), None

            report = replay_manifest(manifest_path, env_factory=env_factory, rotation_backend="numpy")
            result, = report["episodes"]
            self.assertEqual(report["status"], "FAIL")
            self.assertTrue(result["relative"]["success_final"])
            self.assertTrue(result["absolute"]["success_ever"])
            self.assertFalse(result["absolute"]["success_final"])
            self.assertEqual(result["absolute"]["post_done_action_steps"], 2)
            self.assertEqual(result["gripper_commands_changed"], 0)
            for env, use_delta in ((relative_env, True), (absolute_env, False)):
                self.assertEqual(env.seed_value, 0)
                np.testing.assert_array_equal(env.initial_state, initial_state)
                self.assertEqual([delta for delta, _ in env.calls], [True] * 10 + [use_delta] * 15)
                self.assertTrue(env.closed)
            np.testing.assert_allclose(absolute_env.calls[-1][1], [0.014, 0, 0, 0, 0, 0, -1])


if __name__ == "__main__":
    unittest.main()
