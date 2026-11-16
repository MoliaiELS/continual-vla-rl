import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tools.xvla.robotwin import official_rollout as client
from tools.xvla.robotwin import protocol as p


class Env:
    actor_pose = True

    def __init__(self):
        self.take_action_cnt = 0
        self.calls = []

    def take_action(self, command, action_type):
        self.calls.append((command, action_type))
        if self.take_action_cnt < 2:
            self.take_action_cnt += 1


def raw_actions():
    actions = np.zeros((p.ANCHORS, 20), dtype=np.float32)
    actions[:, [3, 6, 13, 16]] = 1.0
    return actions


class OfficialRolloutTests(unittest.TestCase):
    def test_exact_source_is_checked_without_running_top_level_code(self):
        source = b'''raise RuntimeError("would run when imported as a module")
def rotate6D_to_quat(value):
    return value
def _rollout(env, policy):
    return rotate6D_to_quat((0, []))
'''
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            file = root / client.CLIENT_PATH
            file.parent.mkdir(parents=True)
            file.write_bytes(source)
            with patch.object(client, "CLIENT_SHA256", hashlib.sha256(source).hexdigest()):
                self.assertEqual(client.load_official_rollout(root)(None, None), (0, []))
                file.write_bytes(source + b"\n")
                with self.assertRaisesRegex(ValueError, "Wrong X-VLA RoboTwin client SHA256"):
                    client.load_official_rollout(root)

    def test_counts_delivered_noops_and_keeps_original_commands(self):
        env, budget = Env(), p.Budget(step_lim=2)
        observation = object()
        frames = [object()]
        command = np.arange(16, dtype=np.float64)
        inspected = []

        def query(obs):
            self.assertIs(obs, observation)
            return raw_actions()

        def official(backend, policy):
            self.assertEqual(policy.step(observation).dtype, np.float64)
            for _ in range(p.ANCHORS):
                backend.take_action(command, action_type="ee")
            return 0, frames

        result = client.execute(official, env, SimpleNamespace(step=query), budget, return_frames=True,
                                check_command=lambda action, kind: inspected.append((action, kind)))
        self.assertEqual(result[:2], ("failure", "step_limit"))
        self.assertIs(result[2], frames)
        self.assertEqual((budget.queries, budget.anchors_sent, budget.anchors), (1, p.ANCHORS, 2))
        self.assertTrue(all(actual is command and kind == "ee" for actual, kind in env.calls))
        self.assertEqual(len(inspected), len(env.calls))
        self.assertNotIn("take_action", vars(env))

    def test_invalid_policy_chunk_never_reaches_robot(self):
        env, budget = Env(), p.Budget()

        def official(backend, policy):
            policy.step(None)
            self.fail("Official rollout accepted an invalid response")

        result = client.execute(official, env, SimpleNamespace(step=lambda _: np.zeros((1, 20))), budget)
        self.assertEqual(result, ("failure", "invalid_policy_action", None))
        self.assertEqual((budget.queries, budget.anchors_sent, budget.anchors), (1, 0, 0))
        self.assertNotIn("take_action", vars(env))


if __name__ == "__main__":
    unittest.main()
