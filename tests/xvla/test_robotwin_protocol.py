import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from tools.xvla.robotwin import protocol as p
from tools.xvla.robotwin.runner import make_seed, resume_journal, same_scene


def observation():
    return {
        "observation": {name: {"rgb": np.full(p.RGB_SHAPE, i, np.uint8)}
                        for i, name in enumerate(p.CAMERAS)},
        "endpose": {"left_endpose": [0., 1., 2., 1., 0., 0., 0.], "left_gripper": 0.8,
                    "right_endpose": [3., 4., 5., 0., 0., 0., 1.], "right_gripper": 0.3},
    }


class ProtocolTests(unittest.TestCase):
    def test_raw_quaternion_slots_and_command_then_measured_history(self):
        obs = observation()
        images, state = p.observation_values(obs)
        self.assertEqual([int(im[0, 0, 0]) for im in images], [0, 1, 2])
        # [1,0,0,0] is raw wxyz from RoboTwin, but official SciPy treats it
        # as a half-turn about X. This catches a tempting reorder regression.
        np.testing.assert_allclose(state, [0, 1, 2, 1, 0, 0, -1, 0, 0, -0.6,
                                           3, 4, 5, 1, 0, 0, 1, 0, 0, 0.4])
        pred = np.repeat(state[None], p.ANCHORS, axis=0)
        pred[:, 9], pred[:, 19] = 0.7, 0.70001
        pred[:, 0], pred[:, 10] = -1, -2
        command = p.actions_to_env(pred)[0]
        np.testing.assert_allclose(command, [-1, 1, 2, 1, 0, 0, 0, 1,
                                             -2, 4, 5, 0, 0, 0, 1, -1])
        p.update_history(obs, command)
        _, next_state = p.observation_values(obs)
        np.testing.assert_allclose(next_state[[0, 9, 10, 19]], [-1, -0.6, -2, 0.4])

    def test_env_no_ops_dont_inflate_executed_budget(self):
        _, state = p.observation_values(observation())
        actions = np.repeat(state[None], p.ANCHORS, axis=0)

        class Env:
            take_action_cnt = 0
            suc = 0

            def _update_render(self):
                pass

            def get_obs(self):
                return observation()

            def take_action(self, _command, action_type):
                assert action_type == "ee"
                if self.take_action_cnt < 2:
                    self.take_action_cnt += 1

            def check_success(self):
                return False

        budget = p.Budget(step_lim=2)
        self.assertEqual(p.rollout(Env(), SimpleNamespace(step=lambda obs: actions), budget),
                         ("failure", "step_limit"))
        self.assertEqual((budget.queries, budget.anchors_sent, budget.anchors), (10, 300, 2))

    def test_protocol_seed_and_resume_after_partial_append(self):
        key = dict(protocol=p.PROTOCOL, phase="formal", task="turn_switch", condition="easy")
        seed = make_seed(key, 100, 2042, "unique_scene")
        self.assertEqual(seed["batch"], 2)
        self.assertEqual(seed, make_seed(key, 100, 2042, "unique_scene"))
        self.assertNotEqual(seed["policy_seed"], make_seed({**key, "condition": "hard"}, 100, 2042,
                                                          "unique_scene")["policy_seed"])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "episodes.jsonl"
            row = {**seed, "status": "failure"}
            path.write_text(json.dumps(row) + '\n{"episode_id":', encoding="utf-8")
            self.assertEqual(resume_journal(path), [row])
            self.assertEqual(resume_journal(path), [row])
            self.assertEqual(len(list(path.parent.glob("episodes.jsonl.partial-*"))), 1)
        self.assertTrue(same_scene([["obj", 1.0]], [["obj", 1.0005]]))
        self.assertFalse(same_scene([["obj", 1.0]], [["obj", 1.05]]))


if __name__ == "__main__":
    unittest.main()
