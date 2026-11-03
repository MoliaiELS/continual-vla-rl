"""Exercise the real wrapper methods without importing MuJoCo, Ray or Torch."""
import ast
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np


source = Path(__file__).resolve().parents[2] / "rlinf/envs/libero/libero_env.py"
tree = ast.parse(source.read_text())
wrapper = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LiberoEnv")
methods = [n for n in wrapper.body if isinstance(n, ast.FunctionDef)
           and n.name in {"step", "_write_deployment_evidence"}]
namespace = {"np": np, "Path": Path, "json": json, "torch": SimpleNamespace(Tensor=type("Tensor", (), {})),
             "to_tensor": lambda x: np.array(x, copy=True),
             "list_of_dict_to_dict_of_list": lambda x: {}}
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(
    name="Probe", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])), str(source), "exec"), namespace)
Probe = namespace["Probe"]


def make_probe(evidence_dir):
    obj = Probe()
    obj.evidence_dir = evidence_dir
    obj._is_start = False
    obj.is_start = False
    obj.num_envs = 2
    obj.rank = 0
    obj.seed = 0
    obj.task_ids = np.array([0, 0])
    obj.trial_ids = np.array([0, 1])
    obj.task_descriptions = ["task zero", "task zero"]
    obj.cfg = SimpleNamespace(max_episode_steps=4)
    obj._elapsed_steps = np.zeros(2, dtype=int)
    # The actual class exposes this via a property.
    obj.elapsed_steps = obj._elapsed_steps
    obj.first_success_step = np.zeros(2, dtype=int)
    obj.success_once = np.zeros(2, dtype=bool)
    obj.deployment_evidence = {"policy_env_steps": 0, "settling_env_steps": 20,
                               "post_success_executed_steps": 0, "episodes": []}
    obj.video_cfg = SimpleNamespace(save_video=False)
    obj.ignore_terminations = True
    obj.auto_reset = False
    goals = iter([[False, False], [True, False], [True, False], [False, False]])
    obj.env = SimpleNamespace(step=lambda actions: (None, None, np.array(next(goals)), [{}, {}]))
    obj._wrap_obs = lambda raw: None
    obj._calc_step_reward = lambda goal: goal.astype(float) * 5

    def metrics(reward, goal, infos):
        obj.success_once |= goal
        return {"episode": {"success_once": obj.success_once.copy()}}
    obj._record_metrics = metrics
    return obj


class EvidenceTests(unittest.TestCase):
    def test_raw_final_goal_and_executed_post_success_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            probe = make_probe(tmp)
            for _ in range(4):
                probe.step(np.zeros((2, 7)))
            report = json.loads((Path(tmp) / "env_rank_0.json").read_text())
            self.assertEqual(report["policy_env_steps"], 8)
            self.assertEqual(report["settling_env_steps"], 20)
            self.assertEqual(report["post_success_executed_steps"], 2)
            first, second = report["episodes"]
            self.assertTrue(first["success_ever"])
            self.assertFalse(first["success_final"])
            self.assertEqual(first["first_success_step"], 2)
            self.assertEqual(first["post_success_executed_steps"], 2)
            self.assertEqual(second["init_state_id"], 1)
            self.assertFalse(second["success_ever"])
            self.assertIsNone(second["first_success_step"])

    def test_observability_preserves_rewards_and_termination_semantics(self):
        with tempfile.TemporaryDirectory() as tmp:
            observed, baseline = make_probe(tmp), make_probe(None)
            for _ in range(4):
                a, b = observed.step(np.zeros((2, 7))), baseline.step(np.zeros((2, 7)))
                for i in [1, 2, 3]:
                    np.testing.assert_array_equal(a[i], b[i])
                np.testing.assert_array_equal(a[4]["episode"]["success_at_end"],
                                              b[4]["episode"]["success_at_end"])


if __name__ == "__main__":
    unittest.main()
