"""Check that the recovery adapter appends an exclusion to an existing journal."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import sys
import unittest
from unittest.mock import patch

from tools.xvla.robotwin import prescreen_recovery, runner


class PrescreenRecoveryTests(unittest.TestCase):
    def test_resumes_existing_seed_then_moves_on(self):
        class Env:
            plan_success = True

            def play_once(self):
                if self.seed == 2000:
                    raise AssertionError("missing grasp pose")

            def check_success(self):
                return True

            def close_env(self, **kwargs):
                pass

        def reset(env, cfg, index, seed):
            env.plan_success = True
            env.seed = seed

        unstable = ModuleType("envs.utils.create_actor")
        unstable.UnStableError = type("UnStableError", (Exception,), {})
        args = SimpleNamespace(task="turn_switch", condition="easy", phase="formal", limit=1,
                               screen_candidates=1)
        header = runner.common_row(args, "1" * 64)
        audit = {"evaluation_commit": prescreen_recovery.EVALUATION_COMMIT}
        with TemporaryDirectory() as folder, \
                patch.dict(sys.modules, {unstable.__name__: unstable}), \
                patch.object(runner, "reset_env", reset), \
                patch.object(runner, "scene_values", lambda env: [["scene", env.seed]]), \
                patch.object(runner, "load_native", lambda *unused: (None, Env, {"clear_cache_freq": 10}, None)), \
                patch.object(runner, "append", runner.append), \
                patch.object(runner, "emit"), \
                patch.object(prescreen_recovery, "expert_failure_reason", return_value="expert_grasp_pose_unavailable"):
            shard = Path(folder)
            path = shard / "prescreen.jsonl"
            for attempt in (1, 2, 3):
                runner.append(path, {**header, "requested_seed": 2000, "actual_seed": 2000,
                                     "attempt": attempt, "status": "technical_error"})
            original = path.read_bytes()
            prescreen_recovery.install_recovery(runner, audit)
            _, env_class, cfg, _ = runner.load_native(Path(folder), args.task, args.condition)
            runner.screen(args, shard, header, env_class, cfg)
            self.assertTrue(path.read_bytes().startswith(original))
            excluded = runner.resume_journal(path)[-1]
            self.assertEqual((excluded["attempt"], excluded["status"], excluded["reason"]),
                             (4, "excluded", "expert_grasp_pose_unavailable"))
            self.assertEqual(excluded["protocol_fingerprint"], header["protocol_fingerprint"])
            self.assertEqual(excluded["prescreen_recovery"], audit)
            self.assertIn("missing grasp pose", excluded["traceback"])
            runner.screen(args, shard, header, env_class, cfg)
            seed_record = runner.read_json(shard / "seed_manifest.json")["episodes"][0]
            self.assertEqual(seed_record["actual_seed"], 2001)


if __name__ == "__main__":
    unittest.main()
