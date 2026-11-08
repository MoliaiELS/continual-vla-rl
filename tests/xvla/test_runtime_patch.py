"""Regression checks for the opt-in pinned-runtime patch.

These tests exercise Python/NumPy control logic and patch integrity. They do not
claim to test MuJoCo, CUDA, actual controller dynamics, or learned policies.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import numpy as np
except ImportError:
    np = None
try:
    import torch
except ImportError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("xvla_runtime_patch", ROOT / "tools/xvla/apply_runtime_patch.py")
patch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch_module)


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True).stdout


def load_functions(source, names, namespace, class_name=None):
    tree = ast.parse(source)
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name).body if class_name else tree.body
    funcs = [n for n in body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
    assert len(funcs) == len(names), names
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *funcs], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<audited-runtime-functions>", "exec"), namespace)
    return namespace


class PatchApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        git(self.path, "init", "-q")
        (self.path / "file.txt").write_text("original\n")
        git(self.path, "add", "file.txt")
        git(self.path, "-c", "user.name=Patch test", "-c", "user.email=test@example.invalid", "commit", "-qm", "fixture")
        self.commit = git(self.path, "rev-parse", "HEAD").decode().strip()
        self.patch = self.path / "change.patch"
        self.patch.write_text("--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n-original\n+patched\n")
        self.manifest = self.path / "manifest.json"
        self.metadata = {"upstream_commit": self.commit, "patch_sha256": hashlib.sha256(self.patch.read_bytes().replace(b"\r\n", b"\n")).hexdigest(), "files": [{"path": "file.txt", "before_sha256": hashlib.sha256(b"original\n").hexdigest(), "after_sha256": hashlib.sha256(b"patched\n").hexdigest()}]}
        self.manifest.write_text(json.dumps(self.metadata))

    def tearDown(self):
        self.temp.cleanup()

    def apply(self, **kwargs):
        return patch_module.apply_runtime_patch(self.path, patch=self.patch, manifest=self.manifest, **kwargs)

    def test_check_does_not_mutate_and_apply_is_idempotent(self):
        self.assertEqual(self.apply(check_only=True)["status"], "applicable")
        self.assertEqual((self.path / "file.txt").read_text(), "original\n")
        self.assertEqual(self.apply()["status"], "applied")
        self.assertEqual(self.apply()["status"], "already_applied")

    def test_user_edit_is_refused_and_preserved(self):
        (self.path / "file.txt").write_text("user work\n")
        with self.assertRaisesRegex(RuntimeError, "preserved"):
            self.apply()
        self.assertEqual((self.path / "file.txt").read_text(), "user work\n")

    def test_unrelated_dirty_file_is_preserved(self):
        (self.path / "unrelated.txt").write_text("keep\n")
        self.apply()
        self.assertEqual((self.path / "unrelated.txt").read_text(), "keep\n")

    def test_wrong_version_is_refused(self):
        self.metadata["upstream_commit"] = "0" * 40
        self.manifest.write_text(json.dumps(self.metadata))
        with self.assertRaisesRegex(RuntimeError, "HEAD must be"):
            self.apply()

    def test_tampered_patch_is_refused(self):
        self.patch.write_text(self.patch.read_text() + "\n# tampered\n")
        with self.assertRaisesRegex(RuntimeError, "checksum"):
            self.apply()


class RuntimePatchSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        git(cls.root, "init", "-q")
        cls.metadata = json.loads((ROOT / "patches/rlinf-xvla.json").read_text())
        for item in cls.metadata["files"]:
            data = git(ROOT / "third_party/rlinf", "show", f"{cls.metadata['upstream_commit']}:{item['path']}")
            assert hashlib.sha256(data).hexdigest() == item["before_sha256"]
            target = cls.root / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        git(cls.root, "apply", str(ROOT / "patches/rlinf-xvla.patch"))
        cls.sources = {}
        for item in cls.metadata["files"]:
            data = (cls.root / item["path"]).read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(data).hexdigest() == item["after_sha256"]
            ast.parse(data)
            cls.sources[item["path"]] = data.decode()

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_pinned_patch_applies_to_exact_sources(self):
        self.assertEqual(len(self.sources), 7)

    def test_isolated_ray_is_opt_in_bounded_and_preserves_cuda_mask(self):
        ns = load_functions(self.sources["rlinf/scheduler/cluster/cluster.py"], {"_xvla_isolated_ray_kwargs"}, {"os": os, "Path": Path})
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(ns["_xvla_isolated_ray_kwargs"]())
        with tempfile.TemporaryDirectory() as temp:
            config = {"RLINF_XVLA_ISOLATED_RAY": "1", "SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "8", "CUDA_VISIBLE_DEVICES": "2,GPU-123", "RAY_TMPDIR": str(Path(temp) / "ray_12345"), "FYP_XVLA_RAY_SPILL": str(Path(temp) / "spill_12345")}
            with patch.dict(os.environ, config, clear=True):
                actual = ns["_xvla_isolated_ray_kwargs"]()
                self.assertEqual(dict(os.environ), config)
            self.assertEqual(actual["address"], "local")
            self.assertEqual(actual["num_cpus"], 8)
            self.assertEqual(actual["num_gpus"], 2)
            self.assertEqual(actual["object_store_memory"], 2 * 1024**3)
            self.assertEqual(json.loads(actual["_system_config"]["object_spilling_config"])["params"]["directory_path"], config["FYP_XVLA_RAY_SPILL"])

    def test_isolated_ray_rejects_unallocated_or_unbounded_resources(self):
        ns = load_functions(self.sources["rlinf/scheduler/cluster/cluster.py"], {"_xvla_isolated_ray_kwargs"}, {"os": os, "Path": Path})
        with patch.dict(os.environ, {"RLINF_XVLA_ISOLATED_RAY": "1"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "Slurm"):
                ns["_xvla_isolated_ray_kwargs"]()
        with patch.dict(os.environ, {"RLINF_XVLA_ISOLATED_RAY": "1", "SLURM_JOB_ID": "12345", "SLURM_CPUS_PER_TASK": "8", "RLINF_XVLA_RAY_CPUS": "9"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SLURM_CPUS"):
                ns["_xvla_isolated_ray_kwargs"]()

    def test_mode_is_opt_in_and_preserves_sampling_kwargs(self):
        tree = ast.parse(self.sources["rlinf/workers/rollout/hf/huggingface_worker.py"])
        candidates = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Attribute) and n.test.func.attr == "get" and n.test.args and isinstance(n.test.args[0], ast.Constant) and n.test.args[0].value == "pass_predict_mode"]
        self.assertEqual(len(candidates), 1)
        code = compile(ast.fix_missing_locations(ast.Module(body=candidates, type_ignores=[])), "mode-gate", "exec")
        for enabled, dagger, expected in [(False, False, None), (True, False, "train"), (True, True, "eval")]:
            ns = {"self": SimpleNamespace(model_cfg={"pass_predict_mode": enabled}, enable_dagger=dagger), "kwargs": {"temperature": 1.6}, "mode": "train"}
            exec(code, ns)
            self.assertEqual(ns["kwargs"].get("mode"), expected)
            self.assertEqual(ns["kwargs"]["temperature"], 1.6)

    def test_old_models_default_logprob_layout_remains_physical_action_dim(self):
        tree = ast.parse(self.sources["rlinf/workers/actor/embodied_fsdp_actor_worker.py"])
        expression = next(value for node in ast.walk(tree) if isinstance(node, ast.Dict) for key, value in zip(node.keys, node.values) if isinstance(key, ast.Constant) and key.value == "single_action_dim")
        code = compile(ast.Expression(expression), "logprob-layout", "eval")
        for config, expected in [({}, 7), ({"action_dim": 6}, 6), ({"action_dim": 7, "logprob_action_dim": 1}, 1)]:
            self.assertEqual(eval(code, {"self": SimpleNamespace(cfg=SimpleNamespace(actor=SimpleNamespace(model=config)))}), expected)


@unittest.skipIf(np is None, "NumPy not installed: controller/reset logic NOT RUN")
class ControllerAndResetTests(RuntimePatchSource):
    def test_raw_image_opt_in_keeps_old_models_both_view_rotation(self):
        util_source = (ROOT / "third_party/rlinf/rlinf/envs/sim/libero/utils.py").read_text()
        ns = load_functions(util_source, {"get_libero_image", "get_libero_wrist_image"}, {"np": np})
        ns["quat2axisangle"] = lambda quat: np.array([0.0, 0.0, 0.0])
        load_functions(self.sources["rlinf/envs/sim/libero/libero_env.py"], {"_extract_image_and_state"}, ns, "LiberoEnv")
        image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        obs = {"agentview_image": image.copy(), "robot0_eye_in_hand_image": (image + 20).copy(), "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.array([0, 0, 0, 1]), "robot0_gripper_qpos": np.zeros(2)}
        old = ns["_extract_image_and_state"](SimpleNamespace(cfg={}), obs)
        np.testing.assert_array_equal(old["full_image"], image[::-1, ::-1])
        np.testing.assert_array_equal(old["wrist_image"], (image + 20)[::-1, ::-1])
        native = ns["_extract_image_and_state"](SimpleNamespace(cfg={"xvla_raw_observations": True}), obs)
        np.testing.assert_array_equal(native["full_image"], image)
        np.testing.assert_array_equal(native["wrist_image"], image + 20)
        np.testing.assert_array_equal(obs["agentview_image"], image)

    def test_measured_state_uses_rotation_columns_and_keeps_padding_zero(self):
        ns = load_functions(self.sources["rlinf/envs/sim/libero/venv.py"], {"_controllers", "_set_controller_delta", "_measured_xvla_proprio"}, {"np": np})
        matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        controller = SimpleNamespace(use_delta=True, ee_pos=np.array([1, 2, 3]), ee_ori_mat=matrix)
        env = SimpleNamespace(env=SimpleNamespace(robots=[SimpleNamespace(controller=controller)]))
        result = ns["_measured_xvla_proprio"](env)
        np.testing.assert_array_equal(result[:9], [1, 2, 3, 0, 1, 0, -1, 0, 0])
        np.testing.assert_array_equal(result[9:], np.zeros(11))
        ns["_set_controller_delta"](env, False)
        self.assertFalse(controller.use_delta)
        ns["_set_controller_delta"](env, True)
        self.assertTrue(controller.use_delta)

    def test_missing_controller_contract_fails_instead_of_silent_delta(self):
        ns = load_functions(self.sources["rlinf/envs/sim/libero/venv.py"], {"_controllers", "_set_controller_delta"}, {})
        env = SimpleNamespace(robots=[SimpleNamespace(controller=object())])
        with self.assertRaisesRegex(RuntimeError, "use_delta"):
            ns["_set_controller_delta"](env, False)

    def test_reset_settles_in_delta_and_restores_absolute_on_every_reset(self):
        ns = load_functions(self.sources["rlinf/envs/sim/libero/libero_env.py"], {"reset"}, {"np": np}, "LiberoEnv")
        class Config(dict):
            reset_gripper_open = True
        calls = []
        vec = SimpleNamespace(
            set_controller_delta=lambda enabled, id: calls.append(("mode", enabled, tuple(id))),
            step=lambda actions, indices: (calls.append(("step", tuple(indices))) or ([{}] * len(indices), None, None, None)),
        )
        env = SimpleNamespace(num_envs=2, is_start=False, cfg=Config(xvla_absolute_control=True, num_steps_wait=2), env=vec, current_raw_obs=[{}, {}], _xvla_finished=np.array([True, True]), _xvla_reset_mask=np.zeros(2, bool), _xvla_settling_steps=np.zeros(2, np.int64), _reconfigure=lambda ids, indices: None, _wrap_obs=lambda obs: obs, _reset_metrics=lambda ids: None)
        for _ in range(2):
            ns["reset"](env, env_idx=np.array([1]), reset_state_ids=np.array([0]))
        self.assertEqual(calls, [("mode", True, (1,)), ("step", (1,)), ("step", (1,)), ("mode", False, (1,))] * 2)
        np.testing.assert_array_equal(env._xvla_settling_steps, [0, 4])
        np.testing.assert_array_equal(env._xvla_reset_mask, [False, True])
        np.testing.assert_array_equal(env._xvla_finished, [True, False])

    def test_strict_step_never_executes_after_done_or_horizon(self):
        torch_stub = SimpleNamespace(Tensor=type("UnusedTensor", (), {}))
        ns = load_functions(self.sources["rlinf/envs/sim/libero/libero_env.py"], {"_step_xvla"}, {"np": np, "torch": torch_stub, "to_tensor": lambda x: x, "list_of_dict_to_dict_of_list": lambda values: {}}, "LiberoEnv")
        calls = []
        class Config:
            max_episode_steps = 2
        def vector_step(actions, indices):
            calls.append(tuple(indices))
            return ([{} for _ in indices], None, np.array([index == 0 for index in indices]), [{} for _ in indices])
        elapsed = np.zeros(2, np.int64)
        env = SimpleNamespace(cfg=Config(), num_envs=2, ignore_terminations=False, auto_reset=False, current_raw_obs=[{}, {}], env=SimpleNamespace(step=vector_step), _elapsed_steps=elapsed, elapsed_steps=elapsed, _xvla_finished=np.zeros(2, bool), _xvla_policy_steps=np.zeros(2, np.int64), _xvla_settling_steps=np.zeros(2, np.int64), _wrap_obs=lambda raw: raw, _calc_step_reward=lambda term: np.where(term, 5.0, -5.0), _record_metrics=lambda reward, term, infos: {"episode": {}})
        rewards = []
        for _ in range(3):
            rewards.append(ns["_step_xvla"](env, np.zeros((2, 7)))[1])
        self.assertEqual(calls, [(0, 1), (1,)])
        np.testing.assert_array_equal(elapsed, [1, 2])
        np.testing.assert_array_equal(env._xvla_policy_steps, [1, 2])
        np.testing.assert_array_equal(rewards[1][0], 0)
        np.testing.assert_array_equal(rewards[2], [0, 0])


@unittest.skipIf(torch is None, "Torch not installed: real tensor probability reduction NOT RUN")
class ProbabilityLayoutTests(unittest.TestCase):
    def test_observation_reset_masks_are_per_environment_and_one_shot(self):
        metadata = json.loads((ROOT / "patches/rlinf-xvla.json").read_text())
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        runtime = Path(temp.name)
        git(runtime, "init", "-q")
        for item in metadata["files"]:
            target = runtime / item["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(git(ROOT / "third_party/rlinf", "show", f"{metadata['upstream_commit']}:{item['path']}"))
        git(runtime, "apply", str(ROOT / "patches/rlinf-xvla.patch"))
        source = (runtime / "rlinf/envs/sim/libero/libero_env.py").read_text()
        def tensorize(data):
            return {key: torch.as_tensor(np.stack(values)) for key, values in data.items()} if isinstance(data, dict) else torch.as_tensor(data)
        ns = load_functions(source, {"_wrap_obs"}, {"torch": torch, "to_tensor": tensorize, "list_of_dict_to_dict_of_list": lambda rows: {key: [row[key] for row in rows] for key in rows[0]}}, "LiberoEnv")
        measured = np.zeros((2, 20), dtype=np.float32)
        measured[0, 0], measured[1, 0] = 1, 2
        row = {"full_image": np.zeros((2, 2, 3), np.uint8), "wrist_image": np.ones((2, 2, 3), np.uint8), "state": np.zeros(8, np.float32)}
        env = SimpleNamespace(cfg={"xvla_measured_state": True, "xvla_history_reset": True}, _extract_image_and_state=lambda obs: row, env=SimpleNamespace(measured_xvla_proprio=lambda: measured), _xvla_reset_mask=np.array([True, False]), task_descriptions=["a", "b"])
        first, second = ns["_wrap_obs"](env, [{}, {}]), ns["_wrap_obs"](env, [{}, {}])
        torch.testing.assert_close(first["reset_mask"], torch.tensor([True, False]))
        torch.testing.assert_close(second["reset_mask"], torch.tensor([False, False]))
        torch.testing.assert_close(first["states"][:, 0], torch.tensor([1.0, 2.0]))

    def test_joint_per_step_density_sums_chunk_once(self):
        source = (ROOT / "third_party/rlinf/rlinf/algorithms/utils.py").read_text()
        ns = load_functions(source, {"preprocess_loss_inputs", "expand_to_target_dim"}, {"torch": torch})
        logprob = torch.tensor([[[-1.0], [-2.0], [-3.0]], [[-4.0], [-5.0], [-6.0]]])
        result = ns["preprocess_loss_inputs"](logprobs=logprob, old_logprobs=logprob.clone(), advantages=torch.ones(2, 1), reward_type="chunk_level", logprob_type="chunk_level", single_action_dim=1, loss_mask=torch.ones(2, 1, dtype=torch.bool))
        torch.testing.assert_close(result["logprobs"], torch.tensor([-6.0, -15.0]))
        torch.testing.assert_close((result["logprobs"] - result["old_logprobs"]).exp(), torch.ones(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
