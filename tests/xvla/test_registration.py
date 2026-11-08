"""Pinned RLinf registration and configuration tests without model loading.

Mocking Cluster/placement only validates config semantics; actual Ray/Slurm and
GPU placement still require the online smoke job. The child-process tests call
the same extension hook used by workers and checkpoint utilities, without
claiming to construct real Ray workers.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "third_party/rlinf"
HAS_HYDRA = importlib.util.find_spec("hydra") is not None
HAS_TORCH = importlib.util.find_spec("torch") is not None


def compose_config(name, overrides=()):
    from hydra import initialize_config_dir, compose
    from omegaconf import OmegaConf
    with initialize_config_dir(version_base="1.1", config_dir=str(ROOT / "configs/xvla")):
        result = compose(config_name=name, overrides=list(overrides))
    OmegaConf.resolve(result)
    return result


@unittest.skipUnless(HAS_HYDRA, "Hydra unavailable: configuration composition NOT RUN")
class HydraCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        env = {
            "RLINF_RUNTIME": str(RUNTIME),
            "EMBODIED_PATH": str(RUNTIME / "examples/embodiment"),
            "XVLA_RUN_DIR": self.temp.name,
            "XVLA_MODEL_PATH": "unused-native-checkpoint",
            "XVLA_PT_MODEL_PATH": "unused-controlled-sft-checkpoint",
            "FYP_XVLA_DATA": self.temp.name,
        }
        self.environ = patch.dict(os.environ, env)
        self.environ.start()
        self.addCleanup(self.environ.stop)

    def test_all_primary_configs_compose_and_resolve(self):
        for name in ("ppo_smoke", "grpo_smoke", "native_eval_smoke", "stochastic_eval_smoke", "sft_smoke"):
            with self.subTest(name=name):
                cfg = compose_config(name)
                self.assertEqual(cfg.actor.model.model_type, "xvla")
                self.assertFalse(cfg.actor.model.is_lora)
                self.assertEqual(cfg.actor.model.precision, "fp32")

    def test_ppo_and_grpo_have_distinct_correct_loss_and_group_contracts(self):
        ppo = compose_config("ppo_smoke")
        grpo = compose_config("grpo_smoke")
        self.assertEqual((ppo.algorithm.adv_type, ppo.algorithm.loss_type), ("gae", "actor_critic"))
        self.assertEqual((grpo.algorithm.adv_type, grpo.algorithm.loss_type), ("grpo", "actor"))
        self.assertTrue(ppo.actor.model.add_value_head)
        self.assertFalse(grpo.actor.model.add_value_head)
        for cfg in (ppo, grpo):
            self.assertEqual(cfg.runner.val_check_interval, 1)
            self.assertEqual(cfg.actor.model.logprob_action_dim, 1)
            self.assertTrue(cfg.actor.model.pass_predict_mode)
            self.assertTrue(cfg.rollout.model.pass_predict_mode)
            self.assertEqual(cfg.env.train.total_num_envs % cfg.env.train.group_size, 0)
            self.assertEqual(cfg.env.train.max_steps_per_rollout_epoch % cfg.actor.model.num_action_chunks, 0)

    def test_native_and_stochastic_evaluation_use_same_horizon_and_controller(self):
        native, stochastic = compose_config("native_eval_smoke"), compose_config("stochastic_eval_smoke")
        self.assertEqual(native.env.eval, stochastic.env.eval)
        self.assertEqual(native.env.eval.max_episode_steps, 800)
        self.assertEqual(native.env.eval.max_steps_per_rollout_epoch, 810)
        self.assertTrue(native.env.eval.xvla_absolute_control)
        self.assertTrue(native.env.eval.xvla_measured_state)
        self.assertTrue(native.env.eval.xvla_raw_observations)
        self.assertTrue(native.env.eval.xvla_strict_episodes)
        self.assertTrue(native.env.eval.is_eval)
        self.assertEqual(native.actor.model.xvla.eval_sampler, "native_eval")
        self.assertEqual(stochastic.actor.model.xvla.eval_sampler, "terminal_clean_gaussian_v1")
        self.assertEqual(stochastic.actor.model.xvla.eval_history_mode, "native_history")


@unittest.skipUnless(HAS_TORCH, "Torch unavailable: real RLinf registry and config validation NOT RUN")
class RegistrationTests(HydraCompositionTests):
    @classmethod
    def setUpClass(cls):
        import rlinf
        if not Path(rlinf.__file__).resolve().is_relative_to(RUNTIME.resolve()):
            raise RuntimeError("Registration tests require the pinned third_party/rlinf PYTHONPATH, not legacy root RLinf")
        import rlinf.models
        import xvla_rlinf
        cls.models = rlinf.models
        cls.extension = xvla_rlinf

    def test_registration_is_idempotent_and_preserves_all_builtin_builders(self):
        previous = {key: value for key, value in self.models._MODEL_REGISTRY.items() if key != "xvla"}
        self.extension.register()
        self.extension.register()
        self.assertIs(self.models._MODEL_REGISTRY["xvla"], self.extension.build_model)
        for key, value in previous.items():
            self.assertIs(self.models._MODEL_REGISTRY[key], value, key)
        from rlinf.config import SupportedModel, EMBODIED_MODEL
        self.assertIn(SupportedModel("xvla"), EMBODIED_MODEL)

    def test_factory_dispatches_xvla_without_loading_weights_or_double_lora(self):
        import torch
        from omegaconf import OmegaConf
        from xvla_rlinf.policy import XVLAPolicy
        self.extension.register()
        marker = torch.nn.Linear(1, 1)
        cfg = OmegaConf.create({"model_type": "xvla", "precision": "fp32", "is_lora": False, "load_to_device": False})
        with patch.object(XVLAPolicy, "from_checkpoint", return_value=marker) as build:
            self.assertIs(self.models.get_model(cfg), marker)
            build.assert_called_once_with(cfg, torch.float32)
        cfg.is_lora = True
        with self.assertRaisesRegex(ValueError, "factory LoRA"):
            self.models.get_model(cfg)

    def test_worker_and_checkpoint_extension_hook_in_separate_process(self):
        code = r'''
import json, os
import rlinf.models as models
from rlinf.scheduler.cluster import load_user_extension_module
old = dict(models._MODEL_REGISTRY)
load_user_extension_module()
import xvla_rlinf
assert models._MODEL_REGISTRY['xvla'] is xvla_rlinf.build_model
assert all(models._MODEL_REGISTRY[k] is v for k,v in old.items())
load_user_extension_module()
print(json.dumps({'registry': 'PASS', 'role': os.environ['XVLA_TEST_ROLE']}))
'''
        for role in ("actor_hook", "rollout_hook", "checkpoint_hook"):
            env = dict(os.environ, RLINF_EXT_MODULE="xvla_rlinf", XVLA_TEST_ROLE=role,
                       PYTHONPATH=os.pathsep.join((str(RUNTIME), str(ROOT))))
            result = subprocess.run([sys.executable, "-c", code], cwd=RUNTIME, env=env, check=True, capture_output=True, text=True, timeout=120)
            self.assertIn('"registry": "PASS"', result.stdout)

    def test_pinned_validation_with_logical_single_gpu_and_no_ray_start(self):
        import rlinf.config as config
        self.extension.register()
        class LogicalSingleGpuCluster:
            num_nodes = 1
            num_accelerators = 1
            def __init__(self, *args, **kwargs): pass
        class LogicalSingleGpuPlacement:
            def __init__(self, *args, **kwargs): pass
            def get_world_size(self, component): return 1
        for name in ("ppo_smoke", "grpo_smoke", "native_eval_smoke", "stochastic_eval_smoke", "sft_smoke"):
            with self.subTest(name=name):
                cfg = compose_config(name)
                with patch.object(config, "Cluster", LogicalSingleGpuCluster), patch.object(config, "HybridComponentPlacement", LogicalSingleGpuPlacement):
                    validated = config.validate_cfg(cfg)
                self.assertEqual(validated.actor.model.model_type, "xvla")


if __name__ == "__main__":
    unittest.main(verbosity=2)
