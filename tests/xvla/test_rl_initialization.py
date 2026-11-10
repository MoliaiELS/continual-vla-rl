"""Fail closed on mismatched SFT lineage and incomplete policy state."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from xvla_rlinf.rl_initialization import apply_sft_state, read_sft_source, validate_rl_initialization
from xvla_rlinf.sft import PT_ID, PT_REVISION


class PolicyState:
    def __init__(self, add_value_head):
        self.add_value_head = add_value_head
        self.tensors = {"core.vlm.image_projection": np.zeros((2, 3), dtype=np.float32),
                        "core.domain.frozen_weight": np.zeros(2, dtype=np.float32),
                        "sampler.std": np.full(10, 0.04, dtype=np.float32)}
        if add_value_head:
            self.tensors["value_head.0.weight"] = np.ones((1, 3), dtype=np.float32)
        self.loaded = None

    def state_dict(self):
        return self.tensors

    def load_state_dict(self, state, *, strict):
        if not strict or set(state) != set(self.tensors):
            raise AssertionError("The bridge must load one exact complete state dict")
        self.loaded = state


class PolicyStateTests(unittest.TestCase):
    def test_copies_core_state_and_preserves_the_rl_sampler_and_fresh_head(self):
        for add_value_head in (True, False):
            with self.subTest(add_value_head=add_value_head):
                model = PolicyState(add_value_head)
                source = {key: tensor + 2 for key, tensor in model.tensors.items() if key.startswith("core.")}
                source["sampler.std"] = np.full(10, 0.003, dtype=np.float32)
                head = apply_sft_state(model, source)
                self.assertEqual(head, ["value_head.0.weight"] if add_value_head else [])
                for name in source.keys() - {"sampler.std"}:
                    self.assertIs(model.loaded[name], source[name])
                self.assertIs(model.loaded["sampler.std"], model.tensors["sampler.std"])
                for name in head:
                    self.assertIs(model.loaded[name], model.tensors[name])

    def test_rejects_missing_extra_shape_or_dtype_before_load(self):
        for change in ("missing", "missing_sampler", "extra_head", "shape", "dtype", "sampler_shape"):
            with self.subTest(change=change):
                model = PolicyState(True)
                source = {key: tensor.copy() for key, tensor in model.tensors.items()
                          if not key.startswith("value_head.")}
                key = "core.vlm.image_projection"
                if change == "missing":
                    del source[key]
                elif change == "missing_sampler":
                    del source["sampler.std"]
                elif change == "extra_head":
                    source["value_head.0.weight"] = model.tensors["value_head.0.weight"]
                elif change == "shape":
                    source[key] = source[key].reshape(3, 2)
                elif change == "sampler_shape":
                    source["sampler.std"] = source["sampler.std"][:3]
                else:
                    source[key] = source[key].astype(np.float64)
                with self.assertRaisesRegex(ValueError, "keys differ|shape/dtype"):
                    apply_sft_state(model, source)
                self.assertIsNone(model.loaded)


class InitializationReceiptTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        self.base = root / "pt"
        self.base.mkdir()
        self.checkpoint = root / "actor"
        self.weights = self.checkpoint / "model_state_dict" / "full_weights.pt"
        self.weights.parent.mkdir(parents=True)
        self.weights.touch()
        self.source_cfg = self.model_config(add_value_head=False)
        self.evidence = {"model_config": self.source_cfg, "digest": "a" * 64}
        self.check = {"status": "PASS", "strict_state_load": True, "digest": "a" * 64,
                      "exported_digest": "a" * 64, "model_config": self.source_cfg}
        (self.base / "verified.json").write_text(json.dumps({"repo_id": PT_ID, "revision": PT_REVISION}))
        self.write_audit()

    def model_config(self, add_value_head, initialize=False):
        return {"model_type": "xvla", "model_path": str(self.base), "precision": "fp32", "is_lora": False,
                "num_action_chunks": 30, "action_dim": 7, "add_value_head": add_value_head,
                "xvla": {"update_scope": "prompt_heads", "domain_rows": [3], "domain_id": 3, "steps": 10,
                         "initial_sft_checkpoint": str(self.checkpoint) if initialize else None}}

    def write_audit(self):
        (self.checkpoint / "xvla_evidence.json").write_text(json.dumps(self.evidence))
        (self.checkpoint / "checkpoint_check.json").write_text(json.dumps(self.check))

    def cfg(self, *, initialize=True, **runner):
        return SimpleNamespace(actor=SimpleNamespace(model=self.model_config(True, initialize)),
                               rollout=SimpleNamespace(model=self.model_config(True, initialize)),
                               runner={"require_sft_initialization": True, **runner})

    def test_audited_pt_source_passes_with_a_new_rl_value_head(self):
        evidence, weights, _ = read_sft_source(self.checkpoint, self.model_config(True))
        self.assertEqual(evidence, self.evidence)
        self.assertEqual(weights, self.weights)
        validate_rl_initialization(self.cfg())

    def test_rejects_audit_digest_or_configuration_disagreement(self):
        self.check["exported_digest"] = "b" * 64
        self.write_audit()
        with self.assertRaisesRegex(ValueError, "independent strict"):
            read_sft_source(self.checkpoint, self.model_config(False))

    def test_rejects_different_base_or_domain_layout(self):
        different_base = self.model_config(False)
        different_base["model_path"] += "-other"
        with self.assertRaisesRegex(ValueError, "same pinned Pt base"):
            read_sft_source(self.checkpoint, different_base)
        different_domain = self.model_config(False)
        different_domain["xvla"]["domain_rows"] = [4]
        with self.assertRaisesRegex(ValueError, "prompt-head layout"):
            read_sft_source(self.checkpoint, different_domain)

    def test_fresh_requires_checkpoint_and_resume_excludes_it(self):
        with self.assertRaisesRegex(ValueError, "fresh Pt RL job requires"):
            validate_rl_initialization(self.cfg(initialize=False))
        with self.assertRaisesRegex(ValueError, "DCP resume requires"):
            validate_rl_initialization(self.cfg(resume_dir="checkpoint"))
        validate_rl_initialization(self.cfg(initialize=False, resume_dir="checkpoint"))
        with self.assertRaisesRegex(ValueError, "cannot also load"):
            validate_rl_initialization(self.cfg(ckpt_path="full_weights.pt"))

    def test_rejects_different_actor_and_rollout_checkpoint_paths(self):
        cfg = self.cfg()
        cfg.rollout.model["xvla"]["initial_sft_checkpoint"] = None
        with self.assertRaisesRegex(ValueError, "same SFT initialization"):
            validate_rl_initialization(cfg)


if __name__ == "__main__":
    unittest.main()
