"""Checkpoint-checker boundary tests, not real model/DCP restoration evidence."""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from xvla_rlinf import checkpoint_check


class CheckpointBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = {"model_config": {"model_type": "xvla", "precision": "fp32", "is_lora": False,
                                          "xvla": {"update_scope": "prompt_heads", "domain_rows": [3]}},
                         "digest": "a" * 64}
        directory = self.root / "model_state_dict"
        directory.mkdir()
        # Metadata tests never deserialize this placeholder as Torch weights.
        (directory / "full_weights.pt").write_bytes(b"metadata-test-placeholder")
        self.write()

    def write(self):
        (self.root / "xvla_evidence.json").write_text(json.dumps(self.evidence), encoding="utf-8")

    def test_exact_config_and_optional_no_probe_are_preserved(self):
        actual, weights = checkpoint_check.read_evidence(self.root)
        self.assertEqual(actual, self.evidence)
        self.assertEqual(weights, self.root / "model_state_dict/full_weights.pt")
        self.assertNotIn("probe", actual)

    def test_wrong_model_precision_or_factory_lora_is_rejected(self):
        original = deepcopy(self.evidence)
        for field, value in (("model_type", "openpi"), ("precision", "bf16"), ("is_lora", True)):
            with self.subTest(field=field):
                self.evidence = deepcopy(original)
                self.evidence["model_config"][field] = value
                self.write()
                with self.assertRaises(ValueError):
                    checkpoint_check.read_evidence(self.root)

    def test_missing_export_or_invalid_digest_is_rejected(self):
        for invalid in (None, "a" * 63, "z" * 64):
            with self.subTest(invalid=invalid):
                self.evidence["digest"] = invalid
                self.write()
                with self.assertRaises(ValueError):
                    checkpoint_check.read_evidence(self.root)
        self.evidence["digest"] = "a" * 64
        self.write()
        (self.root / "model_state_dict/full_weights.pt").unlink()
        with self.assertRaises(FileNotFoundError):
            checkpoint_check.read_evidence(self.root)

    def test_probe_compares_shapes_finiteness_and_numerical_values(self):
        before = {"raw": [[[.1, .2]]], "actions": [[[.3, .4]]], "value": [[.5]]}
        after = deepcopy(before)
        after["raw"][0][0][0] += 1e-7
        comparison = checkpoint_check.compare_probe(before, after)
        self.assertEqual(set(comparison), set(before))
        self.assertGreater(comparison["raw"]["max_abs_difference"], 0)
        after["raw"][0][0][0] += .1
        with self.assertRaises(AssertionError):
            checkpoint_check.compare_probe(before, after)
        for invalid in ([[float("nan")]], [[float("inf")]], [[]], []):
            with self.subTest(invalid=invalid):
                after = deepcopy(before)
                after["value"] = invalid
                with self.assertRaises(ValueError):
                    checkpoint_check.compare_probe(before, after)

    def test_cli_failure_writes_fail_json_and_returns_nonzero(self):
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            code = checkpoint_check.main(["--checkpoint-dir", str(self.root)])
        self.assertEqual(code, 1)
        result = json.loads((self.root / "checkpoint_check.json").read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Slurm", result["error"])
        self.assertFalse(result["optimizer_resume_checked"])
        self.assertFalse(result["distributed_worker_resume_checked"])
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_launcher_and_registration_are_required_before_loading_torch(self):
        for env in ({"SLURM_JOB_ID": "123"},
                    {"SLURM_JOB_ID": "123", "RLINF_XVLA_ISOLATED_RAY": "1", "RLINF_EXT_MODULE": "other"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    checkpoint_check.assert_context()


if __name__ == "__main__":
    unittest.main()
