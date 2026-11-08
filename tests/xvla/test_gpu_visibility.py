"""Slurm visibility regression logic; no CUDA, EGL or Ray process is launched.

The tested source is reconstructed by applying the seven-file audited patch
to exact pinned Git sources. Driver/CUDA/EGL inventories are controlled inputs;
a real allocated-node check is still required after these tests pass.
"""
from __future__ import annotations

import ast
from functools import cache
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

ROOT = Path(__file__).resolve().parents[2]
NVIDIA_PATH = "rlinf/scheduler/hardware/accelerators/nvidia_gpu.py"
ALLOCATION_ENV = "RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES"


class GPUVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        subprocess.run(["git", "-C", str(cls.root), "init", "-q"], check=True, capture_output=True)
        manifest = json.loads((ROOT / "patches/rlinf-xvla.json").read_text())
        assert len(manifest["files"]) == 7
        for item in manifest["files"]:
            data = subprocess.run(["git", "-C", str(ROOT / "third_party/rlinf"), "show",
                                   manifest["upstream_commit"] + ":" + item["path"]],
                                  check=True, capture_output=True).stdout
            assert hashlib.sha256(data).hexdigest() == item["before_sha256"]
            destination = cls.root / item["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        subprocess.run(["git", "-C", str(cls.root), "apply", "--whitespace=error-all",
                        str(ROOT / "patches/rlinf-xvla.patch")], check=True, capture_output=True)
        for item in manifest["files"]:
            data = (cls.root / item["path"]).read_bytes().replace(b"\r\n", b"\n")
            assert hashlib.sha256(data).hexdigest() == item["after_sha256"]
        cls.source = (cls.root / NVIDIA_PATH).read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        # Load the exact patched functions, without importing hardware drivers.
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        tree = ast.parse(self.source)
        functions = {"_xvla_parse_cuda_tokens", "_xvla_initialize_cuda_visibility",
                     "_xvla_allocation_tokens", "_egl_index_by_cuda_device", "_renders_with_egl"}
        methods = {"get_num_devices", "get_visible_devices", "get_accelerator_env_var", "get_egl_device_id"}
        bodies = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions]
        original = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NvidiaGPUManager")
        bodies.append(ast.ClassDef(name="NvidiaGPUManager", bases=[], keywords=[], decorator_list=[],
                                  body=[node for node in original.body if isinstance(node, ast.FunctionDef) and node.name in methods]))
        self.inventory = Mock(return_value=4)
        self.egl = Mock(return_value={0: 7, 1: 2})
        self.ns = {"os": os, "cache": cache, "logger": logging.getLogger(__name__), "warnings": warnings,
                   "_XVLA_ALLOCATION_ENV": ALLOCATION_ENV, "EGL_DEVICE_ID_ENV_VARS": ("MUJOCO_EGL_DEVICE_ID", "EGL_DEVICE_ID"),
                   "_CPU_RENDERING_BACKENDS": frozenset({"osmesa", "glx"}),
                   "NvidiaGPUAcceleratorManager": SimpleNamespace(get_current_node_num_accelerators=self.inventory),
                   "_query_egl_index_by_cuda_ordinal": self.egl, "_torch_needs_avoid_record_streams": lambda: False}
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *bodies], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), "<patched-slurm-nvidia>", "exec"), self.ns)
        self.manager = self.ns["NvidiaGPUManager"]

    def isolated(self, mask):
        os.environ.update(RLINF_XVLA_ISOLATED_RAY="1", CUDA_VISIBLE_DEVICES=mask)
        self.ns["_xvla_initialize_cuda_visibility"]()

    def test_default_enumeration_and_numeric_masks_are_unchanged(self):
        self.assertEqual(self.manager.get_num_devices(), 4)
        self.inventory.assert_called_once()
        os.environ["CUDA_VISIBLE_DEVICES"] = "3,1"
        self.assertEqual(self.manager.get_visible_devices(), [3, 1])
        self.ns["_xvla_initialize_cuda_visibility"]()
        self.assertNotIn(ALLOCATION_ENV, os.environ)
        self.assertNotIn("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", os.environ)
        os.environ["MUJOCO_GL"] = "osmesa"
        self.assertEqual(self.manager.get_accelerator_env_var(["2"])["CUDA_VISIBLE_DEVICES"], "2")

    def test_one_slurm_gpu_never_becomes_four_nvml_gpus(self):
        self.isolated("0")
        self.assertEqual(self.manager.get_num_devices(), 1)
        self.inventory.assert_not_called()
        self.assertEqual(self.manager.get_visible_devices(), [0])
        self.assertEqual(os.environ[ALLOCATION_ENV], "0")
        self.assertEqual(os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"], "1")

    def test_nonzero_discontiguous_tokens_map_to_scheduler_ordinals_and_egl(self):
        self.isolated("3,1")
        self.assertEqual(self.manager.get_num_devices(), 2)
        self.assertEqual(self.manager.get_visible_devices(), [0, 1])
        worker = self.manager.get_accelerator_env_var(["1"])
        self.assertEqual(worker["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(worker[ALLOCATION_ENV], "3,1")
        self.assertEqual(worker["MUJOCO_EGL_DEVICE_ID"], "2")
        self.assertEqual(worker["EGL_DEVICE_ID"], "2")
        os.environ.update(worker)
        self.assertEqual(self.manager.get_visible_devices(), [1])
        self.assertEqual(self.manager.get_num_devices(), 2)  # cluster allocation, not this worker's subset
        self.assertEqual(["global0", "global1"][self.manager.get_visible_devices()[0]], "global1")

    def test_uuid_mask_is_preserved_and_does_not_require_integer_cast(self):
        self.isolated("GPU-aaaa1111-bbbb,GPU-cccc2222-dddd")
        self.assertEqual(self.manager.get_num_devices(), 2)
        self.assertEqual(self.manager.get_visible_devices(), [0, 1])
        worker = self.manager.get_accelerator_env_var(["1", "0"])
        self.assertEqual(worker["CUDA_VISIBLE_DEVICES"], "GPU-cccc2222-dddd,GPU-aaaa1111-bbbb")
        self.assertEqual(worker["MUJOCO_EGL_DEVICE_ID"], "2")
        os.environ.update(worker)
        self.assertEqual(self.manager.get_visible_devices(), [1, 0])

    def test_worker_local_egl_query_maps_local_zero_back_to_allocation_id(self):
        self.isolated("3,1")
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        self.egl.return_value = {0: 9}
        self.assertEqual(self.manager.get_visible_devices(), [1])
        self.assertEqual(self.manager.get_egl_device_id(1), 9)
        self.assertIsNone(self.manager.get_egl_device_id(0))

    def test_empty_unset_and_disabled_masks_fail_closed(self):
        for value in (None, "", "-1", "NoDevFiles"):
            with self.subTest(value=value), patch.dict(os.environ, {"RLINF_XVLA_ISOLATED_RAY": "1"}, clear=True):
                if value is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = value
                self.assertEqual(self.manager.get_num_devices(), 0)
                self.assertEqual(self.manager.get_visible_devices(), [])
                self.assertEqual(self.manager.get_accelerator_env_var([])["CUDA_VISIBLE_DEVICES"], "")
        self.inventory.assert_not_called()

    def test_out_of_allocation_and_invalid_placements_are_rejected(self):
        self.isolated("2,3")
        for placement in (["2"], ["-1"], ["0", "0"], ["GPU-aaaa"], ["01"]):
            with self.subTest(placement=placement), self.assertRaises(ValueError):
                self.manager.get_accelerator_env_var(placement)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        with self.assertRaisesRegex(ValueError, "exceeds"):
            self.manager.get_visible_devices()

    def test_invalid_original_masks_are_rejected_without_nvml_fallback(self):
        for mask in ("0,0", "0,,1", "0,-1", "GPU-not-a-uuid", "all"):
            with self.subTest(mask=mask), patch.dict(os.environ, {"RLINF_XVLA_ISOLATED_RAY": "1", "CUDA_VISIBLE_DEVICES": mask}, clear=True):
                with self.assertRaises(ValueError):
                    self.manager.get_num_devices()
        self.inventory.assert_not_called()

    def test_egl_mapping_failure_is_rejected_instead_of_guessing_physical_gpu(self):
        self.isolated("3")
        self.egl.return_value = {}
        with self.assertRaisesRegex(RuntimeError, "unverified"):
            self.manager.get_accelerator_env_var(["0"])
        self.ns["_egl_index_by_cuda_device"].cache_clear()
        self.egl.side_effect = OSError("no EGL extension")
        with self.assertRaisesRegex(RuntimeError, "unverified"):
            self.manager.get_accelerator_env_var(["0"])

    def test_cpu_rendering_does_not_attempt_an_egl_gpu(self):
        self.isolated("3")
        os.environ["MUJOCO_GL"] = "osmesa"
        worker = self.manager.get_accelerator_env_var(["0"])
        self.assertEqual(worker["CUDA_VISIBLE_DEVICES"], "3")
        self.assertNotIn("MUJOCO_EGL_DEVICE_ID", worker)
        self.egl.assert_not_called()


if __name__ == "__main__":
    unittest.main()
