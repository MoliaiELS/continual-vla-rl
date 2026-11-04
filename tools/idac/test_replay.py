"""Eligibility, bounded-memory, preprocessing and no-signal update checks.

Run with the project Python on a CPU. HDF5 fixture tests additionally need
numpy/h5py; the dependency-free policy tests also run on local Python.
"""

import ast
from contextlib import redirect_stdout
import io
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
_spec = importlib.util.spec_from_file_location("fyp_replay", REPO / "rlinf/custom/fyp_replay.py")
_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_replay)
for _name in ("FORMAT", "ManifestReplayMemory", "load_manifest", "memory_summary",
              "non_noop_indices", "raw_gripper_to_model", "select_memory", "sha256_file"):
    globals()[_name] = getattr(_replay, _name)

try:
    import h5py
    import numpy as np
except ImportError:
    h5py = np = None


def entry(task, length=30, file_path=None):
    return {"task_id": task, "task_description": f"task {task}",
            "file": str(file_path or Path(tempfile.gettempdir()).resolve() / f"absent-replay-{task}.hdf5"),
            "sha256": "0" * 64, "file_size": 0, "demo": "demo_0",
            "num_steps": length, "step_indices": list(range(length))}


def manifest(*entries):
    return {"format": FORMAT, "suite": "libero_spatial", "demos": list(entries),
            "preprocessing": {"image_rotation_degrees": 180,
                              "action_transform": "oxe_libero_raw_gripper",
                              "noop_filter": "openvla_oft_prev_kept_1e-4"}}


def actor_probe(namespace):
    path = REPO / "rlinf/workers/actor/fsdp_actor_worker.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    actor = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "EmbodiedFSDPActor")
    methods = [node for node in actor.body if isinstance(node, ast.FunctionDef)
               and node.name in {"set_replay_tasks", "run_training"}]
    module = ast.Module(body=[ast.ClassDef(name="Probe", bases=[], keywords=[],
                                         body=methods, decorator_list=[])], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Probe"]


class AttrDict(dict):
    __getattr__ = dict.__getitem__


class MemoryPolicyTests(unittest.TestCase):
    def test_capacity_balanced_deterministic_and_no_future_file_access(self):
        data = manifest(entry(0), entry(1), entry(2))
        selected = select_memory(data, [0, 1], [0, 1, 2], 9, 8, 42)
        self.assertEqual(len(selected), 9)
        self.assertEqual(selected, select_memory(data, [1, 0], [0, 1, 2], 9, 8, 42))
        counts = memory_summary(data, selected, 8)["chunks_per_task"]
        self.assertEqual(counts, {"0": 5, "1": 4})
        self.assertNotIn(2, {data["demos"][index]["task_id"] for index, _ in selected})
        self.assertNotEqual(selected, select_memory(data, [0, 1], [0, 1, 2], 9, 8, 43))

    def test_empty_first_stage_does_not_touch_any_hdf5(self):
        memory = ManifestReplayMemory(manifest(entry(0), entry(1)), [], [0, 1], 8, 8, 0)
        self.assertEqual(len(memory), 0)
        self.assertEqual(memory.file_handles, {})

    def test_ineligible_task_and_too_small_capacity_fail_closed(self):
        data = manifest(entry(0), entry(1))
        with self.assertRaises(ValueError):
            select_memory(data, [1], [0], 8, 8, 0)
        with self.assertRaises(ValueError):
            select_memory(data, [0, 1], [0, 1], 1, 8, 0)

    def test_new_task_only_evicts_old_chunks_never_resurrects_unretained_data(self):
        data = manifest(entry(0, 100), entry(1, 100), entry(2, 100))
        first = set(select_memory(data, [0], [0, 1, 2], 20, 8, 42))
        second = set(select_memory(data, [0, 1], [0, 1, 2], 20, 8, 42))
        third = set(select_memory(data, [0, 1, 2], [0, 1, 2], 20, 8, 42))
        self.assertTrue({item for item in second if item[0] == 0} <= first)
        self.assertTrue({item for item in third if item[0] in (0, 1)} <= second)

    def test_manifest_rejects_foreign_tasks_before_opening_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(json.dumps(manifest(entry(0), entry(9))), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                load_manifest(path, [0, 1])

    def test_noop_filter_matches_bundled_reference(self):
        path = REPO / "openvla-oft/experiments/robot/libero/regenerate_libero_dataset.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "is_noop")
        namespace = {"np": SimpleNamespace(linalg=SimpleNamespace(
            norm=lambda values: math.sqrt(sum(float(value) ** 2 for value in values))))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                     str(path), "exec"), namespace)
        actions = [[0, 0, 0, 0, 0, 0, -1], [0.1, 0, 0, 0, 0, 0, -1],
                   [0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 1],
                   [0, 0, 0, 0, 0, 0, 1], [0.2, 0, 0, 0, 0, 0, 1]]
        kept = []
        for index, action in enumerate(actions):
            previous = actions[kept[-1]] if kept else None
            if not namespace["is_noop"](action, previous):
                kept.append(index)
        self.assertEqual(non_noop_indices(actions), kept)
        self.assertEqual(kept, [1, 3, 5])
        self.assertEqual([raw_gripper_to_model(x) for x in [-1, 0, 1]], [1.0, 1.0, 0.0])

    def test_actor_rejects_future_task_even_when_in_experiment_allowlist(self):
        Probe = actor_probe({"json": json})
        probe = Probe()
        probe.use_experience_replay = True
        probe._manifest_replay = True
        probe._world_size = 1
        probe.cfg = SimpleNamespace(
            algorithm=AttrDict(replay_allowed_task_ids=[0, 1, 2]),
            runner=AttrDict(fyp_experiment=AttrDict(task_ids=[0, 1, 2], method="seq_er")))
        with self.assertRaisesRegex(ValueError, "admission"):
            probe.set_replay_tasks([0, 1], stage=2)

    def test_formal_all_failed_group_cannot_apply_weight_decay_or_bc(self):
        Probe = actor_probe({"json": json})
        probe = Probe()
        zero_calls = []
        probe.cfg = SimpleNamespace(runner=AttrDict(fyp_experiment={"method": "seq_er"}))
        probe._fyp_no_effective_group = True
        probe.optimizer = SimpleNamespace(zero_grad=lambda: zero_calls.append(True),
                                          param_groups=[{"lr": 2e-5}])
        with redirect_stdout(io.StringIO()):
            metrics = probe.run_training()
        self.assertEqual(metrics["fyp_grpo_optimizer_steps"], 0)
        self.assertEqual(metrics["fyp_grpo_skipped"], 1)
        self.assertEqual(metrics["bc/examples"], 0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))
        self.assertEqual(zero_calls, [True])


@unittest.skipIf(h5py is None, "HDF5 fixture requires numpy and h5py")
class HDF5MemoryTests(unittest.TestCase):
    def test_selected_source_hash_orientation_action_chunk_and_future_exclusion(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "task.hdf5"
            actions = np.zeros((12, 7), dtype=np.float32)
            actions[:, 0] = np.arange(12)
            actions[:, -1] = np.where(np.arange(12) % 2, 1, -1)
            images = np.arange(12 * 2 * 3 * 3, dtype=np.uint8).reshape(12, 2, 3, 3)
            with h5py.File(source, "w") as stream:
                demo = stream.create_group("data/demo_0")
                demo.create_dataset("actions", data=actions)
                demo.create_dataset("obs/agentview_rgb", data=images)
            selected_entry = entry(0, 12, source)
            selected_entry.update(sha256=sha256_file(source), file_size=source.stat().st_size,
                                  step_indices=[0, 2, 3, 4, 5, 6, 8, 9, 10, 11])
            memory = ManifestReplayMemory(manifest(selected_entry, entry(1)), [0], [0, 1], 2, 8, 0)
            sample = memory[0]
            _, start = memory.selected[0]
            steps = selected_entry["step_indices"][start:start + 8]
            np.testing.assert_array_equal(sample["obs_rgb"], images[steps[0]][::-1, ::-1])
            np.testing.assert_array_equal(sample["actions"][:, 0], actions[steps, 0])
            np.testing.assert_array_equal(sample["actions"][:, -1],
                                          1 - np.clip(actions[steps, -1], 0, 1))
            self.assertEqual(sample["replay_task_id"], 0)
            self.assertGreater(memory.summary["retained_sample_payload_bytes"], 0)
            memory.close()
            selected_entry["sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "SHA256"):
                ManifestReplayMemory(manifest(selected_entry), [0], [0], 2, 8, 0)


if __name__ == "__main__":
    unittest.main()
