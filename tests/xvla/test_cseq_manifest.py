"""Validate Cseq manifest construction without running the simulator."""
from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

from xvla_rlinf import cseq_manifest as cseq
from xvla_rlinf.sft import SCHEMA, SOURCE_COMMIT, ConvertedLiberoDataset, sha256_file


class CseqManifestTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.output = root / "cseq" / "manifest.json"
        split = json.loads(cseq.SPLIT_PATH.read_text(encoding="utf-8"))
        self.paths = {}
        for task in split["tasks"][:3]:
            task_id = task["task_id"]
            directory = root / f"task{task_id}"
            directory.mkdir()
            episodes = []
            for index in range(10):
                demo_id = f"demo_{index}"
                path = directory / f"{demo_id}.hdf5"
                with h5py.File(path, "w") as data:
                    data.attrs.update(schema=SCHEMA, controller_replay_verified=True)
                    actions = np.zeros((11, 10), dtype=np.float32)
                    actions[:, 0] = np.arange(len(actions)) * .001
                    actions[:, 3] = actions[:, 7] = 1
                    data["abs_action_6d"] = actions
                    data["images/main"] = data["images/wrist"] = np.zeros((12, 4, 4, 3), dtype=np.uint8)
                episodes.append({
                    "task_suite": split["suite"], "task_id": task_id,
                    "task_name": task["task_name"], "language": task["runtime_language"],
                    "demo_id": demo_id, "path": path.name, "sha256": sha256_file(path), "frames": 11,
                    "controller_replay_verified": True, "success_final": True, "control_freq": 20,
                    "absolute_replay": {"success_final": True, "action_env_steps": 11, "control_freq": 20},
                })
            manifest = {
                "schema": SCHEMA, "source_commit": SOURCE_COMMIT,
                "absolute_source": "replayed_controller_goal_pos_goal_ori",
                "image_alignment": "post_action_drop_first",
                **cseq.CONVERSION_PROTOCOL,
                "source_hdf5": f'/data/{split["suite"]}/{task["task_name"]}_demo.hdf5',
                "source_sha256": cseq.RAW_SHA256[task_id], "data_revision": cseq.RAW_REVISION,
                "selection_seed": 1234, "requested_demonstrations": 10, "demonstrations": 10,
                "selected_demo_ids": [episode["demo_id"] for episode in episodes], "episodes": episodes,
            }
            self.paths[task_id] = directory / "manifest.json"
            self.paths[task_id].write_text(json.dumps(manifest), encoding="utf-8")
        self.enterContext(patch.object(cseq, "TASK0_MANIFEST_SHA256", sha256_file(self.paths[0])))

    def modify_task(self, task_id, modify):
        path = self.paths[task_id]
        manifest = json.loads(path.read_text(encoding="utf-8"))
        modify(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_builds_single_loader_compatible_manifest_and_preserves_sources(self):
        cseq.build_cseq_manifest(self.paths, self.output)
        saved = self.output.read_bytes()
        dataset = ConvertedLiberoDataset(self.output, horizon=30)
        self.assertEqual(len(dataset), 30)
        self.assertEqual(dataset.manifest["training_frames_before_static_filter"], 330)
        self.assertEqual(Counter(episode["task_id"] for episode in dataset.manifest["episodes"]),
                         Counter({0: 10, 1: 10, 2: 10}))
        self.assertEqual(dataset.manifest["split_sha256"], cseq.SPLIT_SHA256)
        for source in dataset.manifest["constituent_manifests"]:
            path = (self.output.parent / source["path"]).resolve()
            self.assertEqual(path, self.paths[source["task_id"]])
            self.assertEqual(sha256_file(path), source["sha256"])
        with self.assertRaises(FileExistsError):
            cseq.build_cseq_manifest(self.paths, self.output)
        self.assertEqual(self.output.read_bytes(), saved)

    def test_rejects_raw_source_mismatch(self):
        self.modify_task(1, lambda manifest: manifest.update(source_sha256=cseq.RAW_SHA256[2]))
        with self.assertRaisesRegex(ValueError, "pinned raw dataset"):
            cseq.build_cseq_manifest(self.paths, self.output)
        self.assertFalse(self.output.exists())

    def test_rejects_wrong_split_task(self):
        self.modify_task(1, lambda manifest: manifest["episodes"][0].update(language="wrong task"))
        with self.assertRaisesRegex(ValueError, "outside its frozen task"):
            cseq.build_cseq_manifest(self.paths, self.output)

    def test_requires_paired_absolute_replay(self):
        self.modify_task(1, lambda manifest: manifest["episodes"][0]["absolute_replay"].update(success_final=False))
        with self.assertRaisesRegex(ValueError, "paired controller replay"):
            cseq.build_cseq_manifest(self.paths, self.output)

    def test_rejects_altered_converted_file(self):
        path = self.paths[1].parent / "demo_0.hdf5"
        with path.open("ab") as stream:
            stream.write(b"modified")
        with self.assertRaisesRegex(ValueError, "hash differs"):
            cseq.build_cseq_manifest(self.paths, self.output)

    def test_requires_frozen_task0_manifest(self):
        self.modify_task(0, lambda manifest: manifest.update(slurm_job_id="changed"))
        with self.assertRaisesRegex(ValueError, "reuse the frozen"):
            cseq.build_cseq_manifest(self.paths, self.output)
