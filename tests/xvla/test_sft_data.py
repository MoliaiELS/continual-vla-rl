"""Schema and independent-demo checks; mocks never count as simulator evidence."""
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from xvla_rlinf.sft import (
    SCHEMA, SOURCE_COMMIT, ConvertedLiberoDataset, load_manifest, sha256_file,
)


def fixture(root: Path, *, steps=25, static=False, images_extra=1):
    data = root / "converted.hdf5"
    action = np.zeros((steps, 10), dtype=np.float32)
    if not static:
        action[:, 0] = np.arange(steps) * .001
    action[:, 3] = action[:, 7] = 1
    action[:, 9] = -1
    with h5py.File(data, "w") as stream:
        stream.attrs.update(schema=SCHEMA, controller_replay_verified=True)
        stream.create_dataset("abs_action_6d", data=action)
        pixels = np.zeros((steps + images_extra, 4, 4, 3), dtype=np.uint8)
        pixels[:, 0, 0, 0] = np.arange(len(pixels))
        stream.create_dataset("images/main", data=pixels)
        stream.create_dataset("images/wrist", data=pixels)
    episode = {"path": data.name, "sha256": sha256_file(data), "task_suite": "libero_spatial",
               "task_id": 0, "demo_id": "demo_1", "frames": steps, "language": "put bowl on plate",
               "controller_replay_verified": True, "success_final": True}
    manifest = {"schema": SCHEMA, "source_commit": SOURCE_COMMIT,
                "absolute_source": "replayed_controller_goal_pos_goal_ori",
                "image_alignment": "post_action_drop_first", "demonstrations": 1,
                "episodes": [episode]}
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


class SftDataTests(unittest.TestCase):
    def test_demo_count_is_not_window_count_and_image_drop_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            path, _ = fixture(Path(folder))
            dataset = ConvertedLiberoDataset(path, horizon=30)
            self.assertEqual(dataset.manifest["demonstrations"], 1)
            self.assertEqual(len(dataset), 15)  # official range(T-10), not 15 demos
            sample = dataset[0]
            self.assertEqual(sample["action"].shape, (30, 20))
            self.assertEqual(sample["proprio"].shape, (20,))
            self.assertEqual(sample["images"][0][0, 0, 0], 1)
            self.assertTrue(np.all(sample["action"][:, 10:] == 0))
            self.assertTrue(np.all(sample["action"][:, 9] == 0))

    def test_raw_relative_hdf5_and_unverified_provenance_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manifest = fixture(Path(folder))
            with self.assertRaises(ValueError):
                load_manifest(path.with_suffix(".hdf5"))
            manifest["absolute_source"] = "integrated_relative_deltas"
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_failed_duplicate_or_tampered_demo_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, manifest = fixture(Path(folder))
            manifest["episodes"][0]["success_final"] = False
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path)
            manifest["episodes"][0]["success_final"] = True
            manifest["episodes"] *= 2
            manifest["demonstrations"] = 2
            path.write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_manifest(path)
            manifest["episodes"] = manifest["episodes"][:1]
            manifest["demonstrations"] = 1
            path.write_text(json.dumps(manifest))
            with (Path(folder) / "converted.hdf5").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaises(ValueError):
                load_manifest(path)

    def test_bad_alignment_and_all_static_episodes_rejected(self):
        for kwargs in ({"images_extra": 0}, {"static": True}):
            with tempfile.TemporaryDirectory() as folder:
                path, _ = fixture(Path(folder), **kwargs)
                with self.assertRaises(ValueError):
                    ConvertedLiberoDataset(path, horizon=30)


if __name__ == "__main__":
    unittest.main()
