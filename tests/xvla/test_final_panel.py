from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.xvla import final_panel


class FinalPanelTests(unittest.TestCase):
    def test_frozen_panel_repeats_each_state_with_two_distinct_seed_pairs(self):
        repo = Path(__file__).resolve().parents[2]
        split = final_panel.read(repo / final_panel.PROTOCOL / "split.json")
        shards = final_panel.trial_panels(split)
        self.assertEqual([len(shard) for shard in shards], [50] * 5)
        self.assertEqual([shard[0]["task_id"] for shard in shards], [5, 6, 7, 8, 9])
        for shard in shards:
            self.assertEqual([row["trial_id"] for row in shard[::2]], list(range(5, 30)))
            for first, second in zip(shard[::2], shard[1::2], strict=True):
                self.assertEqual(first["initial_state_sha256"], second["initial_state_sha256"])
                self.assertEqual((first["repetition_index"], second["repetition_index"]), (0, 1))
                self.assertEqual(second["environment_seed"] - first["environment_seed"], 1000)
                self.assertEqual(second["policy_seed"] - first["policy_seed"], 1000)

    def test_refuses_release_before_formal_matrix_completes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root / final_panel.RUN_PROTOCOL
            folder.mkdir()
            final_panel.write(folder / "e3_dispatch.json", {
                "state": "RUNNING", "partition": "GPU", "runs": final_panel.schedule(),
            })
            with self.assertRaisesRegex(ValueError, "H remains sealed"):
                final_panel.build_plan(root, root, "GPU", "commit")
