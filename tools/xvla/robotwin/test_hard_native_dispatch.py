"""Check durable original-Easy delivery without contacting Slurm."""
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


# Production controllers run on Linux. Keep these accounting tests runnable
# on the Windows development checkout without emulating any real file lock.
_fcntl = (patch.dict(sys.modules, {"fcntl": SimpleNamespace(LOCK_EX=1, LOCK_NB=2, flock=Mock())})
          if os.name == "nt" else nullcontext())
with _fcntl:
    from tools.xvla.robotwin.hard_native_dispatch import HardNativeDispatch


class OriginalEasyDeliveryTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.dispatch = object.__new__(HardNativeDispatch)
        self.dispatch.root = root
        self.dispatch.path = root / "state.json"
        self.dispatch.state = {"created_at": 0, "native_jobs": []}
        self.dispatch.tagged_jobs = Mock(return_value=None)
        self.dispatch.repair_original_journals = Mock()
        self.original = SimpleNamespace(cfg=SimpleNamespace(hours=48),
                                        state={"jobs": {"extension:easy": ["20688", "20728"]}},
                                        journal_submission=Mock(return_value=None))

    def test_passes_explicit_helper_token_and_restores_environment(self):
        tokens = []

        def start(stage, lane, hours, retry):
            self.assertEqual((stage, lane, hours, retry), ("extension", "easy", 48, True))
            tokens.append(os.environ["FYP_ROBOTWIN_SUBMISSION_TOKEN"])
            self.assertEqual(os.environ["SBATCH_COMMENT"], "unrelated")
            self.original.state["jobs"]["extension:easy"].append("20901")
            return "20901"

        with patch.dict(os.environ, FYP_ROBOTWIN_SUBMISSION_TOKEN="previous", SBATCH_COMMENT="unrelated"):
            self.assertEqual(self.dispatch.deliver_original_extension(self.original, start), "20901")
            self.assertEqual(os.environ["FYP_ROBOTWIN_SUBMISSION_TOKEN"], "previous")
        self.assertRegex(tokens[0], r"^rw-hard-easy-[0-9a-f]{24}$")
        self.assertEqual(self.dispatch.state["original_resume_job"], "20901")
        self.assertEqual(self.dispatch.state["original_submission"]["known_history"], ["20688", "20728"])

    def test_failed_delivery_preserves_intent_and_refuses_duplicate(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "submission failed"):
                self.dispatch.deliver_original_extension(self.original, Mock(side_effect=RuntimeError("submission failed")))
            self.assertNotIn("FYP_ROBOTWIN_SUBMISSION_TOKEN", os.environ)
            second = Mock()
            with self.assertRaisesRegex(RuntimeError, "delivery is uncertain"):
                self.dispatch.deliver_original_extension(self.original, second)
            second.assert_not_called()
        self.assertIn("original_submission", self.dispatch.state)

    def test_tag_recovers_delivery_before_the_helper_journal(self):
        self.dispatch.state["original_submission"] = {
            "token": "rw-hard-easy-" + "a" * 24, "known_history": ["20688", "20728"],
        }
        self.dispatch.tagged_jobs.return_value = "20902"

        def repair(job):
            self.original.journal_submission.return_value = job

        self.dispatch.repair_original_journals.side_effect = repair
        start = Mock(return_value="20902")
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.dispatch.deliver_original_extension(self.original, start), "20902")
        self.dispatch.repair_original_journals.assert_called_once_with("20902")
        start.assert_called_once_with("extension", "easy", 48, retry=True)


if __name__ == "__main__":
    unittest.main()
