"""Verify the Easy-first Slurm transition without submitting jobs."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.xvla.robotwin import dispatch


class EasyFirstTest(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.easy = {
            "first_batch": {"finished": 5000, "unresolved": 0},
            "cumulative": {"finished": 7000, "unresolved": 0},
        }
        self.runner = object.__new__(dispatch.Dispatch)
        self.runner.state = {"jobs": {"formal:easy": ["20325"], "formal:hard": ["20326"]}}
        self.runner.cfg = SimpleNamespace(hours=48)
        self.runner.save = lambda stage, **values: self.runner.state.update(values)
        self.runner.wait_for = self.wait_for
        self.runner.aggregate = lambda phase: {"overall": {"easy": self.easy}}
        self.enterContext(patch.object(dispatch, "output", side_effect=self.slurm))

    def wait_for(self, stage, lanes, hours):
        self.assertEqual(len(lanes), 1)
        self.assertEqual(hours, 48)
        self.events.append((stage, lanes[0]))

    def slurm(self, *args):
        if args == ("scontrol", "show", "job", "20326", "-o"):
            return "JobId=20326 JobName=rw-b0-formal-hard JobState=PENDING Priority=0 Reason=JobHeldUser"
        self.assertEqual(args, ("scontrol", "release", "20326"))
        self.events.append(("release", "20326"))
        return ""

    def test_finishes_both_easy_cohorts_before_releasing_hard(self):
        self.runner.run_easy_first()
        self.assertEqual(self.events, [
            ("formal", "easy"), ("extension", "easy"), ("release", "20326"),
            ("formal", "hard"), ("extension", "hard"),
        ])

    def test_incomplete_easy_first_batch_does_not_release_hard(self):
        self.easy["first_batch"]["finished"] = 4999
        with self.assertRaisesRegex(RuntimeError, "Easy first_batch incomplete"):
            self.runner.run_easy_first()
        self.assertEqual(self.events, [("formal", "easy")])

    def test_unresolved_easy_extension_does_not_release_hard(self):
        self.easy["cumulative"]["unresolved"] = 1
        with self.assertRaisesRegex(RuntimeError, "Easy cumulative incomplete"):
            self.runner.run_easy_first()
        self.assertEqual(self.events, [("formal", "easy"), ("extension", "easy")])

    def test_restart_after_easy_releases_hard_once(self):
        self.runner.state.update(easy_confirmed_complete=True, held_hard_job="20326")
        self.runner.run_easy_first()
        self.runner.run_easy_first()
        self.assertEqual(self.events, [
            ("release", "20326"), ("formal", "hard"), ("extension", "hard"),
            ("formal", "hard"), ("extension", "hard"),
        ])


if __name__ == "__main__":
    unittest.main()
