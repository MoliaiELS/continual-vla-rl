"""Verify the Easy-first Slurm transition without submitting jobs."""

from pathlib import Path
import json
from subprocess import CalledProcessError
from tempfile import TemporaryDirectory
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
        self.runner.pilot_root = None
        self.runner.paused_easy_job = None
        self.runner.parallel_easy = False
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


class ResumePilotTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.runner = object.__new__(dispatch.Dispatch)
        self.runner.state = {}
        self.runner.cfg = SimpleNamespace(partition="GPU")
        self.runner.pilot_root = root / dispatch.PILOT_COMMIT
        self.runner.evaluation_root = root / dispatch.RECOVERY_COMMIT
        self.runner.data = root / "robotwin-b0"
        self.runner.save = lambda stage, **values: self.runner.state.update(values)
        self.releases = []
        self.runner.wait_for_slot_release = self.releases.append
        self.runner.accounting = lambda job: ("COMPLETED", "0:0")
        self.enterContext(patch.object(dispatch, "output", return_value=dispatch.PILOT_COMMIT))

    def test_pilot_uses_separate_output_and_waits_for_both_slots(self):
        with patch.object(dispatch.subprocess, "run", return_value=SimpleNamespace(stdout="ROBOTWIN_JOB_ID=20399")) as launch:
            self.runner.run_resume_pilot("20325")
            self.runner.run_resume_pilot("20325")
        self.assertEqual(self.releases, ["20325", "20399"])
        self.assertEqual(self.runner.state["easy_pilot_result"],
                         {"job": "20399", "state": "COMPLETED", "exit_code": "0:0"})
        launch.assert_called_once()
        env = launch.call_args.kwargs["env"]
        self.assertEqual(Path(env["FYP_ROBOTWIN_DATA"]), self.runner.data.with_name("robotwin-native"))
        self.assertEqual(Path(env["FYP_ROBOTWIN_INPUT_ROOT"]), self.runner.data)
        self.assertEqual(Path(env["FYP_ROBOTWIN_QUALIFIED_DATA"]), self.runner.data)

    def test_failed_pilot_submission_does_not_hold_formal_resume(self):
        error = CalledProcessError(1, ["sbatch"], stderr="resource unavailable")
        with patch.object(dispatch.subprocess, "run", side_effect=error):
            self.runner.run_resume_pilot("20325")
        self.assertEqual(self.runner.state["easy_pilot_result"], {"error": "resource unavailable"})
        self.assertEqual(self.releases, ["20325"])

    def test_planned_cancel_runs_pilot_before_replacement(self):
        events = []
        self.runner.paused_easy_job = "20325"
        self.runner.accounting = lambda job: ("CANCELLED", "0:0") if job == "20325" else ("COMPLETED", "0:0")
        self.runner.run_resume_pilot = lambda job: events.append(("pilot", job))
        def start(stage, lane, hours, retry=False):
            if retry:
                events.append(("retry", lane))
                return "20400"
            return "20325"
        self.runner.start = start
        with patch.object(dispatch.time, "sleep"):
            result = self.runner.wait_for("formal", ("easy",), 48)
        self.assertEqual(events, [("pilot", "20325"), ("retry", "easy")])
        self.assertEqual(result, {"easy": ("COMPLETED", "0:0")})


class ParallelEasyTest(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        self.runner = object.__new__(dispatch.Dispatch)
        self.runner.state = {"jobs": {}, "easy_pilot_result": {"job": "20569", "state": "COMPLETED", "exit_code": "0:0"}}
        self.runner.cfg = SimpleNamespace(partition="GPU", gpu="4090")
        self.runner.code = dispatch.RECOVERY_COMMIT
        self.runner.easy_first = self.runner.recover_prescreen = self.runner.parallel_easy = True
        self.runner.data = root / "robotwin-b0"
        self.runner.evaluation_root = root / dispatch.RECOVERY_COMMIT
        self.runner.evaluation_tools = self.runner.evaluation_root / "tools/xvla/robotwin"
        self.runner.save = lambda stage, **values: self.runner.state.update(values)
        self.summary = root / "robotwin-native/jobs/20569/concurrency-pilot/summary.json"
        self.summary.parent.mkdir(parents=True)

    def test_parallel_submission_only_for_easy(self):
        with patch.object(dispatch, "output", return_value="ROBOTWIN_JOB_ID=20573") as call:
            self.runner.start("formal", "easy", 48)
            self.runner.start("formal", "hard", 48)
        self.assertEqual(call.call_args_list[0].args[-3:], ("", str(self.runner.evaluation_root), "2"))
        self.assertEqual(call.call_args_list[1].args[-2:], ("", str(self.runner.evaluation_root)))

    def test_rejects_a_pilot_with_changed_outcomes(self):
        pilot = dict(job_id="20569", evaluation_commit=dispatch.RECOVERY_COMMIT, complete=True,
                     outcome_changes=[], serial_over_concurrent=1.4158)
        self.summary.write_text(json.dumps(pilot))
        self.runner.qualify_parallel_easy()
        pilot["outcome_changes"] = ["different-seed-outcome"]
        self.summary.write_text(json.dumps(pilot))
        with self.assertRaisesRegex(ValueError, "did not qualify"):
            self.runner.qualify_parallel_easy()


if __name__ == "__main__":
    unittest.main()
