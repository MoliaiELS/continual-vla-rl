"""Check cohort, retry, and protocol accounting without simulation or model weights."""

import csv
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.xvla.robotwin.summarize import emit_csv, emit_markdown, summarize, wilson


class SummaryTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.tasks = self.root / "tasks.json"
        self.ids = ["move_stapler_pad", "turn_switch", *(f"task_{i}" for i in range(48))]
        self.write_json(self.tasks, {
            "robotwin_commit": "e" * 40,
            "priority_tasks": self.ids[:10],
            "tasks": [{
                "task_id": task,
                "description_zh": "任务说明",
                "prompt": task.replace("_", " "),
                "category": "主类别",
                "tags": ["标签"],
                "success_zh": "成功判定",
                "source_file": f"envs/{task}.py",
                "expert_line": 10,
                "success_line": 20,
            } for task in self.ids],
        })

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def append(path, row):
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    def shard(self, task, condition="easy", phase="formal"):
        context = {
            "task": task,
            "condition": condition,
            "phase": phase,
            "protocol": "official-v1",
            "protocol_fingerprint": "c" * 64,
        }
        shard = self.root / phase / task / condition
        shard.mkdir(parents=True)
        seeds = []
        limit = 5 if phase == "smoke" else (300 if task in self.ids[:10] else 100)
        for index in range(limit):
            row = {
                "episode_id": f"{phase}:{task}:{condition}:{index}",
                "requested_seed": index + 1000,
                "actual_seed": index + 1000,
                "policy_seed": index + 2000,
                "batch": index // 100 + 1,
                "index": index,
                "scene_signature": f"{index:064x}",
            }
            self.append(shard / "prescreen.jsonl", {
                **context, "status": "accepted", "attempt": 1,
                "requested_seed": row["requested_seed"], "actual_seed": row["actual_seed"],
                "scene_signature": row["scene_signature"],
            })
            seeds.append(row)
        manifest = {**context, "limit": limit, "episodes": seeds}
        self.write_json(shard / "seed_manifest.json", manifest)
        return shard, manifest, [{
            **context, **row,
            "instruction": task.replace("_", " "),
            "queries": 2, "anchors": 34, "physics_ticks": 300,
            "wall_seconds": 3.2, "observations_ok": True, "reset_ok": True,
        } for row in seeds]

    def test_separates_independent_cohorts_retries_and_unresolved(self):
        priority, _, episodes = self.shard("turn_switch")
        normal, _, other = self.shard("task_47")
        priority_data = priority / "episodes.jsonl"
        self.append(priority / "prescreen.jsonl", {
            **episodes[0], "requested_seed": 42, "status": "excluded", "attempt": 1,
            "reason": "expert_failure",
        })
        for i, row in enumerate(episodes[:100]):
            self.append(priority_data, {**row, "status": "success" if i < 50 else "failure", "attempt": 1})
        self.append(priority_data, {**episodes[100], "status": "technical_error", "attempt": 1, "reason": "render_error"})
        self.append(priority_data, {**episodes[100], "status": "success", "attempt": 2})
        self.append(priority_data, {**episodes[101], "status": "technical_error", "attempt": 1, "reason": "render_error"})
        self.append(priority_data, {**episodes[101], "status": "unresolved", "attempt": 2, "reason": "render_error"})
        self.append(priority_data, {**episodes[102], "status": "technical_error", "attempt": 1, "reason": "oom"})
        with priority_data.open("ab") as data:
            data.write(b'{"incomplete live')
        for row in other:
            self.append(normal / "episodes.jsonl", {**row, "status": "success", "attempt": 1})
        report = summarize(self.root, self.tasks)
        self.assertEqual(report["overall"]["easy"]["cumulative"]["planned"], 7000)
        self.assertEqual(report["overall"]["easy"]["first_batch"]["planned"], 5000)
        self.assertEqual(report["overall"]["easy"]["first_batch"]["complete_tasks"], 2)
        self.assertEqual(report["overall"]["easy"]["first_batch"]["complete_subset_task_weighted_sr"], 0.75)
        self.assertIsNone(report["overall"]["easy"]["first_batch"]["all_planned_tasks_weighted_sr"])
        self.assertEqual(report["overall"]["easy"]["cumulative"]["complete_subset_task_weighted_sr"], 1)
        summary = report["tasks"][1]["conditions"]["easy"]
        self.assertEqual(summary["prescreen"]["excluded"], 1)
        self.assertEqual(summary["prescreen"]["acceptance_rate"], 300 / 301)
        self.assertEqual(summary["batches"]["1"]["independent_seeds"], 100)
        self.assertEqual(summary["batches"]["1"]["mean_budget"]["anchors"], {"mean": 34, "n": 100})
        second = summary["batches"]["2"]
        self.assertEqual(second["unresolved_only_bounds"], [0.5, 1])
        self.assertEqual(second["technical_attempts"], 4)
        self.assertEqual(second["retries"], 2)
        self.assertEqual(second["retry_pending"], 1)
        self.assertEqual(second["not_started"], 97)
        self.assertEqual(second["successes"], 1)
        self.assertEqual(second["finished"], 1)
        self.assertTrue(report["evidence_files"]["formal/turn_switch/easy/episodes.jsonl"]["trailing_uncommitted_bytes"])
        out = io.StringIO()
        emit_markdown(report, out)
        self.assertIn("未解决全失败至全成功", out.getvalue())
        out = io.StringIO()
        emit_csv(report, out)
        out.seek(0)
        self.assertEqual(len(list(csv.DictReader(out))), 240)

    def test_rejects_duplicate_terminal_or_different_protocol(self):
        directory, _, episodes = self.shard("move_stapler_pad", phase="smoke")
        path = directory / "episodes.jsonl"
        row = {**episodes[0], "status": "failure", "attempt": 1}
        self.append(path, row)
        self.append(path, {**row, "status": "success", "attempt": 2})
        with self.assertRaisesRegex(ValueError, "retried a terminal result"):
            summarize(self.root, self.tasks, phase="smoke")
        path.unlink()
        self.append(path, {**row, "protocol_fingerprint": "d" * 64})
        with self.assertRaisesRegex(ValueError, "multiple protocol fingerprints"):
            summarize(self.root, self.tasks, phase="smoke")

    def test_rejects_seed_substitutions_and_same_actual_seed(self):
        directory, manifest, episodes = self.shard("move_stapler_pad", phase="smoke")
        self.append(directory / "episodes.jsonl", {
            **episodes[0], "actual_seed": 99, "status": "failure", "attempt": 1,
        })
        with self.assertRaisesRegex(ValueError, "identity differs"):
            summarize(self.root, self.tasks, phase="smoke")
        manifest["episodes"][1]["actual_seed"] = episodes[0]["actual_seed"]
        self.write_json(directory / "seed_manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "repeated seed"):
            summarize(self.root, self.tasks, phase="smoke")

    def test_wilson_and_idle_report_do_not_impute_success_rate(self):
        self.assertIsNone(wilson(0, 0))
        self.assertAlmostEqual(wilson(50, 100)[0], 0.4038, places=4)
        self.assertAlmostEqual(wilson(150, 300)[1], 0.5562, places=4)
        self.assertAlmostEqual(wilson(0, 100)[1], 0.0370, places=4)
        report = summarize(self.root, self.tasks, phase="smoke")
        self.assertEqual(report["label"], "smoke only")
        self.assertIsNone(report["protocol"])
        self.assertEqual(report["overall"]["hard"]["cumulative"]["planned"], 10)
        self.assertEqual(len(report["tasks"]), 2)


if __name__ == "__main__":
    unittest.main()
