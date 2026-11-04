"""Check signed metrics, evidence counts and incomplete experiment reporting."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

from analyze_experiment import (analyze, audit_training, continual_metrics,
                                fixed_panel_auc, load_evaluation, wilson_interval)


def cell(ever, final=None, n=10, complete=True):
    final = ever if final is None else final
    return {"episodes": n, "success_ever": int(ever * n), "success_final": int(final * n),
            "success_ever_rate": ever, "success_final_rate": final, "complete": complete}


def episode(task, ever, final, steps=512):
    return {"task_id": task, "success_ever": ever, "success_final": final,
            "policy_env_steps": steps, "init_state_id": 40}


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.matrix = {
            "0": {"0": cell(.2), "1": cell(.3), "2": cell(.4)},
            "1": {"0": cell(.8), "1": cell(.5), "2": cell(.5)},
            "2": {"0": cell(.6), "1": cell(.7), "2": cell(.6)},
            "3": {"0": cell(.9), "1": cell(.5), "2": cell(.8)},
        }

    def test_signed_terminal_and_time_aggregation_differ(self):
        values = continual_metrics(self.matrix, [0, 1, 2], "success_ever")
        self.assertAlmostEqual(values["nbt_final"]["value"], .05)
        self.assertAlmostEqual(values["nbt_time"]["value"], .125)
        self.assertAlmostEqual(values["fwt"]["value"], .2)
        self.assertAlmostEqual(values["per_task_final_drop"]["0"], -.1)
        self.assertAlmostEqual(values["final_average"]["value"], 2.2 / 3)

    def test_missing_cell_is_unknown_not_zero(self):
        del self.matrix["2"]["0"]
        values = continual_metrics(self.matrix, [0, 1, 2], "success_ever")
        self.assertTrue(values["nbt_final"]["complete"])
        self.assertFalse(values["nbt_time"]["complete"])
        self.assertIsNone(values["nbt_time"]["value"])
        self.assertIn([2, "0"], values["nbt_time"]["missing_cells"])

    def test_joint_has_no_artificial_diagonal_nbt(self):
        values = continual_metrics(self.matrix, [0, 1, 2], "success_final", sequential=False)
        self.assertIsNone(values["nbt_final"]["value"])
        self.assertTrue(values["final_average"]["complete"])

    def test_auc_requires_fixed_panel_and_actual_budget_endpoints(self):
        budgets = {"0": 0, "1": 100, "2": 200, "3": 300}
        auc = fixed_panel_auc(self.matrix, budgets, [0, 1, 2], "success_ever", 300)
        self.assertTrue(auc["complete"])
        expected = ((.3 + .6) / 2 + (.6 + 1.9 / 3) / 2 + (1.9 / 3 + 2.2 / 3) / 2) / 3
        self.assertAlmostEqual(auc["value"], expected)
        self.assertIsNone(fixed_panel_auc(self.matrix, budgets, [0, 1, 2], "success_ever", 400)["value"])
        del self.matrix["1"]["2"]
        self.assertFalse(fixed_panel_auc(self.matrix, budgets, [0, 1, 2], "success_ever", 300)["complete"])

    def test_wilson_handles_zero_and_all_success(self):
        self.assertAlmostEqual(wilson_interval(0, 10)[0], 0)
        self.assertLess(wilson_interval(0, 10)[1], .3)
        self.assertAlmostEqual(wilson_interval(10, 10)[1], 1)
        self.assertGreater(wilson_interval(10, 10)[0], .7)
        self.assertIsNone(wilson_interval(0, 0))


class EvidenceTests(unittest.TestCase):
    def test_counted_episodes_override_batch_means_and_partial_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "eval"
            evidence.mkdir()
            data = {"episodes": [episode(0, True, False)] + [episode(0, False, False)] * 9,
                    "policy_env_steps": 5120}
            (evidence / "eval_summary.json").write_text(json.dumps(data))
            report = load_evaluation(root, {"evidence_dir": "eval", "task_ids": [0],
                                            "expected_episodes_per_task": 10})
            self.assertTrue(report["complete"])
            self.assertAlmostEqual(report["per_task"]["0"]["success_ever_rate"], .1)
            short = load_evaluation(root, {"evidence_dir": "eval", "task_ids": [0],
                                           "expected_episodes_per_task": 20})
            self.assertFalse(short["complete"])
            data["policy_env_steps"] = 512
            (evidence / "eval_summary.json").write_text(json.dumps(data))
            self.assertFalse(load_evaluation(root, {"evidence_dir": "eval", "task_ids": [0]})["complete"])

    def test_iteration_ledger_excludes_start_and_complete_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [{"event": "start"}, {"event": "iteration_start"},
                       {"event": "iteration", "policy_env_steps": 4096},
                       {"event": "complete", "policy_env_steps": 4096}]
            (root / "experiment.jsonl").write_text("\n".join(map(json.dumps, records)))
            report = audit_training(root, {"experiment_log_path": "experiment.jsonl", "expected_policy_env_steps": 4096})
            self.assertTrue(report["complete"])
            self.assertEqual(report["iterations"], 1)
            records.append({"event": "iteration", "policy_env_steps": 3})
            (root / "experiment.jsonl").write_text("\n".join(map(json.dumps, records)))
            self.assertFalse(audit_training(root, {"experiment_log_path": "experiment.jsonl"})["complete"])

    def test_shared_baseline_is_used_without_filling_missing_final(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            evidence = root / "base"
            evidence.mkdir()
            (evidence / "eval_summary.json").write_text(json.dumps({
                "episodes": [episode(0, True, False)], "policy_env_steps": 512}))
            plan = {"task_ids": [0, 1], "runs": [{"run_id": "seq", "method": "seq", "task_ids": [0, 1]}],
                    "evaluations": [{"run_id": "sft", "stage": 0, "evidence_dir": "base", "task_ids": [0],
                                      "expected_episodes_per_task": 1, "policy_env_steps": 0},
                                     {"run_id": "seq", "stage": 2, "evidence_dir": "not-yet", "task_ids": [0, 1]}]}
            (root / "plan.json").write_text(json.dumps(plan))
            report = analyze(root)
            self.assertFalse(report["complete"])
            self.assertEqual(report["runs"][0]["matrix"]["0"]["0"]["episodes"], 1)
            self.assertIsNone(report["runs"][0]["metrics"]["success_ever"]["nbt_final"]["value"])

    def test_variant_task_ids_and_roles_do_not_pollute_main_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            entries = []
            for name, run_id, stage, variant, role, success in [
                ("base", "sft", 0, "id", "development", True),
                ("base-object", "sft", 0, "object_transfer", "development", False),
                ("end-object", "seq", 1, "object_transfer", "development", True),
                ("base-final", "sft", 0, "object_transfer", "final", True),
            ]:
                evidence = root / name
                evidence.mkdir()
                (evidence / "eval_summary.json").write_text(json.dumps({
                    "episodes": [episode(0, success, success)], "policy_env_steps": 512}))
                entries.append({"run_id": run_id, "stage": stage, "variant": variant, "role": role,
                                "evidence_dir": name, "task_ids": [0], "expected_episodes_per_task": 1})
            plan = {"runs": [{"run_id": "seq", "method": "seq", "task_ids": [0]}], "evaluations": entries}
            (root / "plan.json").write_text(json.dumps(plan))
            report = analyze(root)
            self.assertEqual(report["runs"][0]["matrix"]["0"]["0"]["success_ever_rate"], 1)
            self.assertNotIn("1", report["runs"][0]["matrix"])
            dev = next(item for item in report["variant_evaluations"] if item["role"] == "development")
            final = next(item for item in report["variant_evaluations"] if item["role"] == "final")
            self.assertEqual(dev["per_task"]["0"]["delta"]["success_ever"], 1)
            self.assertIsNone(final["per_task"]["0"]["delta"]["success_ever"])

    def test_eval_runner_subtracts_nested_cumulative_evidence(self):
        source = Path(__file__).resolve().parents[2] / "rlinf/runners/embodied_eval_runner.py"
        tree = ast.parse(source.read_text())
        wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "_episode_summary")
        method.decorator_list = []
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
        previous = {"rank": 0, "totals": {"policy_env_steps": 512, "settling_env_steps": 10},
                    "episodes": [episode(0, False, False)]}
        current = {"rank": 0, "totals": {"policy_env_steps": 1536, "settling_env_steps": 30,
                                          "post_done_executed_steps": 100},
                   "episodes": previous["episodes"] + [episode(0, True, False), episode(1, True, True)]}
        report = namespace["_episode_summary"]([previous], [current], 1)
        self.assertEqual(report["episode_count"], 2)
        self.assertEqual(report["policy_env_steps"], 1024)
        self.assertEqual(report["settling_env_steps"], 20)
        self.assertEqual(report["success_ever"], 2)
        self.assertEqual(report["success_final"], 1)
        self.assertEqual(report["post_done_executed_steps"], 100)
        current["totals"]["policy_env_steps"] = 1500
        with self.assertRaises(RuntimeError):
            namespace["_episode_summary"]([previous], [current], 1)


if __name__ == "__main__":
    unittest.main()
