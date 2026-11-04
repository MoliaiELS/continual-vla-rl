"""CPU-only schedule/budget tests. No Ray, Torch or MuJoCo imports."""

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fyp_protocol_tested", ROOT / "rlinf/utils/fyp_protocol.py")
PROTOCOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROTOCOL
SPEC.loader.exec_module(PROTOCOL)


def plan(method="seq", **overrides):
    return PROTOCOL.FYPExperiment.from_config(dict(
        method=method, task_ids=overrides.get("task_ids", [0, 1, 2]),
        iterations_per_task=overrides.get("iterations_per_task", 45),
        seed=overrides.get("seed", 1234), train_init_ids=overrides.get("train_init_ids", list(range(40))),
    ))


class ProtocolTests(unittest.TestCase):
    def test_seq_joint_have_identical_task_local_initial_state_streams(self):
        seq, joint = plan(), plan("joint")
        schedules = [[protocol.condition(i) for i in range(protocol.total_iterations)]
                     for protocol in (seq, joint)]
        for task_id in seq.task_ids:
            streams = [[row["init_state_id"] for row in rows if row["task_id"] == task_id]
                       for rows in schedules]
            self.assertEqual(streams[0], streams[1])
            self.assertEqual(len(streams[0]), 45)
            self.assertEqual(set(streams[0][:40]), set(range(40)))
            self.assertTrue(all(0 <= state < 40 for state in streams[0]))
        self.assertEqual([row["task_id"] for row in schedules[0]][:45], [0] * 45)
        self.assertEqual([row["task_id"] for row in schedules[1]][:6], [0, 1, 2, 0, 1, 2])

    def test_replay_never_sees_future_tasks_in_seq(self):
        seq = plan(iterations_per_task=2)
        self.assertEqual([seq.replay_tasks(i) for i in range(6)],
                         [[], [], [0], [0], [0, 1], [0, 1]])
        self.assertEqual(plan("joint").replay_tasks(0), [0, 1, 2])
        self.assertEqual([i + 1 for i in range(6) if seq.condition(i)["stage_end"]], [2, 4, 6])

    def test_task_order_does_not_change_a_tasks_initial_state_stream(self):
        first, reordered = plan(), plan(task_ids=[2, 0, 1])
        first_stream = [first.condition(i)["init_state_id"] for i in range(first.total_iterations)
                        if first.condition(i)["task_id"] == 1]
        other_stream = [reordered.condition(i)["init_state_id"] for i in range(reordered.total_iterations)
                        if reordered.condition(i)["task_id"] == 1]
        self.assertEqual(first_stream, other_stream)
        self.assertNotEqual(first_stream, [plan(seed=99).condition(i)["init_state_id"]
                                         for i in range(first.total_iterations)
                                         if first.condition(i)["task_id"] == 1])

    def test_invalid_protocol_is_rejected(self):
        for overrides in ({"task_ids": []}, {"task_ids": [0, 0]}, {"iterations_per_task": 0},
                          {"train_init_ids": []}, {"train_init_ids": [0, 0]}, {"task_ids": [-1]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                plan(**overrides)
        with self.assertRaises(ValueError):
            plan("adaptive")
        with self.assertRaises(IndexError):
            plan().condition(plan().total_iterations)

    def test_budget_delta_retains_masked_and_startup_cost(self):
        startup = {"totals": dict(PROTOCOL.zero_counters(), settling_env_steps=80),
                   "per_task": {"0": dict(PROTOCOL.zero_counters(), settling_env_steps=80)}, "episodes": []}
        current = copy.deepcopy(startup)
        for counters in (current["totals"], current["per_task"]["0"]):
            counters.update(policy_env_steps=4096, settling_env_steps=160,
                            post_done_executed_steps=1900, episodes_collected=8)
        current["episodes"] = [{"task_id": 0, "init_state_id": 18}] * 8
        delta = PROTOCOL.evidence_delta(current, startup)
        self.assertEqual(delta["totals"]["policy_env_steps"], 4096)
        self.assertEqual(delta["totals"]["settling_env_steps"], 80)
        self.assertEqual(delta["totals"]["post_done_executed_steps"], 1900)
        self.assertEqual(len(delta["episodes"]), 8)
        self.assertEqual(PROTOCOL.merge_evidence([startup, delta])["totals"], current["totals"])
        with self.assertRaises(ValueError):
            PROTOCOL.evidence_delta(startup, current)


def extracted_libero_condition():
    """Compile the production setter with tiny simulator/array stand-ins."""
    source = ROOT / "rlinf/envs/libero/libero_env.py"
    tree = ast.parse(source.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LiberoEnv")
    method = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef)
                  and node.name == "set_experiment_condition")
    module = ast.Module(body=[ast.ClassDef(name="Probe", bases=[], keywords=[], body=[method], decorator_list=[])], type_ignores=[])
    namespace = {"copy": copy, "np": SimpleNamespace(full=lambda n, x, dtype: [x] * n, int64=int)}
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    probe = namespace["Probe"]()
    probe.cfg = SimpleNamespace(only_eval=False)
    probe.auto_reset = probe.ignore_terminations = False
    probe.use_fixed_reset_state_ids = True
    probe.world_size = probe.num_group = 1
    probe.group_size = probe.num_envs = 8
    probe.experiment_condition = None
    probe.task_suite = SimpleNamespace(get_num_tasks=lambda: 10, get_task_init_states=lambda task: range(50))
    probe._compute_total_num_group_envs = lambda: None
    return probe


class WrapperConditionTests(unittest.TestCase):
    def test_group_reset_changes_mapping_without_executing_simulation(self):
        probe = extracted_libero_condition()
        probe.set_experiment_condition(0, 18, 0)
        self.assertEqual(probe.fixed_task_ids, [0])
        self.assertEqual(probe.reset_state_ids, [18] * 8)
        probe.set_experiment_condition(1, 3, 1)
        self.assertEqual(probe.fixed_task_ids, [1])
        self.assertEqual(probe.reset_state_ids, [3] * 8)
        self.assertTrue(probe._is_start)
        with self.assertRaises(ValueError):
            probe.set_experiment_condition(1, 3, 1)

    def test_invalid_group_conditions_fail_before_rollout(self):
        for task, state in ((10, 0), (0, 50), (-1, 0), (0, -1)):
            with self.subTest(task=task, state=state), self.assertRaises(ValueError):
                extracted_libero_condition().set_experiment_condition(task, state, 0)
        probe = extracted_libero_condition()
        probe.auto_reset = True
        with self.assertRaises(ValueError):
            probe.set_experiment_condition(0, 0, 0)


class RealStepEvidenceTests(unittest.TestCase):
    def test_cumulative_task_costs_and_success_are_observed_before_done_mask(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy unavailable locally; run this check in the committed remote environment")
        existing_spec = importlib.util.spec_from_file_location("old_evidence_probe", ROOT / "tools/idac/test_rollout_evidence.py")
        existing = importlib.util.module_from_spec(existing_spec)
        existing_spec.loader.exec_module(existing)
        probe = existing.make_probe(None)
        probe.experiment_condition = {"iteration": 0}
        probe.experiment_evidence = {"totals": PROTOCOL.zero_counters(), "per_task": {}, "episodes": []}
        source = ROOT / "rlinf/envs/libero/libero_env.py"
        tree = ast.parse(source.read_text())
        wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LiberoEnv")
        method = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef)
                      and node.name == "_add_experiment_cost")
        namespace = {"zero_counters": PROTOCOL.zero_counters}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), "exec"), namespace)
        probe._add_experiment_cost = namespace["_add_experiment_cost"].__get__(probe)
        probe._write_experiment_evidence = lambda: None
        for _ in range(4):
            probe.step(np.zeros((2, 7)))
        totals = probe.experiment_evidence["totals"]
        self.assertEqual(totals["policy_env_steps"], 8)
        self.assertEqual(totals["post_done_executed_steps"], 2)
        self.assertEqual(totals["episodes_collected"], 2)
        self.assertEqual(totals["success_ever_count"], 1)
        self.assertEqual(totals["success_final_count"], 0)
        self.assertEqual(totals["steps_through_first_success_or_horizon"], 6)
        self.assertEqual(probe.experiment_evidence["per_task"]["0"], totals)


class RunnerJournalTests(unittest.TestCase):
    def test_journal_checks_actual_group_identity_and_records_consumed_cost(self):
        source = ROOT / "rlinf/runners/embodied_runner.py"
        tree = ast.parse(source.read_text())
        runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EmbodiedRunner")
        selected = [node for node in runner.body if isinstance(node, ast.FunctionDef)
                    and node.name in {"_start_experiment", "_append_experiment_record", "_record_experiment_iteration"}]
        module = ast.Module(body=[ast.ClassDef(name="Probe", bases=[], keywords=[], body=selected, decorator_list=[])], type_ignores=[])
        namespace = {"Path": Path, "time": time, "json": json,
                     "merge_evidence": PROTOCOL.merge_evidence, "evidence_delta": PROTOCOL.evidence_delta}
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
        initial = {"totals": dict(PROTOCOL.zero_counters(), settling_env_steps=80), "per_task": {}, "episodes": []}
        current = copy.deepcopy(initial)
        current["totals"].update(policy_env_steps=4096, settling_env_steps=160, episodes_collected=8)
        condition = plan(iterations_per_task=1).condition(0)
        current["episodes"] = [dict(task_id=condition["task_id"], init_state_id=condition["init_state_id"])] * 8
        reports = iter([initial, current])
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = namespace["Probe"]()
            probe.cfg = SimpleNamespace(runner=SimpleNamespace(logger=SimpleNamespace(log_path=temp_dir)),
                                        env=SimpleNamespace(train=SimpleNamespace(max_episode_steps=512)))
            probe.global_step = 0
            probe.max_steps = 3
            probe.fyp_experiment = plan(iterations_per_task=1)
            probe._experiment_replay_metadata = None
            probe._experiment_optimizer_steps = 0
            probe._experiment_skipped_iterations = 0
            probe.env = SimpleNamespace(get_experiment_evidence=lambda: SimpleNamespace(wait=lambda: [next(reports)]))
            probe._start_experiment()
            probe.global_step = 1
            probe._record_experiment_iteration(condition, {"groups_kept": 1}, {
                "grad_norm": 2, "fyp_grpo_optimizer_steps": 8, "fyp_grpo_skipped": 0,
            }, False)
            records = [json.loads(line) for line in (Path(temp_dir) / "experiment.jsonl").read_text().splitlines()]
            self.assertEqual([record["event"] for record in records], ["start", "iteration"])
            self.assertEqual(records[1]["policy_env_steps"], 4096)
            self.assertEqual(records[1]["budget_delta"]["totals"]["settling_env_steps"], 80)
            self.assertEqual(len(records[1]["budget_delta"]["episodes"]), 8)
            wrong_group = copy.deepcopy(current)
            wrong_group["episodes"][0]["task_id"] = 99
            probe._experiment_previous_evidence = initial
            probe.env = SimpleNamespace(get_experiment_evidence=lambda: SimpleNamespace(wait=lambda: [wrong_group]))
            with self.assertRaises(RuntimeError):
                probe._record_experiment_iteration(condition, {}, {}, False)
            probe.env = SimpleNamespace(get_experiment_evidence=lambda: SimpleNamespace(wait=lambda: [initial]))
            probe.global_step = 0
            with self.assertRaises(FileExistsError):
                probe._start_experiment()


if __name__ == "__main__":
    unittest.main()
