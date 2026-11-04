"""Validate experiment orchestration without Slurm, ML packages or network."""

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("fyp_experiment_tested", ROOT / "tools/idac/experiment.py")
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)
COMMIT = "a" * 40


def create_plan(directory, profile, next_suite=None, paired_baseline_suite=None):
    with patch.dict(os.environ, {"FYP_DATA_ROOT": str(Path(directory).parent / "data")}), contextlib.redirect_stdout(io.StringIO()):
        EXPERIMENT.create(argparse.Namespace(directory=str(directory), profile=profile, commit=COMMIT,
                                             next_suite=str(next_suite) if next_suite else None,
                                             paired_baseline_suite=str(paired_baseline_suite) if paired_baseline_suite else None))
    return EXPERIMENT.load_plan(directory)


def write_training_fixture(plan, task, bc=False, eligible_update=False):
    """A finite completed journal with explicit real-update/skip admission."""
    run = task["run"]
    training = Path(run["training_dir"])
    steps = len(plan["task_ids"]) * plan["iterations_per_task"]
    records = []
    for index in range(steps):
        stage = index // plan["iterations_per_task"] + 1
        active = plan["task_ids"][:stage - 1] if run["method"].startswith("seq") else plan["task_ids"]
        updated = index == 0 or (eligible_update and index == steps - 1)
        records.append({"event": "iteration", "outer_step": index + 1, "stage": stage,
                        "policy_env_steps": (index + 1) * 4096,
                        "replay": [{"stage": stage, "active_task_ids": active}] if run["method"].endswith("_er") else None,
                        "training_metrics": {"fyp_grpo_optimizer_steps": 8 if updated else 0,
                                             "fyp_grpo_skipped": int(not updated), "bc/examples": int(bc and bool(active) and updated)}})
    records.append({"event": "complete", "outer_step": steps, "policy_env_steps": steps * 4096})
    training.mkdir(parents=True, exist_ok=True)
    Path(run["experiment_log_path"]).write_text("\n".join(json.dumps(record) for record in records))
    for stage in range(1, len(plan["task_ids"]) + 1):
        adapter = training / f"checkpoints/global_step_{stage * plan['iterations_per_task']}/actor/adapter_model.bin"
        adapter.parent.mkdir(parents=True, exist_ok=True)
        adapter.write_bytes(b"fixture weights")
        (adapter.parent / "adapter_config.json").write_text('{}')
    update = "FYP_GRPO_UPDATE " + json.dumps({"optimizer_steps": 8, "grad_norms": [1.0], "bc_examples": int(bc)})
    rl = "FYP_GRPO_GRAD " + json.dumps({"isolated_from_bc": True, "isolated_rl_grad_norm": 0.7})
    profile = "FYP_BC_PROFILE " + json.dumps({"stage": len(plan["task_ids"]), "grad_norm": 0.8, "nonzero_gradient_tensors": 4, "samples": 1})
    (training / "run_embodiment.log").write_text(update + "\n" + rl + "\n" + (profile if bc else ""))
    return records


class PlanTests(unittest.TestCase):
    def test_stress10_sparse_panels_anchor_pairing_and_cost_are_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            core_directory = Path(temporary) / "core"
            core = create_plan(core_directory, "core")
            directory = Path(temporary) / "stress10"
            stress = create_plan(directory, "stress10", paired_baseline_suite=core_directory)
            self.assertEqual(stress["task_ids"], list(range(10)))
            self.assertEqual(len(stress["runs"]), 6)
            self.assertEqual({run["method"] for run in stress["runs"]}, {"seq", "seq_er"})
            self.assertEqual({run["seed"] for run in stress["runs"]}, {1234, 2345, 3456})
            for run in stress["runs"]:
                self.assertEqual(run["expected_policy_env_steps"], 409600)
                rows = [entry for entry in stress["evaluations"]
                        if entry["run_id"] == run["run_id"] and entry["variant"] == "id"]
                self.assertEqual(len(rows), 10)
                for entry in rows:
                    stage = entry["stage"]
                    expected = list(range(10)) if stage == 10 else list(range(6)) if stage == 5 else [stage - 1, stage]
                    self.assertEqual(entry["task_ids"], expected)
                    self.assertEqual(entry["expected_episodes_per_task"], 20)
                    self.assertEqual(entry["policy_env_steps"], stage * 40960)
            baseline = next(entry for entry in stress["evaluations"] if entry["run_id"] == "sft" and entry["variant"] == "id")
            self.assertEqual(baseline["task_ids"], list(range(10)))
            self.assertFalse(any(entry["variant"] == "rl_holdout" for entry in stress["evaluations"]))
            self.assertNotIn("rl_holdout", stress["exposure"])
            for variant, tasks in (("camera_shift", [2, 3, 4]), ("object_transfer", [3, 4])):
                entries = [entry for entry in stress["evaluations"] if entry["variant"] == variant]
                self.assertEqual(len(entries), 7)
                self.assertTrue(all(entry["task_ids"] == tasks and entry["role"] == "final" for entry in entries))
            pairing = stress["paired_length_comparison"]
            self.assertEqual(pairing["anchor_task_ids"], list(range(5)))
            self.assertEqual(pairing["reference_plan_sha256"], hashlib.sha256((core_directory / "plan.json").read_bytes()).hexdigest())
            self.assertEqual(pairing["extra_policy_env_steps_per_run"], 204800)
            self.assertFalse(stress["evaluation_design"]["nbt_time_available"])
            self.assertFalse(stress["evaluation_design"]["full_task_macro_auc_available"])
            self.assertEqual(stress["planned_cost"]["training_policy_env_steps_all_runs"], 2457600)
            self.assertEqual(sum(run["expected_policy_env_steps"] for run in core["runs"]), 2457600)
            self.assertEqual(stress["planned_cost"]["evaluation_episodes"], 4740)
            self.assertEqual(stress["planned_cost"]["evaluation_policy_env_steps"], 4740 * 512)
            # Exercise the real analyzer: sparse cells must remain missing,
            # while all required diagonal/final/next-task cells exist.
            analysis_spec = importlib.util.spec_from_file_location("stress_analysis_tested", ROOT / "tools/idac/analyze_experiment.py")
            analysis = importlib.util.module_from_spec(analysis_spec)
            analysis_spec.loader.exec_module(analysis)
            matrix, budgets = {}, {}
            for entry in stress["evaluations"]:
                if entry["variant"] != "id" or entry["run_id"] not in {"sft", "seq_s1234"}:
                    continue
                matrix[str(entry["stage"])] = {str(task): {"success_ever_rate": 0.5, "complete": True}
                                               for task in entry["task_ids"]}
                budgets[str(entry["stage"])] = entry["policy_env_steps"]
            metrics = analysis.continual_metrics(matrix, stress["task_ids"], "success_ever")
            self.assertTrue(metrics["nbt_final"]["complete"])
            self.assertTrue(metrics["fwt"]["complete"])
            self.assertFalse(metrics["nbt_time"]["complete"])
            self.assertIsNone(metrics["nbt_time"]["value"])
            auc = analysis.fixed_panel_auc(matrix, budgets, stress["task_ids"], "success_ever", 409600)
            self.assertFalse(auc["complete"])
            self.assertIsNone(auc["value"])
            # Generating stress10 must not retroactively change the core protocol.
            self.assertIn("rl_holdout", EXPERIMENT.load_plan(core_directory)["exposure"])

    def test_stress10_rejects_absent_or_incompatible_frozen_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "paired-baseline-suite"):
                create_plan(Path(temporary) / "missing", "stress10")
            core_directory = Path(temporary) / "core"
            core = create_plan(core_directory, "core")
            core["dataset_revision"] = "b" * 40
            EXPERIMENT.atomic_json(core_directory / "plan.json", core)
            with self.assertRaisesRegex(ValueError, "dataset_revision"):
                create_plan(Path(temporary) / "incompatible", "stress10", paired_baseline_suite=core_directory)

    def test_stress10_downloads_only_missing_pinned_files_then_verifies_every_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            core_directory = Path(temporary) / "core"
            create_plan(core_directory, "core")
            plan = create_plan(Path(temporary) / "stress10", "stress10", paired_baseline_suite=core_directory)
            map_spec = importlib.util.spec_from_file_location("fyp_stress_task_map", ROOT / "LIBERO/libero/libero/benchmark/libero_suite_task_map.py")
            task_map = importlib.util.module_from_spec(map_spec)
            map_spec.loader.exec_module(task_map)
            raw_root = Path(temporary) / "data/datasets/libero_raw"
            payloads, siblings = {}, []
            for index, name in enumerate(task_map.libero_task_map["libero_spatial"]):
                filename = f"libero_spatial/{name}_demo.hdf5"
                content = f"opaque fixture task {index}".encode()
                payloads[filename] = content
                siblings.append(SimpleNamespace(rfilename=filename, size=len(content),
                                                lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())))
                if index < 5:
                    path = raw_root / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
            downloads = []

            def download(**kwargs):
                downloads.append(kwargs)
                self.assertEqual(kwargs["revision"], EXPERIMENT.DATA_REVISION)
                self.assertEqual(kwargs["repo_type"], "dataset")
                self.assertEqual(kwargs["max_workers"], 2)
                self.assertEqual(kwargs["allow_patterns"], sorted(entry.rfilename for entry in siblings[5:]))
                for name in kwargs["allow_patterns"]:
                    (raw_root / name).write_bytes(payloads[name])

            fake_hf = SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(dataset_info=lambda *a, **k:
                                      SimpleNamespace(sha=EXPERIMENT.DATA_REVISION, siblings=siblings)), snapshot_download=download)
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}), patch.dict(os.environ, {
                "FYP_DATA_ROOT": str(Path(temporary) / "data"), "FYP_MODEL_PATH": str(Path(temporary) / "model"),
            }), patch.object(EXPERIMENT.subprocess, "run") as run:
                EXPERIMENT.prepare_manifest(plan, ROOT)
                self.assertEqual(len(downloads), 1)
                EXPERIMENT.prepare_manifest(plan, ROOT)
                self.assertEqual(len(downloads), 1)
                command = run.call_args.args[0]
                self.assertEqual(command[command.index("--task-ids") + 1], "0,1,2,3,4,5,6,7,8,9")
                verified = json.loads((Path(plan["directory"]) / "dataset_verified.json").read_text())
                self.assertEqual(len(verified["files"]), 10)
                corrupted = raw_root / siblings[0].rfilename
                corrupted.write_bytes(b"X" * corrupted.stat().st_size)
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    EXPERIMENT.prepare_manifest(plan, ROOT)

    def test_profiles_have_matching_stage_budgets_and_all_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            for profile, tasks, seed_count, iterations in (
                ("qualification", [0, 1], 1, 2), ("core", [0, 1, 2, 3, 4], 3, 10),
            ):
                with self.subTest(profile=profile):
                    directory = Path(temporary) / profile
                    plan = create_plan(directory, profile)
                    self.assertEqual(plan["task_ids"], tasks)
                    self.assertEqual(len(plan["runs"]), 4 * seed_count)
                    self.assertEqual(plan["iterations_per_task"], iterations)
                    keys = [task["key"] for task in plan["jobs"]]
                    self.assertEqual(len(keys), len(set(keys)))
                    for run in plan["runs"]:
                        self.assertEqual(run["expected_policy_env_steps"], len(tasks) * iterations * 4096)
                        stage_evaluations = [entry for entry in plan["evaluations"]
                                             if entry["run_id"] == run["run_id"] and entry["variant"] == "id"]
                        self.assertEqual([entry["stage"] for entry in stage_evaluations], list(range(1, len(tasks) + 1)))
                        for entry in stage_evaluations:
                            step = entry["stage"] * iterations
                            self.assertEqual(entry["policy_env_steps"], step * 4096)
                            self.assertTrue(entry["checkpoint"].replace("\\", "/").endswith(f"checkpoints/global_step_{step}/actor"))
                    with self.assertRaises(FileExistsError):
                        create_plan(directory, profile)

    def test_rolling_submission_queues_one_child_after_its_own_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "qualification"
            plan = create_plan(directory, "qualification")
            commands = []

            def output(command, **kwargs):
                if command[0] == "git":
                    return COMMIT + "\n"
                self.assertEqual(command[0], "sbatch")
                commands.append(command)
                return str(5000 + len(commands)) + ";cluster\n"

            with patch.dict(os.environ, {"FYP_REPO": str(ROOT)}), patch.object(EXPERIMENT.subprocess, "check_output", side_effect=output), contextlib.redirect_stdout(io.StringIO()):
                EXPERIMENT.submit(argparse.Namespace(directory=str(directory), dependency="4999"))
                self.assertEqual(len(commands), 1)
                receipt = json.loads((directory / "submission.json").read_text())
                self.assertEqual(len(receipt["jobs"]), 1)
                self.assertFalse(receipt["complete"])
                with self.assertRaises(FileExistsError):
                    EXPERIMENT.submit(argparse.Namespace(directory=str(directory), dependency=None))
                for index, task in enumerate(plan["jobs"]):
                    parent_id = str(5001 + index)
                    args = argparse.Namespace(directory=str(directory), key=task["key"])
                    result_file = Path(task["log_dir"]) / "result.json"
                    with patch.dict(os.environ, {"SLURM_JOB_ID": parent_id}):
                        # A completed payload is required; merely being the
                        # running parent must not advance the experiment.
                        with self.assertRaises(FileNotFoundError):
                            EXPERIMENT.advance(args)
                        self.assertEqual(len(commands), index + 1)
                        result_file.parent.mkdir(parents=True, exist_ok=True)
                        good_result = {"ok": True, "kind": task["kind"], "commit": COMMIT,
                                       "slurm_job_id": parent_id}
                        for invalid in (dict(good_result, ok=False), dict(good_result, commit="b" * 40),
                                        dict(good_result, slurm_job_id="1")):
                            result_file.write_text(json.dumps(invalid))
                            with self.assertRaises(ValueError):
                                EXPERIMENT.advance(args)
                            self.assertEqual(len(commands), index + 1)
                        result_file.write_text(json.dumps(good_result))
                        with patch.dict(os.environ, {"SLURM_JOB_ID": "999999"}), self.assertRaises(ValueError):
                            EXPERIMENT.advance(args)
                        EXPERIMENT.advance(args)
                        self.assertEqual(len(commands), min(index + 2, len(plan["jobs"])))
                        persisted = json.loads((directory / "submission.json").read_text())
                        self.assertEqual(len(persisted["jobs"]), len(commands))
                        # Duplicate requests cannot queue a second child, even
                        # when the old parent process has not exited yet.
                        with self.assertRaises(ValueError):
                            EXPERIMENT.advance(args)
                        self.assertEqual(len(commands), min(index + 2, len(plan["jobs"])))
                        self.assertFalse((directory / "submission.lock").exists())
            self.assertEqual(len(commands), len(plan["jobs"]))
            for index, command in enumerate(commands):
                self.assertIn(f"--dependency=afterok:{4999 if index == 0 else 5000 + index}", command)
                self.assertIn("--exclude=4090node1", command)
                resources = [arg for arg in command if arg.startswith("--gres=")]
                kind = plan["jobs"][index]["kind"]
                self.assertEqual(resources, ["--gres=gpu:4090:2"] if kind == "train" else
                                 ["--gres=gpu:4090:1"] if kind in {"eval", "oodcheck"} else [])
                self.assertFalse(any(arg.startswith("--mem=") for arg in command))
            receipt = json.loads((directory / "submission.json").read_text())
            self.assertTrue(receipt["complete"])
            self.assertTrue(receipt["finished"])
            self.assertEqual(receipt["last_job_id"], str(5000 + len(commands)))

    def test_qualification_manifest_contains_only_its_training_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "qualification"
            plan = create_plan(directory, "qualification")
            # Import the task-name map directly: no simulator package imports.
            map_spec = importlib.util.spec_from_file_location("fyp_raw_task_map", ROOT / "LIBERO/libero/libero/benchmark/libero_suite_task_map.py")
            task_map = importlib.util.module_from_spec(map_spec)
            map_spec.loader.exec_module(task_map)
            raw_root = Path(temporary) / "data/datasets/libero_raw"
            siblings = []
            for index, name in enumerate(task_map.libero_task_map["libero_spatial"][:5]):
                filename = f"libero_spatial/{name}_demo.hdf5"
                content = str(index).encode()
                path = raw_root / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                siblings.append(SimpleNamespace(rfilename=filename, size=len(content),
                                                lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest())))
            info = SimpleNamespace(sha=EXPERIMENT.DATA_REVISION, siblings=siblings)
            fake_hf = SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(dataset_info=lambda *a, **k: info),
                                      snapshot_download=lambda **kwargs: self.fail("Qualification must not add a dataset download"))
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}), patch.dict(os.environ, {
                "FYP_DATA_ROOT": str(Path(temporary) / "data"), "FYP_MODEL_PATH": str(Path(temporary) / "model"),
            }), patch.object(EXPERIMENT.subprocess, "run") as run:
                EXPERIMENT.prepare_manifest(plan, ROOT)
            command = run.call_args.args[0]
            task_ids = command[command.index("--task-ids") + 1]
            self.assertEqual(task_ids, "0,1")
            self.assertEqual(command[command.index("--output") + 1], plan["replay_manifest_path"])
            self.assertEqual(Path(plan["replay_manifest_path"]).parent, directory.resolve())

    def test_successful_gate_starts_exactly_one_frozen_continuation_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            core_directory = Path(temporary) / "core"
            create_plan(core_directory, "core")
            directory = Path(temporary) / "qualification"
            plan = create_plan(directory, "qualification", next_suite=core_directory)
            entries = [{"key": task["key"], "job_id": str(6000 + index), "kind": task["kind"]}
                       for index, task in enumerate(plan["jobs"])]
            parent_id = entries[-1]["job_id"]
            EXPERIMENT.atomic_json(directory / "submission.json", {
                "commit": COMMIT, "directory": str(directory.resolve()), "jobs": entries,
                "complete": True, "last_job_id": parent_id,
            })
            gate = plan["jobs"][-1]
            self.assertEqual(gate["key"], "gate")
            EXPERIMENT.atomic_json(Path(gate["log_dir"]) / "result.json", {
                "ok": True, "kind": "gate", "commit": COMMIT, "slurm_job_id": parent_id,
            })
            submitted = []

            def output(command, **kwargs):
                if command[0] == "git":
                    return COMMIT + "\n"
                submitted.append(command)
                return "7000\n"

            with patch.dict(os.environ, {"FYP_REPO": str(ROOT), "SLURM_JOB_ID": parent_id}), \
                    patch.object(EXPERIMENT.subprocess, "check_output", side_effect=output), \
                    contextlib.redirect_stdout(io.StringIO()):
                EXPERIMENT.advance(argparse.Namespace(directory=str(directory), key="gate"))
                with self.assertRaises(ValueError):
                    EXPERIMENT.advance(argparse.Namespace(directory=str(directory), key="gate"))
            self.assertEqual(len(submitted), 1)
            self.assertIn(f"--dependency=afterok:{parent_id}", submitted[0])
            child_receipt = json.loads((core_directory / "submission.json").read_text())
            self.assertEqual(len(child_receipt["jobs"]), 1)
            self.assertEqual(child_receipt["last_job_id"], "7000")
            receipt = json.loads((directory / "submission.json").read_text())
            self.assertTrue(receipt["finished"])
            self.assertEqual(receipt["continuation_started"], str(core_directory.resolve()))

    def test_training_launch_preserves_protocol_and_hard_replay_allowlist(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "qualification"
            plan = create_plan(directory, "qualification")
            task = next(task for task in plan["jobs"] if task["key"] == "train_seq_er_s1234")
            data = Path(temporary) / "data"
            (data / "reports").mkdir(parents=True)
            (data / "reports/model.snapshot.json").write_text(json.dumps({
                "verified": True, "revision": EXPERIMENT.REVISION, "files": [],
            }))
            with patch.dict(os.environ, {
                "FYP_REPO": str(ROOT), "FYP_DATA_ROOT": str(data), "FYP_MODEL_PATH": str(data / "model"),
            }), patch.object(EXPERIMENT.subprocess, "check_output", return_value=COMMIT + "\n"), \
                    patch.object(EXPERIMENT.subprocess, "run"), patch.object(EXPERIMENT, "launch") as launch, \
                    patch.object(EXPERIMENT, "verify_training", return_value={"rl_verified": True}):
                EXPERIMENT.run_task(argparse.Namespace(directory=str(directory), key=task["key"]))
            command = launch.call_args.args[0]
            mapping_text = next(arg.split("=", 1)[1] for arg in command if arg.startswith("+runner.fyp_experiment="))
            # Hydra flow mappings leave keys unquoted, while values are JSON.
            import re
            parsed = json.loads(re.sub(r"([,{])([A-Za-z_]+):", r'\1"\2":', mapping_text))
            self.assertEqual(parsed, {"method": "seq", "task_ids": [0, 1], "iterations_per_task": 2,
                                      "seed": 1234, "train_init_ids": list(range(40))})
            self.assertIn("algorithm.use_experience_replay=true", command)
            self.assertIn("+algorithm.replay_allowed_task_ids=[0,1]", command)
            self.assertIn(f"+algorithm.replay_manifest_path={plan['replay_manifest_path']}", command)
            self.assertIn("actor.model.attn_implementation=sdpa", command)

    def test_er_qualification_requires_separate_rl_and_bc_gradients(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = create_plan(Path(temporary) / "qualification", "qualification")
            task = next(task for task in plan["jobs"] if task["key"] == "train_seq_er_s1234")
            run = task["run"]
            training = Path(run["training_dir"])
            steps = len(plan["task_ids"]) * plan["iterations_per_task"]
            write_training_fixture(plan, task, bc=True, eligible_update=True)
            log_path = training / "run_embodiment.log"
            update = "FYP_GRPO_UPDATE " + json.dumps({"optimizer_steps": 8, "grad_norms": [1.0], "bc_examples": 1})
            rl = "FYP_GRPO_GRAD " + json.dumps({"isolated_from_bc": True, "isolated_rl_grad_norm": 0.7})
            bc = "FYP_BC_PROFILE " + json.dumps({"stage": 2, "grad_norm": 0.8, "nonzero_gradient_tensors": 4, "samples": 1})
            log_path.write_text(update + "\n" + bc + "\n")
            with self.assertRaisesRegex(RuntimeError, "isolated RL"):
                EXPERIMENT.verify_training(plan, task)
            log_path.write_text(update + "\n" + rl + "\n")
            with self.assertRaisesRegex(RuntimeError, "BC gradient"):
                EXPERIMENT.verify_training(plan, task)
            log_path.write_text(update + "\n" + rl + "\n" + bc + "\n")
            EXPERIMENT.verify_training(plan, task)

    def test_unexercised_replay_requires_every_eligible_iteration_to_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = create_plan(Path(temporary) / "qualification", "qualification")
            task = next(task for task in plan["jobs"] if task["key"] == "train_seq_er_s1234")
            records = write_training_fixture(plan, task)
            result = EXPERIMENT.verify_training(plan, task)
            self.assertEqual(result["bc_status"], "not_exercised")
            self.assertEqual(result["eligible_replay_iterations"], [3, 4])
            self.assertEqual(result["replay_eligible_optimizer_iterations"], [])
            self.assertEqual(result["bc_examples"], 0)
            journal = Path(task["run"]["experiment_log_path"])
            bad = json.loads(json.dumps(records))
            bad[2]["training_metrics"]["fyp_grpo_optimizer_steps"] = 8
            bad[2]["training_metrics"]["fyp_grpo_skipped"] = 0
            journal.write_text("\n".join(json.dumps(record) for record in bad))
            with self.assertRaisesRegex(RuntimeError, "BC gradient"):
                EXPERIMENT.verify_training(plan, task)
            bad = json.loads(json.dumps(records))
            bad[2]["replay"][0]["active_task_ids"] = [0, 1]
            journal.write_text("\n".join(json.dumps(record) for record in bad))
            with self.assertRaisesRegex(RuntimeError, "admission"):
                EXPERIMENT.verify_training(plan, task)

    def test_suite_gate_still_requires_actual_isolated_bc_from_an_er_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = create_plan(Path(temporary) / "qualification", "qualification")
            for task in plan["jobs"]:
                if task["kind"] == "train":
                    if task["run"]["method"] == "joint_er":
                        continue
                    write_training_fixture(plan, task)
            joint = next(task for task in plan["jobs"] if task["key"] == "train_joint_er_s1234")
            # Mock one unexercised classification to isolate the suite-level gate.
            real = EXPERIMENT.verify_training
            def validate(plan, task):
                if task["run"]["method"] == "joint_er":
                    return {"run_id": task["run"]["run_id"], "bc_status": "not_exercised", "rl_verified": True}
                return real(plan, task)
            with patch.object(EXPERIMENT, "verify_training", side_effect=validate), self.assertRaisesRegex(RuntimeError, "Suite did not exercise"):
                EXPERIMENT.verify_suite_training(plan)
            write_training_fixture(plan, joint, bc=True, eligible_update=True)
            result = EXPERIMENT.verify_suite_training(plan)
            self.assertTrue(result["isolated_bc_verified"])
            self.assertEqual(next(row for row in result["runs"] if row["run_id"] == "seq_er_s1234")["bc_status"], "not_exercised")

    def test_recovery_preserves_finished_training_and_queues_only_remaining_work(self):
        with tempfile.TemporaryDirectory() as temporary:
            source_root = Path(temporary) / "old"
            source = create_plan(source_root, "qualification")
            first_failed = next(index for index, task in enumerate(source["jobs"]) if task["key"] == "train_seq_er_s1234")
            submitted = []
            for index, task in enumerate(source["jobs"][:first_failed + 1]):
                job_id = str(8000 + index)
                submitted.append({"key": task["key"], "job_id": job_id, "kind": task["kind"]})
                log_dir = Path(task["log_dir"])
                log_dir.mkdir(parents=True, exist_ok=True)
                if task["kind"] == "train":
                    write_training_fixture(source, task)
                elif task["kind"] == "eval":
                    entry = task["evaluation"]
                    episodes = [{"task_id": task_id} for task_id in entry["task_ids"]
                                for _ in range(entry["expected_episodes_per_task"])]
                    EXPERIMENT.atomic_json(Path(entry["evidence_dir"]) / "eval_summary.json", {"episodes": episodes})
                elif task["kind"] == "oodcheck":
                    EXPERIMENT.atomic_json(log_dir / "ood_validation.json", {"verified": True})
                if index < first_failed:
                    EXPERIMENT.atomic_json(log_dir / "result.json", {"ok": True, "commit": COMMIT,
                                          "kind": task["kind"], "slurm_job_id": job_id})
            EXPERIMENT.atomic_json(source_root / "submission.json", {"jobs": submitted, "complete": False})
            error = source_root / "slurm" / f"{submitted[-1]['job_id']}.err"
            error.parent.mkdir()
            error.write_text("RuntimeError: ER run did not verify an isolated nonzero BC gradient")
            original_plan = (source_root / "plan.json").read_bytes()
            original_receipt = (source_root / "submission.json").read_bytes()
            recovery_root = Path(temporary) / "recovery"
            args = argparse.Namespace(directory=str(recovery_root), source_suite=str(source_root),
                                      commit="b" * 40, next_suite=str(Path(temporary) / "newcore"))
            with patch.dict(os.environ, {"FYP_REPO": str(ROOT)}), patch.object(EXPERIMENT.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                EXPERIMENT.recover(args)
            self.assertIn("diff", run.call_args.args[0])
            recovered = EXPERIMENT.load_plan(recovery_root)
            self.assertEqual(recovered["jobs"][0]["key"], "eval_seq_er_s1234_stage1_id")
            self.assertEqual(len(recovered["jobs"]), 8)
            self.assertEqual([task["key"] for task in recovered["jobs"] if task["kind"] == "train"], ["train_joint_er_s1234"])
            self.assertEqual(recovered["recovery"]["additional_training_policy_env_steps"], 16384)
            self.assertEqual(len(recovered["recovery"]["retained_tasks"]), 10)
            runs = {run["run_id"]: run for run in recovered["runs"]}
            self.assertEqual(runs["seq_er_s1234"]["training_validation"]["bc_status"], "not_exercised")
            self.assertEqual(runs["seq_er_s1234"]["training_dir"], source["runs"][2]["training_dir"])
            self.assertTrue(Path(runs["joint_er_s1234"]["training_dir"]).is_relative_to(recovery_root))
            reused_eval = next(entry for entry in recovered["evaluations"] if entry["run_id"] == "seq_s1234")
            self.assertTrue(Path(reused_eval["evidence_dir"]).is_relative_to(source_root))
            pending_eval = next(entry for entry in recovered["evaluations"] if entry["run_id"] == "joint_er_s1234")
            self.assertTrue(Path(pending_eval["checkpoint"]).is_relative_to(recovery_root))
            self.assertEqual((source_root / "plan.json").read_bytes(), original_plan)
            self.assertEqual((source_root / "submission.json").read_bytes(), original_receipt)
            with self.assertRaises(FileExistsError):
                EXPERIMENT.recover(args)


if __name__ == "__main__":
    unittest.main()
