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


def create_plan(directory, profile):
    with patch.dict(os.environ, {"FYP_DATA_ROOT": str(Path(directory).parent / "data")}), contextlib.redirect_stdout(io.StringIO()):
        EXPERIMENT.create(argparse.Namespace(directory=str(directory), profile=profile, commit=COMMIT))
    return EXPERIMENT.load_plan(directory)


class PlanTests(unittest.TestCase):
    def test_profiles_have_matching_stage_budgets_and_all_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            for profile, tasks, seed_count, iterations in (
                ("qualification", [0, 1], 1, 1), ("core", [0, 1, 2, 3, 4], 3, 10),
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

    def test_submission_is_serial_and_never_requests_more_than_two_gpus(self):
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
            fake_hf = SimpleNamespace(HfApi=lambda **kwargs: SimpleNamespace(dataset_info=lambda *a, **k: info))
            with patch.dict(sys.modules, {"huggingface_hub": fake_hf}), patch.dict(os.environ, {
                "FYP_DATA_ROOT": str(Path(temporary) / "data"), "FYP_MODEL_PATH": str(Path(temporary) / "model"),
            }), patch.object(EXPERIMENT.subprocess, "run") as run:
                EXPERIMENT.prepare_manifest(plan, ROOT)
            command = run.call_args.args[0]
            task_ids = command[command.index("--task-ids") + 1]
            self.assertEqual(task_ids, "0,1")
            self.assertEqual(command[command.index("--output") + 1], plan["replay_manifest_path"])
            self.assertEqual(Path(plan["replay_manifest_path"]).parent, directory.resolve())

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
                    patch.object(EXPERIMENT, "verify_training"):
                EXPERIMENT.run_task(argparse.Namespace(directory=str(directory), key=task["key"]))
            command = launch.call_args.args[0]
            mapping_text = next(arg.split("=", 1)[1] for arg in command if arg.startswith("+runner.fyp_experiment="))
            # Hydra flow mappings leave keys unquoted, while values are JSON.
            import re
            parsed = json.loads(re.sub(r"([,{])([A-Za-z_]+):", r'\1"\2":', mapping_text))
            self.assertEqual(parsed, {"method": "seq", "task_ids": [0, 1], "iterations_per_task": 1,
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
            adapter = training / "checkpoints/global_step_2/actor/adapter_model.bin"
            adapter.parent.mkdir(parents=True)
            adapter.write_bytes(b"fixture weights")
            Path(run["experiment_log_path"]).write_text('\n'.join(json.dumps({"event": "iteration"}) for _ in range(2)))
            log_path = training / "run_embodiment.log"
            update = "FYP_GRPO_UPDATE " + json.dumps({"optimizer_steps": 8, "grad_norms": [1.0]})
            rl = "FYP_GRPO_GRAD " + json.dumps({"isolated_from_bc": True, "isolated_rl_grad_norm": 0.7})
            bc = "FYP_BC_PROFILE " + json.dumps({"grad_norm": 0.8, "nonzero_gradient_tensors": 4, "samples": 1})
            log_path.write_text(update + "\n" + bc + "\n")
            with self.assertRaisesRegex(RuntimeError, "isolated RL"):
                EXPERIMENT.verify_training(plan, task)
            log_path.write_text(update + "\n" + rl + "\n")
            with self.assertRaisesRegex(RuntimeError, "BC gradient"):
                EXPERIMENT.verify_training(plan, task)
            log_path.write_text(update + "\n" + rl + "\n" + bc + "\n")
            EXPERIMENT.verify_training(plan, task)


if __name__ == "__main__":
    unittest.main()
