"""Analyze locked E3 LIBERO-Spatial journals with paired state-cluster intervals."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import statistics

import numpy as np


DEFAULT_SPLIT = Path(__file__).resolve().parents[2] / "configs/xvla/protocol_20261110/split.json"
SPLIT_SHA256 = "fa4ad56a5fd7f5b08b9612b7fefa1b3e156d63feb0df1d8aa60522fe062ce651"
TASKS = tuple(range(5, 10))
TRIALS = tuple(range(5, 30))
REPETITIONS = 2
TRAINING_SEEDS = (101, 202, 303)
TRAINED = ("SFT-cont", "B1", "B2", "M")
CONDITIONS = ("B0", *TRAINED)
HORIZON = 800
SETTLING = 10
BOOTSTRAP_SEED = 20261110
BOOTSTRAP_REPLICATES = 10_000
SUBSTANTIAL_DROP = -0.10
CONTRASTS = (
    ("M", "B2", "primary"),
    ("B2", "B1", "secondary"),
    ("B1", "B0", "secondary"),
    ("SFT-cont", "B0", "supervised_reference"),
    ("B2", "B0", "baseline_reference"),
    ("M", "B0", "baseline_reference"),
    ("B1", "SFT-cont", "supervised_reference"),
    ("B2", "SFT-cont", "supervised_reference"),
    ("M", "SFT-cont", "supervised_reference"),
)
IDENTITY = ("task_id", "trial_id", "environment_seed", "policy_seed")


@dataclass(frozen=True)
class Panel:
    role: str
    tasks: tuple[int, ...]
    trials: tuple[int, ...]
    repetitions: int
    journal_key: str
    bootstrap_seed: int
    contrasts: tuple[tuple[str, str, str], ...]


H_PANEL = Panel("final_H", TASKS, TRIALS, REPETITIONS, "episodes", BOOTSTRAP_SEED, CONTRASTS)
SOURCE_A_PANEL = Panel(
    "source_A_final_test", (0,), tuple(range(40, 50)), 5, "source_a_episodes", BOOTSTRAP_SEED + 1,
    tuple((condition, "B0", "source_baseline_reference") for condition in TRAINED),
)


def _expected_panel(split_path, panel=H_PANEL):
    contents = split_path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != SPLIT_SHA256:
        raise ValueError("Task and initial-state split differs from the frozen registration")
    split = json.loads(contents)
    evaluation = split["common_evaluation"]
    if panel == SOURCE_A_PANEL:
        if ((split["source_A"],) != panel.tasks or tuple(evaluation["source_final_test_trial_ids"]) != panel.trials):
            raise ValueError("Unexpected scope in the frozen source A panel")
    elif (tuple(split["final_H"]) != panel.tasks or tuple(evaluation["H_trial_ids"]) != panel.trials
          or evaluation["H_repetitions_per_state"] != panel.repetitions):
        raise ValueError("Unexpected scope in the frozen H panel")
    task_hashes = {task["task_id"]: task["state_sha256_by_trial_id"] for task in split["tasks"]}
    hashes = {}
    for task in panel.tasks:
        for trial in panel.trials:
            digest = task_hashes[task][trial]
            if not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid frozen initial-state digest for task {task}, trial {trial}")
            for rep in range(panel.repetitions):
                key = (task, trial, 142 + trial + 1000 * rep, 10000 + 100 * task + trial + 1000 * rep)
                hashes[key] = (digest, rep)
    if panel == SOURCE_A_PANEL and len({digest for digest, _ in hashes.values()}) != len(panel.trials):
        raise ValueError("The independent source panel requires distinct initial-state hashes")
    return hashes


def _read_policy(entry, root, expected, panel=H_PANEL):
    paths = entry[panel.journal_key]
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list) or not paths or any(not isinstance(path, str) or not path for path in paths):
        raise ValueError("episodes must contain a journal path or a nonempty list of journal paths")
    rows = {}
    journals = []
    requires_repetition_index = panel == SOURCE_A_PANEL
    for path in paths:
        journal = (root / path).resolve()
        contents = journal.read_bytes()
        journals.append({"path": str(journal), "sha256": hashlib.sha256(contents).hexdigest()})
        for index, line in enumerate(contents.decode("utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = tuple(row[field] for field in IDENTITY)
            where = f"{journal}:{index}"
            if any(type(value) is not int for value in key) or key not in expected or key in rows:
                raise ValueError(f"Unexpected or repeated episode identity at {where}: {key}")
            digest, rep = expected[key]
            row_rep = row.get("repetition_index", None if requires_repetition_index else rep)
            if (row["initial_state_sha256"] != digest or row_rep != rep
                    or (requires_repetition_index and type(row_rep) is not int)
                    or row.get("policy_path", "deployed") != "deployed"
                    or row.get("terminal_noise_seed") is not None):
                raise ValueError(f"Episode differs from the frozen deployment panel at {where}")
            success = row["success"]
            steps = row["policy_env_steps"]
            if (type(success) is not bool or type(steps) is not int or not 1 <= steps <= HORIZON
                    or (not success and steps != HORIZON)
                    or row.get("status", "success" if success else "timeout") != (
                        "success" if success else "timeout")
                    or row.get("settling_env_steps", SETTLING) != SETTLING
                    or row.get("post_done_policy_steps", 0) != 0):
                raise ValueError(f"Invalid success, termination or control-step counts at {where}")
            # The benchmark stops on the first success, so the executed steps
            # equal first-success time even in journals without a separate field.
            first_success = steps if success else None
            if row.get("first_success_step", first_success) != first_success:
                raise ValueError(f"Inconsistent first-success time at {where}")
            rows[key] = (success, steps)
    if rows.keys() != expected.keys():
        raise ValueError(f"Policy has {len(rows)} of {len(expected)} frozen episodes")
    shape = (len(panel.tasks), len(panel.trials), panel.repetitions)
    success, steps = np.asarray([rows[key] for key in expected], dtype=np.int64).T
    return success.reshape(shape).astype(bool), steps.reshape(shape), journals


def _ci(samples):
    return np.quantile(samples, [0.025, 0.975], axis=0).T.tolist()


def _spread(values):
    return {
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values),
        "maximum": max(values),
    }


def _time_summary(success, steps):
    successful = steps[success]
    return {
        "mean_first_success_steps": float(successful.mean()) if successful.size else None,
        "median_first_success_steps": float(np.median(successful)) if successful.size else None,
        "mean_penalized_first_success_steps": float(np.where(success, steps, HORIZON).mean()),
        "policy_env_steps": int(steps.sum()),
        "settling_env_steps": SETTLING * steps.size,
        "post_done_policy_steps": 0,
    }


def _policy_result(condition, seed, success, steps, journals, samples, panel):
    per_task = {}
    intervals = _ci(samples)
    for index, task in enumerate(panel.tasks):
        wins, task_steps = success[index], steps[index]
        per_task[str(task)] = {
            "unique_initial_states": len(panel.trials), "episodes": wins.size, "successes": int(wins.sum()),
            "success_rate": float(wins.mean()), "paired_state_cluster_ci95": intervals[index],
            **_time_summary(wins, task_steps),
        }
    return {
        "condition": condition, "training_seed": seed, "journals": journals, "per_task": per_task,
        "macro_success_rate": float(success.mean()), "paired_state_cluster_ci95": _ci(samples.mean(axis=1)),
        "unique_task_initial_state_pairs": len(panel.tasks) * len(panel.trials), "episodes": success.size,
        "successes": int(success.sum()), **_time_summary(success, steps),
    }


def _contrast(left, right, role, scores, bootstraps, tasks):
    by_seed, task_diffs, task_bootstraps = [], [], []
    for seed in TRAINING_SEEDS:
        a = (left, seed)
        b = (right, None if right == "B0" else seed)
        success_left, steps_left = scores[a]
        success_right, steps_right = scores[b]
        gains = success_left & ~success_right
        losses = success_right & ~success_left
        task_delta = success_left.mean(axis=(1, 2)) - success_right.mean(axis=(1, 2))
        task_diffs.append(task_delta)
        task_bootstraps.append(bootstraps[a] - bootstraps[b])
        by_seed.append({
            "training_seed": seed, "macro_success_rate_difference": float(task_delta.mean()),
            "mean_penalized_first_success_steps_difference": float((steps_left - steps_right).mean()),
            "per_task": {
                str(task): {"success_rate_difference": float(task_delta[index]),
                            "paired_new_successes": int(gains[index].sum()),
                            "paired_new_failures": int(losses[index].sum())}
                for index, task in enumerate(tasks)
            },
        })
    average = np.mean(task_diffs, axis=0)
    sampled = np.mean(task_bootstraps, axis=0)
    task_ci = _ci(sampled)
    worst = int(average.argmin())
    return {
        "left": left, "right": right, "role": role,
        "macro_success_rate_difference": float(average.mean()),
        "paired_state_cluster_ci95": _ci(sampled.mean(axis=1)),
        "mean_penalized_first_success_steps_difference": statistics.mean(
            row["mean_penalized_first_success_steps_difference"] for row in by_seed
        ),
        "training_seed_variability": _spread([row["macro_success_rate_difference"] for row in by_seed]),
        "by_training_seed": by_seed,
        "per_task": {
            str(task): {"success_rate_difference": float(average[index]),
                        "paired_state_cluster_ci95": task_ci[index]}
            for index, task in enumerate(tasks)
        },
        "worst_task": {"task_id": tasks[worst], "success_rate_difference": float(average[worst])},
        "substantial_drop_threshold": SUBSTANTIAL_DROP,
        "substantial_drop_task_fraction": float(np.mean(average <= SUBSTANTIAL_DROP + 1e-12)),
    }


def _analyze_panel(entries, order, root, split_path, panel, replicates):
    expected = _expected_panel(split_path, panel)
    # Draw initial-state clusters independently inside each fixed task. Every
    # policy and training seed uses the same draws and all repetitions per state.
    rng = np.random.default_rng(panel.bootstrap_seed)
    trial_count, task_count = len(panel.trials), len(panel.tasks)
    indices = rng.integers(trial_count, size=(replicates, task_count, trial_count))
    task_axis = np.arange(task_count)[:, None]
    scores, bootstraps, policy_results, used_journals = {}, {}, [], set()
    for label in order:
        success, steps, journals = _read_policy(entries[label], root, expected, panel)
        paths = {journal["path"] for journal in journals}
        if paths & used_journals:
            raise ValueError("A journal was reused for two policy entries")
        used_journals.update(paths)
        scores[label] = success, steps
        bootstraps[label] = success.mean(axis=-1)[task_axis, indices].mean(axis=-1)
        policy_results.append(_policy_result(*label, success, steps, journals, bootstraps[label], panel))

    condition_results = {}
    for condition in CONDITIONS:
        labels = [label for label in order if label[0] == condition]
        individual = [row for row in policy_results if row["condition"] == condition]
        values = [row["macro_success_rate"] for row in individual]
        sampled = np.mean([bootstraps[label] for label in labels], axis=0)
        condition_results[condition] = {
            "macro_success_rate": statistics.mean(values),
            "paired_state_cluster_ci95": _ci(sampled.mean(axis=1)),
            "training_seed_variability": _spread(values) if condition != "B0" else None,
            "per_task_success_rate": {
                str(task): statistics.mean(row["per_task"][str(task)]["success_rate"] for row in individual)
                for task in panel.tasks
            },
        }

    return {
        "panel_role": panel.role, "task_ids": panel.tasks, "trial_ids": panel.trials,
        "repetitions_per_state": panel.repetitions, "horizon_policy_steps": HORIZON,
        "interval": {"method": "paired_initial_state_cluster_percentile_bootstrap",
                     "confidence_level": 0.95, "replicates": replicates, "rng_seed": panel.bootstrap_seed,
                     "tasks": "fixed; stratified by task", "training_seeds": "fixed; equally weighted"},
        "policies": policy_results, "conditions": condition_results,
        "contrasts": {
            f"{left}-{right}": _contrast(left, right, role, scores, bootstraps, panel.tasks)
            for left, right, role in panel.contrasts
        },
    }


def analyze(manifest_path, split_path=DEFAULT_SPLIT, *, replicates=BOOTSTRAP_REPLICATES):
    if type(replicates) is not int or replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    manifest_path = Path(manifest_path).resolve()
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    entries = {}
    for entry in manifest["policies"]:
        condition, seed = entry["condition"], entry["seed"]
        if (condition not in CONDITIONS or (condition == "B0" and seed is not None)
                or (condition != "B0" and (type(seed) is not int or seed not in TRAINING_SEEDS))
                or (condition, seed) in entries):
            raise ValueError(f"Unexpected or repeated policy condition and seed: {condition}, {seed}")
        entries[condition, seed] = entry
    order = [("B0", None), *((condition, seed) for condition in TRAINED for seed in TRAINING_SEEDS)]
    if entries.keys() != set(order):
        raise ValueError("Expected one B0 and all three training seeds for each trained condition")
    with_source_a = [SOURCE_A_PANEL.journal_key in entry for entry in entries.values()]
    if any(with_source_a) and not all(with_source_a):
        raise ValueError("Independent source A analysis requires a journal for every policy")
    root, split_path = manifest_path.parent, Path(split_path)
    result = {
        "schema_version": 1, "protocol_id": "xvla-20261110", "suite": "libero_spatial",
        "input_manifest_sha256": hashlib.sha256(contents).hexdigest(), "split_sha256": SPLIT_SHA256,
        **_analyze_panel(entries, order, root, split_path, H_PANEL, replicates),
    }
    if all(with_source_a):
        result["source_a"] = _analyze_panel(entries, order, root, split_path, SOURCE_A_PANEL, replicates)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON policy and journal manifest")
    parser.add_argument("--output", required=True, type=Path, help="New JSON analysis file")
    parser.add_argument("--split", default=DEFAULT_SPLIT, type=Path, help="Path to the frozen split")
    args = parser.parse_args(argv)
    result = analyze(args.input, args.split)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
