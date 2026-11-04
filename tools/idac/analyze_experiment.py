"""Analyze counted episodes and actual budgets without ML dependencies.

Usage: python tools/idac/analyze_experiment.py SUITE_DIRECTORY [--strict]

plan.json contains task_ids and runs [{run_id, method, seed, task_ids,
expected_policy_env_steps, experiment_log_path}], plus evaluations [{run_id,
stage, policy_env_steps, evidence_dir, task_ids, expected_episodes_per_task,
checkpoint, eval_seed, role}]. eval_manifest.json may replace the evaluations
list with the final attempt locations. Relative paths resolve against SUITE.
Each evidence_dir contains eval_summary.json or cumulative env_rank_*.json.
The shared run_id='sft', stage=0 supplies the baseline for all trained runs.
Missing or partial evidence is reported; it never becomes a zero success cell.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SUCCESS_FIELDS = ("success_ever", "success_final")


def wilson_interval(successes, total, z=1.959963984540054):
    if total <= 0:
        return None
    if not 0 <= successes <= total:
        raise ValueError("Success count must lie between zero and episode count")
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def summarize_episodes(episodes):
    tasks = {}
    for episode in episodes:
        task = str(episode["task_id"])
        stats = tasks.setdefault(task, {"episodes": 0, "success_ever": 0,
                                         "success_final": 0, "policy_env_steps": 0})
        stats["episodes"] += 1
        stats["policy_env_steps"] += int(episode["policy_env_steps"])
        for field in SUCCESS_FIELDS:
            if not isinstance(episode[field], bool):
                raise ValueError(f"Episode {field} must be boolean")
            stats[field] += int(episode[field])
    for stats in tasks.values():
        for field in SUCCESS_FIELDS:
            stats[field + "_rate"] = stats[field] / stats["episodes"]
            stats[field + "_wilson95"] = wilson_interval(stats[field], stats["episodes"])
    return tasks


def _path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_evaluation(root, entry):
    """Read one evaluation attempt, retaining partial results and discrepancies."""
    path = _path(root, entry["evidence_dir"])
    summary_path = path / "eval_summary.json"
    files = [summary_path] if summary_path.is_file() else sorted(path.glob("env_rank_*.json"))
    issues, episodes, counters = [], [], defaultdict(int)
    if not files:
        return {"complete": False, "missing": True, "per_task": {},
                "issues": ["Evaluation evidence is not available"], "evidence_dir": str(path)}
    for source in files:
        data = _read_json(source)
        episodes.extend(data.get("episodes", []))
        totals = data.get("totals", data)
        for field in ("policy_env_steps", "settling_env_steps", "post_done_executed_steps"):
            value = totals.get(field, totals.get("post_success_executed_steps", 0) if field == "post_done_executed_steps" else 0)
            counters[field] += int(value)
    tasks = summarize_episodes(episodes)
    wanted = [str(t) for t in entry.get("task_ids", tasks)]
    expected = entry.get("expected_episodes_per_task")
    for task in wanted:
        count = tasks.get(task, {}).get("episodes", 0)
        expected_count = expected.get(task, expected.get(int(task)) if task.isdigit() else None) if isinstance(expected, dict) else expected
        if expected_count is not None and count != int(expected_count):
            issues.append(f"Task {task}: expected {expected_count} episodes, found {count}")
        elif count == 0:
            issues.append(f"Task {task}: no completed episodes")
    unexpected = set(tasks) - set(wanted)
    if unexpected:
        issues.append(f"Unexpected evaluated tasks: {sorted(unexpected)}")
    episode_steps = sum(ep["policy_env_steps"] for ep in episodes)
    if counters["policy_env_steps"] != episode_steps:
        issues.append(f"Episode steps {episode_steps} differ from counter {counters['policy_env_steps']}")
    return {"complete": bool(episodes) and not issues, "missing": False, "per_task": tasks,
            "episode_count": len(episodes), "counters": dict(counters), "issues": issues,
            "evidence_dir": str(path)}


def _estimate(value, missing):
    return {"value": None if missing else value, "complete": not missing, "missing_cells": missing}


def continual_metrics(matrix, task_order, field, sequential=True):
    """Use explicit stage/task cells; retain signed drops and absent cells."""
    tasks = [str(task) for task in task_order]
    n = len(tasks)

    def score(stage, task):
        cell = matrix.get(str(stage), {}).get(task)
        return cell.get(field + "_rate") if cell and cell.get("complete", True) else None

    missing_final = [[n, task] for task in tasks if score(n, task) is None]
    final_values = [score(n, task) for task in tasks]
    avg = None if missing_final or not tasks else sum(final_values) / n
    result = {"final_average": _estimate(avg, missing_final),
              "final_worst_task": _estimate(min(final_values) if not missing_final and tasks else None, missing_final)}
    if not sequential or n < 2:
        for name in ("nbt_final", "nbt_time", "fwt"):
            result[name] = {"value": None, "complete": False,
                            "reason": "Requires at least two sequential task stages"}
        return result
    final_drops, missing_nbt = [], []
    for j, task in enumerate(tasks[:-1], 1):
        diagonal, final = score(j, task), score(n, task)
        for stage, value in ((j, diagonal), (n, final)):
            if value is None:
                missing_nbt.append([stage, task])
        if diagonal is not None and final is not None:
            final_drops.append(diagonal - final)
    result["nbt_final"] = _estimate(sum(final_drops) / (n - 1), missing_nbt)
    result["per_task_final_drop"] = {
        task: score(j, task) - score(n, task)
        for j, task in enumerate(tasks[:-1], 1)
        if score(j, task) is not None and score(n, task) is not None}
    time_drops, missing_time = [], []
    for j, task in enumerate(tasks[:-1], 1):
        diagonal = score(j, task)
        if diagonal is None:
            missing_time.append([j, task])
        later = []
        for stage in range(j + 1, n + 1):
            value = score(stage, task)
            if value is None:
                missing_time.append([stage, task])
            elif diagonal is not None:
                later.append(diagonal - value)
        if len(later) == n - j:
            time_drops.append(sum(later) / (n - j))
    result["nbt_time"] = _estimate(sum(time_drops) / (n - 1), missing_time)
    fwt_values, missing_fwt = [], []
    for j, task in enumerate(tasks[1:], 2):
        baseline, previous = score(0, task), score(j - 1, task)
        for stage, value in ((0, baseline), (j - 1, previous)):
            if value is None:
                missing_fwt.append([stage, task])
        if baseline is not None and previous is not None:
            fwt_values.append(previous - baseline)
    result["fwt"] = _estimate(sum(fwt_values) / (n - 1), missing_fwt)
    return result


def fixed_panel_auc(matrix, budgets, task_ids, field, total_budget):
    """Normalized trapezoidal AUC on a fixed task panel, requiring B=0 and B=total."""
    tasks = [str(task) for task in task_ids]
    missing, points = [], []
    for stage, budget in sorted(budgets.items(), key=lambda item: item[1]):
        row, values = matrix.get(str(stage), {}), []
        for task in tasks:
            cell = row.get(task)
            if not cell or not cell.get("complete", True):
                missing.append([int(stage), task])
            else:
                values.append(cell[field + "_rate"])
        if len(values) == len(tasks) and tasks:
            points.append((int(budget), sum(values) / len(tasks)))
    if total_budget is None or total_budget <= 0:
        return {"value": None, "complete": False, "reason": "No positive declared training budget"}
    if not points or points[0][0] != 0 or points[-1][0] != total_budget:
        missing.append(["budget_endpoints", total_budget])
    if len({point[0] for point in points}) != len(points):
        missing.append(["duplicate_budget", None])
    area = sum((x2 - x1) * (y1 + y2) / 2 for (x1, y1), (x2, y2) in zip(points, points[1:]))
    return {**_estimate(area / total_budget, missing), "points": points, "task_panel": tasks,
            "interpolation": "trapezoidal_between_predeclared_checkpoints"}


def audit_training(root, run):
    if run.get("method", "").lower() == "sft" or run.get("run_id") == "sft":
        return {"complete": True, "expected_policy_env_steps": 0, "actual_policy_env_steps": 0,
                "iterations": 0, "issues": []}
    log = run.get("experiment_log_path")
    if not log and run.get("training_dir"):
        log = str(Path(run["training_dir"]) / "experiment.jsonl")
    expected = run.get("expected_policy_env_steps")
    if not log or not _path(root, log).is_file():
        return {"complete": False, "expected_policy_env_steps": expected,
                "actual_policy_env_steps": None, "issues": ["Training iteration ledger is unavailable"]}
    records = [json.loads(line) for line in _path(root, log).read_text().splitlines() if line.strip()]
    issues, previous, actual, stage_budgets = [], 0, None, {}
    for record in records:
        counters = record.get("cumulative_budget", record.get("budget", record))
        value = counters.get("policy_env_steps")
        if value is None:
            continue
        value = int(value)
        if value < previous:
            issues.append("Cumulative policy budget moved backwards")
        previous, actual = value, value
        if record.get("event") == "iteration" and record.get("stage_end"):
            stage_budgets[str(record["stage"])] = value
    if actual is None:
        issues.append("No cumulative policy step counter in training ledger")
    if expected is not None and actual != int(expected):
        issues.append(f"Expected training policy steps {expected}, found {actual}")
    if not any(record.get("event") == "complete" for record in records):
        issues.append("Training ledger has no successful completion event")
    return {"complete": not issues and actual is not None, "expected_policy_env_steps": expected,
            "actual_policy_env_steps": actual,
            "stage_policy_env_steps": stage_budgets,
            "iterations": sum(record.get("event") == "iteration" for record in records), "issues": issues}


def analyze(root):
    root = Path(root)
    plan = _read_json(root / "plan.json")
    entries = plan.get("evaluations", [])
    if (root / "eval_manifest.json").is_file():
        manifest = _read_json(root / "eval_manifest.json")
        entries = manifest.get("evaluations", []) if isinstance(manifest, dict) else manifest
    reports, cells, budget_sets = [], defaultdict(lambda: defaultdict(dict)), defaultdict(lambda: defaultdict(set))
    checkpoints = defaultdict(set)
    seen = set()
    for entry in entries:
        variant = str(entry.get("variant", "id"))
        role = str(entry.get("role", "development"))
        identity = (variant, role, entry["run_id"], int(entry["stage"]), str(_path(root, entry["evidence_dir"])))
        if identity in seen:
            raise ValueError(f"Duplicate evaluation attempt in manifest: {identity}")
        seen.add(identity)
        report = {**entry, **load_evaluation(root, entry)}
        reports.append(report)
        if not report["per_task"]:
            continue
        run_id, stage = str(entry["run_id"]), str(entry["stage"])
        bucket = run_id if variant == "id" else f"variant:{variant}:role:{role}:run:{run_id}"
        for task, stats in report["per_task"].items():
            # A task_key disambiguates same numeric task IDs in different suites/variants.
            task_key = str(entry.get("task_key", task))
            key = (bucket, stage, task_key)
            checkpoints[key].add(str(entry.get("checkpoint", "")))
            target = cells[bucket][stage].setdefault(task_key, {"episodes": 0, "success_ever": 0,
                                                              "success_final": 0, "complete": True})
            for field in ("episodes", *SUCCESS_FIELDS):
                target[field] += stats[field]
            target["complete"] &= report["complete"]
            if len(checkpoints[key]) > 1:
                target["complete"] = False
                report["complete"] = False
                report["issues"].append("Different checkpoints assigned to the same matrix cell")
        if entry.get("policy_env_steps") is not None:
            budget_sets[bucket][stage].add(int(entry["policy_env_steps"]))
    for rows in cells.values():
        for row in rows.values():
            for cell in row.values():
                for field in SUCCESS_FIELDS:
                    cell[field + "_rate"] = cell[field] / cell["episodes"]
                    cell[field + "_wilson95"] = wilson_interval(cell[field], cell["episodes"])
    baselines = cells.get("sft", {}).get("0", {})
    runs = []
    for run in plan.get("runs", []):
        run_id = str(run["run_id"])
        matrix = dict(cells.get(run_id, {}))
        matrix["0"] = {**baselines, **matrix.get("0", {})}
        budgets = {stage: next(iter(values)) for stage, values in budget_sets[run_id].items() if len(values) == 1}
        budgets["0"] = 0
        budget_conflicts = {stage: sorted(values) for stage, values in budget_sets[run_id].items() if len(values) > 1}
        order = run.get("task_ids", plan.get("task_ids", []))
        audit = audit_training(root, run)
        actual_stage_budgets = audit.get("stage_policy_env_steps", {})
        for stage, actual in actual_stage_budgets.items():
            if stage in budgets and budgets[stage] != actual:
                budget_conflicts[stage] = [budgets[stage], actual]
            budgets[stage] = actual
        sequential = str(run.get("method", "")).lower().startswith("seq")
        metrics = {}
        for field in SUCCESS_FIELDS:
            metrics[field] = continual_metrics(matrix, order, field, sequential=sequential)
            metrics[field]["auc"] = fixed_panel_auc(matrix, budgets, order, field, run.get("expected_policy_env_steps"))
            if budget_conflicts:
                metrics[field]["auc"] = {"value": None, "complete": False,
                                         "reason": "Conflicting matrix-stage budgets", "conflicts": budget_conflicts}
            elif not audit["complete"]:
                metrics[field]["auc"] = {"value": None, "complete": False,
                                         "reason": "Actual completed training budget has not been verified"}
        runs.append({**run, "matrix": matrix, "stage_policy_env_steps": budgets,
                     "metrics": metrics, "training_budget": audit})
    variants = []
    variant_roles = {(str(entry.get("variant", "id")), str(entry.get("role", "development")))
                     for entry in entries if entry.get("variant", "id") != "id"}
    for variant, role in sorted(variant_roles):
        baseline = cells.get(f"variant:{variant}:role:{role}:run:sft", {}).get("0", {})
        for run in plan.get("runs", []):
            if run.get("run_id") == "sft":
                continue
            matrix = cells.get(f"variant:{variant}:role:{role}:run:{run['run_id']}", {})
            final_stage = str(len(run.get("task_ids", plan.get("task_ids", []))))
            final = matrix.get(final_stage, {})
            tasks = {}
            for task in sorted(set(baseline) | set(final)):
                before, after = baseline.get(task), final.get(task)
                complete = bool(before and after and before["complete"] and after["complete"])
                tasks[task] = {"baseline": before, "final": after, "complete": complete,
                               "delta": {field: after[field + "_rate"] - before[field + "_rate"]
                                         if complete else None for field in SUCCESS_FIELDS}}
            variants.append({"variant": variant, "role": role, "run_id": run["run_id"], "method": run.get("method"),
                             "seed": run.get("seed"), "final_stage": int(final_stage), "per_task": tasks,
                             "complete": bool(tasks) and all(task["complete"] for task in tasks.values())})
    incomplete_evals = sum(not item["complete"] for item in reports)
    return {"schema_version": 1, "suite": str(root), "evaluation_attempts": reports, "runs": runs,
            "variant_evaluations": variants, "exposure": plan.get("exposure", {}),
            "complete": bool(reports) and bool(runs) and incomplete_evals == 0
                        and all(run["training_budget"]["complete"] for run in runs),
            "incomplete_evaluations": incomplete_evals,
            "caution": "Wilson intervals describe episode sampling only; aggregate training-seed variation separately."}


def write_report(root, report):
    root = Path(root)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output / "success_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run_id", "method", "seed", "stage", "policy_env_steps", "task_id", "episodes",
                         "success_ever", "success_final", "ever_rate", "final_rate", "complete"])
        for run in report["runs"]:
            for stage, row in sorted(run["matrix"].items(), key=lambda item: int(item[0])):
                for task, cell in row.items():
                    writer.writerow([run["run_id"], run.get("method"), run.get("seed"), stage,
                                     run["stage_policy_env_steps"].get(stage), task, cell["episodes"],
                                     cell["success_ever"], cell["success_final"], cell["success_ever_rate"],
                                     cell["success_final_rate"], cell["complete"]])
    with (output / "variant_rates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "role", "run_id", "task_id", "baseline_episodes", "final_episodes",
                         "baseline_ever", "final_ever", "delta_ever", "baseline_final", "final_final", "delta_final", "complete"])
        for item in report["variant_evaluations"]:
            for task, scores in item["per_task"].items():
                before, after = scores["baseline"] or {}, scores["final"] or {}
                writer.writerow([item["variant"], item["role"], item["run_id"], task,
                                 before.get("episodes"), after.get("episodes"),
                                 before.get("success_ever_rate"), after.get("success_ever_rate"), scores["delta"]["success_ever"],
                                 before.get("success_final_rate"), after.get("success_final_rate"), scores["delta"]["success_final"],
                                 scores["complete"]])
    lines = ["# Continual VLA experiment results", "", f"Complete: {report['complete']}. "
             f"Incomplete evaluations: {report['incomplete_evaluations']}.", "",
             "All success rates and signed NBT values use fractions, not percentages.", "",
             "| Run | Criterion | Final mean | NBT final | NBT time | FWT | AUC |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for run in report["runs"]:
        for field, metrics in run["metrics"].items():
            values = []
            for name in ("final_average", "nbt_final", "nbt_time", "fwt", "auc"):
                value = metrics[name].get("value")
                values.append("pending" if value is None else f"{value:.4f}")
            lines.append("| " + " | ".join([run["run_id"], field, *values]) + " |")
    lines.extend(["", report["caution"], "", "Partial evidence remains visible in success_matrix.csv; "
                  "missing or inconsistent cells do not produce completed metric estimates."])
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero while expected artifacts are incomplete")
    args = parser.parse_args()
    report = analyze(args.suite)
    output = write_report(args.suite, report)
    print(json.dumps({"analysis_dir": str(output), "complete": report["complete"],
                      "incomplete_evaluations": report["incomplete_evaluations"]}))
    if args.strict and not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
