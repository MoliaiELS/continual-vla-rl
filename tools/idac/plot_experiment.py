"""Create standalone research plots from counted, audited experiment analysis.

Usage: python tools/idac/plot_experiment.py SUITE_OR_SUMMARY [--output-dir DIR]
       python tools/idac/plot_experiment.py --self-test

Requires matplotlib/numpy only when rendering. Missing/partial cells stay gray;
learning curves use recorded stage budgets, never declared nominal step counts.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tempfile


FIELDS = ("success_ever", "success_final")


def load_report(source):
    source = Path(source)
    if source.is_dir():
        source = source / "summary.json" if (source / "summary.json").is_file() else source / "analysis/summary.json"
    report = json.loads(source.read_text(encoding="utf-8-sig"))
    if report.get("schema_version") != 1:
        raise ValueError("Unsupported analysis summary schema")
    # The CSV is a reviewable second representation; reject stale/mixed inputs.
    matrix_csv = source.parent / "success_matrix.csv"
    if matrix_csv.is_file():
        runs = {str(run["run_id"]): run for run in report["runs"]}
        with matrix_csv.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                cell = runs[row["run_id"]]["matrix"][row["stage"]][row["task_id"]]
                complete = str(row["complete"]).lower() == "true"
                if complete != bool(cell.get("complete", True)):
                    raise ValueError("Matrix CSV completeness differs from summary JSON")
                for csv_key, field in (("ever_rate", "success_ever"), ("final_rate", "success_final")):
                    if not math.isclose(float(row[csv_key]), float(cell[field + "_rate"]), abs_tol=1e-12):
                        raise ValueError("Matrix CSV success rate differs from summary JSON")
                if int(row["episodes"]) != int(cell["episodes"]):
                    raise ValueError("Matrix CSV episode count differs from summary JSON")
    return source.resolve(), report


def cell_rate(cell, field):
    if not cell or not cell.get("complete", True) or int(cell.get("episodes", 0)) <= 0:
        return None
    value = cell.get(field + "_rate")
    if value is None or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        return None
    return float(value)


def audited_budgets(run):
    """Keep only counters from the actual training ledger (plus baseline zero)."""
    audit = run.get("training_budget", {})
    budgets = {"0": 0}
    if any(run.get("metrics", {}).get(field, {}).get("auc", {}).get("conflicts") for field in FIELDS):
        return budgets
    for stage, value in audit.get("stage_policy_env_steps", {}).items():
        if int(stage) > 0 and isinstance(value, (int, float)) and math.isfinite(value) and value > 0:
            budgets[str(stage)] = int(value)
    # A completed ledger also verifies the final counter, even if an older
    # ledger format did not export intermediate stage-end counters.
    final = audit.get("actual_policy_env_steps")
    if audit.get("complete") and isinstance(final, (int, float)) and math.isfinite(final) and final > 0:
        budgets[str(len(run.get("task_ids", [])))] = int(final)
    return budgets


def measured_points(run, field, panel=None):
    panel = [str(task) for task in (panel if panel is not None else run.get("task_ids", []))]
    if not panel:
        return []
    points = []
    for stage, budget in sorted(audited_budgets(run).items(), key=lambda item: (item[1], int(item[0]))):
        row = run.get("matrix", {}).get(stage, {})
        values = [cell_rate(row.get(task), field) for task in panel]
        if all(value is not None for value in values):
            points.append({"stage": int(stage), "budget": budget, "value": statistics.mean(values)})
    # Duplicate budgets cannot identify a unique checkpoint; omit ambiguous
    # positions rather than selecting an arbitrary stage or averaging them.
    return [point for point in points if sum(other["budget"] == point["budget"] for other in points) == 1]


def aggregate_curves(runs, field, task=None):
    """Group matching task panels/methods, preserving available-seed counts."""
    grouped = {}
    seen_seeds = set()
    for run in runs:
        panel = tuple(sorted(str(value) for value in run.get("task_ids", [])))
        if task is not None and str(task) not in panel:
            continue
        method = str(run.get("method", run["run_id"]))
        seed = str(run.get("seed", run["run_id"]))
        identity = (panel, method, seed)
        if identity in seen_seeds:
            raise ValueError(f"Duplicate training seed in method/panel aggregation: {identity}")
        seen_seeds.add(identity)
        for point in measured_points(run, field, [task] if task is not None else None):
            key = (panel, method, point["budget"])
            grouped.setdefault(key, []).append({**point, "seed": seed})
    result = []
    for (panel, method, budget), points in sorted(grouped.items()):
        values = [point["value"] for point in points]
        stages = sorted({point["stage"] for point in points})
        result.append({"panel": list(panel), "method": method, "budget": budget,
                       "mean": statistics.mean(values),
                       "std": statistics.stdev(values) if len(values) > 1 else None,
                       "n": len(values), "seeds": sorted(point["seed"] for point in points),
                       "stage": stages[0] if len(stages) == 1 else None,
                       "criterion": field, "task": None if task is None else str(task)})
    return result


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:100]


def _panel_name(panel):
    digest = hashlib.sha256(",".join(panel).encode()).hexdigest()[:8]
    return f"panel_{len(panel)}_{digest}"


def _draw_curves(axis, records, colors, show_n=True):
    methods = sorted({record["method"] for record in records})
    for method in methods:
        points = sorted((record for record in records if record["method"] == method), key=lambda point: point["budget"])
        if not points:
            continue
        color = colors[method]
        counts = sorted({point["n"] for point in points if point["budget"] > 0}) or [points[0]["n"]]
        count_label = str(counts[0]) if len(counts) == 1 else f"{counts[0]}-{counts[-1]}"
        axis.plot([point["budget"] for point in points], [point["mean"] * 100 for point in points],
                  linestyle="none", marker="o", markersize=4, color=color,
                  label=f"{method} (n={count_label})")
        for point in points:
            if point["std"] is not None:
                axis.errorbar(point["budget"], point["mean"] * 100,
                              yerr=[[min(point["std"], point["mean"]) * 100],
                                    [min(point["std"], 1 - point["mean"]) * 100]],
                              fmt="none", ecolor=color, capsize=2, elinewidth=.8, alpha=.8)
        # Joining only neighboring measured stages prevents a sparse long
        # experiment's baseline/final points looking like a full learning curve.
        for first, second in zip(points, points[1:]):
            adjacent = (first["stage"] is not None and second["stage"] == first["stage"] + 1)
            if not adjacent:
                continue
            xs = [first["budget"], second["budget"]]
            means = [first["mean"] * 100, second["mean"] * 100]
            axis.plot(xs, means, linewidth=1.5, color=color)
            if first["std"] is not None and second["std"] is not None:
                axis.fill_between(xs,
                                  [max(0, point["mean"] - point["std"]) * 100 for point in (first, second)],
                                  [min(1, point["mean"] + point["std"]) * 100 for point in (first, second)],
                                  color=color, alpha=.14, linewidth=0)
        if show_n and len(counts) > 1:
            for point in points:
                axis.annotate(str(point["n"]), (point["budget"], point["mean"] * 100),
                              xytext=(0, 6 + methods.index(method) * 3), textcoords="offset points",
                              ha="center", fontsize=6, color=color)
    axis.set_ylim(-2, 103)
    axis.set_ylabel("Success (%)")
    axis.set_xlabel("Actual policy control steps")
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axis.grid(alpha=.22)


def render(report, source, output_dir, dpi=180, pdf=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    runs = report.get("runs", [])
    methods = sorted({str(run.get("method", run["run_id"])) for run in runs})
    colors = {method: plt.get_cmap("tab10")(index % 10) for index, method in enumerate(methods)}
    files, curves = [], []

    def save(figure, filename, **metadata):
        path = output_dir / (filename + ".png")
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
        files.append({"path": path.name, **metadata})
        if pdf:
            pdf_path = output_dir / (filename + ".pdf")
            figure.savefig(pdf_path, bbox_inches="tight")
            files.append({"path": pdf_path.name, **metadata})
        plt.close(figure)

    for run in runs:
        panel = [str(task) for task in run.get("task_ids", [])]
        if not panel:
            continue
        stages = list(range(len(panel) + 1))
        for field in FIELDS:
            values = np.full((len(stages), len(panel)), np.nan)
            for row_index, stage in enumerate(stages):
                for column, task in enumerate(panel):
                    value = cell_rate(run.get("matrix", {}).get(str(stage), {}).get(task), field)
                    if value is not None:
                        values[row_index, column] = value * 100
            figure, axis = plt.subplots(figsize=(max(6, .6 * len(panel) + 2), max(4, .43 * len(stages) + 1.5)))
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad("#dddddd")
            rendered = axis.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100, aspect="auto")
            axis.set_xticks(range(len(panel)), [f"T{task}" for task in panel])
            budgets = audited_budgets(run)
            labels = ["SFT (0 steps)" if stage == 0 else
                      f"Stage {stage} ({budgets[str(stage)]:,})" if str(stage) in budgets else
                      f"Stage {stage} (budget unverified)" for stage in stages]
            axis.set_yticks(range(len(stages)), labels)
            axis.set_xlabel("Evaluation task")
            axis.set_ylabel("Checkpoint / actual control steps")
            status = "" if run.get("training_budget", {}).get("complete") else " [partial evidence]"
            axis.set_title(f"{run['run_id']}: {field.removeprefix('success_')}{status}")
            if values.size <= 144:
                for row, column in np.ndindex(values.shape):
                    if np.isfinite(values[row, column]):
                        axis.text(column, row, f"{values[row, column]:.0f}", ha="center", va="center",
                                  fontsize=7, color="black" if values[row, column] > 60 else "white")
            figure.colorbar(rendered, ax=axis, label="Success (%)", fraction=.045, pad=.03)
            axis.legend(handles=[Patch(facecolor="#dddddd", label="Missing / incomplete")],
                        loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False, fontsize=8)
            save(figure, f"heatmap_{_safe_name(run['run_id'])}_{field.removeprefix('success_')}",
                 kind="heatmap", run_id=run["run_id"], criterion=field)

    panels = sorted({tuple(sorted(str(task) for task in run.get("task_ids", []))) for run in runs})
    for field in FIELDS:
        aggregated = aggregate_curves(runs, field)
        curves.extend(aggregated)
        for panel in panels:
            if not panel:
                continue
            records = [record for record in aggregated if tuple(record["panel"]) == panel]
            figure, axis = plt.subplots(figsize=(7.0, 4.5))
            _draw_curves(axis, records, colors)
            if records:
                axis.legend(frameon=False, fontsize=8)
            else:
                axis.text(.5, .5, "No complete fixed-panel points with audited budgets", transform=axis.transAxes,
                          ha="center", va="center", wrap=True)
            status = "" if report.get("complete") else " [partial evidence]"
            axis.set_title(f"Fixed panel ({len(panel)} tasks): {field.removeprefix('success_')}{status}")
            figure.text(.5, -.01, "Markers: measured checkpoints. Bands: training-seed SD. n: available seeds.\n"
                        "Shared SFT baseline; missing stages are not connected. Evaluation error is separate.",
                        ha="center", fontsize=7)
            save(figure, f"learning_{_panel_name(panel)}_{field.removeprefix('success_')}",
                 kind="fixed_panel_learning", task_panel=list(panel), criterion=field)
            # Six tasks per standalone page keeps individual trajectories readable.
            for start in range(0, len(panel), 6):
                page = panel[start:start + 6]
                rows = math.ceil(len(page) / 2)
                figure, axes = plt.subplots(rows, 2, figsize=(11.5, 3.2 * rows), squeeze=False)
                handles = labels = None
                for axis, task in zip(axes.flat, page):
                    task_records = [record for record in aggregate_curves(runs, field, task)
                                    if tuple(record["panel"]) == panel]
                    _draw_curves(axis, task_records, colors, show_n=False)
                    axis.set_title(f"Task {task}")
                    if task_records and handles is None:
                        handles, labels = axis.get_legend_handles_labels()
                for axis in list(axes.flat)[len(page):]:
                    axis.set_visible(False)
                if handles:
                    figure.legend(handles, labels, loc="lower center", ncol=min(4, len(handles)),
                                  frameon=False, fontsize=8)
                figure.suptitle(f"Per-task success: {field.removeprefix('success_')} (training-seed SD){status}")
                figure.tight_layout(rect=(0, .06, 1, .95))
                save(figure, f"tasks_{_panel_name(panel)}_{field.removeprefix('success_')}_{start // 6 + 1}",
                     kind="per_task_learning", task_ids=list(page), criterion=field)
    result = {"schema_version": 1, "source_summary": str(source),
              "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              "complete": report.get("complete", False), "plots": files,
              "fixed_panel_curve_points": curves,
              "rules": ["Missing or partial scores are never zero-filled",
                        "Only actual ledger budgets appear on learning-curve x axes",
                        "Macro points require the same complete task panel",
                        "No interpolation across missing stages; no AUC estimated by plotter",
                        "SD is between available training seeds, not an episode confidence interval"]}
    (output_dir / "plot_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def self_test():
    def cell(value, complete=True):
        return {"episodes": 10, "success_ever_rate": value, "success_final_rate": value / 2,
                "complete": complete}
    first = {"run_id": "seq_1", "method": "seq", "seed": 1, "task_ids": [0, 1],
             "matrix": {"0": {"0": cell(.2), "1": cell(.4)},
                        "1": {"0": cell(.8), "1": cell(.6, complete=False)},
                        "2": {"0": cell(.5), "1": cell(.9)}},
             "stage_policy_env_steps": {"0": 0, "1": 100, "2": 200},
             "training_budget": {"complete": True, "actual_policy_env_steps": 240,
                                 "stage_policy_env_steps": {"1": 120, "2": 240}}}
    points = measured_points(first, "success_ever")
    assert [(point["stage"], point["budget"]) for point in points] == [(0, 0), (2, 240)]
    assert math.isclose(points[-1]["value"], .7)
    second = json.loads(json.dumps(first))
    second.update(run_id="seq_2", seed=2)
    second["matrix"]["2"] = {"0": cell(.9), "1": cell(.9)}
    aggregate = aggregate_curves([first, second], "success_ever")
    final = next(point for point in aggregate if point["budget"] == 240)
    assert final["n"] == 2 and math.isclose(final["mean"], .8)
    assert math.isclose(final["std"], math.sqrt(.02))
    assert len(measured_points(first, "success_ever", [0])) == 3
    unverified = {**first, "training_budget": {"complete": False}}
    assert [point["budget"] for point in measured_points(unverified, "success_ever")] == [0]
    assert cell_rate(cell(0.0), "success_ever") == 0.0
    assert cell_rate(cell(.9, complete=False), "success_ever") is None
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "summary.json"
        path.write_text(json.dumps({"schema_version": 1, "runs": [first]}), encoding="utf-8")
        with path.with_name("success_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["run_id", "stage", "task_id", "episodes", "ever_rate", "final_rate", "complete"])
            writer.writerow(["seq_1", 2, 0, 10, .5, .25, True])
        load_report(path)
        path.with_name("success_matrix.csv").write_text(
            "run_id,stage,task_id,episodes,ever_rate,final_rate,complete\nseq_1,2,0,10,0.0,0.0,True\n",
            encoding="utf-8")
        try:
            load_report(path)
        except ValueError:
            pass
        else:
            raise AssertionError("Stale CSV must be rejected")
    print("Plot data checks passed: actual budgets, partial cells, fixed panels, seed SD and stale CSV")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--pdf", action="store_true", help="Also export vector PDFs")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.source is None:
        parser.error("source is required unless --self-test is used")
    source, report = load_report(args.source)
    output = args.output_dir or source.parent / "plots"
    result = render(report, source, output, args.dpi, args.pdf)
    print(json.dumps({"plots_dir": str(output), "plots": len(result["plots"]),
                      "evidence_complete": result["complete"]}))


if __name__ == "__main__":
    main()
