"""Audit the 50-task Hard evaluation with each task's native action budget."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

if __package__:
    from .summarize import (
        BUDGET_FIELDS, EPISODE_STATUSES, IDENTITY, Evidence, append_attempt,
        cohort_stats, format_interval, frozen_seeds, markdown_row, pct, require, screening,
    )
else:
    from summarize import (
        BUDGET_FIELDS, EPISODE_STATUSES, IDENTITY, Evidence, append_attempt,
        cohort_stats, format_interval, frozen_seeds, markdown_row, pct, require, screening,
    )


CATALOG = Path(__file__).with_name("tasks.json")
PROTOCOL = "robotwin-native-b0-hard-native-budget-v1"
SOURCE_PROTOCOL = "robotwin-native-b0-v2"
SOURCE_FINGERPRINT = "78fa1f315e850c366ae0e36250f3cb633c6667747ebdfbb0608cee0656320f46"
EVALUATION_COMMIT = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
RESULT_IDENTITY = (*IDENTITY, "source_episode_id", "source_manifest_sha256")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def catalog():
    value = json.loads(CATALOG.read_bytes())
    definitions = {row["task_id"]: row for row in value["tasks"]}
    priority = set(value["priority_tasks"])
    require(len(definitions) == len(value["tasks"]) == 50 and len(priority) == 10
            and priority <= definitions.keys(), CATALOG, "expected the official 50 tasks and ten extension tasks")
    return value, definitions, priority


def verify_profile(evidence, definitions, priority):
    path = evidence.root / "profile.json"
    profile = evidence.read(path)
    require(profile is not None, path, "prepare the Hard evaluation plan first")
    limits = {task: row["step_limit"] for task, row in definitions.items()}
    queries = {task: math.ceil(limit / 30) for task, limit in limits.items()}
    counts = {task: 300 if task in priority else 100 for task in definitions}
    expected = {
        "protocol": PROTOCOL, "condition": "hard", "source_protocol": SOURCE_PROTOCOL,
        "source_protocol_fingerprint": SOURCE_FINGERPRINT, "source_evaluation_commit": EVALUATION_COMMIT,
        "task_ids": list(definitions),
        "native_action_limits": limits, "native_query_limits": queries, "per_task_episodes": counts,
        "planned_episodes": 7000, "planned_first_batch": 5000, "planned_extension": 2000,
        "original_max_queries": 10, "original_max_anchors": 300, "action_chunk_size": 30,
        "model_id": "2toINF/X-VLA-RoboTwin2", "model_revision": "a157c580cfe6f9f445614490f3bec1b2f9ef9f18",
    }
    require(all(profile.get(key) == value for key, value in expected.items()), path, "Hard profile differs from the frozen benchmark plan")
    require(set(profile.get("priority_tasks", [])) == priority, path, "extension task selection differs")
    extra_fields = {"protocol_fingerprint", "planned_episodes", "planned_first_batch", "planned_extension",
                    "source_panel", "implementation_commit"}
    basis = {key: value for key, value in profile.items() if key not in extra_fields}
    require(profile.get("protocol_fingerprint") == digest(basis)
            and profile["protocol_fingerprint"] != SOURCE_FINGERPRINT,
            path, "invalid Hard protocol fingerprint")
    return profile


def source_panel(evidence, task, limits):
    shard = evidence.root / "formal" / task / "hard"
    path = shard / "seed_manifest.json"
    manifest = evidence.read(path)
    accepted, prescreen = screening(evidence, shard, task, "hard", "formal")
    require(evidence.protocols <= {SOURCE_PROTOCOL} and evidence.fingerprints <= {SOURCE_FINGERPRINT},
            shard, "source prescreen belongs to another evaluation protocol")
    selected = frozen_seeds(evidence, path, manifest, task, "hard", "formal", limits, accepted)
    if manifest is not None:
        require(manifest["protocol"] == SOURCE_PROTOCOL and manifest["protocol_fingerprint"] == SOURCE_FINGERPRINT,
                path, "source manifest protocol differs")
        for seed in selected.values():
            require(seed["episode_id"] == f"{SOURCE_PROTOCOL}/formal/{task}/hard/{seed['index'] + 1:04d}", path, "unexpected source episode identity")
            require(seed["actual_seed"] == seed["requested_seed"] and seed["scene_signature"] == digest(seed["scene"]),
                    path, "source scene or executed seed differs")
    return manifest, selected, prescreen


def new_seeds(evidence, path, manifest, source, task, profile):
    if manifest is None:
        return {}
    evidence.context(manifest, task, "hard", "formal", path)
    require(source is not None, path, "native-budget manifest lacks a frozen source panel")
    source_manifest_path = f"formal/{task}/hard/seed_manifest.json"
    source_hash = profile["_source_files"][source_manifest_path]["sha256"]
    expected = {
        "protocol": PROTOCOL, "protocol_fingerprint": profile["protocol_fingerprint"],
        "source_protocol": SOURCE_PROTOCOL, "source_protocol_fingerprint": SOURCE_FINGERPRINT,
        "source_manifest_sha256": source_hash, "limit": profile["per_task_episodes"][task],
        "native_step_limit": profile["native_action_limits"][task],
        "native_max_queries": profile["native_query_limits"][task], "original_max_anchors": 300,
    }
    require(all(manifest.get(key) == value for key, value in expected.items()), path, "native-budget manifest differs from profile or source")
    require(len(manifest["episodes"]) == len(source["episodes"]), path, "native-budget initial-state count differs")
    selected = {}
    for index, (seed, original) in enumerate(zip(manifest["episodes"], source["episodes"])):
        require(seed["episode_id"] == f"{PROTOCOL}/formal/{task}/hard/{index + 1:04d}", path, "unexpected native-budget episode ID")
        require(seed["source_episode_id"] == original["episode_id"] and seed["source_manifest_sha256"] == source_hash,
                path, "source episode binding differs")
        require(all(seed.get(key) == value for key, value in original.items() if key != "episode_id"),
                path, "scene or environment/policy seed changed")
        selected[seed["episode_id"]] = seed
    return selected


def attempts_from_rows(evidence, rows, selected, definition, profile):
    task = definition["task_id"]
    cap, query_cap = profile["native_action_limits"][task], profile["native_query_limits"][task]
    attempts = defaultdict(list)
    for where, row in rows:
        evidence.context(row, task, "hard", "formal", where)
        require(row["protocol"] == PROTOCOL and row["protocol_fingerprint"] == profile["protocol_fingerprint"],
                where, "result belongs to another protocol")
        episode = row.get("episode_id")
        require(episode in selected and all(row.get(key) == selected[episode][key] for key in RESULT_IDENTITY),
                where, "result differs from the frozen initial state")
        require(row.get("instruction") == definition["prompt"], where, "logged language differs")
        for key in BUDGET_FIELDS:
            value = row.get(key)
            require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
                    where, f"invalid {key}")
        queries, sent, advanced = (row.get(key, 0) for key in ("queries", "anchors_sent", "anchors"))
        require(all(type(value) is int and value >= 0 for value in (queries, sent, advanced)), where, "invalid execution counter")
        require(queries <= query_cap and advanced <= min(cap, sent) and sent <= 30 * queries,
                where, "execution exceeds the native task budget")
        if row.get("step_lim") is not None or row.get("status") in {"success", "failure"}:
            require(row.get("step_lim") == cap, where, "recorded native step limit differs")
        if row.get("status") == "success":
            require(row.get("reason") == "official_success" and row.get("official_success_evaluated") is True,
                    where, "success lacks the official predicate evaluation")
        append_attempt(attempts[episode], row, EPISODE_STATUSES, where)
    return attempts


def task_report(evidence, baseline, definition, profile):
    task = definition["task_id"]
    count = profile["per_task_episodes"][task]
    limits = {batch: 100 for batch in range(1, count // 100 + 1)}
    shard = evidence.root / "formal" / task / "hard"
    rows = evidence.read(shard / "episodes.jsonl", journal=True)
    runtime = evidence.read(shard / "runtime_manifest.json")
    path = shard / "seed_manifest.json"
    manifest = evidence.read(path)
    source, original, prescreen = source_panel(baseline, task, limits)
    selected = new_seeds(evidence, path, manifest, source, task, profile)
    attempts = attempts_from_rows(evidence, rows, selected, definition, profile)
    require(not attempts or runtime is not None, shard, "attempted episodes lack a runtime manifest")
    if runtime is not None:
        evidence.context(runtime, task, "hard", "formal", shard / "runtime_manifest.json")
        require(runtime["protocol"] == PROTOCOL and runtime["protocol_fingerprint"] == profile["protocol_fingerprint"],
                shard, "runtime fingerprint differs")
        cap, queries = profile["native_action_limits"][task], profile["native_query_limits"][task]
        require(runtime.get("step_lim") == cap and runtime.get("native_max_advanced_anchors") == cap
                and runtime.get("max_queries") == queries and runtime.get("max_anchors_sent") == 30 * queries,
                shard, "runtime budget differs")
        key = path.relative_to(evidence.root).as_posix()
        require(runtime["seed_manifest_sha256"] == evidence.files.get(key, {}).get("sha256"),
                shard, "runtime initial-state panel differs")
    cumulative = cohort_stats(count, selected, attempts)
    stage = "native_manifest_frozen" if manifest is not None else "source_manifest_frozen" if source is not None else "prescreening" if prescreen["candidates"] else "not_started"
    return {
        "task_id": task, "description_zh": definition["description_zh"], "category": definition["category"],
        "tags": definition["tags"], "stage": "complete" if cumulative["status"] == "complete" else stage,
        "native_step_limit": profile["native_action_limits"][task],
        "native_max_queries": profile["native_query_limits"][task],
        "source_frozen": len(original), "prescreen": prescreen,
        "batches": {str(batch): cohort_stats(100, selected, attempts, batch) for batch in limits},
        "cumulative": cumulative,
    }


def overview(cohorts):
    totals = {key: sum(row[key] for row in cohorts) for key in (
        "planned", "frozen", "finished", "successes", "failures", "unresolved", "retry_pending",
        "not_frozen", "not_started", "technical_attempts", "retries", "attempt_wall_seconds",
    )}
    rates = [row["success_rate"] for row in cohorts if row["status"] == "complete"]
    mean = sum(rates) / len(rates) if rates else None
    totals.update(status="complete" if totals["finished"] == totals["planned"] else "incomplete",
                  planned_tasks=len(cohorts), complete_tasks=len(rates),
                  complete_subset_task_weighted_sr=mean,
                  all_planned_tasks_weighted_sr=mean if len(rates) == len(cohorts) else None,
                  planned_bounds=[totals["successes"] / totals["planned"],
                                  (totals["successes"] + totals["planned"] - totals["finished"]) / totals["planned"]])
    return totals


def summarize(input_root, source_panel_root):
    """Count Hard policy outcomes while retaining unfrozen tasks in the plan."""
    input_root, source_root = Path(input_root).resolve(), Path(source_panel_root).resolve()
    require(source_root.name == EVALUATION_COMMIT and input_root != source_root
            and not input_root.is_relative_to(source_root), input_root, "use a separate Hard protocol output and the original fixed source panel")
    _, definitions, priority = catalog()
    evidence, baseline = Evidence(input_root), Evidence(source_root)
    profile = verify_profile(evidence, definitions, priority)
    task_root = input_root / "formal"
    directories = {path.name for path in task_root.iterdir() if path.is_dir()} if task_root.is_dir() else set()
    require(directories <= definitions.keys(), task_root, "unknown task in the Hard output")
    for task in directories:
        require({path.name for path in (task_root / task).iterdir() if path.is_dir()} <= {"hard"},
                task_root / task, "Hard profile contains another condition")
    # Source files are added lazily as the serial prescreen freezes each task.
    bound_profile = {**profile, "_source_files": baseline.files}
    tasks = [task_report(evidence, baseline, definition, bound_profile) for definition in definitions.values()]
    extension = overview([task["batches"][str(batch)] for task in tasks if task["task_id"] in priority for batch in (2, 3)])
    extension["planned_batches"] = extension.pop("planned_tasks")
    extension["complete_batches"] = extension.pop("complete_tasks")
    extension.pop("complete_subset_task_weighted_sr")
    extension.pop("all_planned_tasks_weighted_sr")
    totals = {"first_batch": overview([task["batches"]["1"] for task in tasks]), "extension": extension,
              "cumulative": overview([task["cumulative"] for task in tasks])}
    groups = defaultdict(list)
    for task in tasks:
        groups[task["category"]].append(task["batches"]["1"])
    prescreen = {key: sum(task["prescreen"][key] for task in tasks) for key in (
        "candidates", "accepted", "excluded", "technical_pending", "technical_attempts", "retries", "attempt_wall_seconds",
    )}
    return {
        "schema": 1, "label": "Hard native-budget baseline calibration", "condition": "hard",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(), "status": totals["cumulative"]["status"],
        "protocol": PROTOCOL, "protocol_fingerprint": profile["protocol_fingerprint"], "profile": profile,
        "source_evaluation_commit": EVALUATION_COMMIT, "source_protocol": SOURCE_PROTOCOL,
        "source_protocol_fingerprint": SOURCE_FINGERPRINT, "overall": totals, "prescreen": prescreen,
        "categories_first_batch": {name: overview(rows) for name, rows in groups.items()}, "tasks": tasks,
        "evidence_files": evidence.files, "source_evidence_files": baseline.files,
    }


def result_cell(row):
    return f"{row['successes']}/{row['finished']}（计划 {row['planned']}）"


def rate_cell(row):
    return f"{pct(row['success_rate'])}（{format_interval(row['wilson_95'])}）" if row["finished"] else "—"


def emit_markdown(report, stream):
    stream.write("# X‑VLA RoboTwin Hard 原生预算评测\n\n")
    stream.write(f"协议 `{report['protocol']}`，指纹 `{report['protocol_fingerprint']}`。50 个任务各 100 个初态，十个预定重点任务再各扩测 200 个初态，累计计划 7,000 回合。\n\n")
    stream.write("使用冻结的官方 RoboTwin 适配基座 `2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18`。各任务保留原生环境上限，模型查询上限为锚点上限除以 30 后向上取整。初态经固定原生专家预筛后逐任务冻结。\n\n")
    stream.write("| 范围 | 完成／计划 | 成功 | 失败 | 未冻结 | 已冻结未运行 | 待重试 | 未解决 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for key, label in (("first_batch", "首批"), ("extension", "新增扩测"), ("cumulative", "累计")):
        row = report["overall"][key]
        stream.write(markdown_row((label, f"{row['finished']}/{row['planned']}", row["successes"], row["failures"], row["not_frozen"], row["not_started"], row["retry_pending"], row["unresolved"])))
    screen = report["prescreen"]
    stream.write(f"\n专家预筛已检查 {screen['candidates']} 个候选，接受 {screen['accepted']}、排除 {screen['excluded']}，{screen['technical_pending']} 个候选待处理技术异常。待冻结的计划回合保留在未完成数量内。\n\n")
    stream.write("## 每任务首批 100 回合\n\n")
    stream.write("| Task ID | 中文任务内容 | 原生锚点／query 上限 | 成功／完成（计划） | SR（95% Wilson） | 未冻结／未运行／待重试／未解决 |\n| --- | --- | ---: | ---: | --- | --- |\n")
    for task in report["tasks"]:
        row = task["batches"]["1"]
        stream.write(markdown_row((task["task_id"], task["description_zh"], f"{task['native_step_limit']} / {task['native_max_queries']}",
                                  result_cell(row), rate_cell(row), f"{row['not_frozen']} / {row['not_started']} / {row['retry_pending']} / {row['unresolved']}")))
    stream.write("\n## 预定重点任务扩测与累计\n\n")
    stream.write("| 任务 | 批次 | 成功／完成（计划） | SR（95% Wilson） | 未解决 |\n| --- | --- | ---: | --- | ---: |\n")
    for task in report["tasks"]:
        if "2" not in task["batches"]:
            continue
        for batch in ("2", "3", "cumulative"):
            row = task["cumulative"] if batch == "cumulative" else task["batches"][batch]
            stream.write(markdown_row((task["description_zh"], "累计" if batch == "cumulative" else batch, result_cell(row), rate_cell(row), row["unresolved"])))
    stream.write("\n## 首批主类别统计\n\n")
    stream.write("| 主类别 | 完成任务数 | 完成子集等权 SR | 全类别等权 SR |\n| --- | ---: | ---: | ---: |\n")
    for category, row in report["categories_first_batch"].items():
        stream.write(markdown_row((category, f"{row['complete_tasks']}/{row['planned_tasks']}", pct(row["complete_subset_task_weighted_sr"]), pct(row["all_planned_tasks_weighted_sr"]))))
    first = report["overall"]["first_batch"]
    stream.write(f"\n首批完整任务 {first['complete_tasks']}/50，完成子集的任务等权均值为 {pct(first['complete_subset_task_weighted_sr'])}。完整 50 任务的等权均值为 {pct(first['all_planned_tasks_weighted_sr'])}。每个任务权重相同，扩测不会增加其权重。\n")
    stream.write("\n`summary.json` 保留每批与累计的独立初态数、Wilson 区间、失败原因、技术尝试、重试、执行预算、耗时和未完成结果范围。本轮属于 baseline calibration／能力审计。\n")


def atomic_text(path, text):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    target = args.output.resolve()
    require(target not in (args.input.resolve(), args.source_panel.resolve())
            and not target.is_relative_to(args.input.resolve() / "formal")
            and not target.is_relative_to(args.source_panel.resolve()), target, "reports must be outside raw input directories")
    report = summarize(args.input, args.source_panel)
    target.mkdir(parents=True, exist_ok=True)
    text = io.StringIO()
    emit_markdown(report, text)
    atomic_text(target / "summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    atomic_text(target / "report.md", text.getvalue())
    row = report["overall"]["cumulative"]
    print(f"Hard native-budget: {row['finished']}/{row['planned']} completed, {row['not_frozen']} not frozen, {row['unresolved']} unresolved")


if __name__ == "__main__":
    main()
