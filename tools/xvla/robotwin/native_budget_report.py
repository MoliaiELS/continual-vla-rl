"""Report the five-task Easy cohort evaluated with each task's native budget."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re

if __package__:
    from .native_budget import EPISODES, EVALUATION_COMMIT, NATIVE_LIMITS, PROTOCOL
    from .summarize import (
        BUDGET_FIELDS, EPISODE_STATUSES, IDENTITY, Evidence, append_attempt,
        cohort_stats, format_interval, markdown_row, pct, require,
    )
else:
    from native_budget import EPISODES, EVALUATION_COMMIT, NATIVE_LIMITS, PROTOCOL
    from summarize import (
        BUDGET_FIELDS, EPISODE_STATUSES, IDENTITY, Evidence, append_attempt,
        cohort_stats, format_interval, markdown_row, pct, require,
    )


CATALOG = Path(__file__).with_name("tasks.json")
SOURCE_PROTOCOL = "robotwin-native-b0-v2"
SOURCE_FINGERPRINT = "78fa1f315e850c366ae0e36250f3cb633c6667747ebdfbb0608cee0656320f46"
QUERY_LIMITS = {task: math.ceil(limit / 30) for task, limit in NATIVE_LIMITS.items()}
SOURCE_IDENTITY = (*IDENTITY, "source_episode_id", "source_manifest_sha256")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_profile(evidence):
    path = evidence.root / "profile.json"
    profile = evidence.read(path)
    require(profile is not None, path, "prepare all 900 frozen initial states first")
    expected = {
        "protocol": PROTOCOL, "source_protocol": SOURCE_PROTOCOL,
        "source_protocol_fingerprint": SOURCE_FINGERPRINT,
        "source_evaluation_commit": EVALUATION_COMMIT,
        "native_action_limits": NATIVE_LIMITS, "native_query_limits": QUERY_LIMITS,
        "original_max_queries": 10, "original_max_anchors": 300, "action_chunk_size": 30,
        "planned_episodes": 900, "planned_first_batch": 500, "planned_extension": 400,
    }
    require(all(profile.get(key) == value for key, value in expected.items()), path, "unexpected cohort definition")
    extra_fields = {
        "protocol_fingerprint", "planned_episodes", "planned_first_batch", "planned_extension",
        "source_manifest_sha256", "source_panel", "implementation_commit",
    }
    basis = {key: value for key, value in profile.items() if key not in extra_fields}
    fingerprint = profile.get("protocol_fingerprint")
    require(fingerprint == digest(basis) and fingerprint != SOURCE_FINGERPRINT, path, "invalid new protocol fingerprint")
    hashes = profile.get("source_manifest_sha256", {})
    require(set(hashes) == set(EPISODES), path, "profile must bind all five source manifests")
    require(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()), path, "invalid source manifest digest")
    return profile


def verify_source_manifest(evidence, task, expected_hash):
    path = evidence.root / "formal" / task / "easy/seed_manifest.json"
    manifest = evidence.read(path)
    require(manifest is not None, path, "missing frozen source seed manifest")
    evidence.context(manifest, task, "easy", "formal", path)
    require(manifest["protocol"] == SOURCE_PROTOCOL, path, "unexpected source protocol")
    require(manifest["protocol_fingerprint"] == SOURCE_FINGERPRINT, path, "unexpected source fingerprint")
    require(evidence.files[path.relative_to(evidence.root).as_posix()]["sha256"] == expected_hash, path, "source manifest differs from profile")
    seeds = manifest["episodes"]
    require(manifest["limit"] == len(seeds) == EPISODES[task], path, "incomplete source initial-state panel")
    actual, requested, signatures, ids = set(), set(), set(), set()
    for index, seed in enumerate(seeds):
        require(seed["index"] == index and seed["batch"] == index // 100 + 1, path, "source ordering or batch differs")
        require(seed["episode_id"] == f"{SOURCE_PROTOCOL}/formal/{task}/easy/{index + 1:04d}", path, "unexpected source episode ID")
        require(all(type(seed[key]) is int for key in ("index", "batch", "actual_seed", "requested_seed", "policy_seed")), path, "invalid seed type")
        require(seed["actual_seed"] == seed["requested_seed"], path, "source automatically changed seed")
        require(seed["scene_signature"] == digest(seed["scene"]), path, "source initial scene hash differs")
        require(seed["episode_id"] not in ids and seed["actual_seed"] not in actual
                and seed["requested_seed"] not in requested and seed["scene_signature"] not in signatures,
                path, "repeated source episode or initial state")
        ids.add(seed["episode_id"])
        actual.add(seed["actual_seed"])
        requested.add(seed["requested_seed"])
        signatures.add(seed["scene_signature"])
    return manifest


def verify_manifest(evidence, task, profile, source):
    path = evidence.root / "formal" / task / "easy/seed_manifest.json"
    manifest = evidence.read(path)
    require(manifest is not None, path, "missing frozen native-budget seed manifest")
    evidence.context(manifest, task, "easy", "formal", path)
    expected = {
        "protocol": PROTOCOL, "protocol_fingerprint": profile["protocol_fingerprint"],
        "source_protocol": SOURCE_PROTOCOL, "source_protocol_fingerprint": SOURCE_FINGERPRINT,
        "source_manifest_sha256": profile["source_manifest_sha256"][task],
        "native_step_limit": NATIVE_LIMITS[task], "native_max_queries": QUERY_LIMITS[task],
        "original_max_anchors": 300, "limit": EPISODES[task],
    }
    require(all(manifest.get(key) == value for key, value in expected.items()), path, "manifest differs from native-budget profile")
    seeds = manifest["episodes"]
    require(len(seeds) == len(source["episodes"]), path, "native-budget seed count differs")
    selected = {}
    for index, (seed, original) in enumerate(zip(seeds, source["episodes"])):
        require(seed["episode_id"] == f"{PROTOCOL}/formal/{task}/easy/{index + 1:04d}", path, "unexpected native-budget episode ID")
        require(seed["source_episode_id"] == original["episode_id"], path, "source episode ID changed")
        require(seed["source_manifest_sha256"] == expected["source_manifest_sha256"], path, "source manifest binding differs")
        require(all(seed.get(key) == value for key, value in original.items() if key != "episode_id"),
                path, "native-budget seed, policy seed, or scene changed")
        selected[seed["episode_id"]] = seed
    return selected


def read_attempts(evidence, task, selected, definition, fingerprint, *, native):
    path = evidence.root / "formal" / task / "easy/episodes.jsonl"
    attempts = defaultdict(list)
    # The first read fixes a journal prefix while live workers append later rows.
    for where, row in evidence.read(path, journal=True):
        evidence.context(row, task, "easy", "formal", where)
        require(row["protocol"] == (PROTOCOL if native else SOURCE_PROTOCOL), where, "unexpected result protocol")
        require(row["protocol_fingerprint"] == fingerprint, where, "result fingerprint differs")
        episode = row.get("episode_id")
        require(episode in selected, where, "result lies outside the frozen initial-state panel")
        identity = SOURCE_IDENTITY if native else IDENTITY
        require(all(row.get(key) == selected[episode][key] for key in identity), where, "result initial state differs from manifest")
        require(row.get("instruction") == definition["prompt"], where, "logged instruction differs")
        for key in BUDGET_FIELDS:
            value = row.get(key)
            require(value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0), where, f"invalid {key}")
        query_limit = QUERY_LIMITS[task] if native else 10
        query, sent, anchors = (row.get(key, 0) for key in ("queries", "anchors_sent", "anchors"))
        require(all(type(value) is int and value >= 0 for value in (query, sent, anchors)), where, "invalid query or anchor count")
        require(query <= query_limit and anchors <= min(NATIVE_LIMITS[task], sent)
                and sent <= 30 * query and sent <= 30 * query_limit,
                where, "execution exceeds the frozen native budget")
        if row.get("step_lim") is not None:
            require(row["step_lim"] == NATIVE_LIMITS[task], where, "step limit differs from native task")
        if row.get("status") in {"success", "failure"}:
            require(row.get("step_lim") == NATIVE_LIMITS[task], where, "completed result lacks native step limit")
        if row.get("status") == "success":
            require(row.get("reason") == "official_success" and row.get("official_success_evaluated") is True,
                    where, "success lacks official success evaluation")
        append_attempt(attempts[episode], row, EPISODE_STATUSES, where)
    return attempts


def panels(selected, attempts, count):
    return {
        "batches": {str(batch): cohort_stats(100, selected, attempts, batch) for batch in range(1, count // 100 + 1)},
        "cumulative": cohort_stats(count, selected, attempts),
    }


def comparison(new, old):
    complete = new["status"] == old["status"] == "complete"
    return {
        "status": "complete" if complete else "awaiting_complete_cohorts",
        "planned": new["planned"], "native_completed": new["finished"], "source_completed": old["finished"],
        "success_rate_difference_pp": 100 * (new["success_rate"] - old["success_rate"]) if complete else None,
    }


def overview(cohorts):
    require(bool(cohorts), "overview", "empty cohort list")
    result = {key: sum(row[key] for row in cohorts) for key in (
        "planned", "frozen", "finished", "successes", "failures", "unresolved", "retry_pending",
        "not_frozen", "not_started", "technical_attempts", "retries", "attempt_wall_seconds",
    )}
    rates = [row["success_rate"] for row in cohorts if row["status"] == "complete"]
    result.update(
        status="complete" if result["finished"] == result["planned"] else "incomplete",
        planned_tasks=len(cohorts), complete_tasks=len(rates),
        complete_subset_task_weighted_sr=sum(rates) / len(rates) if rates else None,
        all_planned_tasks_weighted_sr=sum(rates) / len(rates) if len(rates) == len(cohorts) else None,
        planned_bounds=[result["successes"] / result["planned"],
                        (result["successes"] + result["planned"] - result["finished"]) / result["planned"]],
    )
    return result


def summarize(input_root, source_panel):
    """Audit the frozen 900-state profile and count each terminal episode once."""
    input_root, source_panel = Path(input_root).resolve(), Path(source_panel).resolve()
    require(input_root != source_panel and not input_root.is_relative_to(source_panel), input_root, "new protocol requires a separate output root")
    require(source_panel.name == EVALUATION_COMMIT, source_panel, "expected the original fixed evaluation panel")
    evidence, baseline = Evidence(input_root), Evidence(source_panel)
    profile = verify_profile(evidence)
    task_dirs = set(path.name for path in (input_root / "formal").iterdir() if path.is_dir())
    require(task_dirs == set(EPISODES), input_root, "expected exactly the five native-budget tasks")
    catalog = json.loads(CATALOG.read_bytes())
    definitions = {row["task_id"]: row for row in catalog["tasks"]}
    tasks = []
    for task, count in EPISODES.items():
        definition = definitions[task]
        require(definition["step_limit"] == NATIVE_LIMITS[task], CATALOG, "catalog budget differs")
        require({path.name for path in (input_root / "formal" / task).iterdir() if path.is_dir()} == {"easy"}, input_root, "native-budget cohort is Easy only")
        source = verify_source_manifest(baseline, task, profile["source_manifest_sha256"][task])
        selected = verify_manifest(evidence, task, profile, source)
        original = {row["episode_id"]: row for row in source["episodes"]}
        attempts = read_attempts(evidence, task, selected, definition, profile["protocol_fingerprint"], native=True)
        old_attempts = read_attempts(baseline, task, original, definition, SOURCE_FINGERPRINT, native=False)
        native_panel, source_panel_stats = panels(selected, attempts, count), panels(original, old_attempts, count)
        shard = input_root / "formal" / task / "easy"
        runtime = evidence.read(shard / "runtime_manifest.json")
        require(not attempts or runtime is not None, shard, "attempted episodes lack a runtime manifest")
        if runtime is not None:
            evidence.context(runtime, task, "easy", "formal", shard / "runtime_manifest.json")
            require(runtime["protocol_fingerprint"] == profile["protocol_fingerprint"], shard, "runtime fingerprint differs")
            require(runtime["native_step_limit"] == NATIVE_LIMITS[task] and runtime["max_queries"] == QUERY_LIMITS[task], shard, "runtime budget differs")
            key = (shard / "seed_manifest.json").relative_to(input_root).as_posix()
            require(runtime["seed_manifest_sha256"] == evidence.files[key]["sha256"], shard, "runtime seed panel differs")
        tasks.append({
            "task_id": task, "description_zh": definition["description_zh"], "category": definition["category"],
            "native_step_limit": NATIVE_LIMITS[task], "native_max_queries": QUERY_LIMITS[task],
            **native_panel, "source_300_anchor_baseline": source_panel_stats,
            "comparison": {
                "first_batch": comparison(native_panel["batches"]["1"], source_panel_stats["batches"]["1"]),
                "cumulative": comparison(native_panel["cumulative"], source_panel_stats["cumulative"]),
            },
        })
    extended = [task for task in tasks if EPISODES[task["task_id"]] == 300]
    totals = {
        "first_batch": overview([task["batches"]["1"] for task in tasks]),
        "extension": overview([task["batches"][str(batch)] for task in extended for batch in (2, 3)]),
        "cumulative": overview([task["cumulative"] for task in tasks]),
    }
    extension = totals["extension"]
    extension["planned_batches"] = extension.pop("planned_tasks")
    extension["complete_batches"] = extension.pop("complete_tasks")
    extension.pop("complete_subset_task_weighted_sr")
    extension.pop("all_planned_tasks_weighted_sr")
    return {
        "schema": 1, "label": "five-task native-budget baseline calibration", "condition": "easy",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "status": totals["cumulative"]["status"], "protocol": PROTOCOL,
        "protocol_fingerprint": profile["protocol_fingerprint"], "evaluation_commit": EVALUATION_COMMIT,
        "source_protocol": SOURCE_PROTOCOL, "source_protocol_fingerprint": SOURCE_FINGERPRINT,
        "profile": profile, "overall": totals, "tasks": tasks,
        "evidence_files": evidence.files, "source_evidence_files": baseline.files,
    }


def rate_cell(data):
    return f"{pct(data['success_rate'])}（{format_interval(data['wilson_95'])}）" if data["finished"] else "—"


def result_cell(data):
    return f"{data['successes']}/{data['finished']}（计划 {data['planned']}）"


def emit_markdown(report, stream):
    stream.write("# X‑VLA 五任务 Easy 原生预算评测\n\n")
    stream.write(f"协议 `{report['protocol']}`，指纹 `{report['protocol_fingerprint']}`。复用原固定初态和 policy seed，模型与原生成功判定保持固定。\n\n")
    stream.write("评测对象为冻结的官方 RoboTwin 适配基座 `2toINF/X-VLA-RoboTwin2@a157c580cfe6f9f445614490f3bec1b2f9ef9f18`，本轮用于 baseline calibration／能力审计。\n\n")
    stream.write("计划包括五项首批各 100 回合，以及微波炉开门和闹钟触碰各追加 200 回合，共 900 个独立初态。以下成功率来自本协议实际完成的回合，95% 区间使用 Wilson 方法。\n\n")
    stream.write("| 范围 | 完成／计划 | 成功 | 失败 | 未运行 | 待重试 | 未解决 |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for scope, label in (("first_batch", "首批"), ("extension", "新增扩测"), ("cumulative", "累计")):
        data = report["overall"][scope]
        stream.write(markdown_row((label, f"{data['finished']}/{data['planned']}", data["successes"], data["failures"], data["not_started"], data["retry_pending"], data["unresolved"])))
    stream.write("\n## 首批每任务 100 回合\n\n")
    stream.write("| 任务 | 原生锚点／query 上限 | 成功／完成（计划） | SR（95% Wilson） | 原 300 锚点预算 | 完整队列差值 |\n| --- | ---: | ---: | --- | --- | ---: |\n")
    for task in report["tasks"]:
        current = task["batches"]["1"]
        source = task["source_300_anchor_baseline"]["batches"]["1"]
        delta = task["comparison"]["first_batch"]["success_rate_difference_pp"]
        stream.write(markdown_row((
            f"{task['description_zh']} `{task['task_id']}`", f"{task['native_step_limit']} / {task['native_max_queries']}",
            result_cell(current), rate_cell(current), f"{source['successes']}/{source['finished']} · {pct(source['success_rate'])}" if source["finished"] else "—",
            f"{delta:+.1f} pp" if delta is not None else "待两队列完成",
        )))
    stream.write("\n## 扩测分批及累计\n\n")
    stream.write("| 任务 | 批次 | 成功／完成（计划） | SR（95% Wilson） | 原 300 锚点预算 | 完整队列差值 |\n| --- | --- | ---: | --- | --- | ---: |\n")
    for task in report["tasks"]:
        if EPISODES[task["task_id"]] != 300:
            continue
        for batch in ("2", "3", "cumulative"):
            current = task["cumulative"] if batch == "cumulative" else task["batches"][batch]
            source_panel = task["source_300_anchor_baseline"]
            source = source_panel["cumulative"] if batch == "cumulative" else source_panel["batches"][batch]
            delta = comparison(current, source)["success_rate_difference_pp"]
            stream.write(markdown_row((task["description_zh"], "累计" if batch == "cumulative" else f"批次 {batch}", result_cell(current), rate_cell(current),
                                      f"{source['successes']}/{source['finished']} · {pct(source['success_rate'])}" if source["finished"] else "—",
                                      f"{delta:+.1f} pp" if delta is not None else "待两队列完成")))
    stream.write("\n## 完成情况及执行预算\n\n")
    stream.write("| 任务 | 技术异常尝试／重试 | 未解决 | 未完成结果范围 | 平均 query | 平均实际／发送锚点 | 平均模拟秒／耗时秒 |\n| --- | ---: | ---: | --- | ---: | --- | --- |\n")
    for task in report["tasks"]:
        data = task["cumulative"]
        budget = data["mean_budget"]
        number = lambda field: "—" if budget[field]["mean"] is None else f"{budget[field]['mean']:.2f}"
        stream.write(markdown_row((task["description_zh"], f"{data['technical_attempts']} / {data['retries']}", data["unresolved"],
                                  format_interval(data["planned_bounds"]) if data["status"] != "complete" else "已完成",
                                  number("queries"), f"{number('anchors')} / {number('anchors_sent')}", f"{number('simulated_seconds')} / {number('wall_seconds')}")))
    first = report["overall"]["first_batch"]
    stream.write(f"\n首批已完成 {first['complete_tasks']}/5 个完整任务，完成子集的任务等权均值为 {pct(first['complete_subset_task_weighted_sr'])}。")
    if first["complete_tasks"] == 5:
        stream.write(f"五任务首批均值为 {pct(first['all_planned_tasks_weighted_sr'])}。")
    stream.write("扩测样本量不改变任务权重。与原预算的差值在相同计划初态队列都完成后计算。独立仿真重置存在数值变化，该差值描述本次两组实测。\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-panel", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output not in (args.input.resolve(), args.source_panel.resolve())
            and not output.is_relative_to(args.source_panel.resolve())
            and not output.is_relative_to(args.input.resolve() / "formal"), output, "report output must be outside the raw episode directories")
    report = summarize(args.input, args.source_panel)
    output.mkdir(parents=True, exist_ok=True)
    temporary = output / "summary.json.tmp"
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output / "summary.json")
    with (output / "report.md.tmp").open("w", encoding="utf-8") as stream:
        emit_markdown(report, stream)
    (output / "report.md.tmp").replace(output / "report.md")
    data = report["overall"]["cumulative"]
    print(f"Native-budget Easy: {data['finished']}/{data['planned']} completed, {data['successes']} successes, {data['unresolved']} unresolved")


if __name__ == "__main__":
    main()
