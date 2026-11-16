"""Audit and summarize an X-VLA RoboTwin calibration directory (stdlib only)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

TASKS = Path(__file__).with_name("tasks.json")
CONDITIONS = ("easy", "hard")
SMOKE_TASKS = ("move_stapler_pad", "turn_switch")
SEED_FIELDS = ("requested_seed", "actual_seed", "policy_seed", "batch", "index")
IDENTITY = ("episode_id", *SEED_FIELDS, "scene_signature")
EPISODE_STATUSES = {"success", "failure", "technical_error", "unresolved"}
SCREEN_STATUSES = {"accepted", "excluded", "technical_error"}
BUDGET_FIELDS = (
    "queries",
    "anchors",
    "anchors_sent",
    "step_lim",
    "physics_ticks",
    "settling_ticks",
    "left_plan_failures",
    "right_plan_failures",
    "simulated_seconds",
    "query_seconds",
    "reset_seconds",
    "wall_seconds",
)


def wilson(successes, completed):
    """Return the two-sided 95% Wilson interval, or no estimate without data."""
    if not completed:
        return None
    p = successes / completed
    z = 1.959963984540054
    z_squared = z * z
    denominator = 1 + z_squared / completed
    center = (p + z_squared / (2 * completed)) / denominator
    margin = z * math.sqrt(
        p * (1 - p) / completed + z_squared / (4 * completed**2)
    ) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def require(condition, where, message):
    if not condition:
        raise ValueError(f"{where}: {message}")


class Evidence:
    def __init__(self, root):
        self.root = root
        self.files = {}
        self.protocols = set()
        self.fingerprints = set()

    def read(self, path, journal=False):
        if not path.exists():
            return [] if journal else None
        content = path.read_bytes()
        # The remote writer can be appending while a progress snapshot is read.
        # Do not consume an unterminated line until its newline is visible.
        trailing = journal and bool(content) and not content.endswith(b"\n")
        if trailing:
            content = content[: content.rfind(b"\n") + 1]
        self.files[path.relative_to(self.root).as_posix()] = {
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "trailing_uncommitted_bytes": trailing,
        }
        if not journal:
            return json.loads(content)
        return [
            (f"{path}:{number}", json.loads(line))
            for number, line in enumerate(content.splitlines(), 1)
            if line.strip()
        ]

    def context(self, record, task_id, condition, phase, where):
        require(record.get("task") == task_id, where, "unexpected task")
        require(str(record.get("condition")).lower() == condition, where, "unexpected condition")
        require(record.get("phase") == phase, where, "unexpected phase")
        protocol = record.get("protocol")
        require(isinstance(protocol, str) and protocol.strip(), where, "missing protocol")
        self.protocols.add(protocol)
        require(len(self.protocols) == 1, where, "multiple evaluation protocols cannot be pooled")
        fingerprint = record.get("protocol_fingerprint")
        require(
            isinstance(fingerprint, str) and re.fullmatch(r"[0-9a-f]{64}", fingerprint),
            where, "invalid protocol fingerprint",
        )
        self.fingerprints.add(fingerprint)
        require(len(self.fingerprints) == 1, where, "multiple protocol fingerprints cannot be pooled")


def append_attempt(group, row, allowed, where):
    require(row.get("status") in allowed, where, "unknown status")
    require(type(row.get("attempt")) is int, where, "invalid attempt")
    require(row["attempt"] == len(group) + 1, where, "duplicate or nonconsecutive attempt")
    require(not group or group[-1]["status"] == "technical_error", where, "retried a terminal result")
    group.append(row)


def screening(evidence, shard, task_id, condition, phase):
    path = shard / "prescreen.jsonl"
    candidates = defaultdict(list)
    for where, row in evidence.read(path, journal=True):
        evidence.context(row, task_id, condition, phase, where)
        requested = row.get("requested_seed")
        require(type(requested) is int, where, "invalid candidate seed")
        if row.get("status") == "accepted":
            require(type(row.get("actual_seed")) is int, where, "accepted candidate lacks actual seed")
        append_attempt(candidates[requested], row, SCREEN_STATUSES, where)
    accepted = {
        key: (rows[-1]["actual_seed"], rows[-1].get("scene_signature"))
        for key, rows in candidates.items()
        if rows[-1]["status"] == "accepted"
    }
    latest_counts = Counter(rows[-1]["status"] for rows in candidates.values())
    eligible = latest_counts["accepted"] + latest_counts["excluded"]
    all_attempts = [row for rows in candidates.values() for row in rows]
    technical = [row for row in all_attempts if row["status"] == "technical_error"]
    return accepted, {
        "candidates": len(candidates),
        "accepted": latest_counts["accepted"],
        "excluded": latest_counts["excluded"],
        "acceptance_rate": latest_counts["accepted"] / eligible if eligible else None,
        "technical_pending": latest_counts["technical_error"],
        "technical_attempts": len(technical),
        "retries": sum(len(rows) - 1 for rows in candidates.values()),
        "attempt_wall_seconds": sum(row.get("wall_seconds") or 0 for row in all_attempts),
        "technical_reasons": dict(Counter(row.get("reason") or "unspecified" for row in technical).most_common()),
        "exclusion_reasons": dict(Counter(
            rows[-1].get("reason") or "unspecified"
            for rows in candidates.values()
            if rows[-1]["status"] == "excluded"
        )),
    }


def frozen_seeds(evidence, manifest_path, manifest, task_id, condition, phase, batch_limits, accepted):
    if manifest is None:
        return {}
    evidence.context(manifest, task_id, condition, phase, manifest_path)
    rows = manifest["episodes"]
    limit = sum(batch_limits.values())
    require(manifest["limit"] == limit, manifest_path, "manifest target differs from planned cohort")
    require(len(rows) == limit, manifest_path, "seed list is not fully frozen")
    selected = {}
    requested, actual, signatures = set(), set(), set()
    for index, row in enumerate(rows):
        episode = row["episode_id"]
        require(isinstance(episode, str) and bool(episode), manifest_path, "invalid episode ID")
        require(episode not in selected, manifest_path, f"duplicate episode ID {episode}")
        req, act = row["requested_seed"], row["actual_seed"]
        require(
            all(type(row[key]) is int for key in SEED_FIELDS),
            manifest_path,
            f"invalid seed or batch for {episode}",
        )
        require(row["index"] == index and row["batch"] == index // 100 + 1, manifest_path, "unexpected episode ordering")
        signature = row["scene_signature"]
        require(isinstance(signature, str) and re.fullmatch(r"[0-9a-f]{64}", signature), manifest_path, "invalid initial scene signature")
        require(signature not in signatures, manifest_path, f"repeated initial scene for {episode}")
        require(req not in requested and act not in actual, manifest_path, f"repeated seed for {episode}")
        require(accepted.get(req) == (act, signature), manifest_path, f"seed/scene for {episode} lacks matching accepted prescreen entry")
        requested.add(req)
        actual.add(act)
        signatures.add(signature)
        selected[episode] = row
    require(Counter(row["batch"] for row in rows) == batch_limits, manifest_path, "incorrect batch allocation")
    return selected


def cohort_stats(planned, selected, attempts, batch=None):
    indices = [key for key, row in selected.items() if batch is None or row["batch"] == batch]
    rows = [attempts[key][-1] for key in indices if attempts.get(key)]
    all_attempts = [row for key in indices for row in attempts.get(key, ())]
    statuses = Counter(row["status"] for row in rows)
    successes, failures = statuses["success"], statuses["failure"]
    completed = successes + failures
    unresolved = statuses["unresolved"]
    ready = successes + failures + unresolved
    budgets = {}
    for key in BUDGET_FIELDS:
        values = [row[key] for row in rows if row["status"] in ("success", "failure") and row.get(key) is not None]
        budgets[key] = {"mean": sum(values) / len(values) if values else None, "n": len(values)}
    return {
        "status": "complete" if completed == planned else "incomplete",
        "planned": planned,
        "frozen": len(indices),
        "finished": completed,
        "independent_seeds": len({row["actual_seed"] for row in rows if row["status"] in ("success", "failure")}),
        "distinct_scenes": len({row["scene_signature"] for row in rows if row["status"] in ("success", "failure")}),
        "successes": successes,
        "failures": failures,
        "success_rate": successes / completed if completed else None,
        "wilson_95": wilson(successes, completed),
        "unresolved": unresolved,
        "unresolved_only_bounds": [successes / ready, (successes + unresolved) / ready] if unresolved else None,
        "planned_bounds": [successes / planned, (successes + planned - completed) / planned],
        "not_frozen": planned - len(indices),
        "not_started": len(indices) - len(rows),
        "retry_pending": statuses["technical_error"],
        "technical_attempts": sum(row["status"] in ("technical_error", "unresolved") for row in all_attempts),
        "retries": len(all_attempts) - len(rows),
        "attempt_wall_seconds": sum(row.get("wall_seconds") or 0 for row in all_attempts),
        "failure_reasons": dict(Counter(row.get("reason") or "unspecified" for row in rows if row["status"] == "failure").most_common()),
        "technical_reasons": dict(Counter(row.get("reason") or "unspecified" for row in all_attempts if row["status"] in ("technical_error", "unresolved")).most_common()),
        "mean_budget": budgets,
    }


def summarize_shard(evidence, task, condition, phase, batch_limits):
    task_id = task["task_id"]
    directory = evidence.root / phase / task_id
    matches = [path for path in directory.glob("*") if path.is_dir() and path.name.lower() == condition]
    require(len(matches) <= 1, directory, "duplicate condition directory")
    shard = matches[0] if matches else directory / condition
    # Read the most dependent journals first: preparation appends screen rows,
    # freezes the manifest, and only then can evaluation advertise results.
    results = evidence.read(shard / "episodes.jsonl", journal=True)
    runtime = evidence.read(shard / "runtime_manifest.json")
    parity = evidence.read(shard / "parity.json")
    manifest_path = shard / "seed_manifest.json"
    manifest = evidence.read(manifest_path)
    accepted, prescreen = screening(evidence, shard, task_id, condition, phase)
    selected = frozen_seeds(evidence, manifest_path, manifest, task_id, condition, phase, batch_limits, accepted)
    attempts = defaultdict(list)
    for where, row in results:
        evidence.context(row, task_id, condition, phase, where)
        episode = row.get("episode_id")
        require(episode in selected, where, "episode is outside the frozen manifest")
        require(
            all(row.get(key) == selected[episode][key] for key in IDENTITY),
            where,
            "episode identity differs from frozen manifest",
        )
        require(row.get("instruction") == task["prompt"], where, "logged language differs from task catalog")
        for key in BUDGET_FIELDS:
            value = row.get(key)
            require(
                value is None or (type(value) in (int, float) and math.isfinite(value) and value >= 0),
                where,
                f"invalid {key}",
            )
        require((row.get("queries") or 0) <= 10, where, "query budget exceeds official limit")
        if row.get("anchors_sent") is not None:
            require((row.get("anchors") or 0) <= row["anchors_sent"] <= 300, where, "invalid anchor budget")
        if row.get("step_lim") is not None:
            require(row["step_lim"] == task["step_limit"], where, "step limit differs from pinned catalog")
        append_attempt(attempts[episode], row, EPISODE_STATUSES, where)
    if runtime is not None:
        evidence.context(runtime, task_id, condition, phase, shard / "runtime_manifest.json")
        manifest_key = (shard / "seed_manifest.json").relative_to(evidence.root).as_posix()
        require(
            runtime.get("seed_manifest_sha256") == evidence.files.get(manifest_key, {}).get("sha256"),
            shard / "runtime_manifest.json", "runtime used a different seed manifest",
        )
    if parity is not None:
        evidence.context(parity, task_id, condition, phase, shard / "parity.json")
        require(parity.get("status") in ("pass", "fail"), shard / "parity.json", "invalid parity status")
    totals = cohort_stats(sum(batch_limits.values()), selected, attempts)
    return {
        "episode_ids": selected.keys(),
        "prescreen": prescreen,
        "runtime_manifest_present": runtime is not None,
        "parity_status": parity.get("status") if parity else None,
        "batches": {str(batch): cohort_stats(limit, selected, attempts, batch) for batch, limit in batch_limits.items()},
        "cumulative": totals,
        "videos": [
            {key: group[-1][key] for key in ("episode_id", "status", "video")}
            for group in attempts.values() if group and group[-1].get("video")
        ],
        "coverage": {
            "discovered": True,
            "reset_observed": bool(accepted) or any(row.get("reset_ok") is True for group in attempts.values() for row in group),
            "official_success_expert_verified": bool(accepted),
            "policy_observation_observed": any(row.get("observations_ok") is True for group in attempts.values() for row in group),
            "policy_action_observed": any((row.get("anchors") or 0) > 0 for group in attempts.values() for row in group),
            "official_policy_success_predicate_observed": any(row.get("official_success_evaluated") is True for group in attempts.values() for row in group),
            "planned_rollouts_complete": totals["status"] == "complete",
        },
    }


def panel_stats(tasks, condition, cohort):
    def data(task):
        shard = task["conditions"][condition]
        return shard["cumulative"] if cohort == "cumulative" else shard["batches"]["1"]

    cohorts = [data(task) for task in tasks]
    full = [row["success_rate"] for row in cohorts if row["status"] == "complete"]
    subset_mean = sum(full) / len(full) if full else None
    return {
        "planned_tasks": len(tasks),
        "complete_tasks": len(full),
        "complete_subset_task_weighted_sr": subset_mean,
        "all_planned_tasks_weighted_sr": subset_mean if len(full) == len(tasks) else None,
        **{key: sum(row[key] for row in cohorts) for key in (
            "planned", "frozen", "finished", "successes", "failures", "unresolved",
            "not_frozen", "not_started", "retry_pending", "technical_attempts", "retries",
        )},
    }


def future_use(task):
    roles = []
    for condition, shard in task["conditions"].items():
        if shard["batches"]["1"]["status"] != "complete":
            continue
        cohort = shard["cumulative"] if shard["cumulative"]["status"] == "complete" else shard["batches"]["1"]
        interval, rate = cohort["wilson_95"], cohort["success_rate"]
        if shard["cumulative"]["unresolved"] or shard["cumulative"]["retry_pending"]:
            roles.append(f"{condition.title()} 有未解技术异常，暂不承担主要证据")
        elif 0.30 <= rate <= 0.70:
            roles.append(f"{condition.title()} RL source 初选（复核区间、吞吐与成功轨迹）")
        elif interval[0] >= 0.90:
            roles.append(f"{condition.title()} 基线接近饱和，保留至能力保持面板")
        elif interval[0] >= 0.70:
            roles.append(f"{condition.title()} 能力保持候选（基线稳定偏高）")
        elif interval[1] < 0.30:
            roles.append(f"{condition.title()} 成功信号偏少，暂缓作为主 source")
        else:
            roles.append(f"{condition.title()} 可复核为迁移候选（主类：{task['category']}）")
    return roles or ["正式首批尚未完成，暂无循证建议"]


def summarize(input_root, tasks_path=TASKS, phase="formal", smoke_episodes=5):
    """Return a strict, versioned summary of the immutable cohort identities."""
    catalog_raw = tasks_path.read_bytes()
    catalog = json.loads(catalog_raw)
    official = catalog["tasks"]
    ids = [row["task_id"] for row in official]
    priority = set(catalog["priority_tasks"])
    require(len(ids) == len(set(ids)) == 50, tasks_path, "expected 50 distinct official tasks")
    require(len(priority) == 10 and priority <= set(ids), tasks_path, "expected 10 preregistered expansion tasks")
    require(set(SMOKE_TASKS) <= set(ids), tasks_path, "missing smoke tasks")
    evidence = Evidence(input_root)
    tasks = []
    all_episode_ids = set()
    for definition in official:
        task_id = definition["task_id"]
        if phase == "smoke" and task_id not in SMOKE_TASKS:
            continue
        limits = {1: smoke_episodes if phase == "smoke" else 100}
        if phase == "formal" and task_id in priority:
            limits.update({2: 100, 3: 100})
        task = {
            **definition,
            "priority": task_id in priority,
            "conditions": {
                condition: summarize_shard(evidence, definition, condition, phase, limits)
                for condition in CONDITIONS
            },
        }
        for shard in task["conditions"].values():
            ids_for_shard = shard.pop("episode_ids")
            require(all_episode_ids.isdisjoint(ids_for_shard), input_root, "episode ID repeated across shards")
            all_episode_ids.update(ids_for_shard)
        if phase == "formal":
            task["future_use_zh"] = future_use(task)
        tasks.append(task)
    categories, tags = defaultdict(list), defaultdict(list)
    for task in tasks:
        categories[task["category"]].append(task)
        for tag in set(task["tags"]):
            tags[tag].append(task)
    overviews = {}
    for condition in CONDITIONS:
        first = panel_stats(tasks, condition, "first")
        cumulative = panel_stats(tasks, condition, "cumulative")
        overviews[condition] = {
            "first_batch": first,
            "cumulative": cumulative,
            "prescreen": {
                name: sum(task["conditions"][condition]["prescreen"][name] for task in tasks)
                for name in ("candidates", "accepted", "excluded", "technical_pending", "technical_attempts", "retries", "attempt_wall_seconds")
            },
            "primary_categories_first_batch": {
                name: panel_stats(group, condition, "first") for name, group in sorted(categories.items())
            },
            "overlapping_tags_first_batch": {
                name: panel_stats(group, condition, "first") for name, group in sorted(tags.items())
            },
        }
    return {
        "schema": 1,
        "label": "baseline calibration" if phase == "formal" else "smoke only",
        "phase": phase,
        "protocol": next(iter(evidence.protocols), None),
        "protocol_fingerprint": next(iter(evidence.fingerprints), None),
        "official_task_count": len(official),
        "robotwin_commit": catalog["robotwin_commit"],
        "tasks_sha256": hashlib.sha256(catalog_raw).hexdigest(),
        "evidence_files": evidence.files,
        "overall": overviews,
        "tasks": tasks,
    }


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def format_interval(interval):
    return "—" if interval is None else f"{pct(interval[0])}–{pct(interval[1])}"


def markdown_cell(value):
    return str(value).replace("|", r"\|").replace("\n", " ")


def markdown_row(fields):
    return "| " + " | ".join(markdown_cell(field) for field in fields) + " |\n"


def emit_markdown(report, output):
    write = output.write
    phase = report["phase"]
    write(f"# X‑VLA RoboTwin 2.0 {'基座能力评测' if phase == 'formal' else 'Smoke 验收'}\n\n")
    write(f"用途：**{report['label']}**。RoboTwin 源码固定为 `{report['robotwin_commit']}`。")
    write(f"已读取的运行协议：`{report['protocol'] or '尚无运行证据'}`。\n\n")
    if report["protocol_fingerprint"]:
        write(f"协议指纹：`{report['protocol_fingerprint']}`。\n\n")
    if phase == "formal":
        write("评测对象为官方 RoboTwin 适配基座。该模型已使用 RoboTwin 示教数据适配；本表用于选择后续研究任务与校准，不能视为 SFT 未见任务或独立最终测试。公开参考成绩不计入本地分母。Easy 与 Hard 的相同整数种子未证明对应同一个场景。\n\n")
    else:
        write("Smoke 数据仅用于验收，不进入正式成功率。\n\n")
    write("| 条件 | 首批完成任务 | 首批完成子集的任务等权均值 | 完整首批均值 | 累计完成任务 | 累计完成回合 | 预筛排除 | 评测重试 | 未解决 |\n")
    write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    for condition, overview in report["overall"].items():
        first, all_batches = overview["first_batch"], overview["cumulative"]
        write(markdown_row((
            condition.title(),
            f"{first['complete_tasks']}/{first['planned_tasks']}",
            pct(first["complete_subset_task_weighted_sr"]),
            pct(first["all_planned_tasks_weighted_sr"]),
            f"{all_batches['complete_tasks']}/{all_batches['planned_tasks']}",
            f"{all_batches['finished']}/{all_batches['planned']}",
            overview["prescreen"]["excluded"], all_batches["retries"], all_batches["unresolved"],
        )))
    write("\n均值给每个任务相同权重。完整首批均值须同条件全部任务的首批均达到规定样本数；子集均值明确只用已完成的任务。95% Wilson 区间仅基于完成的独立初态，不涵盖未解决回合。100 个样本在约 50% 时区间约 40.4%–59.6%，300 个约 44.4%–55.6%。零成功也不能证明一种能力完全不存在。\n\n")
    write("## 任务目录\n\n")
    write("| Task ID | 中文任务内容 | 主要类别 | 操作标签 | 官方成功条件 | 代码依据 | 运行覆盖 |\n")
    write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for task in report["tasks"]:
        def coverage(shard):
            stages = shard["coverage"]
            if stages["planned_rollouts_complete"]:
                return "达样本目标"
            if stages["policy_action_observed"]:
                return "有策略观测与动作证据"
            if stages["policy_observation_observed"]:
                return "有策略观测证据"
            if stages["official_success_expert_verified"]:
                return "已验专家 reset／成功"
            if stages["reset_observed"]:
                return "已验 reset"
            return "待运行验证"

        source = f"`{task['source_file']}`：专家 L{task['expert_line']}；成功判定 L{task['success_line']}"
        write(markdown_row((
            f"`{task['task_id']}`", task["description_zh"], task["category"], "、".join(task["tags"]),
            task["success_zh"], source,
            "；".join(f"{c.title()} {coverage(s)}" for c, s in task["conditions"].items())
            + ("；RLinf 联接层未支持，使用原生入口核实" if task.get("rlinf_wrapper_support") is False else ""),
        )))
    write("\n## 能力结果（累计）\n\n")
    write("| Task ID | Easy 成功／完成／计划 | Easy SR（95% Wilson） | Hard 成功／完成／计划 | Hard SR（95% Wilson） | 异常／缺失 | 后续用途建议 |\n")
    write("| --- | ---: | ---: | ---: | ---: | --- | --- |\n")
    for task in report["tasks"]:
        parts = [f"`{task['task_id']}`"]
        exceptions = []
        for condition, shard in task["conditions"].items():
            data = shard["cumulative"]
            parts.extend((
                f"{data['successes']}/{data['finished']}/{data['planned']}",
                f"{pct(data['success_rate'])} ({format_interval(data['wilson_95'])})" if data["finished"] else "—",
            ))
            if data["status"] != "complete" or data["technical_attempts"] or shard["prescreen"]["technical_attempts"]:
                missing = (
                    f"{condition.title()} 未冻结 {data['not_frozen']}／未运行 {data['not_started']}／"
                    f"待重试 {data['retry_pending']}／未解决 {data['unresolved']}／重试 {data['retries']}"
                )
                if data["unresolved"]:
                    missing += f"（未解决全失败至全成功：{format_interval(data['unresolved_only_bounds'])}）"
                if data["technical_reasons"]:
                    missing += f"；首要技术原因：{next(iter(data['technical_reasons']))}"
                if shard["prescreen"]["technical_reasons"]:
                    missing += f"；预筛异常：{next(iter(shard['prescreen']['technical_reasons']))}"
                exceptions.append(missing)
        parts += ["；".join(exceptions) or "—", "；".join(task.get("future_use_zh", ["仅用于验收"]))]
        write(markdown_row(parts))
    write("\n`cohorts.csv` 按任务和条件列出每批与累计数据，包括独立初态数、重试数、执行预算、耗时、异常／失败类型和缺失回合的边界。`summary.json` 保留文件校验、预筛证据汇总、覆盖和重叠标签统计。\n\n")
    if phase == "formal":
        write("建议先筛有持续成功信号的 source，再从可测任务中按主要类别与辅助标签选相关和跨类迁移对照。表内建议至少要求完成正式首批；后续仍须复核轨迹质量和吞吐，并冻结新的评测初态。\n\n")
    else:
        write("| Task ID | Easy 原生协议对齐 | Hard 原生协议对齐 |\n")
        write("| --- | --- | --- |\n")
        for task in report["tasks"]:
            write(markdown_row((f"`{task['task_id']}`", *(
                task["conditions"][condition]["parity_status"] or "暂无证据" for condition in CONDITIONS
            ))))
        write("\n")
    write("## 首批主类别统计\n\n")
    write("| 条件 | 主类别（项目分析分类） | 已完成任务数 | 完成子集等权 SR | 全类别等权 SR |\n")
    write("| --- | --- | ---: | ---: | ---: |\n")
    for condition, overview in report["overall"].items():
        for category, row in overview["primary_categories_first_batch"].items():
            write(markdown_row((
                condition.title(), category, f"{row['complete_tasks']}/{row['planned_tasks']}",
                pct(row["complete_subset_task_weighted_sr"]), pct(row["all_planned_tasks_weighted_sr"]),
            )))


def emit_csv(report, output):
    stats = [
        "status", "planned", "frozen", "finished", "independent_seeds", "distinct_scenes", "successes",
        "failures", "success_rate", "unresolved", "not_frozen", "not_started", "retry_pending",
        "technical_attempts", "retries",
    ]
    header = ["task_id", "condition", "batch", "category", *stats]
    pairs = ["wilson_95", "unresolved_only_bounds", "planned_bounds"]
    for key in pairs:
        header.extend((f"{key}_low", f"{key}_high"))
    for key in BUDGET_FIELDS:
        header.extend((f"mean_{key}", f"n_{key}"))
    header += ["failure_reasons", "technical_reasons", "prescreen_candidates", "prescreen_accepted", "prescreen_excluded", "prescreen_acceptance_rate", "prescreen_retries", "prescreen_technical_attempts"]
    writer = csv.writer(output)
    writer.writerow(header)
    for task in report["tasks"]:
        for condition, shard in task["conditions"].items():
            for batch, data in (*shard["batches"].items(), ("cumulative", shard["cumulative"])):
                row = [task["task_id"], condition, batch, task["category"], *(data[key] for key in stats)]
                for key in pairs:
                    row.extend(data[key] if data[key] is not None else (None, None))
                for key in BUDGET_FIELDS:
                    stat = data["mean_budget"][key]
                    row.extend((stat["mean"], stat["n"]))
                prescreen = shard["prescreen"]
                row.extend(json.dumps(data[key], ensure_ascii=False) for key in ("failure_reasons", "technical_reasons"))
                row.extend(prescreen[key] if batch == "cumulative" else None for key in (
                    "candidates", "accepted", "excluded", "acceptance_rate", "retries", "technical_attempts",
                ))
                writer.writerow(row)


def main():
    """Write the auditable machine results and a Chinese progress report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="run root containing the phase directories")
    parser.add_argument("--output", type=Path, required=True, help="write a derived report here")
    parser.add_argument("--tasks", type=Path, default=TASKS)
    parser.add_argument("--phase", choices=("formal", "smoke"), default="formal")
    parser.add_argument("--smoke-episodes", type=int, choices=range(5, 11), default=5)
    args = parser.parse_args()
    report = summarize(args.input, args.tasks, args.phase, args.smoke_episodes)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (args.output / "report.md").open("w", encoding="utf-8") as stream:
        emit_markdown(report, stream)
    with (args.output / "cohorts.csv").open("w", encoding="utf-8", newline="") as stream:
        emit_csv(report, stream)
    for condition, overview in report["overall"].items():
        totals = overview["cumulative"]
        print(f"{args.phase} {condition}: {totals['finished']}/{totals['planned']} completed ({totals['unresolved']} unresolved)")


if __name__ == "__main__":
    main()
