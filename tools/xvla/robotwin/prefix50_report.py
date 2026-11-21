"""Report deterministic first-50 cohorts without changing frozen evaluation plans."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import uuid

EVALUATION = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
WORKER = "bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460"
NATIVE_EASY = "15728f76ba267fef3bd75c36c7d825c4ab2a4360"
SOURCE_FINGERPRINT = "78fa1f315e850c366ae0e36250f3cb633c6667747ebdfbb0608cee0656320f46"
PREFIX = 50


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(root, expected):
    actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    require(actual == expected, f"Wrong frozen report source: {root}")
    subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "tools/xvla/robotwin"], check=True)


def load(root, name, package):
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(root / "tools/xvla/robotwin")]
    sys.modules[package] = namespace
    spec = importlib.util.spec_from_file_location(f"{package}.{name}", Path(namespace.__path__[0]) / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(base, root):
    class CachedEvidence(base):
        def __init__(self):
            super().__init__(root)
            self.cache = {}

        def read(self, path, journal=False):
            key = (Path(path), journal)
            if key not in self.cache:
                self.cache[key] = super().read(path, journal=journal)
            return self.cache[key]
    return CachedEvidence()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prefix_stats(module, evidence, task, condition, planned, protocol):
    shard = evidence.root / "formal" / task / condition
    manifest = evidence.read(shard / "seed_manifest.json")
    selected = {row["episode_id"]: row for row in manifest["episodes"]} if manifest else {}
    prefix = {key: row for key, row in selected.items() if row["index"] < PREFIX}
    if manifest:
        require([row["index"] for row in prefix.values()] == list(range(PREFIX)), f"Incomplete deterministic prefix: {shard}")
        for seed in prefix.values():
            require(seed["episode_id"] == f"{protocol}/formal/{task}/{condition}/{seed['index'] + 1:04d}", f"Unexpected prefix ID: {shard}")
            require(seed["scene_signature"] == digest(seed["scene"]), f"Prefix scene hash differs: {shard}")
    # All rows were validated by the frozen full-panel/task report before here.
    attempts = defaultdict(list)
    for _, row in evidence.read(shard / "episodes.jsonl", journal=True):
        attempts[row["episode_id"]].append(row)
    stats = module.cohort_stats(PREFIX, prefix, attempts)
    extra = {key: row for key, row in selected.items() if key not in prefix}
    stats.update(outside_prefix=module.cohort_stats(planned - PREFIX, extra, attempts),
                 prefix_identity_sha256=digest(list(prefix.values())) if prefix else None,
                 prefix_indices=[0, 49], manifest_sha256=evidence.files.get(
                     (shard / "seed_manifest.json").relative_to(evidence.root).as_posix(), {}).get("sha256"))
    return stats


def summarize(evaluation_root, worker_root, hard_root, source_root):
    """Return matched sample sizes; Easy and Hard retain different action budgets."""
    evaluation, worker, hard_root, source_root = (Path(path).resolve() for path in
                                                (evaluation_root, worker_root, hard_root, source_root))
    pin(evaluation, EVALUATION)
    pin(worker, WORKER)
    require(source_root.name == EVALUATION and hard_root != source_root
            and not hard_root.is_relative_to(source_root), "Keep native Hard separate from the frozen source panel")
    package = "_prefix50_" + uuid.uuid4().hex
    try:
        original = load(evaluation, "summarize", package + "_source")
        native = load(worker, "hard_native_report", package + "_worker")
        _, definitions, priority = native.catalog()
        baseline, evidence = snapshot(original.Evidence, source_root), snapshot(native.Evidence, hard_root)
        profile = native.verify_profile(evidence, definitions, priority)
        require(profile["implementation_commit"] == WORKER, "The frozen Hard profile implementation changed")
        formal = hard_root / "formal"
        folders = {path.name for path in formal.iterdir() if path.is_dir()} if formal.is_dir() else set()
        require(folders <= definitions.keys(), "Unknown task in native Hard formal output")
        # Read dependent native journals first. Source preparation precedes these
        # files, so a concurrent freeze cannot produce an orphaned native result.
        for task in definitions:
            shard = formal / task / "hard"
            if shard.parent.is_dir():
                require({path.name for path in shard.parent.iterdir() if path.is_dir()} <= {"hard"}, "Unexpected native condition")
            evidence.read(shard / "episodes.jsonl", journal=True)
            evidence.read(shard / "runtime_manifest.json")
            evidence.read(shard / "seed_manifest.json")
        # The full pinned source validator and prefix derivation share every byte.
        original.Evidence = lambda root: baseline
        legacy = original.summarize(source_root)
        require(legacy["protocol_fingerprint"] == SOURCE_FINGERPRINT, "Source protocol fingerprint changed")
        legacy_first = legacy["overall"]["easy"]["first_batch"]
        require(legacy_first["planned"] == legacy_first["finished"] == 5000
                and legacy_first["unresolved"] == legacy_first["retry_pending"] == 0,
                "The original Easy first-100 cohorts must be complete and independently validated")
        legacy_tasks = {row["task_id"]: row for row in legacy["tasks"]}
        require(list(legacy_tasks) == list(definitions), "Frozen source and native task catalogs differ")
        bound = {**profile, "_source_files": baseline.files}
        tasks = []
        for task, definition in definitions.items():
            validated = native.task_report(evidence, baseline, definition, bound)
            planned = profile["per_task_episodes"][task]
            easy = prefix_stats(original, baseline, task, "easy", planned, legacy["protocol"])
            hard = prefix_stats(native, evidence, task, "hard", planned, profile["protocol"])
            easy.update(budget_tag="original_300_anchors", max_queries=10, max_anchors_sent=300,
                        native_step_limit=definition["step_limit"])
            hard.update(budget_tag="native_task_limit", max_queries=profile["native_query_limits"][task],
                        max_anchors_sent=30 * profile["native_query_limits"][task],
                        native_step_limit=profile["native_action_limits"][task])
            tasks.append(dict(task_id=task, description_zh=definition["description_zh"], category=definition["category"],
                              status=hard["status"], easy=easy, hard=hard, prescreen=validated["prescreen"],
                              frozen_plan_episodes=planned, hard_full_plan=validated["cumulative"],
                              legacy_easy_first100=legacy_tasks[task]["conditions"]["easy"]["batches"]["1"]))
        overall = {condition: native.overview([task[condition] for task in tasks]) for condition in ("easy", "hard")}
        for condition, stats in overall.items():
            stats["micro_success_rate"] = stats["successes"] / stats["finished"] if stats["finished"] else None
            stats["outside_prefix"] = native.overview([task[condition]["outside_prefix"] for task in tasks])
        legacy_first = {**legacy_first, "micro_success_rate": legacy_first["successes"] / legacy_first["finished"]}
        data_root = source_root.parent.parent
        native_easy_root = data_root / "native-budget" / NATIVE_EASY
        return dict(schema=1, snapshot_utc=datetime.now(timezone.utc).isoformat(), status=overall["hard"]["status"],
                    protocol=profile["protocol"], protocol_fingerprint=profile["protocol_fingerprint"],
                    source_protocol=legacy["protocol"], source_protocol_fingerprint=SOURCE_FINGERPRINT,
                    selection=dict(rule="first frozen manifest indices 0..49 per task; outcomes do not select the cohort",
                                   episodes_per_task=50, tasks=50, planned=2500, frozen_full_plan_episodes=7000),
                    input_paths=dict(evaluation_root=str(evaluation), worker_root=str(worker), hard_root=str(hard_root), source_root=str(source_root)),
                    overall=overall, legacy_easy_first100=legacy_first,
                    native_easy_five=dict(included_in_comparison=False, root=str(native_easy_root),
                                          profile_available=(native_easy_root / "profile.json").is_file(),
                                          summary_available=(native_easy_root / "reports/summary.json").is_file()),
                    tasks=tasks, evidence_files=evidence.files, source_evidence_files=baseline.files)
    finally:
        for name in list(sys.modules):
            if name.startswith(package):
                del sys.modules[name]


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}%"


def result(row):
    interval = row["wilson_95"]
    ci = "—" if interval is None else f"{100 * interval[0]:.1f}–{100 * interval[1]:.1f}%"
    return f"{row['successes']}/{row['finished']}（计划50）；{percent(row['success_rate'])} [{ci}]"


def markdown(report):
    lines = ["# X‑VLA RoboTwin 每任务前50回合", "",
             "每项使用冻结清单的索引0–49，共50任务×50初态；原100／300清单与前缀以外结果完整保留。", "",
             "Easy 使用原300锚点／10次查询预算；Hard 使用任务原生400–1,700锚点预算。样本数相同不代表预算或场景相同，两列差异不能归因于难度或预算的单一影响。", "",
             "| 条件 | 完成／计划 | 成功 | 失败 | 未冻结／未运行／待重试／未解决 | Micro SR | 完整50任务等权 SR |",
             "| --- | ---: | ---: | ---: | --- | ---: | ---: |"]
    for condition in ("easy", "hard"):
        row = report["overall"][condition]
        lines.append(f"| {condition.title()} | {row['finished']}/{row['planned']} | {row['successes']} | {row['failures']} | "
                     f"{row['not_frozen']} / {row['not_started']} / {row['retry_pending']} / {row['unresolved']} | "
                     f"{percent(row['micro_success_rate'])} | {percent(row['all_planned_tasks_weighted_sr'])} |")
    legacy = report["legacy_easy_first100"]
    lines += ["", f"原项目 Easy 首批每任务100回合：{legacy['successes']}/{legacy['finished']}，"
              f"Micro SR {percent(legacy['micro_success_rate'])}，任务等权 SR {percent(legacy['all_planned_tasks_weighted_sr'])}。该结果单列，未以50回合口径覆盖。", "",
              "| Task ID | 中文内容 | Easy 原300预算：成功／完成、SR [95% Wilson] | Hard 原生预算：成功／完成、SR [95% Wilson] | Hard原生锚点上限 |",
              "| --- | --- | --- | --- | ---: |"]
    for task in report["tasks"]:
        description = task["description_zh"].replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {task['task_id']} | {description} | {result(task['easy'])} | {result(task['hard'])} | {task['hard']['native_step_limit']} |")
    extras = report["overall"]["hard"]["outside_prefix"]
    lines += ["", f"Hard 前缀以外已完成 {extras['finished']} 回合，保留在原始记录中，不计入本报告2,500回合分母。",
              "每任务 Wilson 区间仅描述已完成初态的抽样不确定性；未完成回合不计为策略失败。Micro 在完成子集上汇总，完整任务等权均值须全部任务达到50回合后才给出。",
              "五任务 Easy 原生预算重评属于另一协议；本报告仅标记其文件是否可用，不混入 Easy 前50或原首100结果。",
              "论文应披露50回合口径的确定时点、预算、固定版本和预筛规则；这是官方 RoboTwin 适配基座的能力审计。", ""]
    return "\n".join(lines)


def write_report(report, output_path):
    target = Path(output_path).resolve()
    paths = {key: Path(value).resolve() for key, value in report["input_paths"].items()}
    protected = [paths["evaluation_root"], paths["worker_root"], paths["source_root"],
                 *(paths["hard_root"] / name for name in ("formal", "pilot", "reports"))]
    require(target != paths["hard_root"] and not any(target == path or target.is_relative_to(path) for path in protected),
            "Use a separate prefix50 report directory outside frozen sources and original reports")
    target.mkdir(parents=True, exist_ok=True)
    for name, text in (("summary.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
                       ("report.md", markdown(report))):
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target, prefix=f".{name}.", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(text); stream.flush(); os.fsync(stream.fileno())
            temporary.replace(target / name)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-root", "worker-root", "input", "source-panel", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    report = summarize(args.evaluation_root, args.worker_root, args.input, args.source_panel)
    write_report(report, args.output)
    row = report["overall"]["hard"]
    print(f"Hard first50: {row['finished']}/{row['planned']}, {row['successes']} successes, {row['unresolved']} unresolved")


if __name__ == "__main__":
    main()
