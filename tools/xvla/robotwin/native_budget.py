"""Rerun five Easy tasks with native action budgets in a separate cohort."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
import time

if __package__:
    from . import audit_rollout
else:
    import audit_rollout


PROTOCOL = "robotwin-native-b0-native-budget-v1"
EVALUATION_COMMIT = audit_rollout.EVALUATION_COMMIT
NATIVE_LIMITS = {"open_microwave": 1500, "stack_bowls_three": 1200, "blocks_ranking_rgb": 1200,
                 "blocks_ranking_size": 1200, "click_alarmclock": 400}
EPISODES = {task: 300 if task in {"open_microwave", "click_alarmclock"} else 100 for task in NATIVE_LIMITS}


def source_manifest(runner, source_panel, task):
    path = source_panel / "formal" / task / "easy/seed_manifest.json"
    manifest = runner.read_json(path)
    count = EPISODES[task]
    if (manifest["task"], manifest["condition"], manifest["phase"], manifest["limit"]) != (task, "easy", "formal", count):
        raise ValueError(f"Source manifest does not describe the frozen {task} cohort")
    seeds = manifest["episodes"]
    if len(seeds) != count:
        raise ValueError(f"Incomplete source manifest for {task}")
    identifiers, scene_signatures, actual_seeds = set(), set(), set()
    for index, seed in enumerate(seeds):
        expected_id = f"{manifest['protocol']}/formal/{task}/easy/{index + 1:04d}"
        if (seed["episode_id"] != expected_id or seed["index"] != index or seed["batch"] != index // 100 + 1
                or type(seed["policy_seed"]) is not int or seed["actual_seed"] != seed["requested_seed"]
                or seed["episode_id"] in identifiers or seed["actual_seed"] in actual_seeds
                or seed["scene_signature"] in scene_signatures or runner.digest(seed["scene"]) != seed["scene_signature"]):
            raise ValueError(f"Invalid frozen source identity for {task} episode {index + 1}")
        identifiers.add(seed["episode_id"])
        scene_signatures.add(seed["scene_signature"])
        actual_seeds.add(seed["actual_seed"])
    return manifest, runner.sha256(path)


def write_or_validate(runner, path, value, create):
    if path.is_file():
        if runner.read_json(path) != value:
            raise ValueError(f"Existing native-budget profile differs from frozen inputs: {path}")
    elif create:
        runner.write_json(path, value)
    else:
        raise ValueError(f"Prepare the frozen native-budget profile before evaluating: {path}")


def prepare_profile(args, runner):
    """Freeze all 900 initial states without importing the simulator or loading weights."""
    source = Path(__file__).resolve()
    manifests = {task: source_manifest(runner, args.input_panel, task) for task in NATIVE_LIMITS}
    source_protocols = {row[0]["protocol"] for row in manifests.values()}
    source_fingerprints = {row[0]["protocol_fingerprint"] for row in manifests.values()}
    if len(source_protocols) != 1 or len(source_fingerprints) != 1:
        raise ValueError("Source manifests belong to different evaluation protocols")
    source_protocol, source_fingerprint = source_protocols.pop(), source_fingerprints.pop()
    if source_protocol != "robotwin-native-b0-v2":
        raise ValueError("Expected the original frozen v2 cohort")
    basis = {"protocol": PROTOCOL, "source_protocol": source_protocol,
             "source_protocol_fingerprint": source_fingerprint, "source_evaluation_commit": EVALUATION_COMMIT,
             "native_action_limits": NATIVE_LIMITS, "original_max_queries": 10, "original_max_anchors": 300,
             "native_query_limits": {task: math.ceil(limit / runner.p.ANCHORS) for task, limit in NATIVE_LIMITS.items()},
             "action_chunk_size": runner.p.ANCHORS, "policy_seed_rule": "retain the recorded original policy seed",
             "official_client_sha256": runner.CLIENT_SHA256,
             "author_loop_change": "Only outer range(10) becomes range(ceil(native step_lim / 30))",
             "source_sha256": {path.name: runner.sha256(path) for path in (source, source.with_name("audit_rollout.py"))},
             "pinned_rollout_sha256": runner.sha256(args.pinned_evaluation_root / "tools/xvla/robotwin/official_rollout.py")}
    fingerprint = runner.digest(basis)
    profile = {**basis, "protocol_fingerprint": fingerprint, "planned_episodes": sum(EPISODES.values()),
               "planned_first_batch": 500, "planned_extension": 400,
               "source_manifest_sha256": {task: row[1] for task, row in manifests.items()},
               "source_panel": str(args.input_panel),
               "implementation_commit": runner.git(source.parent, "rev-parse", "HEAD")}
    if args.prepare_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
    lock = runner.lock_shard(args.output_root) if args.prepare_only else nullcontext()
    with lock:
        write_or_validate(runner, args.output_root / "profile.json", profile, args.prepare_only)
        for task, (original, checksum) in manifests.items():
            shard = args.output_root / "formal" / task / "easy"
            if args.prepare_only:
                shard.mkdir(parents=True, exist_ok=True)
            episodes = []
            for seed in original["episodes"]:
                identifier = f"{PROTOCOL}/formal/{task}/easy/{seed['index'] + 1:04d}"
                episodes.append({**seed, "episode_id": identifier, "source_episode_id": seed["episode_id"],
                                 "source_manifest_sha256": checksum})
            header = {"task": task, "condition": "easy", "phase": "formal", "protocol": PROTOCOL,
                      "protocol_fingerprint": fingerprint}
            manifest = {**original, **header, "episodes": episodes, "source_protocol": source_protocol,
                        "source_protocol_fingerprint": source_fingerprint, "source_manifest_sha256": checksum,
                        "native_step_limit": NATIVE_LIMITS[task], "original_max_anchors": 300,
                        "native_max_queries": basis["native_query_limits"][task]}
            write_or_validate(runner, shard / "seed_manifest.json", manifest, args.prepare_only)
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pinned-evaluation-root", "input-panel", "output-root", "robotwin-root", "xvla-root", "checkpoint"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--task", choices=tuple(NATIVE_LIMITS))
    parser.add_argument("--batch", type=int, choices=(1, 2, 3))
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    for key in ("pinned_evaluation_root", "input_panel", "output_root", "robotwin_root", "xvla_root", "checkpoint"):
        setattr(args, key, getattr(args, key).resolve())
    if args.input_panel.name != EVALUATION_COMMIT or args.input_panel.parent.name != "panels":
        parser.error("Expected the original frozen evaluation panel")
    if any(args.output_root.is_relative_to(path) for path in (args.input_panel.parent, args.pinned_evaluation_root,
                                                            args.robotwin_root, args.xvla_root, args.checkpoint)):
        parser.error("Native-budget results must be outside all frozen input paths")
    if not args.prepare_only and (args.task is None or args.batch is None):
        parser.error("Evaluation requires --task and --batch")
    if not args.prepare_only and args.batch > 1 and EPISODES[args.task] != 300:
        parser.error("Only the two originally prioritized tasks receive extension batches")
    sys.path.insert(0, str(args.pinned_evaluation_root))
    from tools.xvla.robotwin import runner, official_rollout

    if Path(runner.__file__).resolve() != args.pinned_evaluation_root / "tools/xvla/robotwin/runner.py":
        raise ImportError("Native-budget evaluation must import the exact frozen runner")
    runner.pinned_checkout(args.pinned_evaluation_root, EVALUATION_COMMIT)
    profile = prepare_profile(args, runner)
    if args.prepare_only:
        runner.emit({"event": "native_budget_profile_frozen", "output": str(args.output_root),
                     "planned": profile["planned_episodes"], "protocol_fingerprint": profile["protocol_fingerprint"]})
        return
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Evaluation requires a GPU Slurm allocation")
    runner.pinned_checkout(args.robotwin_root, runner.p.ROBOTWIN_COMMIT)
    runner.pinned_checkout(args.xvla_root, runner.p.XVLA_COMMIT)
    args.phase, args.condition, args.limit = "formal", "easy", EPISODES[args.task]
    shard = args.output_root / "formal" / args.task / "easy"
    with runner.lock_shard(shard):
        manifest = runner.read_json(shard / "seed_manifest.json")
        header = {key: manifest[key] for key in ("task", "condition", "phase", "protocol", "protocol_fingerprint")}
        if runner.already_evaluated(args, shard, manifest["episodes"], header):
            return
        module, cls, cfg, robot_config = runner.load_native(args.robotwin_root, args.task, "easy")
        task = runner.read_json(args.pinned_evaluation_root / "tools/xvla/robotwin/tasks.json")
        task = next(row for row in task["tasks"] if row["task_id"] == args.task)
        if task["step_limit"] != NATIVE_LIMITS[args.task]:
            raise ValueError("Catalog action budget differs from the preregistered native limit")
        queries = profile["native_query_limits"][args.task]
        rollout, ast_evidence = audit_rollout.expanded_author_rollout(args.xvla_root, queries, official_rollout)
        policy, receipt = runner.checkpoint(args.checkpoint, "cuda:0", False)
        runtime = {**header, **profile, **ast_evidence, "slurm_job": os.environ["SLURM_JOB_ID"],
                   "batch": args.batch, "max_queries": queries, "anchors_per_query": runner.p.ANCHORS,
                   "max_anchors_sent": queries * runner.p.ANCHORS,
                   "native_max_advanced_anchors": NATIVE_LIMITS[args.task],
                   "native_step_limit": NATIVE_LIMITS[args.task], "step_lim": NATIVE_LIMITS[args.task],
                   "original_max_queries": 10, "original_max_anchors": 300,
                   "official_client_sha256": official_rollout.CLIENT_SHA256,
                   "seed_manifest_sha256": runner.sha256(shard / "seed_manifest.json"),
                   "checkpoint": str(args.checkpoint), "checkpoint_receipt": receipt,
                   "robotwin_root": str(args.robotwin_root), "embodiment_config": str(robot_config),
                   "embodiment_config_sha256": runner.sha256(robot_config),
                   "task_source_sha256": runner.sha256(Path(module.__file__)), "versions": runner.package_versions()}
        runner.write_json(shard / "runtime_manifest.json", runtime)
        audit_dir = shard / "runtime"
        audit_dir.mkdir(exist_ok=True)
        runner.write_json(audit_dir / f"eval-{runtime['slurm_job']}-batch{args.batch}-{time.time_ns()}.json", runtime)
        runner.run_eval(args, shard, header, module, cls, cfg, policy, rollout, manifest["episodes"], task)


if __name__ == "__main__":
    main()
