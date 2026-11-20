"""Freeze, expert-screen and evaluate all 50 Hard tasks with native action budgets."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from . import audit_rollout
else:
    import audit_rollout


EVALUATION_COMMIT = audit_rollout.EVALUATION_COMMIT
PROTOCOL = "robotwin-native-b0-hard-native-budget-v1"
SOURCE_PROTOCOL = "robotwin-native-b0-v2"
SOURCE_FINGERPRINT = "78fa1f315e850c366ae0e36250f3cb633c6667747ebdfbb0608cee0656320f46"
PROFILE_METADATA_KEYS = ("protocol_fingerprint", "planned_episodes", "planned_first_batch", "planned_extension",
                         "source_panel", "implementation_commit")
PILOT_TASKS = ("move_stapler_pad", "turn_switch")
PILOT_LANES = ("serial", "parallel")


def write_or_validate(runner, path, value, create):
    if path.is_file():
        if runner.read_json(path) != value:
            raise ValueError(f"Existing Hard native-budget evidence differs: {path}")
    elif create:
        runner.write_json(path, value)
    else:
        raise ValueError(f"Prepare the Hard native-budget evidence before evaluation: {path}")


def original_fingerprint(args, runner):
    names = ("runner.py", "protocol.py", "official_rollout.py", "parity.py", "bind_renderer.py", "runtime.env",
             "requirements.txt")
    basis = {"robotwin_commit": runner.p.ROBOTWIN_COMMIT, "xvla_commit": runner.p.XVLA_COMMIT,
             "rlinf_commit": runner.p.RLINF_COMMIT, "model_id": runner.p.MODEL_ID,
             "model_revision": runner.p.MODEL_REVISION, "official_client_sha256": runner.CLIENT_SHA256,
             "policy_seed_namespace": runner.p.POLICY_SEED_NAMESPACE,
             "config_sha256": {name: runner.sha256(args.robotwin_root / name) for name in runner.FILES},
             "evaluation_source_sha256": {name: runner.sha256(args.pinned_evaluation_root / "tools/xvla/robotwin" / name)
                                          for name in names}}
    fingerprint = runner.digest(basis)
    if runner.p.PROTOCOL != SOURCE_PROTOCOL or fingerprint != SOURCE_FINGERPRINT:
        raise ValueError("The original Hard expert-screening protocol does not match the qualified evaluation")
    return fingerprint


def prepare_profile(args, runner):
    """Freeze the 7,000-episode plan while keeping unprepared initial states explicit."""
    catalog_path = args.pinned_evaluation_root / "tools/xvla/robotwin/tasks.json"
    catalog = runner.read_json(catalog_path)
    tasks = {row["task_id"]: row for row in catalog["tasks"]}
    priority = catalog["priority_tasks"]
    if len(tasks) != 50 or len(catalog["tasks"]) != 50 or len(set(priority)) != 10 or not set(priority) <= tasks.keys():
        raise ValueError("Expected the original 50 tasks and ten priority tasks")
    limits = {task: row["step_limit"] for task, row in tasks.items()}
    native_file = args.robotwin_root / "task_config/_eval_step_limit.yml"
    native_limits = {key.strip(): int(value.strip()) for line in native_file.read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith("#") for key, value in [line.split(":", 1)]}
    if limits != native_limits or min(limits.values()) != 400 or max(limits.values()) != 1700:
        raise ValueError("Task catalog action limits differ from the pinned native task configuration")
    source = Path(__file__).resolve()
    basis = {"protocol": PROTOCOL, "condition": "hard", "source_protocol": SOURCE_PROTOCOL,
             "source_protocol_fingerprint": original_fingerprint(args, runner),
             "source_evaluation_commit": EVALUATION_COMMIT, "task_ids": list(tasks), "priority_tasks": priority,
             "per_task_episodes": {task: 300 if task in priority else 100 for task in tasks},
             "native_action_limits": limits,
             "native_query_limits": {task: math.ceil(limit / runner.p.ANCHORS) for task, limit in limits.items()},
             "original_max_queries": 10, "original_max_anchors": 300, "action_chunk_size": runner.p.ANCHORS,
             "model_id": runner.p.MODEL_ID, "model_revision": runner.p.MODEL_REVISION,
             "official_client_sha256": runner.CLIENT_SHA256,
             "author_loop_change": "Only outer range(10) becomes range(ceil(native step_lim / 30))",
             "policy_seed_rule": "retain the original expert-screened Hard episode policy seed",
             "pilot": {"tasks": list(PILOT_TASKS), "lanes": list(PILOT_LANES), "episodes_per_task": 5,
                       "counts_toward_formal": False},
             "source_catalog_sha256": runner.sha256(catalog_path),
             "source_sha256": {path.name: runner.sha256(path) for path in
                               (source, source.with_name("audit_rollout.py"), source.with_name("prescreen_recovery.py"),
                                source.with_name("prescreen_failures.py"))}}
    profile = {**basis, "protocol_fingerprint": runner.digest(basis), "planned_episodes": 7000,
               "planned_first_batch": 5000, "planned_extension": 2000, "source_panel": str(args.source_panel),
               "implementation_commit": runner.git(source.parent, "rev-parse", "HEAD")}
    if sum(profile["per_task_episodes"].values()) != profile["planned_episodes"]:
        raise ValueError("The Hard task plan does not total 7,000 episodes")
    if args.prepare_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with runner.lock_shard(args.output_root):
            write_or_validate(runner, args.output_root / "profile.json", profile, True)
        for lane in PILOT_LANES:
            for task in PILOT_TASKS:
                freeze_task(args, runner, profile, create=True, task=task, pilot_lane=lane)
    else:
        write_or_validate(runner, args.output_root / "profile.json", profile, False)
    return profile, tasks


def freeze_task(args, runner, profile, *, create, task=None, pilot_lane=None):
    """Bind the new task cohort to a complete, independently audited expert-screen manifest."""
    from tools.xvla.robotwin import summarize

    if Path(summarize.__file__).resolve() != args.pinned_evaluation_root / "tools/xvla/robotwin/summarize.py":
        raise ImportError("Hard source verification must use the pinned summarizer")
    task = task or args.task
    phase = "smoke" if pilot_lane else "formal"
    source_path = args.source_panel / phase / task / "hard/seed_manifest.json"
    source = runner.read_json(source_path)
    source_header = {"task": task, "condition": "hard", "phase": phase, "protocol": SOURCE_PROTOCOL,
                     "protocol_fingerprint": SOURCE_FINGERPRINT}
    runner.validate_rows([source], source_header)
    evidence = summarize.Evidence(args.source_panel)
    accepted, _ = summarize.screening(evidence, source_path.parent, task, "hard", phase)
    count = 5 if pilot_lane else profile["per_task_episodes"][task]
    batches = {1: 5} if pilot_lane else {batch: 100 for batch in range(1, count // 100 + 1)}
    summarize.frozen_seeds(evidence, source_path, source, task, "hard", phase, batches, accepted)
    checksum = runner.sha256(source_path)
    episodes = []
    for seed in source["episodes"]:
        expected = runner.make_seed(source_header, seed["index"], seed["actual_seed"], seed["scene_signature"])
        if any(seed[key] != value for key, value in expected.items()) or runner.digest(seed["scene"]) != seed["scene_signature"]:
            raise ValueError("Source Hard seed identity or scene differs from the pinned sampling protocol")
        identifier = f"{PROTOCOL}/{phase}/{task}/hard/{seed['index'] + 1:04d}"
        episodes.append({**seed, "episode_id": identifier, "source_episode_id": seed["episode_id"],
                         "source_manifest_sha256": checksum})
    manifest = {**source, "protocol": PROTOCOL, "protocol_fingerprint": profile["protocol_fingerprint"],
                "episodes": episodes, "source_protocol": SOURCE_PROTOCOL,
                "source_protocol_fingerprint": SOURCE_FINGERPRINT, "source_manifest_sha256": checksum,
                "native_step_limit": profile["native_action_limits"][task], "original_max_anchors": 300,
                "native_max_queries": profile["native_query_limits"][task]}
    shard = args.output_root / phase / task / "hard"
    if pilot_lane:
        shard = args.output_root / "pilot" / pilot_lane / phase / task / "hard"
        manifest.update(counts_toward_formal=False, pilot_lane=pilot_lane)
    if create:
        shard.mkdir(parents=True, exist_ok=True)
        with runner.lock_shard(shard):
            write_or_validate(runner, shard / "seed_manifest.json", manifest, True)
    else:
        write_or_validate(runner, shard / "seed_manifest.json", manifest, False)
    return shard, manifest


def screen(args, runner, profile):
    source = Path(__file__).resolve()
    command = [sys.executable, "-u", str(source.with_name("prescreen_recovery.py")),
               "--pinned-evaluation-root", str(args.pinned_evaluation_root), "screen",
               "--robotwin-root", str(args.robotwin_root), "--task", args.task, "--condition", "hard",
               "--phase", "formal", "--output", str(args.source_panel),
               "--limit", str(profile["per_task_episodes"][args.task])]
    # The existing recovery process owns the source shard lock and all bounded retries.
    subprocess.run(command, cwd=args.pinned_evaluation_root,
                   env=dict(os.environ, PYTHONPATH=str(args.pinned_evaluation_root)), check=True)
    return freeze_task(args, runner, profile, create=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pinned-evaluation-root", "source-panel", "output-root", "robotwin-root", "xvla-root", "checkpoint"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--screen-only", action="store_true")
    action.add_argument("--eval", action="store_true")
    parser.add_argument("--task")
    parser.add_argument("--batch", type=int, choices=(1, 2, 3))
    parser.add_argument("--pilot-lane", choices=PILOT_LANES)
    args = parser.parse_args()
    for key in ("pinned_evaluation_root", "source_panel", "output_root", "robotwin_root", "xvla_root", "checkpoint"):
        setattr(args, key, getattr(args, key).resolve())
    if args.source_panel.name != EVALUATION_COMMIT or args.source_panel.parent.name != "panels":
        parser.error("Expected the original frozen evaluation panel for Hard expert-screening evidence")
    if any(args.output_root.is_relative_to(path) for path in (args.source_panel.parent, args.pinned_evaluation_root,
                                                            args.robotwin_root, args.xvla_root, args.checkpoint)):
        parser.error("New Hard results must be outside all original input and panel paths")
    if args.pilot_lane and (not args.eval or args.task not in PILOT_TASKS or args.batch is not None):
        parser.error("Pilot evaluation requires --eval and one pilot task, without --batch")
    if not args.prepare_only and (not args.task or (args.eval and not args.pilot_lane and args.batch is None)):
        parser.error("Screening requires --task, and evaluation also requires --batch")
    sys.path.insert(0, str(args.pinned_evaluation_root))
    from tools.xvla.robotwin import runner, official_rollout

    if Path(runner.__file__).resolve() != args.pinned_evaluation_root / "tools/xvla/robotwin/runner.py":
        raise ImportError("Hard native-budget evaluation must import the frozen runner")
    runner.pinned_checkout(args.pinned_evaluation_root, EVALUATION_COMMIT)
    profile, tasks = prepare_profile(args, runner)
    if args.prepare_only:
        runner.emit({"event": "hard_native_plan_frozen", "planned": 7000, "output": str(args.output_root),
                     "protocol_fingerprint": profile["protocol_fingerprint"]})
        return
    if args.task not in tasks or (args.eval and not args.pilot_lane and args.batch > 1
                                 and args.task not in profile["priority_tasks"]):
        parser.error("Task or extension batch is outside the fixed Hard plan")
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Expert screening and evaluation require a GPU Slurm allocation")
    runner.pinned_checkout(args.robotwin_root, runner.p.ROBOTWIN_COMMIT)
    runner.pinned_checkout(args.xvla_root, runner.p.XVLA_COMMIT)
    if args.screen_only:
        screen(args, runner, profile)
        return
    shard, manifest = freeze_task(args, runner, profile, create=False, pilot_lane=args.pilot_lane)
    args.phase, args.condition, args.limit = manifest["phase"], "hard", manifest["limit"]
    with runner.lock_shard(shard):
        header = {key: manifest[key] for key in ("task", "condition", "phase", "protocol", "protocol_fingerprint")}
        if runner.already_evaluated(args, shard, manifest["episodes"], header):
            return
        module, cls, cfg, robot_config = runner.load_native(args.robotwin_root, args.task, "hard")
        queries = profile["native_query_limits"][args.task]
        rollout, ast_evidence = audit_rollout.expanded_author_rollout(args.xvla_root, queries, official_rollout)
        policy, receipt = runner.checkpoint(args.checkpoint, "cuda:0", False)
        runtime = {**profile, **header, **ast_evidence, "slurm_job": os.environ["SLURM_JOB_ID"], "batch": args.batch,
                   "max_queries": queries, "anchors_per_query": runner.p.ANCHORS,
                   "max_anchors_sent": queries * runner.p.ANCHORS,
                   "native_max_advanced_anchors": profile["native_action_limits"][args.task],
                   "step_lim": profile["native_action_limits"][args.task],
                   "seed_manifest_sha256": runner.sha256(shard / "seed_manifest.json"),
                   "source_manifest_sha256": manifest["source_manifest_sha256"],
                   "checkpoint": str(args.checkpoint), "checkpoint_receipt": receipt,
                   "robotwin_root": str(args.robotwin_root), "embodiment_config": str(robot_config),
                   "embodiment_config_sha256": runner.sha256(robot_config),
                   "task_source_sha256": runner.sha256(Path(module.__file__)), "versions": runner.package_versions()}
        if args.pilot_lane:
            runtime.update(counts_toward_formal=False, pilot_lane=args.pilot_lane)
        runner.write_json(shard / "runtime_manifest.json", runtime)
        audit_dir = shard / "runtime"
        audit_dir.mkdir(exist_ok=True)
        runner.write_json(audit_dir / f"eval-{runtime['slurm_job']}-batch{args.batch}-{time.time_ns()}.json", runtime)
        runner.run_eval(args, shard, header, module, cls, cfg, policy, rollout, manifest["episodes"], tasks[args.task])


if __name__ == "__main__":
    main()
