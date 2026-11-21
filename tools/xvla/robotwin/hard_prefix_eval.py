"""Evaluate only the first 50 frozen Hard seeds using the unchanged bcb evaluator."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types


WORKER = "bcb1bcdd123e2b47d3b6c0a12c1f72e95931b460"
EVALUATION = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
PROTOCOL = "robotwin-native-b0-hard-native-budget-v1"
SOURCE_FINGERPRINT = "78fa1f315e850c366ae0e36250f3cb633c6667747ebdfbb0608cee0656320f46"
EPISODES = 50


def require(ok, message):
    if not ok:
        raise ValueError(message)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, timeout=50).strip()


def frozen_native(worker):
    tools = worker / "tools/xvla/robotwin"
    package = "_robotwin_frozen_hard_prefix"
    namespace = types.ModuleType(package)
    namespace.__path__ = [str(tools)]
    sys.modules[package] = namespace
    name = f"{package}.hard_native"
    spec = importlib.util.spec_from_file_location(name, tools / "hard_native.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def prefix_hooks(runner, args, profile):
    """Filter only the seed-list arguments; restore both original functions on every exit."""
    already, evaluate = runner.already_evaluated, runner.run_eval
    evidence_written = False

    def select(eval_args, shard, seeds, header):
        nonlocal evidence_written
        expected_shard = args.output_root / "formal" / args.task / "hard"
        require(Path(shard).resolve() == expected_shard and eval_args.task == args.task
                and eval_args.phase == "formal" and eval_args.condition == "hard" and eval_args.batch == 1,
                "Prefix selection is limited to the requested formal Hard first batch")
        manifest_path = expected_shard / "seed_manifest.json"
        manifest = runner.read_json(manifest_path)
        context = dict(task=args.task, condition="hard", phase="formal", protocol=PROTOCOL,
                       protocol_fingerprint=profile["protocol_fingerprint"])
        require(header == context and all(manifest.get(key) == value for key, value in context.items()),
                "Prefix selection belongs to another frozen task or protocol")
        count = profile["per_task_episodes"][args.task]
        require(count in {100, 300} and manifest.get("limit") == count and len(seeds) == count
                and seeds == manifest["episodes"], "Prefix selection requires the unchanged full frozen manifest")
        require(all(seed["index"] == index and seed["batch"] == index // 100 + 1
                    and seed["episode_id"] == f"{PROTOCOL}/formal/{args.task}/hard/{index + 1:04d}"
                    for index, seed in enumerate(seeds)), "Frozen episode indices, IDs or batches changed")
        selected = seeds[:EPISODES]
        require(len(selected) == EPISODES and len({seed["episode_id"] for seed in selected}) == EPISODES,
                "The first 50 unique frozen episode IDs are unavailable")
        if not evidence_written:
            data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
            job = Path(os.environ["FYP_ROBOTWIN_JOB"]).resolve(strict=True)
            require(job == data / "jobs" / os.environ["SLURM_JOB_ID"], "Wrong Slurm prefix evidence directory")
            receipt = dict(event="hard_prefix_selection", task=args.task, condition="hard", phase="formal", batch=1,
                           job_id=os.environ["SLURM_JOB_ID"], worker_commit=WORKER, evaluation_commit=EVALUATION,
                           protocol=PROTOCOL, protocol_fingerprint=profile["protocol_fingerprint"],
                           selection="frozen manifest indices 0 through 49, independent of outcomes",
                           selected_episodes=EPISODES, full_manifest_episodes=count,
                           full_manifest_sha256=runner.sha256(manifest_path),
                           source_manifest_sha256=manifest["source_manifest_sha256"],
                           episode_ids=[seed["episode_id"] for seed in selected])
            runner.write_json(job / f"hard-prefix-selection-{args.task}.json", receipt)
            runner.emit(receipt)
            evidence_written = True
        return selected

    def already_evaluated(eval_args, shard, seeds, header):
        return already(eval_args, shard, select(eval_args, shard, seeds, header), header)

    def run_eval(eval_args, shard, header, module, cls, cfg, policy, official, seeds, task):
        return evaluate(eval_args, shard, header, module, cls, cfg, policy, official,
                        select(eval_args, shard, seeds, header), task)

    runner.already_evaluated, runner.run_eval = already_evaluated, run_eval
    try:
        yield
    finally:
        runner.already_evaluated, runner.run_eval = already, evaluate


def run(args):
    require(args.eval is True and args.batch == 1, "Prefix evaluation supports only --eval --batch 1")
    for key in ("worker_root", "pinned_evaluation_root", "source_panel", "output_root", "robotwin_root",
                "xvla_root", "checkpoint"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    data = Path(os.environ["FYP_ROBOTWIN_DATA"]).resolve(strict=True)
    require(args.output_root == data / "hard-native-budget" / WORKER,
            "Use the existing unchanged bcb Hard profile")
    require(args.source_panel == data / "panels" / EVALUATION, "Use the original 9baa source panel")
    for root, commit in ((args.worker_root, WORKER), (args.pinned_evaluation_root, EVALUATION)):
        require(git(root, "rev-parse", "HEAD") == commit
                and Path(git(root, "rev-parse", "--show-toplevel")).resolve() == root,
                "Prefix evaluation requires the fixed original worker and evaluation worktrees")
        require(not git(root, "status", "--porcelain", "--untracked-files=no"), "Frozen worktree has tracked changes")
    frozen = frozen_native(args.worker_root)
    require(frozen.EVALUATION_COMMIT == EVALUATION and frozen.SOURCE_FINGERPRINT == SOURCE_FINGERPRINT
            and frozen.PROTOCOL == PROTOCOL, "Frozen Hard evaluator constants differ")
    profile = json.loads((args.output_root / "profile.json").read_bytes())
    require(profile.get("implementation_commit") == WORKER and profile.get("source_evaluation_commit") == EVALUATION
            and profile.get("source_protocol_fingerprint") == SOURCE_FINGERPRINT
            and profile.get("protocol") == PROTOCOL and profile.get("condition") == "hard"
            and profile.get("planned_episodes") == 7000
            and args.task in profile["per_task_episodes"], "Frozen Hard profile identity or task differs")
    original_path, original_argv = sys.path[:], sys.argv
    try:
        sys.path.insert(0, str(args.pinned_evaluation_root))
        runner = importlib.import_module("tools.xvla.robotwin.runner")
        require(Path(runner.__file__).resolve() == args.pinned_evaluation_root / "tools/xvla/robotwin/runner.py",
                "Prefix evaluation must import the original pinned runner")
        command = [str(args.worker_root / "tools/xvla/robotwin/hard_native.py"), "--eval", "--batch", "1",
                   "--task", args.task]
        for key in ("pinned_evaluation_root", "source_panel", "output_root", "robotwin_root", "xvla_root", "checkpoint"):
            command.extend(["--" + key.replace("_", "-"), str(getattr(args, key))])
        sys.argv = command
        with prefix_hooks(runner, args, profile):
            return frozen.main()
    finally:
        sys.path[:] = original_path
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("worker-root", "pinned-evaluation-root", "source-panel", "output-root", "robotwin-root",
                 "xvla-root", "checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--eval", required=True, action="store_true")
    parser.add_argument("--task", required=True)
    parser.add_argument("--batch", type=int, choices=(1,), required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
