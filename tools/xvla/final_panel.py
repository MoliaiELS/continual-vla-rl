"""Evaluate fixed E3 endpoint policies on the sealed Spatial H panel."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

if __package__:
    from .e2_dispatch import fixed_commit, read, write
    from .e3_dispatch import POOLS, SEEDS, digest, schedule
    from .submit import submit_locked
else:
    from e2_dispatch import fixed_commit, read, write
    from e3_dispatch import POOLS, SEEDS, digest, schedule
    from submit import submit_locked


PROTOCOL = Path("configs/xvla/protocol_20261110")
RUN_PROTOCOL = Path("protocol_20261110")
H_TASKS = (5, 6, 7, 8, 9)
H_TRIALS = range(5, 30)


def trial_panels(split):
    panel = split["common_evaluation"]
    if (split["suite"] != "libero_spatial" or split["task_order_index"] != 0
            or tuple(split["final_H"]) != H_TASKS or panel["H_trial_ids"] != list(H_TRIALS)
            or panel["H_repetitions_per_state"] != 2
            or panel["environment_seed_rule"] != "142 + trial_id + 1000 * repetition_index"
            or panel["policy_seed_rule"] != "10000 + 100 * task_id + trial_id + 1000 * repetition_index"):
        raise ValueError("H differs from the frozen Spatial panel")
    tasks = {task["task_id"]: task for task in split["tasks"]}
    return [[
        {"task_id": task, "trial_id": trial, "repetition_index": rep,
         "environment_seed": 142 + trial + 1000 * rep,
         "policy_seed": 10000 + 100 * task + trial + 1000 * rep,
         "initial_state_sha256": tasks[task]["state_sha256_by_trial_id"][trial]}
        for trial in H_TRIALS for rep in range(2)
    ] for task in H_TASKS]


def slurm_status(jobs):
    records = subprocess.check_output(
        ["sacct", "-X", "-j", ",".join(jobs), "-P", "-n", "-o", "JobIDRaw,State,ExitCode"],
        text=True,
    ).splitlines()
    return {job: (state, code) for job, state, code in (line.split("|") for line in records if line)}


def checked_policy(data, condition, seed, job, checkpoint, sha):
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to((data / "runs" / job).resolve()):
        raise ValueError(f"{condition}/{seed} checkpoint escaped its job output")
    check = read(checkpoint / "checkpoint_check.json")
    if (check["identity"]["slurm_job_id"] != job or check["status"] != "PASS"
            or check["strict_state_load"] is not True or check["digest"] != sha or check["exported_digest"] != sha
            or (condition in POOLS and check["fixed_probe"]["status"] != "PASS")):
        raise ValueError(f"Unverified checkpoint for {condition}/{seed}")
    return {"condition": condition, "seed": seed, "training_job_id": job, "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha, "model": "rl-source10" if condition in POOLS else "sft-fewshot"}


def build_plan(repo, data, partition, commit):
    dispatch_path = data / RUN_PROTOCOL / "e3_dispatch.json"
    e3 = read(dispatch_path)
    expected = [(row["condition"], row["seed"]) for row in schedule()]
    if (e3["state"] != "COMPLETE" or e3["partition"] != partition
            or [(row["condition"], row["seed"]) for row in e3["runs"]] != expected
            or any(row["state"] not in ("COMPLETE_VALID", "COMPLETE_NO_SIGNAL") for row in e3["runs"])):
        raise ValueError("H remains sealed until all twelve registered E3 endpoints are complete")
    jobs = [row["job_id"] for row in e3["runs"]]
    accounting = slurm_status(jobs)
    if len(set(jobs)) != len(expected) or any(accounting.get(job) != ("COMPLETED", "0:0") for job in jobs):
        raise ValueError("Slurm must confirm all E3 training/evaluation jobs completed successfully")

    registry = read(repo / PROTOCOL / "registry.json")
    split_path = repo / registry["split_path"]
    split_sha = digest(split_path)
    if split_sha != registry["split_sha256"] or split_sha != e3["prerequisites"]["split_sha256"]:
        raise ValueError("Frozen H split does not match the E3 preregistration")
    source = registry["common_start"]
    if source["actor_state_sha256"] != e3["prerequisites"]["b0_sha256"]:
        raise ValueError("The B0 policy differs from the registered E3 start")
    policies = [checked_policy(data, "B0", None, str(source["job_id"]), data / source["actor_checkpoint"],
                               source["actor_state_sha256"])]
    for row in e3["runs"]:
        condition, seed, job = row["condition"], row["seed"], row["job_id"]
        last = row["milestones"][-1]
        if condition == "SFT-cont":
            if last["optimizer_steps"] != 2048:
                raise ValueError("H requires the preregistered 2,048-update SFT endpoint")
        elif (last["requested_policy_env_steps"] != 200000
              or not 200000 <= last["policy_env_steps"] < 201200):
            raise ValueError("H requires the preregistered 200k-control-step PPO endpoint")
        policies.append(checked_policy(data, condition, seed, job, Path(last["checkpoint_path"]),
                                       last["checkpoint_sha256"]))
    return {"schema_version": 1, "protocol_id": registry["protocol_id"], "evaluation_commit": commit,
            "e3_commit": e3["commit"], "e3_dispatch_sha256": digest(dispatch_path), "split_sha256": split_sha,
            "primary_comparison": ["M", "B2"], "policy_path": "deployed", "horizon": 800,
            "shards": trial_panels(read(split_path)), "policies": policies}


def submit_next(repo, data, partition):
    import fcntl

    commit = fixed_commit(repo)
    folder = data / RUN_PROTOCOL
    plan_path = folder / "final_h_plan.json"
    with (folder / "final_h.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        plan = build_plan(repo, data, partition, commit)
        if plan_path.exists():
            if read(plan_path) != plan:
                raise ValueError("The H evaluation plan is already locked to different evidence")
        else:
            write(plan_path, plan)
        plan_sha = digest(plan_path)
        submission_prefix = [str(repo / "tools/xvla/job.sh"), str(repo), "benchmark_protocol_final",
                             "1", str(plan_path), plan_sha]
        receipts = {}
        for path in (data / "runs").glob("submission-*.json"):
            receipt = read(path)
            if receipt["mode"] == "benchmark_protocol_final" and receipt["command"][-3:-1] == [str(plan_path), plan_sha]:
                if receipt["command"][-7:-1] != submission_prefix:
                    raise ValueError("H submission does not use the locked evaluation worktree")
                index = int(receipt["command"][-1])
                if index in receipts:
                    raise ValueError("Multiple Slurm receipts found for one H policy")
                receipts[index] = receipt["job_id"].split(";")[0]
        accounting = slurm_status(list(receipts.values())) if receipts else {}
        results = []
        for index, policy in enumerate(plan["policies"]):
            if index not in receipts:
                args = SimpleNamespace(repo=repo, data=data, partition=partition, mode="benchmark_protocol_final",
                                       gpus=1, gpu_model="4090")
                with (data / "submit.lock").open("a") as submit_lock:
                    fcntl.flock(submit_lock, fcntl.LOCK_EX)
                    submit_locked(args, None, [str(plan_path), plan_sha, str(index)], benchmark=policy,
                                  walltime="04:00:00")
                return
            job = receipts[index]
            state = accounting.get(job)
            if state != ("COMPLETED", "0:0"):
                print(json.dumps({"policy": policy, "job_id": job, "slurm_state": state}))
                return
            run_dir = data / "runs" / job
            receipt = read(run_dir / "final_h_summary.json")
            episodes = run_dir / "final_h_episodes.jsonl"
            if (receipt["status"] != "PASS" or receipt["job_id"] != job or receipt["plan_sha256"] != plan_sha
                    or receipt["policy_index"] != index or receipt["episodes_sha256"] != digest(episodes)):
                raise ValueError(f"H completion evidence differs for job {job}")
            results.append({**policy, "job_id": job, "episodes": str(episodes)})
        analysis_manifest = folder / "final_h_analysis_manifest.json"
        write(analysis_manifest, {"plan_sha256": plan_sha, "policies": results})
        print(json.dumps({"state": "COMPLETE", "analysis_manifest": str(analysis_manifest)}))


def evaluate(plan_path, plan_sha, index):
    from xvla_rlinf import benchmark

    if digest(plan_path) != plan_sha:
        raise ValueError("H evaluation plan changed after Slurm submission")
    plan = read(plan_path)
    repo = Path(__file__).resolve().parents[2]
    if fixed_commit(repo) != plan["evaluation_commit"]:
        raise ValueError("H evaluation must use the locked evaluation commit")
    policy = plan["policies"][index]
    job = os.environ["SLURM_JOB_ID"]
    root = Path(os.environ["XVLA_RUN_DIR"])
    summaries = {}
    with (root / "final_h_episodes.jsonl").open("x", encoding="utf-8") as journal:
        try:
            for panel in plan["shards"]:
                task = panel[0]["task_id"]
                output = root / f"task_{task}"
                output.mkdir()
                os.environ["XVLA_RUN_DIR"] = str(output)
                benchmark.main([
                    "--model", policy["model"], "--checkpoint", policy["checkpoint"], "--policy-path", "deployed",
                    "--expected-checkpoint-sha256", policy["checkpoint_sha256"],
                ], trial_plan=panel)
                summary = read(output / "benchmark_summary.json")
                if (summary["status"] != "PASS" or summary["completed_episodes"] != len(panel)
                        or summary["trial_plan"] != panel or summary["job_id"] != job):
                    raise RuntimeError(f"Incomplete H task {task} on job {job}")
                summaries[str(task)] = {"successes": summary["successes"], "episodes": len(panel)}
                journal.write((output / "benchmark_episodes.jsonl").read_text(encoding="utf-8"))
        finally:
            os.environ["XVLA_RUN_DIR"] = str(root)
    write(root / "final_h_summary.json", {
        "status": "PASS", "job_id": job, "plan_sha256": plan_sha, "policy_index": index,
        "policy": policy, "per_task": summaries, "episodes_sha256": digest(root / "final_h_episodes.jsonl"),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="action", required=True)
    submit = subcommands.add_parser("submit")
    submit.add_argument("--repo", type=Path, required=True)
    submit.add_argument("--data", type=Path, required=True)
    submit.add_argument("--partition", required=True)
    run = subcommands.add_parser("evaluate")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--plan-sha256", required=True)
    run.add_argument("--index", type=int, choices=range(1 + len(SEEDS) * (1 + len(POOLS))), required=True)
    args = parser.parse_args()
    if args.action == "submit":
        submit_next(args.repo.resolve(), args.data.resolve(), args.partition)
    else:
        evaluate(args.plan, args.plan_sha256, args.index)


if __name__ == "__main__":
    main()
