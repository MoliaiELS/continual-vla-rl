"""Run the frozen E3 matrix serially after the paired E2 development panel."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

if __package__:
    from .e2_dispatch import BlockedResource, fixed_commit, numeric_job, read, write
    from .protocol_dev import TRIAL_IDS, read_deployed_shard
    from .submit import submit_locked
else:
    from e2_dispatch import BlockedResource, fixed_commit, numeric_job, read, write
    from protocol_dev import TRIAL_IDS, read_deployed_shard
    from submit import submit_locked


SEEDS = (101, 202, 303)
POOLS = {"B1": "reference", "B2": "expanded", "M": "stratified"}
SFT_STEPS = (512, 1024, 2048)
PPO_STEPS = (50000, 100000, 200000)
PROTOCOL = Path("configs/xvla/protocol_20261110")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prerequisites(repo, data, job):
    panel_path = data / "runs" / job / "protocol_dev_summary.json"
    panel = read(panel_path)
    slurm = subprocess.check_output(
        ["sacct", "-X", "-j", job, "-P", "-n", "-o", "JobIDRaw,State,ExitCode"], text=True,
    ).splitlines()
    registry = read(repo / PROTOCOL / "registry.json")
    e2 = read(data / "protocol_20261110/e2_dispatch.json")
    common = registry["common_start"]
    if (f"{job}|COMPLETED|0:0" not in slurm or e2["state"] != "COMPLETE"
            or panel["status"] != "PASS" or panel["complete"] is not True or panel["job_id"] != job
            or panel["protocol_id"] != registry["protocol_id"] or panel["suite"] != "libero_spatial"
            or panel["policy_path"] != "deployed" or panel["source_task_id"] != 0
            or panel["trial_ids"] != TRIAL_IDS or panel["e2_commit"] != e2["commit"]
            or panel["split_sha256"] != registry["split_sha256"]
            or digest(repo / registry["split_path"]) != registry["split_sha256"]
            or digest(data / common["manifest"]) != common["manifest_sha256"]):
        raise ValueError("E3 requires the complete deployed development panel and frozen split/manifest")
    expected = {"B0": (str(common["job_id"]), common["actor_state_sha256"])}
    for condition, pool in POOLS.items():
        pilot = e2["pools"][pool]
        if pilot["state"] != "COMPLETE_VALID":
            raise ValueError(f"{condition} did not demonstrate positive-reward PPO updates in E2")
        expected[condition] = (pilot["job_id"], pilot["checkpoint_sha256"])
    for condition, (checkpoint_job, checkpoint_sha) in expected.items():
        row = panel["conditions"][condition]
        if (row["checkpoint_job_id"] != checkpoint_job or row["checkpoint_state_sha256"] != checkpoint_sha
                or row["episodes"] != 30 or not 0 <= row["successes"] <= 30):
            raise ValueError(f"Development evidence differs from the frozen {condition} checkpoint")
    check = read(data / common["strict_reload_evidence"])
    if check["status"] != "PASS" or check["digest"] != expected["B0"][1]:
        raise ValueError("E3 requires the original B0 independent checkpoint receipt")
    split = read(repo / registry["split_path"])
    source = next(task for task in split["tasks"] if task["task_id"] == 0)
    hashes = [source["state_sha256_by_trial_id"][trial] for trial in TRIAL_IDS]
    if panel["initial_state_sha256_by_trial_id"] != hashes:
        raise ValueError("Development panel state hashes differ from the frozen split")
    return {"job_id": job, "summary_sha256": digest(panel_path), "e2_commit": e2["commit"],
            "split_sha256": registry["split_sha256"], "b0_sha256": expected["B0"][1]}


def schedule():
    return [{"condition": "SFT-cont", "seed": seed, "state": "READY"} for seed in SEEDS] + [
        {"condition": condition, "seed": seed, "state": "READY"} for seed in SEEDS for condition in POOLS
    ]


def retry_first_sft(data, state, commit, job):
    first, *remaining = state["runs"]
    if (state["state"] != "RUNNING" or first["condition"] != "SFT-cont"
            or first["state"] != "SUBMITTED" or first["job_id"] != job
            or any(row["state"] != "READY" for row in remaining)):
        raise ValueError("Retry requires the failed initial SFT-cont job and an untouched remaining matrix")
    if any((data / "runs" / job / "sft_cont/checkpoints").glob("global_step_*")):
        raise ValueError("Inspect existing checkpoints before retrying the initial SFT-cont job")
    accounting = subprocess.check_output(
        ["sacct", "-X", "-j", job, "-P", "-n", "-o", "JobIDRaw,State,ExitCode"], text=True,
    ).splitlines()
    failed = [line.split("|") for line in accounting if line.split("|", 1)[0] == job]
    if len(failed) != 1 or failed[0][1] != "FAILED" or failed[0][2] == "0:0":
        raise ValueError("Slurm must confirm the original SFT-cont job failed")
    first.setdefault("failed_attempts", []).append(
        {"job_id": job, "commit": state["commit"], "exit_code": failed[0][2]},
    )
    del first["job_id"]
    first["state"] = "READY"
    state["commit"] = commit


def submission(repo, data, partition, row):
    checkpoint = str(data / "runs/19525/sft_source10/checkpoints/global_step_1024/actor")
    if row["condition"] == "SFT-cont":
        mode, walltime = "sft_cont", "02:00:00"
        extra = [checkpoint, str(row["seed"])]
    else:
        mode, walltime = "ppo_e3", "24:00:00"
        pool = POOLS[row["condition"]]
        plan = repo / "configs/xvla/source_pools" / f"{'expanded' if pool == 'stratified' else pool}.json"
        sampling = "stratified" if pool == "stratified" else "uniform"
        extra = [checkpoint, str(plan), digest(plan), sampling, str(row["seed"])]
    return SimpleNamespace(repo=repo, data=data, partition=partition, mode=mode, gpus=1, gpu_model="4090"), extra, walltime


def verify_benchmark(root, job, folder, checkpoint, checkpoint_sha, panel, *, require_fixed_probe):
    check_path = checkpoint / "checkpoint_check.json"
    check = read(check_path)
    if (not checkpoint.is_relative_to(root) or check["identity"]["slurm_job_id"] != job
            or check["status"] != "PASS" or check["strict_state_load"] is not True
            or check["digest"] != checkpoint_sha or check["exported_digest"] != checkpoint_sha
            or (require_fixed_probe and check["fixed_probe"]["status"] != "PASS")):
        raise ValueError(f"Independent checkpoint check failed: {check_path}")
    eval_path = root / "source_dev" / folder / "benchmark_summary.json"
    episodes = read_deployed_shard(eval_path.parent, job=job, digest=checkpoint_sha,
                                   panel=panel, trial_ids=TRIAL_IDS)
    return {"checkpoint_path": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
            "successes": sum(row["success"] for row in episodes), "episodes": len(episodes),
            "benchmark_path": str(eval_path)}


def endpoint(repo, data, partition, row, panel):
    job = row["job_id"]
    root = (data / "runs" / job).resolve()
    args, extra, walltime = submission(repo, data, partition, row)
    receipt = read(root.parent / f"submission-{job}.json")
    expected = [str(repo / "tools/xvla/job.sh"), str(repo), args.mode, "1", *extra]
    if (receipt["command"][-len(expected):] != expected or f"--time={walltime}" not in receipt["command"]
            or receipt["requested_gpus"] != 1 or receipt["requested_gpu_model"] != "4090"):
        raise ValueError("E3 job does not match the frozen submission")
    results = []
    if row["condition"] == "SFT-cont":
        for step in SFT_STEPS:
            checkpoint = root / f"sft_cont/checkpoints/global_step_{step}/actor"
            state = read(checkpoint / "xvla_sft_data_state.json")
            if [rank["optimizer_steps"] for rank in state["ranks"]] != [step]:
                raise ValueError(f"SFT-cont checkpoint did not reach {step} optimizer updates")
            check = read(checkpoint / "checkpoint_check.json")
            results.append({"optimizer_steps": step, **verify_benchmark(
                root, job, f"global_step_{step}", checkpoint, check["digest"], panel,
                require_fixed_probe=False,
            )})
        return {"state": "COMPLETE_VALID", "milestones": results}

    budget = read(root / "control_step_budget.json")
    if (budget["state"] not in ("COMPLETE_VALID", "COMPLETE_NO_SIGNAL")
            or budget["training_seed"] != row["seed"] or budget["partial_rollout"]
            or budget["reset_plan_sha256"] != extra[2] or budget["sampling_mode"] != extra[3]
            or budget["requested_policy_env_steps"] != PPO_STEPS[-1]
            or budget["max_rollout_batch_control_steps"] != 1200
            or [item["requested_policy_env_steps"] for item in budget["milestones"]] != list(PPO_STEPS)):
        raise ValueError("Formal PPO budget or seed does not match the registry")
    for item in budget["milestones"]:
        step = item["requested_policy_env_steps"]
        if not step <= item["policy_env_steps"] < step + 1200:
            raise ValueError("Formal PPO milestone exceeded the rollout overshoot allowance")
        results.append({"requested_policy_env_steps": step, "policy_env_steps": item["policy_env_steps"],
                        "optimizer_updates": item["optimizer_updates"], **verify_benchmark(
                            root, job, f"control_step_{step}", Path(item["checkpoint_path"]).resolve(),
                            item["checkpoint_sha256"], panel, require_fixed_probe=True,
                        )})
    return {"state": budget["state"], "milestones": results}


def dispatch_next(repo, data, state, path):
    import fcntl

    row = next((row for row in state["runs"] if not row["state"].startswith("COMPLETE_")), None)
    if row is None:
        state["state"] = "COMPLETE"
        write(path, state)
        return
    if "job_id" in row:
        return
    args, extra, walltime = submission(repo, data, state["partition"], row)
    expected = [str(repo / "tools/xvla/job.sh"), str(repo), args.mode, "1", *extra]
    receipts = data / "runs"
    if row["state"] == "SUBMITTING":
        prior = set(row["prior_receipts"])
        recovered = []
        for candidate in receipts.glob("submission-*.json"):
            if candidate.name not in prior:
                record = read(candidate)
                if record["command"][-len(expected):] == expected:
                    recovered.append(record)
        if len(recovered) != 1:
            raise RuntimeError("Reconcile the interrupted E3 submission with Slurm before retrying")
        receipt = recovered[0]
    else:
        row.update(state="SUBMITTING", prior_receipts=sorted(p.name for p in receipts.glob("submission-*.json")))
        write(path, state)
        try:
            with (data / "submit.lock").open("a") as lock, contextlib.redirect_stdout(io.StringIO()):
                fcntl.flock(lock, fcntl.LOCK_EX)
                receipt = submit_locked(args, None, extra, initial_sft={"checkpoint": extra[0]}, walltime=walltime)
        except RuntimeError as error:
            if any(code in str(error) for code in ("BLOCKED_GPU_BUDGET", "AssocMaxSubmitJobLimit")):
                row["state"] = "BLOCKED_RESOURCE"
                write(path, state)
                raise BlockedResource(str(error)) from error
            raise
    row.update(state="SUBMITTED", job_id=numeric_job(receipt["job_id"].split(";")[0]))
    row.pop("prior_receipts", None)
    write(path, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "advance"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--partition")
    parser.add_argument("--dev-job-id", type=numeric_job)
    parser.add_argument("--job-id", type=numeric_job)
    parser.add_argument("--failed-job-id", type=numeric_job)
    args = parser.parse_args()
    if args.action == "start" and (not args.partition or not args.dev_job_id or args.job_id):
        parser.error("start requires --partition and --dev-job-id")
    if args.action == "advance" and (not args.job_id or args.partition or args.dev_job_id or args.failed_job_id):
        parser.error("advance requires --job-id")
    import fcntl

    repo, data = args.repo.resolve(), args.data.resolve()
    commit = fixed_commit(repo)
    path = data / "protocol_20261110/e3_dispatch.json"
    with path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(path) if path.exists() else None
        if state is not None and state["commit"] != commit and not args.failed_job_id:
            raise ValueError("E3 dispatch is frozen to a different commit")
        if args.action == "start":
            evidence = prerequisites(repo, data, args.dev_job_id)
            if state is None:
                if args.failed_job_id:
                    raise ValueError("No existing E3 dispatch to retry")
                state = {"schema_version": 1, "commit": commit, "partition": args.partition,
                         "state": "RUNNING", "prerequisites": evidence, "runs": schedule()}
                write(path, state)
            elif (state["partition"], state["prerequisites"]) != (args.partition, evidence):
                raise ValueError("E3 dispatch prerequisites differ from the registered run")
            elif args.failed_job_id:
                retry_first_sft(data, state, commit, args.failed_job_id)
                write(path, state)
        else:
            matches = [row for row in state["runs"] if row.get("job_id") == args.job_id] if state else []
            if len(matches) != 1:
                raise ValueError("Job ID is absent from the E3 dispatch chain")
            prerequisite = state["prerequisites"]
            panel_path = data / "runs" / prerequisite["job_id"] / "protocol_dev_summary.json"
            if digest(panel_path) != prerequisite["summary_sha256"]:
                raise ValueError("The paired development evidence changed after the E3 freeze")
            hashes = read(panel_path)["initial_state_sha256_by_trial_id"]
            matches[0].update(endpoint(repo, data, state["partition"], matches[0], hashes))
            write(path, state)
        dispatch_next(repo, data, state, path)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BlockedResource as error:
        print(f"XVLA_DISPATCH_BLOCKED {error}", file=sys.stderr, flush=True)
        raise SystemExit(75) from None
