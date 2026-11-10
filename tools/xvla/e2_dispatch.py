"""Dispatch one durable, auditable E2 successor per completed pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


POOLS = ("reference", "expanded", "stratified")
TERMINAL = ("COMPLETE_VALID", "COMPLETE_NO_SIGNAL")
B0_DIGEST = "b584a11131874fc5d1c190af5859c4fab61e8c18d7f9de5b186b8885e6977e2f"


class BlockedResource(RuntimeError):
    """The scheduler rejected a successor before it could be queued."""


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def numeric_job(value):
    if not re.fullmatch(r"[0-9]+", value):
        raise argparse.ArgumentTypeError("Expected a numeric Slurm job ID")
    return value


def fixed_commit(repo):
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.run(["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
                            stdout=subprocess.DEVNULL, check=False)
    dirty = subprocess.run(
        ["git", "-C", str(repo), "diff", "--quiet", "--ignore-submodules=all", "HEAD", "--"],
        check=False,
    )
    if repo.name != commit or branch.returncode != 1 or dirty.returncode != 0:
        raise RuntimeError("E2 dispatcher requires the detached, clean fixed-commit worktree")
    return commit


def e1_evidence(repo, data, deployed, online, physical):
    if len({deployed, online, physical}) != 3:
        raise ValueError("E1 requires separate deployed, online and physical jobs")
    accounting = subprocess.check_output(
        ["sacct", "-X", "-j", f"{deployed},{online},{physical}", "-P", "-n", "-o", "JobIDRaw,State,ExitCode"],
        text=True,
    )
    completed = {job for job, state, code in (line.split("|") for line in accounting.splitlines() if line)
                 if state == "COMPLETED" and code == "0:0"}
    if not {deployed, online, physical} <= completed:
        raise RuntimeError("E1 prerequisites have not all completed successfully in Slurm")
    evidence = {}
    for mode, job, start, count, horizon in (("deployed", deployed, 10, 20, 800), ("online", online, 30, 10, 300)):
        path = data / "runs" / job / "benchmark_summary.json"
        contents = path.read_bytes()
        summary = json.loads(contents)
        plan = [(row["task_id"], row["trial_id"]) for row in summary["trial_plan"]]
        successes = summary["successes"]
        if (summary["status"] != "PASS" or summary["complete"] is not True or summary["job_id"] != job
                or summary["suite"] != "libero_spatial" or summary["completed_episodes"] != count
                or summary["protocol"]["policy_path"] != mode or summary["protocol"]["horizon"] != horizon
                or summary["checkpoint"]["checkpoint_state_sha256"] != B0_DIGEST
                or plan != [(0, trial) for trial in range(start, start + count)]
                or type(successes) is not int or not 0 <= successes <= count):
            raise ValueError(f"E1 {mode} evidence does not match the frozen B0 panel: {path}")
        evidence[mode] = {"job_id": job, "summary_sha256": hashlib.sha256(contents).hexdigest(),
                          "successes": successes, "episodes": count}
    registry = read(repo / "configs/xvla/protocol_20261110/registry.json")
    split_digest = hashlib.sha256((repo / registry["split_path"]).read_bytes()).hexdigest()
    source = registry["common_start"]
    demo_digest = hashlib.sha256((data / source["manifest"]).read_bytes()).hexdigest()
    path = data / "runs" / physical / "e1_source_audit.json"
    contents = path.read_bytes()
    audit = json.loads(contents)
    if (audit["status"] != "PASS" or audit["job_id"] != physical
            or audit["audit_scope"] != "libero_spatial_task0_source"
            or audit["split_sha256"] != split_digest or split_digest != registry["split_sha256"]
            or audit["demo_manifest_sha256"] != demo_digest or demo_digest != source["manifest_sha256"]):
        raise ValueError(f"E1 physical audit does not match the frozen split and B0 manifest: {path}")
    evidence["physical"] = {"job_id": physical, "summary_sha256": hashlib.sha256(contents).hexdigest(),
                            "split_sha256": split_digest, "demo_manifest_sha256": demo_digest}
    return evidence


def receipt_matches(receipt, args, pool):
    command = receipt.get("command", [])
    marker = str(args.repo / "tools/xvla/job.sh")
    if marker not in command:
        return False
    tail = command[command.index(marker) + 1:]
    plan_name = "expanded" if pool == "stratified" else pool
    plan = args.repo / "configs/xvla/source_pools" / f"{plan_name}.json"
    digest = hashlib.sha256(plan.read_bytes()).hexdigest()
    sampling = "stratified" if pool == "stratified" else "uniform"
    checkpoint = args.data / "runs/19525/sft_source10/checkpoints/global_step_1024/actor"
    return (
        receipt.get("mode") == "ppo_e2" and receipt.get("requested_gpu_model") == "4090"
        and receipt.get("requested_gpus") == 1
        and receipt.get("initial_sft", {}).get("checkpoint_job_id") == "19525"
        and receipt.get("initial_sft", {}).get("sft_steps") == 1024
        and tail == [str(args.repo), "ppo_e2", "1", str(checkpoint), str(plan), digest, sampling,
                     pool, args.partition]
    )


def endpoint(args):
    root = (args.data / "runs" / args.job_id).resolve()
    receipt = read(root.parent / f"submission-{args.job_id}.json")
    if not receipt_matches(receipt, args, args.pool):
        raise ValueError("Completed job does not match its fixed E2 submission")
    budget_path = root / "control_step_budget.json"
    budget = read(budget_path)
    observed, updates = budget["policy_env_steps"], budget["optimizer_updates"]
    expected_plan, expected_sampling = receipt["command"][-4:-2]
    if (budget["state"] not in TERMINAL or budget["requested_policy_env_steps"] != 25000
            or budget["max_rollout_batch_control_steps"] != 1200 or not 25000 <= observed < 26200
            or budget["overshoot_policy_env_steps"] != observed - 25000 or budget["partial_rollout"]
            or budget["reset_plan_sha256"] != expected_plan or budget["sampling_mode"] != expected_sampling
            or (budget["state"] == "COMPLETE_VALID") != (updates > 0)):
        raise RuntimeError("E2 endpoint lacks a valid completed control-step budget")
    checkpoint = Path(budget["checkpoint_path"]).resolve()
    if not checkpoint.is_relative_to(root):
        raise RuntimeError("E2 checkpoint lies outside its own Slurm output directory")
    check_path = checkpoint / "checkpoint_check.json"
    check = read(check_path)
    if (check["status"] != "PASS" or check["strict_state_load"] is not True
            or check["identity"]["slurm_job_id"] != args.job_id
            or Path(check["checkpoint_dir"]).resolve() != checkpoint
            or check["digest"] != check["exported_digest"]
            or check["fixed_probe"]["status"] != "PASS"):
        raise RuntimeError("E2 successor requires successful independent checkpoint validation")
    return {"job_id": args.job_id, "state": budget["state"], "policy_env_steps": observed,
            "optimizer_updates": updates, "checkpoint_sha256": check["digest"],
            "budget_path": str(budget_path), "checkpoint_check_path": str(check_path)}


def submit(args, pool, state, path):
    group = state["pools"].setdefault(pool, {})
    if "job_id" in group:
        return
    receipts = args.data / "runs"
    if group.get("state") in ("SUBMITTING", "SUBMISSION_UNCERTAIN"):
        # Recover the receipt if sbatch finished before the dispatcher committed
        # its state. Without a receipt, an accepted job cannot be ruled out.
        known = set(group["prior_receipts"])
        recovered = [
            value for receipt_path in receipts.glob("submission-*.json")
            if receipt_path.name not in known and receipt_matches(value := read(receipt_path), args, pool)
        ]
        if len(recovered) == 1:
            job = numeric_job(recovered[0]["job_id"].split(";")[0])
            state["pools"][pool] = {"state": "SUBMITTED", "job_id": job}
            write(path, state)
            return
        group["state"] = "SUBMISSION_UNCERTAIN"
        write(path, state)
        raise RuntimeError("Ambiguous E2 submission. Reconcile Slurm job and receipt before retrying")

    group.update(state="SUBMITTING", prior_receipts=sorted(p.name for p in receipts.glob("submission-*.json")))
    write(path, state)
    command = [
        sys.executable, str(args.repo / "tools/xvla/submit.py"), "--repo", str(args.repo),
        "--data", str(args.data), "--partition", args.partition, "--mode", "ppo_e2",
        "--reset-pool", pool, "--checkpoint-job-id", "19525", "--sft-steps", "1024",
        "--gpu-model", "4090", "--gpus", "1",
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        reason = result.stderr
        blocked = any(code in reason for code in ("BLOCKED_GPU_BUDGET", "AssocMaxSubmitJobLimit"))
        group["state"] = "BLOCKED_RESOURCE" if blocked else "SUBMISSION_UNCERTAIN"
        group["error"] = reason[-3000:]
        write(path, state)
        error = BlockedResource if blocked else RuntimeError
        raise error(f"E2 successor submission failed: {group['state']}\n{reason}")
    receipt = json.loads(result.stdout)
    if not receipt_matches(receipt, args, pool):
        raise RuntimeError("E2 submitter returned an unexpected receipt")
    job = numeric_job(receipt["job_id"].split(";")[0])
    state["pools"][pool] = {"state": "SUBMITTED", "job_id": job}
    write(path, state)


def dispatch_next(args, state, path):
    for pool in POOLS:
        if state["pools"].get(pool, {}).get("state") not in TERMINAL:
            submit(args, pool, state, path)
            return
    state["state"] = "COMPLETE"
    write(path, state)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "advance"))
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--deployed-job-id", type=numeric_job)
    parser.add_argument("--online-job-id", type=numeric_job)
    parser.add_argument("--physical-job-id", type=numeric_job)
    parser.add_argument("--job-id", type=numeric_job)
    parser.add_argument("--pool", choices=POOLS)
    args = parser.parse_args(argv)
    if args.action == "start":
        if not all((args.deployed_job_id, args.online_job_id, args.physical_job_id)) or args.job_id or args.pool:
            parser.error("start requires the three E1 job IDs")
    elif (not args.job_id or not args.pool
          or any((args.deployed_job_id, args.online_job_id, args.physical_job_id))):
        parser.error("advance requires the completed E2 job ID and pool")
    args.repo, args.data = args.repo.resolve(), args.data.resolve()
    commit = fixed_commit(args.repo)

    import fcntl

    folder = args.data / "protocol_20261110"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "e2_dispatch.json"
    with (folder / "e2_dispatch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = read(path) if path.is_file() else None
        if state is not None and (state["commit"], state["partition"]) != (commit, args.partition):
            raise RuntimeError("E2 chain is frozen to a different commit or partition")
        if args.action == "start":
            evidence = e1_evidence(
                args.repo, args.data, args.deployed_job_id, args.online_job_id, args.physical_job_id
            )
            if state is None:
                state = {"schema_version": 1, "commit": commit, "partition": args.partition,
                         "e1": evidence, "pools": {},
                         "state": "RUNNING" if evidence["online"]["successes"] else "COMPLETE_NO_SIGNAL"}
                write(path, state)
            elif state["e1"] != evidence:
                raise RuntimeError("Existing E2 chain has different prerequisite evidence")
            if state["state"] == "RUNNING":
                dispatch_next(args, state, path)
        else:
            if state is None or state["pools"].get(args.pool, {}).get("job_id") != args.job_id:
                raise RuntimeError("Completed E2 job is not registered for this pool")
            state["pools"][args.pool] = endpoint(args)
            write(path, state)
            dispatch_next(args, state, path)
        print(json.dumps(state), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BlockedResource as error:
        print(f"XVLA_DISPATCH_BLOCKED {error}", file=sys.stderr, flush=True)
        raise SystemExit(75) from None
