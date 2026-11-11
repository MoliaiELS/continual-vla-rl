"""Resolve the frozen E2 checkpoints and evaluate the paired source panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

if __package__:
    from . import e2_dispatch
else:
    import e2_dispatch


CONDITIONS = ("B0", "B1", "B2", "M")
PROTOCOL_DIR = Path("protocol_20261110")
TRIAL_IDS = list(range(30))


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_completed_jobs(jobs):
    accounting = subprocess.check_output(
        ["sacct", "-X", "-j", ",".join(jobs), "-P", "-n", "-o", "JobIDRaw,State,ExitCode"],
        text=True,
    )
    completed = {job for job, state, code in (line.split("|") for line in accounting.splitlines() if line)
                 if state == "COMPLETED" and code == "0:0"}
    if not set(jobs) <= completed:
        raise RuntimeError("The B0, E1 and E2 prerequisite jobs must all be COMPLETED with exit 0:0")


def _check_b0(data, registry):
    source = registry["common_start"]
    checkpoint = (data / source["actor_checkpoint"]).resolve()
    check = e2_dispatch.read(checkpoint / "checkpoint_check.json")
    job = str(source["job_id"])
    digest = source["actor_state_sha256"]
    if (job != "19525" or digest != e2_dispatch.B0_DIGEST
            or check["status"] != "PASS" or check["strict_state_load"] is not True
            or check["identity"]["slurm_job_id"] != job
            or Path(check["checkpoint_dir"]).resolve() != checkpoint
            or check["digest"] != digest or check["exported_digest"] != digest):
        raise ValueError("B0 does not match its frozen independent checkpoint evidence")
    return {"checkpoint_job_id": job, "checkpoint": str(checkpoint), "checkpoint_state_sha256": digest}


def read_deployed_shard(folder, *, job, digest, panel, trial_ids):
    summary = e2_dispatch.read(folder / "benchmark_summary.json")
    protocol = summary["protocol"]
    if (summary["status"] != "PASS" or summary["complete"] is not True or summary["job_id"] != job
            or summary["suite"] != "libero_spatial" or summary["completed_episodes"] != len(trial_ids)
            or summary["checkpoint"]["checkpoint_state_sha256"] != digest
            or (protocol["policy_path"], protocol["history"], protocol["horizon"],
                protocol["terminal_noise"]) != ("deployed", "measured_state", 800, "off")):
        raise ValueError(f"Unmatched deployed benchmark summary: {folder}")
    rows = [json.loads(line) for line in (folder / "benchmark_episodes.jsonl").read_text(encoding="utf-8").splitlines()]
    planned = [(0, trial, 142 + trial, 10000 + trial) for trial in trial_ids]
    fields = ("task_id", "trial_id", "environment_seed", "policy_seed")
    if ([tuple(row[field] for field in fields) for row in summary["trial_plan"]] != planned
            or [tuple(row[field] for field in fields) for row in rows] != planned
            or any(row["initial_state_sha256"] != panel[row["trial_id"]] or row["policy_path"] != "deployed"
                   or type(row["success"]) is not bool for row in rows)
            or sum(row["success"] for row in rows) != summary["successes"]):
        raise ValueError(f"Deployed benchmark trials do not match the frozen panel: {folder}")
    return rows


def development_plan(repo, data, partition):
    """Return a bounded, immutable job request built from audited evidence."""
    repo, data = repo.resolve(), data.resolve()
    state = e2_dispatch.read(data / PROTOCOL_DIR / "e2_dispatch.json")
    commit = state["commit"]
    if (state["state"] != "COMPLETE" or state["partition"] != partition
            or not re.fullmatch("[0-9a-f]{40}", commit)):
        raise ValueError("Development benchmark requires the complete registered E2 chain")

    registry = e2_dispatch.read(repo / "configs/xvla/protocol_20261110/registry.json")
    split_path = repo / registry["split_path"]
    split = e2_dispatch.read(split_path)
    split_digest = _sha256(split_path)
    panel = split["common_evaluation"]
    if (split_digest != registry["split_sha256"] or split["protocol_id"] != "xvla-20261110"
            or split["suite"] != "libero_spatial" or panel["source_dev_task_id"] != 0
            or panel["source_dev_trial_ids"] != TRIAL_IDS):
        raise ValueError("Source development panel differs from the frozen registration")
    source = next(task for task in split["tasks"] if task["task_id"] == 0)
    # Keep all role and trial identities bound to the registered split.
    hashes = [source["state_sha256_by_trial_id"][trial] for trial in TRIAL_IDS]
    conditions = {"B0": _check_b0(data, registry)}
    legacy = state["e1"]["deployed"]
    legacy_job = e2_dispatch.numeric_job(legacy["job_id"])
    legacy_dir = data / "runs" / legacy_job
    summary = legacy_dir / "benchmark_summary.json"
    if _sha256(summary) != legacy["summary_sha256"]:
        raise ValueError("B0 deployed E1 summary changed after dispatch")
    read_deployed_shard(legacy_dir, job=legacy_job, digest=conditions["B0"]["checkpoint_state_sha256"],
                        panel=hashes, trial_ids=TRIAL_IDS[10:])
    for condition, pool in zip(CONDITIONS[1:], e2_dispatch.POOLS, strict=True):
        recorded = state["pools"][pool]
        job = e2_dispatch.numeric_job(recorded["job_id"])
        endpoint = e2_dispatch.endpoint(
            SimpleNamespace(repo=repo.parent / commit, data=data, job_id=job, pool=pool, partition=partition)
        )
        if endpoint != recorded or not re.fullmatch("[0-9a-f]{64}", endpoint["checkpoint_sha256"]):
            raise ValueError(f"{condition} no longer matches its recorded E2 endpoint")
        conditions[condition] = {
            "checkpoint_job_id": job,
            "checkpoint": str(Path(endpoint["checkpoint_check_path"]).parent),
            "checkpoint_state_sha256": endpoint["checkpoint_sha256"],
            "endpoint_state": endpoint["state"],
            "optimizer_updates": endpoint["optimizer_updates"],
            "training_policy_env_steps": endpoint["policy_env_steps"],
        }
    jobs = [group["checkpoint_job_id"] for group in conditions.values()] + [legacy_job]
    if len(set(jobs)) != len(jobs):
        raise ValueError("Expected distinct B0, E1 and E2 Slurm jobs")
    _check_completed_jobs(jobs)
    return {
        "schema_version": 1, "protocol_id": split["protocol_id"], "split_sha256": split_digest,
        "e2_commit": commit, "suite": split["suite"], "source_task_id": 0, "policy_path": "deployed",
        "trial_ids": TRIAL_IDS, "initial_state_sha256_by_trial_id": hashes, "conditions": conditions,
        "b0_reused_shard": {"job_id": legacy_job, "directory": str(legacy_dir),
                            "summary_sha256": legacy["summary_sha256"],
                            "episodes_sha256": _sha256(legacy_dir / "benchmark_episodes.jsonl")},
    }


def evaluate(plan, run_dir, job_id):
    """Run four independently loaded evaluations on the same allocation."""
    with (run_dir / "protocol_dev_plan.json").open("x", encoding="utf-8") as file:
        json.dump(plan, file, indent=2)
        file.write("\n")
    conditions = {}
    panel = plan["initial_state_sha256_by_trial_id"]
    legacy = plan["b0_reused_shard"]
    legacy_dir = Path(legacy["directory"])
    if (_sha256(legacy_dir / "benchmark_summary.json") != legacy["summary_sha256"]
            or _sha256(legacy_dir / "benchmark_episodes.jsonl") != legacy["episodes_sha256"]):
        raise ValueError("The submitted B0 reused shard has changed")
    for condition in CONDITIONS:
        checkpoint = plan["conditions"][condition]
        digest = checkpoint["checkpoint_state_sha256"]
        is_baseline = condition == "B0"
        directory = run_dir / condition
        directory.mkdir()
        trials = TRIAL_IDS[:10] if is_baseline else TRIAL_IDS
        subprocess.run([
            sys.executable, "-m", "xvla_rlinf.benchmark", "--model", "sft-fewshot" if is_baseline else "rl-source10",
            "--policy-path", "deployed", "--start-task", "0", "--task-count", "1",
            "--start-trial", "0", "--trials", str(len(trials)), "--checkpoint", checkpoint["checkpoint"],
            "--expected-checkpoint-sha256", digest,
        ], env={**os.environ, "XVLA_RUN_DIR": str(directory)}, check=True)
        rows = read_deployed_shard(directory, job=job_id, digest=digest, panel=panel, trial_ids=trials)
        eval_jobs = [job_id]
        if is_baseline:
            rows.extend(read_deployed_shard(legacy_dir, job=legacy["job_id"], digest=digest, panel=panel,
                                            trial_ids=TRIAL_IDS[10:]))
            eval_jobs.append(legacy["job_id"])
        conditions[condition] = {
            **checkpoint, "eval_job_ids": eval_jobs, "successes": sum(row["success"] for row in rows),
            "episodes": len(rows), "successful_trial_ids": [row["trial_id"] for row in rows if row["success"]],
            "evaluation_policy_env_steps": sum(row["policy_env_steps"] for row in rows),
            "evaluation_settling_env_steps": sum(row["settling_env_steps"] for row in rows),
        }
    result = {**plan, "job_id": job_id, "conditions": conditions, "complete": True, "status": "PASS"}
    e2_dispatch.write(run_dir / "protocol_dev_summary.json", result)
    print("XVLA_PROTOCOL_DEV " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=json.loads)
    args = parser.parse_args()
    evaluate(args.plan, Path(os.environ["XVLA_RUN_DIR"]), os.environ["SLURM_JOB_ID"])
