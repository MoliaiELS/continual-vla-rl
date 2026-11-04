"""Create, submit and execute a bounded, immutable Slurm experiment plan."""
import argparse
from contextlib import contextmanager
import datetime as dt
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

REVISION = "39e5240e879c80b6cda6b3a83763dad717f5c05d"
DATA_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def load_plan(directory):
    directory = Path(directory).resolve()
    plan = json.loads((directory / "plan.json").read_text(encoding="utf-8"))
    if Path(plan["directory"]).resolve() != directory:
        raise ValueError("Plan directory identity mismatch")
    return plan


def create(args):
    root = Path(args.directory).resolve()
    if (root / "plan.json").exists():
        raise FileExistsError("Use a new suite directory; existing experiment plans are immutable")
    if not re.fullmatch(r"[a-f0-9]{40}", args.commit):
        raise ValueError("A full immutable code commit is required")
    qualification = args.profile == "qualification"
    tasks = [0, 1] if qualification else [0, 1, 2, 3, 4]
    seeds = [1234] if qualification else [1234, 2345, 3456]
    iterations = 1 if qualification else 10
    eval_ids = list(range(40, 44 if qualification else 50))
    episodes = len(eval_ids) if qualification else 20
    data = Path(os.environ["FYP_DATA_ROOT"])
    plan = {
        "schema_version": 1, "profile": args.profile, "directory": str(root),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "commit": args.commit, "model_revision": REVISION,
        "dataset_revision": DATA_REVISION, "task_ids": tasks,
        "train_init_ids": list(range(40)), "eval_init_ids": eval_ids,
        "iterations_per_task": iterations, "episodes_per_eval_task": episodes,
        "seed_ids": seeds, "replay_manifest_path": str(root / "replay_manifest.json"),
        "replay_capacity_chunks": 256, "bc_coeff": 0.03,
        "empty_grpo_group_rule": "skip_all_optimizer_and_bc_updates_count_all_interaction",
        "optimizer_at_task_switch": "retained_in_same_process",
        "maximum_concurrent_gpus": 2, "runs": [], "evaluations": [], "jobs": [],
        "next_suite": str(Path(args.next_suite).resolve()) if getattr(args, "next_suite", None) else None,
        "exposure": {
            "id": {"suite": "libero_spatial", "adaptation_sft": "seen", "online_rl": "train"},
            "rl_holdout": {"suite": "libero_spatial", "adaptation_sft": "seen", "online_rl": "never", "bc": "never"},
            "camera_shift": {"suite": "libero_spatial_cam2", "shift": "bundled_fixed_camera_pose", "online_shift": "never"},
            "object_transfer": {"suite": "libero_object", "adaptation_sft": "not_in_declared_spatial_suite", "all_ancestor_training": "unknown", "online_rl": "never", "bc": "never", "action_statistics": "fixed_spatial_sft_prior"},
        },
    }

    def job(key, kind, **fields):
        entry = {"key": key, "kind": kind, "log_dir": str(root / "tasks" / key), **fields}
        plan["jobs"].append(entry)
        return entry

    def evaluation(run_id, stage, checkpoint, variant="id", panel=None, seed=99001, role="development"):
        panel = tasks if panel is None else panel
        key = f"eval_{run_id}_stage{stage}_{variant}"
        entry = {
            "run_id": run_id, "stage": stage,
            "policy_env_steps": stage * iterations * 4096,
            "task_ids": panel, "checkpoint": checkpoint, "eval_seed": seed,
            "expected_episodes_per_task": episodes, "variant": variant, "role": role,
            "evidence_dir": str(root / "tasks" / key / "evidence"),
        }
        plan["evaluations"].append(entry)
        job(key, "eval", evaluation=entry)

    job("prepare", "prepare")
    if qualification:
        job("oodcheck", "oodcheck")
    evaluation("sft", 0, None)
    for seed in seeds:
        for method in ["seq", "joint", "seq_er", "joint_er"]:
            run_id = f"{method}_s{seed}"
            training_dir = str(root / "tasks" / f"train_{run_id}")
            run = {"run_id": run_id, "method": method, "seed": seed,
                   "task_ids": tasks, "training_dir": training_dir,
                   "experiment_log_path": str(Path(training_dir) / "experiment.jsonl"),
                   "expected_policy_env_steps": len(tasks) * iterations * 4096}
            plan["runs"].append(run)
            job(f"train_{run_id}", "train", run=run)
            for stage in range(1, len(tasks) + 1):
                checkpoint = str(Path(training_dir) / f"checkpoints/global_step_{stage * iterations}/actor")
                evaluation(run_id, stage, checkpoint)
    # Qualification uses only development variants. The core final variants start
    # after every training/ID evaluation, under a protocol frozen before scores.
    variants = {"camera_shift": tasks, "object_transfer": [0, 1] if qualification else [3, 4]}
    if not qualification:
        variants["rl_holdout"] = [7, 8, 9]
    for variant, panel in variants.items():
        role = "development" if qualification else "final"
        evaluation("sft", 0, None, variant, panel, role=role)
        if not qualification:
            for run in plan["runs"]:
                checkpoint = str(Path(run["training_dir"]) / f"checkpoints/global_step_{len(tasks) * iterations}/actor")
                evaluation(run["run_id"], len(tasks), checkpoint, variant, panel, role=role)
    job("gate", "gate")
    atomic_json(root / "plan.json", plan)
    print(json.dumps({"directory": str(root), "jobs": len(plan["jobs"]), "runs": len(plan["runs"]), "profile": args.profile}))


@contextmanager
def submission_lock(root):
    path = Path(root) / "submission.lock"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        yield
    finally:
        path.unlink()


def _submit_one(plan, task, dependency):
    root = Path(plan["directory"])
    repo = Path(os.environ["FYP_REPO"])
    gpu = 2 if task["kind"] == "train" else 1 if task["kind"] in {"eval", "oodcheck"} else 0
    command = ["sbatch", "--parsable", f"--partition={os.environ.get('FYP_PARTITION', 'GPU')}",
               "--exclude=4090node1", "--nodes=1", "--ntasks=1",
               f"--cpus-per-task={8 if gpu else 2}",
               f"--time={'12:00:00' if task['kind'] == 'train' else '04:00:00'}",
               "--kill-on-invalid-dep=yes", f"--chdir={repo}",
               f"--job-name=fyp-{plan['profile'][:4]}-{task['kind']}",
               f"--output={root}/slurm/%j.out", f"--error={root}/slurm/%j.err"]
    if gpu:
        command.append(f"--gres=gpu:4090:{gpu}")
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command += [str(repo / "tools/idac/experiment_job.sh"), str(root), task["key"]]
    (root / "slurm").mkdir(exist_ok=True)
    result = subprocess.check_output(command, text=True).strip().split(";")[0]
    if not result.isdigit():
        raise RuntimeError(f"Invalid Slurm submission result: {result}")
    return {"key": task["key"], "job_id": result, "kind": task["kind"]}


def submit(args):
    plan = load_plan(args.directory)
    root = Path(plan["directory"])
    if (root / "submission.json").exists():
        raise FileExistsError("Already submitted; inspect status rather than duplicating jobs")
    repo = Path(os.environ["FYP_REPO"])
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != plan["commit"]:
        raise ValueError("Submission checkout differs from frozen plan")
    with submission_lock(root):
        if (root / "submission.json").exists():
            raise FileExistsError("Already submitted")
        first = _submit_one(plan, plan["jobs"][0], args.dependency)
        receipt = {"commit": actual, "directory": str(root), "jobs": [first],
                   "mode": "rolling_one_child", "complete": len(plan["jobs"]) == 1,
                   "last_job_id": first["job_id"]}
        atomic_json(root / "submission.json", receipt)
    print(json.dumps({"submitted": 1, "planned": len(plan["jobs"]), "last_job_id": first["job_id"], "directory": str(root)}))


def advance(args):
    """Only a successful allocated parent may enqueue one dependent successor."""
    plan = load_plan(args.directory)
    root = Path(plan["directory"])
    with submission_lock(root):
        receipt = json.loads((root / "submission.json").read_text())
        current = receipt["jobs"][-1]
        if (current["key"] != args.key or current["job_id"] != os.environ.get("SLURM_JOB_ID")
                or receipt.get("finished")):
            raise ValueError("Only the current allocated parent can advance this suite")
        task = next(entry for entry in plan["jobs"] if entry["key"] == args.key)
        result = json.loads((Path(task["log_dir"]) / "result.json").read_text())
        if not result.get("ok") or result.get("commit") != plan["commit"] or result.get("slurm_job_id") != current["job_id"]:
            raise ValueError("Parent task did not provide matching successful evidence")
        index = len(receipt["jobs"])
        if index < len(plan["jobs"]):
            child = _submit_one(plan, plan["jobs"][index], current["job_id"])
            receipt["jobs"].append(child)
            receipt.update(last_job_id=child["job_id"], complete=len(receipt["jobs"]) == len(plan["jobs"]))
        else:
            receipt["finished"] = True
            if plan.get("next_suite"):
                following = load_plan(plan["next_suite"])
                if following["commit"] != plan["commit"]:
                    raise ValueError("Continuation suite uses another code commit")
                submit(argparse.Namespace(directory=following["directory"], dependency=current["job_id"]))
                receipt["continuation_started"] = following["directory"]
        atomic_json(root / "submission.json", receipt)
    print(json.dumps({"advanced_from": args.key, "last_job_id": receipt["last_job_id"], "finished": receipt.get("finished", False)}))


def status(args):
    plan = load_plan(args.directory)
    root = Path(plan["directory"])
    receipt_path = root / "submission.json"
    if not receipt_path.exists():
        print(json.dumps({"submitted": False, "directory": str(root)}))
        return
    receipt = json.loads(receipt_path.read_text())
    ids = ",".join(entry["job_id"] for entry in receipt["jobs"])
    result = subprocess.check_output(["sacct", "-j", ids, "-X", "-n", "-P", "--format=JobID,State,ExitCode,Elapsed,NodeList"], text=True)
    states = {}
    for line in result.splitlines():
        fields = line.split("|")
        if len(fields) >= 5:
            states[fields[0]] = dict(state=fields[1], exit_code=fields[2], elapsed=fields[3], node=fields[4])
    counts = {}
    interesting = []
    for entry in receipt["jobs"]:
        state = states.get(entry["job_id"], {"state": "UNKNOWN"})
        name = state["state"]
        counts[name] = counts.get(name, 0) + 1
        if name not in {"COMPLETED", "PENDING"}:
            interesting.append({**entry, **state})
    print(json.dumps({"directory": str(root), "counts": counts, "planned_jobs": len(plan["jobs"]),
                      "submitted_jobs": len(receipt["jobs"]), "finished": receipt.get("finished", False),
                      "active_or_failed": interesting, "last_job_id": receipt.get("last_job_id")}, indent=2))
    for entry in interesting:
        if entry["state"] in {"RUNNING", "COMPLETING", "UNKNOWN"}:
            continue
        for suffix in ["out", "err"]:
            path = root / "slurm" / f"{entry['job_id']}.{suffix}"
            if path.is_file():
                print(f"FAILED_TASK_LOG {path}\n{path.read_text(errors='replace')[-6000:]}")


def export_evidence(args):
    plan = load_plan(args.directory)
    root = Path(plan["directory"])
    names = {"plan.json", "submission.json", "replay_manifest.json", "dataset_verified.json",
             "result.json", "task.json", "identity.json", "eval_summary.json", "experiment.jsonl",
             "ood_validation.json", "fyp_protocol.json", "summary.json", "summary.md",
             "success_matrix.csv", "variant_rates.csv"}
    destination = root / "evidence.tar.gz"
    with tarfile.open(destination, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink() and (path.name in names or (path.suffix == ".png" and path.parent.name == "oodcheck")):
                archive.add(path, arcname=str(path.relative_to(root)))
    print(destination)


def run_task(args):
    plan = load_plan(args.directory)
    task = next(entry for entry in plan["jobs"] if entry["key"] == args.key)
    log_dir = Path(task["log_dir"])
    if (log_dir / "result.json").exists():
        raise FileExistsError("Task already has a result; preserve it and create an explicit new attempt")
    log_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LOG_DIR"] = str(log_dir)
    os.environ["FYP_DEPLOYMENT_EVIDENCE"] = str(log_dir / "evidence")
    atomic_json(log_dir / "task.json", {**task, "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "commit": plan["commit"]})
    repo = Path(os.environ["FYP_REPO"])
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != plan["commit"]:
        raise ValueError("Execution checkout differs from the frozen plan")
    if task["kind"] in {"train", "eval"}:
        snapshot = json.loads((Path(os.environ["FYP_DATA_ROOT"]) / "reports/model.snapshot.json").read_text())
        if not snapshot.get("verified") or snapshot["revision"] != plan["model_revision"]:
            raise ValueError("Model snapshot is not verified or changed")
        model = Path(os.environ["FYP_MODEL_PATH"])
        for file in snapshot["files"]:
            if not (model / file["name"]).is_file() or (model / file["name"]).stat().st_size != file["size"]:
                raise ValueError(f"Missing or truncated model file: {file['name']}")
        with (log_dir / "identity.json").open("w") as output:
            subprocess.run([sys.executable, "tools/idac/doctor.py"], stdout=output, check=True)
    args_list = []
    if task["kind"] == "prepare":
        prepare_manifest(plan, repo)
    elif task["kind"] == "oodcheck":
        subprocess.run([sys.executable, "tools/idac/validate_ood.py", "--output", str(log_dir)], check=True)
    elif task["kind"] == "train":
        run = task["run"]
        method = run["method"].removesuffix("_er")
        er = run["method"].endswith("_er")
        experiment = {"method": method, "task_ids": plan["task_ids"],
                      "iterations_per_task": plan["iterations_per_task"],
                      "seed": run["seed"], "train_init_ids": plan["train_init_ids"]}
        # Hydra's flow mapping parser accepts JSON-style strings with unquoted keys.
        mapping = "{" + ",".join(f"{k}:{json.dumps(v, separators=(',', ':'))}" for k, v in experiment.items()) + "}"
        args_list = ["bash", "examples/embodiment/run_embodiment.sh", "crl_experiment/fyp_spatial_smoke",
                     f"+runner.fyp_experiment={mapping}", f"env.fixed_task_ids={json.dumps(plan['task_ids'], separators=(',', ':'))}",
                     f"actor.seed={run['seed']}", "actor.model.attn_implementation=sdpa",
                     f"runner.save_interval={plan['iterations_per_task']}",
                     f"algorithm.use_experience_replay={str(er).lower()}",
                     f"algorithm.bc_coeff={plan['bc_coeff'] if er else 0.0}"]
        if er:
            args_list += [f"+algorithm.replay_manifest_path={plan['replay_manifest_path']}",
                          f"+algorithm.replay_allowed_task_ids={json.dumps(plan['task_ids'], separators=(',', ':'))}",
                          f"+algorithm.replay_capacity_chunks={plan['replay_capacity_chunks']}",
                          f"+algorithm.replay_seed={run['seed']}", "+algorithm.replay_num_workers=0",
                          "+algorithm.replay_profile_gradient=True"]
        launch(args_list, log_dir, "train")
        verify_training(plan, task)
    elif task["kind"] == "eval":
        entry = task["evaluation"]
        panel = entry["task_ids"]
        suite = "libero_object" if entry["variant"] == "object_transfer" else "libero_spatial"
        args_list = ["bash", "examples/embodiment/eval_embodiment.sh", "crl_experiment/fyp_spatial_eval",
                     f"env.fixed_task_ids={json.dumps(panel, separators=(',', ':'))}",
                     f"env.eval.num_envs={len(panel)}", "env.eval.eval_per_task=1",
                     f"algorithm.eval_rollout_epoch={entry['expected_episodes_per_task']}",
                     f"+env.eval.fyp_eval_init_ids={json.dumps(plan['eval_init_ids'], separators=(',', ':'))}",
                     f"env.eval.task_suite_name={suite}", f"actor.seed={entry['eval_seed']}",
                     "actor.model.attn_implementation=sdpa", "actor.micro_batch_size=1",
                     "+rollout.inference_micro_batch_size=1", "rollout.enable_action_logging=False"]
        if entry["variant"] == "camera_shift":
            args_list += ["env/eval=libero_spatial_cam2", "env.eval.task_suite_name=libero_spatial_cam2", "env.eval.video_cfg.save_video=False"]
        if entry["checkpoint"]:
            args_list += ["actor.model.is_lora=True", f"+actor.model.lora_path={entry['checkpoint']}"]
        launch(args_list, log_dir, "eval")
        summary = json.loads((log_dir / "evidence/eval_summary.json").read_text())
        actual = len(summary.get("episodes", []))
        if actual != len(panel) * entry["expected_episodes_per_task"]:
            raise RuntimeError(f"Incomplete evaluation: {actual} episodes")
    elif task["kind"] == "gate":
        subprocess.run([sys.executable, "tools/idac/analyze_experiment.py", str(plan["directory"]), "--strict"], check=True)
        for run in plan["runs"]:
            verify_training(plan, {"run": run, "log_dir": run["training_dir"]})
    else:
        raise ValueError(task["kind"])
    atomic_json(log_dir / "result.json", {"ok": True, "kind": task["kind"], "commit": plan["commit"], "slurm_job_id": os.environ.get("SLURM_JOB_ID")})


def launch(command, log_dir, mode):
    resolved = subprocess.run(command + ["--cfg", "job", "--resolve"], check=True, text=True, capture_output=True)
    (log_dir / "resolved.yaml").write_text(resolved.stdout)
    atomic_json(log_dir / "command.json", command)
    subprocess.run(command, check=True)


def prepare_manifest(plan, repo):
    # Exact pinned files are verified before a manifest can enable ER.
    from huggingface_hub import HfApi
    import hashlib
    info = HfApi(endpoint="https://hf-mirror.com").dataset_info(
        "yifengzhu-hf/LIBERO-datasets", revision=DATA_REVISION, files_metadata=True)
    if info.sha != DATA_REVISION:
        raise ValueError("Dataset revision changed")
    sys.path.insert(0, str(repo / "LIBERO/libero/libero/benchmark"))
    from libero_suite_task_map import libero_task_map
    raw_root = Path(os.environ["FYP_DATA_ROOT"]) / "datasets/libero_raw"
    wanted = {f"libero_spatial/{name}_demo.hdf5" for name in libero_task_map["libero_spatial"][:5]}
    verified = []
    for entry in info.siblings:
        if entry.rfilename not in wanted:
            continue
        path = raw_root / entry.rfilename
        if not path.is_file() or path.stat().st_size != entry.size:
            raise ValueError(f"Replay source incomplete: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        expected_hash = entry.lfs.sha256 if entry.lfs else None
        if not expected_hash or digest.hexdigest() != expected_hash:
            raise ValueError(f"Replay source hash mismatch: {path}")
        verified.append({"file": entry.rfilename, "sha256": expected_hash, "size": entry.size})
    if len(verified) != 5:
        raise ValueError("Dataset did not provide all five allowed training files")
    atomic_json(Path(plan["directory"]) / "dataset_verified.json", {"revision": DATA_REVISION, "files": verified})
    subprocess.run([sys.executable, str(repo / "tools/idac/prepare_replay.py"),
                    "--source-dir", str(raw_root), "--output", plan["replay_manifest_path"],
                    "--task-ids", ",".join(map(str, plan["task_ids"])), "--demo-ids", "demo_0",
                    "--model-path", os.environ["FYP_MODEL_PATH"], "--source-revision", DATA_REVISION], check=True)


def verify_training(plan, task):
    run = task["run"]
    journal = Path(run["experiment_log_path"])
    records = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    iterations = [record for record in records if record.get("event") == "iteration"]
    expected = len(plan["task_ids"]) * plan["iterations_per_task"]
    if len(iterations) != expected:
        raise RuntimeError(f"Training journal incomplete: {run['run_id']} {len(iterations)}/{expected}")
    if not (journal.parent / f"checkpoints/global_step_{expected}/actor/adapter_model.bin").is_file():
        raise RuntimeError("Final adapter missing")
    log = (Path(task["log_dir"]) / "run_embodiment.log").read_text(errors="replace")
    updates, rl_gradients, bc_profiles = [], [], []
    for line in log.splitlines():
        for marker, collection in [("FYP_GRPO_UPDATE ", updates), ("FYP_GRPO_GRAD ", rl_gradients), ("FYP_BC_PROFILE ", bc_profiles)]:
            if marker in line:
                payload = line.split(marker, 1)[1]
                try:
                    collection.append(json.JSONDecoder().raw_decode(payload)[0])
                except json.JSONDecodeError:
                    pass
    if not any(any(math.isfinite(norm) and norm > 0 for norm in update.get("grad_norms", [])) for update in updates):
        raise RuntimeError("No verified nonzero RL gradient in this training run")
    if not any(entry.get("isolated_from_bc") and math.isfinite(entry["isolated_rl_grad_norm"])
               and entry["isolated_rl_grad_norm"] > 0 for entry in rl_gradients):
        raise RuntimeError("No isolated RL gradient evidence; combined BC gradients cannot certify GRPO")
    if run["method"].endswith("_er") and not any(
            math.isfinite(entry["grad_norm"]) and entry["grad_norm"] > 0
            and entry["nonzero_gradient_tensors"] > 0 and entry["samples"] > 0
            for entry in bc_profiles):
        raise RuntimeError("ER run did not verify an isolated nonzero BC gradient")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("directory")
    create_parser.add_argument("--profile", choices=["qualification", "core"], required=True)
    create_parser.add_argument("--commit", required=True)
    create_parser.add_argument("--next-suite")
    for name in ["submit", "status", "run-task", "advance", "export"]:
        p = sub.add_parser(name)
        p.add_argument("directory")
        if name == "submit":
            p.add_argument("--dependency")
        if name in {"run-task", "advance"}:
            p.add_argument("key")
    args = parser.parse_args()
    {"create": create, "submit": submit, "status": status, "run-task": run_task, "advance": advance, "export": export_evidence}[args.command](args)


if __name__ == "__main__":
    main()
