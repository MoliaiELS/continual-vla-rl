"""Evaluate fixed, paired LIBERO-Spatial trials using the native X-VLA policy."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


SUITE = "libero_spatial"
TASKS = 10
MAX_TRIALS_PER_JOB = 50
HORIZON = 800
SETTLING = 10
CHUNK = 30
SOLVER_STEPS = 10
DOMAIN = 3
MODEL_REVISIONS = {
    "libero": ("2toINF/X-VLA-Libero", "129e71460678b7236cee6fc9707f09d9fa0c3590"),
    "pt": ("2toINF/X-VLA-Pt", "c1c4a64a7e03ac5b95c468bf1578f3d03651b53b"),
}
TRIAL_FIELDS = ("task_id", "trial_id", "environment_seed", "policy_seed")
SFT_MODELS = ("sft-smoke", "sft-fewshot")


def build_trials(start_task: int, task_count: int, trials: int) -> list[dict]:
    if any(type(value) is not int for value in (start_task, task_count, trials)):
        raise ValueError("Task and trial arguments must be integers")
    if start_task < 0 or task_count < 1 or start_task + task_count > TASKS:
        raise ValueError("Spatial task range must be a nonempty interval within IDs 0 through 9")
    if trials < 1 or task_count * trials > MAX_TRIALS_PER_JOB:
        raise ValueError(f"Evaluate between 1 and {MAX_TRIALS_PER_JOB} episodes per job")
    # The environment seed matches the pinned official client's default.
    return [
        {"task_id": task, "trial_id": trial, "environment_seed": 142 + trial,
         "policy_seed": 10_000 + 100 * task + trial}
        for task in range(start_task, start_task + task_count)
        for trial in range(trials)
    ]


def summarize(episodes: list[dict], plan: list[dict]) -> dict:
    def identity(row):
        if any(type(row.get(field)) is not int for field in TRIAL_FIELDS):
            raise ValueError("Episode identity requires task, trial and both seeds")
        return tuple(row[field] for field in TRIAL_FIELDS)

    expected = {identity(row) for row in plan}
    if not expected or len(expected) != len(plan):
        raise ValueError("Trial plan must be nonempty and unique")
    seen, per_task = set(), defaultdict(list)
    for row in episodes:
        key = identity(row)
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unexpected trial identity: {key}")
        steps = row.get("policy_env_steps")
        settling = row.get("settling_env_steps")
        if (type(row.get("success")) is not bool or type(steps) is not int
                or not 1 <= steps <= HORIZON or type(settling) is not int or settling != SETTLING):
            raise ValueError(f"Invalid episode outcome or step counts: {key}")
        seen.add(key)
        per_task[row["task_id"]].append(row)
    if seen != expected:
        raise ValueError(f"Missing {len(expected - seen)} planned trial results")

    def counts(rows):
        successes = sum(row["success"] for row in rows)
        return {"episodes": len(rows), "successes": successes,
                "success_rate": successes / len(rows),
                "policy_env_steps": sum(row["policy_env_steps"] for row in rows),
                "settling_env_steps": sum(row["settling_env_steps"] for row in rows)}

    tasks = {str(task): counts(rows) for task, rows in sorted(per_task.items())}
    return {"complete": True, "planned_episodes": len(plan), "completed_episodes": len(episodes),
            "per_task": tasks, **counts(episodes),
            "macro_success_rate": sum(row["success_rate"] for row in tasks.values()) / len(tasks)}


def _load_policy(model_name, checkpoint):
    import torch
    from omegaconf import OmegaConf
    from .policy import XVLAPolicy

    is_sft = model_name in SFT_MODELS
    base = "pt" if is_sft else model_name
    model_path = Path(os.environ["XVLA_PT_MODEL_PATH" if base == "pt" else "XVLA_MODEL_PATH"]).resolve()
    repo_id, revision = MODEL_REVISIONS[base]
    receipt = json.loads((model_path / "verified.json").read_text(encoding="utf-8"))
    if (receipt.get("repo_id"), receipt.get("revision")) != (repo_id, revision):
        raise ValueError("Selected model does not match its pinned download receipt")
    config = OmegaConf.create({
        "model_type": "xvla", "model_path": str(model_path), "precision": "fp32",
        "is_lora": False, "num_action_chunks": CHUNK, "add_value_head": False,
        "xvla": {"update_scope": "prompt_heads", "domain_rows": [DOMAIN], "domain_id": DOMAIN,
                 "steps": SOLVER_STEPS, "eval_sampler": "native_eval", "eval_history_mode": "native_history",
                 "image_orientation": "native_raw", "rotation_backend": "robosuite"},
    })
    provenance = {"name": model_name, "repo_id": repo_id, "revision": revision, "model_path": str(model_path)}
    if is_sft:
        from .audit_workers import state_digest
        from .checkpoint_check import read_evidence

        checkpoint = checkpoint.resolve()
        evidence, weights = read_evidence(checkpoint)
        check = json.loads((checkpoint / "checkpoint_check.json").read_text(encoding="utf-8"))
        if (check.get("status") != "PASS" or check.get("strict_state_load") is not True
                or check.get("digest") != evidence["digest"]):
            raise ValueError("SFT checkpoint must pass the independent strict full-state audit")
        saved = evidence["model_config"]
        saved_xvla = saved.get("xvla", {})
        if (Path(saved["model_path"]).resolve() != model_path or saved.get("add_value_head") is not False
                or saved.get("num_action_chunks") != CHUNK
                or any(saved_xvla.get(key) != value for key, value in {
                    "update_scope": "prompt_heads", "domain_rows": [DOMAIN], "domain_id": DOMAIN,
                    "steps": SOLVER_STEPS,
                }.items())):
            raise ValueError("SFT policy layout must match the fixed Pt prompt-head protocol")

    model = XVLAPolicy.from_checkpoint(config, torch.float32)
    if model.core.num_actions != CHUNK:
        raise ValueError("Native X-VLA checkpoint action horizon must be 30")
    if is_sft:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if state_digest(state) != evidence["digest"]:
            raise RuntimeError("SFT export changed since its independent audit")
        model.load_state_dict(state, strict=True)
        del state
        provenance.update(checkpoint=str(checkpoint), checkpoint_state_sha256=evidence["digest"])
    model = model.to(device="cuda:0", dtype=torch.float32).eval().requires_grad_(False)
    return model, provenance


def _make_environment(plan):
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    from rlinf.envs.sim.libero.utils import get_benchmark_overridden

    if not hasattr(LiberoEnv, "_step_xvla"):
        raise RuntimeError("The pinned RLinf runtime must have the audited X-VLA patch applied")
    suite = get_benchmark_overridden(SUITE)()
    if suite.get_num_tasks() != TASKS:
        raise RuntimeError("Unexpected task count for the pinned Spatial benchmark")
    states = [suite.get_task_init_states(task) for task in range(TASKS)]
    offsets = [sum(len(previous) for previous in states[:task]) for task in range(TASKS)]
    for row in plan:
        if row["trial_id"] >= len(states[row["task_id"]]):
            raise ValueError(f"Requested trial exceeds available initial states: {row}")
    first = plan[0]
    config = OmegaConf.create({
        "task_suite_name": SUITE, "seed": first["environment_seed"], "group_size": 1, "is_eval": True,
        "specific_reset_id": offsets[first["task_id"]] + first["trial_id"], "use_fixed_reset_state_ids": True,
        "use_ordered_reset_state_ids": True, "auto_reset": False, "ignore_terminations": False,
        "max_episode_steps": HORIZON, "max_steps_per_rollout_epoch": CHUNK * ((HORIZON + CHUNK - 1) // CHUNK),
        "num_steps_wait": SETTLING, "reset_gripper_open": False, "use_rel_reward": True, "reward_coef": 1.0,
        "xvla_absolute_control": True, "xvla_raw_observations": True, "xvla_measured_state": True,
        "xvla_history_reset": True, "xvla_strict_episodes": True,
        "init_params": {"camera_heights": 256, "camera_widths": 256}, "video_cfg": {"save_video": False},
    })
    env = LiberoEnv(config, num_envs=1, seed_offset=0, total_num_processes=1, worker_info=None)
    return env, states, offsets


def _run_episode(model, env, row, states, offsets):
    import numpy as np
    import torch

    started = time.monotonic()
    task, trial = row["task_id"], row["trial_id"]
    # LiberoEnv._reconfigure seeds the simulator child before reset. A single
    # environment ensures its seed multiplier equals one.
    env.seed = row["environment_seed"]
    env.specific_reset_id = offsets[task] + trial
    env.is_start = True
    old_policy = int(env._xvla_policy_steps[0])
    old_settling = int(env._xvla_settling_steps[0])
    obs, _ = env.reset()
    if (int(env.task_ids[0]), int(env.trial_ids[0])) != (task, trial) or not bool(obs["reset_mask"][0]):
        raise RuntimeError("Simulator reset did not honor the explicit trial or history reset")
    # Generate on CPU so the paired initial-noise stream is independent of
    # the allocated GPU's CUDA random-kernel launch geometry.
    generator = torch.Generator(device="cpu").manual_seed(row["policy_seed"])
    count = 0
    for query in range(1, (HORIZON + CHUNK - 1) // CHUNK + 1):
        noise = torch.randn((1, CHUNK, 20), generator=generator, dtype=torch.float32).to("cuda:0")
        actions, _ = model.predict_action_batch(obs, mode="eval", compute_values=False, initial_noise=noise)
        if actions.shape != (1, CHUNK, 7) or not np.isfinite(actions).all():
            raise RuntimeError("Unexpected native action shape or non-finite actions")
        observations, _, terminated, truncated, infos = env.chunk_step(actions)
        executed = infos[-1]["executed_mask"][0].cpu().numpy()
        valid = int(executed.sum())
        if valid < 1 or not np.array_equal(executed, np.arange(CHUNK) < valid):
            raise RuntimeError("Simulator did not execute a nonempty action prefix")
        count += valid
        terminal = (terminated | truncated)[0].cpu().numpy()
        if terminal.any():
            if np.flatnonzero(terminal).tolist() != [valid - 1]:
                raise RuntimeError("Control steps occurred after the first terminal action")
            success = bool(terminated.any())
            break
        if valid != CHUNK:
            raise RuntimeError("Simulator dropped actions without terminating the episode")
        obs = observations[-1]
    else:
        raise RuntimeError("Episode did not terminate or truncate by the fixed horizon")
    settling = int(env._xvla_settling_steps[0]) - old_settling
    if (count != int(env._xvla_policy_steps[0]) - old_policy or settling != SETTLING
            or count > HORIZON or (not success and count != HORIZON)):
        raise RuntimeError("Execution masks and simulator step counters disagree")
    description = env.task_suite.get_task(task)
    initial_state = np.ascontiguousarray(states[task][trial])
    return {**row, "task_name": description.name, "language": description.language,
            "initial_state_sha256": hashlib.sha256(initial_state.tobytes()).hexdigest(),
            "status": "success" if success else "timeout", "success": success,
            "policy_env_steps": count, "settling_env_steps": settling, "policy_queries": query,
            "post_done_policy_steps": 0, "elapsed_seconds": time.monotonic() - started}


def _write_summary(path, record):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("libero", "pt", *SFT_MODELS), required=True)
    parser.add_argument("--start-task", type=int, default=0)
    parser.add_argument("--task-count", type=int, default=10)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, help="Audited SFT actor checkpoint directory")
    args = parser.parse_args(argv)
    if (args.checkpoint is not None) != (args.model in SFT_MODELS):
        parser.error("--checkpoint is required for SFT models and applies only to them")
    try:
        plan = build_trials(args.start_task, args.task_count, args.trials)
    except ValueError as error:
        parser.error(str(error))

    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] == "4090node1":
        raise RuntimeError("Benchmark requires the project Slurm launcher on a compute node")
    if os.environ.get("LIBERO_TYPE", "standard").lower() != "standard":
        raise ValueError("This benchmark requires standard LIBERO")
    os.environ["LIBERO_TYPE"] = "standard"
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not mask or mask != os.environ.get("RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Benchmark requires the unchanged Slurm CUDA visibility mask")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("Benchmark requires the launcher EGL backend")

    run_dir = Path(os.environ["XVLA_RUN_DIR"])
    summary_path = run_dir / "benchmark_summary.json"
    episodes = []
    record = {"schema_version": 1, "suite": SUITE, "model": args.model,
              "job_id": os.environ["SLURM_JOB_ID"], "trial_plan": plan,
              "protocol": {"sampler": "native_eval", "precision": "fp32", "domain_id": DOMAIN,
                           "chunk": CHUNK, "solver_steps": SOLVER_STEPS, "horizon": HORIZON,
                           "settling_steps": SETTLING, "concurrent_environments": 1,
                           "initial_noise_rng": "torch_cpu_per_trial"}}
    if summary_path.exists():
        raise FileExistsError(summary_path)
    with (run_dir / "benchmark_episodes.jsonl").open("x", encoding="utf-8") as journal:
        try:
            from .checkpoint_check import assert_context
            from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

            record["runtime"] = assert_context()
            egl_id = NvidiaGPUManager.get_egl_device_id(0)
            if egl_id is None:
                raise RuntimeError("Cannot verify an EGL device for Slurm CUDA ordinal 0")
            os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["EGL_DEVICE_ID"] = str(egl_id)
            record["egl_device_id"] = egl_id
            repository = Path(__file__).resolve().parents[1]
            record["code_commit"] = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
            model, record["checkpoint"] = _load_policy(args.model, args.checkpoint)
            env, states, offsets = _make_environment(plan)
            try:
                for row in plan:
                    result = _run_episode(model, env, row, states, offsets)
                    serialized = json.dumps(result, ensure_ascii=False)
                    journal.write(serialized + "\n")
                    journal.flush()
                    episodes.append(result)
                    print("XVLA_BENCHMARK_EPISODE " + serialized, flush=True)
            finally:
                # LiberoEnv inherits gym.Env.close(), which does not close the subprocess.
                env.env.close()
            record.update(status="PASS", **summarize(episodes, plan))
        except Exception as error:
            record.update(status="FAIL", complete=False, planned_episodes=len(plan),
                          completed_episodes=len(episodes), error_type=type(error).__name__, error=str(error))
            _write_summary(summary_path, record)
            raise
    _write_summary(summary_path, record)
    print("XVLA_BENCHMARK_SUMMARY " + json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
