"""Screen physically admissible task-0 reset shifts with the native reference policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from .benchmark import (
    CHUNK, DOMAIN, HORIZON, SETTLING, SOLVER_STEPS,
    _load_policy, _make_environment, _run_episode, _write_summary, build_trials,
)
from .sft import sha256_file
from .source_reset_audit import BOWL, PLATE, RAMEKIN, SUITE, TASK_ID, TASK_NAME


OFFSET_M = 0.01
TRIAL_IDS = tuple(range(30, 40))


def _read_audit(path):
    audit = json.loads(path.read_text(encoding="utf-8"))
    if (audit["schema_version"] != 1 or audit["suite"] != SUITE or audit["task_id"] != TASK_ID
            or audit["task_name"] != TASK_NAME or audit["along_offset_m"] != OFFSET_M
            or audit["standard_unexpected_successes"]):
        raise ValueError("Expected the audited +1 cm source-task reset candidates")
    rows = audit["resets"]
    if tuple(row["trial_id"] for row in rows) != TRIAL_IDS:
        raise ValueError("The solvability check uses exactly source-training candidates 30–39")
    for row in rows:
        if (row["environment_seed"] != 142 + row["trial_id"]
                or type(row["passed_physics_screen"]) is not bool
                or row["passed_physics_screen"] != (not row["rejection_reasons"])):
            raise ValueError("Invalid paired reset identity or physics-screen outcome")
    return audit


def _position(env, name):
    return np.asarray(env.sim.data.body_xpos[env.env.obj_body_id[name]])


def _candidate_state(env, state, row):
    env.seed(row["environment_seed"])
    env.reset()
    env.set_init_state(state)
    original = row["standard"]["unmodified"]
    shifted = row["perturbed"]["before_settling"]["object_positions_m"]
    for name in (BOWL, PLATE, RAMEKIN):
        np.testing.assert_allclose(_position(env, name), original["object_positions_m"][name],
                                   atol=1e-8, rtol=0)
    joints = env.env.objects_dict[BOWL].joints
    if len(joints) != 1:
        raise RuntimeError("Source bowl must have exactly one free joint")
    joint = joints[0]
    start, end = env.sim.model.get_joint_qpos_addr(joint)
    qpos = np.asarray(env.sim.data.qpos)
    if end - start != 7 or not np.array_equal(state[1:1 + len(qpos)], qpos):
        raise RuntimeError("Unexpected MuJoCo state serialization for the source bowl")
    direction = np.asarray(original["plate_to_ramekin_axis_xy"])
    if direction.shape != (2,) or not np.isclose(np.linalg.norm(direction), 1):
        raise ValueError("Physics receipt lacks a unit shift direction")
    candidate = state.copy()
    candidate[1 + start:3 + start] += OFFSET_M * direction
    env.set_init_state(candidate)
    for name in (BOWL, PLATE, RAMEKIN):
        np.testing.assert_allclose(_position(env, name), shifted[name], atol=1e-8, rtol=0)
    if env.check_success():
        raise RuntimeError("The shifted state already satisfies the goal")
    return candidate


class _CandidateSuite:
    """Use alternate task-0 states in this benchmark instance only."""

    def __init__(self, suite, states):
        self.suite = suite
        self.states = states

    def get_task_init_states(self, task_id):
        return self.states[task_id]

    def __getattr__(self, name):
        return getattr(self.suite, name)


def _run(audit, run_dir, report, checkpoint):
    from libero.libero import get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    by_trial = {row["trial_id"]: row for row in audit["resets"]}
    plan = [
        row for row in build_trials(TASK_ID, 1, len(TRIAL_IDS), start_trial=TRIAL_IDS[0])
        if by_trial[row["trial_id"]]["passed_physics_screen"]
    ]
    report["physics_rejected"] = [
        {"trial_id": row["trial_id"], "reasons": row["rejection_reasons"]}
        for row in audit["resets"] if not row["passed_physics_screen"]
    ]
    report["physics_accepted_trial_ids"] = [row["trial_id"] for row in plan]
    if not plan:
        raise RuntimeError("No source reset passed the physics screen")
    model_name = "libero" if checkpoint is None else "sft-fewshot"
    model, report["checkpoint"] = _load_policy(model_name, checkpoint)
    env, states, offsets = _make_environment(plan)
    try:
        task = env.task_suite.get_task(TASK_ID)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
        if (task.name != TASK_NAME or sha256_file(bddl) != audit["bddl_sha256"]
                or sha256_file(init_file) != audit["bundled_init_sha256"]):
            raise ValueError("Source assets do not match the physics audit")

        if checkpoint is not None:
            report["standard_episodes"] = []
            with (run_dir / "source_reset_standard_episodes.jsonl").open("x", encoding="utf-8") as journal:
                for row in plan:
                    result = _run_episode(model, env, row, states, offsets)
                    if result["initial_state_sha256"] != by_trial[row["trial_id"]]["initial_state_sha256"]:
                        raise ValueError("Standard evaluation state differs from its physics receipt")
                    line = json.dumps(result)
                    journal.write(line + "\n")
                    journal.flush()
                    report["standard_episodes"].append(result)
                    print("XVLA_SOURCE_RESET_STANDARD_EPISODE " + line, flush=True)
        states[TASK_ID] = np.array(states[TASK_ID], copy=True)
        control = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
        try:
            for row in plan:
                trial = row["trial_id"]
                receipt = by_trial[trial]
                state = np.ascontiguousarray(states[TASK_ID][trial])
                if hashlib.sha256(state.tobytes()).hexdigest() != receipt["initial_state_sha256"]:
                    raise ValueError(f"Source initial state {trial} differs from its physics receipt")
                states[TASK_ID][trial] = _candidate_state(control, state, receipt)
        finally:
            control.close()
        env.task_suite = _CandidateSuite(env.task_suite, states)

        with (run_dir / "source_reset_solvability_episodes.jsonl").open("x", encoding="utf-8") as journal:
            for row in plan:
                result = _run_episode(model, env, row, states, offsets)
                result["standard_initial_state_sha256"] = by_trial[row["trial_id"]]["initial_state_sha256"]
                line = json.dumps(result)
                journal.write(line + "\n")
                journal.flush()
                report["episodes"].append(result)
                print("XVLA_SOURCE_RESET_SOLVABILITY_EPISODE " + line, flush=True)
    finally:
        env.env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Score an audited Pt-SFT candidate on the same source shifts")
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("XVLA_RUN_DIR"))
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error("Require --output-dir or the Slurm launcher run directory")
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] == "4090node1":
        raise RuntimeError("Solvability screening requires a project Slurm compute-node job")
    if os.environ.get("LIBERO_TYPE", "standard").lower() != "standard":
        raise RuntimeError("Solvability screening requires standard LIBERO")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not mask or mask != os.environ.get("RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Solvability screening requires the unchanged Slurm visibility mask")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("Solvability screening requires the launcher EGL backend")

    from .checkpoint_check import assert_context
    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

    runtime = assert_context()
    egl_id = NvidiaGPUManager.get_egl_device_id(0)
    if egl_id is None:
        raise RuntimeError("Cannot verify an EGL device for Slurm CUDA ordinal 0")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["EGL_DEVICE_ID"] = str(egl_id)
    audit = _read_audit(args.audit)
    output = args.output_dir / "source_reset_solvability.json"
    if output.exists():
        raise FileExistsError(output)
    report = {"schema_version": 1, "suite": SUITE, "task_id": TASK_ID, "along_offset_m": OFFSET_M,
              "audit_path": str(args.audit.resolve()), "audit_sha256": sha256_file(args.audit),
              "runtime": runtime, "egl_device_id": egl_id, "episodes": [],
              "protocol": {"sampler": "native_eval", "history": "native_history", "precision": "fp32",
                           "domain_id": DOMAIN, "chunk": CHUNK, "solver_steps": SOLVER_STEPS,
                           "horizon": HORIZON, "settling_steps": SETTLING}}
    try:
        _run(audit, args.output_dir, report, args.checkpoint)
        report["solvable_trial_ids"] = [row["trial_id"] for row in report["episodes"] if row["success"]]
        report["policy_timeouts"] = [row["trial_id"] for row in report["episodes"] if not row["success"]]
        report["policy_env_steps"] = sum(row["policy_env_steps"] for row in report["episodes"])
        report["settling_env_steps"] = sum(row["settling_env_steps"] for row in report["episodes"])
        report["status"] = "PASS" if args.checkpoint is not None or report["solvable_trial_ids"] else "FAIL"
    except Exception as error:
        report.update(status="FAIL", error_type=type(error).__name__, error=str(error))
        _write_summary(output, report)
        raise
    _write_summary(output, report)
    if report["status"] == "FAIL":
        raise RuntimeError("No candidate was solved by the fixed reference policy")
    print(f"XVLA_SOURCE_RESET_SOLVABILITY {output}", flush=True)


if __name__ == "__main__":
    main()
