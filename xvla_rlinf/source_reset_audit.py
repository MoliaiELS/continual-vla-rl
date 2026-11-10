"""Measure and screen physical source-task resets without running a policy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from .sft import sha256_file


SUITE = "libero_spatial"
TASK_ID = 0
TASK_NAME = "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate"
BOWL = "akita_black_bowl_1"
PLATE = "plate_1"
RAMEKIN = "glazed_rim_porcelain_ramekin_1"
SETTLING_STEPS = 10
MAX_OFFSET_M = 0.02


def relative_geometry(bowl, plate, ramekin):
    """Describe bowl position along and across the plate-to-ramekin segment."""
    axis = ramekin[:2] - plate[:2]
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Reset has a degenerate plate-to-ramekin axis")
    direction = axis / length
    displacement = bowl[:2] - plate[:2]
    along = float(displacement @ direction)
    return {
        "plate_to_ramekin_axis_xy": direction.tolist(),
        "plate_to_ramekin_length_m": length,
        "bowl_along_m": along,
        "bowl_along_fraction": along / length,
        "bowl_cross_m": float(displacement @ np.array([-direction[1], direction[0]])),
        "bowl_to_plate_xy_m": float(np.linalg.norm(displacement)),
        "bowl_to_ramekin_xy_m": float(np.linalg.norm(bowl[:2] - ramekin[:2])),
    }


def _bowl_contacts(env):
    sim = env.sim
    body_ids = env.env.obj_body_id
    bodies = {idx: name for name, idx in body_ids.items()}

    def object_for_geom(geom_id):
        body = int(sim.model.geom_bodyid[geom_id])
        while body and body not in bodies:
            body = int(sim.model.body_parentid[body])
        return bodies.get(body)

    contacts = []
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        objects = (object_for_geom(contact.geom1), object_for_geom(contact.geom2))
        if objects.count(BOWL) == 1:
            other = contact.geom2 if objects[0] == BOWL else contact.geom1
            contacts.append({
                "other_geom": sim.model.geom_id2name(other),
                "other_object": objects[1] if objects[0] == BOWL else objects[0],
                "contact_distance_m": float(contact.dist),
            })
    return contacts


def _snapshot(env, bowl_joint):
    sim = env.sim
    positions = {name: np.asarray(sim.data.body_xpos[idx]).copy()
                 for name, idx in env.env.obj_body_id.items()}
    velocity = np.asarray(sim.data.get_joint_qvel(bowl_joint))
    if velocity.shape != (6,) or not all(
        np.isfinite(values).all() for values in (*positions.values(), velocity)
    ):
        raise ValueError("Non-finite object geometry or invalid bowl free-joint velocity")
    return {
        "object_positions_m": {name: pos.tolist() for name, pos in positions.items()},
        "bowl_linear_speed_m_per_s": float(np.linalg.norm(velocity[:3])),
        "bowl_angular_speed_rad_per_s": float(np.linalg.norm(velocity[3:])),
        "bowl_contacts": _bowl_contacts(env),
        "goal_satisfied": bool(env.check_success()),
        **relative_geometry(positions[BOWL], positions[PLATE], positions[RAMEKIN]),
    }


def _run_reset(env, state, seed, offset):
    env.seed(seed)
    env.reset()
    env.set_init_state(state)
    for robot in env.robots:
        robot.controller.use_delta = True
    joints = env.env.objects_dict[BOWL].joints
    if len(joints) != 1:
        raise RuntimeError("Source bowl must have exactly one free joint")
    bowl_joint = joints[0]
    initial = _snapshot(env, bowl_joint)
    before = initial
    if offset:
        qpos = np.asarray(env.sim.data.get_joint_qpos(bowl_joint)).copy()
        if qpos.shape != (7,) or not np.isfinite(qpos).all():
            raise RuntimeError("Unexpected source-bowl free-joint state")
        qpos[:2] += offset * np.asarray(initial["plate_to_ramekin_axis_xy"])
        env.sim.data.set_joint_qpos(bowl_joint, qpos)
        env.sim.forward()
        before = _snapshot(env, bowl_joint)
    early_success = before["goal_satisfied"]
    for _ in range(SETTLING_STEPS):
        # Matches the patched native benchmark's reset_gripper_open=False.
        _, _, done, _ = env.step(np.zeros(7))
        early_success = early_success or bool(done)
    after = _snapshot(env, bowl_joint)
    delta = np.asarray(after["object_positions_m"][BOWL]) - before["object_positions_m"][BOWL]
    return {
        "unmodified": initial,
        "before_settling": before,
        "after_settling": after,
        "bowl_settling_delta_xyz_m": delta.tolist(),
        "settling_env_steps": SETTLING_STEPS,
        "policy_env_steps": 0,
        "success_during_reset": early_success or after["goal_satisfied"],
    }


def _screen_perturbation(baseline, perturbed, offset):
    before = perturbed["before_settling"]
    after = perturbed["after_settling"]
    baseline_after = baseline["after_settling"]
    delta = np.asarray(perturbed["bowl_settling_delta_xyz_m"])
    errors = []
    if baseline["unmodified"] != perturbed["unmodified"]:
        errors.append("paired_pre_shift_reset_differs")
    if baseline["success_during_reset"] or perturbed["success_during_reset"]:
        errors.append("success_before_policy")
    if not all(0 < snapshot["bowl_along_fraction"] < 1 for snapshot in (before, after)):
        errors.append("bowl_outside_plate_to_ramekin_segment")
    if any(contact["other_object"] for snapshot in (before, after)
           for contact in snapshot["bowl_contacts"]):
        errors.append("bowl_contact_with_tracked_object")
    expected_contacts = {contact["other_geom"] for snapshot in
                         (baseline["before_settling"], baseline_after)
                         for contact in snapshot["bowl_contacts"]}
    if any(contact["other_geom"] not in expected_contacts for snapshot in (before, after)
           for contact in snapshot["bowl_contacts"]):
        errors.append("new_bowl_contact_geometry")
    if np.linalg.norm(delta[:2]) > 0.006 or abs(delta[2]) > 0.02:
        errors.append("bowl_drift_during_settling")
    if abs(after["object_positions_m"][BOWL][2] - baseline_after["object_positions_m"][BOWL][2]) > 0.01:
        errors.append("different_settled_bowl_height")
    if after["bowl_linear_speed_m_per_s"] > 0.05:
        errors.append("bowl_linear_speed_after_settling")
    if abs((after["bowl_along_m"] - baseline_after["bowl_along_m"]) - offset) > 0.006:
        errors.append("requested_shift_not_retained")
    return errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-trial", type=int, default=0)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("XVLA_RUN_DIR"))
    parser.add_argument("--along-offset-m", type=float, default=0.0,
                        help="Optionally add a paired shift toward the ramekin, in metres")
    args = parser.parse_args(argv)
    if args.output_dir is None or args.start_trial < 0 or not 1 <= args.trials <= 50:
        parser.error("Require a run directory, nonnegative starting index and 1–50 trials")
    if not -MAX_OFFSET_M <= args.along_offset_m <= MAX_OFFSET_M:
        parser.error("Candidate offset must be finite and at most 2 cm in magnitude")
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] == "4090node1":
        raise RuntimeError("Source reset audit requires a Slurm compute-node allocation")
    if os.environ.get("XVLA_ALLOCATED_GPUS") != "0":
        raise RuntimeError("The unrendered source reset audit requires a CPU-only allocation")
    if os.environ.get("LIBERO_TYPE", "standard").lower() != "standard":
        raise RuntimeError("Source reset audit requires standard LIBERO")

    from .sft_entry import assert_pinned_runtime

    assert_pinned_runtime()
    runtime = {"job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(), "gpus": 0}
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(TASK_ID)
    if task.name != TASK_NAME:
        raise ValueError("Unexpected task ordering for Spatial task 0")
    states = suite.get_task_init_states(TASK_ID)
    end = args.start_trial + args.trials
    if end > len(states):
        parser.error(f"Requested trial range exceeds the {len(states)} bundled source resets")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init_file = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    output = args.output_dir / "source_reset_audit.json"
    if output.exists():
        raise FileExistsError(output)
    report = {"schema_version": 1, "suite": SUITE, "task_id": TASK_ID, "task_name": task.name,
              "language": task.language, "runtime": runtime, "rendered": False,
              "bddl_sha256": sha256_file(bddl), "bundled_init_sha256": sha256_file(init_file),
              "available_trials": len(states), "along_offset_m": args.along_offset_m, "resets": []}
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
    try:
        for trial in range(args.start_trial, end):
            seed = 142 + trial
            state = np.ascontiguousarray(states[trial])
            baseline = _run_reset(env, state, seed, offset=0.0)
            row = {"trial_id": trial, "environment_seed": seed, "standard": baseline,
                   "initial_state_sha256": hashlib.sha256(state.tobytes()).hexdigest()}
            if args.along_offset_m:
                perturbed = _run_reset(env, state, seed, offset=args.along_offset_m)
                reasons = _screen_perturbation(baseline, perturbed, args.along_offset_m)
                row.update(perturbed=perturbed, passed_physics_screen=not reasons, rejection_reasons=reasons)
            report["resets"].append(row)
    finally:
        env.close()
    report["standard_unexpected_successes"] = [
        row["trial_id"] for row in report["resets"] if row["standard"]["success_during_reset"]
    ]
    report["perturbations_passing_physics_screen"] = [
        row["trial_id"] for row in report["resets"] if row.get("passed_physics_screen")
    ]
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"XVLA_SOURCE_RESET_AUDIT {output}", flush=True)
    if report["standard_unexpected_successes"]:
        raise RuntimeError("Source reset reaches the goal before policy execution")


if __name__ == "__main__":
    main()
