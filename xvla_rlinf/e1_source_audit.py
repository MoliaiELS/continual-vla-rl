"""Audit frozen Spatial task-0 reset overlap using CPU simulation and no policy."""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations, product
import json
import os
from pathlib import Path
import socket

import numpy as np

from .sft import load_manifest, sha256_file
from .source_reset_audit import BOWL, PLATE, RAMEKIN, SETTLING_STEPS, SUITE, TASK_ID, TASK_NAME, _run_reset


PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "configs" / "xvla" / "protocol_20261110"
DEMO_ROLE = "source_sft_raw_demos"
FINAL_ROLE = "source_final_test_sealed"
TOLERANCE_M = 0.001
PHASES = ("before_settling", "after_settling")
GEOMETRY_FIELDS = ("bowl_along_m", "bowl_along_fraction", "bowl_cross_m",
                   "bowl_to_plate_xy_m", "bowl_to_ramekin_xy_m")


def _state_row(state):
    row = np.ascontiguousarray(state, dtype="<f8")
    if row.shape != (92,) or not np.isfinite(row).all():
        raise ValueError("Expected a finite, 92-value simulator state")
    return row, hashlib.sha256(row.tobytes()).hexdigest()


def _load_demos(path, common_start):
    import h5py

    if sha256_file(path) != common_start["manifest_sha256"]:
        raise ValueError("SFT manifest differs from the frozen 19525 manifest")
    manifest = load_manifest(path, verify_files=False)
    demo_ids = [f"demo_{value}" for value in common_start["demo_ids"]]
    actual = [(row["task_suite"], row["task_id"], row["demo_id"]) for row in manifest["episodes"]]
    if actual != [(SUITE, TASK_ID, demo_id) for demo_id in demo_ids]:
        raise ValueError("SFT manifest differs from the frozen demonstration order or task")
    raw = Path(manifest["source_hdf5"])
    if (manifest["data_revision"] != common_start["raw_demo_data_revision"]
            or manifest["source_sha256"] != common_start["raw_demo_file_sha256"]
            or sha256_file(raw) != manifest["source_sha256"]):
        raise ValueError("Raw demonstration asset differs from the frozen source")
    with h5py.File(raw, "r") as handle:
        return {demo_id: _state_row(handle["data"][demo_id]["states"][0]) for demo_id in demo_ids}


def _standard_roles(task_split):
    roles = {}
    for record in task_split["trial_roles"]:
        first, last = record["range_inclusive"]
        if record["role"] in roles:
            raise ValueError("Repeated source role in split")
        roles[record["role"]] = list(range(first, last + 1))
    if roles != {"source_dev": list(range(30)), "source_train_standard": list(range(30, 40)),
                 FINAL_ROLE: list(range(40, 50))}:
        raise ValueError("Unexpected source role assignment")
    return roles


def _distance(left, right):
    left = left["object_positions_m"]
    right = right["object_positions_m"]
    if left.keys() != right.keys():
        raise ValueError("Physical layouts have different tracked object names")
    distances = {name: float(np.linalg.norm(np.subtract(left[name], right[name]))) for name in left}
    return {"all_objects": max(distances.values()),
            "bowl_plate_ramekin": max(distances[name] for name in (BOWL, PLATE, RAMEKIN))}


def _distance_summary(pairs):
    minimum, ids = min(pairs, key=lambda pair: pair[0])
    return {"minimum_m": minimum, "minimum_pair": ids,
            "exact_center_pairs": [pair for distance, pair in pairs if distance == 0.0],
            "within_tolerance_pairs": [pair for distance, pair in pairs if distance <= TOLERANCE_M]}


def _compare_roles(left, right):
    exact_pairs = []
    distances = {phase: {group: [] for group in ("all_objects", "bowl_plate_ramekin")} for phase in PHASES}
    for (left_id, lhs), (right_id, rhs) in product(left.items(), right.items()):
        ids = [left_id, right_id]
        if lhs["state_sha256"] == rhs["state_sha256"]:
            exact_pairs.append(ids)
        for phase in PHASES:
            for group, distance in _distance(lhs[phase], rhs[phase]).items():
                distances[phase][group].append((distance, ids))
    return {
        "pair_count": len(left) * len(right),
        "exact_state_hash_pairs": exact_pairs,
        "center_distance": {
            phase: {group: _distance_summary(pairs) for group, pairs in groups.items()}
            for phase, groups in distances.items()
        },
    }


def _overlap(comparison):
    return bool(comparison["exact_state_hash_pairs"]) or any(
        result["within_tolerance_pairs"]
        for phase in comparison["center_distance"].values() for result in phase.values()
    )


def _geometry_summary(rows):
    result = {}
    for phase in PHASES:
        result[phase] = {}
        for field in GEOMETRY_FIELDS:
            values = [row[phase][field] for row in rows.values()]
            result[phase][field] = {"min": min(values), "median": float(np.median(values)), "max": max(values)}
    return result


def _measure(bddl, inputs):
    from libero.libero.envs.env_wrapper import ControlEnv

    rows = {}
    env = ControlEnv(bddl_file_name=str(bddl), use_camera_obs=False, has_offscreen_renderer=False)
    try:
        for role, states in inputs.items():
            rows[role] = {}
            for identity, (state, digest) in states.items():
                seed = 0 if role == DEMO_ROLE else 142 + identity
                reset = _run_reset(env, state, seed, offset=0.0)
                if reset["settling_env_steps"] != SETTLING_STEPS or reset["policy_env_steps"] != 0:
                    raise RuntimeError("Unexpected reset-only control-step accounting")
                rows[role][identity] = {**reset, "state_sha256": digest}
    finally:
        env.close()
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("XVLA_RUN_DIR"))
    args = parser.parse_args(argv)
    if args.output_dir is None:
        parser.error("Require --output-dir or XVLA_RUN_DIR")
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id or hostname.split(".")[0] == "4090node1" or os.environ.get("XVLA_ALLOCATED_GPUS") != "0":
        raise RuntimeError("E1 source audit requires a CPU-only Slurm compute-node allocation")
    if os.environ.get("LIBERO_TYPE", "standard").lower() != "standard":
        raise RuntimeError("E1 source audit requires standard LIBERO")
    output = args.output_dir / "e1_source_audit.json"
    if output.exists():
        raise FileExistsError(output)

    registry = json.loads((PROTOCOL_DIR / "registry.json").read_text(encoding="utf-8"))
    split_sha256 = sha256_file(args.split)
    if split_sha256 != registry["split_sha256"]:
        raise ValueError("Split differs from the frozen protocol registry")
    split = json.loads(args.split.read_text(encoding="utf-8"))
    task_split = next(row for row in split["tasks"] if row["task_id"] == TASK_ID)
    roles = _standard_roles(task_split)

    from .sft_entry import assert_pinned_runtime

    assert_pinned_runtime()
    from libero.libero import benchmark, get_libero_path

    suite = benchmark.get_benchmark_dict()[SUITE](task_order_index=split["task_order_index"])
    task = suite.get_task(TASK_ID)
    if (split["suite"], split["source_A"], task_split["task_name"], task.name) != (SUITE, TASK_ID, TASK_NAME, TASK_NAME):
        raise ValueError("Source task differs from the frozen split")
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    init = Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    bddl_hash = sha256_file(bddl)
    init_hash = sha256_file(init)
    if bddl_hash != task_split["bddl_sha256"] or init_hash != task_split["init_archive_sha256"]:
        raise ValueError("Runtime source BDDL or init archive differs from the frozen split")
    states = [_state_row(state) for state in suite.get_task_init_states(TASK_ID)]
    if len(states) != 50 or [digest for _, digest in states] != task_split["state_sha256_by_trial_id"]:
        raise ValueError("Standard state hashes differ from the frozen split")

    common_start = registry["common_start"]
    inputs = {role: {trial_id: states[trial_id] for trial_id in ids} for role, ids in roles.items()}
    inputs[DEMO_ROLE] = _load_demos(args.manifest, common_start)
    rows = _measure(bddl, inputs)
    comparisons = [dict(roles=[left, right], **_compare_roles(rows[left], rows[right]))
                   for left, right in combinations(rows, 2)]
    final_overlaps = [item["roles"] for item in comparisons if FINAL_ROLE in item["roles"] and _overlap(item)]
    premature = {role: [identity for identity, row in pool.items() if row["success_during_reset"]]
                 for role, pool in rows.items()}
    report = {
        "schema_version": 1,
        "status": "FAIL" if final_overlaps or any(premature.values()) else "PASS",
        "audit_scope": "libero_spatial_task0_source",
        "job_id": job_id,
        "hostname": hostname,
        "gpus": 0,
        "split_sha256": split_sha256,
        "demo_manifest_sha256": common_start["manifest_sha256"],
        "raw_demo_sha256": common_start["raw_demo_file_sha256"],
        "raw_demo_revision": common_start["raw_demo_data_revision"],
        "bddl_sha256": bddl_hash,
        "init_archive_sha256": init_hash,
        "state_hashes_match_split": True,
        "settling_steps_per_state": SETTLING_STEPS,
        "policy_env_steps": 0,
        "geometry_method": {
            "center_distance": "maximum Euclidean center displacement over named bodies in world coordinates",
            "tolerance_m_inclusive": TOLERANCE_M,
            "task_relevant_objects": [BOWL, PLATE, RAMEKIN],
            "scope": "Center proximity screens possible layout overlap. Rotations and trajectories are unmeasured.",
            "settling": "Same benchmark reset and zero-action settling for standard states and raw demo starts.",
            "demo_note": "Raw demo starts are measured independently of action-label conversion and expert replay.",
        },
        "role_sizes": {role: len(pool) for role, pool in rows.items()},
        "demo_initial_state_hashes": {identity: row["state_sha256"] for identity, row in rows[DEMO_ROLE].items()},
        "success_during_reset_ids": premature,
        "geometry_by_development_role": {role: _geometry_summary(pool) for role, pool in rows.items()
                                         if role != FINAL_ROLE},
        "cross_role_comparisons": comparisons,
        "sealed_role_overlaps": final_overlaps,
        "sealed_role_per_state_geometry_exported": False,
    }
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"XVLA_E1_SOURCE_AUDIT {report['status']} {output}", flush=True)
    if report["status"] != "PASS":
        raise RuntimeError("Sealed layout overlap or success before policy execution")


if __name__ == "__main__":
    main()
