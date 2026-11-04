"""Slurm-only simulation preflight for bundled camera and Object variants.

No policy model is loaded. Images are real LIBERO renders of the same saved
initial state and zero-action settling sequence. Camera configuration comes
directly from the bundled libero_spatial_cam2.yaml, without new perturbations.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import traceback


def require_slurm(environment=None, hostname=None):
    environment = os.environ if environment is None else environment
    hostname = socket.gethostname() if hostname is None else hostname
    if not str(environment.get("SLURM_JOB_ID", "")).isdigit():
        raise RuntimeError("OOD simulation preflight must run in an allocated Slurm job")
    if hostname.split(".")[0] == "4090node1":
        raise RuntimeError("OOD simulation preflight refuses the login node")
    if not environment.get("CUDA_VISIBLE_DEVICES", "").strip():
        raise RuntimeError("Slurm GPU visibility is missing")
    if environment.get("MUJOCO_GL") != "egl":
        raise RuntimeError("OOD preflight expects the project EGL configuration")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector_equal(left, right, atol=1e-10):
    """Small pose comparison remains testable without NumPy or simulator imports."""
    return len(left) == len(right) and all(abs(float(a) - float(b)) <= atol for a, b in zip(left, right))


def verify_camera_poses(requested, stages):
    for stage, poses in stages.items():
        for name, request in requested.items():
            if name not in poses:
                raise RuntimeError(f"Camera {name} is absent after {stage}")
            for field in ("pos", "quat"):
                if field in request and not _vector_equal(request[field], poses[name][field]):
                    raise RuntimeError(f"Camera {name} {field} override was not applied after {stage}")


def _camera_poses(env, names):
    result = {}
    for name in names:
        camera = env.sim.model.camera_name2id(name)
        result[name] = {"pos": env.sim.model.cam_pos[camera].tolist(),
                        "quat": env.sim.model.cam_quat[camera].tolist(),
                        "fovy": float(env.sim.model.cam_fovy[camera])}
    return result


def _array_signature(model):
    """Physical model arrays must stay unchanged by the camera-only variant."""
    names = ("body_pos", "body_quat", "body_mass", "body_inertia", "geom_pos",
             "geom_quat", "geom_size", "jnt_pos", "jnt_axis", "dof_damping",
             "actuator_ctrlrange")
    return {name: hashlib.sha256(getattr(model, name).tobytes()).hexdigest()
            for name in names if hasattr(model, name)}


def _render_condition(OffScreenRenderEnv, suite, task_id, init_id, kwargs, output, name):
    import numpy as np
    from PIL import Image

    state = suite.get_task_init_states(task_id)[init_id]
    env = OffScreenRenderEnv(**kwargs)
    try:
        env.seed(0)
        env.reset()
        poses = {"reset": _camera_poses(env, ["agentview"])}
        obs = env.set_init_state(state)
        poses["set_init_state"] = _camera_poses(env, ["agentview"])
        action_low, action_high = env.env.action_spec
        if np.asarray(action_low).shape != (7,) or np.asarray(action_high).shape != (7,):
            raise RuntimeError("Expected the unchanged seven-dimensional LIBERO action interface")
        for _ in range(10):
            obs, _reward, _done, _info = env.step(np.zeros(7))
        poses["after_settling"] = _camera_poses(env, ["agentview"])
        image = np.asarray(obs["agentview_image"])
        if image.shape != (256, 256, 3) or not np.isfinite(image).all() or float(image.std()) <= 1:
            raise RuntimeError(f"Invalid rendered image for {name}")
        image_path = output / f"{name}.png"
        # LIBERO's agentview array is vertically reversed for a human preview.
        # Hashes and statistics below refer to the original observation bytes.
        Image.fromarray(image[::-1]).save(image_path)
        report = {"task_id": task_id, "init_state_id": init_id, "seed": 0,
                  "settling_steps": 10, "camera_poses": poses,
                  "goal": bool(env.check_success()), "language": env.language_instruction,
                  "action_low": np.asarray(action_low).tolist(), "action_high": np.asarray(action_high).tolist(),
                  "physical_model_sha256": _array_signature(env.sim.model),
                  "image": {"path": str(image_path), "raw_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                            "png_sha256": sha256_file(image_path), "shape": list(image.shape),
                            "mean": float(image.mean()), "std": float(image.std()),
                            "min": int(image.min()), "max": int(image.max())}}
        return report, image.copy(), np.asarray(env.get_sim_state()).copy()
    finally:
        env.close()


def validate(output, report):
    require_slurm()
    import numpy as np
    import torch
    import yaml
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("OOD preflight requires exactly one allocated visible CUDA device")
    report["execution"] = {"slurm_job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
                           "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                           "gpu_name": torch.cuda.get_device_name(0), "mujoco_gl": os.environ["MUJOCO_GL"],
                           "mujoco_egl_device_id": os.environ.get("MUJOCO_EGL_DEVICE_ID")}
    repo = Path(__file__).resolve().parents[2]
    config_dir = repo / "examples/embodiment/config/env/eval"
    base_path, shifted_path = config_dir / "libero_spatial.yaml", config_dir / "libero_spatial_cam2.yaml"
    baseline_config = yaml.safe_load(base_path.read_text())
    variant_config = yaml.safe_load(shifted_path.read_text())
    # A camera-only check must keep the goal, reward, control horizon and robot unchanged.
    for field in ("simulator_type", "max_episode_steps", "use_rel_reward", "reward_coef", "num_images_in_input"):
        if baseline_config[field] != variant_config[field]:
            raise RuntimeError(f"Bundled camera variant unexpectedly changes {field}")
    variant_init = dict(variant_config["init_params"])
    overrides = variant_init.pop("camera_overrides_per_task")
    if variant_init != baseline_config["init_params"]:
        raise RuntimeError("Camera variant differs from baseline beyond the declared camera overrides")
    requested = overrides[0]
    report["configuration"] = {"baseline_path": str(base_path), "baseline_sha256": sha256_file(base_path),
                               "variant_path": str(shifted_path), "variant_sha256": sha256_file(shifted_path),
                               "requested_camera_overrides": requested,
                               "reward_and_horizon_unchanged": True}
    spatial = benchmark.get_benchmark_dict()["libero_spatial"](task_order_index=0)
    task = spatial.get_task(0)
    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    common = {**variant_init, "bddl_file_name": str(bddl)}
    baseline, base_image, base_state = _render_condition(
        OffScreenRenderEnv, spatial, 0, 40, common, output, "camera_baseline")
    report["camera_baseline"] = baseline
    shifted, shifted_image, shifted_state = _render_condition(
        OffScreenRenderEnv, spatial, 0, 40, {**common, "camera_overrides": requested}, output, "camera_shift")
    report["camera_shift"] = shifted
    verify_camera_poses(requested, shifted["camera_poses"])
    base_pose = baseline["camera_poses"]["after_settling"]["agentview"]
    shifted_pose = shifted["camera_poses"]["after_settling"]["agentview"]
    if _vector_equal(base_pose["pos"], shifted_pose["pos"]) and _vector_equal(base_pose["quat"], shifted_pose["quat"]):
        raise RuntimeError("Camera variant did not change the simulated camera pose")
    if baseline["physical_model_sha256"] != shifted["physical_model_sha256"]:
        raise RuntimeError("Camera-only variant unexpectedly changed physical model arrays")
    if not np.allclose(base_state, shifted_state, atol=1e-9, rtol=0):
        raise RuntimeError("Camera-only variant changed the physical simulation state")
    if (baseline["goal"] != shifted["goal"] or baseline["language"] != shifted["language"]
            or baseline["action_low"] != shifted["action_low"] or baseline["action_high"] != shifted["action_high"]):
        raise RuntimeError("Camera-only variant changed the task predicate, instruction or action interface")
    difference = np.abs(base_image.astype(np.float64) - shifted_image.astype(np.float64))
    changed = int(np.count_nonzero(np.any(base_image != shifted_image, axis=2)))
    if not changed or baseline["image"]["raw_sha256"] == shifted["image"]["raw_sha256"]:
        raise RuntimeError("Applied camera variant did not change the rendered observation")
    report["camera_comparison"] = {"verified": True, "bddl_path": str(bddl), "bddl_sha256": sha256_file(bddl),
                                   "physical_state_equal": True, "physical_model_equal": True,
                                   "changed_pixels": changed, "changed_pixel_fraction": changed / (256 * 256),
                                   "mean_absolute_pixel_difference": float(difference.mean()),
                                   "max_absolute_pixel_difference": float(difference.max())}

    objects = benchmark.get_benchmark_dict()["libero_object"](task_order_index=0)
    object_reports = []
    report["object_tasks"] = object_reports
    for task_id in range(5):
        object_task = objects.get_task(task_id)
        states = objects.get_task_init_states(task_id)
        state_arrays = [np.asarray(states[index]) for index in range(40, 50)]
        shapes = {tuple(state.shape) for state in state_arrays}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1 or not all(np.isfinite(state).all() for state in state_arrays):
            raise RuntimeError(f"Invalid Object task {task_id} initial-state vectors 40..49")
        object_bddl = Path(get_libero_path("bddl_files")) / object_task.problem_folder / object_task.bddl_file
        init_file = Path(get_libero_path("init_states")) / object_task.problem_folder / object_task.init_states_file
        if not object_bddl.is_file() or not init_file.is_file():
            raise RuntimeError(f"Object task {task_id} is missing required assets")
        condition, _image, physical_state = _render_condition(
            OffScreenRenderEnv, objects, task_id, 40,
            {**baseline_config["init_params"], "bddl_file_name": str(object_bddl)}, output, f"object_task_{task_id}")
        object_reports.append({"task_id": task_id, "language": object_task.language,
                               "bddl_path": str(object_bddl), "bddl_sha256": sha256_file(object_bddl),
                               "init_states_path": str(init_file), "init_states_sha256": sha256_file(init_file),
                               "available_init_states": len(states), "verified_init_ids": list(range(40, 50)),
                               "saved_state_shape": list(state_arrays[0].shape), "sim_state_shape": list(physical_state.shape),
                               "action_dim": len(condition["action_low"]), "action_low": condition["action_low"],
                               "action_high": condition["action_high"], "image": condition["image"],
                               "settling_steps": 10, "policy_model_loaded": False})

    stats_path = Path(os.environ["FYP_MODEL_PATH"]) / "dataset_statistics.json"
    statistics = json.loads(stats_path.read_text())
    key = "libero_spatial_no_noops"
    action = statistics[key]["action"]
    for name in ("q01", "q99"):
        values = np.asarray(action[name])
        if values.shape != (7,) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid fixed Spatial action statistics: {name}")
    report["action_statistics"] = {"path": str(stats_path), "sha256": sha256_file(stats_path),
                                   "unnorm_key": key, "available_keys": sorted(statistics),
                                   "action_dim": 7, "object_statistics_used": False,
                                   "provenance": "Fixed Spatial SFT checkpoint prior; no Object-specific statistics estimated"}
    report["preflight_cost"] = {"environments_instantiated": 7, "zero_action_settling_steps": 70,
                               "training_policy_steps": 0, "policy_model_loaded": False}
    report["interpretation"] = "Verifies simulation assets and camera shift; does not establish policy success or all-history unseen task exposure."
    report["verified"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Result directory under personal FYP storage")
    args = parser.parse_args()
    require_slurm()
    output = args.output.resolve()
    data_root = Path(os.environ["FYP_DATA_ROOT"]).resolve()
    if not output.is_relative_to(data_root):
        raise RuntimeError("Preflight output must remain in the personal FYP storage root")
    output.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "verified": False}
    try:
        validate(output, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        (output / "ood_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verified": report["verified"], "output": str(output)}))


if __name__ == "__main__":
    main()
