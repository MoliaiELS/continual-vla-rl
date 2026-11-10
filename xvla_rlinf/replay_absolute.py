"""Replay converted LIBERO demonstration targets with native absolute control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import numpy as np

from .sft import _default_env, load_manifest, sha256_file
from .transforms import env_actions_from_model


SETTLING_STEPS = 10


def _replay(env_factory, episode, initial_state, actions, *, use_delta):
    env, _ = env_factory(episode["task_suite"], episode["task_id"], 256)
    try:
        env.seed(0)
        env.reset()
        env.set_init_state(initial_state)
        for robot in env.env.robots:
            robot.controller.use_delta = True
        for _ in range(SETTLING_STEPS):
            env.step(np.array([0., 0., 0., 0., 0., 0., -1.]))
        for robot in env.env.robots:
            robot.controller.use_delta = use_delta
        success_steps = []
        for step, action in enumerate(actions, 1):
            _, _, done, _ = env.step(action)
            if done:
                success_steps.append(step)
        first_success = success_steps[0] if success_steps else None
        return {"success_ever": bool(success_steps),
                "success_final": bool(success_steps and success_steps[-1] == len(actions)),
                "first_success_step": first_success, "successful_steps": len(success_steps),
                "action_env_steps": len(actions), "settling_env_steps": SETTLING_STEPS,
                "post_done_action_steps": 0 if first_success is None else len(actions) - first_success,
                "control_freq": env.env.control_freq}
    finally:
        env.close()


def replay_manifest(path, *, limit=1, env_factory=_default_env, rotation_backend="robosuite"):
    import h5py

    manifest = load_manifest(path)
    if not 1 <= limit <= manifest["demonstrations"]:
        raise ValueError("Replay limit must be within the manifest's independent demonstration count")
    source = Path(manifest["source_hdf5"])
    if sha256_file(source) != manifest["source_sha256"]:
        raise ValueError("Raw demonstration file differs from the conversion manifest")
    report = {"schema_version": 1, "job_id": os.environ.get("SLURM_JOB_ID"),
              "manifest": manifest["manifest_path"], "manifest_sha256": manifest["manifest_sha256"],
              "source_sha256": manifest["source_sha256"], "data_revision": manifest["data_revision"],
              "protocol": "paired_full_trajectory_delta_and_native_absolute_at_simulator_rate",
              "environment_seed": 0, "episodes": []}
    with h5py.File(source, "r") as raw:
        for episode in manifest["episodes"][:limit]:
            demo = raw["data"][episode["demo_id"]]
            initial_state = demo["states"][0]
            relative = np.asarray(demo["actions"])
            with h5py.File(episode["resolved_path"], "r") as converted:
                labels = np.asarray(converted["abs_action_6d"], dtype=np.float32)
            frames = episode["frames"]
            if (labels.shape != (frames, 10) or relative.shape != (frames, 7)
                    or not np.isfinite(labels).all() or not np.isfinite(relative).all()):
                raise ValueError(f"Invalid paired demonstration actions: {episode['demo_id']}")
            # Stored labels use signed gripper commands. SFT turns them into
            # binary targets before the native probability-to-command mapping.
            labels[:, 9] = labels[:, 9] > 0
            absolute = env_actions_from_model(labels, rotation_backend=rotation_backend)
            result = {key: episode[key] for key in ("task_suite", "task_id", "demo_id", "frames")}
            result.update(converted_sha256=episode["sha256"], initial_state_sha256=hashlib.sha256(
                np.ascontiguousarray(initial_state).tobytes()).hexdigest(),
                gripper_commands_changed=int(np.count_nonzero(absolute[:, 6] != relative[:, 6])))
            for name, actions, delta in (("relative", relative, True), ("absolute", absolute, False)):
                result[name] = _replay(env_factory, episode, initial_state, actions, use_delta=delta)
                if result[name]["control_freq"] != episode["control_freq"]:
                    raise RuntimeError("Replay control frequency differs from the conversion manifest")
            report["episodes"].append(result)
            print("XVLA_ABSOLUTE_REPLAY " + json.dumps(result), flush=True)
    report["status"] = ("PASS" if all(result[mode]["success_final"] for result in report["episodes"]
                                     for mode in ("relative", "absolute")) else "FAIL")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", default=1, type=int)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or socket.gethostname().split(".")[0] == "4090node1":
        raise RuntimeError("Replay requires the project Slurm launcher on a compute node")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not mask or mask != os.environ.get("RLINF_XVLA_ALLOCATION_CUDA_VISIBLE_DEVICES"):
        raise RuntimeError("Replay requires the unchanged Slurm CUDA visibility mask")
    if os.environ.get("MUJOCO_GL") != "egl" or os.environ.get("PYOPENGL_PLATFORM") != "egl":
        raise RuntimeError("Replay requires the launcher EGL backend")

    from .checkpoint_check import assert_context
    from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager

    runtime = assert_context()
    egl_id = NvidiaGPUManager.get_egl_device_id(0)
    if egl_id is None:
        raise RuntimeError("Cannot verify an EGL device for Slurm CUDA ordinal 0")
    os.environ["MUJOCO_EGL_DEVICE_ID"] = os.environ["EGL_DEVICE_ID"] = str(egl_id)
    if args.output.exists():
        raise FileExistsError(args.output)
    report = replay_manifest(args.manifest, limit=args.limit)
    report.update(runtime=runtime, egl_device_id=egl_id)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    if report["status"] != "PASS":
        raise RuntimeError(f"Demonstration did not reproduce final success. See {args.output}")


if __name__ == "__main__":
    main()
