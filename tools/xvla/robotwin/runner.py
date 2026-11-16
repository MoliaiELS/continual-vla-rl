"""Run one resumable RoboTwin task/condition shard with a frozen X-VLA."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
from dataclasses import asdict
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

from . import protocol as p


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CONDITIONS = {"easy": "demo_clean", "hard": "demo_randomized"}
SMOKE = ("move_stapler_pad", "turn_switch")
FILES = (*[f"task_config/{name}.yml" for name in CONDITIONS.values()],
         "task_config/_camera_config.yml", "task_config/_embodiment_config.yml",
         "task_config/_eval_step_limit.yml", "envs/camera/camera.py")
MAX_ATTEMPTS = 3
TERMINAL = {"success", "failure", "unresolved"}
VIDEO_PER_OUTCOME = 2


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def sha256(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def pinned_checkout(root, commit):
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError(f"Not the root of a checkout: {root}")
    if git(root, "rev-parse", "HEAD") != commit:
        raise ValueError(f"Expected {commit} in {root}")
    if git(root, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Tracked source differs from {commit}: {root}")


def emit(record):
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


def append(path, record):
    with path.open("ab") as stream:
        stream.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    emit(record)


def resume_journal(path):
    if not path.exists():
        return []
    with path.open("r+b") as stream:
        raw = stream.read()
        complete = raw.rfind(b"\n") + 1
        if complete < len(raw):
            # A killed process can leave its last attempt only partially written.
            archive = path.with_name(path.name + f".partial-{time.time_ns()}")
            archive.write_bytes(raw[complete:])
            stream.truncate(complete)
        return [json.loads(line) for line in raw[:complete].splitlines()]


@contextmanager
def lock_shard(shard):
    import fcntl

    shard.mkdir(parents=True, exist_ok=True)
    with (shard / ".lock").open("a") as fd:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def default_limit(catalog, task, phase):
    if phase == "smoke":
        return 5
    if phase == "coverage":
        return 1
    return 300 if task in catalog["priority_tasks"] else 100


def common_row(args, fingerprint):
    return dict(task=args.task, condition=args.condition, phase=args.phase,
                protocol=p.PROTOCOL, protocol_fingerprint=fingerprint)


def make_seed(row, index, seed, signature):
    scope = [row[key] for key in ("protocol", "phase", "task", "condition")]
    return dict(episode_id="/".join([*scope, f"{index + 1:04d}"]), index=index,
                requested_seed=seed, actual_seed=seed,
                policy_seed=int(digest([*scope, seed])[:15], 16), batch=index // 100 + 1,
                scene_signature=signature)


def validate_rows(rows, header):
    for row in rows:
        if any(row.get(key) != value for key, value in header.items()):
            raise ValueError("Existing journal belongs to a different task or protocol")


def load_native(root, task, condition):
    import yaml
    from .bind_renderer import bind_renderer

    os.chdir(root)
    sys.path.insert(0, str(root))
    bind_renderer()
    module = importlib.import_module(f"envs.{task}")
    if not Path(module.__file__).resolve().is_relative_to(root):
        raise ImportError(f"Unexpected RoboTwin module: {module.__file__}")
    name = CONDITIONS[condition]
    cfg = yaml.safe_load((root / "task_config" / f"{name}.yml").read_text())
    embodiment = cfg["embodiment"]
    if embodiment != ["aloha-agilex"]:
        raise ValueError(f"Unexpected embodiment {embodiment}")
    cameras = yaml.safe_load((root / "task_config/_camera_config.yml").read_text())
    camera = cameras[cfg["camera"]["head_camera_type"]]
    robots = yaml.safe_load((root / "task_config/_embodiment_config.yml").read_text())
    robo = robots[embodiment[0]]["file_path"]
    cfg.update(task_name=task, task_config=name, head_camera_h=camera["h"],
               head_camera_w=camera["w"], embodiment_name=embodiment[0],
               dual_arm_embodied=True, policy_name="V4", ckpt_setting="60k",
               eval_mode=True, render_freq=0)
    for side in ("left", "right"):
        cfg[f"{side}_robot_file"] = robo
        cfg[f"{side}_embodiment_config"] = yaml.safe_load((root / robo / "config.yml").read_text())
    return module, getattr(module, task), cfg, root / robo / "config.yml"


def init_success_fields(module, task, env):
    """Reproduce expert-only initial fields without moving any actor."""
    if task == "open_laptop":
        prod = module.get_face_prod(env.laptop.get_pose().q, [1, 0, 0], [1, 0, 0])
        env.arm_tag = module.ArmTag("left" if prod > 0 else "right")
    elif task in ("place_object_scale", "put_object_cabinet"):
        pos = env.object.get_pose().p
        env.arm_tag = module.ArmTag("right" if pos[0] > 0 else "left")
        if task == "put_object_cabinet":
            env.origin_z = pos[2]


class SceneCounter:
    """Count exact Python calls to native Scene.step without changing physics."""
    def __init__(self, scene, tick):
        self._scene, self._tick = scene, tick

    def __getattr__(self, name):
        return getattr(self._scene, name)

    def step(self):
        result = self._scene.step()
        self._tick()
        return result


@contextmanager
def count_actions(env, budget):
    # The policy loop passes no scene to pybind functions; only attribute calls
    # on Base_Task.scene are intercepted, with the original Scene restored after.
    scene = env.scene

    def tick():
        budget.physics_ticks += 1

    def planner(fn, side):
        def call(*args, **kwargs):
            result = fn(*args, **kwargs)
            if result["status"] != "Success":
                setattr(budget, side, getattr(budget, side) + 1)
            return result
        return call

    robot = env.robot
    old_left, old_right = robot.left_plan_path, robot.right_plan_path
    robot.left_plan_path = planner(old_left, "left_plan_failures")
    robot.right_plan_path = planner(old_right, "right_plan_failures")
    env.scene = SceneCounter(scene, tick)
    try:
        yield
    finally:
        env.scene = scene
        robot.left_plan_path, robot.right_plan_path = old_left, old_right


def scene_values(env):
    result = []
    for actor in [*env.scene.get_all_actors(), *env.scene.get_all_articulations()]:
        pose = actor.get_pose() if hasattr(actor, "get_pose") else actor.get_root_pose()
        qpos = list(actor.get_qpos()) if hasattr(actor, "get_qpos") else []
        q = list(pose.q)
        if max(q, key=abs) < 0:
            q = [-value for value in q]
        values = [round(float(x), 5) for x in (*pose.p, *q, *qpos)]
        result.append((actor.get_name(), *values))
    return sorted(result)


def same_scene(expected, observed, tolerance=1e-3):
    return len(expected) == len(observed) and all(
        a[0] == b[0] and len(a) == len(b)
        and max((abs(v - w) for v, w in zip(a[1:], b[1:])), default=0) <= tolerance
        for a, b in zip(expected, observed)
    )


def reset_env(env, cfg, index, seed, budget=None):
    from copy import deepcopy
    import random
    import numpy as np
    import torch

    env.suc, env.test_num = 0, 0
    start = time.perf_counter()
    # Pinned get_cluttered_table mutates its default lists in place. Use fresh
    # copies so the two tasks with nonzero table XY bias don't accumulate bias
    # across candidates, resets, or resumed shards.
    clutter = env.get_cluttered_table
    defaults = deepcopy(clutter.__func__.__defaults__)
    env.get_cluttered_table = lambda: clutter(*defaults)
    load_robot = env.load_robot

    def load_without_rng_side_effects(**kw):
        # The first reset warms up cuRobo before load_actors(). Preserve the
        # scene RNG stream also when the warm planner is reused after resume.
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices):
                load_robot(**kw)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)

    env.load_robot = load_without_rng_side_effects
    # check_stable does 2,000 + 500 direct Scene.step calls in the pinned code.
    # This intentionally isolates its settling from physics during rollout.
    if budget is not None:
        original = env.check_stable

        def check_stable():
            scene = env.scene

            def tick():
                budget.settling_ticks += 1

            env.scene = SceneCounter(scene, tick)
            try:
                return original()
            finally:
                env.scene = scene
        env.check_stable = check_stable

    try:
        env.setup_demo(now_ep_num=index, seed=seed, is_test=True, **cfg)
        if env.robot.communication_flag:
            raise RuntimeError("Pinned ALOHA config unexpectedly selected subprocess planners")
        # Keep the native warmed planners across resets, while resetting their
        # random streams for reproducibility across shards and retries. RoboTwin
        # applies the same reset_seed=True operation to subprocess planners.
        for planner in (env.robot.left_planner, env.robot.right_planner):
            planner.motion_gen.reset(reset_seed=True)
            planner.motion_gen_batch.reset(reset_seed=True)
    finally:
        del env.get_cluttered_table
        del env.load_robot
        if budget is not None:
            del env.check_stable
            budget.reset_seconds = time.perf_counter() - start
    if budget is not None:
        budget.step_lim = env.step_lim
        budget.timestep_seconds = float(env.scene.get_timestep())


def screen(args, shard, header, cls, cfg):
    from envs.utils.create_actor import UnStableError

    path = shard / "prescreen.jsonl"
    manifest_path = shard / "seed_manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        validate_rows([manifest], header)
        if manifest["limit"] != args.limit:
            raise ValueError("Frozen manifest does not have the requested episode limit")
        emit({"event": "seed_manifest_already_frozen", "path": str(manifest_path)})
        return
    rows = resume_journal(path)
    validate_rows(rows, header)
    accepted = [row for row in rows if row["status"] == "accepted"]
    seen = {row["scene_signature"] for row in accepted}
    attempts = {}
    seed_start = {"formal": 2000, "coverage": 100_000_000, "smoke": 200_000_000}[args.phase]
    seed = seed_start
    for row in rows:
        attempts[row["requested_seed"]] = row["attempt"]
        if row["status"] in ("accepted", "excluded"):
            seed = row["requested_seed"] + 1
    scanned = 0
    env = cls()
    while len(accepted) < args.limit:
        if seed >= seed_start + 100_000:
            raise RuntimeError("Exhausted reserved candidate seed range")
        attempt = attempts.get(seed, 0) + 1
        row = {**header, "requested_seed": seed, "actual_seed": seed, "attempt": attempt}
        result = {"status": "technical_error"}
        started = time.perf_counter()
        try:
            reset_env(env, cfg, len(accepted), seed)
            scene = scene_values(env)
            signature = digest(scene)
            env.play_once()
            if not env.plan_success:
                result = {"status": "excluded", "reason": "expert_plan_failed"}
            elif not env.check_success():
                result = {"status": "excluded", "reason": "expert_success_false"}
            elif signature in seen:
                result = {"status": "excluded", "reason": "duplicate_initial_scene"}
            else:
                result = {"status": "accepted", "reason": "expert_success",
                          "scene_signature": signature, "scene": scene}
        except UnStableError as error:
            result = {"status": "excluded", "reason": "unstable_initial_scene", "error": str(error)}
        except Exception as error:
            result.update(reason=type(error).__name__, traceback=traceback.format_exc())
        finally:
            with suppress(Exception):
                env.close_env(clear_cache=(seed - seed_start + 1) % cfg["clear_cache_freq"] == 0)
        row.update(result, wall_seconds=time.perf_counter() - started)
        append(path, row)
        if row["status"] == "technical_error":
            env = cls()  # A failed reset can leave a partially initialized planner.
            attempts[seed] = attempt
            if attempt % MAX_ATTEMPTS == 0:
                raise RuntimeError(f"Prescreen blocked at seed {seed}; rerun resumes this same seed")
            continue
        if row["status"] == "accepted":
            accepted.append(row)
            seen.add(row["scene_signature"])
        seed += 1
        scanned += 1
        if args.screen_candidates is not None and scanned >= args.screen_candidates and len(accepted) < args.limit:
            emit({"event": "partial_prescreen", "accepted": len(accepted), "limit": args.limit})
            return
    episodes = []
    for index, row in enumerate(accepted):
        item = make_seed(header, index, row["actual_seed"], row["scene_signature"])
        episodes.append({**item, "scene": row["scene"]})
    manifest = {**header, "limit": args.limit, "candidate_start": seed_start, "episodes": episodes}
    write_json(manifest_path, manifest)
    emit({"event": "seed_manifest_frozen", "path": str(manifest_path), "count": len(episodes)})


def checkpoint(path, device, full_verify):
    import torch
    from xvla_rlinf.vendor.xvla.modeling_xvla import XVLA
    from xvla_rlinf.vendor.xvla.processing_xvla import XVLAProcessor

    receipt = read_json(path / "verified.json")
    if (receipt["repo_id"], receipt["revision"]) != (p.MODEL_ID, p.MODEL_REVISION):
        raise ValueError(f"Checkpoint must be {p.MODEL_ID}@{p.MODEL_REVISION}")
    names = {record["file"] for record in receipt["files"]}
    if "config.json" not in names or not any(s.endswith(".safetensors") for s in names):
        raise ValueError("Receipt does not cover configuration and safetensors")
    for record in receipt["files"]:
        file = path / record["file"]
        if (not file.is_file() or file.stat().st_size != record["bytes"]
                or ((full_verify or file.suffix != ".safetensors") and sha256(file) != record["sha256"])):
            raise ValueError(f"Checkpoint file differs from verified receipt: {file}")
    config = read_json(path / "config.json")
    expected = {"action_mode": "ee6d", "use_proprio": True, "num_actions": p.ANCHORS}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Unexpected checkpoint configuration; expected {expected}")
    torch.set_default_dtype(torch.float32)
    processor = XVLAProcessor.from_pretrained(str(path), local_files_only=True)
    model = XVLA.from_pretrained(str(path), trust_remote_code=False, local_files_only=True,
                                 torch_dtype=torch.float32).to(device).float().eval().requires_grad_(False)
    if model.action_space.dim_action != 20 or processor.num_views != 3:
        raise ValueError("Unexpected X-VLA input/output dimensions")
    return p.NativePolicy(model, processor, device), receipt


def package_versions():
    versions = {}
    for name in ("torch", "torchvision", "sapien", "curobo", "nvidia-curobo", "mplib", "toppra",
                 "numpy", "scipy", "transformers", "Pillow", "imageio", "gymnasium"):
        with suppress(importlib.metadata.PackageNotFoundError):
            versions[name] = importlib.metadata.version(name)
    return versions


def save_video(shard, episode, status, frames):
    import imageio.v2 as imageio

    panel = shard / "videos"
    panel.mkdir(exist_ok=True)
    name = f"{episode['index'] + 1:04d}-{status}.mp4"
    temp = panel / f".{name}"
    imageio.mimsave(temp, frames, fps=30, macro_block_size=1)
    temp.replace(panel / name)
    return f"videos/{name}"


def run_eval(args, shard, header, module, cls, cfg, policy, seeds, task):
    path = shard / "episodes.jsonl"
    rows = resume_journal(path)
    validate_rows(rows, header)
    latest = {}
    for row in rows:
        key = row["episode_id"]
        if key in latest and latest[key]["status"] in TERMINAL:
            raise ValueError(f"Multiple terminal outcomes for {key}")
        latest[key] = row
    counts = {status: sum(row["status"] == status for row in latest.values()) for status in TERMINAL}
    unresolved = 0
    env = cls()
    for seed in seeds:
        if args.batch is not None and seed["batch"] != args.batch:
            continue
        key = seed["episode_id"]
        prior = latest.get(key, {})
        if prior.get("status") in TERMINAL:
            unresolved += int(prior["status"] == "unresolved")
            continue
        for attempt in range(prior.get("attempt", 0) + 1, MAX_ATTEMPTS + 1):
            start = time.perf_counter()
            budget = p.Budget()
            frames = []
            record = {**header, **{k: v for k, v in seed.items() if k != "scene"},
                      "attempt": attempt, "instruction": task["prompt"],
                      "reset_ok": False, "observations_ok": False}
            try:
                reset_env(env, cfg, seed["index"], seed["actual_seed"], budget)
                if budget.step_lim != task["step_limit"]:
                    raise ValueError(f"Unexpected episode step limit {budget.step_lim}")
                init_success_fields(module, args.task, env)
                measured = scene_values(env)
                if not same_scene(seed["scene"], measured):
                    raise RuntimeError("Seed reset differs from frozen prescreen scene by > 1 mm/rad-component")
                record["reset_ok"] = True
                policy.begin(task["prompt"], seed["policy_seed"])

                def step(obs):
                    values = policy.step(obs)
                    record["observations_ok"] = True
                    return values

                from types import SimpleNamespace
                collect = min(counts["success"], counts["failure"]) < VIDEO_PER_OUTCOME
                with count_actions(env, budget):
                    status, reason = p.rollout(env, SimpleNamespace(step=step), budget,
                                               (lambda image: frames.append(image.copy())) if collect else None)
                record.update(status=status, reason=reason, official_success_evaluated=budget.anchors > 0)
            except Exception as error:
                record.update(status="technical_error" if attempt < MAX_ATTEMPTS else "unresolved",
                              reason=type(error).__name__, traceback=traceback.format_exc())
            finally:
                with suppress(Exception):
                    env.close_env(clear_cache=(seed["index"] + 2) % cfg["clear_cache_freq"] == 0)
            status = record["status"]
            if status in {"technical_error", "unresolved"}:
                env = cls()
            record.update(asdict(budget), wall_seconds=time.perf_counter() - start,
                          simulated_seconds=budget.physics_ticks * budget.timestep_seconds
                          if budget.timestep_seconds else None)
            if status in {"success", "failure"} and frames and counts[status] < VIDEO_PER_OUTCOME:
                try:
                    record["video"] = save_video(shard, seed, status, frames)
                except Exception as error:
                    record["video_error"] = str(error)
            append(path, record)
            if status in TERMINAL:
                counts[status] += 1
                unresolved += int(status == "unresolved")
                break
    emit({"event": "shard_eval_finished", **header, **counts})
    if unresolved:
        raise RuntimeError(f"Requested batch contains {unresolved} unresolved episodes")


def already_evaluated(args, shard, seeds, header):
    rows = resume_journal(shard / "episodes.jsonl")
    validate_rows(rows, header)
    seen = {row["episode_id"]: row["status"] for row in rows}
    requested = [seed for seed in seeds if args.batch is None or seed["batch"] == args.batch]
    if not requested:
        raise ValueError("Requested batch is not part of the frozen manifest")
    if all(seen.get(seed["episode_id"]) in TERMINAL for seed in requested):
        failures = sum(seen[seed["episode_id"]] == "unresolved" for seed in requested)
        if failures:
            raise RuntimeError(f"Requested batch contains {failures} unresolved episodes")
        emit({"event": "batch_already_evaluated", **header, "batch": args.batch, "count": len(requested)})
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "eval", "parity"))
    parser.add_argument("--robotwin-root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--xvla-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--phase", required=True, choices=("smoke", "coverage", "formal"))
    parser.add_argument("--condition", required=True, choices=tuple(CONDITIONS))
    parser.add_argument("--task", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch", type=int, choices=(1, 2, 3))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--verify-weights", action="store_true")
    parser.add_argument("--screen-candidates", type=int,
                        help="Stop prescreen after this many candidate seeds, without freezing a partial panel")
    args = parser.parse_args()
    for attr in ("checkpoint", "output", "robotwin_root", "xvla_root"):
        if getattr(args, attr) is not None:
            setattr(args, attr, getattr(args, attr).resolve())
    catalog = read_json(HERE / "tasks.json")
    if catalog["robotwin_commit"] != p.ROBOTWIN_COMMIT or len(catalog["tasks"]) != 50:
        raise ValueError("Unexpected task catalog")
    tasks = {item["task_id"]: item for item in catalog["tasks"]}
    if args.task not in tasks:
        parser.error("Not in official task panel: " + args.task)
    task = tasks[args.task]
    if task["prompt"] != args.task.replace("_", " "):
        raise ValueError("Catalog prompt differs from native task-name language")
    prescribed = default_limit(catalog, args.task, args.phase)
    args.limit = prescribed if args.limit is None else args.limit
    if args.phase == "smoke" and not 5 <= args.limit <= 10:
        parser.error("Smoke limit must be between 5 and 10")
    if args.phase == "coverage" and args.limit != 1:
        parser.error("Coverage probe limit is one episode")
    if args.phase == "formal" and args.limit != prescribed:
        parser.error(f"Formal limit is fixed at {prescribed} for this task")
    if args.stage == "parity" and (args.phase != "smoke" or args.xvla_root is None):
        parser.error("Parity requires --phase smoke and a pinned --xvla-root")
    if args.stage != "screen" and args.checkpoint is None:
        parser.error("This stage needs --checkpoint")

    pinned_checkout(args.robotwin_root, p.ROBOTWIN_COMMIT)
    if args.xvla_root:
        pinned_checkout(args.xvla_root, p.XVLA_COMMIT)
    config_hashes = {name: sha256(args.robotwin_root / name) for name in FILES}
    files = ("runner.py", "protocol.py", "parity.py", "bind_renderer.py", "runtime.env", "requirements.txt")
    source_hashes = {name: sha256(HERE / name) for name in files}
    basis = dict(robotwin_commit=p.ROBOTWIN_COMMIT, xvla_commit=p.XVLA_COMMIT, rlinf_commit=p.RLINF_COMMIT,
                 model_id=p.MODEL_ID, model_revision=p.MODEL_REVISION, config_sha256=config_hashes,
                 evaluation_source_sha256=source_hashes)
    fingerprint = digest(basis)
    header = common_row(args, fingerprint)
    shard = args.output / args.phase / args.task / args.condition
    with lock_shard(shard):
        module, cls, cfg, robot_cfg = load_native(args.robotwin_root, args.task, args.condition)
        if args.stage == "screen":
            screen(args, shard, header, cls, cfg)
            return
        panel = read_json(shard / "seed_manifest.json")
        validate_rows([panel], header)
        if panel["limit"] != args.limit or len(panel["episodes"]) != args.limit:
            raise ValueError("No complete frozen seed manifest for the requested panel")
        if args.stage == "eval" and already_evaluated(args, shard, panel["episodes"], header):
            return
        import torch
        policy, receipt = checkpoint(args.checkpoint, args.device, args.verify_weights)
        runtime = {**header, **basis, "slurm_job": os.environ.get("SLURM_JOB_ID"),
                   "hostname": platform.node(), "python": sys.version, "python_executable": sys.executable,
                   "versions": package_versions(), "cuda": torch.version.cuda,
                   "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                   "robotwin_root": str(args.robotwin_root), "embodiment_config": str(robot_cfg),
                   "embodiment_config_sha256": sha256(robot_cfg),
                   "worktree_commit": git(REPO, "rev-parse", "HEAD"),
                   "checkpoint": str(args.checkpoint), "checkpoint_receipt": receipt,
                   "camera_order": p.CAMERAS, "rgb_shape_hwc": p.RGB_SHAPE,
                   "camera_near": 0.1, "camera_far": 100, "crop": None,
                   "domain_id": p.DOMAIN, "denoising_steps": p.DENOISE_STEPS,
                   "max_queries": p.QUERIES, "anchors_per_query": p.ANCHORS,
                   "step_lim": task["step_limit"], "instruction": task["prompt"],
                   "seed_manifest_sha256": sha256(shard / "seed_manifest.json")}
        write_json(shard / "runtime_manifest.json", runtime)
        audit_dir = shard / "runtime"
        audit_dir.mkdir(exist_ok=True)
        write_json(audit_dir / f"{args.stage}-{runtime['slurm_job'] or 'local'}-{time.time_ns()}.json", runtime)
        if args.stage == "parity":
            from .parity import run_parity
            run_parity(args, shard, header, module, cls, cfg, policy, panel["episodes"][0], task)
        else:
            run_eval(args, shard, header, module, cls, cfg, policy, panel["episodes"], task)


if __name__ == "__main__":
    main()
