"""Diagnose the released X-VLA rollout with the native task action budget."""
from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


EVALUATION_COMMIT = "9baa17360abe242cb36e8f9c78f0aadbe64b3bee"
TASKS = ("open_microwave", "stack_bowls_three", "blocks_ranking_rgb", "blocks_ranking_size", "click_alarmclock")


def expanded_author_rollout(root, query_limit, official):
    """Verify the source with its existing loader, then replace one outer range literal."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    official.load_official_rollout(root)
    path = root / official.CLIENT_PATH
    syntax = ast.parse(path.read_bytes(), filename=str(path))
    definitions = [node for node in syntax.body if getattr(node, "name", None) in {"_rollout", "rotate6D_to_quat"}]
    rollout = next(node for node in definitions if node.name == "_rollout")
    loops = [node for node in rollout.body if isinstance(node, ast.For)]
    if (len(loops) != 1 or ast.dump(loops[0].iter, include_attributes=False)
            != ast.dump(ast.parse("range(10)", mode="eval").body, include_attributes=False)):
        raise ValueError("Pinned author rollout no longer has the expected outer range(10)")
    before = ast.dump(ast.Module(body=definitions, type_ignores=[]), include_attributes=False)
    loops[0].iter.args[0].value = query_limit
    tree = ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[]))
    after = ast.dump(tree, include_attributes=False)
    namespace = {"np": np, "R": Rotation, "tqdm": lambda iterable: iterable}
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace["_rollout"], {"original_ast_sha256": hashlib.sha256(before.encode()).hexdigest(),
                                    "diagnostic_ast_sha256": hashlib.sha256(after.encode()).hexdigest(),
                                    "change": {"outer_query_range": [10, query_limit]}}


def task_state(env, task):
    """Read native success operands without invoking the stateful predicate again."""
    import numpy as np

    positions = lambda names: {name: getattr(env, name).get_pose().p.tolist() for name in names}
    opened = [bool(env.is_left_gripper_open()), bool(env.is_right_gripper_open())]
    state = {"table_z_bias_m": float(env.table_z_bias), "grippers_open": opened,
             "tcp_pose": {side: getattr(env.robot, f"get_{side}_tcp_pose")() for side in ("left", "right")},
             "gripper_value": {side: float(getattr(env.robot, f"get_{side}_gripper_val")()) for side in ("left", "right")}}
    state["tcp_pose"] = {key: np.asarray(value).tolist() for key, value in state["tcp_pose"].items()}
    if task == "open_microwave":
        qpos, limits = env.microwave.get_qpos(), env.microwave.get_qlimits()
        state.update(model_id=int(env.model_id), qpos=np.asarray(qpos).tolist(), first_joint_limits=np.asarray(limits[0]).tolist(),
                     native_target_fraction=0.6, target_qpos=float(limits[0][1] * 0.6),
                     remaining_qpos=float(limits[0][1] * 0.6 - qpos[0]),
                     predicate_operands_pass=bool(qpos[0] >= limits[0][1] * 0.6))
    elif task == "stack_bowls_three":
        poses = positions(("bowl1", "bowl2", "bowl3"))
        ordered = np.array(sorted(poses.values(), key=lambda xyz: xyz[2]))
        heights = np.array([0.74, 0.77, 0.81]) + env.table_z_bias
        xy = np.abs(ordered[:-1, :2] - ordered[1:, :2])
        z = ordered[:, 2] - heights
        state.update(positions_m=poses, sorted_heights_m=ordered[:, 2].tolist(), target_heights_m=heights.tolist(),
                     adjacent_xy_abs_delta_m=xy.tolist(), upper_height_delta_m=z.tolist(),
                     native_xy_tolerance_m=0.04, native_upper_height_tolerance_m=0.02,
                     predicate_operands_pass=bool(np.all(xy < 0.04) and np.all(z < 0.02) and all(opened)))
    elif task in {"blocks_ranking_rgb", "blocks_ranking_size"}:
        poses = positions(("block1", "block2", "block3"))
        ordered = np.array(list(poses.values()))
        delta = np.abs(ordered[:-1, :2] - ordered[1:, :2])
        sorted_x = bool(np.all(np.diff(ordered[:, 0]) > 0))
        state.update(positions_m=poses, adjacent_xy_abs_delta_m=delta.tolist(), native_xy_tolerance_m=[0.13, 0.03],
                     increasing_x=sorted_x, predicate_operands_pass=bool(np.all(delta < [0.13, 0.03])
                                                                 and sorted_x and all(opened)))
    elif task == "click_alarmclock":
        target = np.asarray(env.alarm.get_contact_point(0)[:3])
        contacts = np.asarray(env.get_gripper_actor_contact_position("046_alarm-clock")).reshape(-1, 3)
        delta = contacts - target
        closed = bool(env.check_arm_function())
        near = bool(len(contacts) and np.any(np.all(np.abs(delta) < [0.03, 0.03, 0.03], axis=1)))
        state.update(model_id=int(env.alarmclock_id), alarm_pose=np.asarray(env.alarm.get_pose().p).tolist(),
                     target_contact_xyz_m=target.tolist(),
                     gripper_contacts_xyz_m=contacts.tolist(), contact_minus_target_xyz_m=delta.tolist(),
                     native_contact_tolerance_m=[0.03, 0.03, 0.03], required_arm_closed=closed,
                     stage_success_tag=bool(env.stage_success_tag),
                     target_minus_tcp_xyz_m={side: (target - np.asarray(pose[:3])).tolist()
                                             for side, pose in state["tcp_pose"].items()},
                     predicate_operands_pass=bool(env.stage_success_tag or (closed and near)))
    return state


@contextmanager
def instrument(env, budget, task, folder, wrist):
    import imageio.v2 as imageio
    import numpy as np

    original_action, original_obs, original_check = env.take_action, env.get_obs, env.check_success
    frame_number, pending, inside_action = 0, False, False
    last_check = None
    started = time.perf_counter()
    result = {"original_300_boundary": None, "video": {"fps": 30, "sampling": "one frame per advanced action anchor"}}
    with ExitStack() as stack:
        trace = stack.enter_context((folder / "trace.jsonl").open("w", encoding="utf-8"))
        trace.write(json.dumps({"event": "header", "label": "isolated native-budget diagnostic",
                                "counts_toward_formal": False, "task": task,
                                "frame_time": "physics_ticks * native timestep, video uses one frame per anchor at 30fps",
                                "tcp_frame": "native TCP; observation EE has a different link offset"}) + "\n")
        trace.flush()
        cameras = ("head_camera", "left_camera", "right_camera") if wrist else ("head_camera",)
        writers = {camera: stack.enter_context(imageio.get_writer(str(folder / f"{camera}-anchors-30fps.mp4"),
                                                                 fps=30, macro_block_size=1)) for camera in cameras}

        def snapshot():
            return {"budget": asdict(budget), "simulated_seconds": budget.physics_ticks * budget.timestep_seconds,
                    "elapsed_wall_seconds": time.perf_counter() - started, "native_last_check_success": last_check,
                    "native_eval_success": bool(env.eval_success), "task_state": task_state(env, task)}

        def action(command, action_type):
            nonlocal pending, inside_action
            before = env.take_action_cnt
            inside_action = True
            try:
                original_action(command, action_type=action_type)
            finally:
                inside_action = False
            budget.anchors = env.take_action_cnt
            pending = env.take_action_cnt > before
            if pending:
                row = {"event": "anchor", "command": command.tolist(), **snapshot()}
                trace.write(json.dumps(row, allow_nan=False) + "\n")
                trace.flush()
                if env.take_action_cnt == 300:
                    result["original_300_boundary"] = row

        def obs():
            nonlocal pending, frame_number
            observation = original_obs()
            if pending and not inside_action:
                for camera, writer in writers.items():
                    writer.append_data(observation["observation"][camera]["rgb"])
                trace.write(json.dumps({"event": "video_frame", "frame_index": frame_number,
                                        "anchor": env.take_action_cnt, "physics_ticks": budget.physics_ticks,
                                        "simulated_seconds": budget.physics_ticks * budget.timestep_seconds,
                                        "measured_endpose": {key: np.asarray(value).tolist()
                                                             for key, value in observation["endpose"].items()}}) + "\n")
                trace.flush()
                frame_number += 1
                pending = False
            return observation

        def check(*args, **kwargs):
            nonlocal last_check
            answer = original_check(*args, **kwargs)
            last_check = bool(answer)
            return answer

        env.take_action, env.get_obs, env.check_success = action, obs, check
        try:
            yield result
        finally:
            with suppress(Exception):
                result["terminal_snapshot"] = snapshot()
            result["video"]["frames"] = frame_number
            env.take_action, env.get_obs, env.check_success = original_action, original_obs, original_check


def select_episodes(panel, task, count):
    shard = panel / "formal" / task / "easy"
    manifest = json.loads((shard / "seed_manifest.json").read_bytes())
    seeds = {row["episode_id"]: row for row in manifest["episodes"]}
    rows = [json.loads(line) for line in (shard / "episodes.jsonl").read_bytes().splitlines(keepends=True)
            if line.endswith(b"\n")]
    terminal = {row["episode_id"]: row for row in rows if row["batch"] == 1 and row["status"] in {"success", "failure"}}
    failures = sorted((row for row in terminal.values() if row["status"] == "failure"), key=lambda row: row["index"])
    selected = failures[:count]
    if len(selected) != count:
        raise ValueError(f"Expected {count} original failed episodes for {task}")
    if task == "click_alarmclock":
        selected.append(min((row for row in terminal.values() if row["status"] == "success"), key=lambda row: row["index"]))
    for row in selected:
        seed = seeds[row["episode_id"]]
        for field in ("task", "condition", "phase", "protocol", "protocol_fingerprint"):
            if row[field] != manifest[field]:
                raise ValueError(f"Original journal and manifest context differ for {row['episode_id']}")
        for field in ("actual_seed", "policy_seed", "scene_signature", "index", "batch"):
            if row[field] != seed[field]:
                raise ValueError(f"Original journal and manifest disagree for {row['episode_id']}")
    return [(row, seeds[row["episode_id"]], manifest) for row in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("pinned-evaluation-root", "input-panel", "output-root", "robotwin-root", "xvla-root", "checkpoint"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--episodes-per-task", type=int, choices=(1, 2), default=2)
    parser.add_argument("--record-wrist", action="store_true")
    args = parser.parse_args()
    for name in ("pinned_evaluation_root", "input_panel", "output_root", "robotwin_root", "xvla_root", "checkpoint"):
        setattr(args, name, getattr(args, name).resolve())
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("CUDA_VISIBLE_DEVICES"):
        parser.error("Run this diagnostic through a GPU Slurm allocation")
    if len(args.tasks) != len(set(args.tasks)):
        parser.error("Diagnostic tasks must be unique")
    if args.input_panel.name != EVALUATION_COMMIT or any(args.output_root.is_relative_to(root) for root in
            (args.input_panel.parent, args.pinned_evaluation_root, args.robotwin_root, args.xvla_root, args.checkpoint)):
        parser.error("Use the frozen panel and a fresh output outside all input roots")
    args.output_root.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.pinned_evaluation_root))
    from tools.xvla.robotwin import runner, official_rollout

    runner.pinned_checkout(args.pinned_evaluation_root, EVALUATION_COMMIT)
    if Path(runner.__file__).resolve() != args.pinned_evaluation_root / "tools/xvla/robotwin/runner.py":
        raise ImportError("Diagnostic must import the exact frozen runner")
    runner.pinned_checkout(args.robotwin_root, runner.p.ROBOTWIN_COMMIT)
    runner.pinned_checkout(args.xvla_root, runner.p.XVLA_COMMIT)
    source = Path(__file__).resolve()
    provenance = {"label": "isolated native-budget diagnostic", "counts_toward_formal": False,
                  "job_id": os.environ["SLURM_JOB_ID"], "evaluation_commit": EVALUATION_COMMIT,
                  "diagnostic_commit": runner.git(source.parent, "rev-parse", "HEAD"),
                  "diagnostic_sha256": runner.sha256(source), "official_client_sha256": official_rollout.CLIENT_SHA256,
                  "input_panel": str(args.input_panel), "policy_noise": "same recorded per-episode policy seed",
                  "original_max_queries": 10, "anchors_per_query": runner.p.ANCHORS}
    runner.write_json(args.output_root / "provenance.json", provenance)
    summary = []
    policy, receipt = runner.checkpoint(args.checkpoint, "cuda:0", True)
    runner.write_json(args.output_root / "checkpoint-receipt.json", receipt)
    catalog = {row["task_id"]: row for row in runner.read_json(args.pinned_evaluation_root / "tools/xvla/robotwin/tasks.json")["tasks"]}
    for task in args.tasks:
        module, cls, cfg, robot_config = runner.load_native(args.robotwin_root, task, "easy")
        for original, seed, manifest in select_episodes(args.input_panel, task, args.episodes_per_task):
            folder = args.output_root / task / f"episode-{seed['index'] + 1:04d}"
            folder.mkdir(parents=True)
            runner.write_json(folder / "original-record.json", original)
            runner.write_json(folder / "initial-manifest.json", {key: value for key, value in manifest.items() if key != "episodes"}
                              | {"episode": seed, "source_manifest_sha256": runner.sha256(args.input_panel / "formal" / task / "easy/seed_manifest.json")})
            result = {"task": task, "episode_id": seed["episode_id"], "original_status": original["status"],
                      "label": provenance["label"], "counts_toward_formal": False,
                      "task_source_sha256": runner.sha256(Path(module.__file__)),
                      "embodiment_config_sha256": runner.sha256(robot_config),
                      "actual_seed": seed["actual_seed"], "policy_seed": seed["policy_seed"], "status": "technical_error"}
            budget, env, trace = runner.p.Budget(), cls(), None
            try:
                runner.reset_env(env, cfg, seed["index"], seed["actual_seed"], budget)
                runner.init_success_fields(module, task, env)
                measured = runner.scene_values(env)
                runner.write_json(folder / "reset-scene.json", {"frozen": seed["scene"], "measured": measured,
                                  "matches": runner.same_scene(seed["scene"], measured), "task_state": task_state(env, task)})
                if not runner.same_scene(seed["scene"], measured) or budget.step_lim != catalog[task]["step_limit"]:
                    raise ValueError("Diagnostic reset or native step limit differs from frozen evidence")
                queries = math.ceil(budget.step_lim / runner.p.ANCHORS)
                rollout, changed_ast = expanded_author_rollout(args.xvla_root, queries, official_rollout)
                result.update(native_step_limit=budget.step_lim, native_max_queries=queries, **changed_ast)
                policy.begin(catalog[task]["prompt"], seed["policy_seed"])
                with runner.count_actions(env, budget), instrument(env, budget, task, folder, args.record_wrist) as trace:
                    status, reason, _ = official_rollout.execute(rollout, env, policy, budget)
                result.update(status=status, reason=reason, **trace, budget=asdict(budget),
                              success_at_or_before_original_300=status == "success" and budget.anchors <= 300)
            except Exception:
                if trace is not None:
                    result.update(trace)
                result["traceback"] = traceback.format_exc()
                result["budget"] = asdict(budget)
            finally:
                with suppress(Exception):
                    env.close_env(clear_cache=True)
                runner.write_json(folder / "terminal.json", result)
            summary.append(result)
            runner.write_json(args.output_root / "summary.json", {**provenance, "episodes": summary})
    return int(any(row["status"] == "technical_error" for row in summary))


if __name__ == "__main__":
    raise SystemExit(main())
