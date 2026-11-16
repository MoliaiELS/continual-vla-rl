"""Compare the runner with the pinned X-VLA client and /act endpoint."""
from __future__ import annotations

import ast
from contextlib import contextmanager, suppress
import json
import traceback
from types import MethodType, SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from . import protocol as p


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def extract(path, names, ns, class_name=None):
    # Loading the script verbatim would chdir to the original author's machine.
    # Execute only the untouched definitions being audited, under local bindings.
    syntax = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = syntax.body
    if class_name is not None:
        body = next(node for node in body if isinstance(node, ast.ClassDef) and node.name == class_name).body
    definitions = [node for node in body if getattr(node, "name", None) in names]
    if {node.name for node in definitions} != set(names):
        raise ValueError(f"Missing official reference definitions: {names}")
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), ns)
    return ns


def reference_client(xvla_root, policy, seed, expected_inputs, stats):
    """Run original ClientModel.step against original act, without HTTP transport."""
    import cv2
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from PIL import Image
    import json_numpy
    import logging
    from typing import Any, Dict

    from .runner import sha256

    source = xvla_root / "models/modeling_xvla.py"
    client_file = xvla_root / "evaluation/robotwin-2.0/client.py"
    torch = policy.torch
    official = extract(source, ("generate_actions", "_build_app"),
                       dict(np=np, torch=torch, Dict=Dict, Any=Any, Image=Image, json_numpy=json_numpy,
                            cv2=cv2, logging=logging, traceback=traceback, FastAPI=FastAPI,
                            JSONResponse=JSONResponse), "XVLA")
    original_generate = MethodType(official["generate_actions"], policy.model)
    native_generate = policy.model.generate_actions
    random_stream = p.PolicyRandomStream(torch, policy.device, seed)

    def generate(**inputs):
        steps = inputs.pop("steps")
        require(steps == p.DENOISE_STEPS, "Wrong denoising budget")
        require(expected_inputs.keys() == inputs.keys(), "Official model input fields differ")
        for name, actual in inputs.items():
            expected = expected_inputs[name]
            if not torch.equal(actual, expected):
                raise AssertionError(f"Official model input differs: {name}")
        return random_stream.call(original_generate, **inputs, steps=steps)

    official["_build_app"](policy.model, policy.processor)
    route = next(r for r in policy.model.app.routes if r.path == "/act")

    def post(_url, **kwargs):
        policy.model.generate_actions = generate
        try:
            response = route.endpoint(kwargs["json"])
        finally:
            policy.model.generate_actions = native_generate
        if response.status_code != 200:
            raise RuntimeError(f"Original /act failed: {response.body.decode()}")
        return SimpleNamespace(json=lambda: json.loads(response.body))

    client = extract(client_file, ("quat_to_rotate6D", "rotate6D_to_quat", "ClientModel", "_rollout"),
                     dict(np=np, R=R, requests=SimpleNamespace(post=post), json_numpy=json_numpy,
                          tqdm=lambda iterable: iterable))
    stats.update(official_client_sha256=sha256(client_file), official_model_source_sha256=sha256(source))
    return client["ClientModel"]("in-process", 0), client["_rollout"]


@contextmanager
def trajectory(env, budget, steps, calls):
    originals = {name: getattr(env, name) for name in ("take_action", "_update_render", "get_obs", "check_success")}
    inside_action = False

    def outer(name):
        def call(*args, **kwargs):
            if not inside_action:
                calls.append(name)
            return originals[name](*args, **kwargs)
        return call

    def action(command, action_type):
        nonlocal inside_action
        calls.append(f"take_action:{action_type}")
        before = budget.physics_ticks
        inside_action = True
        try:
            originals["take_action"](command, action_type)
        finally:
            inside_action = False
        # No extra camera reads: in Hard, get_obs could advance randomized lights.
        state = np.concatenate([env.robot.get_left_tcp_pose(), env.robot.get_right_tcp_pose(),
                                [env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()]])
        steps.append((np.copy(command), state, budget.physics_ticks - before))

    env.take_action = action
    for name in ("_update_render", "get_obs", "check_success"):
        setattr(env, name, outer(name))
    try:
        yield
    finally:
        for name, original in originals.items():
            setattr(env, name, original)


def scene_difference(first, second):
    if len(first) != len(second) or any(a[0] != b[0] or len(a) != len(b) for a, b in zip(first, second)):
        return None
    return max((abs(x - y) for a, b in zip(first, second) for x, y in zip(a[1:], b[1:])), default=0.0)


def compare_trajectories(first, second):
    command_diff, tcp_diff, tick_deltas = [], [], []
    for (command_a, state_a, ticks_a), (command_b, state_b, ticks_b) in zip(first, second):
        command_diff.append(float(np.max(np.abs(command_a - command_b))))
        # q and -q are equivalent for measured poses only. Commands must match
        # the official raw quaternion slots exactly as delivered to RoboTwin.
        state_b = state_b.copy()
        for offset in (3, 10):
            if np.dot(state_a[offset:offset + 4], state_b[offset:offset + 4]) < 0:
                state_b[offset:offset + 4] *= -1
        tcp_diff.append(float(np.max(np.abs(state_a - state_b))))
        tick_deltas.append(ticks_a - ticks_b)
    return dict(command_max_abs_diff=max(command_diff, default=None),
                trajectory_max_abs_diff=max(tcp_diff, default=None),
                anchors_compared=len(tick_deltas),
                tick_deltas_by_anchor=tick_deltas,
                first_tick_divergence_anchor=next((i for i, delta in enumerate(tick_deltas, 1) if delta), None),
                first_tcp_gate_divergence_anchor=next((i for i, delta in enumerate(tcp_diff, 1) if delta > 1e-3), None))


def compare_calls(first, second):
    index = next((i for i, (a, b) in enumerate(zip(first, second)) if a != b), min(len(first), len(second)))
    if first == second:
        return None
    return dict(index=index + 1, first=first[index] if index < len(first) else None,
                second=second[index] if index < len(second) else None)


def run_parity(args, shard, header, module, cls, cfg, policy, seed, task):
    from .runner import count_actions, emit, init_success_fields, reset_env, same_scene, scene_values, write_json

    result = {**header, "episode_id": seed["episode_id"], "actual_seed": seed["actual_seed"],
              "policy_seed": seed["policy_seed"], "instruction": task["prompt"],
              "status": "fail", "queries_compared": 0, "anchors_compared": 0,
              "same_observation_comparison": True, "same_predicted_chunks_replayed": True,
              "action_max_abs_diff": None, "command_max_abs_diff": None,
              "trajectory_max_abs_diff": None, "unused_front_view_fallback": False}
    expected_inputs = {}
    chunks = []
    native_steps, replay_steps = [], []
    native_calls, replay_calls = [], []
    env = None
    try:
        official, original_rollout = reference_client(args.xvla_root, policy, seed["policy_seed"],
                                                       expected_inputs, result)
        policy.begin(task["prompt"], seed["policy_seed"])
        official.set_instruction(task["prompt"])

        def cross_check(obs):
            native_calls.append("query")
            expected_inputs.clear()
            expected_inputs.update(policy.inputs(obs))
            predicted = policy.rng.call(policy.model.generate_actions, **expected_inputs,
                                        steps=p.DENOISE_STEPS).squeeze(0).float().cpu().numpy().astype(np.float64)
            # The official client stacks a fourth image only for its local vision
            # log. Don't let a missing log-only camera change the /act input.
            if "front_camera" not in obs["observation"]:
                obs = {**obs, "observation": {**obs["observation"],
                                              "front_camera": obs["observation"]["head_camera"]}}
                result["unused_front_view_fallback"] = True
            official_action = official.step(obs)
            result["action_max_abs_diff"] = max(result["action_max_abs_diff"] or 0.0,
                                                float(np.max(np.abs(official_action - predicted))))
            np.testing.assert_allclose(official_action, predicted, rtol=1e-5, atol=1e-6)
            result["queries_compared"] += 1
            chunks.append(predicted)
            return predicted

        def reset(budget, name):
            instance = cls()
            try:
                reset_env(instance, cfg, seed["index"], seed["actual_seed"], budget)
                init_success_fields(module, args.task, instance)
                scene = scene_values(instance)
                result[f"{name}_frozen_scene_max_abs_diff"] = scene_difference(seed["scene"], scene)
                if not same_scene(seed["scene"], scene):
                    raise RuntimeError("Parity reset does not match frozen initial scene")
                if budget.step_lim != task["step_limit"]:
                    raise RuntimeError("Parity step limit differs from catalog")
            except BaseException:
                instance.close_env()
                raise
            return instance, scene

        def replay(instance, budget, steps, calls):
            iterator = iter(chunks)

            def step(_obs):
                calls.append("query")
                return next(iterator)

            with count_actions(instance, budget), trajectory(instance, budget, steps, calls):
                success, _ = original_rollout(instance, SimpleNamespace(step=step))
            return bool(success)

        native_budget = p.Budget()
        env, native_scene = reset(native_budget, "native_reset")
        with count_actions(env, native_budget), trajectory(env, native_budget, native_steps, native_calls):
            native_status, result["native_reason"] = p.rollout(env, SimpleNamespace(step=cross_check), native_budget)
        env.close_env()
        env = None
        replay_budget = p.Budget()
        env, replay_scene = reset(replay_budget, "official_reset")
        replay_success = replay(env, replay_budget, replay_steps, replay_calls)
        result.update(native_success=native_status == "success", official_success=replay_success,
                      native_ticks=native_budget.physics_ticks, official_ticks=replay_budget.physics_ticks,
                      native_anchors=len(native_steps), official_anchors=len(replay_steps),
                      reset_pair_scene_max_abs_diff=scene_difference(native_scene, replay_scene),
                      native_motion_calls=native_calls, official_motion_calls=replay_calls,
                      first_motion_call_divergence=compare_calls(native_calls, replay_calls),
                      **compare_trajectories(native_steps, replay_steps))
        command_diff = result["command_max_abs_diff"]
        trajectory_diff = result["trajectory_max_abs_diff"]
        physical_mismatch = (native_budget.physics_ticks != replay_budget.physics_ticks
                             or result["native_success"] != replay_success
                             or (trajectory_diff is not None and trajectory_diff > 1e-3))
        if (physical_mismatch and len(native_steps) == len(replay_steps) and command_diff is not None
                and command_diff <= 1e-10 and result["first_motion_call_divergence"] is None):
            # Quantify whether the same client itself varies on an independent
            # seeded reset. This is diagnostic and cannot pass the parity gate.
            repeat = {}
            result["official_repeat"] = repeat
            repeat_steps, repeat_calls = [], []
            try:
                env.close_env()
                env = None
                repeat_budget = p.Budget()
                env, repeat_scene = reset(repeat_budget, "official_repeat_reset")
                repeat_success = replay(env, repeat_budget, repeat_steps, repeat_calls)
                repeat.update(success=repeat_success, reference_success=replay_success,
                              ticks=repeat_budget.physics_ticks, reference_ticks=replay_budget.physics_ticks,
                              anchors=len(repeat_steps), reference_anchors=len(replay_steps),
                              reset_pair_scene_max_abs_diff=scene_difference(replay_scene, repeat_scene),
                              motion_calls=repeat_calls,
                              first_motion_call_divergence=compare_calls(replay_calls, repeat_calls),
                              **compare_trajectories(replay_steps, repeat_steps))
            except Exception:
                repeat["traceback"] = traceback.format_exc()
        require(result["native_success"] == result["official_success"], "Official rollout success differs")
        require(len(native_steps) == len(replay_steps), "Official rollout anchor count differs")
        require(native_budget.physics_ticks == replay_budget.physics_ticks, "Official rollout tick count differs")
        require(result["queries_compared"] > 0 and len(native_steps) > 0, "No command traced")
        require(command_diff <= 1e-10, "Official command arrays differ")
        require(trajectory_diff <= 1e-3, "Official execution trajectory differs")
        result["status"] = "pass"
    except Exception:
        result["traceback"] = traceback.format_exc()
        raise
    finally:
        if env is not None:
            with suppress(Exception):
                env.close_env(clear_cache=True)
        write_json(shard / "parity.json", result)
        emit(result)
