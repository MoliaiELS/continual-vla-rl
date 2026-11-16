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
def trajectory(env, steps):
    original = env.take_action

    def action(command, action_type):
        original(command, action_type)
        # No extra camera reads: in Hard, get_obs could advance randomized lights.
        state = np.concatenate([env.robot.get_left_tcp_pose(), env.robot.get_right_tcp_pose(),
                                [env.robot.get_left_gripper_val(), env.robot.get_right_gripper_val()]])
        steps.append((np.copy(command), state))

    env.take_action = action
    try:
        yield
    finally:
        env.take_action = original


def run_parity(args, shard, header, module, cls, cfg, policy, seed, task):
    from .runner import count_actions, emit, init_success_fields, reset_env, same_scene, scene_values, write_json

    result = {**header, "episode_id": seed["episode_id"], "actual_seed": seed["actual_seed"],
              "policy_seed": seed["policy_seed"], "instruction": task["prompt"],
              "status": "fail", "queries_compared": 0, "anchors_compared": 0,
              "same_observation_comparison": True, "same_predicted_chunks_replayed": True,
              "action_max_abs_diff": 0.0, "command_max_abs_diff": 0.0,
              "trajectory_max_abs_diff": 0.0, "unused_front_view_fallback": False}
    expected_inputs = {}
    chunks = []
    native_steps, replay_steps = [], []
    env = None
    try:
        official, original_rollout = reference_client(args.xvla_root, policy, seed["policy_seed"],
                                                       expected_inputs, result)
        policy.begin(task["prompt"], seed["policy_seed"])
        official.set_instruction(task["prompt"])

        def cross_check(obs):
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
            result["action_max_abs_diff"] = max(result["action_max_abs_diff"],
                                                float(np.max(np.abs(official_action - predicted))))
            np.testing.assert_allclose(official_action, predicted, rtol=1e-5, atol=1e-6)
            result["queries_compared"] += 1
            chunks.append(predicted)
            return predicted

        def reset(budget):
            instance = cls()
            try:
                reset_env(instance, cfg, seed["index"], seed["actual_seed"], budget)
                init_success_fields(module, args.task, instance)
                if not same_scene(seed["scene"], scene_values(instance)):
                    raise RuntimeError("Parity reset does not match frozen initial scene")
                if budget.step_lim != task["step_limit"]:
                    raise RuntimeError("Parity step limit differs from catalog")
            except BaseException:
                instance.close_env()
                raise
            return instance

        native_budget = p.Budget()
        env = reset(native_budget)
        with count_actions(env, native_budget), trajectory(env, native_steps):
            native_status, result["native_reason"] = p.rollout(env, SimpleNamespace(step=cross_check), native_budget)
        env.close_env()
        env = None
        iterator = iter(chunks)
        replay_budget = p.Budget()
        env = reset(replay_budget)
        with count_actions(env, replay_budget), trajectory(env, replay_steps):
            replay_success, _ = original_rollout(env, SimpleNamespace(step=lambda obs: next(iterator)))
        result.update(native_success=native_status == "success", official_success=bool(replay_success),
                      native_ticks=native_budget.physics_ticks, official_ticks=replay_budget.physics_ticks,
                      anchors_compared=len(native_steps))
        require(result["native_success"] == result["official_success"], "Official rollout success differs")
        require(len(native_steps) == len(replay_steps), "Official rollout anchor count differs")
        require(native_budget.physics_ticks == replay_budget.physics_ticks, "Official rollout tick count differs")
        require(result["queries_compared"] > 0 and len(native_steps) > 0, "No command traced")
        command_diff, trajectory_diff = 0.0, 0.0
        for native, replay in zip(native_steps, replay_steps):
            command_diff = max(command_diff, float(np.max(np.abs(native[0] - replay[0]))))
            # Quaternion sign represents the same orientation. This equivalence
            # applies only to trajectory measurements, not command-array inputs.
            state = replay[1].copy()
            for offset in (3, 10):
                if np.dot(native[1][offset:offset + 4], state[offset:offset + 4]) < 0:
                    state[offset:offset + 4] *= -1
            trajectory_diff = max(trajectory_diff, float(np.max(np.abs(native[1] - state))))
        result.update(command_max_abs_diff=command_diff, trajectory_max_abs_diff=trajectory_diff)
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
