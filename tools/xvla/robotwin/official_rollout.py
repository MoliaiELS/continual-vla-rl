"""Run the pinned X-VLA client loop with episode accounting at its boundaries."""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from . import protocol as p


CLIENT_PATH = Path("evaluation/robotwin-2.0/client.py")
CLIENT_SHA256 = "642e9c316cd57faf138023e5e0d77159e8f1007e2c4efc5aa68d61453db9b34b"


def load_official_rollout(xvla_root):
    """Load only the verified rollout and quaternion conversion definitions."""
    path = Path(xvla_root).resolve() / CLIENT_PATH
    source = path.read_bytes()
    actual = hashlib.sha256(source).hexdigest()
    if actual != CLIENT_SHA256:
        raise ValueError(f"Wrong X-VLA RoboTwin client SHA256 at {path}: {actual}")
    names = {"_rollout", "rotate6D_to_quat"}
    syntax = ast.parse(source, filename=str(path))
    definitions = [node for node in syntax.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(definitions) != len(names) or {node.name for node in definitions} != names:
        raise ValueError(f"Pinned X-VLA client is missing the required rollout definitions: {path}")
    # Avoid importing the original script's machine-specific paths and HTTP
    # entrypoint. The executable loop and quaternion conversion stay untouched.
    globals_ = {"np": np, "R": Rotation, "tqdm": lambda iterable: iterable}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), "exec"), globals_)
    return globals_["_rollout"]


def execute(official, env, policy, budget, *, return_frames=False, check_command=None):
    """Return (status, reason, frames) with native query and anchor accounting."""
    original = env.take_action
    saved = vars(env).get("take_action")
    had_override = "take_action" in vars(env)

    def take_action(command, action_type):
        if check_command is not None:
            check_command(command, action_type)
        budget.anchors_sent += 1
        try:
            return original(command, action_type=action_type)
        finally:
            budget.anchors = env.take_action_cnt

    def step(obs):
        budget.queries += 1
        started = time.perf_counter()
        try:
            raw = policy.step(obs)
        finally:
            budget.query_seconds += time.perf_counter() - started
        try:
            raw = np.asarray(raw, dtype=np.float64)
            p.actions_to_env(raw)  # Validate only. The official loop converts and delivers actions.
        except (TypeError, ValueError) as error:
            raise p.PolicyActionError("Invalid anchor chunk for the official client") from error
        return raw

    env.take_action = take_action
    try:
        success, images = official(env, SimpleNamespace(step=step))
    except p.PolicyActionError:
        return "failure", "invalid_policy_action", None
    finally:
        if had_override:
            env.take_action = saved
        else:
            del env.take_action
    frames = images if return_frames else None
    if success:
        return "success", "official_success", frames
    if env.actor_pose == False:  # Match the original client's comparison.
        return "failure", "actor_pose_false", frames
    if budget.step_lim is not None and budget.anchors >= budget.step_lim:
        return "failure", "step_limit", frames
    if budget.left_plan_failures or budget.right_plan_failures:
        return "failure", "query_limit_with_planning_failures", frames
    return "failure", "query_limit", frames
