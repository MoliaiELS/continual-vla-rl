"""Native X-VLA RoboTwin observation, inference and execution protocol."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


MODEL_ID = "2toINF/X-VLA-RoboTwin2"
MODEL_REVISION = "a157c580cfe6f9f445614490f3bec1b2f9ef9f18"
ROBOTWIN_COMMIT = "6e75b8b80ee32d2f495eb756cf5687e4e16f0eea"
XVLA_COMMIT = "6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2"
RLINF_COMMIT = "1bf78980d9339cbc67497eddceb9a1e3a74e2acc"
PROTOCOL = "robotwin-native-b0-v1"
CAMERAS = ("head_camera", "left_camera", "right_camera")
RGB_SHAPE = (240, 320, 3)
DOMAIN = 6
QUERIES = 10
ANCHORS = 30
DENOISE_STEPS = 10


def observation_values(obs):
    images = [np.asarray(obs["observation"][camera]["rgb"]) for camera in CAMERAS]
    for camera, rgb in zip(CAMERAS, images):
        if rgb.dtype != np.uint8 or rgb.shape != RGB_SHAPE:
            raise ValueError(f"{camera}: expected RGB8 {RGB_SHAPE}, got {rgb.dtype} {rgb.shape}")
    endpose = obs["endpose"]
    arms = []
    for side in ("left", "right"):
        pose = np.asarray(endpose[f"{side}_endpose"], dtype=np.float64)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(f"Invalid measured {side} endpose")
        grip = 1 - 2 * np.asarray(endpose[f"{side}_gripper"])
        if np.ndim(grip) != 0 or not np.isfinite(grip):
            raise ValueError(f"Invalid measured {side} gripper")
        # The author supplies RoboTwin's raw wxyz slots to SciPy as xyzw.
        # Preserve this checkpoint's convention on both sides of inference.
        rot = Rotation.from_quat(pose[3:]).as_matrix()[:, :2].reshape(6)
        arms.extend((pose[:3], rot, np.array([grip])))
    return images, np.concatenate(arms)


class PolicyActionError(ValueError):
    """A model output is unusable for the native action interface."""


def actions_to_env(actions):
    # Native /act round trips float32 predictions through JSON; the official
    # np.array(reply['action']) consequently computes rotations in float64.
    actions = np.asarray(actions, dtype=np.float64)
    if actions.shape != (ANCHORS, 20) or not np.isfinite(actions).all():
        raise PolicyActionError(f"Invalid policy action shape/values: {actions.shape}")
    arms = []
    for offset in (0, 10):
        a1 = actions[:, offset + 3:offset + 8:2]
        a2 = actions[:, offset + 4:offset + 9:2]
        n1 = np.linalg.norm(a1, axis=-1, keepdims=True)
        if np.any(n1 == 0):
            raise PolicyActionError("Degenerate first rotation axis")
        b1 = a1 / n1
        b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
        n2 = np.linalg.norm(b2, axis=-1, keepdims=True)
        if np.any(n2 == 0):
            raise PolicyActionError("Degenerate second rotation axis")
        b2 = b2 / n2
        rot = np.stack((b1, b2, np.cross(b1, b2)), axis=-1)
        # Deliberately no xyzw/wxyz reorder; the official client emits these
        # raw slots and RoboTwin's planner consumes them as wxyz.
        quat = Rotation.from_matrix(rot).as_quat()
        grip = 1 - 2 * (actions[:, offset + 9:offset + 10] > 0.7)
        arms.extend((actions[:, offset:offset + 3], quat, grip))
    return np.concatenate(arms, axis=-1)


def update_history(obs, command):
    # This only replaces xyz and quaternion. Gripper slots stay measured.
    obs["endpose"]["left_endpose"] = list(command[:7])
    obs["endpose"]["right_endpose"] = list(command[8:15])


class PolicyRandomStream:
    """Independent episode random stream without changing generate_actions."""

    def __init__(self, torch, device, seed):
        self.torch, self.device = torch, device
        self.state = torch.Generator(device=device).manual_seed(seed).get_state()

    def call(self, fn, *args, **kwargs):
        if self.device.type == "cuda":
            get = lambda: self.torch.cuda.get_rng_state(self.device)
            set_ = lambda state: self.torch.cuda.set_rng_state(state, self.device)
        else:
            get, set_ = self.torch.get_rng_state, self.torch.set_rng_state
        previous = get()
        set_(self.state)
        try:
            return fn(*args, **kwargs)
        finally:
            self.state = get()
            set_(previous)


class NativePolicy:
    def __init__(self, model, processor, device):
        import torch

        self.torch = torch
        self.model, self.processor, self.device = model, processor, torch.device(device)
        self.dtype = next(model.parameters()).dtype

    def begin(self, instruction, seed):
        self.instruction = instruction
        self.rng = PolicyRandomStream(self.torch, self.device, seed)

    def inputs(self, obs):
        from PIL import Image

        images, proprio = observation_values(obs)
        inputs = self.processor([Image.fromarray(im) for im in images], self.instruction)
        inputs["proprio"] = self.torch.as_tensor(proprio[None])
        inputs["domain_id"] = self.torch.tensor([DOMAIN], dtype=self.torch.long)
        return {
            k: v.to(device=self.device, dtype=self.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()
        }

    def step(self, obs):
        action = self.rng.call(self.model.generate_actions, **self.inputs(obs), steps=DENOISE_STEPS)
        return action.squeeze(0).float().cpu().numpy().astype(np.float64)


@dataclass
class Budget:
    queries: int = 0
    anchors_sent: int = 0
    anchors: int = 0
    physics_ticks: int = 0
    settling_ticks: int = 0
    step_lim: int | None = None
    timestep_seconds: float | None = None
    left_plan_failures: int = 0
    right_plan_failures: int = 0
    query_seconds: float = 0.0
    reset_seconds: float = 0.0


def rollout(env, policy, budget, record_frame=None):
    """Execute the original loop, including no-ops once step_lim is reached."""
    import time

    env._update_render()
    env.actor_pose = True
    obs = env.get_obs()
    for _ in range(QUERIES):
        budget.queries += 1
        start = time.perf_counter()
        try:
            actions = policy.step(obs)
        finally:
            budget.query_seconds += time.perf_counter() - start
        try:
            commands = actions_to_env(actions)
        except PolicyActionError:
            return "failure", "invalid_policy_action"
        for command in commands:
            budget.anchors_sent += 1
            try:
                env.take_action(command, action_type="ee")
            finally:
                budget.anchors = env.take_action_cnt
            obs = env.get_obs()
            update_history(obs, command)
            if record_frame is not None:
                record_frame(obs["observation"]["head_camera"]["rgb"])
            if env.check_success():
                env.suc += 1
                return "success", "official_success"
            if env.actor_pose == False:  # Keep the author's comparison.
                return "failure", "actor_pose_false"
        env._update_render()
    if budget.step_lim is not None and budget.anchors >= budget.step_lim:
        return "failure", "step_limit"
    if budget.left_plan_failures or budget.right_plan_failures:
        return "failure", "query_limit_with_planning_failures"
    return "failure", "query_limit"
