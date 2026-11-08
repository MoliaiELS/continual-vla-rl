"""Native X-VLA/LIBERO boundary transforms (no model or simulator imports).

The reference uses column-based Rotate6D, an absolute EEF controller, and
postprocessed gripper probabilities. These are not OpenVLA delta actions.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

EPS = 1e-6  # Exact stabilizer in the fixed official LIBERO client.


def as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def matrix_to_rotation_6d(matrix: Any) -> np.ndarray:
    matrix = as_numpy(matrix)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("Expected rotation matrices [...,3,3]")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def rotation_6d_to_matrix(rotation: Any) -> np.ndarray:
    """Official Gram-Schmidt with column vectors, including its +1e-6."""
    rotation = as_numpy(rotation)
    if rotation.shape[-1] != 6 or not np.isfinite(rotation).all():
        raise ValueError("Rotate6D must contain six finite values")
    a1, a2 = rotation[..., :3], rotation[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
    orthogonal = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = orthogonal / (np.linalg.norm(orthogonal, axis=-1, keepdims=True) + EPS)
    return np.stack((b1, b2, np.cross(b1, b2)), axis=-1)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Eigenvector conversion used by robosuite's mat2quat; output xyzw."""
    values = []
    for m in matrix.reshape(-1, 3, 3):
        qxx, qyx, qzx, qxy, qyy, qzy, qxz, qyz, qzz = m.flat
        k = np.array([
            [qxx - qyy - qzz, 0, 0, 0],
            [qyx + qxy, qyy - qxx - qzz, 0, 0],
            [qzx + qxz, qzy + qyz, qzz - qxx - qyy, 0],
            [qyz - qzy, qzx - qxz, qxy - qyx, qxx + qyy + qzz],
        ]) / 3.0
        eigenvalues, eigenvectors = np.linalg.eigh(k)
        xyzw = eigenvectors[:, eigenvalues.argmax()]
        if xyzw[3] < 0:
            xyzw *= -1
        values.append(xyzw)
    return np.asarray(values).reshape(matrix.shape[:-2] + (4,))


def rotation_6d_to_axis_angle(rotation: Any, *, backend: str = "numpy") -> np.ndarray:
    """Convert Rotate6D via a quaternion, never by reshaping six coordinates.

    ``robosuite`` selects the exact reference library for native alignment.
    ``numpy`` implements the same eigenvector convention without importing the
    simulator, allowing boundary tests to run independently.
    """
    matrix = rotation_6d_to_matrix(rotation)
    flat = matrix.reshape(-1, 3, 3)
    if backend == "robosuite":
        from robosuite.utils import transform_utils as t
        aa = np.asarray([t.quat2axisangle(t.mat2quat(m)) for m in flat])
    elif backend == "numpy":
        quaternion = _matrix_to_quaternion(flat)
        w = np.clip(quaternion[..., 3], -1.0, 1.0)
        denominator = np.sqrt(np.maximum(1.0 - w * w, 0.0))
        scale = np.divide(2.0 * np.arccos(w), denominator,
                          out=np.zeros_like(w), where=~np.isclose(denominator, 0))
        aa = quaternion[..., :3] * scale[..., None]
    else:
        raise ValueError(f"Unknown rotation backend: {backend}")
    return aa.reshape(matrix.shape[:-2] + (3,))


def axis_angle_to_matrix(axis_angle: Any) -> np.ndarray:
    aa = as_numpy(axis_angle).astype(np.float64)
    if aa.shape[-1] != 3 or not np.isfinite(aa).all():
        raise ValueError("Expected finite axis-angle [...,3]")
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    axis = np.divide(aa, theta, out=np.zeros_like(aa), where=theta > 1e-12)
    x, y, z = np.moveaxis(axis, -1, 0)
    skew = np.zeros(aa.shape[:-1] + (3, 3), dtype=np.float64)
    skew[..., 0, 1], skew[..., 0, 2] = -z, y
    skew[..., 1, 0], skew[..., 1, 2] = z, -x
    skew[..., 2, 0], skew[..., 2, 1] = -y, x
    eye = np.broadcast_to(np.eye(3), skew.shape)
    return eye + np.sin(theta)[..., None] * skew + (1 - np.cos(theta))[..., None] * (skew @ skew)


def measured_proprio_from_states(states: Any) -> np.ndarray:
    """Build the official 20D single-arm proprio, leaving both grippers zero.

    Prefer controller-derived state20/state9 supplied by the controlled LIBERO
    patch. Ordinary RLinf state8 is [xyz, axis-angle, finger0, finger1].
    """
    state = as_numpy(states)
    if state.ndim == 1:
        state = state[None]
    if state.ndim != 2 or state.shape[1] not in (8, 9, 20):
        raise ValueError("Expected batched controller state9/20 or RLinf state8")
    if not np.isfinite(state).all():
        raise ValueError("Non-finite robot state")
    out = np.zeros((state.shape[0], 20), dtype=np.float32)
    out[:, :3] = state[:, :3]
    out[:, 3:9] = (matrix_to_rotation_6d(axis_angle_to_matrix(state[:, 3:6]))
                    if state.shape[1] == 8 else state[:, 3:9])
    return out


def env_actions_from_model(action: Any, *, gripper_logits: bool = False,
                           rotation_backend: str = "numpy") -> np.ndarray:
    """Convert model20/single-arm10 into absolute LIBERO xyz+rotvec+grip7.

    Native generate_actions has already applied sigmoid. RL raw logits callers
    must explicitly request sigmoid here. No OpenVLA gripper inversion follows.
    """
    action = as_numpy(action)
    if action.shape[-1] not in (10, 20) or not np.isfinite(action).all():
        raise ValueError("Expected finite ee6d internal20/single-arm10 actions")
    grip = action[..., 9:10]
    if gripper_logits:
        # Stable sigmoid, with no influence on the raw policy probability.
        positive = grip >= 0
        e = np.exp(-np.abs(grip))
        grip = np.where(positive, 1.0 / (1.0 + e), e / (1.0 + e))
    aa = rotation_6d_to_axis_angle(action[..., 3:9], backend=rotation_backend)
    return np.concatenate((action[..., :3], aa, np.where(grip > .5, 1., -1.)), axis=-1).astype(np.float32)


def absolute_actions_to_6d(absolute_actions: Any) -> np.ndarray:
    """Absolute controller xyz/axis-angle/raw-grip7 -> official HDF5 10D.

    This function does NOT convert relative actions to absolute control. Obtain
    controller.goal_pos/goal_ori by actual environment replay first.
    """
    action = as_numpy(absolute_actions)
    if action.shape[-1] != 7 or not np.isfinite(action).all():
        raise ValueError("Expected finite absolute controller actions [...,7]")
    return np.concatenate((action[..., :3], matrix_to_rotation_6d(axis_angle_to_matrix(action[..., 3:6])),
                           action[..., 6:7]), axis=-1).astype(np.float32)


def absolute_actions_to_model(absolute_actions: Any) -> np.ndarray:
    """Absolute controller7 -> clean EE6D20 target; grip=(raw_grip>0)."""
    six = absolute_actions_to_6d(absolute_actions)
    left = six.copy()
    left[..., 9] = left[..., 9] > 0
    return np.concatenate((left, np.zeros_like(left)), axis=-1)


def absolute_sft_window(absolute_6d: Any, index: int, *, num_actions: int,
                        frequency: float = 30., duration: float = 1.) -> dict[str, Any]:
    """Fixed official LiberoHandler + BaseHDF5Handler + action_slice semantics.

    Input is one episode's ``abs_action_6d[T,10]`` with raw signed gripper.
    Candidate indices are [0,T-10); interpolate a one-second window to H+1
    float32 query times, take point0 as proprio and points1:H+1 as targets.
    Returns ``is_static`` for the caller to filter (never silently substitutes
    another episode/frame), plus ``image_index=index+1`` for dropped-first-image
    alignment. No image preprocessing or randomness occurs in this function.
    """
    data = as_numpy(absolute_6d)
    if data.ndim != 2 or data.shape[1] != 10 or not np.isfinite(data).all():
        raise ValueError("Expected episode abs_action_6d[T,10]")
    if not 0 <= int(index) < len(data) - 10:
        raise ValueError("Official LIBERO candidate index must satisfy 0 <= index < T-10")
    if num_actions < 1 or frequency <= 0 or duration <= 0:
        raise ValueError("Positive num_actions, frequency and duration required")
    # Exactly like LiberoHandler: binarize BEFORE temporal interpolation.
    left = np.concatenate((data[:, :9], (data[:, 9:] > 0).astype(data.dtype)), axis=-1)
    times = np.arange(len(data), dtype=np.float64) / float(frequency)
    query = np.linspace(times[index], min(times[index] + duration, times[-1]),
                        num_actions + 1, dtype=np.float32)
    sampled = np.stack([np.interp(query, times, left[:, column]) for column in range(10)], axis=-1)
    static = bool(np.max(np.abs(sampled[1] - sampled[0])) < 1e-5)
    trajectory = np.concatenate((sampled, np.zeros_like(sampled)), axis=-1).astype(np.float32)
    return {"proprio": trajectory[0].copy(), "action": trajectory[1:].copy(),
            "query_times": query, "is_static": static, "image_index": int(index) + 1,
            "episode_length": len(data), "source_frequency": float(frequency)}


class NativeProprioHistory:
    """Each environment keeps the last full predicted chunk's final pose.

    prepare() returns a copy: actor replay must cache this actual model input,
    not reconstruct it later from the current simulator state. reset_mask is
    applied before input construction and environment identities may be routed.
    """
    def __init__(self, mode: str = "native_history"):
        if mode not in ("native_history", "measured"):
            raise ValueError("proprio mode must be native_history or measured")
        self.mode = mode
        self._states: dict[int, np.ndarray] = {}

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._states.clear()
        else:
            for key in env_ids:
                self._states.pop(int(key), None)

    @staticmethod
    def _ids(batch: int, env_ids: Sequence[int] | None) -> list[int]:
        ids = list(range(batch)) if env_ids is None else [int(i) for i in as_numpy(env_ids).reshape(-1)]
        if len(ids) != batch or len(set(ids)) != batch:
            raise ValueError("One unique environment identity per batch item is required")
        return ids

    def prepare(self, measured: Any, reset_mask: Any | None = None,
                env_ids: Sequence[int] | None = None) -> np.ndarray:
        measured = measured_proprio_from_states(measured)
        ids = self._ids(len(measured), env_ids)
        reset = np.zeros(len(ids), bool) if reset_mask is None else as_numpy(reset_mask).astype(bool).reshape(-1)
        if len(reset) != len(ids):
            raise ValueError("reset_mask must match batch size")
        result = []
        for index, key in enumerate(ids):
            if reset[index] or self.mode == "measured" or key not in self._states:
                self._states[key] = measured[index].copy()
            result.append(self._states[key].copy())
        return np.stack(result)

    def update(self, predicted_actions: Any, env_ids: Sequence[int] | None = None) -> None:
        predicted = as_numpy(predicted_actions)
        if predicted.ndim != 3 or predicted.shape[-1] not in (10, 20) or predicted.shape[1] == 0:
            raise ValueError("Expected predicted [B,H,10|20]")
        if not np.isfinite(predicted).all():
            raise ValueError("Non-finite native proprio prediction")
        ids = self._ids(len(predicted), env_ids)
        for index, key in enumerate(ids):
            if key not in self._states:
                raise RuntimeError("prepare() must initialize each environment before update()")
            self._states[key][:9] = predicted[index, -1, :9]


def prepare_processor_inputs(processor: Any, env_obs: dict[str, Any], *,
                             image_orientation: str = "native_raw") -> dict[str, Any]:
    """Use the audited official processor; do not add valid zero views.

    native_raw: raw agentview+wrist; flip main only as official client does.
    rlinf_libero: upstream helper flipped both; undo wrist only.
    native_aligned: images have already been explicitly aligned by caller.
    """
    from PIL import Image
    if image_orientation not in ("native_raw", "rlinf_libero", "native_aligned"):
        raise ValueError("Unknown image orientation")
    main = as_numpy(env_obs["main_images"])
    if main.ndim != 4 or main.shape[-1] != 3:
        raise ValueError("Expected HWC RGB main_images [B,H,W,3]")
    wrist = env_obs.get("wrist_images")
    wrist = None if wrist is None else as_numpy(wrist)
    if wrist is not None and (wrist.ndim != 4 or len(wrist) != len(main) or wrist.shape[-1] != 3):
        raise ValueError("Expected optional wrist_images [B,H,W,3]")
    samples = []
    for index in range(len(main)):
        m = main[index]
        if image_orientation == "native_raw":
            m = m[::-1, ::-1]
        views = [Image.fromarray(np.ascontiguousarray(m))]
        if wrist is not None:
            w = wrist[index]
            if image_orientation == "rlinf_libero":
                w = w[::-1, ::-1]
            views.append(Image.fromarray(np.ascontiguousarray(w)))
        samples.append(views)
    instructions = env_obs["task_descriptions"]
    if isinstance(instructions, str):
        instructions = [instructions]
    if len(instructions) != len(samples):
        raise ValueError("One instruction per environment is required")
    # XVLAProcessor pads to three slots and marks padding image_mask=False.
    return processor(images=samples, language_instruction=list(instructions))
