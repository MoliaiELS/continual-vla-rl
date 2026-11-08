# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

"""Independent numerical reference extracted from the pinned official LIBERO client.

Source: evaluation/libero/libero_client.py at
2toinf/X-VLA@6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2.
Apache-2.0; the complete upstream license is retained at xvla/LICENSE.

The image helper and rotation processor below are exact source excerpts. The
batch wrapper adapts ClientModel.step's postprocessing without importing any
adapter transform, running an HTTP server, or constructing a simulator.
Only alignment checks should use this reference, never the policy execution.
"""
from __future__ import annotations

from typing import List
import numpy as np
import robosuite.utils.transform_utils as T

EPS = 1e-6
SOURCE_COMMIT = "6c4ecfb8be458ae7c0c96094e4b0add4174f4fe2"
SOURCE_PATH = "evaluation/libero/libero_client.py"
SOURCE_SHA256 = 'af52caebd5c1d923d07ef5a77e43cd148445a1118da46131ff0818fa216d080b'
SOURCE_MEMBER_SHA256 = {'_flip_agentview': '4c00ed862d0856fd546c01f5667066ebf16bb9efc7730f4cd1acde1b0f1468f8', 'LiberoAbsActionProcessor': 'c0dd05f9cf04f0e157d42665e6742a5180ee99c6a012736f0b1691255f9a48a2'}


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    """Match original code behavior: vertical+horizontal flips."""
    return np.flip(np.flip(img, 0), 1)



class LiberoAbsActionProcessor:
    """Helpers to convert between 6D rotation (Zhou et al.) and axis-angle."""

    def Rotate6D_to_AxisAngle(self, r6d: np.ndarray) -> np.ndarray:
        """Convert 6D rotation representation to axis-angle.

        Args:
            r6d: array with shape (N, 6) or (6,)
        Returns:
            array with shape (N, 3) or (3,)
        """
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        # b1
        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)

        # b2
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)

        # b3
        b3 = np.cross(b1, b2, axis=-1)

        R = np.stack([b1, b2, b3], axis=-1)  # (N, 3, 3)

        axis_angle_list: List[np.ndarray] = []
        for i in range(R.shape[0]):
            quat = T.mat2quat(R[i])
            axis_angle = T.quat2axisangle(quat)
            axis_angle_list.append(axis_angle)

        axis_angle_array = np.stack(axis_angle_list, axis=0)
        return axis_angle_array[0] if single else axis_angle_array

    def Mat_to_Rotate6D(self, R: np.ndarray) -> np.ndarray:
        if R.ndim == 2:
            return np.concatenate([R[:3, 0], R[:3, 1]], axis=-1)
        elif R.ndim == 3:
            return np.concatenate([R[:, :3, 0], R[:, :3, 1]], axis=-1)
        else:
            raise ValueError("Rotation matrix must be (...,3,3)")

    def AxisAngle_to_Rotate6D(self, aa: np.ndarray) -> np.ndarray:
        if aa.ndim == 1:
            return self.Mat_to_Rotate6D(T.quat2mat(T.axisangle2quat(aa)))
        else:
            raise ValueError("Only 1D axis-angle supported here")

    def action_6d_to_axisangle(self, action: np.ndarray) -> np.ndarray:
        """Convert action [..., 3(pos)+6(rot6d)+1(grip)] -> [..., 3(pos)+3(aa)+1(grip)]"""
        if action.ndim == 1:
            final_ori = self.Rotate6D_to_AxisAngle(action[3:9])
            return np.concatenate([action[0:3], final_ori, action[-1:]])
        elif action.ndim == 2:
            final_ori = self.Rotate6D_to_AxisAngle(action[:, 3:9])
            return np.concatenate([action[:, 0:3], final_ori, action[:, -1:]], axis=-1)
        else:
            raise ValueError("Unsupported action shape")



def reference_environment_actions(postprocessed_action: np.ndarray) -> np.ndarray:
    """Native ClientModel.step mapping: post-sigmoid ee6d -> float32 EEF7.

    The wrapper batches independent official client chunks. Both the class
    above and this wrapper are independent of xvla_rlinf.transforms.
    """
    action = np.asarray(postprocessed_action)
    if action.ndim < 2 or action.shape[-1] not in (10, 20):
        raise ValueError("Expected official postprocessed [...,H,10|20] actions")
    if not np.isfinite(action).all():
        raise ValueError("Reference actions must be finite")
    horizon = action.shape[-2]
    if horizon < 1:
        raise ValueError("Reference action chunk cannot be empty")
    processor = LiberoAbsActionProcessor()
    chunks = []
    for chunk in action.reshape(-1, horizon, action.shape[-1]):
        # Exact ClientModel.step order, with deque/HTTP transport removed.
        target_eef = chunk[:, :3]
        target_axis = processor.Rotate6D_to_AxisAngle(chunk[:, 3:9])
        target_act = chunk[:, 9:10]
        final_action = np.concatenate([target_eef, target_axis, target_act], axis=-1)
        rows = []
        for row in final_action.tolist():
            action_predict = np.array(row, dtype=np.float32)
            action_predict[-1] = 1.0 if action_predict[-1] > 0.5 else -1.0
            rows.append(action_predict)
        chunks.append(np.stack(rows))
    return np.stack(chunks).reshape(action.shape[:-1] + (7,))
