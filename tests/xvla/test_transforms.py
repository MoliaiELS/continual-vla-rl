"""Boundary tests: these do not claim a real model rollout has passed."""
import unittest
import numpy as np

from xvla_rlinf.transforms import (
    NativeProprioHistory, axis_angle_to_matrix, env_actions_from_model,
    matrix_to_rotation_6d, measured_proprio_from_states,
    prepare_processor_inputs, rotation_6d_to_axis_angle, rotation_6d_to_matrix,
    absolute_actions_to_6d, absolute_actions_to_model, absolute_sft_window,
)


class RotationTests(unittest.TestCase):
    def test_columns_are_not_rows(self):
        r = axis_angle_to_matrix(np.array([0., 0., np.pi / 2]))
        six = matrix_to_rotation_6d(r)
        np.testing.assert_allclose(six, [0, 1, 0, -1, 0, 0], atol=1e-10)
        np.testing.assert_allclose(rotation_6d_to_matrix(six), r, atol=2e-6)
        np.testing.assert_allclose(rotation_6d_to_axis_angle(six), [0, 0, np.pi / 2], atol=2e-6)

    def test_random_roundtrip(self):
        aa = np.random.default_rng(4).normal(0, .5, (20, 3))
        matrix = axis_angle_to_matrix(aa)
        converted = rotation_6d_to_axis_angle(matrix_to_rotation_6d(matrix))
        np.testing.assert_allclose(axis_angle_to_matrix(converted), matrix, atol=3e-6)

    def test_identity_and_pi(self):
        for aa in ([0, 0, 0], [np.pi, 0, 0], [0, -np.pi, 0]):
            matrix = axis_angle_to_matrix(aa)
            converted = rotation_6d_to_axis_angle(matrix_to_rotation_6d(matrix))
            np.testing.assert_allclose(axis_angle_to_matrix(converted), matrix, atol=3e-6)

    def test_20_to_10_to_7_and_gripper_boundary(self):
        action = np.zeros((2, 3, 20))
        action[..., :3] = [.4, .1, .9]
        action[..., 3:9] = [0, 1, 0, -1, 0, 0]
        action[0, :, 9] = .5
        action[1, :, 9] = .6
        result = env_actions_from_model(action)
        self.assertEqual(result.shape, (2, 3, 7))
        np.testing.assert_allclose(result[..., 3:6], np.broadcast_to([0, 0, np.pi/2], (2, 3, 3)), atol=2e-6)
        np.testing.assert_array_equal(result[:, 0, -1], [-1, 1])
        # Large padding data never becomes an EEF coordinate or gripper.
        action[..., 10:] = 300
        np.testing.assert_array_equal(env_actions_from_model(action), result)

    def test_explicit_logits(self):
        x = np.zeros((1, 20)); x[:, 3:9] = [1, 0, 0, 0, 1, 0]
        x[:, 9] = -.01
        self.assertEqual(env_actions_from_model(x, gripper_logits=True)[0, -1], -1)
        x[:, 9] = .01
        self.assertEqual(env_actions_from_model(x, gripper_logits=True)[0, -1], 1)

    def test_reject_bad_dimensions(self):
        with self.assertRaises(ValueError):
            env_actions_from_model(np.zeros((1, 7)))
        with self.assertRaises(ValueError):
            rotation_6d_to_matrix([1, 2, 3])

    def test_against_installed_official_robosuite(self):
        try:
            from robosuite.utils import transform_utils as t
        except ImportError:
            self.skipTest("Real robosuite library not installed in this test environment")
        six = matrix_to_rotation_6d(axis_angle_to_matrix(np.random.default_rng(5).normal(0, .5, (10, 3))))
        official = np.array([t.quat2axisangle(t.mat2quat(m)) for m in rotation_6d_to_matrix(six)])
        np.testing.assert_allclose(rotation_6d_to_axis_angle(six), official, atol=2e-6)


class HistoryTests(unittest.TestCase):
    def test_per_environment_history_reset_and_copy(self):
        state = np.zeros((2, 20)); state[:, :3] = [[1, 2, 3], [4, 5, 6]]
        state[:, 3:9] = [1, 0, 0, 0, 1, 0]
        history = NativeProprioHistory()
        initial = history.prepare(state)
        action = np.repeat(initial[:, None], 3, axis=1); action[:, -1, :3] += 10
        history.update(action)
        state[:, :3] += 100
        used = history.prepare(state, [False, True])
        np.testing.assert_array_equal(used[0, :3], [11, 12, 13])
        np.testing.assert_array_equal(used[1, :3], [104, 105, 106])
        used[:] = -200
        self.assertEqual(history.prepare(state)[0, 0], 11)  # cached copy is immutable to callers
        history.reset([0])
        self.assertEqual(history.prepare(state)[0, 0], 101)

    def test_explicit_identity_routing(self):
        state = np.zeros((2, 20))
        history = NativeProprioHistory()
        history.prepare(state, env_ids=[9, 3])
        action = np.zeros((2, 2, 20)); action[0, -1, 0] = 9; action[1, -1, 0] = 3
        history.update(action, env_ids=[9, 3])
        self.assertEqual(history.prepare(state, env_ids=[3, 9])[0, 0], 3)

    def test_measured_mode_is_separate(self):
        history = NativeProprioHistory(mode="measured")
        state = np.zeros((1, 20)); history.prepare(state)
        history.update(np.ones((1, 2, 20)))
        np.testing.assert_array_equal(history.prepare(state), state)

    def test_ordinary_rlinf_state_and_padding(self):
        state = np.zeros((1, 8)); state[:, :3] = [.2, .3, .4]
        actual = measured_proprio_from_states(state)
        np.testing.assert_array_equal(actual[0, 3:9], [1, 0, 0, 0, 1, 0])
        np.testing.assert_array_equal(actual[0, 9:], 0)


class SFTWindowTests(unittest.TestCase):
    def test_absolute_action_target_and_raw_gripper(self):
        absolute = np.array([[.2, .3, .8, 0, 0, np.pi/2, -1.], [.2, .3, .8, 0, 0, 0, 1.]])
        raw = absolute_actions_to_6d(absolute)
        np.testing.assert_array_equal(raw[:, 9], [-1, 1])
        model = absolute_actions_to_model(absolute)
        np.testing.assert_array_equal(model[:, 9], [0, 1])
        np.testing.assert_array_equal(model[:, 10:], 0)
        np.testing.assert_allclose(model[0, 3:9], [0, 1, 0, -1, 0, 0], atol=1e-6)

    def test_official_resampling_proprio_and_drop_first_image(self):
        absolute = np.zeros((41, 10)); absolute[:, 3:9] = [1, 0, 0, 0, 1, 0]
        absolute[:, 0] = np.arange(41) / 30
        absolute[:, 9] = np.where(np.arange(41) >= 20, 1, -1)
        window = absolute_sft_window(absolute, 0, num_actions=30)
        self.assertFalse(window["is_static"])
        self.assertEqual(window["image_index"], 1)
        self.assertEqual(window["action"].shape, (30, 20))
        self.assertEqual(float(window["proprio"][0]), 0)
        self.assertAlmostEqual(float(window["action"][-1, 0]), 1)
        np.testing.assert_array_equal(window["action"][:, 10:], 0)
        # Endpoint clamp: index30 is valid but the horizon ends at frame40.
        tail = absolute_sft_window(absolute, 30, num_actions=30)
        self.assertAlmostEqual(float(tail["action"][-1, 0]), 40/30, places=6)

    def test_static_filter_and_candidate_boundary(self):
        absolute = np.zeros((31, 10))
        self.assertTrue(absolute_sft_window(absolute, 0, num_actions=30)["is_static"])
        with self.assertRaises(ValueError):
            absolute_sft_window(absolute, 21, num_actions=30)


class ProcessorBoundaryTests(unittest.TestCase):
    def test_no_double_flip_and_missing_slot_is_not_supplied(self):
        main = np.arange(36, dtype=np.uint8).reshape(1, 3, 4, 3)
        wrist = main + 30
        received = {}
        def processor(**kwargs):
            received.update(kwargs)
            return {"recorded": True}
        obs = {"main_images": main, "wrist_images": wrist, "task_descriptions": ["put bowl"]}
        self.assertTrue(prepare_processor_inputs(processor, obs)["recorded"])
        self.assertEqual(len(received["images"][0]), 2)
        np.testing.assert_array_equal(np.asarray(received["images"][0][0]), main[0, ::-1, ::-1])
        np.testing.assert_array_equal(np.asarray(received["images"][0][1]), wrist[0])
        aligned = dict(obs, main_images=main[:, ::-1, ::-1], wrist_images=wrist[:, ::-1, ::-1])
        prepare_processor_inputs(processor, aligned, image_orientation="rlinf_libero")
        np.testing.assert_array_equal(np.asarray(received["images"][0][1]), wrist[0])
        del obs["wrist_images"]
        prepare_processor_inputs(processor, obs)
        self.assertEqual(len(received["images"][0]), 1)

    def test_actual_processor_missing_views_mask(self):
        try:
            import torch
            from xvla_rlinf.vendor.xvla.processing_xvla import XVLAProcessor
        except ImportError:
            self.skipTest("Torch/Transformers unavailable; this requires the real processor runtime")
        # Exercise the *actual* source method without network/tokenizer loading.
        processor = object.__new__(XVLAProcessor)
        class ImageProcessor:
            def __call__(self, images, **kwargs):
                return {"pixel_values": torch.stack([torch.as_tensor(np.asarray(i).copy()).permute(2, 0, 1) for i in images])}
        processor.image_processor = ImageProcessor()
        result = processor.encode_image([[np.ones((3, 4, 3), np.uint8)], [np.ones((3, 4, 3), np.uint8)]])
        self.assertEqual(result["image_input"].shape, (2, 3, 3, 3, 4))
        self.assertEqual(result["image_mask"].tolist(), [[True, False, False], [True, False, False]])
        self.assertEqual(int(result["image_input"][:, 1:].count_nonzero()), 0)


if __name__ == "__main__":
    unittest.main()
