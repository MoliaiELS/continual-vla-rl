"""Guard and pose checks without importing CUDA, MuJoCo or LIBERO."""
import unittest

from validate_ood import require_slurm, verify_camera_poses


class OODValidationTests(unittest.TestCase):
    def test_refuses_login_node_and_missing_allocation(self):
        env = {"SLURM_JOB_ID": "12345", "CUDA_VISIBLE_DEVICES": "2", "MUJOCO_GL": "egl"}
        require_slurm(env, "4090node2")
        for changed, hostname in [({}, "4090node2"), (env, "4090node1.domain"),
                                  ({**env, "CUDA_VISIBLE_DEVICES": ""}, "4090node2"),
                                  ({**env, "MUJOCO_GL": "glfw"}, "4090node2")]:
            with self.assertRaises(RuntimeError):
                require_slurm(changed, hostname)

    def test_detects_lost_override_after_state_restore(self):
        wanted = {"agentview": {"pos": [.6, 0, 1.6], "quat": [.6, .3, .3, .6]}}
        stages = {"reset": wanted, "set_init_state": wanted, "after_settling": wanted}
        verify_camera_poses(wanted, stages)
        stages["set_init_state"] = {"agentview": {"pos": [.65, 0, 1.6], "quat": [.6, .3, .3, .6]}}
        with self.assertRaises(RuntimeError):
            verify_camera_poses(wanted, stages)


if __name__ == "__main__":
    unittest.main()
