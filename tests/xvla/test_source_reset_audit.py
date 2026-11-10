import unittest

import numpy as np

from xvla_rlinf.source_reset_audit import BOWL, _screen_perturbation, relative_geometry


class SourceResetAuditTests(unittest.TestCase):
    def test_relative_geometry_uses_plate_to_ramekin_world_axis(self):
        plate = np.array([1.0, 2.0, 0.9])
        ramekin = plate + np.array([0.0, 0.2, 0.0])
        bowl = plate + np.array([-0.01, 0.08, 0.0])
        geometry = relative_geometry(bowl, plate, ramekin)
        self.assertAlmostEqual(geometry["bowl_along_fraction"], 0.4)
        self.assertAlmostEqual(geometry["bowl_cross_m"], 0.01)
        with self.assertRaisesRegex(ValueError, "degenerate"):
            relative_geometry(bowl, plate, plate)

    def test_physics_screen_rejects_contact_and_shift_loss(self):
        def snapshot(along, contacts=()):
            return {"object_positions_m": {BOWL: [along, 0.0, 0.9]}, "bowl_along_fraction": 4 * along,
                    "bowl_along_m": along, "bowl_contacts": contacts, "bowl_linear_speed_m_per_s": 0.0}

        original = snapshot(0.08)
        moved = snapshot(0.09)
        baseline = {"unmodified": original, "before_settling": original, "after_settling": original,
                    "success_during_reset": False}
        perturbed = {"unmodified": original, "before_settling": moved, "after_settling": moved,
                     "bowl_settling_delta_xyz_m": [0.0, 0.0, 0.0], "success_during_reset": False}
        self.assertEqual(_screen_perturbation(baseline, perturbed, 0.01), [])
        perturbed["after_settling"] = snapshot(0.08, [{"other_geom": "g", "other_object": "plate_1"}])
        self.assertCountEqual(_screen_perturbation(baseline, perturbed, 0.01), [
            "bowl_contact_with_tracked_object", "new_bowl_contact_geometry", "requested_shift_not_retained",
        ])
