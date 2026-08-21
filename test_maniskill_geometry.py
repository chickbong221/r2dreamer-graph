"""Containment features and CPU contact geometry (step 11)."""

import math
import unittest
from types import SimpleNamespace

import numpy as np

from scenegraph.adapters.contact_geometry import (
    contact_anchor, directions_to_local, paired_contact_frame,
    pairwise_contact_points, spherical_radius, symmetry_of, to_local,
)
from scenegraph.adapters.maniskill_containment import (
    CAPABILITY_PEG, CAPABILITY_SLOT, containment_features, detect_capability,
    peg_features, slot_features,
)


def _qmul(a, b):
    w1, x1, y1, z1 = a[0]
    w2, x2, y2, z2 = b[0]
    return np.array([[
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]])


class _Pose:
    """Batched ManiSkill Pose stand-in supporting inv() and composition."""

    def __init__(self, p, q):
        self.p = np.asarray(p, dtype=float).reshape(-1, 3)
        self.q = np.asarray(q, dtype=float).reshape(-1, 4)

    @property
    def raw_pose(self):
        return np.concatenate([self.p, self.q], axis=1)

    def inv(self):
        return _Pose(-self.p, self.q * np.array([1.0, -1, -1, -1]))

    def __mul__(self, other):
        return _Pose(other.p + self.p, _qmul(self.q, other.q))


_ID = [[1.0, 0.0, 0.0, 0.0]]


def _peg_env(head=(0.05, 0.0, 0.0), radius=0.02, quat=_ID):
    base = SimpleNamespace(
        box_hole_pose=_Pose([[0.3, 0.1, 0.2]], _ID),
        box_hole_radii=np.array([radius]),
        peg_head_pose=_Pose([[0.3 + head[0], 0.1 + head[1], 0.2 + head[2]]],
                            quat),
        peg_half_sizes=np.array([[0.1, radius, radius]]),
    )
    return SimpleNamespace(unwrapped=base)


def _slot_env(offset=(0.0, 0.0, 0.0), quat=_ID):
    base = SimpleNamespace(
        goal_pose=_Pose([[0.1, 0.2, 0.3]], _ID),
        charger=SimpleNamespace(pose=_Pose(
            [[0.1 + offset[0], 0.2 + offset[1], 0.3 + offset[2]]], quat)),
        receptacle=SimpleNamespace(pose=_Pose([[0.1, 0.2, 0.3]], _ID)),
        _peg_size=np.array([0.02, 0.01, 0.005]),
        _base_size=np.array([0.03, 0.02, 0.01]),
    )
    return SimpleNamespace(unwrapped=base)


class CapabilityTest(unittest.TestCase):
    def test_peg_detected(self):
        self.assertEqual(detect_capability(_peg_env()), CAPABILITY_PEG)

    def test_slot_detected(self):
        self.assertEqual(detect_capability(_slot_env()), CAPABILITY_SLOT)

    def test_peg_wins_when_goal_pose_is_also_present(self):
        """PegInsertionSide defines goal_pose too, so order is load-bearing."""
        env = _peg_env()
        env.unwrapped.goal_pose = _Pose([[0.0, 0, 0]], _ID)
        env.unwrapped.charger = object()
        env.unwrapped.receptacle = object()
        self.assertEqual(detect_capability(env), CAPABILITY_PEG)

    def test_plain_task_has_no_capability(self):
        env = SimpleNamespace(unwrapped=SimpleNamespace(cube=object()))
        self.assertIsNone(detect_capability(env))
        self.assertIsNone(containment_features(env))


class PegFeatureTest(unittest.TestCase):
    def test_hole_offset_vanishes_in_the_hole_frame(self):
        a = peg_features(_peg_env())
        env_b = _peg_env()
        env_b.unwrapped.box_hole_pose = _Pose([[-0.9, 0.4, 0.05]], _ID)
        env_b.unwrapped.peg_head_pose = _Pose([[-0.9 + 0.05, 0.4, 0.05]], _ID)
        b = peg_features(env_b)
        self.assertAlmostEqual(a["axial"], b["axial"])
        self.assertAlmostEqual(a["lateral"], b["lateral"])

    def test_radius_variation_does_not_vanish(self):
        wide = peg_features(_peg_env(head=(0.05, 0.015, 0.0), radius=0.02))
        tight = peg_features(_peg_env(head=(0.05, 0.015, 0.0), radius=0.01))
        self.assertLess(wide["lateral_norm"], 1.0)
        self.assertGreater(tight["lateral_norm"], 1.0)

    def test_square_opening_uses_max_of_y_and_z(self):
        f = peg_features(_peg_env(head=(0.05, 0.012, -0.018), radius=0.02))
        self.assertAlmostEqual(f["lateral"], 0.018)

    def test_inserted_matches_the_env_criterion(self):
        self.assertTrue(peg_features(_peg_env(head=(0.05, 0.0, 0.0)))["holds"])
        self.assertFalse(
            peg_features(_peg_env(head=(0.05, 0.03, 0.0)))["holds"])
        self.assertFalse(
            peg_features(_peg_env(head=(-0.05, 0.0, 0.0)))["holds"])

    def test_edge_of_the_mouth_still_counts(self):
        self.assertTrue(peg_features(_peg_env(head=(-0.014, 0, 0)))["holds"])

    def test_container_and_containee_keys(self):
        f = peg_features(_peg_env())
        self.assertEqual(f["container_key"], "actor:box_with_hole")
        self.assertEqual(f["containee_key"], "actor:peg")


class SlotFeatureTest(unittest.TestCase):
    def test_seated_charger_holds(self):
        self.assertTrue(slot_features(_slot_env())["holds"])

    def test_offset_beyond_tolerance_does_not_hold(self):
        self.assertFalse(slot_features(_slot_env(offset=(0.02, 0, 0)))["holds"])

    def test_rotation_beyond_tolerance_does_not_hold(self):
        half = 0.3 / 2.0
        quat = [[math.cos(half), 0.0, 0.0, math.sin(half)]]
        self.assertFalse(slot_features(_slot_env(quat=quat))["holds"])

    def test_dimensions_are_recorded_for_future_randomization(self):
        f = slot_features(_slot_env())
        self.assertEqual(len(f["containee_dims"]), 3)
        self.assertEqual(f["container_key"], "actor:receptacle")


class ContactGeometryTest(unittest.TestCase):
    def test_world_to_local_round_trip(self):
        pose = [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]
        pts = np.array([[1.5, 2.0, 3.0]])
        self.assertTrue(np.allclose(to_local(pts, pose), [[0.5, 0.0, 0.0]]))

    def test_directions_ignore_translation(self):
        pose = [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0]
        d = directions_to_local(np.array([[0.0, 0.0, 1.0]]), pose)
        self.assertTrue(np.allclose(d, [[0.0, 0.0, 1.0]]))

    def test_anchor_is_impulse_weighted(self):
        points = {
            "positions": np.array([[0.0, 0, 0], [1.0, 0, 0]]),
            "normals": np.array([[0.0, 0, 1.0], [0.0, 0, 1.0]]),
            "impulses": np.array([[0.0, 0, 9.0], [0.0, 0, 1.0]]),
        }
        anchor, normal = contact_anchor(points)
        self.assertAlmostEqual(anchor[0], 0.1)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0)

    def test_zero_impulse_falls_back_to_uniform(self):
        points = {
            "positions": np.array([[0.0, 0, 0], [2.0, 0, 0]]),
            "normals": np.array([[1.0, 0, 0], [1.0, 0, 0]]),
            "impulses": np.zeros((2, 3)),
        }
        anchor, _ = contact_anchor(points)
        self.assertAlmostEqual(anchor[0], 1.0)

    def test_no_contact_api_returns_none(self):
        scene = SimpleNamespace(px=None)
        self.assertIsNone(pairwise_contact_points(scene, object(), object()))

    def test_paired_frame_gives_both_endpoints(self):
        pt = SimpleNamespace(position=[1.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0],
                             impulse=[0.0, 0.0, 1.0])
        ea, eb = object(), object()
        contact = SimpleNamespace(
            bodies=[SimpleNamespace(entity=ea), SimpleNamespace(entity=eb)],
            points=[pt])
        scene = SimpleNamespace(px=SimpleNamespace(
            get_contacts=lambda: [contact]))
        a = SimpleNamespace(_objs=[ea])
        b = SimpleNamespace(_objs=[eb])
        out = paired_contact_frame(
            scene, a, b,
            [0.0, 0, 0, 1.0, 0, 0, 0], [2.0, 0, 0, 1.0, 0, 0, 0])
        self.assertEqual(out["anchor_a_local"], [1.0, 0.0, 0.0])
        self.assertEqual(out["anchor_b_local"], [-1.0, 0.0, 0.0])
        self.assertEqual(out["normal_b_local"], [0.0, 0.0, -1.0])
        self.assertEqual(out["anchor_source"], "contact_points")

    def test_normal_sign_is_corrected_when_the_pair_is_reversed(self):
        pt = SimpleNamespace(position=[1.0, 0.0, 0.0], normal=[0.0, 0.0, 1.0],
                             impulse=[0.0, 0.0, 1.0])
        ea, eb = object(), object()
        contact = SimpleNamespace(
            bodies=[SimpleNamespace(entity=eb), SimpleNamespace(entity=ea)],
            points=[pt])
        scene = SimpleNamespace(px=SimpleNamespace(
            get_contacts=lambda: [contact]))
        got = pairwise_contact_points(
            scene, SimpleNamespace(_objs=[ea]), SimpleNamespace(_objs=[eb]))
        self.assertTrue(np.allclose(got["normals"], [[0.0, 0.0, -1.0]]))


class SymmetryTest(unittest.TestCase):
    def _entity(self, shapes):
        comp = SimpleNamespace(get_collision_shapes=lambda: shapes)
        return SimpleNamespace(_objs=[SimpleNamespace(components=[comp])])

    def test_sphere_detected_with_radius(self):
        sphere = type("PhysxCollisionShapeSphere", (), {"radius": 0.02})()
        self.assertAlmostEqual(spherical_radius(self._entity([sphere])), 0.02)
        sym = symmetry_of(self._entity([sphere]))
        self.assertEqual(sym["symmetry"], "spherical")
        self.assertTrue(sym["orientation_invariant"])

    def test_box_is_not_spherical(self):
        box = type("PhysxCollisionShapeBox", (), {"half_size": [1, 1, 1]})()
        self.assertIsNone(spherical_radius(self._entity([box])))
        self.assertEqual(symmetry_of(self._entity([box]))["symmetry"], "none")

    def test_compound_shape_is_not_spherical(self):
        sphere = type("PhysxCollisionShapeSphere", (), {"radius": 0.02})()
        box = type("PhysxCollisionShapeBox", (), {})()
        self.assertIsNone(spherical_radius(self._entity([sphere, box])))


if __name__ == "__main__":
    unittest.main()
