"""Object-to-camera coverage, the MS-HAB relation gate.

No simulator: cameras are matrices and entities are stubs holding collision
shapes. The point is the geometry, and the geometry is where this can be wrong
in ways that look plausible -- a sign flip behind the camera, a compound body
whose off-origin parts get ignored, an object spanning the whole image with no
corner inside it.
"""

import unittest

import numpy as np

from scenegraph.adapters.camera_projection import (
    CameraCoverage,
    box_covers_image,
    corners_world,
    local_aabb_corners,
)


W, H = 128, 128
# 90-degree field of view, principal point at the centre.
K = np.array([[64.0, 0.0, 64.0], [0.0, 64.0, 64.0], [0.0, 0.0, 1.0]])
# Camera at the origin looking down +z, world axes aligned.
EYE = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])


class _Geom:
    def __init__(self, **kw):
        for key, value in kw.items():
            setattr(self, key, value)


class _Pose:
    def __init__(self, p=(0.0, 0.0, 0.0), q=(1.0, 0.0, 0.0, 0.0)):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)


class _Shape:
    """Exposes ``local_pose`` as a property, as some SAPIEN builds do."""

    def __init__(self, geometry, local_pose=None):
        self.geometry = geometry
        self.local_pose = local_pose


class _GetterShape:
    """Exposes it as ``get_local_pose()``, as other builds do."""

    def __init__(self, geometry, local_pose=None):
        self.geometry = geometry
        self._pose = local_pose

    def get_local_pose(self):
        return self._pose


class _Body:
    def __init__(self, shapes):
        self._shapes = list(shapes)

    def get_collision_shapes(self):
        return self._shapes


class _Obj:
    """One env row of a ManiSkill Actor wrapper."""

    def __init__(self, shapes):
        self.components = [_Body(shapes)]


class _Entity:
    def __init__(self, shapes):
        self._objs = [_Obj(shapes)]
        self._scene_idxs = [0]


def _box(half, at=None):
    pose = _Pose(at) if at is not None else None
    return _Shape(_Geom(half_size=np.asarray(half, dtype=float)), pose)


def _identity_pose(xyz):
    return [*xyz, 1.0, 0.0, 0.0, 0.0]


class LocalAabbTest(unittest.TestCase):
    def test_a_single_box_gives_its_own_corners(self):
        corners = local_aabb_corners(_Entity([_box((0.1, 0.2, 0.3))]))
        np.testing.assert_allclose(corners.min(0), [-0.1, -0.2, -0.3])
        np.testing.assert_allclose(corners.max(0), [0.1, 0.2, 0.3])
        self.assertEqual(corners.shape, (8, 3))

    def test_compound_bodies_include_each_shape_offset(self):
        """Ignoring the shape's own transform shrinks a compound body to
        whichever part happens to sit on the origin."""
        entity = _Entity([_box((0.05, 0.05, 0.05)),
                          _box((0.05, 0.05, 0.05), at=(0.5, 0.0, 0.0))])
        corners = local_aabb_corners(entity)
        self.assertAlmostEqual(corners.max(0)[0], 0.55)
        self.assertAlmostEqual(corners.min(0)[0], -0.05)

    def test_a_rotated_shape_widens_the_box(self):
        rot45 = _Pose(q=(np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)))
        flat = _Shape(_Geom(half_size=np.array([0.4, 0.05, 0.05])), rot45)
        corners = local_aabb_corners(_Entity([flat]))
        self.assertGreater(corners.max(0)[1], 0.2)

    def test_spheres_and_capsules_are_handled(self):
        corners = local_aabb_corners(_Entity([_Shape(_Geom(radius=0.25))]))
        np.testing.assert_allclose(corners.max(0), [0.25, 0.25, 0.25])

    def test_meshes_use_their_vertex_extent(self):
        verts = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.2]])
        corners = local_aabb_corners(_Entity([_Shape(_Geom(vertices=verts))]))
        np.testing.assert_allclose(corners.max(0), [0.3, 0.1, 0.2])

    def test_no_geometry_returns_none(self):
        self.assertIsNone(local_aabb_corners(_Entity([])))

    def test_a_rotated_mesh_is_bounded_by_all_eight_corners(self):
        """Reducing a mesh to its two diagonal extremes and rotating those
        bounds a line through the shape, not the shape."""
        verts = np.array([[-0.4, -0.05, -0.05], [0.4, 0.05, 0.05]])
        rot90 = _Pose(q=(np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)))
        corners = local_aabb_corners(
            _Entity([_Shape(_Geom(vertices=verts), rot90)]))
        # A 0.8-long bar turned a quarter turn about z is 0.8 wide in y.
        self.assertAlmostEqual(corners.max(0)[1], 0.4, places=6)
        self.assertAlmostEqual(corners.max(0)[0], 0.05, places=6)

    def test_the_getter_form_of_local_pose_is_honoured(self):
        """SAPIEN has exposed this both ways; missing one silently drops every
        compound offset."""
        entity = _Entity([
            _GetterShape(_Geom(half_size=np.array([0.05, 0.05, 0.05])),
                         _Pose((0.5, 0.0, 0.0))),
        ])
        self.assertAlmostEqual(local_aabb_corners(entity).max(0)[0], 0.55)


class CoverageTest(unittest.TestCase):
    def _corners(self, at, half=(0.05, 0.05, 0.05)):
        return corners_world(local_aabb_corners(_Entity([_box(half)])),
                             _identity_pose(at))

    def test_an_object_in_front_of_the_camera_is_covered(self):
        self.assertTrue(box_covers_image(self._corners((0, 0, 1.0)), EYE, K, W, H))

    def test_an_object_behind_the_camera_is_not(self):
        """The case a naive projection gets wrong: negative depth flips the
        sign and would place it in the middle of the frame."""
        self.assertFalse(box_covers_image(self._corners((0, 0, -1.0)), EYE, K, W, H))

    def test_an_object_off_to_the_side_is_not_covered(self):
        self.assertFalse(box_covers_image(self._corners((5.0, 0, 1.0)), EYE, K, W, H))

    def test_an_object_spanning_the_whole_image_is_covered(self):
        """No corner projects inside the frame, but the box still fills it.
        Testing corners alone would call this invisible."""
        corners = self._corners((0, 0, 1.0), half=(4.0, 4.0, 0.05))
        self.assertTrue(box_covers_image(corners, EYE, K, W, H))

    def test_an_object_straddling_the_near_plane_is_covered(self):
        corners = self._corners((0, 0, 0.0), half=(0.05, 0.05, 0.5))
        self.assertTrue(box_covers_image(corners, EYE, K, W, H))

    def test_a_translated_camera_sees_a_different_object(self):
        """Extrinsics are recomputed every frame precisely because both Fetch
        cameras move in the world frame."""
        moved = np.array([[1.0, 0, 0, -5.0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
        corners = self._corners((5.0, 0, 1.0))
        self.assertFalse(box_covers_image(corners, EYE, K, W, H))
        self.assertTrue(box_covers_image(corners, moved, K, W, H))


class _Camera:
    def __init__(self, extrinsic, width=W, height=H):
        self._ext = np.asarray(extrinsic, dtype=float)
        self.width, self.height = width, height
        self.extrinsic_reads = 0

    def get_intrinsic_matrix(self):
        return K[None]

    def get_extrinsic_matrix(self):
        self.extrinsic_reads += 1
        return self._ext[None]


class _Scene:
    def __init__(self, sensors):
        self.sensors = sensors


class _Env:
    def __init__(self, sensors):
        self.scene = _Scene(sensors)
        self.unwrapped = self


class CameraCoverageTest(unittest.TestCase):
    def _coverage(self, **cams):
        return CameraCoverage(_Env(cams)), cams

    def test_either_camera_is_enough(self):
        far = np.array([[1.0, 0, 0, -50.0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
        coverage, _ = self._coverage(head=_Camera(far), hand=_Camera(EYE))
        entity = _Entity([_box((0.05, 0.05, 0.05))])
        self.assertTrue(coverage.covers(entity, 0, _identity_pose((0, 0, 1.0)), 0))

    def test_neither_camera_means_out_of_frame(self):
        far = np.array([[1.0, 0, 0, -50.0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]])
        coverage, _ = self._coverage(head=_Camera(far), hand=_Camera(far))
        entity = _Entity([_box((0.05, 0.05, 0.05))])
        self.assertFalse(coverage.covers(entity, 0, _identity_pose((0, 0, 1.0)), 0))

    def test_intrinsics_are_cached_and_extrinsics_are_not(self):
        """Fetch drives its base and pans its head, so a cached extrinsic
        would freeze both cameras where they were on the first frame."""
        coverage, cams = self._coverage(head=_Camera(EYE))
        entity = _Entity([_box((0.05, 0.05, 0.05))])
        for _ in range(3):
            coverage.covers(entity, 0, _identity_pose((0, 0, 1.0)), 0)
        self.assertEqual(cams["head"].extrinsic_reads, 3)

    def test_the_aabb_cache_is_reused_then_dropped_on_reconfigure(self):
        coverage, _ = self._coverage(head=_Camera(EYE))
        entity = _Entity([_box((0.05, 0.05, 0.05))])
        coverage.covers(entity, 0, _identity_pose((0, 0, 1.0)), 0)
        self.assertEqual(len(coverage._aabb), 1)
        coverage.invalidate()
        self.assertEqual(len(coverage._aabb), 0)

    def test_geometryless_entity_projects_as_its_centroid(self):
        """Assuming True would make "no collision geometry" mean "always
        relational", which is not a camera test at all."""
        coverage, _ = self._coverage(head=_Camera(EYE))
        bare = _Entity([])
        self.assertTrue(
            coverage.covers(bare, 0, _identity_pose((0.0, 0.0, 2.0)), 0))
        self.assertFalse(
            coverage.covers(bare, 0, _identity_pose((9.0, 0.0, 2.0)), 0))

    def test_camera_dimensions_come_from_the_camera(self):
        """2 * principal point is only the size for a centred principal
        point, so it is the last resort rather than the rule."""
        cam = _Camera(EYE)
        cam.get_width = lambda: 32
        cam.get_height = lambda: 32
        coverage = CameraCoverage(_Env({"head": cam}))
        entity = _Entity([_box((0.02, 0.02, 0.02))])
        # x = 0.35 at z = 1 lands near u = 86: inside 128, outside 32.
        self.assertFalse(
            coverage.covers(entity, 0, _identity_pose((0.35, 0.0, 1.0)), 0))

    def test_moving_the_object_changes_coverage(self):
        coverage, _ = self._coverage(head=_Camera(EYE))
        entity = _Entity([_box((0.05, 0.05, 0.05))])
        self.assertTrue(coverage.covers(entity, 0, _identity_pose((0, 0, 1.0)), 0))
        self.assertFalse(coverage.covers(entity, 0, _identity_pose((9.0, 0, 1.0)), 0))


if __name__ == "__main__":
    unittest.main()
