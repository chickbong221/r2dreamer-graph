"""Live goal geometry: the readers that turn an environment into a SiteSpec.

Fake environments, real geometry. What is under test is the arithmetic and the
failure behaviour -- where the hole mouth lands, which tolerance is used, when
the box-depth cache re-reads, and what happens when the environment does not
expose what a declaration asked for.
"""

import unittest

import numpy as np

from scenegraph.adapters import site_providers as sp
from scenegraph.core.sites import (
    PROVIDER_PEG_HOLE_MOUTH,
    PROVIDER_PICK_CUBE_GOAL,
    PROVIDER_ROBOT_BASE_REGION,
    SiteDeclaration,
    SiteError,
    reached_holds,
    site_distance,
)
from scenegraph.adapters.site_providers import site_specs


class _Pose:
    def __init__(self, p, q=(1.0, 0.0, 0.0, 0.0)):
        self.raw_pose = np.array([[*p, *q]], dtype=float)


class _Shape:
    def __init__(self, half):
        self.half_size = np.asarray(half, dtype=float)


class _Actor:
    def __init__(self, shapes=None, pose=None):
        self.pose = pose
        self._shapes = shapes

        class _Obj:
            def __init__(inner, shapes):
                inner._shapes = shapes

            def get_collision_shapes(inner):
                return inner._shapes

        self._objs = [_Obj(shapes)] if shapes is not None else []


def _peg_env(depth=0.11, radius=0.018, hole=(0.0, 0.3, 0.10),
             head=(-0.25, 0.3, 0.10), blocks=None):
    """A PegInsertionSide-shaped environment.

    Defaults mirror one recorded episode: the hole frame on the box centre
    plane, the peg approaching along -x from outside.
    """
    shapes = blocks if blocks is not None else [
        _Shape([depth, 0.02, 0.05]) for _ in range(4)]

    class Env:
        box_hole_pose = _Pose(hole)
        box_hole_radii = np.array([radius])
        peg_head_pose = _Pose(head)
        peg_half_sizes = np.array([[depth, 0.015, 0.015]])
        box = _Actor(shapes=shapes)

    return Env()


def _decl(key="spatial:hole_site", subject="actor:peg", site_type="surface",
          metric="euclidean", source="provider",
          provider=PROVIDER_PEG_HOLE_MOUTH):
    return SiteDeclaration(
        key=key, site_type=site_type, subject_key=subject, metric=metric,
        source=source, provider=provider, provenance="test",
    )


class PegHoleMouthTest(unittest.TestCase):

    def setUp(self):
        sp.reset_depth_cache()

    def test_the_mouth_sits_a_half_depth_before_the_hole_frame(self):
        """box_hole_pose is on the box centre plane -- its x offset is
        identically zero in all 300 recorded episodes -- so the entrance is a
        half-depth back along the hole axis."""
        spec = sp.peg_hole_mouth(_peg_env(depth=0.11), 0, _decl())
        np.testing.assert_allclose(spec.pose_world[:3], [-0.11, 0.3, 0.10])

    def test_the_mouth_is_not_the_box_centre(self):
        """Targeting the hole frame would put the milestone inside the box and
        fire it only once the peg was already half inserted."""
        spec = sp.peg_hole_mouth(_peg_env(depth=0.11), 0, _decl())
        self.assertNotAlmostEqual(float(spec.pose_world[0]), 0.0)

    def test_a_deeper_box_moves_the_mouth_further_out(self):
        shallow = sp.peg_hole_mouth(_peg_env(depth=0.085), 0, _decl())
        sp.reset_depth_cache()
        deep = sp.peg_hole_mouth(_peg_env(depth=0.125), 0, _decl())
        self.assertLess(float(deep.pose_world[0]), float(shallow.pose_world[0]))

    def test_the_tolerance_is_the_live_aperture(self):
        spec = sp.peg_hole_mouth(_peg_env(radius=0.0227), 0, _decl())
        self.assertAlmostEqual(spec.tolerance, 0.0227)

    def test_the_source_point_is_the_peg_head(self):
        spec = sp.peg_hole_mouth(_peg_env(head=(-0.25, 0.3, 0.10)), 0, _decl())
        np.testing.assert_allclose(spec.subject_point_world, [-0.25, 0.3, 0.10])

    def test_distance_is_head_to_mouth_not_origin_to_box(self):
        spec = sp.peg_hole_mouth(
            _peg_env(depth=0.11, head=(-0.25, 0.3, 0.10)), 0, _decl())
        self.assertAlmostEqual(site_distance(spec, None), 0.14)

    def test_approaching_the_mouth_shortens_the_distance(self):
        far = sp.peg_hole_mouth(_peg_env(head=(-0.30, 0.3, 0.10)), 0, _decl())
        near = sp.peg_hole_mouth(_peg_env(head=(-0.13, 0.3, 0.10)), 0, _decl())
        self.assertLess(site_distance(near, None), site_distance(far, None))

    def test_reached_fires_inside_the_aperture(self):
        at_mouth = _peg_env(depth=0.11, radius=0.018, head=(-0.111, 0.3, 0.10))
        self.assertTrue(reached_holds(
            sp.peg_hole_mouth(at_mouth, 0, _decl()), None))

    def test_moving_the_box_origin_does_not_move_the_mouth_relative_to_the_hole(self):
        """The mouth is derived from the live hole frame, so a box that slid
        keeps the same head-to-mouth distance for a head that slid with it."""
        a = sp.peg_hole_mouth(
            _peg_env(hole=(0.0, 0.3, 0.10), head=(-0.25, 0.3, 0.10)),
            0, _decl())
        sp.reset_depth_cache()
        b = sp.peg_hole_mouth(
            _peg_env(hole=(0.4, 0.7, 0.10), head=(0.15, 0.7, 0.10)),
            0, _decl())
        self.assertAlmostEqual(site_distance(a, None), site_distance(b, None))


class BoxDepthCacheTest(unittest.TestCase):

    def setUp(self):
        sp.reset_depth_cache()

    def test_the_depth_comes_from_collision_geometry(self):
        sp.peg_hole_mouth(_peg_env(), 0, _decl())
        self.assertIn("collision", sp.depth_provenance(0))

    def test_it_falls_back_when_there_is_no_collision_geometry(self):
        env = _peg_env(depth=0.11)
        env.box = _Actor(shapes=None)
        spec = sp.peg_hole_mouth(env, 0, _decl())
        np.testing.assert_allclose(spec.pose_world[:3], [-0.11, 0.3, 0.10])
        self.assertIn("peg_half_sizes", sp.depth_provenance(0))

    def test_a_fallback_that_disagrees_with_the_geometry_raises(self):
        """Upstream draws the box depth and the peg half-length from one
        sample. If that ever stops being true, the mouth would land on the
        wrong plane -- so it fails rather than picking one."""
        env = _peg_env(depth=0.11)
        env.box = _Actor(shapes=[_Shape([0.20, 0.02, 0.05]) for _ in range(4)])
        with self.assertRaises(SiteError) as ctx:
            sp.peg_hole_mouth(env, 0, _decl())
        self.assertIn("half-depth", str(ctx.exception))

    def test_blocks_that_disagree_with_each_other_raise(self):
        blocks = [_Shape([0.11, 0.02, 0.05]) for _ in range(3)]
        blocks.append(_Shape([0.09, 0.02, 0.05]))
        with self.assertRaises(SiteError) as ctx:
            sp.peg_hole_mouth(_peg_env(blocks=blocks), 0, _decl())
        self.assertIn("x half-extent", str(ctx.exception))

    def test_re_randomized_geometry_invalidates_the_cache(self):
        """PegInsertionSide reconfigures on every reset at num_envs=1, so a
        cache that did not notice would place the mouth using the previous
        episode's box."""
        first = sp.peg_hole_mouth(_peg_env(depth=0.085, radius=0.018),
                                  0, _decl())
        second = sp.peg_hole_mouth(_peg_env(depth=0.125, radius=0.0227),
                                   0, _decl())
        self.assertAlmostEqual(float(first.pose_world[0]), -0.085)
        self.assertAlmostEqual(float(second.pose_world[0]), -0.125)

    def test_unchanged_geometry_reuses_the_cached_depth(self):
        env = _peg_env(depth=0.11)
        sp.peg_hole_mouth(env, 0, _decl())
        env.box = _Actor(shapes=None)  # geometry no longer readable
        spec = sp.peg_hole_mouth(env, 0, _decl())
        # Still the cached value, because the signature did not change.
        np.testing.assert_allclose(spec.pose_world[:3], [-0.11, 0.3, 0.10])
        self.assertIn("collision", sp.depth_provenance(0))

    def test_sub_scenes_keep_separate_entries(self):
        sp.peg_hole_mouth(_peg_env(depth=0.11), 0, _decl())
        self.assertIsNone(sp.depth_provenance(1))


class PickCubeGoalTest(unittest.TestCase):

    def _env(self, thresh=0.025, goal=(0.1, 0.2, 0.3)):
        class Env:
            goal_site = _Actor(pose=_Pose(goal))
            goal_thresh = np.array([thresh])
        return Env()

    def _decl(self):
        return _decl(key="actor:goal_site", subject="actor:cube",
                     site_type="point", source="origin",
                     provider=PROVIDER_PICK_CUBE_GOAL)

    def test_it_reads_the_live_goal_pose_and_threshold(self):
        spec = sp.pick_cube_goal(self._env(), 0, self._decl())
        np.testing.assert_allclose(spec.pose_world[:3], [0.1, 0.2, 0.3])
        self.assertAlmostEqual(spec.tolerance, 0.025)

    def test_the_threshold_is_not_hardcoded(self):
        """A task configured with a different goal_thresh must be scored
        against its own."""
        spec = sp.pick_cube_goal(self._env(thresh=0.05), 0, self._decl())
        self.assertAlmostEqual(spec.tolerance, 0.05)

    def test_reached_uses_full_3d_distance(self):
        spec = sp.pick_cube_goal(self._env(goal=(0.0, 0.0, 0.0)), 0,
                                 self._decl())
        cube_above = [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
        self.assertFalse(reached_holds(spec, cube_above))

    def test_a_missing_goal_site_raises(self):
        class Env:
            goal_thresh = np.array([0.025])
        with self.assertRaises(SiteError):
            sp.pick_cube_goal(Env(), 0, self._decl())


class RobotBaseRegionTest(unittest.TestCase):

    def _env(self, base=(0.0, 0.0, 0.0)):
        class Robot:
            def get_links(self):
                return [_Actor(pose=_Pose(base)), _Actor(pose=_Pose((9, 9, 9)))]

        class Agent:
            robot = Robot()

        class Env:
            agent = Agent()
        return Env()

    def _decl(self):
        return _decl(key="spatial:pull_goal_region", subject="actor:cube",
                     site_type="region", metric="planar", source="origin",
                     provider=PROVIDER_ROBOT_BASE_REGION)

    def test_the_region_is_centred_on_the_base_link(self):
        spec = sp.robot_base_region(self._env(base=(0.3, -0.2, 0.0)), 0,
                                    self._decl())
        np.testing.assert_allclose(spec.pose_world[:3], [0.3, -0.2, 0.0])

    def test_the_radius_is_the_success_radius(self):
        self.assertAlmostEqual(sp.PULL_SUCCESS_RADIUS, 0.6)
        spec = sp.robot_base_region(self._env(), 0, self._decl())
        self.assertAlmostEqual(spec.tolerance, 0.6)

    def test_reached_matches_the_environment_predicate_at_the_boundary(self):
        """PullCubeTool evaluates ``< 0.6``, so 0.6 exactly is not success."""
        spec = sp.robot_base_region(self._env(), 0, self._decl())
        self.assertFalse(reached_holds(spec, [0.6, 0.0, 0.0, 1, 0, 0, 0]))
        self.assertTrue(reached_holds(spec, [0.5999, 0.0, 0.0, 1, 0, 0, 0]))

    def test_height_is_ignored(self):
        """A cube on the floor and one on the table are equally pulled in."""
        spec = sp.robot_base_region(self._env(), 0, self._decl())
        self.assertTrue(reached_holds(spec, [0.5, 0.0, 5.0, 1, 0, 0, 0]))

    def test_distance_is_unclamped_inside_the_region(self):
        """The ladder needs to keep resolving once the cube is inside; a hinge
        clamped at the boundary would flatten the last stretch."""
        spec = sp.robot_base_region(self._env(), 0, self._decl())
        self.assertAlmostEqual(
            site_distance(spec, [0.2, 0.0, 0.0, 1, 0, 0, 0]), 0.2)

    def test_a_robot_with_no_links_raises(self):
        class Robot:
            def get_links(self):
                return []

        class Agent:
            robot = Robot()

        class Env:
            agent = Agent()
        with self.assertRaises(SiteError):
            sp.robot_base_region(Env(), 0, self._decl())


class DispatchTest(unittest.TestCase):

    def setUp(self):
        sp.reset_depth_cache()

    def test_declarations_are_served_in_stable_key_order(self):
        env = _peg_env()
        decls = {"spatial:hole_site": _decl()}
        specs = site_specs(env, 0, decls)
        self.assertEqual([s.key for s in specs], ["spatial:hole_site"])

    def test_an_unimplemented_provider_raises(self):
        decl = SiteDeclaration(
            key="spatial:x", site_type="point", subject_key="actor:cube",
            metric="euclidean", source="origin", provider="pick_cube_goal",
            provenance="p",
        )
        object.__setattr__(decl, "provider", "not_implemented")
        with self.assertRaises(SiteError):
            site_specs(_peg_env(), 0, {"spatial:x": decl})

    def test_a_site_the_environment_cannot_serve_raises(self):
        """Skipping it would drop a scored fact, and a missing scored fact
        masks the whole frame's potential -- a much quieter failure."""
        class Env:
            pass
        with self.assertRaises(SiteError):
            site_specs(Env(), 0, {"spatial:hole_site": _decl()})

    def test_no_declarations_means_no_specs(self):
        self.assertEqual(site_specs(_peg_env(), 0, {}), [])


if __name__ == "__main__":
    unittest.main()
