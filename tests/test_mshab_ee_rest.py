"""The MS-HAB end-effector rest position, and the scale it is measured on.

MS-HAB Pick ends with the gripper back at a rest position defined relative to
the robot base, still holding the object. It is the one site whose subject is
the end effector rather than a manipuland -- the task provides no object
destination at all -- and both the runtime provider and the calibration
collector have to read it from the same place, or the mined scale describes a
different point than the labels do.

No simulator here: the provider reads three attributes and composes a pose, so
a stand-in environment exercises every branch.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from scenegraph.adapters.site_providers import ee_rest_geometry, mshab_ee_rest
from scenegraph.core.sites import (
    METRIC_EUCLIDEAN,
    PROVIDER_MSHAB_EE_REST,
    PROVIDERS,
    SITE_EE_REST,
    SITE_POINT,
    SOURCE_ORIGIN,
    SiteDeclaration,
    SiteError,
)
from scenegraph.core.spatial_metrics import (
    EE_SITE_HEIGHT_KEY,
    EE_SITE_PLANAR_KEY,
    EE_SITE_SCOPE,
    SPATIAL_SCOPES,
    stat_key,
)


class _Pose:
    def __init__(self, p, q):
        self.p, self.q = np.asarray(p, float), np.asarray(q, float)


class _Link:
    def __init__(self, pose):
        self.pose = pose


class _PickCfg:
    def __init__(self, thresh):
        self.ee_rest_thresh = thresh


class _Env:
    """The three things the rest position is composed from, per sub-scene.

    ``ee_rest_pos_wrt_base`` is a Pose, not an XYZ array: MS-HAB builds it
    with ``Pose.create_from_pq`` and composes it against the base pose. An
    array stand-in here passed while the real attribute raised a bare
    TypeError on the first collected sample, so the default is a Pose and the
    array form is exercised separately.
    """

    def __init__(self, base_p=((0.0, 0.0, 0.0),), base_q=((1.0, 0.0, 0.0, 0.0),),
                 offset=((0.1, 0.0, 0.5),), thresh=(0.05,), offset_as_array=False):
        self.agent = type("Agent", (), {
            "base_link": _Link(_Pose(base_p, base_q))})()
        self.ee_rest_pos_wrt_base = (
            np.asarray(offset, float) if offset_as_array
            else _Pose(offset, [(1.0, 0.0, 0.0, 0.0)] * len(offset))
        )
        self.pick_cfg = _PickCfg(np.asarray(thresh, float))


def _decl():
    return SiteDeclaration(
        key=SITE_EE_REST, site_type=SITE_POINT, subject_key="ee",
        metric=METRIC_EUCLIDEAN, source=SOURCE_ORIGIN,
        provider=PROVIDER_MSHAB_EE_REST,
        provenance="PickSubtask: ee_rest = norm(tcp - base*ee_rest_pos_wrt_base)"
                   " <= pick_cfg.ee_rest_thresh",
    )


class CompositionTest(unittest.TestCase):
    """``base_link.pose x ee_rest_pos_wrt_base``, both halves applied."""

    def test_an_untransformed_base_gives_the_bare_offset(self):
        world, _ = ee_rest_geometry(_Env(), 0)
        np.testing.assert_allclose(world, [0.1, 0.0, 0.5])

    def test_the_base_translation_is_added(self):
        world, _ = ee_rest_geometry(_Env(base_p=((1.0, 2.0, 3.0),)), 0)
        np.testing.assert_allclose(world, [1.1, 2.0, 3.5])

    def test_the_base_rotation_is_applied(self):
        """A half turn about z mirrors the offset's horizontal part. Adding
        the translation without rotating would leave x at +0.1."""
        world, _ = ee_rest_geometry(
            _Env(base_p=((1.0, 0.0, 0.0),), base_q=((0.0, 0.0, 0.0, 1.0),)), 0)
        np.testing.assert_allclose(world, [0.9, 0.0, 0.5], atol=1e-12)

    def test_rotation_and_translation_together(self):
        """Quarter turn about z: (0.1, 0, 0.5) -> (0, 0.1, 0.5), then shifted."""
        s = np.sqrt(0.5)
        world, _ = ee_rest_geometry(
            _Env(base_p=((2.0, -1.0, 0.25),), base_q=((s, 0.0, 0.0, s),)), 0)
        np.testing.assert_allclose(world, [2.0, -0.9, 0.75], atol=1e-9)

    def test_the_offset_is_read_as_a_pose(self):
        """``Pose.create_from_pq`` is what MS-HAB stores. Reading it as a
        numeric array raises TypeError on the first real sample."""
        world, _ = ee_rest_geometry(_Env(base_p=((1.0, 2.0, 3.0),)), 0)
        np.testing.assert_allclose(world, [1.1, 2.0, 3.5])

    def test_a_raw_pose_offset_is_read_too(self):
        """Some SAPIEN versions expose the batched form instead."""
        env = _Env()
        env.ee_rest_pos_wrt_base = SimpleNamespace(
            raw_pose=np.array([[0.4, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(ee_rest_geometry(env, 0)[0], [0.4, 0.0, 0.5])

    def test_a_plain_array_offset_still_works(self):
        """Only the translation is ever used, and both forms carry it."""
        world, _ = ee_rest_geometry(_Env(offset_as_array=True), 0)
        np.testing.assert_allclose(world, [0.1, 0.0, 0.5])

    def test_the_offset_rotation_is_ignored(self):
        """The site is a point. A rotated offset Pose still contributes only
        its translation, so the two forms cannot disagree."""
        turned = _Env()
        turned.ee_rest_pos_wrt_base = _Pose(
            [(0.1, 0.0, 0.5)], [(0.0, 0.0, 0.0, 1.0)])
        np.testing.assert_allclose(
            ee_rest_geometry(turned, 0)[0],
            ee_rest_geometry(_Env(offset_as_array=True), 0)[0])

    def test_the_tolerance_is_read_not_assumed(self):
        """0.05 is its current value, not its definition."""
        _, tol = ee_rest_geometry(_Env(thresh=(0.0123,)), 0)
        self.assertAlmostEqual(tol, 0.0123)


class VectorEnvTest(unittest.TestCase):
    """Every field is per sub-scene; the wrong row is a different robot."""

    def _env(self):
        return _Env(
            base_p=((0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 20.0, 0.0)),
            base_q=((1.0, 0.0, 0.0, 0.0),) * 3,
            offset=((0.1, 0.0, 0.5), (0.2, 0.0, 0.5), (0.3, 0.0, 0.5)),
            thresh=(0.05, 0.06, 0.07),
        )

    def test_each_row_composes_its_own_base_and_offset(self):
        for idx, expected in enumerate(
            ([0.1, 0.0, 0.5], [10.2, 0.0, 0.5], [0.3, 20.0, 0.5])
        ):
            with self.subTest(env_idx=idx):
                world, _ = ee_rest_geometry(self._env(), idx)
                np.testing.assert_allclose(world, expected)

    def test_each_row_reads_its_own_tolerance(self):
        for idx, expected in enumerate((0.05, 0.06, 0.07)):
            with self.subTest(env_idx=idx):
                self.assertAlmostEqual(ee_rest_geometry(self._env(), idx)[1],
                                       expected)


class MissingFieldTest(unittest.TestCase):
    """Raise, never substitute. A default rest position would be a plausible
    number for a place the robot was never asked to return to."""

    def test_no_base_link_raises(self):
        env = _Env()
        env.agent = None
        with self.assertRaises(SiteError) as ctx:
            ee_rest_geometry(env, 0)
        self.assertIn("base_link", str(ctx.exception))

    def test_no_rest_offset_raises_and_names_it(self):
        env = _Env()
        del env.ee_rest_pos_wrt_base
        with self.assertRaises(SiteError) as ctx:
            ee_rest_geometry(env, 0)
        self.assertIn("ee_rest_pos_wrt_base", str(ctx.exception))

    def test_no_threshold_raises_and_names_it(self):
        env = _Env()
        env.pick_cfg = None
        with self.assertRaises(SiteError) as ctx:
            ee_rest_geometry(env, 0)
        self.assertIn("ee_rest_thresh", str(ctx.exception))

    def test_a_short_offset_raises(self):
        with self.assertRaises(SiteError):
            ee_rest_geometry(_Env(offset=((0.1, 0.0),)), 0)

    def test_a_non_finite_pose_raises(self):
        with self.assertRaises(SiteError):
            ee_rest_geometry(_Env(base_p=((float("nan"), 0.0, 0.0),)), 0)

    def test_a_non_finite_tolerance_raises(self):
        with self.assertRaises(SiteError):
            ee_rest_geometry(_Env(thresh=(float("inf"),)), 0)


class ProviderTest(unittest.TestCase):

    def test_the_provider_is_registered_under_its_declared_name(self):
        from scenegraph.adapters.site_providers import _PROVIDERS
        self.assertIn(PROVIDER_MSHAB_EE_REST, PROVIDERS)
        self.assertIn(PROVIDER_MSHAB_EE_REST, _PROVIDERS)

    def test_the_spec_carries_the_live_pose_and_tolerance(self):
        spec = mshab_ee_rest(_Env(base_p=((1.0, 2.0, 3.0),)), 0, _decl())
        spec.validate("test")
        np.testing.assert_allclose(spec.pose_world[:3], [1.1, 2.0, 3.5])
        self.assertAlmostEqual(spec.tolerance, 0.05)

    def test_the_site_has_no_orientation_of_its_own(self):
        """It is a point. Borrowing the base's rotation would imply an axis
        nothing measures against."""
        spec = mshab_ee_rest(
            _Env(base_q=((0.0, 0.0, 0.0, 1.0),)), 0, _decl())
        np.testing.assert_allclose(spec.pose_world[3:], [1.0, 0.0, 0.0, 0.0])

    def test_nothing_is_cached_across_a_moved_base(self):
        """MS-HAB re-places the robot on reset, so a held pose names the
        previous episode's rest position."""
        decl = _decl()
        first = mshab_ee_rest(_Env(base_p=((0.0, 0.0, 0.0),)), 0, decl)
        second = mshab_ee_rest(_Env(base_p=((5.0, 5.0, 0.0),)), 0, decl)
        self.assertFalse(
            np.allclose(first.pose_world[:3], second.pose_world[:3]))

    def test_the_subject_is_the_end_effector(self):
        """Unlike every other shipped site, whose subject is an object."""
        self.assertEqual(_decl().subject_key, "ee")
        _decl().validate()


class ScaleKeyTest(unittest.TestCase):
    """Dedicated bins. Pooling a return-to-base distance into ``ee-object-*``
    would stretch the very band a two-centimetre approach registers against."""

    def test_the_keys_are_their_own_scope(self):
        self.assertEqual(EE_SITE_PLANAR_KEY, "ee-site-planar-distance")
        self.assertEqual(EE_SITE_HEIGHT_KEY, "ee-site-height-offset")

    def test_the_collector_statistic_names_match_the_bin_keys(self):
        self.assertEqual(stat_key(EE_SITE_SCOPE, "planar-distance"),
                         "ee_site_planar_distance")
        self.assertEqual(stat_key(EE_SITE_SCOPE, "height-offset"),
                         "ee_site_height_offset")

    def test_the_scope_is_not_in_the_cross_product_generator(self):
        """``SPATIAL_SCOPES`` is looped against both relations by every
        consumer, so adding it there would create and then require keys
        nothing measures."""
        self.assertNotIn(EE_SITE_SCOPE, SPATIAL_SCOPES)

    def test_the_ordinary_ee_object_scale_is_untouched(self):
        from scenegraph.core.spatial_metrics import (
            EE_OBJECT_SCOPE, spatial_bin_key,
        )
        self.assertEqual(spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance"),
                         "ee-object-planar-distance")
        self.assertNotEqual(EE_SITE_PLANAR_KEY,
                            spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance"))


if __name__ == "__main__":
    unittest.main()
