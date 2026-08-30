"""Spatial sites and the ``reached`` relation.

``reached`` is the one relation whose threshold is not mined. It mirrors an
environment's own success geometry, so every test here is a way that mirror
could be off by a little and still look right: the wrong comparison at the
boundary, the wrong source point, a tolerance frozen at mining time, or a pair
that never declared a goal at all.
"""

import unittest

import numpy as np

from scenegraph.core.relation_rules import (
    GOAL_RELATIONS,
    RELATION_TYPES,
    abs_labels_for,
)
from scenegraph.core.sites import (
    METRIC_EUCLIDEAN,
    METRIC_PLANAR,
    SITE_POINT,
    SITE_REGION,
    SITE_SURFACE,
    SOURCE_ORIGIN,
    SOURCE_PROVIDER,
    SiteDeclaration,
    SiteError,
    SiteSpec,
    goal_pairs,
    parse_site_declarations,
    reached_holds,
    site_distance,
    site_pair_points,
)


def _pose(x=0.0, y=0.0, z=0.0):
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


def _decl(key="actor:goal_site", subject="actor:cube", site_type=SITE_POINT,
          metric=METRIC_EUCLIDEAN, source=SOURCE_ORIGIN):
    return SiteDeclaration(
        key=key, site_type=site_type, subject_key=subject, metric=metric,
        source=source, provenance="PickCube-v1.evaluate: is_obj_placed",
    )


def _spec(decl=None, pose=None, tolerance=0.025, subject_point=None):
    return SiteSpec(
        declaration=decl or _decl(),
        pose_world=np.asarray(pose if pose is not None else _pose(), float),
        tolerance=tolerance,
        subject_point_world=(None if subject_point is None
                             else np.asarray(subject_point, float)),
    )


class VocabularyTest(unittest.TestCase):
    """The ten existing relations keep their ids. A trained head reads the
    relation embedding by position, so inserting ``reached`` anywhere but the
    end would silently rescore every affordance fact."""

    def test_reached_is_appended_last(self):
        self.assertEqual(RELATION_TYPES[-1], "reached")
        self.assertEqual(GOAL_RELATIONS, ("reached",))

    def test_existing_relation_order_is_unchanged(self):
        self.assertEqual(RELATION_TYPES[:10], (
            "contact", "grasp", "support", "contain",
            "planar-distance", "height-offset",
            "grasp-compatibility", "contact-compatibility",
            "support-compatibility", "contain-compatibility",
        ))

    def test_reached_reuses_the_physical_labels(self):
        """sigma must not grow: a new absolute label would shift every id
        above it and invalidate the decoder head the same way."""
        self.assertEqual(abs_labels_for()["reached"], ["not-holds", "holds"])

    def test_reached_carries_no_temporal_label(self):
        from scenegraph.core.relation_rules import TEMPORAL_RELATIONS
        self.assertNotIn("reached", TEMPORAL_RELATIONS)


class ThresholdTest(unittest.TestCase):
    """The comparison at the boundary is the environment's, not ours."""

    def test_point_goal_holds_at_exactly_the_tolerance(self):
        """PickCube's is_obj_placed is ``<= goal_thresh``."""
        spec = _spec(tolerance=0.025)
        self.assertTrue(reached_holds(spec, _pose(x=0.025)))

    def test_point_goal_does_not_hold_just_past_the_tolerance(self):
        spec = _spec(tolerance=0.025)
        self.assertFalse(reached_holds(spec, _pose(x=0.0250001)))

    def test_region_goal_is_strictly_inside(self):
        """PullCubeTool's evaluate is ``< 0.6``, so the boundary is out."""
        spec = _spec(_decl(site_type=SITE_REGION, metric=METRIC_PLANAR),
                     tolerance=0.6)
        self.assertFalse(reached_holds(spec, _pose(x=0.6)))
        self.assertTrue(reached_holds(spec, _pose(x=0.5999)))

    def test_point_goal_uses_full_3d_distance(self):
        """A cube directly above the goal marker has not reached it, however
        good its planar distance looks."""
        spec = _spec(tolerance=0.025)
        self.assertFalse(reached_holds(spec, _pose(z=0.2)))
        self.assertAlmostEqual(site_distance(spec, _pose(z=0.2)), 0.2)

    def test_region_goal_ignores_height(self):
        spec = _spec(_decl(site_type=SITE_REGION, metric=METRIC_PLANAR),
                     tolerance=0.6)
        self.assertAlmostEqual(site_distance(spec, _pose(x=0.3, z=5.0)), 0.3)

    def test_distance_is_unclamped(self):
        """The ladder reads the raw distance; only ``reached`` knows the
        radius. A hinge clamped at the boundary would make every frame inside
        the region report the same distance and flatten the approach."""
        spec = _spec(_decl(site_type=SITE_REGION, metric=METRIC_PLANAR),
                     tolerance=0.6)
        self.assertAlmostEqual(site_distance(spec, _pose(x=0.2)), 0.2)


class SourcePointTest(unittest.TestCase):
    """PegInsertionSide measures the peg *head*. Calibrating on the peg origin
    and reading the head is exactly the drift spatial_metrics exists to stop."""

    def test_provider_point_overrides_the_subject_origin(self):
        spec = _spec(_decl(source=SOURCE_PROVIDER),
                     subject_point=[0.1, 0.0, 0.0])
        source, site = site_pair_points(spec, _pose(x=0.9))
        np.testing.assert_allclose(source, [0.1, 0.0, 0.0])
        np.testing.assert_allclose(site, [0.0, 0.0, 0.0])

    def test_origin_source_uses_the_subject_pose(self):
        source, _ = site_pair_points(_spec(), _pose(x=0.9))
        np.testing.assert_allclose(source, [0.9, 0.0, 0.0])

    def test_provider_source_without_a_point_is_rejected(self):
        """Falling back to the origin here would silently measure a different
        quantity than the bins were calibrated on."""
        spec = _spec(_decl(source=SOURCE_PROVIDER))
        with self.assertRaises(SiteError):
            spec.validate()

    def test_unresolvable_subject_reports_none_rather_than_zero(self):
        self.assertIsNone(site_distance(_spec(), None))
        self.assertIsNone(reached_holds(_spec(), None))


class SpecValidationTest(unittest.TestCase):
    """A site whose provider failed must raise, not emit a confident
    'not-holds' from a stale pose."""

    def test_non_finite_pose_is_rejected(self):
        with self.assertRaises(SiteError):
            _spec(pose=[float("nan")] + _pose()[1:]).validate()

    def test_non_positive_tolerance_is_rejected(self):
        with self.assertRaises(SiteError):
            _spec(tolerance=0.0).validate()

    def test_a_site_cannot_be_its_own_subject(self):
        with self.assertRaises(SiteError):
            _decl(key="actor:x", subject="actor:x").validate()

    def test_unknown_site_type_is_rejected(self):
        with self.assertRaises(SiteError):
            _decl(site_type="blob").validate()

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(SiteError):
            _decl(metric="manhattan").validate()

    def test_missing_provenance_is_rejected(self):
        decl = SiteDeclaration(
            key="actor:goal_site", site_type=SITE_POINT,
            subject_key="actor:cube", metric=METRIC_EUCLIDEAN,
            source=SOURCE_ORIGIN, provenance="",
        )
        with self.assertRaises(SiteError):
            decl.validate()

    def test_a_surface_site_is_a_valid_type(self):
        _decl(site_type=SITE_SURFACE).validate()


class DeclarationParsingTest(unittest.TestCase):

    def test_absent_section_means_no_sites(self):
        self.assertEqual(parse_site_declarations(None), {})

    def test_a_parsed_declaration_round_trips(self):
        raw = {"spatial:hole_site": {
            "site_type": "surface", "subject": "actor:peg",
            "metric": "euclidean", "source": "provider",
            "provenance": "PegInsertionSide-v1.box_hole_pose mouth",
        }}
        parsed = parse_site_declarations(raw)
        self.assertEqual(set(parsed), {"spatial:hole_site"})
        self.assertEqual(parsed["spatial:hole_site"].subject_key, "actor:peg")
        self.assertEqual(parsed["spatial:hole_site"].source, SOURCE_PROVIDER)

    def test_an_invalid_declaration_fails_at_parse_time(self):
        """Not at the first frame that needs it."""
        raw = {"spatial:x": {"site_type": "surface", "subject": "actor:peg",
                             "metric": "nonsense", "provenance": "p"}}
        with self.assertRaises(SiteError):
            parse_site_declarations(raw)

    def test_goal_pairs_are_unordered(self):
        raw = {"actor:goal_site": {
            "site_type": "point", "subject": "actor:cube",
            "metric": "euclidean", "provenance": "p"}}
        self.assertEqual(
            goal_pairs(parse_site_declarations(raw)),
            {("actor:cube", "actor:goal_site")},
        )


# --------------------------------------------------------------------------- #
# runtime emission
# --------------------------------------------------------------------------- #
from scenegraph.core.relation_rules import goal_edges
from scenegraph.core.schema import Graph, Node


def _node(node_id, key, pose):
    return Node(node_id=node_id, node_type="object", name=node_id,
                pose_world=list(pose), attributes={"whitelist_key": key})


def _graph(*nodes):
    return Graph(frame=3, env_id="T-v1", camera="base",
                 nodes=list(nodes), edges=[])


class GoalEdgeEmissionTest(unittest.TestCase):
    """The replay potential requires exactly one edge per scored relation and
    masks the whole frame when one is missing, so ``reached`` is emitted every
    frame or it raises. Going quiet is the one thing it must never do."""

    def _cfg(self, spec):
        return {"site_specs": [spec]}

    def _scene(self, cube_x=0.9):
        return _graph(
            _node("actor:cube", "actor:cube", _pose(x=cube_x)),
            _node("actor:goal_site", "actor:goal_site", _pose()),
        )

    def test_one_edge_per_declared_pair(self):
        edges = goal_edges(self._scene(), None, self._cfg(_spec(
            _decl(key="actor:goal_site", subject="actor:cube"))))
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0].relation, "reached")

    def test_a_false_goal_emits_not_holds_rather_than_nothing(self):
        edges = goal_edges(self._scene(cube_x=0.9), None, self._cfg(_spec(
            _decl(key="actor:goal_site", subject="actor:cube"))))
        self.assertEqual(edges[0].label, "not-holds")

    def test_a_satisfied_goal_emits_holds(self):
        edges = goal_edges(self._scene(cube_x=0.01), None, self._cfg(_spec(
            _decl(key="actor:goal_site", subject="actor:cube"))))
        self.assertEqual(edges[0].label, "holds")

    def test_the_raw_distance_travels_with_the_edge(self):
        edges = goal_edges(self._scene(cube_x=0.4), None, self._cfg(_spec(
            _decl(key="actor:goal_site", subject="actor:cube"))))
        self.assertAlmostEqual(edges[0].raw_value, 0.4)

    def test_endpoints_are_stored_in_sorted_key_order(self):
        """The compiler resolves a clause to sorted key order, so a fact
        stored the other way round would never be found."""
        edges = goal_edges(self._scene(), None, self._cfg(_spec(
            _decl(key="actor:goal_site", subject="actor:cube"))))
        self.assertEqual((edges[0].src, edges[0].dst),
                         ("actor:cube", "actor:goal_site"))

    def test_a_site_sorting_first_still_stores_sorted(self):
        graph = _graph(
            _node("actor:zcube", "actor:zcube", _pose(x=0.9)),
            _node("actor:agoal", "actor:agoal", _pose()),
        )
        edges = goal_edges(graph, None, self._cfg(_spec(
            _decl(key="actor:agoal", subject="actor:zcube"))))
        self.assertEqual((edges[0].src, edges[0].dst),
                         ("actor:agoal", "actor:zcube"))

    def test_a_missing_subject_raises(self):
        graph = _graph(_node("actor:goal_site", "actor:goal_site", _pose()))
        with self.assertRaises(ValueError) as ctx:
            goal_edges(graph, None, self._cfg(_spec(
                _decl(key="actor:goal_site", subject="actor:cube"))))
        self.assertIn("actor:cube", str(ctx.exception))

    def test_a_missing_site_raises(self):
        graph = _graph(_node("actor:cube", "actor:cube", _pose(x=0.9)))
        with self.assertRaises(ValueError):
            goal_edges(graph, None, self._cfg(_spec(
                _decl(key="actor:goal_site", subject="actor:cube"))))

    def test_a_duplicated_key_raises_rather_than_picking_one(self):
        graph = _graph(
            _node("actor:cube-0", "actor:cube", _pose(x=0.9)),
            _node("actor:cube-1", "actor:cube", _pose(x=0.5)),
            _node("actor:goal_site", "actor:goal_site", _pose()),
        )
        with self.assertRaises(ValueError):
            goal_edges(graph, None, self._cfg(_spec(
                _decl(key="actor:goal_site", subject="actor:cube"))))

    def test_a_failed_provider_raises_rather_than_emitting_not_holds(self):
        spec = SiteSpec(
            declaration=_decl(key="actor:goal_site", subject="actor:cube"),
            pose_world=np.asarray([float("nan")] + _pose()[1:], float),
            tolerance=0.025,
        )
        with self.assertRaises(Exception):
            goal_edges(self._scene(), None, self._cfg(spec))

    def test_no_declared_sites_emits_nothing(self):
        self.assertEqual(goal_edges(self._scene(), None, {}), [])

    def test_the_peg_head_override_is_what_gets_measured(self):
        """The cube node sits 0.9 m away; the provider's source point is at
        0.01 m. The edge must report the provider's."""
        spec = _spec(_decl(key="actor:goal_site", subject="actor:cube",
                           source=SOURCE_PROVIDER),
                     subject_point=[0.01, 0.0, 0.0])
        edges = goal_edges(self._scene(cube_x=0.9), None, self._cfg(spec))
        self.assertAlmostEqual(edges[0].raw_value, 0.01)
        self.assertEqual(edges[0].label, "holds")


if __name__ == "__main__":
    unittest.main()
