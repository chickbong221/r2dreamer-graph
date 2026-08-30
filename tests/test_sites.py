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
    PROVIDER_PICK_CUBE_GOAL,
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
        source=source, provider=PROVIDER_PICK_CUBE_GOAL,
        provenance="PickCube-v1.evaluate: is_obj_placed",
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
            source=SOURCE_ORIGIN, provider=PROVIDER_PICK_CUBE_GOAL,
            provenance="",
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
            "provider": "peg_hole_mouth",
            "provenance": "PegInsertionSide-v1.box_hole_pose mouth",
        }}
        parsed = parse_site_declarations(raw)
        self.assertEqual(set(parsed), {"spatial:hole_site"})
        self.assertEqual(parsed["spatial:hole_site"].subject_key, "actor:peg")
        self.assertEqual(parsed["spatial:hole_site"].source, SOURCE_PROVIDER)

    def test_an_invalid_declaration_fails_at_parse_time(self):
        """Not at the first frame that needs it."""
        raw = {"spatial:x": {"site_type": "surface", "subject": "actor:peg",
                             "metric": "nonsense", "provider": "peg_hole_mouth",
                             "provenance": "p"}}
        with self.assertRaises(SiteError):
            parse_site_declarations(raw)

    def test_goal_pairs_are_unordered(self):
        raw = {"actor:goal_site": {
            "site_type": "point", "subject": "actor:cube",
            "metric": "euclidean", "provider": "pick_cube_goal",
            "provenance": "p"}}
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


# --------------------------------------------------------------------------- #
# object-site ladder
# --------------------------------------------------------------------------- #
from scenegraph.core.relation_rules import (
    OBJECT_SITE_HEIGHT_KEY,
    OBJECT_SITE_PLANAR_KEY,
    object_object_spatial_edges,
    required_bin_keys,
)
from scenegraph.core.spatial_metrics import (
    OBJECT_REGION_PLANAR_KEY,
    OBJECT_OBJECT_SCOPE,
    spatial_bin_key,
)

PEG = "actor:peg"
HOLE = "spatial:hole_site"
REGION = "spatial:pull_goal_region"

LADDER_BINS = {
    OBJECT_SITE_PLANAR_KEY: [0.05, 0.10, 0.20, 0.30],
    OBJECT_SITE_HEIGHT_KEY: [-0.06, -0.02, 0.02, 0.06],
    OBJECT_REGION_PLANAR_KEY: [0.2, 0.4, 0.6, 0.8],
    spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance"): [.03, .07, .11, .15],
    spatial_bin_key(OBJECT_OBJECT_SCOPE, "height-offset"): [-.09, -.03, .03, .09],
}


def _obj(key, pos, dynamic=True):
    return Node(node_id=key, node_type="object", name=key,
                pose_world=[*pos, 1.0, 0.0, 0.0, 0.0],
                attributes={"whitelist_key": key, "entity_key": key,
                            "body_type": "dynamic" if dynamic else "kinematic",
                            "interaction_types": []})


def _ladder_cfg(spec, decl):
    return {
        "bin_edges": dict(LADDER_BINS),
        "site_declarations": {decl.key: decl},
        "site_specs": [spec],
        "object_object_spatial": True,
        "structural_surfaces": set(),
        "affordance_set": None,
    }


def _hole_spec(mouth=(0.0, 0.3, 0.10), head=(-0.25, 0.3, 0.10)):
    decl = SiteDeclaration(
        key=HOLE, site_type=SITE_SURFACE, subject_key=PEG,
        metric=METRIC_EUCLIDEAN, source=SOURCE_PROVIDER,
        provider="peg_hole_mouth", provenance="test",
    )
    spec = SiteSpec(
        declaration=decl,
        pose_world=np.asarray([*mouth, 1.0, 0.0, 0.0, 0.0], float),
        tolerance=0.018,
        subject_point_world=np.asarray(head, float),
    )
    return spec, decl


class ObjectSiteLadderTest(unittest.TestCase):
    """The peg-head-to-mouth rungs. Their whole reason for existing is that
    they measure a different point from the peg's origin, on a different scale
    from the peg-to-box pair that is still emitted beside them."""

    def _edges(self, peg_at=(-0.30, 0.3, 0.10), head=(-0.25, 0.3, 0.10),
               mouth=(0.0, 0.3, 0.10)):
        spec, decl = _hole_spec(mouth=mouth, head=head)
        graph = _graph(_obj(PEG, peg_at), _obj(HOLE, mouth, dynamic=False))
        edges = object_object_spatial_edges(
            graph, None, _ladder_cfg(spec, decl))
        return {e.relation: e for e in edges}

    def test_both_rungs_are_emitted(self):
        edges = self._edges()
        self.assertEqual(set(edges), {"planar-distance", "height-offset"})

    def test_they_carry_the_object_site_scale(self):
        """Not object-object: the peg-to-box origin pair is still emitted and
        must not share a scale with this one."""
        edges = self._edges()
        self.assertEqual(edges["planar-distance"].bin_key,
                         OBJECT_SITE_PLANAR_KEY)
        self.assertEqual(edges["height-offset"].bin_key,
                         OBJECT_SITE_HEIGHT_KEY)

    def test_the_distance_is_measured_from_the_peg_head(self):
        """The peg node sits at -0.30 and its head at -0.25. Measuring the
        origin would report 0.30; the head reports 0.25."""
        edges = self._edges(peg_at=(-0.30, 0.3, 0.10), head=(-0.25, 0.3, 0.10))
        self.assertAlmostEqual(edges["planar-distance"].raw_value, 0.25)

    def test_the_height_is_measured_from_the_peg_head(self):
        edges = self._edges(head=(-0.25, 0.3, 0.16), mouth=(0.0, 0.3, 0.10))
        self.assertAlmostEqual(edges["height-offset"].raw_value, 0.06)

    def test_approaching_the_mouth_improves_the_label(self):
        far = self._edges(head=(-0.30, 0.3, 0.10))["planar-distance"].label
        near = self._edges(head=(-0.02, 0.3, 0.10))["planar-distance"].label
        self.assertEqual(far, "very-far")
        self.assertEqual(near, "very-near")

    def test_the_sign_follows_pair_order(self):
        """``actor:peg`` sorts before ``spatial:hole_site``, so a head above
        the mouth reads positive."""
        edges = self._edges(head=(-0.25, 0.3, 0.16))
        edge = edges["height-offset"]
        self.assertEqual((edge.src, edge.dst), (PEG, HOLE))
        self.assertGreater(edge.raw_value, 0.0)

    def test_a_declared_site_with_no_live_spec_raises(self):
        """The provider has to have run. Emitting from the peg origin instead
        would silently measure the wrong point."""
        spec, decl = _hole_spec()
        cfg = _ladder_cfg(spec, decl)
        cfg["site_specs"] = []
        graph = _graph(_obj(PEG, (-0.3, 0.3, 0.1)),
                       _obj(HOLE, (0.0, 0.3, 0.1), dynamic=False))
        with self.assertRaises(ValueError):
            object_object_spatial_edges(graph, None, cfg)


class RequiredBinKeyTest(unittest.TestCase):
    """A task must not be made to carry a scale nothing in it produces."""

    def _decl(self, key, site_type):
        return SiteDeclaration(
            key=key, site_type=site_type, subject_key=PEG,
            metric=METRIC_EUCLIDEAN, source=SOURCE_ORIGIN,
            provider="peg_hole_mouth", provenance="test",
        )

    def test_no_sites_requires_neither_new_scale(self):
        keys = required_bin_keys({})
        self.assertNotIn(OBJECT_SITE_PLANAR_KEY, keys)
        self.assertNotIn(OBJECT_REGION_PLANAR_KEY, keys)

    def test_a_ladder_site_requires_both_object_site_keys(self):
        keys = required_bin_keys(
            {"site_declarations": {HOLE: self._decl(HOLE, SITE_SURFACE)}})
        self.assertIn(OBJECT_SITE_PLANAR_KEY, keys)
        self.assertIn(OBJECT_SITE_HEIGHT_KEY, keys)

    def test_a_region_requires_planar_only(self):
        keys = required_bin_keys(
            {"site_declarations": {REGION: self._decl(REGION, SITE_REGION)}})
        self.assertIn(OBJECT_REGION_PLANAR_KEY, keys)
        self.assertNotIn("object-region-height-offset", keys)
        self.assertNotIn(OBJECT_SITE_PLANAR_KEY, keys)

    def test_an_actor_backed_site_needs_no_new_scale(self):
        """PickCube's goal marker is a real actor whose object-object ladder
        already works; re-scoping it would mean re-mining a working task."""
        keys = required_bin_keys({"site_declarations": {
            "actor:goal_site": self._decl("actor:goal_site", SITE_POINT)}})
        self.assertNotIn(OBJECT_SITE_PLANAR_KEY, keys)
        self.assertNotIn(OBJECT_REGION_PLANAR_KEY, keys)


# --------------------------------------------------------------------------- #
# virtual sites in the graph
# --------------------------------------------------------------------------- #
from scenegraph.core.relation_rules import (
    ee_object_spatial_edges,
    is_virtual_site,
)
from scenegraph.core.spatial_metrics import (
    EE_OBJECT_SCOPE,
    ee_family_bin_key,
)
from scenegraph.core.schedule import scorable_relations

VS_BINS = {
    spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance"): [.03, .07, .11, .15],
    ee_family_bin_key("manipuland"): [-.07, -.02, .02, .07],
    ee_family_bin_key("goal-marker"): [-.13, -.04, .04, .13],
}


def _vs_node(key, pos, ntype="object"):
    return Node(node_id=key, node_type=ntype, name=key,
                pose_world=[*pos, 1.0, 0.0, 0.0, 0.0],
                attributes={"whitelist_key": key, "entity_key": key,
                            "body_type": "kinematic",
                            "interaction_types": []})


def _vs_cfg(families):
    return {"bin_edges": dict(VS_BINS), "families": dict(families),
            "structural_surfaces": set(), "site_declarations": {},
            "affordance_set": None}


class VirtualSiteEmissionTest(unittest.TestCase):
    """A virtual site has no body, so the end effector is never near it or
    above it in any sense the runtime can label.

    It is also given no height family by the miner -- deliberately, since
    inventing one would measure a distance to nothing on a real object's
    scale. Before the exclusion the two facts collided: the site was included
    like any object, the height scale was demanded, and the whole task raised
    on its first frame.
    """

    EE = Node(node_id="ee", node_type="ee", name="ee",
              pose_world=[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    def _edges(self, node, families):
        graph = Graph(frame=0, env_id="T-v1", camera="base",
                      nodes=[self.EE, node], edges=[])
        return ee_object_spatial_edges(graph, None, _vs_cfg(families))

    def test_a_virtual_site_is_recognised_by_its_namespace(self):
        self.assertTrue(is_virtual_site(_vs_node("spatial:hole_site", (0, 0, 0))))
        self.assertFalse(is_virtual_site(_vs_node("actor:goal_site", (0, 0, 0))))
        self.assertFalse(is_virtual_site(_vs_node("actor:cube", (0, 0, 0))))

    def test_no_ee_spatial_edges_for_a_virtual_site(self):
        edges = self._edges(_vs_node("spatial:hole_site", (0.1, 0.2, 0.9)),
                            {"actor:peg": "manipuland"})
        self.assertEqual(edges, [])

    def test_it_does_not_raise_for_a_family_it_was_never_given(self):
        """The exact first-frame failure: a classified asset plus an
        unclassified site meant the fail-closed height check fired on every
        Peg and Pull episode."""
        try:
            self._edges(_vs_node("spatial:pull_goal_region", (0.5, 0, 0)),
                        {"actor:cube": "manipuland"})
        except ValueError as exc:  # pragma: no cover - the regression itself
            self.fail(f"virtual site still demands a height family: {exc}")

    def test_an_actor_backed_site_keeps_its_ee_edges(self):
        """PickCube's goal marker is a real sphere with a mined family. The
        exclusion must not reach it."""
        edges = self._edges(_vs_node("actor:goal_site", (0.1, 0.0, 0.9)),
                            {"actor:goal_site": "goal-marker"})
        self.assertEqual(
            sorted(e.relation for e in edges),
            ["height-offset", "planar-distance"])

    def test_ordinary_objects_are_unaffected(self):
        edges = self._edges(_vs_node("actor:cube", (0.1, 0.0, 0.9)),
                            {"actor:cube": "manipuland"})
        self.assertEqual(len(edges), 2)


class VirtualSiteScorabilityTest(unittest.TestCase):
    """The compiler has to refuse what the runtime will not emit, or a clause
    scores zero for a whole episode with nothing to show for it."""

    MEMBERS = {
        "actor:peg": {"interaction_types": ["contact", "grasp"],
                      "family": "manipuland"},
        "spatial:hole_site": {"interaction_types": [], "kind": "spatial"},
    }
    BINS = {
        spatial_bin_key(scope, rel): (
            [.03, .07, .11, .15] if rel == "planar-distance"
            else [-.09, -.03, .03, .09])
        for scope in (EE_OBJECT_SCOPE, OBJECT_OBJECT_SCOPE)
        for rel in ("planar-distance", "height-offset")
    }

    def _table(self):
        sites = parse_site_declarations({"spatial:hole_site": {
            "site_type": "surface", "subject": "actor:peg",
            "metric": "euclidean", "source": "provider",
            "provider": "peg_hole_mouth", "provenance": "p"}})
        bins = dict(self.BINS)
        bins[ee_family_bin_key("manipuland")] = [-.07, -.02, .02, .07]
        return scorable_relations({}, self.MEMBERS, bins, sites)

    def test_ee_to_virtual_site_spatial_is_unscorable(self):
        pair = self._table()["ee / spatial:hole_site"]
        self.assertFalse(pair["planar-distance"])
        self.assertFalse(pair["height-offset"])

    def test_ee_to_a_real_object_stays_scorable(self):
        pair = self._table()["ee / actor:peg"]
        self.assertTrue(pair["planar-distance"])
        self.assertTrue(pair["height-offset"])

    def test_object_to_site_stays_scorable(self):
        """Only the end-effector pair is excluded. The peg-head-to-mouth
        ladder is the whole point of the site."""
        pair = self._table()["actor:peg / spatial:hole_site"]
        self.assertTrue(pair["reached"])


class SiteNodeMergeTest(unittest.TestCase):
    """A site key may name an actor already in the graph.

    PickCube's ``goal_site`` is a real kinematic sphere the cameras see.
    Overwriting it with a synthetic vertex discarded its segmentation ids, its
    per-camera boxes and its appearance, and left the decoder reconstructing a
    blank where a visible object stands -- silently, because poses and
    ``reached`` went on working.
    """

    class _Builder:
        """Just the merge, without a simulator behind it."""
        from scenegraph.core.graph_builder import GraphBuilder as _G
        _merge_site_nodes = _G._merge_site_nodes
        _site_node = _G._site_node

    def _real_actor(self):
        return Node(
            node_id="actor:goal_site", node_type="object", name="goal_site",
            visible=True, segmentation_ids=[7], pixel_area=140,
            pose_world=[0.1, 0.2, 0.9, 1.0, 0.0, 0.0, 0.0],
            bbox=[[0.1, 0.2, 0.3, 0.4]],
            attributes={"whitelist_key": "actor:goal_site",
                        "entity_key": "actor:goal_site"},
        )

    def _merge(self, nodes, spec):
        self._Builder()._merge_site_nodes(nodes, [spec])
        return nodes

    def test_an_actor_backed_site_keeps_its_pixels(self):
        nodes = {"actor:goal_site": self._real_actor()}
        merged = self._merge(nodes, _spec(_decl(key="actor:goal_site")))
        node = merged["actor:goal_site"]
        self.assertEqual(node.segmentation_ids, [7])
        self.assertEqual(node.pixel_area, 140)
        self.assertIsNotNone(node.bbox)
        self.assertTrue(node.visible)

    def test_it_is_still_marked_as_a_site(self):
        nodes = {"actor:goal_site": self._real_actor()}
        merged = self._merge(nodes, _spec(_decl(key="actor:goal_site")))
        self.assertTrue(merged["actor:goal_site"].attributes["is_site"])
        self.assertEqual(
            merged["actor:goal_site"].attributes["site_type"], SITE_POINT)

    def test_the_actor_keeps_its_own_pose(self):
        """The builder already refreshes actor poses from the simulator; the
        spec's copy of the same pose must not become a second source."""
        nodes = {"actor:goal_site": self._real_actor()}
        merged = self._merge(nodes, _spec(_decl(key="actor:goal_site"),
                                          pose=_pose(9.0, 9.0, 9.0)))
        self.assertEqual(merged["actor:goal_site"].pose_world[:3],
                         [0.1, 0.2, 0.9])

    def test_a_virtual_site_still_gets_a_node(self):
        nodes = {}
        decl = _decl(key="spatial:hole_site", subject="actor:peg",
                     site_type=SITE_SURFACE)
        merged = self._merge(nodes, _spec(decl, pose=_pose(0.3, 0.1, 0.9)))
        node = merged["spatial:hole_site"]
        self.assertEqual(node.segmentation_ids, [])
        self.assertFalse(node.visible)
        self.assertTrue(node.in_frame)
        self.assertEqual(node.pose_world[:3], [0.3, 0.1, 0.9])
        self.assertEqual(node.attributes["interaction_types"], [])

    def test_a_virtual_site_never_displaces_a_real_node(self):
        nodes = {"actor:cube": self._real_actor()}
        decl = _decl(key="spatial:hole_site", subject="actor:peg",
                     site_type=SITE_SURFACE)
        merged = self._merge(nodes, _spec(decl))
        self.assertEqual(sorted(merged), ["actor:cube", "spatial:hole_site"])


class GlobalBinBindingTest(unittest.TestCase):
    """Binding the shipped assets, which is the first thing a run does.

    What an asset is *required* to calibrate depends on what it *classified*:
    a families asset carries per-family height scales and deliberately drops
    the single shared one. So the classification has to reach the config
    before the requirement is computed. It did not, and every one of the four
    tasks died on ``env.reset()`` demanding ``ee-object-height-offset`` --
    the one key the miner had just, correctly, removed.

    Compilation never caught it: the schedule compiler reads the asset
    directly and never binds.
    """

    from scenegraph.core.graph_builder import GraphBuilder, TASK_LEVEL_SUBTASK

    TASKS = ("PickCube-v1", "PlaceSphere-v1", "PegInsertionSide-v1",
             "PullCubeTool-v1")

    def _builder(self, env_id):
        import os
        cfg = {
            "temporal": {"K": 2},
            "selection": {"n_max": 8},
            "whitelist_dir": os.path.join(
                "scenegraph", "configs", "subtask_whitelists", env_id),
            "object_object_spatial": True,
        }
        return self.GraphBuilder(
            None, cfg, env_id=env_id, use_target_flag=False)

    def test_every_shipped_asset_binds(self):
        for env_id in self.TASKS:
            with self.subTest(env=env_id):
                builder = self._builder(env_id)
                builder._bind_global_bin_edges(self.TASK_LEVEL_SUBTASK)
                self.assertTrue(builder.cfg["bin_edges"])

    def test_binding_publishes_the_classification(self):
        """The three things every later emission decision reads."""
        builder = self._builder("PegInsertionSide-v1")
        builder._bind_global_bin_edges(self.TASK_LEVEL_SUBTASK)
        self.assertEqual(builder.cfg["structural_surfaces"],
                         {"actor:table-workspace"})
        self.assertEqual(builder.cfg["families"]["actor:peg"], "manipuland")
        self.assertIn("spatial:hole_site", builder.cfg["site_declarations"])

    def test_a_families_asset_does_not_need_the_shared_height_scale(self):
        """The regression itself, stated as the property it violated."""
        for env_id in self.TASKS:
            with self.subTest(env=env_id):
                builder = self._builder(env_id)
                builder._bind_global_bin_edges(self.TASK_LEVEL_SUBTASK)
                self.assertNotIn("ee-object-height-offset",
                                 builder.cfg["bin_edges"])

    def test_each_family_present_has_its_own_scale_bound(self):
        for env_id in self.TASKS:
            with self.subTest(env=env_id):
                builder = self._builder(env_id)
                builder._bind_global_bin_edges(self.TASK_LEVEL_SUBTASK)
                edges = builder.cfg["bin_edges"]
                for family in set(builder.cfg["families"].values()):
                    self.assertTrue(
                        edges.get(ee_family_bin_key(family)),
                        f"{env_id}: {family} has no height scale",
                    )


class DeclaredSitePairTest(unittest.TestCase):
    """A site relates to one subject. Every other pairing is a fact the
    runtime does not emit and no sample calibrated.

    PullCubeTool's goal region was emitting a second edge for the *tool*,
    scored on a scale mined from the cube -- an extra target row the schedule
    never reads. The miner already filtered to the declared pair; emission did
    not.
    """

    REGION = "spatial:pull_goal_region"

    def _cfg(self, subject="actor:cube"):
        decl = SiteDeclaration(
            key=self.REGION, site_type=SITE_REGION, subject_key=subject,
            metric=METRIC_PLANAR, source=SOURCE_ORIGIN,
            provider="robot_base_region", provenance="p")
        spec = SiteSpec(declaration=decl,
                        pose_world=np.asarray(_pose(), float), tolerance=0.6)
        return {
            "bin_edges": {OBJECT_REGION_PLANAR_KEY: [0.7, 0.75, 0.8, 0.86],
                          spatial_bin_key(OBJECT_OBJECT_SCOPE,
                                          "planar-distance"): [.1, .2, .3, .4],
                          spatial_bin_key(OBJECT_OBJECT_SCOPE,
                                          "height-offset"): [-.2, -.06, .06, .2]},
            "site_declarations": {self.REGION: decl},
            "site_specs": [spec],
            "families": {"actor:cube": "manipuland",
                         "actor:l_shape_tool": "manipuland"},
            "structural_surfaces": set(),
            "object_object_spatial": True,
            "affordance_set": None,
        }

    def _region_edges(self, cfg):
        graph = _graph(
            _obj("actor:cube", (0.8, 0.0, 0.9)),
            _obj("actor:l_shape_tool", (0.5, 0.1, 0.9)),
            _obj(self.REGION, (0.0, 0.0, 0.0), dynamic=False),
        )
        return [e for e in object_object_spatial_edges(graph, None, cfg)
                if e.bin_key == OBJECT_REGION_PLANAR_KEY]

    def test_only_the_declared_subject_gets_a_region_edge(self):
        edges = self._region_edges(self._cfg())
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0].src, edges[0].dst},
                         {"actor:cube", self.REGION})

    def test_the_other_movable_gets_none(self):
        edges = self._region_edges(self._cfg())
        self.assertNotIn("actor:l_shape_tool",
                         {e.src for e in edges} | {e.dst for e in edges})

    def test_changing_the_declared_subject_moves_the_edge(self):
        """Driven by the declaration, not by a name."""
        edges = self._region_edges(self._cfg(subject="actor:l_shape_tool"))
        self.assertEqual(len(edges), 1)
        self.assertEqual({edges[0].src, edges[0].dst},
                         {"actor:l_shape_tool", self.REGION})

    def test_a_region_pair_still_emits_no_height(self):
        graph = _graph(
            _obj("actor:cube", (0.8, 0.0, 0.9)),
            _obj(self.REGION, (0.0, 0.0, 0.0), dynamic=False),
        )
        edges = object_object_spatial_edges(graph, None, self._cfg())
        self.assertEqual([e.relation for e in edges], ["planar-distance"])


# --------------------------------------------------------------------------- #
# the reset frame
# --------------------------------------------------------------------------- #
from scenegraph.core.relation_rules import (
    ee_object_physical_edges,
    object_object_physical_edges,
)

# The exact force PlaceSphere carried across its reset: the sphere's own
# weight, bit-identical between the previous episode's last frame and the new
# episode's first.
STALE_FORCE = 0.32873624563217163


class _StaleForceState:
    """A simulator that answers every force query from the last step.

    Which is what SAPIEN does before the first post-reset step: the pairwise
    buffer has not been recomputed, so it still holds the previous episode's
    impulses. ``queried`` records whether anything asked.
    """

    active_obj = None
    seg_id_map: dict = {}
    tcp_pose_world = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    gripper_width = 0.04

    def __init__(self):
        self.queried = []

    def pairwise_force_vector(self, a, b):
        self.queried.append(("pairwise", a, b))
        return [0.0, 0.0, -STALE_FORCE]

    def ee_object_contact_force(self, ent):
        self.queried.append(("ee_contact", ent))
        return STALE_FORCE

    def is_grasping(self, ent, max_angle=None):
        self.queried.append(("is_grasping", ent))
        return True


def _phys_node(key, pos, types=("contact", "support")):
    return Node(node_id=key, node_type="object", name=key,
                pose_world=[*pos, 1.0, 0.0, 0.0, 0.0], segmentation_ids=[],
                attributes={"whitelist_key": key, "entity_key": key,
                            "body_type": "dynamic",
                            "interaction_types": list(types)})


class ResetFrameForcePolicyTest(unittest.TestCase):
    """Before the first post-reset physics step the force buffer is invalid.

    PlaceSphere reported ``contact=holds`` at 0.329N between a bin and a
    sphere 17cm apart -- the previous episode's impulse, bit-identical. That
    completed the terminal phase, cumulative credit filled in every earlier
    phase, and the episode opened at potential 1.0 before collapsing to 0.075
    on the next frame: a -0.925 spike on the first transition of the episode.

    A distance guard cannot fix this. The value is a real force, just from the
    wrong episode, and two objects that genuinely start in contact would slip
    past any threshold. The only sound answer is not to ask.
    """

    CFG = {"contact": {"eps_force": 0.05}, "support": {},
           "grasp": {"max_angle": 30}, "affordance_set": None}

    def _scene(self):
        # The recorded PlaceSphere frame-0 poses: 17cm apart.
        return Graph(frame=0, env_id="PlaceSphere-v1", camera="base", edges=[],
                     nodes=[_phys_node("actor:bin", (0.0987, 0.0216, 0.0025)),
                            _phys_node("actor:sphere", (-0.0714, 0.0108, 0.02))])

    def _obj_edges(self, force_valid):
        state = _StaleForceState()
        edges = object_object_physical_edges(
            self._scene(), state, dict(self.CFG), force_valid=force_valid)
        return {e.relation: e for e in edges}, state

    def test_stale_forces_cannot_produce_holds(self):
        edges, _ = self._obj_edges(force_valid=False)
        self.assertEqual(edges["contact"].label, "unobserved")
        self.assertEqual(edges["support"].label, "unobserved")

    def test_no_force_query_happens_on_the_reset_frame(self):
        """Not merely discarded afterwards -- never asked."""
        _, state = self._obj_edges(force_valid=False)
        self.assertEqual(state.queried, [])

    def test_the_raw_value_is_none_not_zero(self):
        """Zero is a measurement. This is the absence of one."""
        edges, _ = self._obj_edges(force_valid=False)
        self.assertIsNone(edges["contact"].raw_value)
        self.assertIsNone(edges["support"].raw_value)

    def test_frame_one_reports_genuine_contact(self):
        edges, state = self._obj_edges(force_valid=True)
        self.assertEqual(edges["contact"].label, "holds")
        self.assertEqual(edges["support"].label, "src-holds")
        self.assertAlmostEqual(edges["contact"].raw_value, STALE_FORCE)
        self.assertTrue(state.queried)

    def test_the_edge_is_still_present_so_the_frame_stays_readable(self):
        """Omitting it would mark the frame invalid, which is a different and
        worse answer than 'this fact was not observed'."""
        edges, _ = self._obj_edges(force_valid=False)
        self.assertEqual(sorted(edges), ["contact", "support"])

    def test_ee_contact_and_grasp_are_unobserved_too(self):
        graph = Graph(frame=0, env_id="T-v1", camera="base", edges=[],
                      nodes=[Node(node_id="ee", node_type="ee", name="ee",
                                  pose_world=[0, 0, 1.0, 1, 0, 0, 0]),
                             _phys_node("actor:sphere", (0.0, 0.0, 0.9),
                                        types=("contact", "grasp"))])
        state = _StaleForceState()
        edges = ee_object_physical_edges(
            graph, state, dict(self.CFG), force_valid=False)
        self.assertEqual({e.relation: e.label for e in edges},
                         {"contact": "unobserved", "grasp": "unobserved"})
        self.assertEqual(state.queried, [])

    def test_unobserved_is_admissible_for_the_force_derived_predicates(self):
        from scenegraph.core.relation_rules import abs_labels_for
        labels = abs_labels_for()
        for relation in ("contact", "grasp", "support"):
            self.assertIn("unobserved", labels[relation])

    def test_pose_derived_predicates_never_carry_unobserved(self):
        """``contain`` is the key-in-entry-volume test and ``reached`` is a
        distance: both are computable the moment poses exist, so admitting
        'unobserved' for them would widen the decoder mask for a label they
        can never take."""
        from scenegraph.core.relation_rules import abs_labels_for
        labels = abs_labels_for()
        self.assertNotIn("unobserved", labels["contain"])
        self.assertNotIn("unobserved", labels["reached"])

    def test_the_absolute_vocabulary_did_not_grow(self):
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        self.assertEqual(len(build_absolute_vocab()), 19)

    def test_pose_derived_relations_still_emit_at_reset(self):
        """Spatial facts are unaffected: nothing about them reads the force
        buffer, and a reset frame is a perfectly good frame to measure in."""
        from scenegraph.core.relation_rules import object_object_spatial_edges
        cfg = dict(self.CFG)
        cfg.update({
            "bin_edges": {
                spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance"):
                    [.03, .07, .11, .15],
                spatial_bin_key(OBJECT_OBJECT_SCOPE, "height-offset"):
                    [-.09, -.03, .03, .09]},
            "object_object_spatial": True, "structural_surfaces": set(),
            "families": {}, "site_declarations": {},
        })
        edges = object_object_spatial_edges(self._scene(), _StaleForceState(),
                                            cfg)
        self.assertEqual(sorted(e.relation for e in edges),
                         ["height-offset", "planar-distance"])


if __name__ == "__main__":
    unittest.main()
