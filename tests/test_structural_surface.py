"""Structural surfaces: tables measured as planes, not as origins.

A table's link origin sits ~0.9m below its own top. Measuring against it made
every end-effector a metre in the air, and because the height scale is mined
from the range the run spans, that one pair set the deadband for every other
pair in the scene -- which is why PlaceSphere labelled every end-effector
height ``level`` for a whole episode.

Two corrections, tested here: height is taken against the surface plane, and
no public planar-distance is emitted, because the origin of a metre-wide plane
names no place to approach.
"""

import unittest

import numpy as np

from scenegraph.core.affordance import (
    AffordanceSet,
    BottomComponent,
    ReferenceSurface,
)
from scenegraph.core.relation_rules import (
    ee_object_spatial_edges,
    is_structural_surface,
    object_object_spatial_edges,
    reference_plane_world,
    surface_relative_height,
)
from scenegraph.core.schema import Graph, Node
from scenegraph.core.spatial_metrics import (
    OBJECT_OBJECT_SCOPE,
    EE_OBJECT_SCOPE,
    oriented_normal,
    spatial_bin_key,
    surface_height,
)

TABLE_KEY = "actor:table-workspace"
SPHERE_KEY = "actor:sphere"
BIN_KEY = "actor:bin"

# The table's own top, in its object frame. The shipped PlaceSphere asset
# records exactly this: an anchor 0.92m up and a normal pointing *down*,
# because support normals are mined from contact forces.
TABLE_TOP_LOCAL = [0.0, 0.0, 0.92]
MINED_DOWNWARD_NORMAL = [0.0, 0.0, -1.0]

BINS = {
    spatial_bin_key(scope, "planar-distance"): [0.03, 0.07, 0.11, 0.15]
    for scope in (EE_OBJECT_SCOPE, OBJECT_OBJECT_SCOPE)
}
BINS.update({
    spatial_bin_key(scope, "height-offset"): [-0.09, -0.03, 0.03, 0.09]
    for scope in (EE_OBJECT_SCOPE, OBJECT_OBJECT_SCOPE)
})


def _node(key, pos, node_id=None, radius=None):
    return Node(
        node_id=node_id or key, node_type="object", name=key,
        pose_world=[*pos, 1.0, 0.0, 0.0, 0.0],
        attributes={"whitelist_key": key, "entity_key": key,
                    "body_type": "kinematic" if key == TABLE_KEY else "dynamic",
                    "interaction_types": ["contact", "support"]},
    )


def _ee(pos):
    return Node(node_id="ee", node_type="ee", name="ee",
                pose_world=[*pos, 1.0, 0.0, 0.0, 0.0])


def _aff(table_origin_z=0.0, normal=None, sphere_radius=0.02):
    """Assets for a table whose origin sits ``0.92 - table_origin_z`` below its
    own top, plus a sphere that knows where its own bottom is."""
    aff = AffordanceSet()
    aff.reference_surface_by_object[TABLE_KEY] = ReferenceSurface(
        anchor_obj_frame=np.asarray(TABLE_TOP_LOCAL, float),
        outward_normal_obj_frame=np.asarray(
            normal if normal is not None else MINED_DOWNWARD_NORMAL, float),
    )
    aff.bottom_by_object[SPHERE_KEY] = [BottomComponent(
        bottom_anchor_obj_frame=np.zeros(3),
        bottom_normal_obj_frame=np.asarray([0.0, 0.0, -1.0]),
        partner_key=TABLE_KEY, radial_offset=sphere_radius,
    )]
    return aff


def _cfg(aff=None, structural=(TABLE_KEY,)):
    return {
        "bin_edges": dict(BINS),
        "structural_surfaces": set(structural),
        "affordance_set": aff if aff is not None else _aff(),
        "object_object_spatial": True,
    }


def _graph(*nodes):
    return Graph(frame=0, env_id="PlaceSphere-v1", camera="base",
                 nodes=list(nodes), edges=[])


def _by_relation(edges):
    out = {}
    for e in edges:
        out.setdefault(e.relation, []).append(e)
    return out


class NormalConventionTest(unittest.TestCase):
    """The shipped assets record downward normals. Reading them literally
    inverts every height, and ``level`` being symmetric around zero hides it."""

    def test_a_downward_mined_normal_is_re_oriented(self):
        np.testing.assert_allclose(
            oriented_normal(MINED_DOWNWARD_NORMAL), [0.0, 0.0, 1.0])

    def test_both_conventions_give_the_same_height(self):
        up = surface_height([0, 0, 1.02], [0, 0, 0.92], [0, 0, 1])
        down = surface_height([0, 0, 1.02], [0, 0, 0.92], [0, 0, -1])
        self.assertAlmostEqual(up, 0.10)
        self.assertAlmostEqual(down, 0.10)

    def test_a_horizontal_normal_has_no_up(self):
        """Guessing would be a silent half-metre error."""
        self.assertIsNone(oriented_normal([1.0, 0.0, 0.0]))
        self.assertIsNone(surface_height([0, 0, 1], [0, 0, 0], [1, 0, 0]))


class SurfaceRelativeHeightTest(unittest.TestCase):

    def test_an_object_resting_on_the_table_is_level(self):
        table, sphere = _node(TABLE_KEY, (0, 0, 0)), _node(SPHERE_KEY,
                                                           (0.1, 0.2, 0.94))
        edges = _by_relation(object_object_spatial_edges(
            _graph(table, sphere), None, _cfg()))
        height = edges["height-offset"][0]
        self.assertAlmostEqual(height.raw_value, 0.0, places=9)
        self.assertEqual(height.label, "level")

    def test_lifting_it_reads_above_then_far_above(self):
        table = _node(TABLE_KEY, (0, 0, 0))
        labels = []
        for z in (0.94, 0.99, 1.10):
            edges = _by_relation(object_object_spatial_edges(
                _graph(table, _node(SPHERE_KEY, (0.1, 0.2, z))), None, _cfg()))
            labels.append(edges["height-offset"][0].label)
        self.assertEqual(labels, ["level", "above", "far-above"])

    def test_the_table_origin_depth_does_not_move_the_surface(self):
        """The whole point: a table 0.92m tall and one 5m tall report the same
        height for an object resting on either."""
        heights = []
        for origin_z in (0.0, -4.0):
            table = _node(TABLE_KEY, (0, 0, origin_z))
            sphere = _node(SPHERE_KEY, (0.1, 0.2, origin_z + 0.94))
            edges = _by_relation(object_object_spatial_edges(
                _graph(table, sphere), None, _cfg()))
            heights.append(edges["height-offset"][0].raw_value)
        self.assertAlmostEqual(heights[0], heights[1], places=9)

    def test_the_sign_follows_pair_order(self):
        """Stored order is by key, and ``actor:sphere`` sorts before
        ``actor:table-workspace``, so a lifted sphere reads positive."""
        table = _node(TABLE_KEY, (0, 0, 0))
        sphere = _node(SPHERE_KEY, (0.1, 0.2, 1.02))
        edge = _by_relation(object_object_spatial_edges(
            _graph(table, sphere), None, _cfg()))["height-offset"][0]
        self.assertEqual((edge.src, edge.dst), (SPHERE_KEY, TABLE_KEY))
        self.assertGreater(edge.raw_value, 0.0)

    def test_ee_height_is_measured_from_the_tabletop(self):
        """0.15m above the table, not the 1.07m its origin would report."""
        graph = _graph(_ee((0.0, 0.0, 1.07)), _node(TABLE_KEY, (0, 0, 0)))
        edges = _by_relation(ee_object_spatial_edges(graph, None, _cfg()))
        self.assertAlmostEqual(edges["height-offset"][0].raw_value, 0.15)

    def test_a_rotating_sphere_keeps_its_height(self):
        """Its bottom is centre - r * up whatever the quaternion says."""
        table = _node(TABLE_KEY, (0, 0, 0))
        sphere = _node(SPHERE_KEY, (0.1, 0.2, 0.94))
        sphere.pose_world = [0.1, 0.2, 0.94, 0.7071, 0.7071, 0.0, 0.0]
        edge = _by_relation(object_object_spatial_edges(
            _graph(table, sphere), None, _cfg()))["height-offset"][0]
        self.assertAlmostEqual(edge.raw_value, 0.0, places=9)


class PlanarSuppressionTest(unittest.TestCase):
    """Only the public edge goes. Internal planar geometry stays available to
    the near gates that gate compatibility scoring."""

    def test_no_ee_table_planar_edge(self):
        graph = _graph(_ee((0.0, 0.0, 1.07)), _node(TABLE_KEY, (0, 0, 0)))
        edges = _by_relation(ee_object_spatial_edges(graph, None, _cfg()))
        self.assertNotIn("planar-distance", edges)
        self.assertIn("height-offset", edges)

    def test_no_object_table_planar_edge(self):
        graph = _graph(_node(TABLE_KEY, (0, 0, 0)),
                       _node(SPHERE_KEY, (0.1, 0.2, 0.94)))
        edges = _by_relation(object_object_spatial_edges(graph, None, _cfg()))
        self.assertNotIn("planar-distance", edges)

    def test_a_receptacle_keeps_its_planar_edge(self):
        """The bin is a localized destination: approaching its horizontal
        position is exactly what the task asks for."""
        graph = _graph(_node(BIN_KEY, (0.3, 0.0, 0.94)),
                       _node(SPHERE_KEY, (0.1, 0.2, 0.94)))
        edges = _by_relation(object_object_spatial_edges(graph, None, _cfg()))
        self.assertIn("planar-distance", edges)
        self.assertEqual(len(edges["planar-distance"]), 1)

    def test_ee_keeps_planar_to_a_manipuland(self):
        graph = _graph(_ee((0.0, 0.0, 1.07)),
                       _node(SPHERE_KEY, (0.1, 0.2, 0.94)))
        edges = _by_relation(ee_object_spatial_edges(graph, None, _cfg()))
        self.assertIn("planar-distance", edges)

    def test_an_unmarked_table_still_emits_planar(self):
        """The suppression is driven by the mined property, not by the name."""
        graph = _graph(_ee((0.0, 0.0, 1.07)), _node(TABLE_KEY, (0, 0, 0)))
        edges = _by_relation(
            ee_object_spatial_edges(graph, None, _cfg(structural=())))
        self.assertIn("planar-distance", edges)


class FailClosedTest(unittest.TestCase):
    """A member the asset called structural but never gave a plane cannot be
    measured. The only available fallback is the actor origin, which is the
    error this change exists to remove."""

    def test_a_structural_surface_without_a_plane_raises(self):
        aff = _aff()
        aff.reference_surface_by_object.clear()
        with self.assertRaises(ValueError) as ctx:
            reference_plane_world(_node(TABLE_KEY, (0, 0, 0)), aff)
        self.assertIn("reference_surface", str(ctx.exception))

    def test_emission_raises_rather_than_measuring_from_the_origin(self):
        aff = _aff()
        aff.reference_surface_by_object.clear()
        graph = _graph(_ee((0.0, 0.0, 1.07)), _node(TABLE_KEY, (0, 0, 0)))
        with self.assertRaises(ValueError):
            ee_object_spatial_edges(graph, None, _cfg(aff=aff))

    def test_an_unmarked_member_is_never_asked_for_a_plane(self):
        aff = _aff()
        aff.reference_surface_by_object.clear()
        graph = _graph(_ee((0.0, 0.0, 1.07)), _node(TABLE_KEY, (0, 0, 0)))
        ee_object_spatial_edges(graph, None, _cfg(aff=aff, structural=()))

    def test_membership_comes_from_the_mined_set(self):
        node = _node(TABLE_KEY, (0, 0, 0))
        self.assertTrue(is_structural_surface(node, _cfg()))
        self.assertFalse(is_structural_surface(node, _cfg(structural=())))
        self.assertFalse(
            is_structural_surface(_node(SPHERE_KEY, (0, 0, 1)), _cfg()))

    def test_surface_relative_height_needs_a_pose(self):
        node = _node(TABLE_KEY, (0, 0, 0))
        node.pose_world = None
        self.assertIsNone(surface_relative_height(node, np.zeros(3), _aff()))


class PlaceSphereRegressionTest(unittest.TestCase):
    """The recorded failure, reproduced from the numbers in the shipped asset.

    Every end-effector height in the PlaceSphere rollout fell inside the
    +/-0.206m ``level`` deadband, because the deadband was set by the
    0.94-1.07m end-effector-to-table-origin range.
    """

    RECORDED_EE_TABLE_ORIGIN_HEIGHTS = (0.9409, 1.0747)

    def test_the_table_range_collapses_once_measured_from_the_surface(self):
        table = _node(TABLE_KEY, (0, 0, 0))
        surface_heights = []
        for z in self.RECORDED_EE_TABLE_ORIGIN_HEIGHTS:
            graph = _graph(_ee((0.0, 0.0, z)), table)
            edges = _by_relation(
                ee_object_spatial_edges(graph, None, _cfg()))
            surface_heights.append(edges["height-offset"][0].raw_value)
        # From ~1m down to the same range the sphere and the bin occupy, which
        # is what stops one pair setting the scale for all of them.
        self.assertLess(max(surface_heights), 0.16)
        self.assertGreater(min(surface_heights), 0.0)

    def test_ee_heights_over_the_recorded_range_are_no_longer_all_level(self):
        table = _node(TABLE_KEY, (0, 0, 0))
        labels = set()
        for z in (0.9409, 0.98, 1.0747):
            graph = _graph(_ee((0.0, 0.0, z)), table)
            edges = _by_relation(ee_object_spatial_edges(graph, None, _cfg()))
            labels.add(edges["height-offset"][0].label)
        self.assertGreater(len(labels), 1)


if __name__ == "__main__":
    unittest.main()
