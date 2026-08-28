"""Retention: which vertices survive, what they carry, and what they relate to.

Everything here is synthetic -- no simulator, no torch. Retention is now
unconditional, so there is no snapshot rule left to exercise; what matters is
that a node without pixels keeps its row and its live centroid, that ``in_frame``
rather than ``visible`` decides eligibility, and that the protected target still
pairs with whatever the cameras do cover.
"""

import unittest
from types import SimpleNamespace
from typing import Optional

import numpy as np

from scenegraph.adapters.graph_pack import (
    _row_assignment,
    pack_graph,
)
from scenegraph.adapters.graph_vocab import (
    EntityVocab,
    GraphVocab,
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.graph_builder import (
    VISIBILITY_KEEP,
    VISIBILITY_PROJECTED,
    GraphBuilder,
)
from scenegraph.core.spatial_metrics import (
    EE_OBJECT_SCOPE,
    OBJECT_OBJECT_SCOPE,
    SPATIAL_SCOPES,
    change_bin_key,
    spatial_bin_key,
)
from scenegraph.core.relation_rules import (
    _eligible_objects,
    _ee_object_nodes,
    _object_pairs,
    ee_object_spatial_edges,
    height_offset_xyz,
    object_object_spatial_edges,
    planar_distance_xyz,
)
from scenegraph.core.schema import Graph, Node

N_MAX = 8
E_MAX = 24


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _node(node_id, name, *, visible=True, in_frame=None, pose=(0.0, 0.0, 0.0),
          seg=(1,), node_type="object", box=0.4,
          quat=(1.0, 0.0, 0.0, 0.0)):
    node = Node(
        node_id=node_id,
        node_type=node_type,
        name=name,
        visible=visible,
        in_frame=visible if in_frame is None else in_frame,
        segmentation_ids=list(seg),
        pose_world=[*pose, *quat],
        attributes={"whitelist_key": name, "interaction_types": {"grasp", "contact"}},
    )
    node.bbox = np.full((2, 4), box, np.float32)
    node.bbox[:, 1] += 0.1
    node.bbox[:, 3] += 0.1
    return node


def _retained(node_id, name, **kw):
    """A node re-injected by ``merge_retained``: no pixels, live pose."""
    node = _node(node_id, name, visible=False, seg=(), **kw)
    node.bbox = None
    return node


def _graph(nodes, target_id: Optional[str] = "obj-1"):
    return Graph(
        frame=0, env_id="env0", camera="cam", nodes=list(nodes), edges=[],
        meta=dict(active_target_node_id=target_id, node_uids={}),
    )


def _vocab(names):
    tokens = {"<pad>": 0, "<ee>": 1}
    for i, name in enumerate(sorted(set(names))):
        tokens[name] = i + 2
    relation = build_relation_vocab()
    return GraphVocab(
        entity=EntityVocab(token_to_id=tokens),
        relation=relation,
        absolute=build_absolute_vocab(),
        temporal=build_temporal_vocab(),
        abs_valid=np.zeros((len(relation), 1), bool),
        temp_valid=np.zeros(len(relation), bool),
    )


def _pack(graph, names, n_max=N_MAX):
    return pack_graph(
        graph, _vocab(names), n_max=n_max, e_max=E_MAX, n_cams=2,
    )


# Labels come from the relation vocabulary; cfg supplies edges only.
_BINS = {
    k: ([0.1, 0.2, 0.6, 1.0] if k.endswith("planar-distance")
        else [-0.4, -0.1, 0.1, 0.4])
    for scope in SPATIAL_SCOPES
    for k in (spatial_bin_key(scope, "planar-distance"),
              spatial_bin_key(scope, "height-offset"))
}


# --------------------------------------------------------------------------- #
# Row assignment
# --------------------------------------------------------------------------- #
class FixedRowTest(unittest.TestCase):
    """Row 0 is the end effector, row 1 the subtask target."""

    def test_ee_is_row_zero_even_when_it_arrives_last(self):
        graph = _graph([_node("obj-1", "apple"),
                        _node("ee", "ee", node_type="ee", seg=())])
        rows, _ = _row_assignment(graph, N_MAX, "obj-1", True)
        self.assertEqual({r for r, n in rows if n.node_type == "ee"}, {0})

    def test_target_takes_row_one_whatever_its_registry_index(self):
        graph = _graph([_node("ee", "ee", node_type="ee", seg=()),
                        _node("obj-0", "bowl"), _node("obj-1", "apple")])
        rows, _ = _row_assignment(graph, N_MAX, "obj-1", True)
        by_id = {n.node_id: r for r, n in rows}
        self.assertEqual(by_id["obj-1"], 1)
        self.assertEqual(by_id["obj-0"], 2)

    def test_row_one_is_padding_before_the_target_is_observed(self):
        graph = _graph([_node("ee", "ee", node_type="ee", seg=()),
                        _node("obj-0", "bowl")], target_id="obj-1")
        rows, _ = _row_assignment(graph, N_MAX, "obj-1", True)
        self.assertNotIn(1, {r for r, _ in rows})

    def test_reserving_row_one_raises_rather_than_dropping(self):
        """The old packer returned a drop count here. Retention leaves no
        vertex that is safe to lose, so it is an error instead."""
        nodes = [_node("ee", "ee", node_type="ee", seg=())]
        nodes += [_node(f"obj-{i}", f"o{i}") for i in range(N_MAX)]
        graph = _graph(nodes, target_id="missing")
        with self.assertRaises(RuntimeError):
            _row_assignment(graph, N_MAX, "missing", True)

    def test_more_vertices_than_rows_raises(self):
        nodes = [_node(f"obj-{i}", f"o{i}") for i in range(N_MAX + 1)]
        graph = _graph(nodes, target_id=None)
        with self.assertRaises(RuntimeError) as cm:
            _row_assignment(graph, N_MAX, None, False)
        self.assertIn("Retention never evicts", str(cm.exception))


# --------------------------------------------------------------------------- #
# What a node without pixels carries
# --------------------------------------------------------------------------- #
class RetainedNodePackingTest(unittest.TestCase):
    NAMES = ["apple", "bowl"]

    def _packed(self, nodes, target="obj-1"):
        return _pack(_graph(nodes, target), self.NAMES)

    def test_retained_node_keeps_its_centroid(self):
        """Decided deliberately: the position stays live. ``bbox = 0000`` is the
        pixel-visibility signal, and zeroing the centroid too would delete the
        only thing that says where an inserted object went."""
        packed = self._packed([
            _node("ee", "ee", node_type="ee", seg=()),
            _retained("obj-1", "apple", pose=(0.4, 0.1, 0.9)),
        ])
        np.testing.assert_allclose(
            packed["graph_node_centroid"][1], [0.4, 0.1, 0.9], atol=1e-6)

    def test_retained_node_boxes_go_to_zero(self):
        packed = self._packed([
            _node("ee", "ee", node_type="ee", seg=()),
            _retained("obj-1", "apple"),
        ])
        np.testing.assert_allclose(packed["graph_node_bbox"][1], 0.0)

    def test_a_retained_non_target_keeps_its_row_too(self):
        """The old rule kept exactly one node alive. Every admitted node now
        survives, so a hidden receptacle is still a vertex."""
        packed = self._packed([
            _node("ee", "ee", node_type="ee", seg=()),
            _node("obj-1", "apple"),
            _retained("obj-0", "bowl", pose=(0.2, 0.0, 0.5)),
        ])
        self.assertNotEqual(int(packed["graph_node_ent"][2]), 0)
        np.testing.assert_allclose(
            packed["graph_node_centroid"][2], [0.2, 0.0, 0.5], atol=1e-6)


# --------------------------------------------------------------------------- #
# Eligibility
# --------------------------------------------------------------------------- #
class EligibilityTest(unittest.TestCase):
    """``in_frame`` decides relations; ``visible`` decides pixels."""

    def test_robot_blocked_object_is_still_eligible(self):
        blocked = _node("obj-0", "bowl", visible=False, in_frame=True, seg=())
        graph = _graph([_node("ee", "ee", node_type="ee", seg=()), blocked],
                       target_id=None)
        self.assertEqual([n.node_id for n in _eligible_objects(graph)],
                         ["obj-0"])

    def test_out_of_frame_object_is_not_eligible(self):
        gone = _node("obj-0", "bowl", visible=False, in_frame=False, seg=())
        self.assertEqual(_eligible_objects(_graph([gone], target_id=None)), [])

    def test_segmentation_ids_are_no_longer_required_for_pairing(self):
        """They used to gate object pairs, which excluded exactly the
        robot-blocked case the projection exists to keep."""
        a = _node("obj-0", "bowl", visible=False, in_frame=True, seg=())
        b = _node("obj-1", "apple")
        graph = _graph([a, b], target_id=None)
        self.assertEqual(len(_object_pairs(graph)), 1)


class ProtectedTargetPairingTest(unittest.TestCase):
    """The target bypasses its own eligibility, and only its own."""

    def _pairs(self, nodes):
        pairs = _object_pairs(_graph(nodes, "obj-1"))
        return {frozenset((a.node_id, b.node_id)) for a, b in pairs}

    def test_out_of_frame_target_pairs_with_an_in_frame_object(self):
        """A place subtask's defining fact is target-to-receptacle. Losing it
        when the camera turns away deletes the evidence mid-subtask."""
        target = _node("obj-1", "apple", visible=False, in_frame=False, seg=())
        other = _node("obj-0", "bowl")
        self.assertIn(frozenset(("obj-1", "obj-0")),
                      self._pairs([target, other]))

    def test_two_out_of_frame_objects_do_not_pair(self):
        target = _node("obj-1", "apple", visible=False, in_frame=False, seg=())
        other = _node("obj-0", "bowl", visible=False, in_frame=False, seg=())
        self.assertEqual(self._pairs([target, other]), set())

    def test_an_ordinary_out_of_frame_object_gets_no_exception(self):
        stray = _node("obj-2", "bowl", visible=False, in_frame=False, seg=())
        other = _node("obj-0", "bowl")
        self.assertEqual(self._pairs([stray, other]), set())

    def test_an_in_frame_target_is_paired_exactly_once(self):
        """The exception must not double-emit when the target is visible."""
        pairs = _object_pairs(
            _graph([_node("obj-1", "apple"), _node("obj-0", "bowl")], "obj-1"))
        self.assertEqual(len(pairs), 1)

    def test_canonical_orientation_holds_for_the_exception_pair(self):
        """A single stored edge only means one thing if the pair order is
        stable, target exception or not."""
        target = _node("obj-1", "zebra", visible=False, in_frame=False, seg=())
        other = _node("obj-0", "apple")
        pairs = _object_pairs(_graph([target, other], "obj-1"))
        self.assertEqual([(a.node_id, b.node_id) for a, b in pairs],
                         [("obj-0", "obj-1")])


# --------------------------------------------------------------------------- #
# Relations that continue while the target is hidden
# --------------------------------------------------------------------------- #
class RetainedTargetRelationTest(unittest.TestCase):
    def _graph_with(self, ee_xyz, target_xyz, in_frame=False):
        ee = _node("ee", "ee", node_type="ee", pose=ee_xyz, seg=())
        target = _node("obj-1", "apple", visible=in_frame, in_frame=in_frame,
                       pose=target_xyz, seg=())
        return _graph([ee, target], "obj-1")

    def test_iterator_keeps_the_target_when_it_is_out_of_frame(self):
        graph = self._graph_with((0, 0, 0), (0.3, 0, 0.2))
        self.assertEqual([n.node_id for n in _ee_object_nodes(graph)],
                         ["obj-1"])

    def test_distance_and_height_match_the_centroids_exactly(self):
        graph = self._graph_with((0.0, 0.0, 0.0), (0.3, 0.4, 0.2))
        edges = {e.relation: e
                 for e in ee_object_spatial_edges(graph, None, {"bin_edges": _BINS})}
        target = np.array([0.3, 0.4, 0.2])
        self.assertAlmostEqual(edges["planar-distance"].raw_value,
                               planar_distance_xyz(np.zeros(3), target))
        self.assertAlmostEqual(edges["height-offset"].raw_value,
                               height_offset_xyz(np.zeros(3), target))

    def test_moving_the_end_effector_moves_the_hidden_targets_relations(self):
        cfg = {"bin_edges": _BINS}
        far = ee_object_spatial_edges(
            self._graph_with((0.0, 0.0, 0.0), (0.9, 0.0, 0.0)), None, cfg)[0]
        near = ee_object_spatial_edges(
            self._graph_with((0.8, 0.0, 0.0), (0.9, 0.0, 0.0)), None, cfg)[0]
        self.assertGreater(far.raw_value, near.raw_value)


# --------------------------------------------------------------------------- #
# The ladder a carry phase reads
# --------------------------------------------------------------------------- #
class ObjectObjectSpatialTest(unittest.TestCase):
    CFG = {"bin_edges": _BINS}

    def _edges(self, a_xyz, b_xyz):
        a = _node("obj-0", "cubeA", pose=a_xyz)
        b = _node("obj-1", "cubeB", pose=b_xyz)
        return object_object_spatial_edges(_graph([a, b], None), None, self.CFG)

    def _of(self, edges, relation):
        return [e for e in edges if e.relation == relation][0]

    def test_a_pair_gets_both_ladders(self):
        rels = {e.relation for e in self._edges((0, 0, 0), (0.3, 0, 0))}
        self.assertEqual(rels, {"planar-distance", "height-offset"})

    def test_distance_shrinks_as_the_objects_approach(self):
        """Without this a carry phase has to score the gripper's distance to
        the destination, which stops being the object's the moment it is let
        go."""
        far = self._of(self._edges((0.0, 0, 0), (0.9, 0, 0)), "planar-distance")
        near = self._of(self._edges((0.0, 0, 0), (0.15, 0, 0)), "planar-distance")
        self.assertGreater(far.raw_value, near.raw_value)
        self.assertNotEqual(far.label, near.label)

    def test_height_direction_rides_in_the_label(self):
        """One edge per pair, so above/below has to be readable from the label
        rather than from which endpoint is the source."""
        above = self._of(self._edges((0, 0, 0.5), (0, 0, 0.0)), "height-offset")
        below = self._of(self._edges((0, 0, 0.0), (0, 0, 0.5)), "height-offset")
        self.assertNotEqual(above.label, below.label)


# --------------------------------------------------------------------------- #
# Builder-side retention
# --------------------------------------------------------------------------- #
class _State:
    """The privileged lookups retention and seeding need. ``step`` wants a live
    env; these methods do not."""

    active_subtask_type = "pick"
    env_idx = 0
    robot_links = frozenset()
    robot_link_names = frozenset()

    def __init__(self, seg_id_map=None):
        self.seg_id_map = dict(seg_id_map or {})


class _Entity:
    def __init__(self, name):
        self.name = name


def _builder(policy=VISIBILITY_KEEP, n_max=3):
    from scenegraph.core.selector import EntityRegistry

    builder = object.__new__(GraphBuilder)
    builder.env_id, builder.env_idx = "env0", 0
    builder._task_group = "set_table"
    builder.registry = EntityRegistry(n_max=n_max)
    builder._entities = {}
    builder._coverage = None
    builder.visibility_policy = policy
    return builder


class CapacityIsFatalTest(unittest.TestCase):
    def test_overflow_raises_before_anything_is_dropped(self):
        builder = _builder(n_max=3)          # ee plus two objects
        nodes = {f"o{i}": _node(f"o{i}", f"n{i}") for i in range(3)}
        with self.assertRaises(RuntimeError) as cm:
            builder._check_capacity(nodes, 7, _State())
        message = str(cm.exception)
        self.assertIn("frame=7", message)
        self.assertIn("n_max=3", message)
        self.assertIn("set_table", message)
        self.assertIn("o0", message)

    def test_exactly_at_capacity_is_silent(self):
        builder = _builder(n_max=3)
        nodes = {f"o{i}": _node(f"o{i}", f"n{i}") for i in range(2)}
        builder._check_capacity(nodes, 0, _State())


class EntityCacheTest(unittest.TestCase):
    """The association has to be made while the node still has pixels."""

    def test_resolution_survives_the_pixels_going_away(self):
        builder = _builder()
        apple = _Entity("apple")
        seen = _node("o1", "apple", seg=(5,))
        self.assertIs(builder._entity_for(seen, _State({5: apple})), apple)
        hidden = _node("o1", "apple", visible=False, seg=())
        self.assertIs(builder._entity_for(hidden, _State({})), apple)

    def test_an_unseen_node_resolves_to_nothing(self):
        """A force query on None reads zero, which would be emitted as a
        confident not-holds. Nothing may reach that path unresolved."""
        builder = _builder()
        stray = _node("o9", "ghost", visible=False, seg=())
        self.assertIsNone(builder._entity_for(stray, _State({})))

    def test_name_match_wins_over_first_segmentation_hit(self):
        builder = _builder()
        other, apple = _Entity("bowl"), _Entity("apple")
        node = _node("o1", "apple", seg=(4, 5))
        self.assertIs(builder._entity_for(node, _State({4: other, 5: apple})),
                      apple)


class VisibilityPolicyTest(unittest.TestCase):
    def test_keep_tabletop_makes_everything_eligible(self):
        builder = _builder(VISIBILITY_KEEP)
        nodes = {"o1": _node("o1", "apple", visible=False, seg=())}
        builder._apply_visibility(nodes, _State())
        self.assertTrue(nodes["o1"].in_frame)

    def test_projected_camera_needs_a_camera_to_agree(self):
        builder = _builder(VISIBILITY_PROJECTED)
        nodes = {"o1": _node("o1", "apple", visible=False, seg=())}
        builder._apply_visibility(nodes, _State())
        self.assertFalse(nodes["o1"].in_frame)

    def test_a_visible_node_is_eligible_without_projecting(self):
        builder = _builder(VISIBILITY_PROJECTED)
        nodes = {"o1": _node("o1", "apple")}
        builder._apply_visibility(nodes, _State())
        self.assertTrue(nodes["o1"].in_frame)

    def test_an_unknown_policy_is_refused_at_construction(self):
        with self.assertRaises(ValueError):
            GraphBuilder(None, {"selection": {"n_max": 8},
                                "temporal": {"K": 5}},
                         visibility_policy="sometimes")


class SceneSeedingTest(unittest.TestCase):
    """A whitelisted actor that renders no pixels must still be a vertex.

    PickCube's goal marker is the case: hidden before sensor capture and
    collisionless, so it produces neither pixels nor contacts. Mining admits it
    from poses, so without seeding the whitelist declares an object the runtime
    graph can never produce, and the schedule role bound to it never resolves.
    """

    def _seed(self, nodes, entities, admit=None):
        from scenegraph.core.node_builder import seed_scene_nodes
        state = _State({i: e for i, e in enumerate(entities, start=1)})
        seed_scene_nodes(nodes, state, admit=admit)
        return nodes

    def test_an_unrendered_actor_becomes_a_vertex(self):
        nodes = self._seed({"ee": _node("ee", "end_effector")},
                           [_Entity("goal_site")])
        self.assertIn("object:goal_site", nodes)
        node = nodes["object:goal_site"]
        self.assertFalse(node.visible)
        self.assertEqual(node.pixel_area, 0)
        self.assertEqual(node.segmentation_ids, [])
        self.assertEqual(node.source, "scene")

    def test_a_rendered_actor_is_left_alone(self):
        """Seeding must not clobber the pixels a camera did capture."""
        seen = _node("object:cube", "cube", seg=(5,))
        seen.pixel_area = 41
        nodes = self._seed({"object:cube": seen}, [_Entity("cube")])
        self.assertEqual(len(nodes), 1)
        self.assertTrue(nodes["object:cube"].visible)
        self.assertEqual(nodes["object:cube"].pixel_area, 41)

    def test_the_gate_keeps_the_scenery_out(self):
        """Without a whitelist gate this would admit the ground and the walls,
        and capacity would fail on scenery."""
        nodes = self._seed(
            {}, [_Entity("goal_site"), _Entity("ground")],
            admit=lambda e: e.name != "ground",
        )
        self.assertEqual(sorted(nodes), ["object:goal_site"])

    def test_a_seeded_node_carries_no_box(self):
        """bbox all-zero is the agreed signal for no pixels this frame; the
        centroid still comes from the simulator."""
        import numpy as np
        from scenegraph.core.node_builder import fill_bboxes
        nodes = self._seed({}, [_Entity("goal_site")])
        fill_bboxes(nodes, [np.zeros((4, 4), np.int64)])
        self.assertTrue((nodes["object:goal_site"].bbox == 0).all())

    def test_a_seeded_node_resolves_to_its_entity(self):
        """Otherwise every force query on it reads zero, which is emitted as a
        confident not-holds rather than as nothing."""
        builder = _builder()
        goal = _Entity("goal_site")
        nodes = self._seed({}, [goal])
        state = _State({1: goal})
        self.assertIs(builder._entity_for(nodes["object:goal_site"], state), goal)


class SeedGateTest(unittest.TestCase):
    def test_tabletop_seeds_from_the_scene(self):
        builder = _builder(VISIBILITY_KEEP)
        self.assertEqual(builder._seed_gate, builder._entity_admitted)

    def test_projected_camera_does_not(self):
        """MS-HAB's scene is a whole apartment: admissibility is not the
        question there, camera coverage is."""
        builder = _builder(VISIBILITY_PROJECTED)
        self.assertIsNone(builder._seed_gate)


# --------------------------------------------------------------------------- #
# Surface anchors and scoped calibration
# --------------------------------------------------------------------------- #
def _anchored(partner="actor:bin", radial=None):
    from scenegraph.core.affordance import (
        AffordanceSet, BottomComponent, SupportComponent,
    )
    return AffordanceSet(
        support_by_object={"table": [SupportComponent(
            surface_anchor_obj_frame=np.array([0.0, 0.0, 0.9]),
            surface_normal_obj_frame=np.array([0.0, 0.0, 1.0]),
            footprint_radius=0.5,
            partner_key=partner,
        )]},
        bottom_by_object={"bin": [BottomComponent(
            bottom_anchor_obj_frame=np.array([0.0, 0.0, -0.05]),
            bottom_normal_obj_frame=np.array([0.0, 0.0, -1.0]),
            partner_key="actor:table",
            radial_offset=radial,
        )]},
    )


class SupportAnchorSpatialTest(unittest.TestCase):
    """A link origin is not a contact point.

    A table's origin sits ~0.9m below its own top, so origin geometry reported
    a bin resting on it as far-above and a metre away.
    """

    def _pair(self, bin_z, quat=(1.0, 0.0, 0.0, 0.0), radial=None,
              bin_x=0.0, aff=None):
        table = _node("obj-0", "table", pose=(0.0, 0.0, 0.0))
        table.attributes["entity_key"] = "actor:table"
        table.attributes["whitelist_key"] = "actor:table"
        obj = _node("obj-1", "bin", pose=(bin_x, 0.0, bin_z), quat=quat)
        obj.attributes["entity_key"] = "actor:bin"
        obj.attributes["whitelist_key"] = "actor:bin"
        cfg = {"bin_edges": _BINS,
               "affordance_set": aff if aff is not None else _anchored(radial=radial)}
        edges = object_object_spatial_edges(_graph([table, obj], None), None, cfg)
        # Pair order is by key, so ``actor:bin`` is the source: a positive
        # height offset means the bin is above the table.
        return {e.relation: e for e in edges}

    def test_a_resting_pair_is_very_near_and_level(self):
        # Bin bottom at 0.95 - 0.05 = 0.90: exactly the table's top face.
        edges = self._pair(0.95)
        self.assertAlmostEqual(edges["planar-distance"].raw_value, 0.0)
        self.assertAlmostEqual(edges["height-offset"].raw_value, 0.0)
        self.assertEqual(edges["height-offset"].label, "level")
        self.assertEqual(edges["planar-distance"].label, "very-near")

    def test_origin_geometry_would_not_have_agreed(self):
        """The bug this replaces, pinned so it cannot come back quietly."""
        edges = self._pair(0.95)
        self.assertNotAlmostEqual(edges["height-offset"].raw_value, 0.95)

    def test_lifting_changes_only_height(self):
        rest = self._pair(0.95)
        lifted = self._pair(1.25)
        self.assertAlmostEqual(lifted["planar-distance"].raw_value,
                               rest["planar-distance"].raw_value)
        self.assertAlmostEqual(lifted["height-offset"].raw_value, 0.30)
        self.assertEqual(lifted["height-offset"].label, "above")

    def test_lateral_movement_changes_only_planar_distance(self):
        rest = self._pair(0.95)
        moved = self._pair(0.95, bin_x=0.4)
        self.assertAlmostEqual(moved["planar-distance"].raw_value, 0.4)
        self.assertAlmostEqual(moved["height-offset"].raw_value,
                               rest["height-offset"].raw_value)

    def test_anchors_survive_support_ending(self):
        """Physical support is false once lifted; the pair keeps its scale."""
        lifted = self._pair(1.55)
        self.assertEqual(lifted["planar-distance"].label, "very-near")
        self.assertIsNotNone(lifted["height-offset"].label)

    def test_rotating_a_stationary_sphere_changes_nothing(self):
        """A fixed local bottom point would orbit with the ball."""
        upright = self._pair(0.95, radial=0.05)
        spun = self._pair(0.95, quat=(0.0, 1.0, 0.0, 0.0), radial=0.05)
        self.assertAlmostEqual(upright["height-offset"].raw_value,
                               spun["height-offset"].raw_value)
        self.assertAlmostEqual(upright["planar-distance"].raw_value,
                               spun["planar-distance"].raw_value)

    def test_a_rotated_local_anchor_would_have_moved(self):
        """Control for the test above: without the radial form, it does move."""
        upright = self._pair(0.95)
        spun = self._pair(0.95, quat=(0.0, 1.0, 0.0, 0.0))
        self.assertNotAlmostEqual(upright["height-offset"].raw_value,
                                  spun["height-offset"].raw_value)

    def test_an_anchor_mined_against_another_partner_does_not_match(self):
        edges = self._pair(0.95, aff=_anchored(partner="actor:sphere"))
        # Falls back to origins rather than borrowing the sphere's anchor.
        self.assertAlmostEqual(edges["height-offset"].raw_value, 0.95)

    def test_a_pair_with_no_anchor_falls_back_to_origins(self):
        from scenegraph.core.affordance import AffordanceSet
        edges = self._pair(0.95, aff=AffordanceSet())
        self.assertAlmostEqual(edges["height-offset"].raw_value, 0.95)


class ScopedCalibrationTest(unittest.TestCase):
    """One vocabulary, two scales. Change bins must split with absolute ones."""

    def _both(self, cfg, offset):
        a = _node("obj-0", "cubeA", pose=(0.0, 0.0, 0.0))
        b = _node("obj-1", "cubeB", pose=(0.3 + offset, 0.0, 0.0))
        graph = _graph([a, b], None)
        graph.edges.extend(object_object_spatial_edges(graph, None, cfg))
        ee = _node("ee", "ee", pose=(0.0, 0.0, 0.0), node_type="ee")
        ee_graph = _graph([ee, _node("obj-1", "cubeB",
                                     pose=(0.3 + offset, 0.0, 0.0))], None)
        graph.edges.extend(ee_object_spatial_edges(ee_graph, None, cfg))
        return graph

    def test_each_scope_labels_with_its_own_key(self):
        graph = self._both({"bin_edges": _BINS}, 0.0)
        by_src = {(e.src, e.relation): e.bin_key for e in graph.edges}
        self.assertEqual(by_src[("ee", "planar-distance")],
                         spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance"))
        self.assertEqual(by_src[("obj-0", "planar-distance")],
                         spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance"))

    def test_identical_raw_changes_take_different_change_bins(self):
        from scenegraph.core.temporal_buffer import TemporalBuffer

        ee_change = change_bin_key(
            spatial_bin_key(EE_OBJECT_SCOPE, "planar-distance"))
        obj_change = change_bin_key(
            spatial_bin_key(OBJECT_OBJECT_SCOPE, "planar-distance"))
        cfg = {
            "temporal": {"K": 1},
            "bin_edges": {
                **_BINS,
                # Wide for the arm, tight for two objects on a table.
                ee_change: [-0.4, -0.2, 0.2, 0.4],
                obj_change: [-0.04, -0.02, 0.02, 0.04],
            },
        }
        buffer = TemporalBuffer(K=1)
        for offset in (0.0, 0.10):
            graph = self._both(cfg, offset)
            buffer.annotate(graph, cfg)

        by_src = {(e.src, e.relation): e for e in graph.edges}
        ee_edge = by_src[("ee", "planar-distance")]
        obj_edge = by_src[("obj-0", "planar-distance")]
        self.assertAlmostEqual(ee_edge.raw_value, obj_edge.raw_value)
        self.assertEqual(ee_edge.temp_label, "stable")
        self.assertEqual(obj_edge.temp_label, "increase-fast")


class ImmobilePairTest(unittest.TestCase):
    """A pair neither endpoint can move is scene layout, not a fact.

    PlaceSphere's bin and table are both kinematic, so PhysX solves no contact
    between them and no anchor is ever mined. Left in, the pair reports link
    origins -- a table's sits ~0.9m below its own top -- every frame, and its
    fixed offset sets the height scale for every pair that does move.
    """

    def _pair(self, a_type, b_type):
        a = _node("obj-0", "bin", pose=(0.0, 0.0, 0.95))
        b = _node("obj-1", "table", pose=(0.0, 0.0, 0.0))
        a.attributes["body_type"] = a_type
        b.attributes["body_type"] = b_type
        cfg = {"bin_edges": _BINS}
        return object_object_spatial_edges(_graph([a, b], None), None, cfg)

    def test_two_kinematic_bodies_emit_nothing(self):
        self.assertEqual(self._pair("kinematic", "kinematic"), [])

    def test_one_dynamic_endpoint_is_enough(self):
        self.assertTrue(self._pair("dynamic", "kinematic"))

    def test_static_counts_as_immobile(self):
        self.assertEqual(self._pair("static", "kinematic"), [])

    def test_an_unknown_body_type_keeps_the_pair(self):
        """Fail open: a missing field must not silently delete facts."""
        self.assertTrue(self._pair("", ""))

    def test_the_miner_drops_the_same_pairs(self):
        """Mining and runtime have to agree on which pairs exist at all."""
        from scenegraph.adapters.interaction_events import BinStats
        poses = {"bin": [0, 0, 0.95, 1, 0, 0, 0],
                 "table": [0, 0, 0.0, 1, 0, 0, 0],
                 "sphere": [0, 0, 1.0, 1, 0, 0, 0]}
        stats = BinStats(horizon=1)
        for frame in range(3):
            stats.observe(poses, frame, dynamic={"sphere"})
        pairs = {tuple(sorted((r["key_a"], r["key_b"])))
                 for r in stats.pose_samples()}
        self.assertNotIn(("bin", "table"), pairs)
        self.assertEqual(pairs, {("bin", "sphere"), ("sphere", "table")})


class InitialCaptureFrameTest(unittest.TestCase):
    """Frame zero is the reset observation, and its contact buffer lies.

    On GPU sim the pairwise buffer has not been filled by a physics step yet
    and reported PlaceSphere's sphere touching a bin 15cm away, which
    suppressed the one affordance the task is about.
    """

    def _builder(self, frame_cfg=None):
        cfg = {"selection": {"n_max": 8},
               "temporal": {"K": 5},
               "bin_edges": _BINS,
               "affordances": dict(frame_cfg or {})}
        builder = GraphBuilder.__new__(GraphBuilder)
        builder.cfg = cfg
        builder._initial_captured = False
        builder._initial_capture_frame = int(
            cfg["affordances"].get("initial_physical_pair_frame", 1))
        return builder

    def _capture_frames(self, builder, frames):
        seen = []
        for frame in frames:
            capture = (not builder._initial_captured
                       and frame >= builder._initial_capture_frame)
            if capture:
                builder._initial_captured = True
            seen.append(capture)
        return seen

    def test_the_reset_frame_is_skipped(self):
        builder = self._builder()
        self.assertEqual(self._capture_frames(builder, [0, 1, 2, 3]),
                         [False, True, False, False])

    def test_capture_happens_exactly_once(self):
        builder = self._builder()
        self.assertEqual(sum(self._capture_frames(builder, range(20))), 1)

    def test_the_frame_is_configurable(self):
        builder = self._builder({"initial_physical_pair_frame": 3})
        self.assertEqual(self._capture_frames(builder, [0, 1, 2, 3, 4]),
                         [False, False, False, True, False])

    def test_zero_restores_the_reset_frame(self):
        builder = self._builder({"initial_physical_pair_frame": 0})
        self.assertEqual(self._capture_frames(builder, [0, 1]), [True, False])


class WhitelistAdmissionTest(unittest.TestCase):
    """A whitelist that rejects every object must fail, not emit a stub graph.

    PegInsertionSide names its actors ``peg_0`` / ``box_with_hole_0`` per
    sub-scene while the mined whitelist says ``actor:peg`` / ``actor:box_with_hole``.
    Without merged-view aliasing the hard gate drops both, and what survives is
    the end effector and the table -- a graph that stays finite, reports every
    relation as unobserved, and trains on nothing.
    """

    class _Selector:
        def __init__(self, keep):
            self.whitelist = SimpleNamespace(by_key={"actor:peg": {"interacted"}})
            self._keep = keep

        def apply_whitelist(self, nodes):
            return {k: v for k, v in nodes.items() if k in self._keep}

    def _builder(self, keep):
        builder = GraphBuilder.__new__(GraphBuilder)
        builder._checked_admission = False
        builder.selector = self._Selector(keep)
        return builder

    def _nodes(self):
        ee = _node("ee", "ee", node_type="ee")
        peg = _node("actor:peg_0", "peg_0")
        box = _node("actor:box_with_hole_0", "box_with_hole_0")
        return {n.node_id: n for n in (ee, peg, box)}

    def _run(self, builder, keep):
        nodes = self._nodes()
        built = {k: n for k, n in nodes.items() if n.node_type != "ee"}
        builder._check_admission(built, {k: nodes[k] for k in keep})

    def test_rejecting_every_object_raises(self):
        builder = self._builder({"ee"})
        with self.assertRaises(ValueError) as caught:
            self._run(builder, {"ee"})
        message = str(caught.exception)
        self.assertIn("actor:peg_0", message)
        self.assertIn("actor:peg", message)
        self.assertIn("merged-view aliasing", message)

    def test_one_admitted_object_is_enough(self):
        builder = self._builder({"ee", "actor:peg_0"})
        self._run(builder, {"ee", "actor:peg_0"})

    def test_a_scene_with_no_objects_is_not_an_error(self):
        """Nothing was rejected, so there is nothing to diagnose."""
        builder = self._builder({"ee"})
        builder._check_admission({}, {"ee": _node("ee", "ee", node_type="ee")})

    def test_the_check_runs_once(self):
        builder = self._builder({"ee", "actor:peg_0"})
        self._run(builder, {"ee", "actor:peg_0"})
        # Already checked, so a later empty frame does not raise.
        self._run(builder, {"ee"})


if __name__ == "__main__":
    unittest.main()
