"""Retention: which vertices survive, what they carry, and what they relate to.

Everything here is synthetic -- no simulator, no torch. Retention is now
unconditional, so there is no snapshot rule left to exercise; what matters is
that a node without pixels keeps its row and its live centroid, that ``in_frame``
rather than ``visible`` decides eligibility, and that the protected target still
pairs with whatever the cameras do cover.
"""

import unittest
from typing import Optional

import numpy as np

from scenegraph.adapters.graph_pack import (
    SCHEMA_SIMPLE_POOLED,
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
          seg=(1,), node_type="object", box=0.4):
    node = Node(
        node_id=node_id,
        node_type=node_type,
        name=name,
        visible=visible,
        in_frame=visible if in_frame is None else in_frame,
        segmentation_ids=list(seg),
        pose_world=[*pose, 1.0, 0.0, 0.0, 0.0],
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
        graph, _vocab(names), n_max=n_max, e_max=E_MAX, n_cams=2, app_dim=4,
        schema=SCHEMA_SIMPLE_POOLED, uid_vocab=256,
    )


# Labels come from the relation vocabulary; cfg supplies edges only.
_BINS = {
    "planar-distance": [0.1, 0.2, 0.6, 1.0],
    "height-offset": [-0.4, -0.1, 0.1, 0.4],
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
    """The two privileged lookups retention needs. ``step`` wants a live env;
    these methods do not."""

    active_subtask_type = "pick"

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


if __name__ == "__main__":
    unittest.main()
