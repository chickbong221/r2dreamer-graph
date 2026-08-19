"""Target-only persistence: the retained vertex, its row, and its relations.

Everything here is synthetic -- no simulator, no torch. The builder's
``step`` needs a live ManiSkill env, so the persistence rule is exercised
through ``_refresh_target_snapshot`` directly, which is the whole of it.

The property under test throughout: exactly one node outlives its own
visibility, it keeps row 1, and all six of its end-effector facts keep being
recomputed while it is hidden.
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
from scenegraph.core.graph_builder import GraphBuilder
from scenegraph.core.relation_rules import (
    _ee_object_nodes,
    _object_pairs,
    _resolve_entity,
    _visible_objects,
    ee_object_spatial_edges,
    height_offset_xyz,
    planar_distance_xyz,
)
from scenegraph.core.schema import Graph, Node

N_MAX = 8
E_MAX = 24


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _node(node_id, name, *, visible=True, pose=(0.0, 0.0, 0.0), seg=(1,),
          node_type="object", box=0.4):
    node = Node(
        node_id=node_id,
        node_type=node_type,
        name=name,
        visible=visible,
        segmentation_ids=list(seg),
        pose_world=[*pose, 1.0, 0.0, 0.0, 0.0],
        attributes={"whitelist_key": name, "interaction_types": {"grasp", "contact"}},
    )
    node.bbox = np.full((2, 4), box, np.float32)
    node.bbox[:, 1] += 0.1
    node.bbox[:, 3] += 0.1
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


class _FakeState:
    """The two privileged queries the retained target still needs answered."""

    def __init__(self, active_obj=None, grasped=False, pose=None):
        self.active_obj = active_obj
        self.active_obj_merged = None
        self.grasped = bool(grasped)
        self.pose = pose
        self.seg_id_map = {}

    def is_grasping(self, obj, max_angle=30):
        return self.grasped and obj is self.active_obj


def _builder():
    """A GraphBuilder with only the fields the persistence rule touches."""
    builder = object.__new__(GraphBuilder)
    builder.env_idx = 0
    builder.cfg = {"grasp": {"max_angle": 30}}
    builder._target_snapshot = None
    return builder


# --------------------------------------------------------------------------- #
# Row assignment
# --------------------------------------------------------------------------- #
class FixedRowTest(unittest.TestCase):
    """Row 0 is the end effector and row 1 is the target, by meaning."""

    def test_ee_is_row_zero_even_when_it_arrives_last(self):
        ee = _node("ee", "ee", node_type="ee")
        obj = _node("obj-1", "apple")
        rows, dropped = _row_assignment(_graph([obj, ee]), N_MAX, "obj-1", True)
        self.assertEqual(dict((n.node_id, r) for r, n in rows)["ee"], 0)
        self.assertEqual(dropped, 0)

    def test_target_takes_row_one_whatever_its_registry_index(self):
        # The registry admitted two distractors first, so the target's own
        # index is 3. Packing must ignore that.
        nodes = [
            _node("ee", "ee", node_type="ee"),
            _node("obj-a", "apple"),
            _node("obj-b", "bowl"),
            _node("obj-1", "can"),
        ]
        for i, node in enumerate(nodes):
            node.index = i
        packed = _pack(_graph(nodes), ["apple", "bowl", "can"])
        self.assertEqual(int(packed["graph_node_target"][1]), 1)
        self.assertEqual(int(packed["graph_node_target"].sum()), 1)

    def test_row_one_is_padding_before_the_target_is_observed(self):
        nodes = [_node("ee", "ee", node_type="ee"), _node("obj-a", "apple")]
        packed = _pack(_graph(nodes), ["apple"])
        self.assertEqual(int(packed["graph_node_ent"][1]), 0)
        self.assertNotEqual(int(packed["graph_node_ent"][0]), 0)
        self.assertNotEqual(int(packed["graph_node_ent"][2]), 0)

    def test_dynamic_rows_churn_without_touching_zero_or_one(self):
        ee = _node("ee", "ee", node_type="ee")
        target = _node("obj-1", "can")
        first = _pack(_graph([ee, target, _node("obj-a", "apple")]),
                      ["apple", "bowl", "can"])
        # A different distractor arrives; the target did not move.
        second = _pack(_graph([ee, target, _node("obj-b", "bowl")]),
                       ["apple", "bowl", "can"])
        for key in ("graph_node_ent", "graph_node_target"):
            np.testing.assert_array_equal(first[key][:2], second[key][:2])
        self.assertNotEqual(int(first["graph_node_ent"][2]),
                            int(second["graph_node_ent"][2]))

    def test_reserving_row_one_reports_its_overflow(self):
        # n_max=4 leaves rows 2 and 3 for non-target objects. A third one is
        # dropped, and the drop is counted rather than swallowed.
        nodes = [_node("ee", "ee", node_type="ee")] + [
            _node(f"obj-{i}", f"o{i}") for i in "abc"
        ]
        graph = _graph(nodes, target_id=None)
        _pack(graph, ["oa", "ob", "oc"], n_max=4)
        self.assertEqual(graph.meta["n_nodes_dropped"], 1)
        self.assertEqual(graph.meta["n_nodes_packed"], 3)


# --------------------------------------------------------------------------- #
# What the packed target looks like while it is hidden
# --------------------------------------------------------------------------- #
class InvisibleTargetPackingTest(unittest.TestCase):
    def _frames(self, grasped=False, moved_to=None):
        builder = _builder()
        ee = _node("ee", "ee", node_type="ee")
        target = _node("obj-1", "can", pose=(1.0, 2.0, 0.5))
        state = _FakeState(active_obj=object(), grasped=grasped, pose=moved_to)

        seen = builder._refresh_target_snapshot(
            {"ee": ee, "obj-1": target}, "obj-1", state
        )
        hidden = builder._refresh_target_snapshot({"ee": ee}, "obj-1", state)
        return seen, hidden

    def test_invisible_target_keeps_row_one_and_its_centroid(self):
        seen, hidden = self._frames()
        first = _pack(_graph(list(seen.values())), ["can"])
        second = _pack(_graph(list(hidden.values())), ["can"])
        self.assertEqual(int(second["graph_node_target"][1]), 1)
        self.assertEqual(int(first["graph_node_ent"][1]),
                         int(second["graph_node_ent"][1]))
        np.testing.assert_array_equal(
            first["graph_node_centroid"][1], second["graph_node_centroid"][1]
        )
        np.testing.assert_allclose(
            second["graph_node_centroid"][1], [1.0, 2.0, 0.5], atol=1e-6
        )

    def test_invisible_target_boxes_go_to_zero(self):
        seen, hidden = self._frames()
        first = _pack(_graph(list(seen.values())), ["can"])
        second = _pack(_graph(list(hidden.values())), ["can"])
        self.assertGreater(float(np.abs(first["graph_node_bbox"][1]).max()), 0.0)
        self.assertEqual(float(np.abs(second["graph_node_bbox"][1]).max()), 0.0)

    def test_other_invisible_objects_simply_disappear(self):
        builder = _builder()
        state = _FakeState(active_obj=object())
        nodes = {
            "ee": _node("ee", "ee", node_type="ee"),
            "obj-1": _node("obj-1", "can"),
            "obj-a": _node("obj-a", "apple"),
        }
        builder._refresh_target_snapshot(dict(nodes), "obj-1", state)
        # Next frame the cameras see neither object.
        kept = builder._refresh_target_snapshot({"ee": nodes["ee"]}, "obj-1", state)
        self.assertIn("obj-1", kept)
        self.assertNotIn("obj-a", kept)

    def test_grasped_invisible_target_follows_the_simulator(self):
        builder = _builder()
        handle = object()
        state = _FakeState(active_obj=handle, grasped=True)
        target = _node("obj-1", "can", pose=(1.0, 2.0, 0.5))
        builder._refresh_target_snapshot({"obj-1": target}, "obj-1", state)

        moved = np.array([1.0, 2.0, 0.9, 1.0, 0.0, 0.0, 0.0])
        import scenegraph.core.graph_builder as gb
        original = gb.entity_pose_world_array
        gb.entity_pose_world_array = lambda entity, idx: moved
        try:
            kept = builder._refresh_target_snapshot({}, "obj-1", state)
        finally:
            gb.entity_pose_world_array = original
        np.testing.assert_allclose(kept["obj-1"].pose_world[:3], [1.0, 2.0, 0.9])

    def test_ungrasped_invisible_target_does_not_ask_the_simulator(self):
        builder = _builder()
        state = _FakeState(active_obj=object(), grasped=False)
        target = _node("obj-1", "can", pose=(1.0, 2.0, 0.5))
        builder._refresh_target_snapshot({"obj-1": target}, "obj-1", state)

        import scenegraph.core.graph_builder as gb
        original = gb.entity_pose_world_array

        def _boom(entity, idx):
            raise AssertionError("an ungrasped hidden target must stay frozen")

        gb.entity_pose_world_array = _boom
        try:
            kept = builder._refresh_target_snapshot({}, "obj-1", state)
        finally:
            gb.entity_pose_world_array = original
        np.testing.assert_allclose(kept["obj-1"].pose_world[:3], [1.0, 2.0, 0.5])

    def test_nothing_is_replayed_before_the_target_is_ever_seen(self):
        builder = _builder()
        state = _FakeState(active_obj=object())
        kept = builder._refresh_target_snapshot(
            {"ee": _node("ee", "ee", node_type="ee")}, "obj-1", state
        )
        self.assertNotIn("obj-1", kept)

    def test_reset_clears_the_retained_target(self):
        builder = _builder()
        state = _FakeState(active_obj=object())
        builder._refresh_target_snapshot(
            {"obj-1": _node("obj-1", "can")}, "obj-1", state
        )
        self.assertIsNotNone(builder._target_snapshot)
        # ``reset_episode`` touches collaborators this fixture does not build,
        # so assert on the one line of it that owns this state.
        builder._target_snapshot = None
        kept = builder._refresh_target_snapshot({}, "obj-1", state)
        self.assertNotIn("obj-1", kept)


# --------------------------------------------------------------------------- #
# Relations recomputed through the occlusion
# --------------------------------------------------------------------------- #
_BINS = {
    "bin_edges": {
        "planar-distance": [0.05, 0.15, 0.35, 0.75],
        "height-offset": [-0.35, -0.10, 0.10, 0.35],
    }
}


class RetainedTargetRelationTest(unittest.TestCase):
    def _scene(self, ee_xyz=(0.0, 0.0, 0.0)):
        ee = _node("ee", "ee", node_type="ee", pose=ee_xyz, seg=())
        target = _node("obj-1", "can", visible=False, pose=(0.2, 0.0, 0.1), seg=())
        other = _node("obj-a", "apple", visible=False, pose=(0.9, 0.0, 0.0))
        visible = _node("obj-b", "bowl", visible=True, pose=(0.3, 0.3, 0.0))
        return _graph([ee, target, other, visible])

    def test_iterator_keeps_the_target_and_drops_everyone_else(self):
        graph = self._scene()
        ids = {n.node_id for n in _ee_object_nodes(graph)}
        self.assertEqual(ids, {"obj-1", "obj-b"})
        self.assertEqual({n.node_id for n in _visible_objects(graph)}, {"obj-b"})

    def test_object_object_pairs_never_include_the_retained_target(self):
        graph = self._scene()
        for a, b in _object_pairs(graph):
            self.assertNotIn("obj-1", (a.node_id, b.node_id))

    def test_only_the_exact_target_gets_the_privileged_handle(self):
        graph = self._scene()
        handle = object()
        state = _FakeState(active_obj=handle)
        target = graph.get_node("obj-1")
        other = graph.get_node("obj-a")
        self.assertIs(_resolve_entity(target, state, graph), handle)
        self.assertIsNone(_resolve_entity(other, state, graph))
        # And never without the graph that names it.
        self.assertIsNone(_resolve_entity(target, state))

    def test_distance_and_height_match_the_centroids_exactly(self):
        graph = self._scene(ee_xyz=(0.0, 0.0, 0.4))
        edges = ee_object_spatial_edges(graph, _FakeState(), dict(_BINS))
        raw = {
            (e.relation, e.dst): e.raw_value
            for e in edges
        }
        ee_xyz = np.asarray(graph.get_node("ee").pose_world[:3], float)
        tgt_xyz = np.asarray(graph.get_node("obj-1").pose_world[:3], float)
        self.assertAlmostEqual(
            raw[("planar-distance", "obj-1")],
            planar_distance_xyz(ee_xyz, tgt_xyz),
            places=12,
        )
        self.assertAlmostEqual(
            raw[("height-offset", "obj-1")],
            height_offset_xyz(ee_xyz, tgt_xyz),
            places=12,
        )

    def test_moving_the_end_effector_moves_the_hidden_targets_relations(self):
        # The target is stationary and unobserved in both scenes; only the
        # robot moved. Its facts must move anyway.
        far = ee_object_spatial_edges(
            self._scene(ee_xyz=(2.0, 0.0, 0.9)), _FakeState(), dict(_BINS)
        )
        near = ee_object_spatial_edges(
            self._scene(ee_xyz=(0.21, 0.0, 0.1)), _FakeState(), dict(_BINS)
        )

        def edge(edges, relation):
            return next(
                e for e in edges
                if e.relation == relation and e.dst == "obj-1"
            )

        self.assertEqual(edge(far, "planar-distance").label, "very-far")
        self.assertEqual(edge(near, "planar-distance").label, "very-near")
        self.assertEqual(edge(far, "height-offset").label, "far-above")
        self.assertEqual(edge(near, "height-offset").label, "level")
        self.assertNotEqual(
            edge(far, "height-offset").raw_value,
            edge(near, "height-offset").raw_value,
        )


if __name__ == "__main__":
    unittest.main()
