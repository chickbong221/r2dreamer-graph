"""How many facts one maximally-admissible graph emits, against e_max.

The configured ceiling is documented as ``3 * n_max * (n_max - 1)`` = 168, but
that is the count only when each unordered pair emits six facts. Support and
contain currently emit both orderings, in the physical family and again in the
affordance family, so the real legacy count is higher and the packer truncates.

Affordance component lookups and compatibility maths are stubbed: this measures
emission structure, not affordance scoring.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml

from scenegraph.core import relation_rules as rr
from scenegraph.core.schema import Graph, Node

ALL_TYPES = ["contact", "grasp", "support", "contain"]
_SENTINEL = [object()]


def _cfg(contract=rr.EDGE_CONTRACT_LEGACY):
    bins = {r: [0.1, 0.2, 0.3, 0.4] for r in rr.SPATIAL_RELATIONS}
    bins.update({r: [1 / 3, 2 / 3] for r in rr.AFFORDANCE_RELATIONS})
    return {
        "edge_contract": contract,
        "bin_edges": bins,
        "contact": {"eps_force": 0.05},
        "grasp": {"max_angle": 30, "tcp_approach_axis_local": [0.0, 0.0, 1.0]},
        "support": {"eps_z": 0.02, "min_vertical_force_ratio": 0.5},
        # All three mirror thresholds.yaml; the code defaults
        # object_object_support_compatibility to False.
        "affordances": {"object_object_compatibility": True,
                        "object_object_contact_compatibility": True,
                        "object_object_support_compatibility": True,
                        "object_object_contain_compatibility": True,
                        "object_object_compat_max_distance": 2.0},
        "affordance_set": object(),
        "compat_norm": {"pos": 0.1, "orient": 1.57, "width": 0.04, "xy": 0.05,
                        "vertical": 0.03, "radial": 0.02, "axial": 0.03},
    }


def _graph(n_objects):
    ee = Node("ee", "ee", "ee", pose_world=[0.0] * 3 + [1.0, 0, 0, 0], index=0)
    nodes = [ee]
    for i in range(n_objects):
        nodes.append(Node(
            f"actor:o{i}", "object", f"o{i}",
            visible=True, segmentation_ids=[i + 1],
            pose_world=[0.01 * i, 0.01 * i, 0.01 * i, 1.0, 0, 0, 0],
            index=i + 1,
            attributes={"interaction_types": list(ALL_TYPES),
                        "whitelist_key": f"actor:o{i}"},
        ))
    return Graph(frame=0, env_id="t", camera="c", nodes=nodes, meta={})


def _state(graph):
    seg = {n.segmentation_ids[0]: object()
           for n in graph.nodes if n.segmentation_ids}
    return SimpleNamespace(
        tcp_pose_world=np.array([0.0, 0, 0, 1.0, 0, 0, 0]),
        gripper_width=0.04,
        seg_id_map=seg,
        pairwise_force_vector=lambda a, b: np.zeros(3),
        pairwise_force=lambda a, b: 0.0,
        ee_object_contact_force=lambda o: 1.0,
        is_grasping=lambda o, **k: True,
        active_obj=None,
    )


def _role(ids):
    """Components resolve for one role per node, as a mined asset gives."""
    return lambda _aff, node: _SENTINEL if node.node_id in ids else None


def _stubs(roles=None):
    """Stub affordance lookups. ``roles`` splits container/containee so no pair
    claims both orientations, which canonical_v2 refuses."""
    symmetric = ("lookup_components", "lookup_contact_components")
    directed = ("lookup_support_components", "lookup_contain_components",
                "lookup_bottom_components", "lookup_key_components")
    p = [mock.patch.object(rr, n, return_value=_SENTINEL) for n in symmetric]
    if roles is None:
        p += [mock.patch.object(rr, n, return_value=_SENTINEL) for n in directed]
    else:
        upper, lower = roles
        p += [mock.patch.object(rr, n, side_effect=_role(upper))
              for n in ("lookup_support_components", "lookup_contain_components")]
        p += [mock.patch.object(rr, n, side_effect=_role(lower))
              for n in ("lookup_bottom_components", "lookup_key_components")]
    p += [mock.patch.object(rr, n, return_value=None) for n in (
        "obj_contact_compatibility", "support_compatibility",
        "contain_compatibility", "compatibility_components",
        "_compatibility_score")]
    p.append(mock.patch.object(rr, "contain_holds", return_value=False))
    p.append(mock.patch.object(
        rr, "_resolve_active_anchor",
        return_value=(np.zeros(3), object(), 0)))
    return p


def _emit(n_objects, contract=rr.EDGE_CONTRACT_LEGACY, roles=None,
          nodes=None):
    graph = _graph(n_objects) if nodes is None else nodes
    state, cfg = _state(graph), _cfg(contract)
    stubs = _stubs(roles)
    for s in stubs:
        s.start()
    try:
        rr.build_absolute_edges(graph, state, cfg)
    finally:
        for s in stubs:
            s.stop()
    return graph.edges


def _by_relation(edges, ee=False):
    """Counts per relation, for ee-object edges or object-object edges."""
    out = {}
    for e in edges:
        if (e.src == "ee") != ee:
            continue
        out[e.relation] = out.get(e.relation, 0) + 1
    return out


class LegacyEmissionTest(unittest.TestCase):
    def test_support_and_contain_emit_both_orderings(self):
        counts = _by_relation(_emit(2))          # one object-object pair
        self.assertEqual(counts["contact"], 1)
        self.assertEqual(counts["support"], 2)
        self.assertEqual(counts["contain"], 2)
        self.assertEqual(counts["support-compatibility"], 2)
        self.assertEqual(counts["contain-compatibility"], 2)
        self.assertEqual(counts["contact-compatibility"], 1)

    def test_ee_object_pair_emits_six(self):
        counts = _by_relation(_emit(1), ee=True)
        self.assertEqual(sum(counts.values()), 6)

    def test_full_graph_exceeds_configured_e_max(self):
        n_max = 8
        edges = _emit(n_max - 1)
        with open("configs/model/_base_.yaml") as f:
            e_max = yaml.safe_load(f)["graph"]["e_max"]
        self.assertEqual(e_max, 3 * n_max * (n_max - 1))
        # 7 ee pairs x 6, plus 21 object pairs x 10.
        self.assertEqual(len(edges), 42 + 210)
        self.assertGreater(len(edges), e_max)

    def test_canonical_v2_target_would_fit(self):
        """One edge per unordered pair for support/contain lands exactly on 168."""
        edges = _emit(8 - 1)
        pair = _by_relation(edges)
        halved = sum(
            c // 2 if r in ("support", "contain",
                            "support-compatibility", "contain-compatibility")
            else c
            for r, c in pair.items())
        ee = sum(_by_relation(edges, ee=True).values())
        self.assertEqual(halved + ee, 3 * 8 * (8 - 1))


if __name__ == "__main__":
    unittest.main()


CANON = rr.EDGE_CONTRACT_CANONICAL
_ROLES = ({"actor:o0"}, {"actor:o1"})   # o0 supports/contains o1


class CanonicalV2EmissionTest(unittest.TestCase):
    def test_object_pair_emits_six(self):
        counts = _by_relation(_emit(2, CANON, _ROLES))
        self.assertEqual(counts, {
            "contact": 1, "support": 1, "contain": 1,
            "contact-compatibility": 1, "support-compatibility": 1,
            "contain-compatibility": 1,
        })

    def test_ee_pair_unchanged(self):
        legacy = _by_relation(_emit(1), ee=True)
        canon = _by_relation(_emit(1, CANON, _ROLES), ee=True)
        self.assertEqual(legacy, canon)

    def test_ceiling_is_exactly_e_max(self):
        with open("configs/model/_base_.yaml") as f:
            e_max = yaml.safe_load(f)["graph"]["e_max"]
        n_max = 8
        per_ee = sum(_by_relation(_emit(1, CANON, _ROLES), ee=True).values())
        per_pair = sum(_by_relation(_emit(2, CANON, _ROLES)).values())
        total = (n_max - 1) * per_ee + 21 * per_pair
        self.assertEqual(total, e_max)

    def test_direction_does_not_flip_when_predicate_is_false(self):
        """The bug one-direction emission could have introduced."""
        edges = _emit(2, CANON, _ROLES)
        support = [e for e in edges if e.relation == "support"]
        self.assertEqual(support[0].label, rr.NOT_HOLDS)
        self.assertEqual((support[0].src, support[0].dst),
                         ("actor:o0", "actor:o1"))

    def test_supporter_role_rides_in_the_label(self):
        graph = _graph(2)
        a, b = graph.nodes[1], graph.nodes[2]
        # Force b under a: get_pairwise_contact_forces is "force on a due to b",
        # so fz < 0 makes a the supporter.
        state, cfg = _state(graph), _cfg(CANON)
        state.pairwise_force_vector = lambda x, y: np.array([0.0, 0.0, -1.0])
        stubs = _stubs(_ROLES)
        for s in stubs:
            s.start()
        try:
            rr.build_absolute_edges(graph, state, cfg)
        finally:
            for s in stubs:
                s.stop()
        support = [e for e in graph.edges if e.relation == "support"]
        self.assertEqual(len(support), 1)
        self.assertEqual(support[0].label, rr.SRC_HOLDS)
        self.assertEqual(support[0].src, a.node_id)
        self.assertEqual(support[0].dst, b.node_id)

    def test_ambiguous_role_raises(self):
        with self.assertRaises(ValueError):
            _emit(2, CANON, roles=None)      # both orientations resolve


class PairOrderStabilityTest(unittest.TestCase):
    def _orientations(self, graph):
        pairs = rr._object_pairs(graph, CANON)
        return [(a.node_id, b.node_id) for a, b in pairs]

    def test_node_order_does_not_change_orientation(self):
        g1 = _graph(4)
        g2 = _graph(4)
        g2.nodes = [g2.nodes[0]] + list(reversed(g2.nodes[1:]))
        self.assertEqual(self._orientations(g1), self._orientations(g2))

    def test_legacy_follows_node_order(self):
        """Documents why canonical ordering is a real change, not a no-op."""
        g1, g2 = _graph(4), _graph(4)
        g2.nodes = [g2.nodes[0]] + list(reversed(g2.nodes[1:]))
        legacy = lambda g: [(a.node_id, b.node_id)
                            for a, b in rr._object_pairs(g)]
        self.assertNotEqual(legacy(g1), legacy(g2))

    def test_index_reuse_does_not_flip_a_pair(self):
        g = _graph(3)
        for i, n in enumerate(g.nodes[1:]):
            n.index = 3 - i          # registry hands out different slots
        self.assertEqual(self._orientations(g), self._orientations(_graph(3)))


class ContractVocabTest(unittest.TestCase):
    def test_absolute_vocab_widths(self):
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        self.assertEqual(len(build_absolute_vocab(rr.EDGE_CONTRACT_LEGACY)), 17)
        self.assertEqual(len(build_absolute_vocab(CANON)), 19)

    def test_directional_labels_only_in_canonical(self):
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        legacy = build_absolute_vocab(rr.EDGE_CONTRACT_LEGACY).token_to_id
        canon = build_absolute_vocab(CANON).token_to_id
        self.assertNotIn(rr.SRC_HOLDS, legacy)
        self.assertIn(rr.SRC_HOLDS, canon)
        self.assertIn(rr.DST_HOLDS, canon)

    def test_support_labels_are_non_contiguous_in_canonical(self):
        """Why the hardcoded slice mask in graph.py could not be kept."""
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        vocab = build_absolute_vocab(CANON)
        contact = {vocab.encode(l)
                   for l in rr.abs_labels_for(CANON)["contact"]}
        support = {vocab.encode(l)
                   for l in rr.abs_labels_for(CANON)["support"]}
        self.assertEqual(contact, {1, 2})
        self.assertEqual(support, {1, 3, 4})

    def test_unknown_contract_raises(self):
        with self.assertRaises(ValueError):
            rr.abs_labels_for("v3")
