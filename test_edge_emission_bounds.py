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


def _cfg():
    bins = {r: [0.1, 0.2, 0.3, 0.4] for r in rr.SPATIAL_RELATIONS}
    bins.update({r: [1 / 3, 2 / 3] for r in rr.AFFORDANCE_RELATIONS})
    return {
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


def _stubs():
    """Make every affordance lookup resolve so nothing is skipped for data."""
    p = [mock.patch.object(rr, name, return_value=_SENTINEL) for name in (
        "lookup_components", "lookup_contact_components",
        "lookup_support_components", "lookup_bottom_components",
        "lookup_contain_components", "lookup_key_components")]
    p += [mock.patch.object(rr, name, return_value=None) for name in (
        "obj_contact_compatibility", "support_compatibility",
        "contain_compatibility", "compatibility_components",
        "_compatibility_score")]
    p.append(mock.patch.object(rr, "contain_holds", return_value=False))
    p.append(mock.patch.object(
        rr, "_resolve_active_anchor",
        return_value=(np.zeros(3), object(), 0)))
    return p


def _emit(n_objects):
    graph, = (_graph(n_objects),)
    state, cfg = _state(graph), _cfg()
    stubs = _stubs()
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
