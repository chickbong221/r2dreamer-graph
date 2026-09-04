"""How many facts one maximally-admissible graph emits, against e_max.

Each physical object pair emits at most six physical/compatibility facts,
plus two spatial facts when object_object_spatial is enabled. Directional
relations emit once per pair. Overflow raises rather than truncating.

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
from scenegraph.core.sites import SITE_EE_REST, SiteDeclaration, SiteSpec

ALL_TYPES = ["contact", "grasp", "support", "contain"]
_SENTINEL = [object()]


def _cfg():
    bins = {rr.spatial_bin_key(scope, r): [0.1, 0.2, 0.3, 0.4]
            for scope in rr.SPATIAL_SCOPES for r in rr.SPATIAL_RELATIONS}
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
        # ManiSkill's setting: the object-to-object approach ladders exist.
        "object_object_spatial": True,
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
    claims both orientations, which the contract refuses."""
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


def _emit(n_objects, roles=None, nodes=None, cfg=None):
    graph = _graph(n_objects) if nodes is None else nodes
    state, cfg = _state(graph), _cfg() if cfg is None else cfg
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


_ROLES = ({"actor:o0"}, {"actor:o1"})   # o0 supports/contains o1


class EmissionTest(unittest.TestCase):
    def test_object_pair_emits_eight(self):
        """Six physical/affordance facts plus the two spatial ladders a
        schedule needs for object-to-object approach."""
        counts = _by_relation(_emit(2, _ROLES))
        self.assertEqual(counts, {
            "contact": 1, "support": 1, "contain": 1,
            "contact-compatibility": 1, "support-compatibility": 1,
            "contain-compatibility": 1,
            "planar-distance": 1, "height-offset": 1,
        })

    def test_ee_pair_unchanged(self):
        legacy = _by_relation(_emit(1), ee=True)
        canon = _by_relation(_emit(1, _ROLES), ee=True)
        self.assertEqual(legacy, canon)

    def test_disable_object_pairs_skips_work_and_preserves_ee_facts(self):
        expected = [e for e in _emit(2, _ROLES) if "ee" in (e.src, e.dst)]
        for spatial in (False, True):
            with self.subTest(object_object_spatial=spatial):
                cfg = _cfg()
                cfg.update(disable_object_object_relations=True,
                           object_object_spatial=spatial)
                with mock.patch.object(rr, "_object_pairs", side_effect=AssertionError(
                        "disabled pairs must not even be enumerated")):
                    actual = _emit(2, _ROLES, cfg=cfg)
                self.assertEqual(actual, expected)
                self.assertTrue(all("ee" in (e.src, e.dst) for e in actual))
                cfg["disable_object_object_relations"] = False
                restored = _emit(2, _ROLES, cfg=cfg)
                self.assertGreater(len(restored), len(actual))

    def test_disabled_object_pairs_require_only_ee_calibration(self):
        cfg = _cfg()
        cfg["disable_object_object_relations"] = True
        keys = rr.required_bin_keys(cfg)
        self.assertIn("grasp-compatibility", keys)
        self.assertIn("contact-compatibility", keys)
        self.assertIn("ee-object-planar-distance", keys)
        self.assertIn("ee-object-height-offset", keys)
        for key in ("object-object-planar-distance", "object-object-height-offset",
                    "support-compatibility", "contain-compatibility"):
            self.assertNotIn(key, keys)

    def test_mshab_twelve_node_edge_bound_includes_physical_pairs_and_site(self):
        cfg = _cfg()
        cfg["object_object_spatial"] = False
        per_ee = sum(_by_relation(_emit(1, _ROLES, cfg=cfg), ee=True).values())
        pairs = _by_relation(_emit(2, _ROLES, cfg=cfg))
        self.assertEqual(per_ee, 6)
        self.assertNotIn("planar-distance", pairs)
        self.assertNotIn("height-offset", pairs)
        self.assertEqual(sum(pairs.values()), 6)

        graph = _graph(0)
        pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        graph.nodes.append(Node(
            SITE_EE_REST, "object", "rest", pose_world=pose.tolist(),
            attributes={"whitelist_key": SITE_EE_REST,
                        "interaction_types": []},
        ))
        declaration = SiteDeclaration(
            key=SITE_EE_REST, site_type="point", subject_key="ee",
            metric="euclidean", source="origin", provider="mshab_ee_rest",
            provenance="MS-HAB Pick capacity bound",
        )
        cfg["site_declarations"] = {SITE_EE_REST: declaration}
        cfg["site_specs"] = [SiteSpec(declaration, pose, 0.05)]
        cfg["bin_edges"].update({
            rr.EE_SITE_PLANAR_KEY: [0.1, 0.2, 0.3, 0.4],
            rr.EE_SITE_HEIGHT_KEY: [-0.2, -0.1, 0.1, 0.2],
        })
        site_edges = list(_emit(0, nodes=graph, cfg=cfg))
        self.assertCountEqual([e.relation for e in site_edges],
                              ["planar-distance", "height-offset", "reached"])
        cfg["disable_object_object_relations"] = True
        graph.edges.clear()
        self.assertEqual(_emit(0, nodes=graph, cfg=cfg), site_edges)
        # EE + one empty-token virtual site leave at most ten physical nodes.
        bound = 10 * per_ee + (10 * 9 // 2) * sum(pairs.values()) + len(site_edges)
        self.assertEqual(bound, 333)
        self.assertLessEqual(bound, 384)

    def test_saturated_graph_needs_more_than_configured_e_max(self):
        """n_max=8 is one end effector plus seven objects, so a saturated
        canonical graph is 21*8 + 7*6 = 210. e_max stays at 168 because the
        scenes actually run hold far fewer -- and overflow now raises."""
        with open("configs/model/_base_.yaml") as f:
            e_max = yaml.safe_load(f)["graph"]["e_max"]
        per_ee = sum(_by_relation(_emit(1, _ROLES), ee=True).values())
        per_pair = sum(_by_relation(_emit(2, _ROLES)).values())
        total = 7 * per_ee + 21 * per_pair
        self.assertEqual(total, 210)
        self.assertGreater(total, e_max)

    def test_three_object_tabletop_scene_fits(self):
        """Table plus two task objects, which is every selected task."""
        with open("configs/model/_base_.yaml") as f:
            e_max = yaml.safe_load(f)["graph"]["e_max"]
        edges = _emit(3, _ROLES)
        self.assertLessEqual(len(edges), e_max)

    def test_direction_does_not_flip_when_predicate_is_false(self):
        """The bug one-direction emission could have introduced."""
        edges = _emit(2, _ROLES)
        support = [e for e in edges if e.relation == "support"]
        self.assertEqual(support[0].label, rr.NOT_HOLDS)
        self.assertEqual((support[0].src, support[0].dst),
                         ("actor:o0", "actor:o1"))

    def test_supporter_role_rides_in_the_label(self):
        graph = _graph(2)
        a, b = graph.nodes[1], graph.nodes[2]
        # Force b under a: get_pairwise_contact_forces is "force on a due to b",
        # so fz < 0 makes a the supporter.
        state, cfg = _state(graph), _cfg()
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

    def test_initial_physical_pair_keeps_facts_but_loses_affordances(self):
        graph = _graph(2)
        state, cfg = _state(graph), _cfg()
        state.pairwise_force_vector = lambda x, y: np.array([0.0, 0.0, -1.0])
        initial = set()
        stubs = _stubs(_ROLES)
        for s in stubs:
            s.start()
        try:
            rr.build_absolute_edges(
                graph, state, cfg, initial_physical_pairs=initial,
                capture_initial=True,
            )
        finally:
            for s in stubs:
                s.stop()

        self.assertEqual(initial, {("actor:o0", "actor:o1")})
        counts = _by_relation(graph.edges)
        self.assertEqual(counts, {
            "contact": 1, "support": 1, "contain": 1,
            "planar-distance": 1, "height-offset": 1,
        })

        # The decision is episode-scoped: compatibility must not reappear just
        # because the initially supported object is later lifted away. Capture
        # is off from here, so a pair that turns physical later is unaffected.
        later = _graph(2)
        later.frame = 1
        later_state = _state(later)
        stubs = _stubs(_ROLES)
        for s in stubs:
            s.start()
        try:
            rr.build_absolute_edges(
                later, later_state, cfg, initial_physical_pairs=initial,
            )
        finally:
            for s in stubs:
                s.stop()
        later_counts = _by_relation(later.edges)
        self.assertNotIn("contact-compatibility", later_counts)
        self.assertNotIn("support-compatibility", later_counts)
        self.assertNotIn("contain-compatibility", later_counts)

    def test_initially_separate_pair_keeps_affordances(self):
        graph = _graph(2)
        state, cfg = _state(graph), _cfg()
        initial = set()
        stubs = _stubs(_ROLES)
        for s in stubs:
            s.start()
        try:
            rr.build_absolute_edges(
                graph, state, cfg, initial_physical_pairs=initial,
                capture_initial=True,
            )
        finally:
            for s in stubs:
                s.stop()
        self.assertFalse(initial)
        counts = _by_relation(graph.edges)
        self.assertEqual(counts["contact-compatibility"], 1)
        self.assertEqual(counts["support-compatibility"], 1)
        self.assertEqual(counts["contain-compatibility"], 1)

    def test_ambiguous_role_raises(self):
        with self.assertRaises(ValueError):
            _emit(2, roles=None)      # both orientations resolve


class PairOrderStabilityTest(unittest.TestCase):
    def _orientations(self, graph):
        pairs = rr._object_pairs(graph)
        return [(a.node_id, b.node_id) for a, b in pairs]

    def test_node_order_does_not_change_orientation(self):
        g1 = _graph(4)
        g2 = _graph(4)
        g2.nodes = [g2.nodes[0]] + list(reversed(g2.nodes[1:]))
        self.assertEqual(self._orientations(g1), self._orientations(g2))

    def test_index_reuse_does_not_flip_a_pair(self):
        g = _graph(3)
        for i, n in enumerate(g.nodes[1:]):
            n.index = 3 - i          # registry hands out different slots
        self.assertEqual(self._orientations(g), self._orientations(_graph(3)))


class AbsoluteVocabTest(unittest.TestCase):
    def test_vocab_width_matches_the_configured_n_abs(self):
        import yaml
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        with open("configs/model/_base_.yaml") as f:
            n_abs = yaml.safe_load(f)["graph"]["n_abs"]
        self.assertEqual(len(build_absolute_vocab()), 19)
        self.assertEqual(n_abs, 19)

    def test_directional_labels_are_in_the_vocabulary(self):
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        tokens = build_absolute_vocab().token_to_id
        self.assertIn(rr.SRC_HOLDS, tokens)
        self.assertIn(rr.DST_HOLDS, tokens)

    def test_support_labels_are_non_contiguous(self):
        """Why the hardcoded slice mask in graph.py could not be kept.

        The ids moved when the force-derived predicates gained ``unobserved``:
        sigma is built in first-seen order over the relation table, ``contact``
        is the first relation, so ``unobserved`` now sits at 3 and everything
        that used to follow it shifted up by one. The vocabulary is the same
        size and every consumer derives its mask, so nothing breaks -- but a
        packed graph written before that change encodes different labels under
        the same ids and cannot be replayed against this vocabulary.
        """
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        vocab = build_absolute_vocab()
        labels = rr.abs_labels_for()
        contact = {vocab.encode(l) for l in labels["contact"]}
        support = {vocab.encode(l) for l in labels["support"]}
        self.assertEqual(contact, {1, 2, 3})
        self.assertEqual(support, {1, 3, 4, 5})
        # The property the mask has to express. Only the directional
        # predicates are non-contiguous: they skip ``holds``, which the
        # symmetric ones use and they do not.
        contain = {vocab.encode(l) for l in labels["contain"]}
        for ids in (support, contain):
            self.assertNotEqual(len(ids), max(ids) - min(ids) + 1)

    def test_pose_derived_predicates_stay_off_the_unobserved_label(self):
        """``contain`` and ``reached`` are computed from poses, so they are
        observable whenever the graph is. Admitting a label they can never
        take would widen the decoder's mask for nothing."""
        from scenegraph.adapters.graph_vocab import build_absolute_vocab
        vocab = build_absolute_vocab()
        unobserved = vocab.encode("unobserved")
        labels = rr.abs_labels_for()
        for relation in ("contain", "reached"):
            ids = {vocab.encode(l) for l in labels[relation]}
            self.assertNotIn(unobserved, ids)


if __name__ == "__main__":
    unittest.main()
