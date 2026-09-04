"""The live potential probe scores what training would score.

The probe is the only thing that answers "does a real success end at 1.0", so
a probe that scores a *different* graph than training packs answers nothing.
Two ways it did:

* it matched the schedule's ``$active_target`` sentinel against real entity
  ids, which never match, so every fact about the target was dropped before
  scoring -- and the Pick schedule is almost entirely facts about the target;
* it never packed, so the protected rows the sentinel resolves through were
  never checked.

The row-index contract is exercised without torch. The scoring itself needs
it, so those tests run on the server with the rest of the graph stack.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from scenegraph.adapters.graph_pack import pack_graph
from scenegraph.core.schema import Edge, Graph, Node
from scenegraph.core.schedule import ACTIVE_TARGET_ENTITY_ID
from scenegraph.core.sites import SITE_EE_REST

TARGET = "actor:004_sugar_box"
COUNTER = "link:kitchen_counter-0/body"


def _pose(x=0.0, y=0.0, z=0.0):
    return [x, y, z, 1.0, 0.0, 0.0, 0.0]


class _Vocab:
    """The three id spaces, with the pad ids the packer checks against."""

    def __init__(self, tokens, pad_id=0):
        self.pad_id = pad_id
        self.token_to_id = {"<pad>": pad_id}
        for i, token in enumerate(tokens, start=1):
            self.token_to_id[token] = i
        self.ee_id = self.token_to_id.get("<ee>", 1)

    def encode(self, token):
        return self.token_to_id.get(token, self.pad_id)


def _vocab():
    return SimpleNamespace(
        entity=_Vocab(["<ee>", TARGET, SITE_EE_REST, COUNTER]),
        relation=_Vocab(["planar-distance", "height-offset", "grasp",
                         "reached"]),
        absolute=_Vocab(["very-near", "near", "level", "holds", "not-holds"]),
        temporal=_Vocab(["stable", "closer", "farther"]),
    )


def _graph():
    """One frame with the three protected nodes and facts about each."""
    ee = Node(node_id="ee", node_type="ee", name="ee",
              pose_world=_pose(z=1.0), attributes={})
    target = Node(node_id=TARGET, node_type="object", name="obj_0",
                  pose_world=_pose(z=0.5),
                  attributes={"whitelist_key": TARGET})
    site = Node(node_id=SITE_EE_REST, node_type="object", name=SITE_EE_REST,
                pose_world=_pose(z=0.9),
                attributes={"whitelist_key": SITE_EE_REST, "is_site": True})
    counter = Node(node_id=COUNTER, node_type="object", name="env-0_body",
                   pose_world=_pose(), attributes={"whitelist_key": COUNTER})
    return Graph(
        frame=0, env_id="PickSubtaskTrain-v0", camera="fetch_head",
        nodes=[counter, site, target, ee],       # deliberately out of order
        edges=[
            Edge("ee", TARGET, "planar-distance", "very-near"),
            Edge("ee", TARGET, "grasp", "holds"),
            Edge("ee", SITE_EE_REST, "reached", "holds"),
            Edge("ee", COUNTER, "height-offset", "level"),
        ],
        meta={"active_target_node_id": TARGET,
              "protected_site_node_id": SITE_EE_REST})


class PackedRowsAreWhatTheScorerReadsTest(unittest.TestCase):
    """No torch: the packing contract alone."""

    def _packed(self, graph=None, n_max=6, e_max=12):
        return pack_graph(graph or _graph(), _vocab(), n_max=n_max,
                          e_max=e_max, n_cams=2)

    def test_the_end_effector_is_row_zero(self):
        packed = self._packed()
        self.assertEqual(int(packed["graph_node_ent"][0]),
                         _vocab().entity.ee_id)

    def test_the_target_is_row_one_and_flagged(self):
        packed = self._packed()
        self.assertEqual(int(packed["graph_node_ent"][1]),
                         _vocab().entity.encode(TARGET))
        self.assertEqual(int(packed["graph_node_target"][1]), 1)
        self.assertEqual(int(packed["graph_node_target"].sum()), 1)

    def test_the_site_is_row_two(self):
        self.assertEqual(int(self._packed()["graph_node_ent"][2]),
                         _vocab().entity.encode(SITE_EE_REST))

    def test_edge_endpoints_are_rows_not_entity_ids(self):
        """The distinction the probe used to get wrong. The counter's entity
        id is 4 and its row is 3; reading one as the other silently addresses
        a different node."""
        packed = self._packed()
        vocab = _vocab()
        rel = packed["graph_edge_rel"]
        real = rel != vocab.relation.pad_id
        srcs = set(int(x) for x in packed["graph_edge_src"][real])
        self.assertEqual(srcs, {0})              # every fact is about the ee
        counter_row = int(np.where(
            packed["graph_node_ent"] == vocab.entity.encode(COUNTER))[0][0])
        self.assertNotEqual(counter_row, vocab.entity.encode(COUNTER))
        dsts = set(int(x) for x in packed["graph_edge_dst"][real])
        self.assertEqual(dsts, {1, 2, counter_row})

    def test_padded_edge_slots_carry_the_pad_relation(self):
        """Which is how the probe tells a real fact from an empty slot."""
        packed = self._packed(e_max=12)
        self.assertEqual(int((packed["graph_edge_rel"] != 0).sum()), 4)

    def test_a_frame_with_no_target_refuses_to_pack(self):
        """Row 1 is where the sentinel resolves, so an unnamed target makes
        every phase unreadable -- the probe must not score such a frame."""
        graph = _graph()
        graph.meta["active_target_node_id"] = None
        with self.assertRaises(RuntimeError) as ctx:
            self._packed(graph)
        self.assertIn("row 1", str(ctx.exception))


class TheSentinelIsNotAnEntityIdTest(unittest.TestCase):
    """Why the old comparison silently dropped the facts that matter."""

    def test_the_sentinel_matches_no_real_entity(self):
        vocab = _vocab()
        self.assertLess(ACTIVE_TARGET_ENTITY_ID, 0)
        self.assertNotIn(ACTIVE_TARGET_ENTITY_ID,
                         set(vocab.entity.token_to_id.values()))

    def test_matching_ids_against_it_discards_every_target_fact(self):
        """The old probe built ``(rel, src_entity, dst_entity)`` and asked
        whether the schedule named that slot. For the target the schedule
        names ``(rel, ee_id, -1)``, so the test was id-vs-sentinel and failed
        on every frame."""
        vocab = _vocab()
        scheduled = {("planar-distance", vocab.entity.ee_id,
                      ACTIVE_TARGET_ENTITY_ID)}
        observed = ("planar-distance", vocab.entity.ee_id,
                    vocab.entity.encode(TARGET))
        self.assertNotIn(observed, scheduled)


def _torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch is not installed")


class ScoringTest(unittest.TestCase):
    """End to end through the real scorer. Needs torch."""

    @classmethod
    def setUpClass(cls):
        _torch()

    def _schedule(self):
        """A one-phase schedule naming a target fact and a site fact."""
        from scenegraph.core.schedule import compile_schedule

        vocab = _vocab()
        spec = {
            "env_id": "PickSubtaskTrain-v0",
            "roles": {"movable": TARGET},
            "phases": [{
                "name": "hold_at_rest",
                "weight": 1.0,
                "clauses": [
                    {"src": "ee", "rel": "grasp", "dst": "movable",
                     "labels": ["holds"], "weight": 0.5},
                    {"src": "ee", "rel": "reached", "dst": SITE_EE_REST,
                     "labels": ["holds"], "weight": 0.5},
                ],
                "completion": {"all_of": [
                    {"src": "ee", "rel": "grasp", "dst": "movable",
                     "labels": ["holds"]},
                    {"src": "ee", "rel": "reached", "dst": SITE_EE_REST,
                     "labels": ["holds"]},
                ]},
            }],
        }
        return compile_schedule(spec, vocab.entity)

    def _score(self, graph):
        from tests.probes.probe_policy_potential import score_frames
        return score_frames([graph], self._schedule(), _vocab(),
                            n_max=6, e_max=12)

    def test_a_satisfied_frame_scores_one(self):
        potentials, valids = self._score(_graph())
        self.assertTrue(valids[0])
        self.assertAlmostEqual(potentials[0], 1.0, places=5)

    def test_the_target_fact_is_read_rather_than_discarded(self):
        """Drop the grasp and the score has to fall. Under the old id-vs-
        sentinel comparison it was never read, so removing it changed
        nothing."""
        graph = _graph()
        graph.edges = [e for e in graph.edges if e.relation != "grasp"]
        potentials, _ = self._score(graph)
        self.assertLess(potentials[0], 1.0)

    def test_the_site_fact_is_read_too(self):
        graph = _graph()
        graph.edges = [e for e in graph.edges if e.relation != "reached"]
        potentials, _ = self._score(graph)
        self.assertLess(potentials[0], 1.0)


if __name__ == "__main__":
    unittest.main()
