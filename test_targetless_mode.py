"""Targetless normal-ManiSkill packing: no reserved row, no target bit.

MS-HAB pins row 1 for the subtask target. Normal ManiSkill names no target, so
row 1 must go back to the object pool and graph_node_target must stay zero
while remaining in the packed schema.
"""

import unittest

import numpy as np

from scenegraph.adapters import graph_pack as gp
from scenegraph.adapters.graph_vocab import (
    EE_TOKEN, PAD_TOKEN, EntityVocab, GraphVocab, build_absolute_vocab,
    build_relation_vocab, build_temporal_vocab,
)
from scenegraph.core.relation_rules import (
    EDGE_CONTRACT_LEGACY, RELATION_TYPES, TEMPORAL_RELATIONS, abs_labels_for,
)
from scenegraph.core.schema import Graph, Node

N_MAX = 8


def _vocab(contract=EDGE_CONTRACT_LEGACY):
    keys = [PAD_TOKEN, EE_TOKEN] + [f"actor:o{i}" for i in range(7)]
    entity = EntityVocab(token_to_id={k: i for i, k in enumerate(keys)})
    relation, temporal = build_relation_vocab(), build_temporal_vocab()
    absolute = build_absolute_vocab(contract)
    labels = abs_labels_for(contract)
    abs_valid = np.zeros((len(relation), len(absolute)), dtype=bool)
    temp_valid = np.zeros((len(relation),), dtype=bool)
    for name in RELATION_TYPES:
        rid = relation.encode(name)
        for lab in labels[name]:
            abs_valid[rid, absolute.encode(lab)] = True
        temp_valid[rid] = name in TEMPORAL_RELATIONS
    return GraphVocab(entity, relation, absolute, temporal,
                      abs_valid, temp_valid)


def _graph(n_objects, target_id=None):
    ee = Node("ee", "ee", "ee", pose_world=[0.0] * 3 + [1.0, 0, 0, 0], index=0,
              attributes={"whitelist_key": EE_TOKEN})
    nodes = [ee]
    for i in range(n_objects):
        nodes.append(Node(
            f"actor:o{i}", "object", f"o{i}", visible=True,
            segmentation_ids=[i + 1],
            pose_world=[0.01 * i, 0.0, 0.0, 1.0, 0, 0, 0], index=i + 1,
            attributes={"whitelist_key": f"actor:o{i}",
                        "interaction_types": ["contact"]},
        ))
    return Graph(frame=0, env_id="t", camera="c", nodes=nodes,
                 meta={"active_target_node_id": target_id, "node_uids": {}})


def _pack(graph, use_target_flag):
    return gp.pack_graph(
        graph, _vocab(), n_max=N_MAX, e_max=168, n_cams=2, app_dim=384,
        schema=gp.SCHEMA_SIMPLE_POOLED, use_target_flag=use_target_flag,
    )


class TargetlessPackingTest(unittest.TestCase):
    def test_target_row_reserved_when_flag_on(self):
        out = _pack(_graph(3, target_id="actor:o2"), True)
        self.assertEqual(int(out["graph_node_target"][1]), 1)

    def test_no_target_bit_when_flag_off(self):
        out = _pack(_graph(3, target_id="actor:o2"), False)
        self.assertEqual(int(out["graph_node_target"].sum()), 0)

    def test_field_stays_in_the_schema(self):
        out = _pack(_graph(3), False)
        self.assertIn("graph_node_target", out)
        self.assertEqual(out["graph_node_target"].shape, (N_MAX,))

    def test_row_one_returns_to_the_object_pool(self):
        """With the flag on, an unresolved target wastes row 1; off, it doesn't."""
        on = _pack(_graph(N_MAX - 1), True)
        off = _pack(_graph(N_MAX - 1), False)
        self.assertEqual(int((on["graph_node_ent"] != 0).sum()), N_MAX - 1)
        self.assertEqual(int((off["graph_node_ent"] != 0).sum()), N_MAX)

    def test_ee_stays_at_row_zero(self):
        out = _pack(_graph(3), False)
        self.assertEqual(int(out["graph_node_ent"][0]),
                         _vocab().entity.encode(EE_TOKEN))


if __name__ == "__main__":
    unittest.main()
