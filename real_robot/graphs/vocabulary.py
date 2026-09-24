"""The dataset's entity vocabulary beside the repository's label vocabularies.

Relation, absolute and temporal ids come from the shared builders in
:mod:`scenegraph.adapters.graph_vocab`; only the entity table is new. It is
built from the global entity list, so an entity has the same id in every task.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from scenegraph.adapters.graph_vocab import (
    EE_TOKEN,
    PAD_TOKEN,
    EntityVocab,
    GraphVocab,
    build_absolute_vocab,
    build_relation_vocab,
    build_temporal_vocab,
)
from scenegraph.core.relation_rules import RELATION_TYPES, TEMPORAL_RELATIONS, abs_labels_for

from ..common import stable_hash
from .schema import GraphConfig


def build_vocab(config: GraphConfig) -> GraphVocab:
    """Pad at 0, the end effector at 1, then objects in configuration order."""
    tokens = [PAD_TOKEN, EE_TOKEN] + [e.key for e in config.entities if e.type == "object"]
    entity = EntityVocab(token_to_id={token: index for index, token in enumerate(tokens)})
    relation = build_relation_vocab()
    absolute = build_absolute_vocab()
    temporal = build_temporal_vocab()
    labels = abs_labels_for()
    abs_valid = np.zeros((len(relation), len(absolute)), dtype=bool)
    temp_valid = np.zeros((len(relation),), dtype=bool)
    for name in RELATION_TYPES:
        rid = relation.encode(name)
        for label in labels[name]:
            abs_valid[rid, absolute.encode(label)] = True
        temp_valid[rid] = name in TEMPORAL_RELATIONS
    return GraphVocab(entity=entity, relation=relation, absolute=absolute,
                      temporal=temporal, abs_valid=abs_valid, temp_valid=temp_valid)


def vocab_sizes(vocab: GraphVocab) -> Dict[str, int]:
    """The four embedding widths the model config takes, by their config names."""
    return {
        "entity_vocab": len(vocab.entity),
        "n_rel": len(vocab.relation),
        "n_abs": len(vocab.absolute),
        "n_temp": len(vocab.temporal),
    }


def vocab_tables(vocab: GraphVocab) -> Dict[str, Dict[str, int]]:
    """Token -> id for each vocabulary, padding (id 0) included."""
    labels = {name: {PAD_TOKEN: 0, **table.token_to_id}
              for name, table in (("relation", vocab.relation), ("absolute", vocab.absolute),
                                  ("temporal", vocab.temporal))}
    return {"entity": dict(vocab.entity.token_to_id), **labels}


def vocab_identity(vocab: GraphVocab) -> Dict[str, Any]:
    """Digests of the numeric encodings, independent of dictionary order."""
    return {name: f"{len(table)}:{stable_hash(sorted(table.items()))}"
            for name, table in vocab_tables(vocab).items()}
