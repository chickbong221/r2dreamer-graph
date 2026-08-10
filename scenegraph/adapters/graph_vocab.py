"""Closed vocabularies for the semantic graph encoder, one per learned table.

entity (kappa): whitelist-key union + ``<ee>`` / ``<pad>``. relation (rho): the
ten types. absolute (sigma): union of all family labels. temporal (delta): the
shared signed change vocabulary. Index 0 is padding everywhere.

sigma is shared across relations, so the decoder head is conditioned on rho and
masked by ``abs_valid`` / ``temp_valid``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..core.relation_rules import (
    ABS_LABELS,
    CHANGE_LABELS,
    RELATION_TYPES,
    TEMPORAL_RELATIONS,
)


PAD_TOKEN = "<pad>"
EE_TOKEN = "<ee>"


@dataclass
class Vocab:
    token_to_id: Dict[str, int]

    def __len__(self) -> int:
        return len(self.token_to_id) + 1  # +1 for pad at index 0

    @property
    def pad_id(self) -> int:
        return 0

    def encode(self, token: Optional[str]) -> int:
        if token is None:
            return 0
        idx = self.token_to_id.get(token)
        if idx is None:
            raise KeyError(f"unknown token {token!r}")
        return idx


@dataclass
class EntityVocab:
    token_to_id: Dict[str, int]

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def ee_id(self) -> int:
        return self.token_to_id[EE_TOKEN]

    def encode(self, key: Optional[str]) -> int:
        if key is None:
            return self.pad_id
        idx = self.token_to_id.get(key)
        if idx is None:
            raise KeyError(
                f"EntityVocab: unknown entity key {key!r}. Vocab must be built "
                f"from a whitelist directory that covers every runtime asset."
            )
        return idx


@dataclass
class GraphVocab:
    entity: EntityVocab
    relation: Vocab
    absolute: Vocab
    temporal: Vocab
    # [n_relations, n_abs] bool: which sigma each rho may take.
    abs_valid: np.ndarray
    # [n_relations] bool: which rho carry a delta at all (mu^rho).
    temp_valid: np.ndarray

    @property
    def sizes(self) -> Dict[str, int]:
        return {
            "entity": len(self.entity),
            "relation": len(self.relation),
            "absolute": len(self.absolute),
            "temporal": len(self.temporal),
        }


def _index(tokens: List[str]) -> Dict[str, int]:
    return {tok: i + 1 for i, tok in enumerate(tokens)}


def build_relation_vocab() -> Vocab:
    return Vocab(token_to_id=_index(list(RELATION_TYPES)))


def build_absolute_vocab() -> Vocab:
    seen: List[str] = []
    for relation in RELATION_TYPES:
        for label in ABS_LABELS[relation]:
            if label not in seen:
                seen.append(label)
    return Vocab(token_to_id=_index(seen))


def build_temporal_vocab() -> Vocab:
    return Vocab(token_to_id=_index(list(CHANGE_LABELS)))


def build_entity_vocab(whitelist_dir: str) -> EntityVocab:
    if not os.path.isdir(whitelist_dir):
        raise FileNotFoundError(
            f"whitelist_dir does not exist: {whitelist_dir!r}. "
            "Mine assets with tools/build_subtask_whitelists.py first."
        )
    keys: List[str] = [PAD_TOKEN, EE_TOKEN]
    seen = set(keys)
    for name in sorted(os.listdir(whitelist_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(whitelist_dir, name)
        with open(path, "r") as f:
            raw = json.load(f)
        members = raw.get("members", {}) if isinstance(raw, dict) else {}
        for member_key in members.keys():
            if not isinstance(member_key, str) or member_key.startswith("_"):
                continue
            if member_key not in seen:
                seen.add(member_key)
                keys.append(member_key)
    if len(keys) <= 2:
        raise ValueError(
            f"whitelist_dir {whitelist_dir!r} contained no member keys"
        )
    return EntityVocab(token_to_id={k: i for i, k in enumerate(keys)})


def build_graph_vocab(whitelist_dir: str) -> GraphVocab:
    entity = build_entity_vocab(whitelist_dir)
    relation = build_relation_vocab()
    absolute = build_absolute_vocab()
    temporal = build_temporal_vocab()

    abs_valid = np.zeros((len(relation), len(absolute)), dtype=bool)
    temp_valid = np.zeros((len(relation),), dtype=bool)
    for name in RELATION_TYPES:
        rid = relation.encode(name)
        for label in ABS_LABELS[name]:
            abs_valid[rid, absolute.encode(label)] = True
        temp_valid[rid] = name in TEMPORAL_RELATIONS

    return GraphVocab(
        entity=entity,
        relation=relation,
        absolute=absolute,
        temporal=temporal,
        abs_valid=abs_valid,
        temp_valid=temp_valid,
    )


def entity_key_for(node) -> Optional[str]:
    """Entity-type key for one node. ``None`` encodes to pad."""
    if node.node_type == "ee":
        return EE_TOKEN
    return node.attributes.get("whitelist_key")
