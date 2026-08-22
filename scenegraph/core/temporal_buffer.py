"""Temporal change labels over a horizon K, for spatial and affordance facts.

``signed(value[t] - value[t-K])`` binned into 5 buckets, written onto the fact
as ``temp_label``. History survives the affordance near gate (the score is
measured either way, so an approach shows up before the label flips) and is
dropped when the fact leaves the admissible set, so it never bridges a gap.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Set, Tuple

from .schema import Graph
from .relation_rules import (
    TEMPORAL_RELATIONS,
    _get_bin_spec,
    bin_label,
    temporal_bin_key,
)


def _edge_key(src: str, dst: str, relation: str) -> Tuple[str, str, str]:
    return (src, dst, relation)


class TemporalBuffer:
    """Per-fact continuous-value history keyed by (src, dst, relation)."""

    def __init__(self, K: int = 5):
        self.K = K
        self._values: Dict[Tuple[str, str, str], Deque[float]] = {}

    def annotate(self, graph: Graph, cfg: dict) -> None:
        """Push this frame's values and write ``temp_label`` in place.

        Facts with fewer than ``K + 1`` samples keep ``temp_label = None``:
        there is no value at ``t - K`` to difference against.
        """
        present: Set[Tuple[str, str, str]] = set()
        for e in graph.edges:
            if e.relation not in TEMPORAL_RELATIONS:
                continue
            present.add(_edge_key(e.src, e.dst, e.relation))

        for key in list(self._values):
            if key not in present:
                del self._values[key]

        for e in graph.edges:
            if e.relation not in TEMPORAL_RELATIONS:
                continue
            key = _edge_key(e.src, e.dst, e.relation)
            if e.raw_value is None:
                self._values.pop(key, None)
                continue
            hist = self._values.setdefault(key, deque(maxlen=self.K + 1))
            hist.append(float(e.raw_value))
            if len(hist) <= self.K:
                continue
            spec = _get_bin_spec(cfg, temporal_bin_key(e.relation))
            if spec is None:
                continue
            e.temp_label = bin_label(hist[-1] - hist[0], spec[0], spec[1])

    def purge(self, node_ids) -> None:
        """Drop history for any key whose src or dst is in ``node_ids``."""
        if not node_ids:
            return
        drop_set = set(node_ids)
        for key in [k for k in self._values if k[0] in drop_set or k[1] in drop_set]:
            del self._values[key]
