"""Deterministic, distinct color assignment for graph nodes / masks.

A palette is assigned in *insertion order* of node ids, so the same object keeps
the same color across frames within a run. The palette is a hand-picked set of
saturated, well-separated hues (similar in spirit to the reference SGG figure),
falling back to evenly-spaced HSV colors if there are more nodes than swatches.

``ee`` is pinned to a fixed green so the end-effector is always recognizable.
"""

from __future__ import annotations

import colorsys
from typing import Dict, List, Tuple

# RGB floats in [0, 1]. Distinct, saturated, print-friendly.
_PALETTE: List[Tuple[float, float, float]] = [
    (0.91, 0.30, 0.24),   # red
    (0.20, 0.46, 0.78),   # blue
    (0.95, 0.61, 0.16),   # orange
    (0.56, 0.27, 0.68),   # purple
    (0.18, 0.71, 0.55),   # teal
    (0.85, 0.37, 0.62),   # pink/magenta
    (0.55, 0.55, 0.20),   # olive
    (0.30, 0.69, 0.91),   # sky
    (0.78, 0.25, 0.44),   # crimson
    (0.40, 0.55, 0.30),   # moss
    (0.60, 0.40, 0.20),   # brown
    (0.45, 0.45, 0.80),   # periwinkle
]

_EE_COLOR: Tuple[float, float, float] = (0.18, 0.62, 0.40)   # pinned green


def _hsv_fallback(i: int, n: int) -> Tuple[float, float, float]:
    h = (i / max(n, 1)) % 1.0
    return colorsys.hsv_to_rgb(h, 0.65, 0.85)


class ColorMap:
    """Assigns and remembers a color per node id."""

    def __init__(self):
        self._colors: Dict[str, Tuple[float, float, float]] = {"ee": _EE_COLOR}
        self._next = 0

    def color(self, node_id: str) -> Tuple[float, float, float]:
        if node_id not in self._colors:
            if self._next < len(_PALETTE):
                c = _PALETTE[self._next]
            else:
                c = _hsv_fallback(self._next, self._next + 4)
            self._colors[node_id] = c
            self._next += 1
        return self._colors[node_id]

    def assign_all(self, node_ids: List[str]) -> Dict[str, Tuple[float, float, float]]:
        """Pre-assign colors for a list of ids (ee first, then in order)."""
        for nid in node_ids:
            self.color(nid)
        return dict(self._colors)

    def as_dict(self) -> Dict[str, Tuple[float, float, float]]:
        return dict(self._colors)


def color_list_for(node_ids: List[str]) -> Dict[str, Tuple[float, float, float]]:
    """One-shot helper: deterministic color per id for a single frame."""
    return ColorMap().assign_all(node_ids)
