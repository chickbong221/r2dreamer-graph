"""Snapshot helpers used by the selector to retain nodes that left the view."""

from __future__ import annotations

from typing import Dict

from .schema import Node


# Per-frame affordance selection must be recomputed after a fresh observation.
_DYNAMIC_ATTRS = ("affordance_a_star",)


def _stripped_attrs(attrs: Dict[str, object]) -> Dict[str, object]:
    return {k: v for k, v in attrs.items() if k not in _DYNAMIC_ATTRS}


def _snapshot(node: Node) -> Node:
    """Copy of a visible node used as the retention seed."""
    return Node(
        node_id=node.node_id,
        node_type=node.node_type,
        name=node.name,
        visible=node.visible,
        segmentation_ids=list(node.segmentation_ids),
        pixel_area=node.pixel_area,
        pose_world=list(node.pose_world) if node.pose_world else None,
        index=node.index,
        source=node.source,
        attributes=_stripped_attrs(node.attributes),
    )
