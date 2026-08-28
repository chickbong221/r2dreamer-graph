"""Entity-name callouts drawn onto a rendered frame.

The paper figure names four things -- the sphere, the bin, the gripper and the
table -- and nothing else. A ManiSkill scene holds more actors than that and the
graph carries a vertex for each, so the label set is an explicit map rather than
"every node": an unlabelled vertex is a deliberate omission here, not an
oversight. ``label_every_node`` is available for a debugging pass.

Colours come from the same :class:`ColorMap` the node-link diagram uses, so the
chip beside the sphere in the photo is the colour of the sphere's circle in the
graph printed next to it. That is the whole reason the two figures read as one.

Text is drawn with PIL at the frame's own resolution. Compositing through
matplotlib would resample the render, and an unresampled 1000px frame is exactly
what capturing at this size is for.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..core.entity_identity import display_name
from ..core.schema import Graph, Node
from ..viz.palette import ColorMap

# Node ids for PlaceSphere-v1, which is what the figure is of. ``ee`` is the
# gripper and is task-independent; the three actors come from the mined
# whitelist at scenegraph/configs/subtask_whitelists/PlaceSphere-v1.
DEFAULT_LABELS: Dict[str, str] = {
    "ee": "ee",
    "actor:sphere": "sphere",
    "actor:bin": "bin",
    "actor:table-workspace": "table",
}

LabelFn = Callable[[Node], Optional[str]]


@dataclass
class Callout:
    """One name, the pixel it points at, and where its chip ended up."""

    node_id: str
    text: str
    anchor: Tuple[float, float]
    color: Tuple[int, int, int]
    box: Optional[Tuple[float, float, float, float]] = None
    offset: Tuple[float, float] = (0.0, 0.0)
    # Filled in by ``draw_callouts``; the caller records it so a figure can be
    # re-tuned from the manifest instead of by re-running the simulator.
    chip: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "text": self.text,
            "anchor": [round(float(v), 2) for v in self.anchor],
            "color": list(self.color),
            "box": None if self.box is None else [round(float(v), 2) for v in self.box],
            "chip": None if self.chip is None else [round(float(v), 2) for v in self.chip],
        }


def fixed_labels(mapping: Optional[Mapping[str, str]] = None) -> LabelFn:
    """Label only the named nodes. ``None`` means :data:`DEFAULT_LABELS`."""
    table = dict(DEFAULT_LABELS if mapping is None else mapping)

    def pick(node: Node) -> Optional[str]:
        return table.get(node.node_id)

    return pick


def label_every_node(node: Node) -> Optional[str]:
    """Label every vertex, with the short name the diagram prints for it."""
    return "ee" if node.node_type == "ee" else display_name(node.name)


def build_callouts(
    graph: Graph,
    camera,
    entities: Mapping[str, object],
    *,
    labels: Optional[LabelFn] = None,
    colormap: Optional[ColorMap] = None,
    offsets: Optional[Mapping[str, Sequence[float]]] = None,
) -> List[Callout]:
    """Project every labelled node into the frame.

    ``entities`` maps node id to the simulator actor, which is what supplies the
    collision AABB. A node with no entity -- ``ee`` is one, since the gripper is
    a tool-centre pose rather than an actor -- is anchored on its own world
    position instead, so it is labelled either way.
    """
    pick = labels or fixed_labels()
    cmap = colormap or ColorMap()
    cmap.assign_all(graph.node_ids())
    nudge = {str(k): tuple(float(v) for v in val)
             for k, val in (offsets or {}).items()}

    out: List[Callout] = []
    for node in graph.nodes:
        text = pick(node)
        if not text or node.pose_world is None:
            continue
        box = None
        entity = entities.get(node.node_id)
        if entity is not None:
            box = camera.entity_box(entity, node.pose_world)
        if box is not None:
            anchor: Optional[Tuple[float, float]] = (
                0.5 * (box[0] + box[2]), 0.5 * (box[1] + box[3])
            )
        else:
            anchor = camera.point_pixel(
                np.asarray(node.pose_world, dtype=float).reshape(-1)[:3]
            )
        if anchor is None:
            continue
        rgb = tuple(int(round(255.0 * c)) for c in cmap.color(node.node_id))
        out.append(
            Callout(
                node_id=node.node_id,
                text=str(text),
                anchor=(float(anchor[0]), float(anchor[1])),
                color=rgb,                                   # type: ignore[arg-type]
                box=box,
                offset=nudge.get(node.node_id, (0.0, 0.0)),  # type: ignore[arg-type]
            )
        )
    return out


def draw_callouts(
    frame: np.ndarray,
    callouts: Sequence[Callout],
    *,
    font_size: int = 0,
    lift: float = 0.0,
    margin: int = 10,
    draw_boxes: bool = False,
) -> np.ndarray:
    """Return ``frame`` with a chip per callout. The input is not modified.

    Chips are laid out top-down and pushed away from ones already placed, the
    same de-collision the node-link diagram runs on its relation chips: two
    objects a metre apart in the world can be twenty pixels apart in a
    third-person view, and two overlapping names would name neither.
    """
    from PIL import Image, ImageDraw

    img = Image.fromarray(np.ascontiguousarray(frame)).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.width, img.height

    size = int(font_size) if font_size else max(13, int(round(height * 0.030)))
    font = _load_font(size)
    lift = float(lift) if lift else height * 0.055
    pad = max(3, int(round(size * 0.38)))
    radius = max(3, int(round(size * 0.30)))
    dot = max(3, int(round(size * 0.20)))
    stroke = max(1, int(round(size / 14.0)))

    placed: List[Tuple[float, float, float, float]] = []
    # Farthest-up first: the object highest in the frame claims the space above
    # it, and the ones below stack downward from there rather than crossing it.
    for call in sorted(callouts, key=lambda c: c.anchor[1]):
        x0, y0, x1, y1 = draw.textbbox((0, 0), call.text, font=font)
        chip_w, chip_h = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad

        top = call.box[1] if call.box is not None else call.anchor[1]
        bottom = call.box[3] if call.box is not None else call.anchor[1]
        cx = call.anchor[0] + call.offset[0]
        cy = top - lift + call.offset[1]
        push = -1.0
        if cy - chip_h / 2.0 < margin:
            # No room above: hang the chip under the object and push downward
            # from there, so a de-collision never walks it back off the top.
            cy = bottom + lift + call.offset[1]
            push = 1.0

        rect = _clamp(cx, cy, chip_w, chip_h, width, height, margin)
        for _ in range(24):
            hit = next((p for p in placed if _overlaps(rect, p, pad)), None)
            if hit is None:
                break
            step = (chip_h + pad) * push
            rect = _clamp(
                0.5 * (rect[0] + rect[2]),
                0.5 * (rect[1] + rect[3]) + step,
                chip_w, chip_h, width, height, margin,
            )
        placed.append(rect)
        call.chip = rect

        if draw_boxes and call.box is not None:
            draw.rectangle(list(call.box), outline=call.color, width=stroke)

        _leader(draw, rect, call.anchor, call.color, stroke)
        draw.ellipse(
            [call.anchor[0] - dot - stroke, call.anchor[1] - dot - stroke,
             call.anchor[0] + dot + stroke, call.anchor[1] + dot + stroke],
            fill=(255, 255, 255),
        )
        draw.ellipse(
            [call.anchor[0] - dot, call.anchor[1] - dot,
             call.anchor[0] + dot, call.anchor[1] + dot],
            fill=call.color,
        )
        draw.rounded_rectangle(
            list(rect), radius=radius, fill=call.color,
            outline=(255, 255, 255), width=stroke,
        )
        draw.text(
            (0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])),
            call.text, font=font, fill=(255, 255, 255), anchor="mm",
        )

    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Layout helpers
# --------------------------------------------------------------------------- #
def _clamp(cx, cy, w, h, width, height, margin):
    """A ``w x h`` rectangle centred on ``(cx, cy)``, kept inside the frame."""
    cx = min(max(float(cx), margin + w / 2.0), width - margin - w / 2.0)
    cy = min(max(float(cy), margin + h / 2.0), height - margin - h / 2.0)
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


def _overlaps(a, b, gap: float = 0.0) -> bool:
    return not (
        a[2] + gap <= b[0] or b[2] + gap <= a[0]
        or a[3] + gap <= b[1] or b[3] + gap <= a[1]
    )


def _leader(draw, rect, anchor, color, stroke: int) -> None:
    """A line from the chip's border to the point it names.

    It starts at the border rather than the centre so the chip's fill does not
    have to be drawn over it, and it carries a white casing because a coloured
    hairline over a rendered tabletop is invisible at print size.
    """
    cx, cy = 0.5 * (rect[0] + rect[2]), 0.5 * (rect[1] + rect[3])
    dx, dy = anchor[0] - cx, anchor[1] - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return
    half_w, half_h = 0.5 * (rect[2] - rect[0]), 0.5 * (rect[3] - rect[1])
    scale = min(
        half_w / abs(dx) if abs(dx) > 1e-6 else np.inf,
        half_h / abs(dy) if abs(dy) > 1e-6 else np.inf,
    )
    if not np.isfinite(scale) or scale >= 1.0:
        return                                  # the anchor is under the chip
    start = (cx + dx * scale, cy + dy * scale)
    draw.line([start, tuple(anchor)], fill=(255, 255, 255), width=stroke * 3)
    draw.line([start, tuple(anchor)], fill=color, width=stroke)


def _load_font(size: int):
    """A real TrueType face at ``size``, falling back to PIL's bitmap font.

    DejaVu Sans ships with matplotlib, which is already a dependency of the
    node-link renderer, so the two figures are set in the same typeface.
    """
    from PIL import ImageFont

    try:
        from matplotlib import font_manager

        path = font_manager.findfont(
            font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
            fallback_to_default=True,
        )
        return ImageFont.truetype(path, size)
    except Exception:                                     # noqa: BLE001
        try:
            return ImageFont.load_default(size)
        except TypeError:                       # Pillow < 9.2 takes no size
            return ImageFont.load_default()
