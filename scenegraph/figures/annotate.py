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

# The graph's end-effector pose is the tool centre between the fingertips. For
# a human-facing figure, the recognisable entity is the solid gripper housing
# immediately above that point. This screen-space rise moves only the callout;
# the graph and its physical TCP pose remain unchanged.
_EE_VISUAL_RISE = 0.05

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
        if node.node_id == "ee" and box is None:
            _k, _ext, (_width, height) = camera.matrices()
            anchor = (
                float(anchor[0]),
                max(0.0, float(anchor[1]) - _EE_VISUAL_RISE * height),
            )
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

    Compact objects prefer a free side. The end effector uses a horizontal
    callout on its left, and large background surfaces are labelled on the
    left in a raised free lane with a leader back to the tabletop.
    Candidate positions on all four sides still provide de-collision when
    projected objects are close.
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
    # Farthest-up first: labels for the robot and background claim their nearby
    # space before labels for foreground objects are considered.
    for call in sorted(callouts, key=lambda c: c.anchor[1]):
        x0, y0, x1, y1 = draw.textbbox((0, 0), call.text, font=font)
        chip_w, chip_h = (x1 - x0) + 2 * pad, (y1 - y0) + 2 * pad

        rect = _place_chip(
            call, chip_w, chip_h, width, height, lift, margin, pad, placed
        )
        placed.append(rect)
        call.chip = rect

        if draw_boxes and call.box is not None:
            draw.rectangle(list(call.box), outline=call.color, width=stroke)

        target = _leader_target(call, rect)
        _leader(draw, rect, target, call.color, stroke)
        draw.ellipse(
            [target[0] - dot - stroke, target[1] - dot - stroke,
             target[0] + dot + stroke, target[1] + dot + stroke],
            fill=(255, 255, 255),
        )
        draw.ellipse(
            [target[0] - dot, target[1] - dot,
             target[0] + dot, target[1] + dot],
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
def _place_chip(call, chip_w, chip_h, width, height, gap, margin, pad, placed):
    """Pick the nearest non-overlapping side of ``call`` for its name chip."""
    if call.box is None:
        left = right = float(call.anchor[0])
        top = bottom = float(call.anchor[1])
    else:
        left, top, right, bottom = (float(v) for v in call.box)

    half_w, half_h = chip_w / 2.0, chip_h / 2.0
    side_y = min(max(float(call.anchor[1]), top), bottom)
    vertical_x = min(max(float(call.anchor[0]), left), right)
    horizontal = [
        (left - gap - half_w, side_y),
        (right + gap + half_w, side_y),
    ]
    # Put the chip toward the nearest frame edge. This leaves the centre of the
    # scene, where the task action happens, as clear as possible.
    if call.anchor[0] >= width / 2.0:
        horizontal.reverse()

    above = (vertical_x, top - gap - half_h)
    below = (vertical_x, bottom + gap + half_h)
    surface = _is_large_surface(call, width, height)
    if surface:
        # Keep the table name on the left, where it reads as the base of the
        # graph/frame composition.  Prefer a raised chip with a visible leader
        # into the tabletop.  If the sphere label already occupies that lane,
        # the ordinary collision check below falls through to the lower lane
        # rather than allowing the two names to overlap.
        ax = margin + half_w
        ay = min(max(float(call.anchor[1]), top), bottom)
        candidates = [
            (ax, ay - gap - half_h),
            (ax, ay + gap + half_h),
            (ax + gap + half_w, ay),
            *horizontal,
            above,
            below,
        ]
    elif call.node_id == "ee":
        # Keep the chip left of the solid gripper housing, giving it the same
        # horizontal visual grammar as the sphere callout below it.
        candidates = [
            (left - gap - half_w, side_y),
            (right + gap + half_w, side_y),
            above,
            below,
        ]
    elif call.box is None:
        candidates = [*horizontal, above, below]
    else:
        candidates = [*horizontal, above, below]
    # At the top edge a sideways chip would be clamped partly over the object;
    # put it below instead. This is common for a raised gripper.
    if top - gap - chip_h < margin:
        candidates.insert(0, below)

    # Extra lanes keep ``--labels-all`` usable when more than four entities
    # project to almost the same pixel.
    lane_y = chip_h + pad
    lane_x = chip_w + pad
    for lane in (1.0, -1.0, 2.0, -2.0):
        candidates.extend([
            (horizontal[0][0], side_y + lane * lane_y),
            (horizontal[1][0], side_y + lane * lane_y),
            (vertical_x + lane * lane_x, top - gap - half_h),
            (vertical_x + lane * lane_x, bottom + gap + half_h),
        ])

    dx, dy = (float(v) for v in call.offset)
    choices = [
        _clamp(cx + dx, cy + dy, chip_w, chip_h, width, height, margin)
        for cx, cy in candidates
    ]
    for rect in choices:
        covers_object = (
            call.box is not None
            and not surface
            and _overlaps(rect, call.box, 1.0)
        )
        covers_chip = any(_overlaps(rect, other, pad) for other in placed)
        if not covers_object and not covers_chip:
            return rect

    # A huge projected box can leave no formally free location. Prefer the
    # candidate that obscures the least object/chip area, with stable ordering
    # as the final tie-breaker.
    def obstruction(rect):
        own = 0.0 if call.box is None else _overlap_area(rect, call.box)
        other = sum(_overlap_area(rect, old) for old in placed)
        return other * 1000.0 + own

    ranked = enumerate(choices)
    return min(
        ranked, key=lambda item: (obstruction(item[1]), item[0])
    )[1]


def _leader_target(call, rect):
    """Nearest point on the projected object, avoiding a line through it."""
    if call.box is None:
        return call.anchor
    cx = 0.5 * (rect[0] + rect[2])
    cy = 0.5 * (rect[1] + rect[3])
    if call.node_id == "actor:table-workspace":
        return (
            min(max(cx, call.box[0]), call.box[2]),
            min(max(float(call.anchor[1]), call.box[1]), call.box[3]),
        )
    if call.box[0] <= cx <= call.box[2] and call.box[1] <= cy <= call.box[3]:
        return call.anchor
    return (
        min(max(cx, call.box[0]), call.box[2]),
        min(max(cy, call.box[1]), call.box[3]),
    )


def _is_large_surface(call, width, height) -> bool:
    if call.box is None:
        return False
    box_w = max(0.0, float(call.box[2]) - float(call.box[0]))
    box_h = max(0.0, float(call.box[3]) - float(call.box[1]))
    return box_w >= 0.65 * width or box_h >= 0.65 * height


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


def _overlap_area(a, b) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
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
