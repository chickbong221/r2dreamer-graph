"""Render the semantic graph as a node-link diagram in the reference style.

Large filled circles for nodes (ee centred, objects on a ring) with bold
colored labels; plain italic relation values stacked along each edge in
family-colored chips. Relations in the same family share one chip background,
so the viewer reads ``event`` vs ``spatial`` vs ``affordance`` at a glance.

Layout is deterministic and scales the ring radius + canvas with the number
of objects so dense graphs spread out, with a per-node de-collision nudge so
distinct same-category instances do not stack on top of one another.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.entity_identity import display_name
from ..core.schema import Edge, Graph
from .palette import ColorMap


# --------------------------------------------------------------------------- #
# Relation -> family classification
# --------------------------------------------------------------------------- #
# Three TEEMO families. Physical-state replaces the old ``event`` family;
# transition edges have been removed from the vocabulary entirely.
_FAMILY_PHYSICAL_STATE = "physical_state"
_FAMILY_SPATIAL = "spatial"
_FAMILY_AFFORDANCE = "affordance"
_FAMILY_GOAL = "goal"

_RELATION_FAMILY: Dict[str, str] = {
    "contact": _FAMILY_PHYSICAL_STATE,
    "grasp": _FAMILY_PHYSICAL_STATE,
    "support": _FAMILY_PHYSICAL_STATE,
    "contain": _FAMILY_PHYSICAL_STATE,
    "planar-distance": _FAMILY_SPATIAL,
    "height-offset": _FAMILY_SPATIAL,
    "grasp-compatibility": _FAMILY_AFFORDANCE,
    "contact-compatibility": _FAMILY_AFFORDANCE,
    "support-compatibility": _FAMILY_AFFORDANCE,
    "contain-compatibility": _FAMILY_AFFORDANCE,
    "reached": _FAMILY_GOAL,
}

_FAMILY_ORDER = (_FAMILY_PHYSICAL_STATE, _FAMILY_SPATIAL, _FAMILY_AFFORDANCE,
                 _FAMILY_GOAL)

_INTRA_FAMILY_ORDER = {
    _FAMILY_PHYSICAL_STATE: ("contact", "grasp", "support", "contain"),
    _FAMILY_SPATIAL: ("planar-distance", "height-offset"),
    _FAMILY_AFFORDANCE: ("grasp-compatibility", "contact-compatibility",
                         "support-compatibility", "contain-compatibility"),
    _FAMILY_GOAL: ("reached",),
}

# Predicate states are useful to the model, but they are not relations a
# person should have to read in a demonstration video.  Negative/unknown
# physical predicates add a dense web of empty facts, while a positive
# predicate is much clearer when named by the relation it establishes.
_HIDDEN_LABELS = {"not-holds", "unobserved"}
_POSITIVE_PHYSICAL_LABELS = {"holds", "src-holds", "dst-holds"}
_PRIMARY_PHYSICAL_RELATIONS = {"grasp", "support", "contain"}
_PRIMARY_AFFORDANCE_RELATIONS = {
    "grasp-compatibility", "support-compatibility", "contain-compatibility",
}

# Per-family chip styling. Affordance uses a green palette so it doesn't
# read as the same family as the (also light+cool) spatial blue. Goal takes a
# warm violet: it is a terminal milestone, so it should not read as either the
# spatial ladder that leads to it or the physical state it sits beside.
_FAMILY_STYLE: Dict[str, Dict[str, str]] = {
    _FAMILY_PHYSICAL_STATE: {"bg": "#ffe0c2", "edge": "#c25a00", "text": "#5a2900"},
    _FAMILY_SPATIAL:        {"bg": "#d4e7ff", "edge": "#2f6ec2", "text": "#13396b"},
    _FAMILY_AFFORDANCE:     {"bg": "#d8f0dc", "edge": "#3a8f5b", "text": "#1c4a2b"},
    _FAMILY_GOAL:           {"bg": "#e8dcf5", "edge": "#7a4fb5", "text": "#3c2160"},
}

# Chip layout scales with n_obj: small graphs use a large, easily readable
# 13pt chip; busy graphs (near ``n_max``) shrink the chip and push it
# outward along its edge to avoid piling up around ee. All chip-collision
# constants scale off the chosen fontsize because they exist to detect
# overlap between real chip rectangles.
def _chip_layout(n_obj: int) -> Dict[str, float]:
    if n_obj <= 4:
        fontsize, fraction = 8.5, 0.50
    elif n_obj <= 7:
        fontsize, fraction = 7.0, 0.60
    else:
        fontsize, fraction = 6.0, 0.70
    # Chip rectangle size scales with fontsize; the collision constants were
    # measured against a 13pt chip that covered ~2.5 x 1.8 axes units, so we
    # scale from that reference.
    scale = fontsize / 13.0
    return {
        "fontsize": fontsize,
        "fraction": fraction,
        "min_sep_x": 2.5 * scale,
        "min_sep_y": 1.8 * scale,
        "nudge_step": 1.0 * scale,
        "max_iters": 15,
    }

# Stale edges override every family chip with the same blue palette used for
# frozen-pose nodes -- a single visual signal that the data is not fresh.


def _radial_layout(
    graph: Graph, radius: float, node_r: float
) -> Dict[str, np.ndarray]:
    pos: Dict[str, np.ndarray] = {}
    objects = [n.node_id for n in graph.nodes if n.node_type == "object"]
    has_ee = graph.get_node("ee") is not None
    if has_ee:
        pos["ee"] = np.array([0.0, 0.0])
    if has_ee and len(objects) == 1:
        pos["ee"] = np.array([-radius * 0.55, 0.0])
        pos[objects[0]] = np.array([radius * 0.55, 0.0])
        return pos
    if has_ee and len(objects) == 2:
        # A diameter layout makes the object--object edge pass straight through
        # ee, which is especially messy in the common object + receptacle
        # scene.  Use a balanced triangle instead.
        pos["ee"] = np.array([0.0, -radius * 0.45])
        pos[objects[0]] = np.array([-radius * 0.65, radius * 0.35])
        pos[objects[1]] = np.array([radius * 0.65, radius * 0.35])
        return pos
    n = max(len(objects), 1)
    for i, nid in enumerate(objects):
        ang = np.pi / 2 - 2 * np.pi * i / n
        pos[nid] = np.array([radius * np.cos(ang), radius * np.sin(ang)])

    min_sep = 2.0 * node_r + 0.35
    nudge_step = max(node_r * 0.5, 0.18)
    max_iters = 80
    for nid in objects:
        for _ in range(max_iters):
            p = pos[nid]
            r = float(np.linalg.norm(p))
            if r < 1e-9:
                pos[nid] = np.array([0.0, max(min_sep, radius)])
                continue
            unit = p / r
            collides = any(
                other != nid and np.linalg.norm(pos[other] - p) < min_sep
                for other in pos
            )
            if not collides:
                break
            pos[nid] = unit * (r + nudge_step)
    return pos


def _family_of(relation: str) -> Optional[str]:
    return _RELATION_FAMILY.get(relation)


def _display_text(
    edge: Edge, *, name_affordance: bool = False,
) -> Optional[str]:
    """Return the human-facing fact text, or ``None`` when it is noise.

    This only changes diagnostics.  The packed graph and its supervision keep
    the full absolute-state vocabulary, including negative and unobserved
    states.
    """
    label = str(edge.label)
    if label in _HIDDEN_LABELS:
        return None
    family = _family_of(edge.relation)
    if family == _FAMILY_PHYSICAL_STATE and label in _POSITIVE_PHYSICAL_LABELS:
        text = edge.relation
    else:
        text = label
    if edge.temp_label and str(edge.temp_label) not in _HIDDEN_LABELS:
        text = f"{text} / {edge.temp_label}"
    if name_affordance and family == _FAMILY_AFFORDANCE:
        relation_type = edge.relation.removesuffix("-compatibility")
        text = f"{relation_type} {text}"
    return text


def _group_by_family(
    elist: List[Edge], *, name_affordance: bool = False,
) -> Dict[str, List[str]]:
    """Bucket facts by family in ``_INTRA_FAMILY_ORDER``, one chip line each.
    Negative and unobserved facts are omitted.  A positive physical predicate
    renders as its relation name (``contact``, ``support``, ...) rather than
    the implementation-level label ``holds``."""
    grouped: Dict[str, List[Tuple[int, str]]] = {f: [] for f in _FAMILY_ORDER}
    for e in elist:
        family = _family_of(e.relation)
        if family is None:
            continue
        text = _display_text(e, name_affordance=name_affordance)
        if text is None:
            continue
        order = _INTRA_FAMILY_ORDER.get(family, ())
        try:
            rank = order.index(e.relation)
        except ValueError:
            rank = len(order)
        grouped[family].append((rank, text))

    out: Dict[str, List[str]] = {}
    for family in _FAMILY_ORDER:
        items = sorted(grouped[family])
        if items:
            out[family] = [text for _r, text in items]
    return out


def _unordered_pair(edge: Edge) -> Tuple[str, str]:
    """Stable visual identity for facts joining the same two vertices."""
    return tuple(sorted((edge.src, edge.dst)))


def _paper_display_edges(graph: Graph) -> List[Edge]:
    """Return a paper-only, presentation-focused view of ``graph.edges``.

    This deliberately does not mutate the graph.  The JSON written beside the
    diagram therefore remains the exact builder output used by training and
    diagnostics.
    """
    edges = list(graph.edges)

    # The bin and table are both kinematic in PlaceSphere, so the runtime graph
    # intentionally omits their unchanging scene-layout pair.  It is useful in
    # the explanatory paper diagram, however: the table supports the bin.
    if graph.env_id == "PlaceSphere-v1":
        node_ids = set(graph.node_ids())
        bin_id = "actor:bin"
        table_id = "actor:table-workspace"
        if {bin_id, table_id}.issubset(node_ids):
            pair = tuple(sorted((bin_id, table_id)))
            relations = {
                edge.relation for edge in edges
                if _unordered_pair(edge) == pair
            }
            if "support" not in relations:
                # Stable physical-pair ordering is bin -> table; dst is the
                # supporter, matching the schema's directional label.
                edges.append(Edge(
                    bin_id, table_id, "support", "dst-holds",
                    attributes={"support_role": "supporter"},
                ))
    # Once a stronger physical predicate holds, positive contact is implied
    # and only repeats the same event.  Apply the same visual hierarchy to an
    # established affordance, where contact-compatibility would otherwise add
    # a second identical ``match`` line.
    established_physical = {
        _unordered_pair(edge) for edge in edges
        if edge.relation in _PRIMARY_PHYSICAL_RELATIONS
        and str(edge.label) in _POSITIVE_PHYSICAL_LABELS
    }
    established_affordance = {
        _unordered_pair(edge) for edge in edges
        if edge.relation in _PRIMARY_AFFORDANCE_RELATIONS
        and str(edge.label) == "match"
    }
    return [
        edge for edge in edges
        if not (
            edge.relation == "contact"
            and _unordered_pair(edge) in established_physical
        )
        and not (
            edge.relation == "contact-compatibility"
            and _unordered_pair(edge) in established_affordance
        )
    ]


def _crop_paper_graph(path: str, background: str, pad: int = 8) -> None:
    """Crop a saved paper diagram to its visible graph content."""
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        pixels = np.asarray(image)
        bg = np.asarray(tuple(bytes.fromhex(background.lstrip("#"))), dtype=int)
        foreground = np.any(
            np.abs(pixels.astype(int) - bg[None, None, :]) > 2,
            axis=2,
        )
        ys, xs = np.nonzero(foreground)
        if not len(xs):
            return
        left = max(0, int(xs.min()) - pad)
        top = max(0, int(ys.min()) - pad)
        right = min(image.width, int(xs.max()) + pad + 1)
        bottom = min(image.height, int(ys.max()) + pad + 1)
        cropped = image.crop((left, top, right, bottom))
        cropped.save(path, format="PNG", optimize=True, dpi=(200, 200))


def render_graph(
    graph: Graph,
    out_path: Optional[str],
    colormap: Optional[ColorMap] = None,
    *,
    paper_style: bool = False,
):
    """Write a PNG and return its path, or return an RGB array when
    ``out_path`` is None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.patches import Circle

    cmap = colormap or ColorMap()
    cmap.assign_all(graph.node_ids())

    n_obj = sum(1 for n in graph.nodes if n.node_type == "object")
    chip = _chip_layout(n_obj)
    if paper_style:
        chip["fontsize"] += 2.0
        scale = chip["fontsize"] / 13.0
        chip.update({
            "min_sep_x": 2.5 * scale,
            "min_sep_y": 1.8 * scale,
            "nudge_step": 1.0 * scale,
        })

    # Square canvas that matches the overlay panels (6" @ 200 dpi -> 1200 px)
    # so the three panels hstack without any padding in the video.
    figsize = (6.0, 6.0)
    # Bigger baseline radius so small graphs still spread across the panel;
    # the growth per extra object is gentle so 10 objects still fit inside
    # the viewport.
    if n_obj <= 1:
        radius = 5.3
    elif n_obj == 2:
        radius = 5.5
    else:
        radius = min(5.5 + 0.1 * (n_obj - 3), 6.0)

    node_r = 0.32
    view_half = 7.0

    pos = _radial_layout(graph, radius, node_r)
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    ax.axis("off")
    bg = "#fdf0e9"
    ax.set_facecolor(bg)
    fig.patch.set_facecolor(bg)
    # Fill the panel with the axes, reserving only a thin title strip at the
    # top -- otherwise matplotlib's default margins leave the ring floating in
    # the middle with empty background around it.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

    # ----------------------------------------------------------------- edges
    by_pair: Dict[Tuple[str, str], List[Edge]] = {}
    display_edges = _paper_display_edges(graph) if paper_style else graph.edges
    for e in display_edges:
        if e.src not in pos or e.dst not in pos:
            continue
        # Do not leave an unlabeled line behind for a pair whose only facts are
        # negative or unobserved.
        if _display_text(e) is None:
            continue
        pair = _unordered_pair(e) if paper_style else (e.src, e.dst)
        by_pair.setdefault(pair, []).append(e)

    # Track every placed chip center across ALL edges so a chip laid down
    # for a later edge can nudge itself away from earlier chips.
    placed_chip_centers: List[Tuple[float, float]] = []
    placed_chip_boxes: List[Tuple[float, float, float, float]] = []

    for pair, elist in by_pair.items():
        if paper_style:
            # Prefer the stable physical direction for the single drawn line.
            # Affordance edges can run in the opposite direction even though
            # they describe the same unordered object pair.
            representative = next(
                (e for e in elist if e.relation in ("support", "contain")),
                elist[0],
            )
            src, dst = representative.src, representative.dst
        else:
            src, dst = pair
        p0, p1 = pos[src], pos[dst]
        d = p1 - p0
        L = float(np.linalg.norm(d)) + 1e-9
        u = d / L
        a0 = p0 + u * node_r
        a1 = p1 - u * node_r
        is_directed_physical = any(
            e.relation in ("support", "contain") for e in elist
        )

        if is_directed_physical:
            edge_color = "#b15a00"
            lw = 2.2
            alpha = 0.95
            linestyle = "solid"
        else:
            edge_color = "#444"
            lw = 1.4
            alpha = 0.85
            linestyle = "solid"

        ax.annotate(
            "", xy=a1, xytext=a0,
            arrowprops=dict(
                arrowstyle="-|>", color=edge_color, lw=lw, alpha=alpha,
                linestyle=linestyle,
                shrinkA=0, shrinkB=0,
                mutation_scale=18,
            ),
            zorder=2,
        )

        grouped = _group_by_family(
            elist, name_affordance=paper_style,
        )
        if not grouped:
            continue

        # Chips stack SCREEN-VERTICALLY along the edge, ordered top-to-bottom
        # as physical / spatial / affordance -- regardless of edge angle. The
        # chip anchor sits at ``chip["fraction"]`` along the edge so a dense
        # ee->object star pushes chips outward toward the ring instead of
        # piling them up near ee. Cross-edge collisions are resolved by
        # nudging the later chip further up or down, so the stack stays
        # vertical instead of drifting off to the side.
        mid = a0 + (a1 - a0) * chip["fraction"]
        families_present = [f for f in _FAMILY_ORDER if f in grouped]
        n = len(families_present)

        if paper_style:
            # Estimate each rounded chip's true height in axes units, then
            # pack the family boxes edge-to-edge.  Moving the whole block for
            # cross-edge collisions keeps one pair's relations together.
            axis_height_px = max(float(ax.bbox.height), 1.0)
            axis_width_px = max(float(ax.bbox.width), 1.0)
            data_per_y_px = (2.0 * view_half) / axis_height_px
            data_per_x_px = (2.0 * view_half) / axis_width_px
            line_px = chip["fontsize"] * fig.dpi / 72.0
            pad_px = 2.0 * 0.35 * line_px
            renderer = fig.canvas.get_renderer()
            font = FontProperties(size=chip["fontsize"], style="italic")
            heights = [
                data_per_y_px * (
                    line_px * (1.0 + 1.05 * (len(grouped[f]) - 1))
                    + pad_px
                )
                for f in families_present
            ]
            widths = [
                data_per_x_px * (
                    max(
                        renderer.get_text_width_height_descent(
                            line, font, ismath=False,
                        )[0]
                        for line in grouped[f]
                    )
                    + pad_px
                )
                for f in families_present
            ]
            total_height = sum(heights)
            cursor = total_height * 0.5
            y_offsets = []
            for height in heights:
                y_offsets.append(cursor - height * 0.5)
                cursor -= height

            anchors = [
                np.array([float(mid[0]), float(mid[1] + offset)])
                for offset in y_offsets
            ]
            push_sign = 0.0
            for _ in range(int(chip["max_iters"])):
                hit = next((
                    (anchor, width, height, box)
                    for anchor, width, height in zip(
                        anchors, widths, heights
                    )
                    for box in placed_chip_boxes
                    if abs(anchor[0] - box[0])
                    < width * 0.5 + box[2]
                    and abs(anchor[1] - box[1])
                    < height * 0.5 + box[3]
                ), None)
                if hit is None:
                    break
                anchor, _width, _height, box = hit
                if push_sign == 0.0:
                    push_sign = 1.0 if anchor[1] >= box[1] else -1.0
                shift = np.array([0.0, chip["nudge_step"] * push_sign])
                anchors = [anchor + shift for anchor in anchors]
            placed_chip_boxes.extend(
                (float(anchor[0]), float(anchor[1]),
                 float(width * 0.5), float(height * 0.5))
                for anchor, width, height in zip(anchors, widths, heights)
            )
        else:
            spacing = 0.55
            if n == 1:
                y_offsets = [0.0]
            elif n == 2:
                y_offsets = [spacing * 0.5, -spacing * 0.5]
            else:
                y_offsets = [spacing, 0.0, -spacing]
            anchors = [
                np.array([float(mid[0]), float(mid[1] + offset)])
                for offset in y_offsets
            ]

        for y_offset, anchor, family in zip(
            y_offsets, anchors, families_present
        ):
            # Direction the chip prefers to escape in when de-colliding:
            # top-slot pushes up, bottom-slot pushes down. The centre slot
            # (spatial when three families present) picks its direction
            # from the first collision it hits so it doesn't oscillate.
            if not paper_style:
                push_sign = (
                    1.0 if y_offset > 0 else (-1.0 if y_offset < 0 else 0.0)
                )
                for _ in range(int(chip["max_iters"])):
                    collided = False
                    first_hit_side = 0.0
                    for px, py in placed_chip_centers:
                        if (
                            abs(anchor[0] - px) < chip["min_sep_x"]
                            and abs(anchor[1] - py) < chip["min_sep_y"]
                        ):
                            collided = True
                            first_hit_side = 1.0 if anchor[1] >= py else -1.0
                            break
                    if not collided:
                        break
                    step_sign = (
                        push_sign if push_sign != 0.0 else first_hit_side
                    )
                    if step_sign == 0.0:
                        step_sign = 1.0
                    anchor = anchor + np.array([
                        0.0, chip["nudge_step"] * step_sign,
                    ])
            placed_chip_centers.append((float(anchor[0]), float(anchor[1])))
            style = _FAMILY_STYLE[family]
            ax.text(
                anchor[0], anchor[1], "\n".join(grouped[family]),
                fontsize=chip["fontsize"], ha="center", va="center", style="italic",
                color=style["text"], zorder=4, linespacing=1.05,
                bbox=dict(
                    facecolor=style["bg"], edgecolor=style["edge"],
                    linewidth=0.7, pad=0.45, alpha=0.96,
                    boxstyle="round,pad=0.35",
                ),
            )

    # ----------------------------------------------------------------- nodes
    for node in graph.nodes:
        nid = node.node_id
        if nid not in pos:
            continue
        x, y = pos[nid]
        if node.visible:
            face = cmap.color(nid)
            edge_col = "white"
            linestyle = "solid"
        else:
            face = (0.29, 0.56, 0.89)
            edge_col = "#1c3d6e"
            linestyle = (0, (3, 2))

        alpha = 0.55 if not node.visible else 1.0
        circ = Circle(
            (x, y), node_r,
            facecolor=face, edgecolor=edge_col, linewidth=1.8,
            linestyle=linestyle, alpha=alpha, zorder=3,
        )
        ax.add_patch(circ)

        label = "ee" if node.node_type == "ee" else display_name(node.name)
        label_color = tuple(0.45 * np.asarray(face))
        label_y = y - node_r - 0.18
        label_va = "top"
        if (
            paper_style
            and graph.env_id == "PlaceSphere-v1"
            and nid == "actor:bin"
        ):
            label_y = y + node_r + 0.18
            label_va = "bottom"
        ax.text(
            x, label_y, label,
            fontsize=12.5, fontweight="bold",
            ha="center", va=label_va,
            color=label_color, zorder=5,
        )

    # ----------------------------------------------------------------- frame
    if not paper_style:
        sub = graph.meta.get("active_subtask")
        title = f"frame {graph.frame}  |  {graph.env_id}"
        if sub:
            title += f"  |  subtask={sub}"
        ax.set_title(title, fontsize=14)

    ax.set_xlim(-view_half, view_half)
    ax.set_ylim(-view_half, view_half)
    ax.set_aspect("equal")
    # No ``bbox_inches='tight'``: cropping to visible content would defeat the
    # fixed figsize/viewport and re-introduce per-frame size drift.
    if out_path is None:
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return rgba[..., :3].copy()
    fig.savefig(out_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    if paper_style:
        _crop_paper_graph(out_path, bg)
    return out_path


def render_graph_array(graph, colormap=None, height=None) -> "np.ndarray":
    """The node-link panel as an RGB array, optionally scaled to ``height``.

    No PNG round-trip: the eval loop composites this straight onto the camera
    strip, once per step.
    """
    img = render_graph(graph, None, colormap=colormap)
    if height is None or img.shape[0] == height:
        return img
    # Nearest-neighbour; the panel is a diagram, so resampling quality does
    # not matter and this avoids pulling in an image library.
    rows = (np.arange(height) * img.shape[0] // height).clip(0, img.shape[0] - 1)
    width = max(1, img.shape[1] * height // img.shape[0])
    cols = (np.arange(width) * img.shape[1] // width).clip(0, img.shape[1] - 1)
    return img[rows][:, cols]
