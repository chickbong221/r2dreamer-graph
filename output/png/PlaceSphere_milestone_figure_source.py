"""Build the PlaceSphere milestone figure from exported frames and graphs.

The selected columns are relation-kind milestones, not every threshold crossed
inside one distance ladder.  Rewards are the undiscounted schedule-potential
change for the transition entering the displayed step: r_t = Phi_t - Phi_t-1.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
EPISODE = ROOT / "data" / "paper_figures" / "PlaceSphere-v1_seed0006"
FRAME_DIR = EPISODE / "frames"
GRAPH_DIR = EPISODE / "graphs"
GRAPH_JSON_DIR = EPISODE / "graph_json"
OUT = Path(__file__).with_name("PlaceSphere_milestone_figure.png")

# Start, affordance introductions/refinements, physical transitions, and settle.
# Distance-only far/medium/very-near crossings are deliberately not separate
# columns, following the supplied figure specification.
MILESTONES = [1, 11, 36, 43, 44, 94, 112]

WIDTH, HEIGHT = 3000, 1210
MARGIN = 28
SIDE_LABEL = 0
GAP = 10
PAPER = (250, 250, 248, 255)
INK = (29, 33, 38, 255)
MUTED = (91, 101, 112, 255)
GRID = (206, 211, 216, 255)
FILM = (139, 143, 147, 255)
BLUE = (68, 154, 196, 255)
BLUE_DARK = (30, 121, 165, 255)
BLUE_LIGHT = (213, 237, 247, 255)
PEACH = (247, 222, 204, 255)
PEACH_DARK = (186, 112, 74, 255)
WHITE = (255, 255, 255, 255)


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    if bold and italic:
        path = r"C:\Windows\Fonts\arialbi.ttf"
    elif bold:
        path = r"C:\Windows\Fonts\arialbd.ttf"
    elif italic:
        path = r"C:\Windows\Fonts\ariali.ttf"
    else:
        path = r"C:\Windows\Fonts\arial.ttf"
    return ImageFont.truetype(path, size)


F_SECTION = font(23, bold=True)
F_REWARD = font(22, bold=True)
F_CELL = font(22, bold=True)
F_SMALL = font(18)
F_TINY = font(16)


def load_graph(step: int) -> dict:
    return json.loads((GRAPH_JSON_DIR / f"graph_{step:04d}.json").read_text(encoding="utf-8"))


def relation_label(graph: dict, src: str, dst: str, relation: str) -> str:
    """Return a label in the requested orientation.

    Object-object edges are stored in canonical order.  Support is
    antisymmetric, so its holder label must be mirrored if the lookup reverses
    that stored order.
    """
    mirror = {"src-holds": "dst-holds", "dst-holds": "src-holds"}
    for edge in graph["edges"]:
        if edge["relation"] != relation:
            continue
        if edge["src"] == src and edge["dst"] == dst:
            return edge["label"]
        if edge["src"] == dst and edge["dst"] == src:
            return mirror.get(edge["label"], edge["label"])
    raise KeyError((src, dst, relation))


def ladder(label: str, ordered: list[str], weights: list[float]) -> float:
    if label not in ordered:
        return 0.0
    return sum(weights[: ordered.index(label) + 1])


def score(graph: dict) -> float:
    """Reproduce the four-phase PlaceSphere schedule potential."""
    ee, sphere, bin_ = "ee", "actor:sphere", "actor:bin"

    approach_quality = (
        ladder(
            relation_label(graph, ee, sphere, "planar-distance"),
            ["far", "medium", "near", "very-near"],
            [0.0375, 0.0375, 0.0375, 0.0375],
        )
        + ladder(
            relation_label(graph, ee, sphere, "grasp-compatibility"),
            ["poor-match", "partial-match", "match"],
            [0.033333, 0.033333, 0.033334],
        )
    )
    grasp_quality = 0.1 if relation_label(graph, ee, sphere, "grasp") == "holds" else 0.0
    transport_quality = (
        ladder(
            relation_label(graph, sphere, bin_, "planar-distance"),
            ["far", "medium", "near", "very-near"],
            [0.05, 0.05, 0.05, 0.05],
        )
        + ladder(
            relation_label(graph, sphere, bin_, "support-compatibility"),
            ["poor-match", "partial-match", "match"],
            [0.033333, 0.033333, 0.033334],
        )
    )
    settle_quality = (
        0.35
        if relation_label(graph, sphere, bin_, "support") == "dst-holds"
        else 0.0
    )

    done = [
        relation_label(graph, ee, sphere, "contact") == "holds",
        relation_label(graph, ee, sphere, "grasp") == "holds",
        relation_label(graph, sphere, bin_, "contact") == "holds",
        relation_label(graph, sphere, bin_, "support") == "dst-holds",
    ]
    qualities = [approach_quality, grasp_quality, transport_quality, settle_quality]
    weights = [0.25, 0.10, 0.30, 0.35]
    credit = []
    for index, quality in enumerate(qualities):
        later_complete = any(done[index + 1 :])
        credit.append(max(quality, weights[index] if later_complete else 0.0))
    return sum(credit)


def trace() -> tuple[dict[int, float], dict[int, float]]:
    potentials = {step: score(load_graph(step)) for step in range(117)}
    rewards = {0: 0.0}
    rewards.update({step: potentials[step] - potentials[step - 1] for step in range(1, 117)})
    return potentials, rewards


def fit_cover(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_w, target_h = size
    scale = max(target_w / source.width, target_h / source.height)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    left = (resized.width - target_w) // 2
    top = (resized.height - target_h) // 2
    return resized.crop((left, top, left + target_w, top + target_h))


def fit_contain(source: Image.Image, size: tuple[int, int], background=WHITE) -> Image.Image:
    target_w, target_h = size
    scale = min(target_w / source.width, target_h / source.height)
    resized = source.resize(
        (round(source.width * scale), round(source.height * scale)),
        Image.Resampling.LANCZOS,
    )
    out = Image.new("RGBA", size, background)
    out.alpha_composite(resized.convert("RGBA"), ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return out


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int],
          fill, width: int = 3) -> None:
    draw.line((*start, *end), fill=fill, width=width)
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = max(1.0, (vx * vx + vy * vy) ** 0.5)
    ux, uy = vx / length, vy / length
    px, py = -uy, ux
    tip = end
    back = (end[0] - ux * 12, end[1] - uy * 12)
    points = [tip, (back[0] + px * 6, back[1] + py * 6), (back[0] - px * 6, back[1] - py * 6)]
    draw.polygon(points, fill=fill)


def relation_box(draw: ImageDraw.ImageDraw, center: tuple[int, int], text: str,
                 fill=BLUE_LIGHT, outline=BLUE_DARK, max_width: int = 180) -> None:
    lines = text.split("\n")
    size = 15
    chosen = font(size)
    while size > 10:
        chosen = font(size)
        widths = [draw.textbbox((0, 0), line, font=chosen)[2] for line in lines]
        if max(widths, default=0) <= max_width - 16:
            break
        size -= 1
    bbox = draw.multiline_textbbox((0, 0), text, font=chosen, spacing=1, align="center")
    w, h = bbox[2] - bbox[0] + 16, bbox[3] - bbox[1] + 12
    rect = (center[0] - w // 2, center[1] - h // 2, center[0] + w // 2, center[1] + h // 2)
    draw.rounded_rectangle(rect, radius=7, fill=fill, outline=outline, width=2)
    draw.multiline_text(center, text, font=chosen, fill=INK, spacing=1, align="center", anchor="mm")


def compact_graph(step: int, size: tuple[int, int]) -> Image.Image:
    """Redraw the corresponding graph with stable, uncluttered node positions."""
    w, h = size
    graph = load_graph(step)
    out = Image.new("RGBA", size, (255, 244, 237, 255))
    draw = ImageDraw.Draw(out)
    table = (42, h - 46)
    sphere = (w // 2, h - 46)
    bin_ = (w - 42, h - 46)
    ee = (w // 2, 45)
    gray = (92, 96, 101, 255)
    orange = (190, 102, 12, 255)

    # Structural supports stay behind the task relations.  The wide gap from
    # table to sphere keeps their node titles distinct at paper scale.
    arrow(draw, (table[0] + 14, table[1] - 2), (sphere[0] - 18, sphere[1] - 2), orange, 4)
    arrow(draw, (table[0] + 10, table[1] - 10), (bin_[0] - 13, bin_[1] - 12), orange, 4)
    arrow(draw, (ee[0], ee[1] + 17), (sphere[0], sphere[1] - 18), gray, 4)
    arrow(draw, (sphere[0] + 18, sphere[1]), (bin_[0] - 18, bin_[1]), gray, 4)

    ee_distance = relation_label(graph, "ee", "actor:sphere", "planar-distance")
    grasp_fit = relation_label(graph, "ee", "actor:sphere", "grasp-compatibility")
    contact = relation_label(graph, "ee", "actor:sphere", "contact")
    grasp = relation_label(graph, "ee", "actor:sphere", "grasp")
    ee_lines = [f"ee is {ee_distance} sphere"]
    if grasp == "holds":
        ee_lines.append("ee grasps sphere")
    elif contact == "holds":
        ee_lines.append("ee contacts sphere")
    if grasp_fit != "unobserved" and grasp != "holds":
        ee_lines.append(f"ee has {grasp_fit} grasp fit")
    relation_box(draw, (ee[0] - 94, h // 2), "\n".join(ee_lines), max_width=188)

    sb_distance = relation_label(graph, "actor:sphere", "actor:bin", "planar-distance")
    support_fit = relation_label(graph, "actor:sphere", "actor:bin", "support-compatibility")
    sb_contact = relation_label(graph, "actor:sphere", "actor:bin", "contact")
    support = relation_label(graph, "actor:sphere", "actor:bin", "support")
    sb_lines = [f"sphere is {sb_distance} bin"]
    if support == "dst-holds":
        sb_lines.append("bin supports sphere")
    elif sb_contact == "holds":
        sb_lines.append("sphere contacts bin")
    if support_fit != "unobserved" and support != "dst-holds":
        sb_lines.append(f"sphere has {support_fit} support fit")
    relation_box(draw, ((sphere[0] + bin_[0]) // 2, sphere[1] - 70), "\n".join(sb_lines), max_width=174)

    node_specs = [
        (table, (242, 156, 41, 255), "table", "left"),
        (sphere, (51, 117, 199, 255), "sphere", "center"),
        (bin_, (232, 76, 61, 255), "bin", "right"),
        (ee, (46, 158, 102, 255), "ee", "center"),
    ]
    node_font = font(19, bold=True)
    for (x, y), color, label, align in node_specs:
        draw.ellipse((x - 16, y - 16, x + 16, y + 16), fill=color, outline=WHITE, width=4)
        if align == "left":
            draw.text((4, y + 19), label, font=node_font, fill=INK, anchor="la")
        elif align == "right":
            draw.text((w - 4, y + 19), label, font=node_font, fill=INK, anchor="ra")
        else:
            draw.text((x, y + 19), label, font=node_font, fill=INK, anchor="ma")
    return out


def rounded_text_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
                     fill, text_fill=WHITE, pad_x=14, pad_y=8,
                     text_font=F_REWARD) -> tuple[int, int, int, int]:
    x, y = xy
    box = draw.textbbox((0, 0), text, font=text_font)
    w, h = box[2] - box[0] + 2 * pad_x, box[3] - box[1] + 2 * pad_y
    rect = (x, y, x + w, y + h)
    draw.rounded_rectangle(rect, radius=12, fill=fill, outline=(255, 255, 255, 225), width=2)
    draw.text((x + pad_x, y + pad_y - box[1]), text, font=text_font, fill=text_fill)
    return rect


def draw_film_holes(draw: ImageDraw.ImageDraw, y_top: int, y_bottom: int) -> None:
    x = MARGIN + SIDE_LABEL + 12
    while x + 28 < WIDTH - MARGIN:
        draw.rounded_rectangle((x, y_top + 7, x + 28, y_top + 22), radius=4, fill=WHITE)
        draw.rounded_rectangle((x, y_bottom - 22, x + 28, y_bottom - 7), radius=4, fill=WHITE)
        x += 48


def center_multiline(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                     text: str, text_font, fill=INK, spacing=3) -> None:
    x0, y0, x1, y1 = box
    bbox = draw.multiline_textbbox((0, 0), text, font=text_font, spacing=spacing, align="center")
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.multiline_text(
        ((x0 + x1 - w) / 2, (y0 + y1 - h) / 2 - bbox[1]),
        text,
        font=text_font,
        fill=fill,
        spacing=spacing,
        align="center",
    )


def fit_center_multiline(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int],
                         text: str, max_size: int = 18, min_size: int = 11,
                         fill=INK) -> None:
    x0, y0, x1, y1 = box
    chosen = font(max_size)
    for size in range(max_size, min_size - 1, -1):
        candidate = font(size)
        bbox = draw.multiline_textbbox((0, 0), text, font=candidate, spacing=2, align="center")
        if bbox[2] - bbox[0] <= x1 - x0 and bbox[3] - bbox[1] <= y1 - y0:
            chosen = candidate
            break
    center_multiline(draw, box, text, chosen, fill=fill, spacing=2)


def draw_schedule_row(draw: ImageDraw.ImageDraw, y: int,
                      cells: list[tuple[int, int, str, str]]) -> None:
    """Draw schedule intervals on the true 1..116 episode timeline."""
    x0 = MARGIN + SIDE_LABEL
    x1 = WIDTH - MARGIN
    plot_x0 = x0
    row_h = 112
    draw.rounded_rectangle((x0, y, x1, y + row_h), radius=12, fill=WHITE, outline=INK, width=3)

    def px(step: int, right: bool = False) -> int:
        numerator = step if right else step - 1
        return round(plot_x0 + numerator / 116 * (x1 - plot_x0))

    narrow_events = []
    for start, end, label, kind in cells:
        left, right = px(start), px(end, right=True)
        fill = BLUE_LIGHT if kind == "reward" else PEACH
        outline = BLUE_DARK if kind == "reward" else PEACH_DARK
        draw.rectangle((left, y + 8, right, y + row_h - 8), fill=fill, outline=outline, width=2)
        if right - left >= 72:
            fit_center_multiline(draw, (left + 5, y + 10, right - 5, y + row_h - 10), label, max_size=19)
        elif label:
            narrow_events.append((left, right, label, outline))

    # Draw one-step event callouts after all interval fills so the following
    # segment cannot cover their text.
    for left, right, label, outline in narrow_events:
        anchor_right = left > (plot_x0 + x1) / 2
        tx = left - 8 if anchor_right else right + 8
        draw.text((tx, y + 14), label, font=F_TINY, fill=outline,
                  anchor="ra" if anchor_right else "la")
        draw.line((left, y + 38, left, y + row_h - 8), fill=outline, width=3)

    # No category prefix or step ticks: every block itself reads as a complete
    # subject-relation-object statement, as in the supplied reference figure.


def build() -> None:
    potentials, rewards = trace()
    canvas = Image.new("RGBA", (WIDTH, HEIGHT), PAPER)
    draw = ImageDraw.Draw(canvas)

    content_x0 = MARGIN + SIDE_LABEL
    content_x1 = WIDTH - MARGIN
    usable = content_x1 - content_x0
    col_w = (usable - GAP * (len(MILESTONES) - 1)) // len(MILESTONES)

    # Paper filmstrip: no headline or step labels, only transition reward.
    strip_top, image_top, image_h = 10, 42, 400
    strip_bottom = image_top + image_h + 30
    draw.rounded_rectangle((content_x0 - 10, strip_top, content_x1 + 10, strip_bottom), radius=8, fill=FILM)
    draw_film_holes(draw, strip_top, strip_bottom)
    for index, step in enumerate(MILESTONES):
        x = content_x0 + index * (col_w + GAP)
        with Image.open(FRAME_DIR / f"frame_{step:04d}.png") as source:
            frame = fit_cover(source.convert("RGBA"), (col_w, image_h))
        canvas.alpha_composite(frame, (x, image_top))
        draw.rectangle((x, image_top, x + col_w, image_top + image_h), outline=WHITE, width=3)
        reward_text = f"r = {rewards[step]:+.3f}"
        rounded_text_box(draw, (x + 12, image_top + 12), reward_text, (20, 27, 34, 220))

    # Original paper-renderer graphs, cropped to fill each panel without white
    # head/foot bands.  The graph renderer itself owns edge-label placement.
    graph_top, graph_h = 486, 390
    for index, step in enumerate(MILESTONES):
        x = content_x0 + index * (col_w + GAP)
        with Image.open(GRAPH_DIR / f"graph_{step:04d}.png") as source:
            graph = fit_cover(source.convert("RGBA"), (col_w, graph_h))
        canvas.alpha_composite(graph, (x, graph_top))
        draw.rectangle((x, graph_top, x + col_w, graph_top + graph_h), outline=GRID, width=2)

    # Complete subject-relation-object phrases; no relation-category prefix.
    draw_schedule_row(
        draw,
        900,
        [
            (1, 10, "ee is medium-distance from sphere", "reward"),
            (11, 35, "ee is near sphere\nee has poor grasp fit with sphere", "reward"),
            (36, 42, "ee has partial grasp fit with sphere", "reward"),
            (43, 43, "ee contacts sphere", "reward"),
            (44, 110, "ee grasps sphere", "reward"),
            (111, 116, "ee is near sphere", "neutral"),
        ],
    )
    draw_schedule_row(
        draw,
        1032,
        [
            (1, 83, "sphere is very-far from bin", "neutral"),
            (84, 93, "sphere approaches bin", "reward"),
            (94, 111, "sphere is near to very-near bin\nsphere has poor support fit with bin", "reward"),
            (112, 116, "bin supports sphere", "reward"),
        ],
    )

    canvas.convert("RGB").save(OUT, format="PNG", optimize=True, dpi=(300, 300))
    print(f"wrote {OUT}")
    print("selected transition rewards:")
    for step in MILESTONES:
        print(f"  step {step:03d}: phi={potentials[step]:.6f}  r={rewards[step]:+.6f}")


if __name__ == "__main__":
    build()
