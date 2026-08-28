"""Build the PlaceSphere scene-graph progress timeline as an editable PDF.

The script validates every displayed interval against the exported graph JSON
before drawing.  Run it from the repository root with the bundled PDF Python.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from PIL import Image
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.utils import ImageReader


ROOT = Path(__file__).resolve().parents[2]
EPISODE = ROOT / "data" / "paper_figures" / "PlaceSphere-v1_seed0006"
GRAPH_DIR = EPISODE / "graph_json"
FRAME_DIR = EPISODE / "frames"
OUT = Path(__file__).with_name("PlaceSphere_progress_timeline.pdf")

PAGE_W, PAGE_H = 504.0, 212.0  # 7.0 in wide, compact full-width paper figure
MARGIN = 5.0
LABEL_W = 19.0
PLOT_X0 = MARGIN + LABEL_W
PLOT_X1 = PAGE_W - MARGIN
PLOT_W = PLOT_X1 - PLOT_X0
FIRST_STEP, LAST_STEP = 1, 116

EE = HexColor("#2E9E66")
SPHERE = HexColor("#3375C7")
BIN = HexColor("#E84C3D")
INK = HexColor("#20252B")
MUTED = HexColor("#66717D")
GRID = HexColor("#D8DEE5")
PAPER = HexColor("#FBFCFD")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("FigureSans", r"C:\Windows\Fonts\arial.ttf"))
    pdfmetrics.registerFont(TTFont("FigureSans-Bold", r"C:\Windows\Fonts\arialbd.ttf"))
    pdfmetrics.registerFont(TTFont("FigureSans-Italic", r"C:\Windows\Fonts\ariali.ttf"))


def edge_map(graph: dict, a: str, b: str) -> dict[str, dict]:
    return {
        edge["relation"]: edge
        for edge in graph["edges"]
        if {edge["src"], edge["dst"]} == {a, b}
    }


def graph_at(step: int) -> dict:
    path = GRAPH_DIR / f"graph_{step:04d}.json"
    return json.loads(path.read_text(encoding="utf-8"))


EE_SPHERE_ABS = [
    (1, 10, {"planar-distance": "medium", "grasp-compatibility": "unobserved", "contact": "not-holds", "grasp": "not-holds"}),
    (11, 34, {"planar-distance": "near", "grasp-compatibility": "poor-match", "contact": "not-holds", "grasp": "not-holds"}),
    (35, 35, {"planar-distance": "very-near", "grasp-compatibility": "poor-match", "contact": "not-holds", "grasp": "not-holds"}),
    (36, 42, {"planar-distance": "very-near", "grasp-compatibility": "partial-match", "contact": "not-holds", "grasp": "not-holds"}),
    (43, 43, {"planar-distance": "very-near", "grasp-compatibility": "match", "contact": "holds", "grasp": "not-holds"}),
    (44, 110, {"planar-distance": "very-near", "grasp-compatibility": "match", "contact": "holds", "grasp": "holds"}),
    (111, 111, {"planar-distance": "very-near", "grasp-compatibility": "match", "contact": "not-holds", "grasp": "not-holds"}),
    (112, 116, {"planar-distance": "near", "grasp-compatibility": "partial-match", "contact": "not-holds", "grasp": "not-holds"}),
]
EE_SPHERE_TEMP = [
    (1, 4, None), (5, 6, "stable"), (7, 27, "decrease-slow"),
    (28, 112, "stable"), (113, 116, "increase-slow"),
]

SPHERE_BIN_ABS = [
    (1, 83, {"planar-distance": "very-far", "support-compatibility": "unobserved", "contact": "not-holds", "support": "not-holds"}),
    (84, 89, {"planar-distance": "far", "support-compatibility": "unobserved", "contact": "not-holds", "support": "not-holds"}),
    (90, 93, {"planar-distance": "medium", "support-compatibility": "unobserved", "contact": "not-holds", "support": "not-holds"}),
    (94, 98, {"planar-distance": "near", "support-compatibility": "poor-match", "contact": "not-holds", "support": "not-holds"}),
    (99, 111, {"planar-distance": "very-near", "support-compatibility": "poor-match", "contact": "not-holds", "support": "not-holds"}),
    (112, 116, {"planar-distance": "very-near", "support-compatibility": "partial-match", "contact": "holds", "support": "src-holds"}),
]
SPHERE_BIN_TEMP = [
    (1, 4, None), (5, 78, "stable"), (79, 87, "decrease-slow"),
    (88, 101, "decrease-fast"), (102, 112, "decrease-slow"),
    (113, 116, "stable"),
]

EE_BIN_ABS = [
    (1, 12, {"planar-distance": "far", "height-offset": "level", "contact": "not-holds"}),
    (13, 87, {"planar-distance": "very-far", "height-offset": "level", "contact": "not-holds"}),
    (88, 91, {"planar-distance": "far", "height-offset": "level", "contact": "not-holds"}),
    (92, 96, {"planar-distance": "medium", "height-offset": "level", "contact": "not-holds"}),
    (97, 116, {"planar-distance": "near", "height-offset": "level", "contact": "not-holds"}),
]
EE_BIN_TEMP = [
    (1, 4, None), (5, 6, "stable"), (7, 28, "increase-slow"),
    (29, 78, "stable"), (79, 86, "decrease-slow"),
    (87, 101, "decrease-fast"), (102, 111, "decrease-slow"),
    (112, 116, "stable"),
]


def validate_intervals() -> None:
    graphs = {step: graph_at(step) for step in range(FIRST_STEP, LAST_STEP + 1)}

    def validate_abs(a: str, b: str, intervals) -> None:
        for start, end, expected in intervals:
            for step in range(start, end + 1):
                edges = edge_map(graphs[step], a, b)
                actual = {relation: edges[relation]["label"] for relation in expected}
                if actual != expected:
                    raise ValueError(f"step {step} {a}/{b}: {actual} != {expected}")

    def validate_temp(a: str, b: str, intervals) -> None:
        for start, end, expected in intervals:
            for step in range(start, end + 1):
                edge = edge_map(graphs[step], a, b)["planar-distance"]
                if edge.get("temp_label") != expected:
                    raise ValueError(
                        f"step {step} {a}/{b} temporal: "
                        f"{edge.get('temp_label')!r} != {expected!r}"
                    )

    validate_abs("ee", "actor:sphere", EE_SPHERE_ABS)
    validate_temp("ee", "actor:sphere", EE_SPHERE_TEMP)
    validate_abs("actor:sphere", "actor:bin", SPHERE_BIN_ABS)
    validate_temp("actor:sphere", "actor:bin", SPHERE_BIN_TEMP)
    validate_abs("ee", "actor:bin", EE_BIN_ABS)
    validate_temp("ee", "actor:bin", EE_BIN_TEMP)


def step_left(step: int) -> float:
    return PLOT_X0 + (step - FIRST_STEP) / (LAST_STEP - FIRST_STEP + 1) * PLOT_W


def step_right(step: int) -> float:
    return PLOT_X0 + step / (LAST_STEP - FIRST_STEP + 1) * PLOT_W


def text(c: Canvas, x: float, y: float, value: str, size: float = 7,
         font: str = "FigureSans", color=INK, align: str = "left") -> None:
    c.setFont(font, size)
    c.setFillColor(color)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def fit_lines(c: Canvas, x: float, y: float, w: float, h: float,
              lines: list[str], max_size: float = 4.7, min_size: float = 3.0,
              color=INK) -> None:
    if w < 3 or not lines:
        return
    size = max_size
    longest = max(lines, key=len) if lines else ""
    while size > min_size and pdfmetrics.stringWidth(longest, "FigureSans", size) > w - 2:
        size -= 0.15
    if pdfmetrics.stringWidth(longest, "FigureSans", size) > w - 1:
        return
    leading = size + 0.35
    total = leading * len(lines)
    baseline = y + (h + total) / 2 - leading + 0.5
    for index, line in enumerate(lines):
        text(c, x + w / 2, baseline - index * leading, line, size, color=color, align="center")


def draw_filmstrip(c: Canvas) -> None:
    steps = [1, 28, 36, 44, 79, 90, 99, 111, 112, 116]
    x0, y0, strip_h = MARGIN, 162.0, 34.0
    gap = 2.0
    thumb_w = (PAGE_W - 2 * MARGIN - gap * (len(steps) - 1)) / len(steps)
    c.setFillColor(HexColor("#9A9DA1"))
    c.rect(x0, y0 - 5, PAGE_W - 2 * MARGIN, strip_h + 10, fill=1, stroke=0)
    for i in range(45):
        hx = x0 + i * (PAGE_W - 2 * MARGIN) / 45 + 1.5
        c.setFillColor(white)
        c.rect(hx, y0 + strip_h + 1.2, 5.7, 2.8, fill=1, stroke=0)
        c.rect(hx, y0 - 4.0, 5.7, 2.8, fill=1, stroke=0)
    for index, step in enumerate(steps):
        x = x0 + index * (thumb_w + gap)
        path = FRAME_DIR / f"frame_{step:04d}.png"
        c.setFillColor(white)
        c.rect(x, y0, thumb_w, strip_h, fill=1, stroke=0)
        with Image.open(path) as source:
            thumbnail = source.convert("RGB").resize((240, 180), Image.Resampling.LANCZOS)
            buffer = BytesIO()
            thumbnail.save(buffer, format="PNG", optimize=True)
            buffer.seek(0)
            c.drawImage(ImageReader(buffer), x, y0, thumb_w, strip_h,
                        preserveAspectRatio=True, anchor="c", mask="auto")
        c.setStrokeColor(white)
        c.setLineWidth(0.45)
        c.rect(x, y0, thumb_w, strip_h, fill=0, stroke=1)
        text(c, x + thumb_w / 2, y0 + strip_h + 7.2, f"Step {step:03d}",
             4.5, "FigureSans-Bold", align="center")


def draw_row(c: Canvas, y: float, h: float, segments, fill, temporal=False) -> None:
    c.setFillColor(HexColor("#FAFAFA") if temporal else white)
    c.setStrokeColor(HexColor("#B8B8B8"))
    c.setLineWidth(0.35)
    c.rect(PLOT_X0, y, PLOT_W, h, fill=1, stroke=1)
    for index, segment in enumerate(segments):
        start, end, lines = segment
        x1, x2 = step_left(start), step_right(end)
        shade = fill if index % 2 == 0 else HexColor("#EEF1F3")
        if not lines:
            continue
        c.setFillColor(shade)
        c.setStrokeColor(white)
        c.setLineWidth(0.35)
        c.rect(x1, y + 0.5, max(0.7, x2 - x1), h - 1, fill=1, stroke=1)
        fit_lines(c, x1, y + 0.5, x2 - x1, h - 1, lines,
                  max_size=4.45,
                  color=HexColor("#26323D"))


def draw_group(c: Canvas, bottom: float, color,
               absolute_segments, temporal_segments, markers=()) -> None:
    group_h = 49.0
    c.setFillColor(white)
    c.setStrokeColor(HexColor("#111111"))
    c.setLineWidth(0.9)
    c.rect(MARGIN, bottom, PAGE_W - 2 * MARGIN, group_h, fill=1, stroke=1)

    abs_y, temp_y, row_h = bottom + 27, bottom + 4, 17
    text(c, PLOT_X0 - 3, abs_y + 5.5, "abs", 4.1,
         "FigureSans-Bold", INK, "right")
    text(c, PLOT_X0 - 3, temp_y + 5.5, "temp", 4.1,
         "FigureSans-Bold", INK, "right")
    draw_row(c, abs_y, row_h, absolute_segments, color, temporal=False)
    draw_row(c, temp_y, row_h, temporal_segments, color, temporal=True)
    for step, label, align in markers:
        x = (step_left(step) + step_right(step)) / 2
        c.setStrokeColor(HexColor("#353535"))
        c.setLineWidth(0.45)
        c.line(x, abs_y + row_h - 1, x, bottom + group_h - 1.5)
        offset = -1.5 if align == "right" else 1.5
        text(c, x + offset, bottom + group_h - 5.0, label, 3.7,
             "FigureSans-Bold", INK, align)


def build() -> None:
    register_fonts()
    validate_intervals()
    c = Canvas(
        str(OUT), pagesize=(PAGE_W, PAGE_H), pageCompression=1,
        initialFontName="FigureSans", initialFontSize=7,
    )
    c.setTitle("PlaceSphere-v1 scene-graph progress timeline")
    c.setAuthor("R2-Dreamer")
    c.setSubject("Absolute and temporal relation changes across a successful episode")
    c.setFillColor(white)
    c.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)

    draw_filmstrip(c)

    draw_group(
        c, 108, HexColor("#CBE8F0"),
        [
            (1, 10, ["ee medium-distance", "from sphere"]),
            (11, 34, ["ee near sphere"]),
            (35, 42, ["ee very-near", "sphere"]),
            (43, 43, []),
            (44, 110, ["ee grasps sphere"]),
            (111, 111, []),
            (112, 116, ["ee", "near", "sphere"]),
        ],
        [
            (1, 4, []),
            (5, 6, []),
            (7, 27, ["ee approaches sphere", "slowly"]),
            (28, 112, ["ee stays near sphere"]),
            (113, 116, ["ee moves", "away from", "sphere"]),
        ],
        markers=[(43, "ee contacts sphere", "left"),
                 (111, "ee releases sphere", "right")],
    )

    draw_group(
        c, 56, HexColor("#F4D5DF"),
        [
            (1, 83, ["sphere very-far from bin"]),
            (84, 89, ["sphere", "far from", "bin"]),
            (90, 93, ["sphere", "medium", "from bin"]),
            (94, 98, ["sphere", "near bin"]),
            (99, 111, ["sphere very-near", "bin"]),
            (112, 116, ["bin", "supports", "sphere"]),
        ],
        [
            (1, 4, []),
            (5, 78, ["sphere stays far from bin"]),
            (79, 87, ["sphere approaches", "bin slowly"]),
            (88, 101, ["sphere approaches", "bin quickly"]),
            (102, 112, ["sphere approaches", "bin slowly"]),
            (113, 116, ["sphere", "stays on", "bin"]),
        ],
    )

    draw_group(
        c, 4, HexColor("#F4E1BF"),
        [
            (1, 12, ["ee far from bin"]),
            (13, 87, ["ee very-far from bin"]),
            (88, 91, ["ee", "far from", "bin"]),
            (92, 96, ["ee", "medium", "from bin"]),
            (97, 116, ["ee near bin"]),
        ],
        [
            (1, 4, []),
            (5, 6, []),
            (7, 28, ["ee moves away", "from bin slowly"]),
            (29, 78, ["ee stays far from bin"]),
            (79, 86, ["ee approaches", "bin slowly"]),
            (87, 101, ["ee approaches", "bin quickly"]),
            (102, 111, ["ee approaches", "bin slowly"]),
            (112, 116, ["ee stays", "near bin"]),
        ],
    )

    c.showPage()
    c.save()
    print(f"validated steps {FIRST_STEP}-{LAST_STEP}")
    print(OUT)


if __name__ == "__main__":
    build()
