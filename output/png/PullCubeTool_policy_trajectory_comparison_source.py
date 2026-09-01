"""Build the two-row PullCubeTool policy-behavior comparison figure."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


BASELINE_DIR = Path(r"C:\Users\Lenovo\Downloads\pullcubetool\baseline")
OURS_DIR = Path(r"C:\Users\Lenovo\Downloads\pullcubetool\our_method")
OUTPUT_PATH = Path(__file__).with_name(
    "PullCubeTool_policy_trajectory_comparison.png"
)

FRAME_COUNT = 7
FRAME_SIZE = (600, 500)
FRAME_GAP = 8
SIDE_RAIL = 12
FILM_RAIL = 42
ROW_GAP = 6
CANVAS_MARGIN = 0

BASELINE_COLOR = "#C62828"
OURS_COLOR = "#168A5B"
WHITE = "#FFFFFF"


def load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path(r"C:\Windows\Fonts\arialbd.ttf"),
        Path(r"C:\Windows\Fonts\segoeuib.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def numbered_frames(folder: Path) -> list[Path]:
    paths = sorted(folder.glob("*.png"), key=lambda path: int(path.stem))
    expected = list(range(1, FRAME_COUNT + 1))
    found = [int(path.stem) for path in paths]
    if found != expected:
        raise ValueError(f"Expected frames {expected} in {folder}, found {found}")
    return paths


def prepare_frame(path: Path) -> Image.Image:
    with Image.open(path) as image:
        image = image.convert("RGB")
        # A modest vertical crop removes unused sky and table foreground while
        # retaining the robot, cube, and tool throughout both trajectories.
        return ImageOps.fit(
            image,
            FRAME_SIZE,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.62),
        )


def rounded_hole(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=6, fill=WHITE)


def draw_perforations(
    draw: ImageDraw.ImageDraw,
    row_x: int,
    row_y: int,
    row_width: int,
    row_height: int,
    label_clearance: int,
) -> None:
    hole_width = 44
    hole_height = 16
    spacing = 10

    top_y = row_y + 9
    bottom_y = row_y + row_height - 9 - hole_height

    top_x = row_x + label_clearance
    while top_x + hole_width <= row_x + row_width - 10:
        rounded_hole(
            draw,
            (top_x, top_y, top_x + hole_width, top_y + hole_height),
        )
        top_x += hole_width + spacing

    bottom_x = row_x + 10
    while bottom_x + hole_width <= row_x + row_width - 10:
        rounded_hole(
            draw,
            (bottom_x, bottom_y, bottom_x + hole_width, bottom_y + hole_height),
        )
        bottom_x += hole_width + spacing


def draw_row(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    row_y: int,
    frames: list[Path],
    color: str,
    label: str,
    font: ImageFont.ImageFont,
) -> None:
    frame_width, frame_height = FRAME_SIZE
    row_width = 2 * SIDE_RAIL + FRAME_COUNT * frame_width + (FRAME_COUNT - 1) * FRAME_GAP
    row_height = frame_height + 2 * FILM_RAIL
    row_x = CANVAS_MARGIN

    draw.rectangle(
        (row_x, row_y, row_x + row_width, row_y + row_height),
        fill=color,
        outline=color,
        width=4,
    )

    label_x = row_x + 18
    label_y = row_y + (FILM_RAIL - font.size) // 2 - 2
    draw.text((label_x, label_y), label, fill=WHITE, font=font)
    label_width = int(draw.textlength(label, font=font))
    draw_perforations(
        draw,
        row_x,
        row_y,
        row_width,
        row_height,
        label_clearance=max(220, label_width + 48),
    )

    image_y = row_y + FILM_RAIL
    for index, path in enumerate(frames):
        image_x = row_x + SIDE_RAIL + index * (frame_width + FRAME_GAP)
        frame = prepare_frame(path)
        canvas.paste(frame, (image_x, image_y))
        draw.rectangle(
            (image_x, image_y, image_x + frame_width - 1, image_y + frame_height - 1),
            outline=WHITE,
            width=4,
        )


def main() -> None:
    baseline_frames = numbered_frames(BASELINE_DIR)
    ours_frames = numbered_frames(OURS_DIR)

    row_width = 2 * SIDE_RAIL + FRAME_COUNT * FRAME_SIZE[0] + (FRAME_COUNT - 1) * FRAME_GAP
    row_height = FRAME_SIZE[1] + 2 * FILM_RAIL
    canvas_width = row_width + 2 * CANVAS_MARGIN
    canvas_height = 2 * row_height + ROW_GAP + 2 * CANVAS_MARGIN

    canvas = Image.new("RGB", (canvas_width, canvas_height), WHITE)
    draw = ImageDraw.Draw(canvas)
    font = load_font(34)

    first_row_y = CANVAS_MARGIN
    second_row_y = first_row_y + row_height + ROW_GAP
    draw_row(
        canvas,
        draw,
        first_row_y,
        baseline_frames,
        BASELINE_COLOR,
        "DreamerV3",
        font,
    )
    draw_row(
        canvas,
        draw,
        second_row_y,
        ours_frames,
        OURS_COLOR,
        "Ours",
        font,
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_PATH, format="PNG", optimize=True, dpi=(300, 300))
    print(OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
