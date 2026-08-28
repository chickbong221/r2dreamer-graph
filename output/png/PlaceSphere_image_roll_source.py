"""Export the selected PlaceSphere frames as a compact paper-width filmstrip."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FRAME_DIR = ROOT / "data" / "paper_figures" / "PlaceSphere-v1_seed0006" / "frames"
OUT = Path(__file__).with_name("PlaceSphere_image_roll.png")

STEPS = [1, 28, 36, 44, 79, 90, 99, 111, 112, 116]
WIDTH, HEIGHT = 2100, 320
MARGIN, GAP = 12, 8
IMAGE_TOP, IMAGE_HEIGHT = 70, 210
STRIP_TOP, STRIP_BOTTOM = 52, 299
FILM_GRAY = (151, 154, 158)


def build() -> None:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(r"C:\Windows\Fonts\arialbd.ttf", 23)

    draw.rectangle((MARGIN, STRIP_TOP, WIDTH - MARGIN - 1, STRIP_BOTTOM), fill=FILM_GRAY)

    hole_w, hole_h, hole_gap = 27, 12, 17
    x = MARGIN + 8
    while x + hole_w < WIDTH - MARGIN:
        draw.rectangle((x, STRIP_TOP + 4, x + hole_w, STRIP_TOP + 4 + hole_h), fill="white")
        draw.rectangle((x, STRIP_BOTTOM - 4 - hole_h, x + hole_w, STRIP_BOTTOM - 4), fill="white")
        x += hole_w + hole_gap

    usable_width = WIDTH - 2 * MARGIN - GAP * (len(STEPS) - 1)
    thumb_width = usable_width // len(STEPS)
    remainder = usable_width - thumb_width * len(STEPS)
    x = MARGIN

    for index, step in enumerate(STEPS):
        width = thumb_width + (1 if index < remainder else 0)
        frame_path = FRAME_DIR / f"frame_{step:04d}.png"
        with Image.open(frame_path) as source:
            frame = source.convert("RGB").resize((width, IMAGE_HEIGHT), Image.Resampling.LANCZOS)
        canvas.paste(frame, (x, IMAGE_TOP))
        draw.rectangle((x, IMAGE_TOP, x + width - 1, IMAGE_TOP + IMAGE_HEIGHT - 1), outline="white", width=3)

        label = f"Step {step:03d}"
        label_box = draw.textbbox((0, 0), label, font=font)
        label_width = label_box[2] - label_box[0]
        draw.text((x + (width - label_width) / 2, 13), label, fill=(24, 24, 24), font=font)
        x += width + GAP

    canvas.save(OUT, format="PNG", optimize=True, dpi=(300, 300))
    print(OUT)


if __name__ == "__main__":
    build()
