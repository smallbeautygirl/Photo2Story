"""Contact sheet for a Stage 0 selection: every photo as a thumbnail in a grid,
selected photos framed, each captioned with its CLIP cluster and LLM relevance
score. Meant for eyeballing why a selection strategy chose what it chose."""

from __future__ import annotations

import math
from pathlib import Path

import pillow_heif
from PIL import Image, ImageDraw, ImageFont, ImageOps

pillow_heif.register_heif_opener()

# Same CJK-capable font Stage 3 uses; PIL's default bitmap font renders Chinese
# filenames as boxes.
CAPTION_FONT_PATH = Path("assets/fonts/jf-openhuninn.ttf")
CAPTION_FONT_PX = 14

HIGHLIGHT_RGB = (255, 200, 0)
BACKGROUND_RGB = (40, 40, 40)
CAPTION_RGB = (230, 230, 230)
BORDER_PX = 6
CAPTION_PAD_PX = 4
MAX_NAME_CHARS = 18


def render_contact_sheet(
    *,
    image_paths: list[str],
    selected_paths: list[str],
    cluster_labels: dict[str, int],
    scores: dict[str, float],
    out_path: Path | str,
    columns: int = 6,
    thumb_px: int = 256,
    caption_px: int = 28,
) -> Path:
    """Render the grid to `out_path` and return it."""
    rows = math.ceil(len(image_paths) / columns)
    tile_px = thumb_px + caption_px
    sheet = Image.new("RGB", (columns * thumb_px, rows * tile_px), BACKGROUND_RGB)
    draw = ImageDraw.Draw(sheet)
    font = _load_caption_font()
    selected = set(selected_paths)

    for index, path in enumerate(image_paths):
        x = (index % columns) * thumb_px
        y = (index // columns) * tile_px
        _paste_thumbnail(sheet, path, x, y, thumb_px)
        if path in selected:
            draw.rectangle(
                (x, y, x + thumb_px - 1, y + thumb_px - 1), outline=HIGHLIGHT_RGB, width=BORDER_PX
            )
        caption = _caption(index, path, cluster_labels, scores, is_selected=path in selected)
        draw.text(
            (x + CAPTION_PAD_PX, y + thumb_px + CAPTION_PAD_PX), caption, fill=CAPTION_RGB, font=font
        )

    out = Path(out_path)
    sheet.save(out)
    return out


def _load_caption_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if CAPTION_FONT_PATH.is_file():
        return ImageFont.truetype(str(CAPTION_FONT_PATH), CAPTION_FONT_PX)
    return ImageFont.load_default()


def _paste_thumbnail(sheet: Image.Image, path: str, x: int, y: int, thumb_px: int) -> None:
    with Image.open(path) as photo:
        # Phone photos store rotation in EXIF; without this, portrait shots lie on their side.
        upright = ImageOps.exif_transpose(photo).convert("RGB")
        thumb = ImageOps.contain(upright, (thumb_px, thumb_px))
    offset_x = x + (thumb_px - thumb.width) // 2
    offset_y = y + (thumb_px - thumb.height) // 2
    sheet.paste(thumb, (offset_x, offset_y))


def _caption(
    index: int,
    path: str,
    cluster_labels: dict[str, int],
    scores: dict[str, float],
    *,
    is_selected: bool,
) -> str:
    parts = [f"{index + 1:02d}", Path(path).stem[:MAX_NAME_CHARS]]
    if path in cluster_labels:
        parts.append(f"c{cluster_labels[path]}")
    if path in scores:
        parts.append(f"{scores[path]:.2f}")
    if is_selected:
        parts.append("*")
    return " ".join(parts)
