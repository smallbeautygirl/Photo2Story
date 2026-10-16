"""Comparison sheet for the character_reference conditions: one row per
condition, one column per spread, so cross-spread drift (skin tone, outfit,
face) can be read along a row and the conditions compared down a column."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Same CJK-capable font as the Stage 0 contact sheet and Stage 3.
LABEL_FONT_PATH = Path("assets/fonts/jf-openhuninn.ttf")
LABEL_FONT_PX = 18
LABEL_PAD_PX = 8

BACKGROUND_RGB = (40, 40, 40)
LABEL_RGB = (230, 230, 230)

# A quarter of the 1408 x 992 spread keeps four spreads readable side by side.
DEFAULT_TILE_SIZE = (352, 248)
DEFAULT_LABEL_PX = 180


def render_character_reference_sheet(
    *,
    rows: dict[str, list[str]],
    out_path: Path | str,
    tile_size: tuple[int, int] = DEFAULT_TILE_SIZE,
    label_px: int = DEFAULT_LABEL_PX,
) -> Path:
    """Render `rows` (condition name -> spread paths in page order) to `out_path`."""
    tile_width, tile_height = tile_size
    columns = max(len(paths) for paths in rows.values())
    sheet = Image.new(
        "RGB", (label_px + columns * tile_width, len(rows) * tile_height), BACKGROUND_RGB
    )
    draw = ImageDraw.Draw(sheet)
    font = _load_label_font()

    for row, (condition, paths) in enumerate(rows.items()):
        y = row * tile_height
        draw.text((LABEL_PAD_PX, y + LABEL_PAD_PX), condition, fill=LABEL_RGB, font=font)
        for column, path in enumerate(paths):
            with Image.open(path) as spread:
                tile = spread.convert("RGB").resize(tile_size, Image.Resampling.LANCZOS)
            sheet.paste(tile, (label_px + column * tile_width, y))

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def _load_label_font() -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    if LABEL_FONT_PATH.exists():
        return ImageFont.truetype(str(LABEL_FONT_PATH), LABEL_FONT_PX)
    return ImageFont.load_default()
