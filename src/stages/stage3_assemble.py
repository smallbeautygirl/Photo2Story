from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.pdfgen import canvas

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = A4
MARGIN = 1.5 * cm

# --- Legacy "image top / text bottom" layout constants ---
IMAGE_H = 380
HEADER_FONT = "Helvetica-Bold"
FONT_SIZE = 12
HEADER_SIZE = 12
LINE_HEIGHT = 16
SECTION_GAP = 0.4 * cm

# --- picture_book layout constants (shared by single-page and spread variants) ---
TOP_BAND_FONT_SIZE = 20
BOTTOM_CAPTION_FONT_SIZE = 15
MAX_LINES_FOR_BOTTOM = 2
TEXT_ZONE_PAD = 0.6 * cm
INTER_ZONE_GAP = 0.3 * cm

# --- picture_book_spread geometry: one PDF page per spread (two A4 widths, one A4 height) ---
SPREAD_PAGE_W = 2 * PAGE_W
SPREAD_PAGE_H = PAGE_H

# ReportLab's TTFont only embeds TrueType (glyf) outlines, so Noto Sans CJK's
# CFF .ttc files fail to load. AR PL UMing TW is a glyf-based CJK font that works.
# (name, path, subfontIndex) — first that registers wins; an explicit path is tried first.
_CJK_FONT_CANDIDATES = [
    ("AR PL UMing TW", "/usr/share/fonts/truetype/arphic/uming.ttc", 0),
    ("AR PL UKai TW", "/usr/share/fonts/truetype/arphic/ukai.ttc", 0),
    ("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc", 0),
]

_cjk_font_name: str | None = None


def _resolve_cjk_font(font_path: str | None = None) -> str:
    """Register and return the first usable CJK TrueType font, or 'Helvetica' if none load.

    Result is cached: the font is registered once per process.
    """
    global _cjk_font_name
    if _cjk_font_name is not None:
        return _cjk_font_name

    candidates = list(_CJK_FONT_CANDIDATES)
    if font_path:
        candidates.insert(0, ("CustomCJK", font_path, 0))

    for name, path, index in candidates:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            _cjk_font_name = name
            logger.info("Registered CJK font", extra={"font": name, "path": path})
            return name
        except TTFError:
            continue

    logger.warning("No CJK TrueType font found; CJK text may not render in the PDF")
    _cjk_font_name = "Helvetica"
    return _cjk_font_name


def _wrap_to_width(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Wrap text to a pixel width. Breaks on spaces for latin text, per character for CJK."""
    text = text.strip()
    if not text:
        return []
    if " " in text:
        units, joiner = text.split(), " "
    else:
        units, joiner = list(text), ""

    lines: list[str] = []
    current = ""
    for unit in units:
        trial = f"{current}{joiner}{unit}" if current else unit
        if not current or pdfmetrics.stringWidth(trial, font, size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = unit
    if current:
        lines.append(current)
    return lines


def _choose_layout_variant(
    pages: list[str], font: str, page_width: float
) -> Literal["top_band", "bottom_caption"]:
    """Decide once per book: top-band for long captions, bottom-caption for short ones.

    Every page's caption is wrapped at `page_width` using the top-band font size as a
    fixed yardstick, regardless of which variant is ultimately chosen -- this keeps the
    decision a pure text-length measurement that never depends on the chosen variant.
    """
    max_width = page_width - 2 * MARGIN
    for page_text in pages:
        lines = _wrap_to_width(page_text, font, TOP_BAND_FONT_SIZE, max_width)
        if len(lines) > MAX_LINES_FOR_BOTTOM:
            return "top_band"
    return "bottom_caption"


def _draw_full_bleed_image(c: canvas.Canvas, img_path: str) -> None:
    """Scale the illustration to cover the whole page (cropping overflow off the page edges)."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    scale = max(PAGE_W / img_w, PAGE_H / img_h)
    draw_w, draw_h = img_w * scale, img_h * scale
    x = (PAGE_W - draw_w) / 2
    y = (PAGE_H - draw_h) / 2
    c.drawImage(ImageReader(img), x, y, width=draw_w, height=draw_h)


def _draw_caption(c: canvas.Canvas, text: str, font: str) -> None:
    """Draw the story text in a semi-transparent scrim across the bottom of the page."""
    max_width = PAGE_W - 2 * CAPTION_PAD_X
    lines = _wrap_to_width(text, font, CAPTION_FONT_SIZE, max_width) or [""]
    scrim_h = 2 * CAPTION_PAD_Y + len(lines) * CAPTION_LINE_HEIGHT

    c.setFillColor(colors.black)
    c.setFillAlpha(SCRIM_ALPHA)
    c.rect(0, 0, PAGE_W, scrim_h, fill=1, stroke=0)

    c.setFillAlpha(1)
    c.setFillColor(colors.white)
    c.setFont(font, CAPTION_FONT_SIZE)
    y = scrim_h - CAPTION_PAD_Y - CAPTION_FONT_SIZE
    for line in lines:
        c.drawCentredString(PAGE_W / 2, y, line)
        y -= CAPTION_LINE_HEIGHT


def build_caption_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
) -> None:
    """Build a picture-book PDF: each page is a full-bleed illustration with the story
    text overlaid in a caption band, the way a real storybook reads."""
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    font = _resolve_cjk_font(font_path)
    c = canvas.Canvas(output_path, pagesize=A4)
    for img_path, page_text in zip(illustration_paths, pages):
        _draw_full_bleed_image(c, img_path)
        _draw_caption(c, page_text, font)
        c.showPage()
    c.save()


def _draw_section(c: canvas.Canvas, label: str, text: str, body_font: str, top_y: float) -> float:
    """Draw a labeled text block starting at `top_y`. Returns the new y after the block."""
    c.setFont(HEADER_FONT, HEADER_SIZE)
    c.setFillColor(colors.black)
    c.drawString(MARGIN, top_y, label)
    y = top_y - LINE_HEIGHT

    c.setFont(body_font, FONT_SIZE)
    for line in _wrap_to_width(text or "(empty)", body_font, FONT_SIZE, PAGE_W - 2 * MARGIN):
        if y < MARGIN:
            return y
        c.drawString(MARGIN, y, line)
        y -= LINE_HEIGHT
    return y


def build_pdf(
    illustration_paths: list[str],
    pages: list[str],
    descriptions: list[str],
    output_path: str,
) -> None:
    """Legacy layout: illustration on top, photo description + story text below."""
    assert len(illustration_paths) == len(pages) == len(descriptions), (
        f"Mismatch: {len(illustration_paths)} illustrations vs "
        f"{len(pages)} pages vs {len(descriptions)} descriptions"
    )
    body_font = _resolve_cjk_font()
    c = canvas.Canvas(output_path, pagesize=A4)

    for img_path, page_text, description in zip(illustration_paths, pages, descriptions):
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        draw_w = PAGE_W - 2 * MARGIN
        draw_h = min(IMAGE_H, draw_w * img_h / img_w)
        x = MARGIN
        y = PAGE_H - MARGIN - draw_h
        c.drawImage(ImageReader(img), x, y, width=draw_w, height=draw_h)

        next_y = y - MARGIN
        next_y = _draw_section(c, "Photo description", description, body_font, next_y)
        next_y -= SECTION_GAP
        _draw_section(c, "Story", page_text, body_font, next_y)

        c.showPage()

    c.save()


def run_stage3(
    stage1_result: dict,
    stage2_result: dict,
    descriptions: list[str],
    config: dict,
    output_path: str,
) -> dict:
    """Returns: {"pdf_path": str}"""
    layout = config.get("page_layout", "image_top_text_bottom")
    if layout == "full_bleed_caption":
        build_caption_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            output_path,
            config.get("font_path"),
        )
    else:
        build_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            descriptions,
            output_path,
        )
    return {"pdf_path": output_path}
