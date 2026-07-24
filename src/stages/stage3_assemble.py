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
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFError, TTFont
from reportlab.pdfgen import canvas

from src.stages.text_placement import (
    CANDIDATE_PAD_PT,
    CONTRAST_SAFE_VARIANCE,
    SCRIM_OPACITY,
    Candidate,
    analyze_badness,
    pick_best,
    search_candidates,
    visible_crop_box,
)
from src.stages.zhuyin_render import draw_zhuyin_line, wrap_zhuyin
from src.utils.text_wrap import wrap_to_width
from src.utils.zhuyin import annotate

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
    """Register and return a usable CJK font.

    Tries an embedded TrueType font file first (an explicit `font_path`,
    then the fixed candidate list), then falls back to ReportLab's built-in
    'STSong-Light' CID font -- no font file needed, since it relies on the
    PDF viewer's own CJK font substitution (confirmed by direct testing to
    render Traditional Chinese and Bopomofo correctly). This fallback always
    succeeds: it ships as reportlab package data, not a filesystem lookup.

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

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    _cjk_font_name = "STSong-Light"
    logger.info("Registered built-in CID CJK font", extra={"font": "STSong-Light"})
    return _cjk_font_name


def _cover_fit_image(
    c: canvas.Canvas, image: Image.Image, page_w: float, page_h: float
) -> None:
    """Scale the already-opened illustration to fill the whole page,
    cropping any excess. Takes a loaded image rather than a path so the
    caller can reuse the same load for badness analysis -- and derives its
    geometry from `visible_crop_box`, the same function that analysis uses,
    so the two can never disagree about what's visible."""
    img_w, img_h = image.size
    left, top, right, _ = visible_crop_box(img_w, img_h, page_w, page_h)
    scale = page_w / (right - left)
    draw_w, draw_h = img_w * scale, img_h * scale
    draw_x, draw_y = -left * scale, -top * scale
    c.drawImage(ImageReader(image), draw_x, draw_y, width=draw_w, height=draw_h)


def _draw_zone_text(
    c: canvas.Canvas,
    text: str,
    font: str,
    font_size: float,
    x: float,
    top_y: float,
    w: float,
    align: Literal["left", "center"],
    language: str = "en",
    fill_color: colors.Color = colors.black,
    offset: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Draw wrapped text with its first line's baseline just below `top_y`.

    zh-tw text is drawn with Zhuyin annotation via zhuyin_render.py; every
    other language uses the plain wrap-and-draw path, unchanged.
    """
    line_height = font_size * 1.3
    dx, dy = offset
    cursor_y = top_y - font_size + dy
    x = x + dx

    if language == "zh-tw":
        zhuyin_lines = wrap_zhuyin(annotate(text), font, font_size, w) or [[]]
        for zhuyin_line in zhuyin_lines:
            draw_zhuyin_line(
                c, zhuyin_line, font, font_size, x, cursor_y, w, align, fill_color=fill_color
            )
            cursor_y -= line_height
        return

    lines = wrap_to_width(text, font, font_size, w) or [""]
    c.setFillColor(fill_color)
    c.setFont(font, font_size)
    for line in lines:
        if align == "center":
            c.drawCentredString(x + w / 2, cursor_y, line)
        else:
            c.drawString(x, cursor_y, line)
        cursor_y -= line_height


def _draw_caption_overlay(
    c: canvas.Canvas,
    candidate: Candidate,
    page_text: str,
    font: str,
    page_w: float,
    page_h: float,
    language: str = "en",
) -> None:
    """Draw `page_text` inside `candidate`'s rectangle, directly on the
    illustration, escalating from plain text to an outline to a translucent
    scrim only as far as needed for contrast against the artwork."""
    x = candidate.x * page_w
    w = candidate.w * page_w
    top_y = page_h - candidate.y * page_h
    h = candidate.h * page_h

    # The rectangle search (search_candidates) reserves CANDIDATE_PAD_PT of
    # padding on every side when it wraps text, so the drawn text must be
    # inset by the same amount to land inside the space that was measured --
    # otherwise it renders flush against the detected rectangle's edges.
    text_x = x + CANDIDATE_PAD_PT
    text_top_y = top_y - CANDIDATE_PAD_PT
    text_w = w - 2 * CANDIDATE_PAD_PT

    ink = colors.black if candidate.brightness > 128 else colors.white
    backdrop = colors.white if ink == colors.black else colors.black

    if candidate.variance <= CONTRAST_SAFE_VARIANCE:
        _draw_zone_text(
            c, page_text, font, candidate.font_size, text_x, text_top_y, text_w, "left",
            language, fill_color=ink,
        )
    elif not candidate.requires_scrim:
        _draw_zone_text(
            c, page_text, font, candidate.font_size, text_x, text_top_y, text_w, "left",
            language, fill_color=backdrop, offset=(0.6, -0.6),
        )
        _draw_zone_text(
            c, page_text, font, candidate.font_size, text_x, text_top_y, text_w, "left",
            language, fill_color=ink,
        )
    else:
        c.saveState()
        c.setFillColor(backdrop)
        c.setFillAlpha(SCRIM_OPACITY)
        c.rect(x, top_y - h, w, h, fill=1, stroke=0)
        c.restoreState()
        _draw_zone_text(
            c, page_text, font, candidate.font_size, text_x, text_top_y, text_w, "left",
            language, fill_color=ink,
        )


def build_picture_book_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
    page_size: tuple[float, float] = A4,
    language: str = "en",
) -> None:
    """Build a picture-book PDF: each page is a full-bleed illustration with
    its caption drawn directly on top, in a programmatically detected
    text-safe region (see text_placement.py) rather than a reserved zone
    outside the art.
    """
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    font = _resolve_cjk_font(font_path)
    page_w, page_h = page_size

    c = canvas.Canvas(output_path, pagesize=page_size)
    for img_path, page_text in zip(illustration_paths, pages):
        image = Image.open(img_path).convert("RGB")
        badness_map = analyze_badness(image, page_w, page_h)
        candidates = search_candidates(badness_map, page_text, font, language, page_w, page_h)
        best = pick_best(candidates)

        _cover_fit_image(c, image, page_w, page_h)
        _draw_caption_overlay(c, best, page_text, font, page_w, page_h, language)
        c.showPage()
    c.save()


def build_spread_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
    language: str = "en",
) -> None:
    """Build a picture-book PDF where each page is a double-page spread (one wide
    illustration spanning two A4 widths at one A4 height). Reuses the same
    full-bleed-illustration-plus-detected-overlay pipeline as
    `build_picture_book_pdf`, just at the spread's doubled page width -- so the
    badness analysis and candidate search run against the wider page.
    """
    build_picture_book_pdf(
        illustration_paths,
        pages,
        output_path,
        font_path,
        page_size=(SPREAD_PAGE_W, SPREAD_PAGE_H),
        language=language,
    )


def _draw_section(c: canvas.Canvas, label: str, text: str, body_font: str, top_y: float) -> float:
    """Draw a labeled text block starting at `top_y`. Returns the new y after the block."""
    c.setFont(HEADER_FONT, HEADER_SIZE)
    c.setFillColor(colors.black)
    c.drawString(MARGIN, top_y, label)
    y = top_y - LINE_HEIGHT

    c.setFont(body_font, FONT_SIZE)
    for line in wrap_to_width(text or "(empty)", body_font, FONT_SIZE, PAGE_W - 2 * MARGIN):
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
    language = stage1_result.get("language", "en")
    if layout == "picture_book":
        build_picture_book_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            output_path,
            config.get("font_path"),
            language=language,
        )
    elif layout == "picture_book_spread":
        build_spread_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            output_path,
            config.get("font_path"),
            language=language,
        )
    else:
        build_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            descriptions,
            output_path,
        )
    return {"pdf_path": output_path}
