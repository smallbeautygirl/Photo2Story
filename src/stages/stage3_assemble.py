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

from src.stages.zhuyin_render import draw_zhuyin_line, wrap_zhuyin
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

# --- picture_book layout constants (shared by single-page and spread variants) ---
TOP_BAND_FONT_SIZE = 24
BOTTOM_CAPTION_FONT_SIZE = 18
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


def _wrapped_line_count(text: str, font: str, font_size: float, max_width: float, language: str) -> int:
    """Number of lines `text` wraps to, using Zhuyin-aware wrapping for zh-tw
    (each character is wider once annotated) and plain wrapping otherwise."""
    if language == "zh-tw":
        return len(wrap_zhuyin(annotate(text), font, font_size, max_width))
    return len(_wrap_to_width(text, font, font_size, max_width))


def _choose_layout_variant(
    pages: list[str], font: str, page_width: float, language: str = "en"
) -> Literal["top_band", "bottom_caption"]:
    """Decide once per book: top-band for long captions, bottom-caption for short ones.

    Every page's caption is wrapped at `page_width` using the top-band font size as a
    fixed yardstick, regardless of which variant is ultimately chosen -- this keeps the
    decision a pure text-length measurement that never depends on the chosen variant.
    zh-tw captions use Zhuyin-aware wrapping, since each character is wider once
    annotated and would otherwise be under-measured.
    """
    max_width = page_width - 2 * MARGIN
    for page_text in pages:
        if _wrapped_line_count(page_text, font, TOP_BAND_FONT_SIZE, max_width, language) > MAX_LINES_FOR_BOTTOM:
            return "top_band"
    return "bottom_caption"


def _contain_fit_image(
    c: canvas.Canvas, img_path: str, x: float, y: float, w: float, h: float
) -> None:
    """Scale the illustration to fit within (w, h) without cropping, centered in the zone."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    scale = min(w / img_w, h / img_h)
    draw_w, draw_h = img_w * scale, img_h * scale
    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2
    c.drawImage(ImageReader(img), draw_x, draw_y, width=draw_w, height=draw_h)


def _text_zone_height(
    pages: list[str], font: str, font_size: float, page_width: float, language: str = "en"
) -> float:
    """Height of the text zone, sized to the longest wrapped caption across all pages.

    Computed once per book so every page reserves an identically sized band.
    zh-tw captions use Zhuyin-aware wrapping (see _wrapped_line_count).
    """
    max_width = page_width - 2 * MARGIN
    line_height = font_size * 1.3
    max_lines = max(
        (_wrapped_line_count(p, font, font_size, max_width, language) or 1 for p in pages),
        default=1,
    )
    return 2 * TEXT_ZONE_PAD + max_lines * line_height


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
) -> None:
    """Draw wrapped text with its first line's baseline just below `top_y`.

    zh-tw text is drawn with Zhuyin annotation via zhuyin_render.py; every
    other language uses the plain wrap-and-draw path, unchanged.
    """
    line_height = font_size * 1.3
    cursor_y = top_y - font_size

    if language == "zh-tw":
        zhuyin_lines = wrap_zhuyin(annotate(text), font, font_size, w) or [[]]
        for zhuyin_line in zhuyin_lines:
            draw_zhuyin_line(c, zhuyin_line, font, font_size, x, cursor_y, w, align)
            cursor_y -= line_height
        return

    lines = _wrap_to_width(text, font, font_size, w) or [""]
    c.setFillColor(colors.black)
    c.setFont(font, font_size)
    for line in lines:
        if align == "center":
            c.drawCentredString(x + w / 2, cursor_y, line)
        else:
            c.drawString(x, cursor_y, line)
        cursor_y -= line_height


def _draw_picture_book_page(
    c: canvas.Canvas,
    img_path: str,
    page_text: str,
    font: str,
    variant: Literal["top_band", "bottom_caption"],
    text_h: float,
    page_w: float,
    page_h: float,
    language: str = "en",
) -> None:
    """Draw one page/spread: a text zone (top or bottom) and a contain-fit image
    filling the rest, with a small gap so text never touches the art."""
    content_w = page_w - 2 * MARGIN
    if variant == "top_band":
        text_top_y = page_h - MARGIN
        _draw_zone_text(
            c, page_text, font, TOP_BAND_FONT_SIZE, MARGIN, text_top_y, content_w, "left", language
        )
        image_h = page_h - MARGIN - text_h - INTER_ZONE_GAP - MARGIN
        _contain_fit_image(c, img_path, MARGIN, MARGIN, content_w, image_h)
    else:
        image_y = MARGIN + text_h + INTER_ZONE_GAP
        image_h = page_h - MARGIN - image_y
        _contain_fit_image(c, img_path, MARGIN, image_y, content_w, image_h)
        text_top_y = MARGIN + text_h
        _draw_zone_text(
            c,
            page_text,
            font,
            BOTTOM_CAPTION_FONT_SIZE,
            MARGIN,
            text_top_y,
            content_w,
            "center",
            language,
        )


def build_picture_book_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
    page_size: tuple[float, float] = A4,
    language: str = "en",
) -> None:
    """Build a picture-book PDF: image contain-fit into a reserved zone, caption text
    in a plain-background zone outside the image that never overlaps it. The
    top-band-vs-bottom-caption choice and the text zone's height are both computed
    once per book, so every page/spread reserves an identically sized, positioned zone.
    zh-tw captions are annotated with Zhuyin (see zhuyin_render.py); every other
    language takes the unchanged plain-text path.
    """
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    font = _resolve_cjk_font(font_path)
    page_w, page_h = page_size
    variant = _choose_layout_variant(pages, font, page_w, language)
    font_size = TOP_BAND_FONT_SIZE if variant == "top_band" else BOTTOM_CAPTION_FONT_SIZE
    text_h = _text_zone_height(pages, font, font_size, page_w, language)

    c = canvas.Canvas(output_path, pagesize=page_size)
    for img_path, page_text in zip(illustration_paths, pages):
        _draw_picture_book_page(
            c, img_path, page_text, font, variant, text_h, page_w, page_h, language
        )
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
    top-band/bottom-caption geometry as `build_picture_book_pdf`, re-measured at
    the spread's doubled width.
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
