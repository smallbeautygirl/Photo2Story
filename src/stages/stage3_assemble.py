from __future__ import annotations
from pathlib import Path
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.units import cm


PAGE_W, PAGE_H = A4
MARGIN = 1.5 * cm
IMAGE_H = 380
BODY_FONT = "Helvetica"
HEADER_FONT = "Helvetica-Bold"
FONT_SIZE = 12
HEADER_SIZE = 12
LINE_HEIGHT = 16
SECTION_GAP = 0.4 * cm


def _wrap_text(text: str, max_chars: int = 65) -> list[str]:
    words = text.split()
    lines, current = [], ""
    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current = f"{current} {word}".strip()
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_section(c: canvas.Canvas, label: str, text: str, top_y: float) -> float:
    """Draw a labeled text block starting at `top_y`. Returns the new y after the block."""
    c.setFont(HEADER_FONT, HEADER_SIZE)
    c.setFillColor(colors.black)
    c.drawString(MARGIN, top_y, label)
    y = top_y - LINE_HEIGHT

    c.setFont(BODY_FONT, FONT_SIZE)
    for line in _wrap_text(text or "(empty)"):
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
    """Build a PDF storybook: illustration + photo description + story text per page."""
    assert len(illustration_paths) == len(pages) == len(descriptions), (
        f"Mismatch: {len(illustration_paths)} illustrations vs "
        f"{len(pages)} pages vs {len(descriptions)} descriptions"
    )
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
        next_y = _draw_section(c, "Photo description", description, next_y)
        next_y -= SECTION_GAP
        _draw_section(c, "Story", page_text, next_y)

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
    build_pdf(
        stage2_result["illustration_paths"],
        stage1_result["pages"],
        descriptions,
        output_path,
    )
    return {"pdf_path": output_path}
