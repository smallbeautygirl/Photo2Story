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
IMAGE_H = 480
FONT_NAME = "Helvetica"
FONT_SIZE = 13
LINE_HEIGHT = 18


def _wrap_text(text: str, max_chars: int = 60) -> list[str]:
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


def build_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
) -> None:
    """Build a PDF storybook: one page per illustration + text."""
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    c = canvas.Canvas(output_path, pagesize=A4)

    for img_path, page_text in zip(illustration_paths, pages):
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        draw_w = PAGE_W - 2 * MARGIN
        draw_h = min(IMAGE_H, draw_w * img_h / img_w)
        x = MARGIN
        y = PAGE_H - MARGIN - draw_h
        c.drawImage(ImageReader(img), x, y, width=draw_w, height=draw_h)

        text_y = y - MARGIN - FONT_SIZE
        c.setFont(FONT_NAME, FONT_SIZE)
        c.setFillColor(colors.black)
        for line in _wrap_text(page_text):
            if text_y < MARGIN:
                break
            c.drawString(MARGIN, text_y, line)
            text_y -= LINE_HEIGHT

        c.showPage()

    c.save()


def run_stage3(stage1_result: dict, stage2_result: dict, config: dict, output_path: str) -> dict:
    """Returns: {"pdf_path": str}"""
    build_pdf(
        stage2_result["illustration_paths"],
        stage1_result["pages"],
        output_path,
    )
    return {"pdf_path": output_path}
