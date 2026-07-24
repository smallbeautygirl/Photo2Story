"""Text wrapping primitives shared by stage3_assemble.py (PDF drawing) and
text_placement.py (candidate rectangle search) -- both need to know how many
lines a caption wraps to at a given width/font/language without depending on
each other.
"""

from __future__ import annotations

from reportlab.pdfbase import pdfmetrics

from src.stages.zhuyin_render import wrap_zhuyin
from src.utils.zhuyin import annotate


def wrap_to_width(text: str, font: str, size: float, max_width: float) -> list[str]:
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


def wrapped_line_count(
    text: str, font: str, font_size: float, max_width: float, language: str
) -> int:
    """Number of lines `text` wraps to, using Zhuyin-aware wrapping for zh-tw
    (each character is wider once annotated) and plain wrapping otherwise."""
    if language == "zh-tw":
        return len(wrap_zhuyin(annotate(text), font, font_size, max_width))
    return len(wrap_to_width(text, font, font_size, max_width))
