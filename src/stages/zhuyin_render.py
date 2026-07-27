"""ReportLab drawing for Zhuyin-annotated Traditional Chinese captions.

Consumes ZhuyinChar lists from src/utils/zhuyin.py. Each character reserves
extra width for its phonetic column (main letters stacked vertically,
tone mark positioned beside the stack), wraps character-by-character (no
spaces to break on, matching this codebase's existing CJK wrapping in
stage3_assemble._wrap_to_width), and draws accordingly.
"""

from __future__ import annotations

from typing import Literal

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from src.utils.zhuyin import ZhuyinChar

# Main Bopomofo letters render at this fraction of the base character's font size.
ZHUYIN_FONT_RATIO = 0.32
# Small fixed breathing room between a character's phonetic column and the next character.
ZHUYIN_GAP_PAD = 1.0


def _zhuyin_font_size(font_size: float) -> float:
    return font_size * ZHUYIN_FONT_RATIO


def _cell_width(zc: ZhuyinChar, font: str, font_size: float) -> float:
    """Total horizontal space one character (plus its annotation, if any) needs."""
    char_w = pdfmetrics.stringWidth(zc.char, font, font_size)
    if not zc.main and not zc.tone_mark:
        return char_w

    zy_size = _zhuyin_font_size(font_size)
    main_w = max((pdfmetrics.stringWidth(ch, font, zy_size) for ch in zc.main), default=0.0)
    tone_w = pdfmetrics.stringWidth(zc.tone_mark, font, zy_size) if zc.tone_mark else 0.0
    return char_w + main_w + tone_w + ZHUYIN_GAP_PAD


def wrap_zhuyin(
    zchars: list[ZhuyinChar], font: str, font_size: float, max_width: float
) -> list[list[ZhuyinChar]]:
    """Wrap Zhuyin-annotated characters to a pixel width, character-by-character."""
    lines: list[list[ZhuyinChar]] = []
    current: list[ZhuyinChar] = []
    current_width = 0.0
    for zc in zchars:
        w = _cell_width(zc, font, font_size)
        if current and current_width + w > max_width:
            lines.append(current)
            current = [zc]
            current_width = w
        else:
            current.append(zc)
            current_width += w
    if current:
        lines.append(current)
    return lines


def draw_zhuyin_line(
    c: canvas.Canvas,
    line: list[ZhuyinChar],
    font: str,
    font_size: float,
    x: float,
    y: float,
    w: float,
    align: Literal["left", "center"],
    fill_color: colors.Color = colors.black,
) -> None:
    """Draw one wrapped line, baseline at `y`.

    Each character is followed by its stacked main Bopomofo letters (small
    font, *centered* on the character's vertical middle -- not pinned to a
    fixed top, so a short stack still spans the character's height instead
    of only ever occupying its upper portion) and its tone mark positioned
    beside the *last* letter of the stack: 2nd tone (ˊ) just above the last
    letter, 3rd tone (ˇ) level with it, 4th tone (ˋ) just below it. Neutral
    tone (˙) is the exception, drawn above the first (topmost) letter. 1st
    tone (no mark) is drawn nowhere.
    """
    zy_size = _zhuyin_font_size(font_size)
    total_width = sum(_cell_width(zc, font, font_size) for zc in line)
    cursor_x = x if align == "left" else x + (w - total_width) / 2

    c.setFillColor(fill_color)
    for zc in line:
        cell_w = _cell_width(zc, font, font_size)
        c.setFont(font, font_size)
        c.drawString(cursor_x, y, zc.char)
        char_w = pdfmetrics.stringWidth(zc.char, font, font_size)

        if zc.main or zc.tone_mark:
            c.setFont(font, zy_size)
            letter_x = cursor_x + char_w + 1.5
            # Center the stack on the character's vertical middle: a stack of
            # N letters spans (N-1) * zy_size * 1.05, so starting half that
            # span above the middle (and working downward) keeps the stack
            # centered regardless of N, instead of always starting at a fixed
            # top -- which left short stacks stranded in the character's
            # upper portion, never reaching its lower half.
            stack_span = max(len(zc.main) - 1, 0) * zy_size * 1.05
            letter_y = y + (font_size - zy_size) / 2 + stack_span / 2
            first_letter_y = letter_y
            last_letter_y = letter_y
            for letter in zc.main:
                c.drawString(letter_x, letter_y, letter)
                last_letter_y = letter_y
                letter_y -= zy_size * 1.05

            if zc.tone_mark:
                main_w = max(
                    (pdfmetrics.stringWidth(ch, font, zy_size) for ch in zc.main), default=0.0
                )
                if zc.tone_mark == "˙":
                    # Above the first (topmost) letter -- not beside the stack like the
                    # other tone marks, per this feature's confirmed layout design.
                    tone_x = letter_x
                    tone_y = first_letter_y + zy_size * 1.05
                else:
                    tone_x = letter_x + main_w + 0.5
                    # Anchored to the *last* main letter's row, not the character
                    # cell's fixed top -- a mark pinned to the absolute top
                    # regardless of stack length drifts away from a multi-letter
                    # stack's actual last letter, reading as floating above it
                    # rather than sitting beside it.
                    if zc.tone_mark == "ˊ":
                        tone_y = last_letter_y + zy_size * 0.3
                    elif zc.tone_mark == "ˇ":
                        tone_y = last_letter_y
                    else:  # "ˋ"
                        tone_y = last_letter_y - zy_size * 0.3
                c.drawString(tone_x, tone_y, zc.tone_mark)

        cursor_x += cell_w
