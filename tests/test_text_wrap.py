from __future__ import annotations

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from src.utils.text_wrap import wrap_to_width, wrapped_line_count


def test_wrap_to_width_breaks_on_spaces_for_latin_text():
    lines = wrap_to_width("one two three four five", "Helvetica", 12, max_width=60)
    assert len(lines) > 1
    assert " ".join(lines) == "one two three four five"


def test_wrap_to_width_breaks_per_character_when_no_spaces():
    lines = wrap_to_width("陶樂蒂的開學日", "Helvetica", 12, max_width=30)
    assert len(lines) > 1
    assert "".join(lines) == "陶樂蒂的開學日"


def test_wrap_to_width_empty_string_returns_empty_list():
    assert wrap_to_width("", "Helvetica", 12, max_width=1000) == []


def test_wrapped_line_count_matches_wrap_to_width_for_en():
    text = "one two three four five six seven"
    assert wrapped_line_count(text, "Helvetica", 12, 60, "en") == len(
        wrap_to_width(text, "Helvetica", 12, 60)
    )


def test_wrapped_line_count_uses_zhuyin_wrapping_for_zh_tw():
    """A Hanzi string that fits in 2 plain-wrapped lines can need more once
    Zhuyin annotation widens each character."""
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    caption = "陶樂蒂的開學日今天真是漂亮的一天大家都很開心呢" + "啊" * 10

    en_count = wrapped_line_count(caption, "STSong-Light", 24, 500, "en")
    zh_count = wrapped_line_count(caption, "STSong-Light", 24, 500, "zh-tw")

    assert zh_count > en_count
