# tests/test_zhuyin_render.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.stages.zhuyin_render import draw_zhuyin_line, wrap_zhuyin
from src.utils.zhuyin import annotate


def test_cell_width_wider_than_plain_character():
    """A Zhuyin-annotated character must reserve more width than its plain
    glyph alone, since there needs to be room for the phonetic column."""
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    from src.stages.zhuyin_render import _cell_width

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    zc = annotate("陶")[0]
    plain_width = pdfmetrics.stringWidth("陶", "STSong-Light", 24)

    assert _cell_width(zc, "STSong-Light", 24) > plain_width


def test_wrap_zhuyin_wraps_at_max_width():
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    zchars = annotate("陶樂蒂的開學日")

    # A narrow width forces multiple lines; a very wide one fits on one line.
    narrow_lines = wrap_zhuyin(zchars, "STSong-Light", 24, max_width=60)
    wide_lines = wrap_zhuyin(zchars, "STSong-Light", 24, max_width=10_000)

    assert len(narrow_lines) > 1
    assert len(wide_lines) == 1
    # every character must appear exactly once across all lines, in order
    flat = [zc.char for line in narrow_lines for zc in line]
    assert flat == list("陶樂蒂的開學日")


def test_wrap_zhuyin_empty_input_returns_empty_list():
    assert wrap_zhuyin([], "Helvetica", 24, max_width=1000) == []


def test_draw_zhuyin_line_runs_without_error(tmp_path):
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    zchars = annotate("我不要上學！")
    lines = wrap_zhuyin(zchars, "STSong-Light", 24, max_width=10_000)

    out_path = str(tmp_path / "zhuyin_test.pdf")
    c = canvas.Canvas(out_path, pagesize=A4)
    for line in lines:
        draw_zhuyin_line(c, line, "STSong-Light", 24, x=50, y=700, w=400, align="center")
    c.save()

    from pathlib import Path

    assert Path(out_path).exists()
