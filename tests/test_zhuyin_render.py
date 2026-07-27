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


def test_draw_zhuyin_line_positions_all_four_tone_marks_distinctly():
    """Regression test: the neutral tone (˙) must render above the letter
    stack, not at the same point as the 2nd tone (ˊ) beside it -- these were
    previously identical due to a copy-paste bug in the tone_y branches."""
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    from src.stages.zhuyin_render import draw_zhuyin_line
    from src.utils.zhuyin import ZhuyinChar

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    calls = []

    class _RecordingCanvas:
        def __init__(self, real):
            self._real = real

        def setFillColor(self, *a, **kw):
            self._real.setFillColor(*a, **kw)

        def setFont(self, *a, **kw):
            self._real.setFont(*a, **kw)

        def drawString(self, x, y, text):
            calls.append((x, y, text))
            self._real.drawString(x, y, text)

    from reportlab.pdfgen import canvas as canvas_module

    real_canvas = canvas_module.Canvas("/dev/null")
    wrapped = _RecordingCanvas(real_canvas)

    zchars = [
        ZhuyinChar(char="媽", main="ㄇㄚ", tone_mark=""),
        ZhuyinChar(char="麻", main="ㄇㄚ", tone_mark="ˊ"),
        ZhuyinChar(char="馬", main="ㄇㄚ", tone_mark="ˇ"),
        ZhuyinChar(char="罵", main="ㄇㄚ", tone_mark="ˋ"),
        ZhuyinChar(char="嗎", main="ㄇㄚ", tone_mark="˙"),
    ]
    draw_zhuyin_line(wrapped, zchars, "STSong-Light", 24, x=50, y=700, w=400, align="left")

    tone_positions = {text: (x, y) for x, y, text in calls if text in "ˊˇˋ˙"}
    assert len(tone_positions) == 4
    # all four tone marks must land at distinct (x, y) points
    assert len(set(tone_positions.values())) == 4
    # neutral tone must be strictly above the 2nd-tone mark (both share the same x-ish
    # region beside/above the stack, but neutral must be higher, not identical)
    assert tone_positions["˙"][1] > tone_positions["ˊ"][1]


def test_draw_zhuyin_line_tone_mark_tracks_last_letter_not_fixed_cell_height():
    """Regression test: a tone mark must be anchored to the stack's *last*
    letter, not a fixed height within the character cell -- otherwise a
    multi-letter stack's tone mark drifts away from its own letters (reads as
    floating above the stack) the more letters that stack has."""
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    from src.stages.zhuyin_render import draw_zhuyin_line
    from src.utils.zhuyin import ZhuyinChar

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    calls = []

    class _RecordingCanvas:
        def __init__(self, real):
            self._real = real

        def setFillColor(self, *a, **kw):
            self._real.setFillColor(*a, **kw)

        def setFont(self, *a, **kw):
            self._real.setFont(*a, **kw)

        def drawString(self, x, y, text):
            calls.append((x, y, text))
            self._real.drawString(x, y, text)

    from reportlab.pdfgen import canvas as canvas_module

    real_canvas = canvas_module.Canvas("/dev/null")
    wrapped = _RecordingCanvas(real_canvas)

    # Same tone mark, different stack lengths (1 letter vs. 3 letters).
    zchars = [
        ZhuyinChar(char="一", main="ㄧ", tone_mark="ˊ"),
        ZhuyinChar(char="娘", main="ㄋㄧㄤ", tone_mark="ˊ"),
    ]
    draw_zhuyin_line(wrapped, zchars, "STSong-Light", 24, x=50, y=700, w=400, align="left")

    tone_ys = [y for _, y, text in calls if text == "ˊ"]
    assert len(tone_ys) == 2
    # The 3-letter stack's last letter sits lower than the 1-letter stack's --
    # its tone mark must track that, landing lower too, not at the same height.
    assert tone_ys[1] < tone_ys[0]


def test_draw_zhuyin_line_single_letter_stack_centers_on_character_middle():
    """Regression test: a single-letter stack must be centered on the
    character's vertical middle, not pinned to the same fixed top used for
    longer stacks -- otherwise a short stack is stranded near the top of the
    character's height, never reaching its lower half."""
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    from src.stages.zhuyin_render import _zhuyin_font_size, draw_zhuyin_line
    from src.utils.zhuyin import ZhuyinChar

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    calls = []

    class _RecordingCanvas:
        def __init__(self, real):
            self._real = real

        def setFillColor(self, *a, **kw):
            self._real.setFillColor(*a, **kw)

        def setFont(self, *a, **kw):
            self._real.setFont(*a, **kw)

        def drawString(self, x, y, text):
            calls.append((x, y, text))
            self._real.drawString(x, y, text)

    from reportlab.pdfgen import canvas as canvas_module

    real_canvas = canvas_module.Canvas("/dev/null")
    wrapped = _RecordingCanvas(real_canvas)

    font_size = 24
    y = 700
    zchars = [ZhuyinChar(char="魚", main="ㄩ", tone_mark="ˊ")]
    draw_zhuyin_line(wrapped, zchars, "STSong-Light", font_size, x=50, y=y, w=400, align="left")

    zy_size = _zhuyin_font_size(font_size)
    letter_y = next(letter_y for _, letter_y, text in calls if text == "ㄩ")
    expected_center = y + (font_size - zy_size) / 2

    assert letter_y == expected_center


def test_draw_zhuyin_line_uses_given_fill_color():
    from reportlab.lib import colors
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    calls = []

    class _RecordingCanvas:
        def __init__(self, real):
            self._real = real

        def setFillColor(self, *a, **kw):
            calls.append(("setFillColor", a, kw))
            self._real.setFillColor(*a, **kw)

        def setFont(self, *a, **kw):
            self._real.setFont(*a, **kw)

        def drawString(self, x, y, text):
            self._real.drawString(x, y, text)

    from reportlab.pdfgen import canvas as canvas_module

    real_canvas = canvas_module.Canvas("/dev/null")
    wrapped = _RecordingCanvas(real_canvas)
    zchars = annotate("媽")

    draw_zhuyin_line(
        wrapped, zchars, "STSong-Light", 24, x=50, y=700, w=400, align="left",
        fill_color=colors.white,
    )

    assert ("setFillColor", (colors.white,), {}) in calls
