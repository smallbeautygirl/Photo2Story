# tests/test_stage3_assemble.py
import pytest
from pathlib import Path
from src.stages.stage3_assemble import build_pdf


def test_build_pdf_creates_file(tmp_path, tmp_images):
    pages = ["Once upon a time.", "They had fun.", "The end."]
    descriptions = ["A sunny beach.", "Children laughing.", "Sunset over the sea."]
    illustrations = tmp_images[:3]
    out_path = str(tmp_path / "storybook.pdf")

    build_pdf(illustrations, pages, descriptions, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_pdf_wrong_count_raises(tmp_path, tmp_images):
    with pytest.raises(AssertionError):
        build_pdf(
            tmp_images[:2],
            ["Only one page."],
            ["Only one description."],
            str(tmp_path / "bad.pdf"),
        )


def test_choose_layout_variant_picks_top_band_for_long_captions():
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W

    long_caption = (
        "This is a very long caption that will definitely need more than two "
        "lines when wrapped at the page width because it just keeps going and "
        "going and going."
    )
    result = _choose_layout_variant(["Short one.", long_caption], "Helvetica", PAGE_W)
    assert result == "top_band"


def test_choose_layout_variant_picks_bottom_caption_for_short_captions():
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W

    result = _choose_layout_variant(["Short one.", "Another short one."], "Helvetica", PAGE_W)
    assert result == "bottom_caption"


def test_choose_layout_variant_uses_spread_width():
    """The same long caption that top-bands at page width must bottom-caption
    at a much wider width, proving the decision is measured against the
    page_width argument rather than a hardcoded constant."""
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W

    long_caption = (
        "This is a long caption that will wrap to several lines at normal "
        "page width because it just keeps going on and on with many more words."
    )
    narrow_result = _choose_layout_variant([long_caption], "Helvetica", PAGE_W)
    very_wide_result = _choose_layout_variant([long_caption], "Helvetica", PAGE_W * 10)

    assert narrow_result == "top_band"
    assert very_wide_result == "bottom_caption"


def test_text_zone_height_scales_with_longest_caption():
    from src.stages.stage3_assemble import _text_zone_height, TOP_BAND_FONT_SIZE, PAGE_W

    short_only = _text_zone_height(["Hi.", "Bye."], "Helvetica", TOP_BAND_FONT_SIZE, PAGE_W)
    with_long = _text_zone_height(
        ["Hi.", "This is a much longer caption that wraps across several lines of text."],
        "Helvetica",
        TOP_BAND_FONT_SIZE,
        PAGE_W,
    )
    assert with_long > short_only


def test_build_picture_book_pdf_creates_file_bottom_caption(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    pages = ["Once upon a time.", "They had fun.", "The end."]
    out_path = str(tmp_path / "picture_book_bottom.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_picture_book_pdf_creates_file_top_band(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    long_caption = (
        "This is a very long caption that will definitely need more than two "
        "lines when wrapped at the page width because it just keeps going and "
        "going."
    )
    pages = [long_caption, "Short.", "The end."]
    out_path = str(tmp_path / "picture_book_top.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_picture_book_pdf_wrong_count_raises(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    with pytest.raises(AssertionError):
        build_picture_book_pdf(tmp_images[:2], ["Only one page."], str(tmp_path / "bad.pdf"))


def test_build_spread_pdf_creates_file(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_spread_pdf

    pages = ["Once upon a time.", "They had fun.", "The end."]
    out_path = str(tmp_path / "spread.pdf")

    build_spread_pdf(tmp_images[:3], pages, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_spread_pdf_zh_tw_creates_file(tmp_path, tmp_images):
    """picture_book_spread is the real pipeline's default layout
    (configs/demo.yaml) -- zh-tw + Zhuyin must work through this entry
    point, not just build_picture_book_pdf."""
    from src.stages.stage3_assemble import build_spread_pdf

    pages = ["我不要上學！", "但是米亞不想動。", "世代相傳。"]
    out_path = str(tmp_path / "spread_zh.pdf")

    build_spread_pdf(tmp_images[:3], pages, out_path, language="zh-tw")

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_spread_pdf_wrong_count_raises(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_spread_pdf

    with pytest.raises(AssertionError):
        build_spread_pdf(tmp_images[:2], ["Only one page."], str(tmp_path / "bad.pdf"))


def test_build_spread_pdf_uses_double_width_page(tmp_path, tmp_images):
    """A spread PDF's page must be twice the width of a single picture_book page."""
    import pypdf

    from src.stages.stage3_assemble import PAGE_W, build_spread_pdf

    out_path = str(tmp_path / "spread_size.pdf")
    build_spread_pdf(tmp_images[:2], ["One.", "Two."], out_path)

    reader = pypdf.PdfReader(out_path)
    page_width_pt = float(reader.pages[0].mediabox.width)
    assert abs(page_width_pt - 2 * PAGE_W) < 1.0


def test_resolve_cjk_font_falls_back_to_builtin_cid_font(monkeypatch):
    """When no embedded TTF font file is found, fall back to ReportLab's
    built-in STSong-Light CID font instead of plain Helvetica (which has no
    CJK glyphs at all)."""
    import src.stages.stage3_assemble as mod

    monkeypatch.setattr(mod, "_cjk_font_name", None)
    monkeypatch.setattr(mod, "_CJK_FONT_CANDIDATES", [])

    result = mod._resolve_cjk_font()

    assert result == "STSong-Light"


def test_choose_layout_variant_uses_zhuyin_wrapping_for_zh_tw():
    """A Hanzi string that fits in 2 plain-wrapped lines can need 3+ lines
    once Zhuyin annotation widens each character -- the decision must use
    the same wrapping the final draw will use, or the reserved zone could
    be too short. Verified numerically before writing this test: at
    TOP_BAND_FONT_SIZE against PAGE_W, this 33-character caption plain-wraps
    to 2 lines but Zhuyin-wraps to 3; at 20x the width it collapses to 1
    Zhuyin-wrapped line."""
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    caption = "陶樂蒂的開學日今天真是漂亮的一天大家都很開心呢" + "啊" * 10

    en_result = _choose_layout_variant([caption], "STSong-Light", PAGE_W, language="en")
    zh_result = _choose_layout_variant([caption], "STSong-Light", PAGE_W, language="zh-tw")
    zh_wide_result = _choose_layout_variant(
        [caption], "STSong-Light", PAGE_W * 20, language="zh-tw"
    )

    assert en_result == "bottom_caption"
    assert zh_result == "top_band"
    assert zh_wide_result == "bottom_caption"


def test_text_zone_height_larger_for_zhuyin_than_plain_at_same_width():
    from src.stages.stage3_assemble import _text_zone_height, TOP_BAND_FONT_SIZE, PAGE_W
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    caption = "陶樂蒂的開學日今天真是漂亮的一天大家都很開心呢" + "啊" * 10

    plain_h = _text_zone_height(
        [caption], "STSong-Light", TOP_BAND_FONT_SIZE, PAGE_W, language="en"
    )
    zhuyin_h = _text_zone_height(
        [caption], "STSong-Light", TOP_BAND_FONT_SIZE, PAGE_W, language="zh-tw"
    )

    assert zhuyin_h > plain_h


def test_build_picture_book_pdf_zh_tw_creates_file(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    pages = ["我不要上學！", "但是米亞不想動。", "世代相傳。"]
    out_path = str(tmp_path / "picture_book_zh.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_path, language="zh-tw")

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_run_stage3_threads_language_from_stage1_result(tmp_path, mocker):
    """run_stage3 must read language from stage1_result and pass it through
    to whichever build_* function it dispatches to -- the integration seam
    between run_stage1 returning language and build_* consuming it, which
    no single-function test exercises directly."""
    from src.stages.stage3_assemble import run_stage3

    mock_build_spread = mocker.patch("src.stages.stage3_assemble.build_spread_pdf")
    stage1_result = {"pages": ["頁一", "頁二"], "language": "zh-tw"}
    stage2_result = {"illustration_paths": ["a.png", "b.png"]}
    config = {"page_layout": "picture_book_spread"}

    run_stage3(stage1_result, stage2_result, ["d1", "d2"], config, str(tmp_path / "out.pdf"))

    mock_build_spread.assert_called_once()
    assert mock_build_spread.call_args.kwargs["language"] == "zh-tw"


def test_run_stage3_defaults_language_to_en_when_absent(tmp_path, mocker):
    """Older stage1_result dicts without a language key must not KeyError --
    run_stage3 should default to "en"."""
    from src.stages.stage3_assemble import run_stage3

    mock_build = mocker.patch("src.stages.stage3_assemble.build_picture_book_pdf")
    stage1_result = {"pages": ["A.", "B."]}
    stage2_result = {"illustration_paths": ["a.png", "b.png"]}
    config = {"page_layout": "picture_book"}

    run_stage3(stage1_result, stage2_result, ["d1", "d2"], config, str(tmp_path / "out.pdf"))

    mock_build.assert_called_once()
    assert mock_build.call_args.kwargs["language"] == "en"


def test_build_picture_book_pdf_default_language_unaffected(tmp_path, tmp_images):
    """Omitting `language` must produce the same rendered content as
    language="en" -- English output takes the untouched plain-text path
    either way. Compares extracted text/page count via pypdf rather than
    raw bytes: reportlab embeds a unique per-build /ID in the PDF trailer
    (confirmed by direct testing -- two back-to-back builds from identical
    inputs differ at the trailer, nowhere in the actual content streams),
    so byte-for-byte comparison is never satisfiable even when nothing
    about the rendering differs."""
    import pypdf

    from src.stages.stage3_assemble import build_picture_book_pdf

    pages = ["Once upon a time.", "They had fun.", "The end."]
    out_no_lang = str(tmp_path / "no_lang.pdf")
    out_en = str(tmp_path / "en.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_no_lang)
    build_picture_book_pdf(tmp_images[:3], pages, out_en, language="en")

    reader_no_lang = pypdf.PdfReader(out_no_lang)
    reader_en = pypdf.PdfReader(out_en)

    assert len(reader_no_lang.pages) == len(reader_en.pages)
    assert [p.extract_text() for p in reader_no_lang.pages] == [
        p.extract_text() for p in reader_en.pages
    ]
