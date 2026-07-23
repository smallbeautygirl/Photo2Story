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


def test_build_picture_book_pdf_creates_file_short_caption(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    pages = ["Once upon a time.", "They had fun.", "The end."]
    out_path = str(tmp_path / "picture_book_bottom.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_picture_book_pdf_creates_file_long_caption(tmp_path, tmp_images):
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


def test_cover_fit_image_fills_entire_page(tmp_path, tmp_images):
    """Unlike contain-fit, cover-fit must never leave visible page margin --
    the drawn image must be at least as large as the page in both dimensions."""
    from PIL import Image
    from reportlab.pdfgen import canvas as canvas_module

    from src.stages.stage3_assemble import PAGE_H, PAGE_W, _cover_fit_image

    calls = []
    c = canvas_module.Canvas(str(tmp_path / "cover.pdf"))
    original_draw_image = c.drawImage

    def _recording_draw_image(image, x, y, width, height, **kwargs):
        calls.append((x, y, width, height))
        return original_draw_image(image, x, y, width=width, height=height, **kwargs)

    c.drawImage = _recording_draw_image
    image = Image.open(tmp_images[0]).convert("RGB")
    _cover_fit_image(c, image, PAGE_W, PAGE_H)

    x, y, width, height = calls[0]
    assert width >= PAGE_W - 0.01
    assert height >= PAGE_H - 0.01


def test_draw_caption_overlay_draws_plain_text_for_low_variance_candidate():
    from reportlab.pdfgen import canvas as canvas_module

    from src.stages.stage3_assemble import _draw_caption_overlay, PAGE_H, PAGE_W, _resolve_cjk_font
    from src.stages.text_placement import Candidate

    font = _resolve_cjk_font()
    candidate = Candidate(
        x=0.1, y=0.1, w=0.5, h=0.2, font_size=20.0, badness=0.1, variance=0.05,
        brightness=230.0, requires_scrim=False,
    )
    calls = []
    c = canvas_module.Canvas("/dev/null")
    original_draw_string = c.drawString

    def _recording_draw_string(x, y, text):
        calls.append(text)
        return original_draw_string(x, y, text)

    c.drawString = _recording_draw_string
    _draw_caption_overlay(c, candidate, "Hello there.", font, PAGE_W, PAGE_H)

    assert calls == ["Hello there."]


def test_draw_caption_overlay_draws_outline_pass_for_high_variance_no_scrim_candidate():
    from reportlab.pdfgen import canvas as canvas_module

    from src.stages.stage3_assemble import _draw_caption_overlay, PAGE_H, PAGE_W, _resolve_cjk_font
    from src.stages.text_placement import Candidate

    font = _resolve_cjk_font()
    candidate = Candidate(
        x=0.1, y=0.1, w=0.5, h=0.2, font_size=20.0, badness=0.3, variance=0.5,
        brightness=230.0, requires_scrim=False,
    )
    calls = []
    c = canvas_module.Canvas("/dev/null")
    original_draw_string = c.drawString

    def _recording_draw_string(x, y, text):
        calls.append((x, y, text))
        return original_draw_string(x, y, text)

    c.drawString = _recording_draw_string
    _draw_caption_overlay(c, candidate, "Hi.", font, PAGE_W, PAGE_H)

    # outline pass (offset) + main pass (no offset) => same text drawn twice at
    # different positions
    matching = [call for call in calls if call[2] == "Hi."]
    assert len(matching) == 2
    assert matching[0][:2] != matching[1][:2]


def test_draw_caption_overlay_draws_scrim_rect_for_requires_scrim_candidate():
    from reportlab.pdfgen import canvas as canvas_module

    from src.stages.stage3_assemble import _draw_caption_overlay, PAGE_H, PAGE_W, _resolve_cjk_font
    from src.stages.text_placement import Candidate

    font = _resolve_cjk_font()
    candidate = Candidate(
        x=0.1, y=0.1, w=0.5, h=0.2, font_size=14.0, badness=0.6, variance=0.6,
        brightness=100.0, requires_scrim=True,
    )
    calls = []
    c = canvas_module.Canvas("/dev/null")
    original_rect = c.rect

    def _recording_rect(*args, **kwargs):
        calls.append((args, kwargs))
        return original_rect(*args, **kwargs)

    c.rect = _recording_rect
    _draw_caption_overlay(c, candidate, "Hi.", font, PAGE_W, PAGE_H)

    assert len(calls) == 1
