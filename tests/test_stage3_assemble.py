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
