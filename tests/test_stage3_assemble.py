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
