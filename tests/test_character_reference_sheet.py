"""Comparison sheet: one row per character_reference condition, one column per spread."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.eval.character_reference_sheet import render_character_reference_sheet

RED, GREEN, BLUE = (220, 30, 30), (30, 200, 30), (30, 30, 220)


def _spreads(folder: Path, color: tuple[int, int, int], pages: int) -> list[str]:
    folder.mkdir(parents=True)
    paths = []
    for page in range(1, pages + 1):
        path = folder / f"page_{page:02d}.png"
        Image.new("RGB", (1408, 992), color).save(path)
        paths.append(str(path))
    return paths


def test_sheet_has_a_label_column_then_one_tile_per_spread_for_each_condition(tmp_path):
    out_path = render_character_reference_sheet(
        rows={
            "none": _spreads(tmp_path / "none", RED, 4),
            "previous_spread": _spreads(tmp_path / "previous", GREEN, 4),
            "source_photo": _spreads(tmp_path / "photo", BLUE, 4),
        },
        out_path=tmp_path / "sheet.png",
        tile_size=(140, 100),
        label_px=120,
    )

    with Image.open(out_path) as sheet:
        # 120 px label column + 4 spreads x 140 px; 3 conditions x 100 px.
        assert sheet.size == (680, 300)
        assert sheet.getpixel((120 + 70, 50)) == RED
        assert sheet.getpixel((120 + 3 * 140 + 70, 150)) == GREEN
        assert sheet.getpixel((120 + 140 + 70, 250)) == BLUE


def test_sheet_for_a_family_album_has_only_the_conditions_it_was_given(tmp_path):
    out_path = render_character_reference_sheet(
        rows={
            "none": _spreads(tmp_path / "none", RED, 4),
            "previous_spread": _spreads(tmp_path / "previous", GREEN, 4),
        },
        out_path=tmp_path / "sheet.png",
        tile_size=(140, 100),
        label_px=120,
    )

    with Image.open(out_path) as sheet:
        assert sheet.size == (680, 200)
