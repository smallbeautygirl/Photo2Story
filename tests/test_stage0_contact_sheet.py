"""Contact sheet: every photo as a thumbnail, selected ones framed, labelled with cluster/score."""

from __future__ import annotations

from PIL import Image

from src.eval.stage0_contact_sheet import HIGHLIGHT_RGB, render_contact_sheet


def test_contact_sheet_lays_out_all_photos_in_a_grid(tmp_images, tmp_path):
    out_path = render_contact_sheet(
        image_paths=tmp_images,
        selected_paths=[tmp_images[1]],
        cluster_labels={p: i % 2 for i, p in enumerate(tmp_images)},
        scores={tmp_images[1]: 0.9},
        out_path=tmp_path / "sheet.jpg",
        columns=3,
        thumb_px=100,
        caption_px=24,
    )

    with Image.open(out_path) as sheet:
        # 5 photos in 3 columns -> 2 rows; each tile is 100 px of image + 24 px of caption.
        assert sheet.size == (300, 248)


def test_contact_sheet_frames_only_the_selected_photos(tmp_images, tmp_path):
    out_path = render_contact_sheet(
        image_paths=tmp_images,
        selected_paths=[tmp_images[1]],
        cluster_labels={},
        scores={},
        out_path=tmp_path / "sheet.png",
        columns=3,
        thumb_px=100,
        caption_px=24,
    )

    with Image.open(out_path) as sheet:
        selected_tile_corner = sheet.getpixel((100 + 2, 2))
        unselected_tile_corner = sheet.getpixel((2, 2))

    assert selected_tile_corner == HIGHLIGHT_RGB
    assert unselected_tile_corner != HIGHLIGHT_RGB
