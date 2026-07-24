from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.stages.text_placement import _integral_image, _rect_sum, _rect_sums_for_size


def test_integral_image_rect_sum_matches_direct_sum():
    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    integral = _integral_image(arr)

    assert _rect_sum(integral, 0, 0, 4, 5) == arr.sum()
    assert _rect_sum(integral, 1, 1, 3, 4) == arr[1:3, 1:4].sum()


def test_rect_sums_for_size_matches_direct_sums_at_every_position():
    arr = np.arange(20, dtype=np.float64).reshape(4, 5)
    integral = _integral_image(arr)

    sums = _rect_sums_for_size(integral, rect_h=2, rect_w=3)

    assert sums.shape == (3, 3)
    for y0 in range(3):
        for x0 in range(3):
            assert sums[y0, x0] == arr[y0 : y0 + 2, x0 : x0 + 3].sum()


def test_mask_gutter_straddling_sets_straddling_positions_to_inf():
    from src.stages.text_placement import _mask_gutter_straddling

    sums = np.zeros((1, 10))

    masked = _mask_gutter_straddling(sums, rect_w=4, gutter_col=5)

    # x0=1: window [1, 5) ends exactly at the gutter column -- doesn't straddle it.
    assert masked[0, 1] == 0.0
    # x0=2: window [2, 6) covers column 5 -- straddles.
    assert masked[0, 2] == np.inf
    # x0=3: window [3, 7) covers column 5 -- straddles.
    assert masked[0, 3] == np.inf
    # x0=5: window [5, 9) starts exactly at the gutter column -- doesn't straddle it.
    assert masked[0, 5] == 0.0


def test_analyze_badness_scores_flat_region_low_and_textured_region_high():
    """A flat light region and a high-contrast checkerboard region: the
    checkerboard must score as less text-safe (higher badness)."""
    from src.stages.text_placement import analyze_badness

    size = 200
    arr = np.full((size, size, 3), 230, dtype=np.uint8)
    half = size // 2
    checker = ((np.indices((size, half)).sum(axis=0) % 2) * 255).astype(np.uint8)
    arr[:, half:, 0] = checker
    arr[:, half:, 1] = checker
    arr[:, half:, 2] = checker
    image = Image.fromarray(arr, mode="RGB")

    badness_map = analyze_badness(image, page_width=200.0, page_height=200.0)
    grid_h, grid_w = badness_map.badness.shape
    flat_region = badness_map.badness[:, : grid_w // 4]
    textured_region = badness_map.badness[:, 3 * grid_w // 4 :]

    assert flat_region.mean() < textured_region.mean()


def test_analyze_badness_brightness_matches_grayscale_level():
    """A uniformly dark image's brightness channel must average low; a
    uniformly light image's must average high."""
    from src.stages.text_placement import analyze_badness

    dark = Image.fromarray(np.full((100, 100, 3), 20, dtype=np.uint8), mode="RGB")
    light = Image.fromarray(np.full((100, 100, 3), 235, dtype=np.uint8), mode="RGB")

    assert analyze_badness(dark, page_width=100.0, page_height=100.0).brightness.mean() < 60
    assert analyze_badness(light, page_width=100.0, page_height=100.0).brightness.mean() > 200


def test_analyze_badness_saliency_weight_drives_most_of_the_badness_gap(monkeypatch):
    """Saliency (weight 0.5, the largest of the three) must account for most
    of the badness gap between a salient shape and plain background -- not
    edge density and variance alone, which only produce a small boundary
    signal. Verified by comparing the gap at full weights against the gap
    with saliency zeroed out."""
    from src.stages import text_placement

    # Construct image: left half plain gray, right half gray with white circle
    size = 200
    arr = np.full((size, size, 3), 128, dtype=np.uint8)
    center_y, center_x = size // 2, 3 * size // 4
    radius = 20
    y, x = np.ogrid[:size, :size]
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius**2
    arr[mask] = 255
    image = Image.fromarray(arr, mode="RGB")

    # Measure badness gap with full weights
    full = text_placement.analyze_badness(image, page_width=200.0, page_height=200.0)
    grid_h, grid_w = full.badness.shape
    left_mean = full.badness[:, : grid_w // 2].mean()
    right_mean = full.badness[:, grid_w // 2 :].mean()
    full_gap = right_mean - left_mean

    # Measure badness gap with saliency zeroed
    monkeypatch.setattr(text_placement, "BADNESS_WEIGHTS", (0.0, 0.3, 0.2))
    no_saliency = text_placement.analyze_badness(image, page_width=200.0, page_height=200.0)
    no_sal_left_mean = no_saliency.badness[:, : grid_w // 2].mean()
    no_sal_right_mean = no_saliency.badness[:, grid_w // 2 :].mean()
    no_saliency_gap = no_sal_right_mean - no_sal_left_mean

    # Saliency must account for > 50% of the gap (causal ablation)
    assert no_saliency_gap < full_gap * 0.5


def test_analyze_badness_only_covers_the_cover_fit_visible_region():
    """A wide illustration on a square page has its left and right margins
    cropped away by cover-fit. If analysis ran on the raw, uncropped image,
    noisy margins would pull the average badness up; analysis must instead
    only see the plain center strip that will actually be shown."""
    from src.stages.text_placement import analyze_badness

    width, height = 400, 100
    rng = np.random.default_rng(0)
    arr = np.full((height, width, 3), 230, dtype=np.uint8)
    noisy = (rng.integers(0, 2, size=(height, 150)) * 255).astype(np.uint8)
    for channel in range(3):
        arr[:, :150, channel] = noisy
        arr[:, 250:, channel] = noisy
    image = Image.fromarray(arr, mode="RGB")

    # Cover-fit onto a 100x100 page only shows image columns [150, 250) --
    # the plain center -- cropping away both noisy margins entirely.
    badness_map = analyze_badness(image, page_width=100.0, page_height=100.0)

    assert badness_map.badness.mean() < 0.1


def test_search_candidates_returns_one_per_shape_preset_with_matching_preset_field():
    from src.stages.text_placement import (
        SHAPE_PRESET_NAMES,
        SHAPE_PRESET_WIDTHS,
        analyze_badness,
        search_candidates,
    )

    image = Image.fromarray(np.full((200, 200, 3), 230, dtype=np.uint8), mode="RGB")
    badness_map = analyze_badness(image, page_width=400.0, page_height=400.0)

    candidates = search_candidates(
        badness_map, "A short caption.", "Helvetica", "en", 400.0, 400.0
    )

    assert len(candidates) == len(SHAPE_PRESET_WIDTHS)
    assert [round(c.w, 2) for c in candidates] == [round(w, 2) for w in SHAPE_PRESET_WIDTHS]
    assert [c.preset for c in candidates] == list(SHAPE_PRESET_NAMES)


def test_search_candidates_narrower_preset_gets_smaller_or_equal_font_for_long_caption():
    """A long caption needs more lines at a narrower width, growing the
    candidate rectangle's height until it crosses from a safe zone into an
    unsafe one and is forced to a smaller font -- a narrower preset must
    never end up with a *larger* font than a wider preset for the same
    caption.

    A flat, uniformly-colored image (as used in the other tests here) turns
    out to score as near-zero badness everywhere at every rectangle size --
    verified empirically up to 300 repeated words with no font-size change --
    so it can never exercise this code path regardless of caption length.
    Instead, a BadnessMap is built directly with a hard safe/unsafe
    boundary partway down the grid: a rectangle that must grow past that
    boundary to fit its text picks up unsafe badness and is forced smaller.
    """
    from src.stages.text_placement import BadnessMap, search_candidates

    grid_size = 200
    safe_rows = 60
    badness = np.zeros((grid_size, grid_size), dtype=np.float32)
    badness[safe_rows:] = 0.9
    badness_map = BadnessMap(
        badness=badness, variance=np.zeros_like(badness), brightness=np.zeros_like(badness)
    )
    long_caption = " ".join(["word"] * 15)

    candidates = search_candidates(badness_map, long_caption, "Helvetica", "en", 400.0, 400.0)
    by_width = sorted(candidates, key=lambda c: c.w)

    assert by_width[0].font_size < by_width[-1].font_size


def test_search_candidates_flags_requires_scrim_when_nothing_clears_threshold():
    """Salt-and-pepper (binary black/white) noise keeps edge density and local
    variance uniformly high across every position -- unlike continuous random
    noise, which by chance can still contain locally smoother pockets that
    dip below SAFE_THRESHOLD -- so no candidate rectangle anywhere can clear
    the threshold."""
    from src.stages.text_placement import analyze_badness, search_candidates

    rng = np.random.default_rng(0)
    arr = (rng.integers(0, 2, size=(200, 200, 3)) * 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    badness_map = analyze_badness(image, page_width=400.0, page_height=400.0)

    candidates = search_candidates(
        badness_map, "A caption that needs placement.", "Helvetica", "en", 400.0, 400.0
    )

    assert all(c.requires_scrim for c in candidates)


def test_search_candidates_skips_wide_short_preset_when_has_gutter():
    """`wide_short` (0.65 of the page width) is wider than half a spread page
    and can never avoid straddling the binding gutter -- it must be skipped
    outright, not returned with a meaningless best-effort placement."""
    from src.stages.text_placement import BadnessMap, search_candidates

    badness_map = BadnessMap(
        badness=np.zeros((200, 200), dtype=np.float32),
        variance=np.zeros((200, 200), dtype=np.float32),
        brightness=np.zeros((200, 200), dtype=np.float32),
    )

    candidates = search_candidates(
        badness_map, "A short caption.", "Helvetica", "en", 400.0, 400.0, has_gutter=True
    )

    assert len(candidates) == 2
    assert all(c.preset != "wide_short" for c in candidates)


def test_pick_best_selects_highest_combined_score():
    from src.stages.text_placement import Candidate, pick_best

    clean_small_font = Candidate(
        x=0.1, y=0.1, w=0.3, h=0.1, font_size=14.0, badness=0.05, variance=0.05,
        brightness=230.0, requires_scrim=False,
    )
    slightly_busier_large_font = Candidate(
        x=0.5, y=0.5, w=0.5, h=0.2, font_size=26.0, badness=0.2, variance=0.2,
        brightness=200.0, requires_scrim=False,
    )
    scrim_required = Candidate(
        x=0.0, y=0.0, w=0.6, h=0.3, font_size=26.0, badness=0.5, variance=0.5,
        brightness=100.0, requires_scrim=True,
    )

    best = pick_best([clean_small_font, slightly_busier_large_font, scrim_required])

    assert best is slightly_busier_large_font


def test_pick_best_penalizes_scrim_requirement():
    """Two candidates identical in every scoring input except `requires_scrim`
    must have the non-scrim one win. This pins the penalty's *sign*: a
    formula that omits the penalty, or applies it as a bonus, would instead
    tie the two candidates (omitted) or pick the scrim one (bonus) -- both
    of which fail this assertion.
    """
    from src.stages.text_placement import Candidate, pick_best

    without_scrim = Candidate(
        x=0.1, y=0.1, w=0.3, h=0.1, font_size=20.0, badness=0.3, variance=0.3,
        brightness=200.0, requires_scrim=False,
    )
    with_scrim = Candidate(
        x=0.1, y=0.1, w=0.3, h=0.1, font_size=20.0, badness=0.3, variance=0.3,
        brightness=200.0, requires_scrim=True,
    )

    best = pick_best([with_scrim, without_scrim])

    assert best is without_scrim


def test_pick_best_weighs_badness_above_font_size():
    """Constructed so the correct weighting (0.6 badness, 0.4 font size) and
    a swapped weighting (0.4 badness, 0.6 font size) disagree on the winner
    -- catching a formula that swaps `RANK_BADNESS_WEIGHT` and
    `RANK_FONT_WEIGHT`.

    low_badness_small_font: badness=0.0 ((1 - badness) term 1.0), font_size at
    FONT_SIZE_MIN (font_ratio 0.0) -> correct score 0.6, swapped score 0.4.
    high_badness_large_font: badness=1.0 ((1 - badness) term 0.0), font_size
    at FONT_SIZE_MAX (font_ratio 1.0) -> correct score 0.4, swapped score 0.6.
    The correct formula must pick the first; a swapped-weight formula would
    pick the second instead.
    """
    from src.stages.text_placement import Candidate, pick_best

    low_badness_small_font = Candidate(
        x=0.1, y=0.1, w=0.3, h=0.1, font_size=14.0, badness=0.0, variance=0.1,
        brightness=200.0, requires_scrim=False,
    )
    high_badness_large_font = Candidate(
        x=0.1, y=0.1, w=0.3, h=0.1, font_size=26.0, badness=1.0, variance=0.1,
        brightness=200.0, requires_scrim=False,
    )

    best = pick_best([low_badness_small_font, high_badness_large_font])

    assert best is low_badness_small_font


def test_pick_best_raises_on_empty_list():
    from src.stages.text_placement import pick_best

    with pytest.raises(ValueError, match="no candidates"):
        pick_best([])
