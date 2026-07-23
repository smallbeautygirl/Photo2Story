from __future__ import annotations

import numpy as np
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


def test_analyze_suitability_scores_flat_region_low_and_textured_region_high():
    """A flat light region and a high-contrast checkerboard region: the
    checkerboard must score as less text-safe (higher badness)."""
    from src.stages.text_placement import analyze_suitability

    size = 200
    arr = np.full((size, size, 3), 230, dtype=np.uint8)
    half = size // 2
    checker = ((np.indices((size, half)).sum(axis=0) % 2) * 255).astype(np.uint8)
    arr[:, half:, 0] = checker
    arr[:, half:, 1] = checker
    arr[:, half:, 2] = checker
    image = Image.fromarray(arr, mode="RGB")

    suitability = analyze_suitability(image)
    grid_h, grid_w = suitability.badness.shape
    flat_region = suitability.badness[:, : grid_w // 4]
    textured_region = suitability.badness[:, 3 * grid_w // 4 :]

    assert flat_region.mean() < textured_region.mean()


def test_analyze_suitability_brightness_matches_grayscale_level():
    """A uniformly dark image's brightness channel must average low; a
    uniformly light image's must average high."""
    from src.stages.text_placement import analyze_suitability

    dark = Image.fromarray(np.full((100, 100, 3), 20, dtype=np.uint8), mode="RGB")
    light = Image.fromarray(np.full((100, 100, 3), 235, dtype=np.uint8), mode="RGB")

    assert analyze_suitability(dark).brightness.mean() < 60
    assert analyze_suitability(light).brightness.mean() > 200
