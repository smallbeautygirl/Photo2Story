from __future__ import annotations

import numpy as np

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
