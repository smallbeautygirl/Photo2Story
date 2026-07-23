"""Text-safe region detection for full-bleed picture-book illustrations.

Analyzes an illustration into a per-pixel "suitability map" using classical
computer vision (saliency + edge density + local variance -- no segmentation
model, no GPU) and searches three candidate rectangle shapes for the
best-scoring, font-fitted placement. See
docs/superpowers/specs/2026-07-22-picture-book-text-overlay-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from src.utils.text_wrap import wrapped_line_count

FONT_SIZE_MIN = 14.0
FONT_SIZE_MAX = 26.0
FONT_SIZE_STEP = 2.0

# A candidate's average badness must be at or below this to avoid a scrim.
SAFE_THRESHOLD = 0.35
# Opacity of the fallback scrim panel when no candidate clears SAFE_THRESHOLD.
SCRIM_OPACITY = 0.55
# A candidate's average variance at or below this draws as plain text;
# above it (but under SAFE_THRESHOLD for badness) gets an outline instead.
CONTRAST_SAFE_VARIANCE = 0.15

# Three shape presets, as a fraction of page width, tried independently.
SHAPE_PRESET_WIDTHS = (0.28, 0.45, 0.65)

# badness = w_saliency * saliency + w_edge * edge_density + w_variance * variance
BADNESS_WEIGHTS = (0.5, 0.3, 0.2)

# combined = RANK_SUITABILITY_WEIGHT * (1 - badness)
#          + RANK_FONT_WEIGHT * font_ratio
#          - (RANK_SCRIM_PENALTY if requires_scrim else 0)
RANK_SUITABILITY_WEIGHT = 0.6
RANK_FONT_WEIGHT = 0.4
RANK_SCRIM_PENALTY = 0.25

# The illustration is downsampled to this long-side size before analysis --
# precision beyond a few dozen cells doesn't matter for region search, and it
# keeps every cv2 op fast regardless of the real illustration resolution.
ANALYSIS_LONG_SIDE = 200

# A candidate rectangle's height is capped at this fraction of the page height.
MAX_HEIGHT_FRACTION = 0.6

# Padding inside a candidate rectangle, in points, on every side of the text.
CANDIDATE_PAD_PT = 10.0


@dataclass(frozen=True)
class SuitabilityMap:
    """Per-pixel analysis of one illustration, at the downsampled analysis
    resolution. All three arrays share the same (H, W) shape."""

    badness: np.ndarray
    variance: np.ndarray
    brightness: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One candidate text placement. x/y/w/h are fractions of the page's
    width/height (x, w relative to width; y, h relative to height), with y
    measured from the top of the page."""

    x: float
    y: float
    w: float
    h: float
    font_size: float
    badness: float
    variance: float
    brightness: float
    requires_scrim: bool


def _integral_image(arr: np.ndarray) -> np.ndarray:
    """Summed-area table: integral[y, x] is the sum of arr[:y, :x]."""
    integral = np.zeros((arr.shape[0] + 1, arr.shape[1] + 1), dtype=np.float64)
    integral[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    return integral


def _rect_sum(integral: np.ndarray, y0: int, x0: int, y1: int, x1: int) -> float:
    """Sum of the original array over rows [y0, y1) and columns [x0, x1)."""
    return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])


def _rect_sums_for_size(integral: np.ndarray, rect_h: int, rect_w: int) -> np.ndarray:
    """Sum of the original array for every valid top-left position of a
    rect_h x rect_w rectangle, vectorized via the integral image."""
    height, width = integral.shape[0] - 1, integral.shape[1] - 1
    a = integral[rect_h:, rect_w:]
    b = integral[: height - rect_h + 1, rect_w:]
    c = integral[rect_h:, : width - rect_w + 1]
    d = integral[: height - rect_h + 1, : width - rect_w + 1]
    return a - b - c + d
