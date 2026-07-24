"""Text-safe region detection for full-bleed picture-book illustrations.

Analyzes an illustration into a per-pixel badness map using classical
computer vision (saliency + edge density + local variance -- no segmentation
model, no GPU) and searches three candidate rectangle shapes for the
best-scoring, font-fitted placement. See
docs/superpowers/specs/2026-07-22-picture-book-text-overlay-design.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
SHAPE_PRESET_NAMES = ("narrow_tall", "medium", "wide_short")
SHAPE_PRESET_WIDTHS = (0.28, 0.45, 0.65)

# badness = w_saliency * saliency + w_edge * edge_density + w_variance * variance
BADNESS_WEIGHTS = (0.5, 0.3, 0.2)

# combined = RANK_BADNESS_WEIGHT * (1 - badness)
#          + RANK_FONT_WEIGHT * font_ratio
#          - (RANK_SCRIM_PENALTY if requires_scrim else 0)
RANK_BADNESS_WEIGHT = 0.6
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

ShapePreset = Literal["narrow_tall", "medium", "wide_short"]


@dataclass(frozen=True)
class BadnessMap:
    """Per-pixel analysis of one illustration, at the downsampled analysis
    resolution. All three arrays share the same (H, W) shape."""

    badness: np.ndarray
    variance: np.ndarray
    brightness: np.ndarray


@dataclass(frozen=True)
class Candidate:
    """One candidate text placement. x/y/w/h are fractions of the final
    rendered page (x, w relative to width; y, h relative to height, measured
    from the top) -- i.e. of the illustration after its cover-fit crop, not
    of the raw illustration file. `preset` is the shape preset that produced
    this candidate; `None` for candidates not produced by `search_candidates`
    (e.g. hand-built in tests)."""

    x: float
    y: float
    w: float
    h: float
    font_size: float
    badness: float
    variance: float
    brightness: float
    requires_scrim: bool
    preset: ShapePreset | None = None


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


def visible_crop_box(
    image_width: float, image_height: float, page_width: float, page_height: float
) -> tuple[float, float, float, float]:
    """The (left, top, right, bottom) region of an image, in the image's own
    pixel coordinates, that remains visible after a cover-fit scale-and-crop
    onto a page_width x page_height page. The single source of truth for
    that geometry -- shared by `analyze_badness` (so its badness map only
    covers what will actually be shown) and `stage3_assemble._cover_fit_image`
    (so drawing and analysis can never disagree about what's visible)."""
    scale = max(page_width / image_width, page_height / image_height)
    draw_w, draw_h = image_width * scale, image_height * scale
    draw_x, draw_y = (page_width - draw_w) / 2, (page_height - draw_h) / 2
    left, top = -draw_x / scale, -draw_y / scale
    return left, top, left + page_width / scale, top + page_height / scale


def analyze_badness(image: Image.Image, page_width: float, page_height: float) -> BadnessMap:
    """Downsample the cover-fit-visible region of `image` -- the sub-rectangle
    that will actually be shown on a page_width x page_height page -- and
    compute its per-pixel text-safety badness. Candidates derived from the
    result are therefore already in page-fraction coordinates."""
    left, top, right, bottom = visible_crop_box(image.width, image.height, page_width, page_height)
    visible = image.crop((round(left), round(top), round(right), round(bottom)))

    rgb = visible.convert("RGB")
    long_side = max(rgb.size)
    scale = ANALYSIS_LONG_SIDE / long_side
    small = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))))
    arr = np.asarray(small)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    saliency_algo = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = saliency_algo.computeSaliency(arr)
    saliency_map = saliency_map.astype(np.float32) if success else np.zeros_like(gray)
    sal_min, sal_max = saliency_map.min(), saliency_map.max()
    saliency_norm = (
        (saliency_map - sal_min) / (sal_max - sal_min)
        if sal_max > sal_min
        else np.zeros_like(saliency_map)
    )

    edges = cv2.Canny(gray.astype(np.uint8), 100, 200).astype(np.float32) / 255.0

    kernel = 5
    local_mean = cv2.blur(gray, (kernel, kernel))
    local_sq_mean = cv2.blur(gray * gray, (kernel, kernel))
    local_var = np.clip(local_sq_mean - local_mean * local_mean, 0.0, None)
    variance_norm = np.clip(np.sqrt(local_var) / 128.0, 0.0, 1.0)

    w_saliency, w_edge, w_variance = BADNESS_WEIGHTS
    badness = np.clip(
        w_saliency * saliency_norm + w_edge * edges + w_variance * variance_norm, 0.0, 1.0
    )

    return BadnessMap(
        badness=badness.astype(np.float32),
        variance=variance_norm.astype(np.float32),
        brightness=gray.astype(np.float32),
    )


def _candidate_from_position(
    variance_integral: np.ndarray,
    brightness_integral: np.ndarray,
    y0: int,
    x0: int,
    rect_h: int,
    rect_w: int,
    grid_h: int,
    grid_w: int,
    width_fraction: float,
    height_fraction: float,
    font_size: float,
    badness: float,
    requires_scrim: bool,
    preset: ShapePreset,
) -> Candidate:
    area = rect_h * rect_w
    variance = _rect_sum(variance_integral, y0, x0, y0 + rect_h, x0 + rect_w) / area
    brightness = _rect_sum(brightness_integral, y0, x0, y0 + rect_h, x0 + rect_w) / area
    return Candidate(
        x=x0 / grid_w,
        y=y0 / grid_h,
        w=width_fraction,
        h=height_fraction,
        font_size=font_size,
        badness=badness,
        variance=variance,
        brightness=brightness,
        requires_scrim=requires_scrim,
        preset=preset,
    )


def search_candidates(
    badness_map: BadnessMap,
    caption: str,
    font: str,
    language: str,
    page_width: float,
    page_height: float,
) -> list[Candidate]:
    """Search each of the three shape presets independently for its best
    text-safe rectangle, font-fitting the caption to each shape as it goes."""
    badness_integral = _integral_image(badness_map.badness)
    variance_integral = _integral_image(badness_map.variance)
    brightness_integral = _integral_image(badness_map.brightness)
    grid_h, grid_w = badness_map.badness.shape

    candidates: list[Candidate] = []
    for preset, width_fraction in zip(SHAPE_PRESET_NAMES, SHAPE_PRESET_WIDTHS):
        rect_w_pt = width_fraction * page_width
        max_text_width_pt = rect_w_pt - 2 * CANDIDATE_PAD_PT

        font_size = FONT_SIZE_MAX
        last_attempt: tuple[int, int, int, int, float, float] | None = None
        chosen: Candidate | None = None
        while font_size >= FONT_SIZE_MIN:
            line_count = max(
                wrapped_line_count(caption, font, font_size, max_text_width_pt, language), 1
            )
            rect_h_pt = 2 * CANDIDATE_PAD_PT + line_count * font_size * 1.3
            height_fraction = min(rect_h_pt / page_height, MAX_HEIGHT_FRACTION)
            rect_h = min(max(round(height_fraction * grid_h), 1), grid_h)
            rect_w = min(max(round(width_fraction * grid_w), 1), grid_w)

            sums = _rect_sums_for_size(badness_integral, rect_h, rect_w)
            area = rect_h * rect_w
            y0, x0 = (int(i) for i in np.unravel_index(np.argmin(sums), sums.shape))
            best_badness = float(sums[y0, x0] / area)
            last_attempt = (y0, x0, rect_h, rect_w, best_badness, height_fraction)

            if best_badness <= SAFE_THRESHOLD:
                chosen = _candidate_from_position(
                    variance_integral,
                    brightness_integral,
                    y0,
                    x0,
                    rect_h,
                    rect_w,
                    grid_h,
                    grid_w,
                    width_fraction,
                    height_fraction,
                    font_size,
                    best_badness,
                    requires_scrim=False,
                    preset=preset,
                )
                break
            font_size -= FONT_SIZE_STEP

        if chosen is None:
            assert last_attempt is not None
            y0, x0, rect_h, rect_w, best_badness, height_fraction = last_attempt
            chosen = _candidate_from_position(
                variance_integral,
                brightness_integral,
                y0,
                x0,
                rect_h,
                rect_w,
                grid_h,
                grid_w,
                width_fraction,
                height_fraction,
                FONT_SIZE_MIN,
                best_badness,
                requires_scrim=True,
                preset=preset,
            )
        candidates.append(chosen)
    return candidates


def pick_best(candidates: list[Candidate]) -> Candidate:
    """Rank candidates by badness (inverted) + achieved font size, penalizing a
    scrim requirement, and return the highest scorer."""
    if not candidates:
        raise ValueError("pick_best called with no candidates")

    def score(candidate: Candidate) -> float:
        font_ratio = (candidate.font_size - FONT_SIZE_MIN) / (FONT_SIZE_MAX - FONT_SIZE_MIN)
        penalty = RANK_SCRIM_PENALTY if candidate.requires_scrim else 0.0
        return (
            RANK_BADNESS_WEIGHT * (1 - candidate.badness)
            + RANK_FONT_WEIGHT * font_ratio
            - penalty
        )

    return max(candidates, key=score)
