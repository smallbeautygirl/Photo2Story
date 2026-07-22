# Picture-book text overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `picture_book`/`picture_book_spread`'s fixed outside-image text zone with captions drawn directly on a full-bleed illustration, positioned in a programmatically detected text-safe region per page.

**Architecture:** A new `src/stages/text_placement.py` module analyzes each illustration into a per-pixel "suitability map" (saliency + edge density + local variance via classical CV, no ML model weights) and searches three candidate rectangle shapes for the best-scoring, font-fitted placement. `src/stages/stage3_assemble.py` calls this module per page, draws the illustration cover-fit (full page), then renders the caption with an escalating contrast strategy (plain → outline → scrim).

**Tech Stack:** Python 3.14, `opencv-contrib-python` (new), `numpy` (existing), `Pillow` (existing), `reportlab` (existing).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-22-picture-book-text-overlay-design.md`
- `image_top_text_bottom` (`build_pdf`) is untouched — do not modify its behavior.
- No `configs/*.yaml` schema changes — `page_layout: picture_book` / `picture_book_spread` keep the same config surface.
- New core dependency: `opencv-contrib-python` (confirmed installable on this repo's Python 3.14 venv via the `cp37-abi3` wheel tag — no GPU, no separate model download).
- Every new/modified `.py` file starts with `from __future__ import annotations`, uses built-in generics (`list[str]`, `X | None`), f-strings only, `pathlib.Path` for paths (not applicable here — no new file-path handling), and no comments explaining *what* code does (only *why*, when non-obvious).
- Formatter/linter: `ruff format` / `ruff check` (`E, F, I, UP, B, SIM, TCH`); type checker `mypy --strict`.
- Test style: one behavior per test, descriptive names, deterministic (seeded `numpy` RNG where randomness is needed), no sleeps.
- Every step that adds/changes code ends with running the relevant tests; every task ends with a commit.

---

### Task 1: Extract text-wrapping helpers into `src/utils/text_wrap.py`

`_wrap_to_width` and `_wrapped_line_count` currently live as private functions in `stage3_assemble.py`. `text_placement.py` (Task 4) needs `_wrapped_line_count`'s logic too, and `stage3_assemble.py` will import `text_placement.py` — so these helpers must move to a shared module both can import without a circular import.

**Files:**
- Create: `src/utils/text_wrap.py`
- Modify: `src/stages/stage3_assemble.py` (remove `_wrap_to_width`/`_wrapped_line_count`, lines 93–124; update the two call sites at the old `_draw_section` and `_draw_zone_text` bodies to use the new imports)
- Test: `tests/test_text_wrap.py`

**Interfaces:**
- Produces: `wrap_to_width(text: str, font: str, size: float, max_width: float) -> list[str]`, `wrapped_line_count(text: str, font: str, font_size: float, max_width: float, language: str) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_text_wrap.py
from __future__ import annotations

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

from src.utils.text_wrap import wrap_to_width, wrapped_line_count


def test_wrap_to_width_breaks_on_spaces_for_latin_text():
    lines = wrap_to_width("one two three four five", "Helvetica", 12, max_width=60)
    assert len(lines) > 1
    assert " ".join(lines) == "one two three four five"


def test_wrap_to_width_breaks_per_character_when_no_spaces():
    lines = wrap_to_width("陶樂蒂的開學日", "Helvetica", 12, max_width=30)
    assert len(lines) > 1
    assert "".join(lines) == "陶樂蒂的開學日"


def test_wrap_to_width_empty_string_returns_empty_list():
    assert wrap_to_width("", "Helvetica", 12, max_width=1000) == []


def test_wrapped_line_count_matches_wrap_to_width_for_en():
    text = "one two three four five six seven"
    assert wrapped_line_count(text, "Helvetica", 12, 60, "en") == len(
        wrap_to_width(text, "Helvetica", 12, 60)
    )


def test_wrapped_line_count_uses_zhuyin_wrapping_for_zh_tw():
    """A Hanzi string that fits in 2 plain-wrapped lines can need more once
    Zhuyin annotation widens each character."""
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    caption = "陶樂蒂的開學日今天真是漂亮的一天大家都很開心呢" + "啊" * 10

    en_count = wrapped_line_count(caption, "STSong-Light", 24, 500, "en")
    zh_count = wrapped_line_count(caption, "STSong-Light", 24, 500, "zh-tw")

    assert zh_count > en_count
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_text_wrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.text_wrap'`

- [ ] **Step 3: Create `src/utils/text_wrap.py`**

```python
"""Text wrapping primitives shared by stage3_assemble.py (PDF drawing) and
text_placement.py (candidate rectangle search) -- both need to know how many
lines a caption wraps to at a given width/font/language without depending on
each other.
"""

from __future__ import annotations

from reportlab.pdfbase import pdfmetrics

from src.stages.zhuyin_render import wrap_zhuyin
from src.utils.zhuyin import annotate


def wrap_to_width(text: str, font: str, size: float, max_width: float) -> list[str]:
    """Wrap text to a pixel width. Breaks on spaces for latin text, per character for CJK."""
    text = text.strip()
    if not text:
        return []
    if " " in text:
        units, joiner = text.split(), " "
    else:
        units, joiner = list(text), ""

    lines: list[str] = []
    current = ""
    for unit in units:
        trial = f"{current}{joiner}{unit}" if current else unit
        if not current or pdfmetrics.stringWidth(trial, font, size) <= max_width:
            current = trial
        else:
            lines.append(current)
            current = unit
    if current:
        lines.append(current)
    return lines


def wrapped_line_count(
    text: str, font: str, font_size: float, max_width: float, language: str
) -> int:
    """Number of lines `text` wraps to, using Zhuyin-aware wrapping for zh-tw
    (each character is wider once annotated) and plain wrapping otherwise."""
    if language == "zh-tw":
        return len(wrap_zhuyin(annotate(text), font, font_size, max_width))
    return len(wrap_to_width(text, font, font_size, max_width))
```

- [ ] **Step 4: Update `stage3_assemble.py` to import from the new module**

Remove the `_wrap_to_width` and `_wrapped_line_count` function definitions (current lines 93–124). Add to the imports at the top of the file:

```python
from src.utils.text_wrap import wrap_to_width, wrapped_line_count
```

Replace every call to `_wrap_to_width(...)` with `wrap_to_width(...)` and every call to `_wrapped_line_count(...)` with `wrapped_line_count(...)` in the rest of the file (`_draw_section`, `_draw_zone_text`, `build_pdf`).

- [ ] **Step 5: Run the new tests and the full existing suite**

Run: `.venv/bin/pytest tests/test_text_wrap.py tests/test_stage3_assemble.py -v`
Expected: all PASS (the existing `test_stage3_assemble.py` tests still reference `_choose_layout_variant`/`_text_zone_height`, which are untouched by this task, so they still pass)

- [ ] **Step 6: Commit**

```bash
git add src/utils/text_wrap.py src/stages/stage3_assemble.py tests/test_text_wrap.py
git commit -m "refactor(pipeline): ♻️ extract text wrapping helpers to shared module"
```

---

### Task 2: Add `opencv-contrib-python` dependency and `text_placement.py` scaffolding

**Files:**
- Modify: `requirements.txt`
- Create: `src/stages/text_placement.py`
- Test: `tests/test_text_placement.py`

**Interfaces:**
- Produces: `SuitabilityMap` (dataclass: `badness`, `variance`, `brightness` — each `np.ndarray` of identical shape), `Candidate` (dataclass: `x`, `y`, `w`, `h`, `font_size`, `badness`, `variance`, `brightness`, `requires_scrim`), constants `FONT_SIZE_MIN`, `FONT_SIZE_MAX`, `FONT_SIZE_STEP`, `SAFE_THRESHOLD`, `SCRIM_OPACITY`, `CONTRAST_SAFE_VARIANCE`, `SHAPE_PRESET_WIDTHS`, `BADNESS_WEIGHTS`, `RANK_SUITABILITY_WEIGHT`, `RANK_FONT_WEIGHT`, `RANK_SCRIM_PENALTY`, `ANALYSIS_LONG_SIDE`, `MAX_HEIGHT_FRACTION`, `CANDIDATE_PAD_PT`, and internal `_integral_image`, `_rect_sum`, `_rect_sums_for_size`.

- [ ] **Step 1: Add the dependency**

In `requirements.txt`, add a new line after `fal-client>=1.0.0`:

```
opencv-contrib-python>=4.10.0
```

Install it:

Run: `.venv/bin/pip install opencv-contrib-python>=4.10.0`
Expected: installs successfully (verified in this repo's Python 3.14 venv — resolves to a `cp37-abi3` wheel, no build-from-source)

- [ ] **Step 2: Write the failing tests for the integral-image helpers**

```python
# tests/test_text_placement.py
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_text_placement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.stages.text_placement'`

- [ ] **Step 4: Create `src/stages/text_placement.py` (constants, dataclasses, integral-image helpers)**

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_text_placement.py -v`
Expected: both tests PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/stages/text_placement.py tests/test_text_placement.py
git commit -m "feat(pipeline): ✨ add text_placement module scaffolding and integral-image search"
```

---

### Task 3: Implement `analyze_suitability`

**Files:**
- Modify: `src/stages/text_placement.py`
- Test: `tests/test_text_placement.py`

**Interfaces:**
- Consumes: `BADNESS_WEIGHTS`, `ANALYSIS_LONG_SIDE` (Task 2)
- Produces: `analyze_suitability(image: Image.Image) -> SuitabilityMap`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_text_placement.py -k analyze_suitability -v`
Expected: FAIL with `ImportError: cannot import name 'analyze_suitability'`

- [ ] **Step 3: Implement `analyze_suitability`**

Append to `src/stages/text_placement.py`:

```python
def analyze_suitability(image: Image.Image) -> SuitabilityMap:
    """Downsample `image` and compute its per-pixel text-safety badness."""
    rgb = image.convert("RGB")
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

    return SuitabilityMap(
        badness=badness.astype(np.float32),
        variance=variance_norm.astype(np.float32),
        brightness=gray.astype(np.float32),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_text_placement.py -k analyze_suitability -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/stages/text_placement.py tests/test_text_placement.py
git commit -m "feat(pipeline): ✨ implement suitability-map analysis via saliency/edges/variance"
```

---

### Task 4: Implement `search_candidates`

**Files:**
- Modify: `src/stages/text_placement.py`
- Test: `tests/test_text_placement.py`

**Interfaces:**
- Consumes: `SuitabilityMap`, `Candidate`, `_integral_image`, `_rect_sum`, `_rect_sums_for_size` (Task 2), `wrapped_line_count` (Task 1)
- Produces: `search_candidates(suitability: SuitabilityMap, caption: str, font: str, language: str, page_width: float, page_height: float) -> list[Candidate]`

- [ ] **Step 1: Write the failing tests**

```python
def test_search_candidates_returns_one_per_shape_preset():
    from src.stages.text_placement import SHAPE_PRESET_WIDTHS, analyze_suitability, search_candidates

    image = Image.fromarray(np.full((200, 200, 3), 230, dtype=np.uint8), mode="RGB")
    suitability = analyze_suitability(image)

    candidates = search_candidates(
        suitability, "A short caption.", "Helvetica", "en", 400.0, 400.0
    )

    assert len(candidates) == len(SHAPE_PRESET_WIDTHS)
    assert [round(c.w, 2) for c in candidates] == [round(w, 2) for w in SHAPE_PRESET_WIDTHS]


def test_search_candidates_narrower_preset_gets_smaller_or_equal_font_for_long_caption():
    """A long caption needs more lines at a narrower width, growing the
    candidate rectangle's height until it hits MAX_HEIGHT_FRACTION and is
    forced to a smaller font -- a narrower preset must never end up with a
    *larger* font than a wider preset for the same caption."""
    from src.stages.text_placement import analyze_suitability, search_candidates

    image = Image.fromarray(np.full((200, 200, 3), 230, dtype=np.uint8), mode="RGB")
    suitability = analyze_suitability(image)
    long_caption = " ".join(["word"] * 40)

    candidates = search_candidates(suitability, long_caption, "Helvetica", "en", 400.0, 400.0)
    by_width = sorted(candidates, key=lambda c: c.w)

    assert by_width[0].font_size <= by_width[-1].font_size


def test_search_candidates_flags_requires_scrim_when_nothing_clears_threshold():
    from src.stages.text_placement import analyze_suitability, search_candidates

    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(200, 200, 3), dtype=np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    suitability = analyze_suitability(image)

    candidates = search_candidates(
        suitability, "A caption that needs placement.", "Helvetica", "en", 400.0, 400.0
    )

    assert all(c.requires_scrim for c in candidates)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_text_placement.py -k search_candidates -v`
Expected: FAIL with `ImportError: cannot import name 'search_candidates'`

- [ ] **Step 3: Implement `search_candidates`**

Append to `src/stages/text_placement.py`:

```python
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
    )


def search_candidates(
    suitability: SuitabilityMap,
    caption: str,
    font: str,
    language: str,
    page_width: float,
    page_height: float,
) -> list[Candidate]:
    """Search each of the three shape presets independently for its best
    text-safe rectangle, font-fitting the caption to each shape as it goes."""
    badness_integral = _integral_image(suitability.badness)
    variance_integral = _integral_image(suitability.variance)
    brightness_integral = _integral_image(suitability.brightness)
    grid_h, grid_w = suitability.badness.shape

    candidates: list[Candidate] = []
    for width_fraction in SHAPE_PRESET_WIDTHS:
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
            )
        candidates.append(chosen)
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_text_placement.py -k search_candidates -v`
Expected: all PASS. If `test_search_candidates_narrower_preset_gets_smaller_or_equal_font_for_long_caption`
fails because 40 words isn't long enough to force a size difference at 400pt page width, increase the
repeat count (e.g. to 60 or 80) until the narrow preset (`0.28`) is measurably forced below `FONT_SIZE_MAX`
while the wide preset (`0.65`) is not — confirm by printing `[(c.w, c.font_size) for c in candidates]`.

- [ ] **Step 5: Commit**

```bash
git add src/stages/text_placement.py tests/test_text_placement.py
git commit -m "feat(pipeline): ✨ implement candidate rectangle search with font fitting"
```

---

### Task 5: Implement `pick_best`

**Files:**
- Modify: `src/stages/text_placement.py`
- Test: `tests/test_text_placement.py`

**Interfaces:**
- Consumes: `Candidate`, `RANK_SUITABILITY_WEIGHT`, `RANK_FONT_WEIGHT`, `RANK_SCRIM_PENALTY`, `FONT_SIZE_MIN`, `FONT_SIZE_MAX` (Task 2)
- Produces: `pick_best(candidates: list[Candidate]) -> Candidate`

- [ ] **Step 1: Write the failing test**

```python
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


def test_pick_best_raises_on_empty_list():
    from src.stages.text_placement import pick_best

    with pytest.raises(ValueError, match="no candidates"):
        pick_best([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_text_placement.py -k pick_best -v`
Expected: FAIL with `ImportError: cannot import name 'pick_best'`

- [ ] **Step 3: Implement `pick_best`**

Append to `src/stages/text_placement.py`:

```python
def pick_best(candidates: list[Candidate]) -> Candidate:
    """Rank candidates by suitability + achieved font size, penalizing a
    scrim requirement, and return the highest scorer."""
    if not candidates:
        raise ValueError("pick_best called with no candidates")

    def score(candidate: Candidate) -> float:
        font_ratio = (candidate.font_size - FONT_SIZE_MIN) / (FONT_SIZE_MAX - FONT_SIZE_MIN)
        penalty = RANK_SCRIM_PENALTY if candidate.requires_scrim else 0.0
        return (
            RANK_SUITABILITY_WEIGHT * (1 - candidate.badness)
            + RANK_FONT_WEIGHT * font_ratio
            - penalty
        )

    return max(candidates, key=score)
```

Add `import pytest` to the top of `tests/test_text_placement.py` if not already present (needed for `pytest.raises`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_text_placement.py -v`
Expected: all tests in the file PASS

- [ ] **Step 5: Commit**

```bash
git add src/stages/text_placement.py tests/test_text_placement.py
git commit -m "feat(pipeline): ✨ implement candidate ranking in pick_best"
```

---

### Task 6: Add `fill_color` parameter to `zhuyin_render.draw_zhuyin_line`

**Files:**
- Modify: `src/stages/zhuyin_render.py:63-119`
- Test: `tests/test_zhuyin_render.py`

**Interfaces:**
- Produces: `draw_zhuyin_line(c, line, font, font_size, x, y, w, align, fill_color: Color = colors.black) -> None` (new optional param, backward compatible)

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_zhuyin_render.py
def test_draw_zhuyin_line_uses_given_fill_color():
    from reportlab.lib import colors
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    calls = []

    class _RecordingCanvas:
        def __init__(self, real):
            self._real = real

        def setFillColor(self, *a, **kw):
            calls.append(("setFillColor", a, kw))
            self._real.setFillColor(*a, **kw)

        def setFont(self, *a, **kw):
            self._real.setFont(*a, **kw)

        def drawString(self, x, y, text):
            self._real.drawString(x, y, text)

    from reportlab.pdfgen import canvas as canvas_module

    real_canvas = canvas_module.Canvas("/dev/null")
    wrapped = _RecordingCanvas(real_canvas)
    zchars = annotate("媽")

    draw_zhuyin_line(
        wrapped, zchars, "STSong-Light", 24, x=50, y=700, w=400, align="left",
        fill_color=colors.white,
    )

    assert ("setFillColor", (colors.white,), {}) in calls
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_zhuyin_render.py -k fill_color -v`
Expected: FAIL — `colors.white` is never passed to `setFillColor` (the current code always calls `setFillColor(colors.black)`)

- [ ] **Step 3: Add the parameter**

In `src/stages/zhuyin_render.py`, change the `draw_zhuyin_line` signature (currently line 63-72):

```python
def draw_zhuyin_line(
    c: canvas.Canvas,
    line: list[ZhuyinChar],
    font: str,
    font_size: float,
    x: float,
    y: float,
    w: float,
    align: Literal["left", "center"],
    fill_color: colors.Color = colors.black,
) -> None:
```

And change line 85 from `c.setFillColor(colors.black)` to `c.setFillColor(fill_color)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_zhuyin_render.py -v`
Expected: all tests PASS (including the pre-existing ones, which don't pass `fill_color` and get the `colors.black` default — behavior-preserving)

- [ ] **Step 5: Commit**

```bash
git add src/stages/zhuyin_render.py tests/test_zhuyin_render.py
git commit -m "feat(pipeline): ✨ add fill_color param to draw_zhuyin_line"
```

---

### Task 7: Add overlay rendering to `stage3_assemble.py`

**Files:**
- Modify: `src/stages/stage3_assemble.py`
- Test: `tests/test_stage3_assemble.py`

**Interfaces:**
- Consumes: `Candidate`, `CONTRAST_SAFE_VARIANCE`, `SCRIM_OPACITY` (Task 2), `draw_zhuyin_line(..., fill_color=...)` (Task 6)
- Produces: `_cover_fit_image(c, image, page_w, page_h) -> None` (takes an already-opened `Image.Image`, not a path — Task 8's caller loads the file once and reuses it for both suitability analysis and drawing), `_draw_zone_text(..., fill_color=colors.black, offset=(0.0, 0.0))` (extends existing signature), `_draw_caption_overlay(c, candidate, page_text, font, page_w, page_h, language="en") -> None`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_stage3_assemble.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_stage3_assemble.py -k "cover_fit or draw_caption_overlay" -v`
Expected: FAIL with `ImportError: cannot import name '_cover_fit_image'` (and similarly for `_draw_caption_overlay`)

- [ ] **Step 3: Implement the drawing functions**

In `src/stages/stage3_assemble.py`, add to the imports:

```python
from src.stages.text_placement import CONTRAST_SAFE_VARIANCE, SCRIM_OPACITY, Candidate
```

Replace `_contain_fit_image` (current lines 146-156) with:

```python
def _cover_fit_image(
    c: canvas.Canvas, image: Image.Image, page_w: float, page_h: float
) -> None:
    """Scale the already-opened illustration to fill the whole page,
    cropping any excess. Takes a loaded image rather than a path so the
    caller can reuse the same load for suitability analysis."""
    img_w, img_h = image.size
    scale = max(page_w / img_w, page_h / img_h)
    draw_w, draw_h = img_w * scale, img_h * scale
    draw_x = (page_w - draw_w) / 2
    draw_y = (page_h - draw_h) / 2
    c.drawImage(ImageReader(image), draw_x, draw_y, width=draw_w, height=draw_h)
```

Change `_draw_zone_text`'s signature (current lines 176-186) to:

```python
def _draw_zone_text(
    c: canvas.Canvas,
    text: str,
    font: str,
    font_size: float,
    x: float,
    top_y: float,
    w: float,
    align: Literal["left", "center"],
    language: str = "en",
    fill_color: colors.Color = colors.black,
    offset: tuple[float, float] = (0.0, 0.0),
) -> None:
```

And its body (current lines 192-210) to:

```python
    line_height = font_size * 1.3
    dx, dy = offset
    cursor_y = top_y - font_size + dy
    x = x + dx

    if language == "zh-tw":
        zhuyin_lines = wrap_zhuyin(annotate(text), font, font_size, w) or [[]]
        for zhuyin_line in zhuyin_lines:
            draw_zhuyin_line(c, zhuyin_line, font, font_size, x, cursor_y, w, align, fill_color=fill_color)
            cursor_y -= line_height
        return

    lines = wrap_to_width(text, font, font_size, w) or [""]
    c.setFillColor(fill_color)
    c.setFont(font, font_size)
    for line in lines:
        if align == "center":
            c.drawCentredString(x + w / 2, cursor_y, line)
        else:
            c.drawString(x, cursor_y, line)
        cursor_y -= line_height
```

Add a new function, after `_draw_zone_text`:

```python
def _draw_caption_overlay(
    c: canvas.Canvas,
    candidate: Candidate,
    page_text: str,
    font: str,
    page_w: float,
    page_h: float,
    language: str = "en",
) -> None:
    """Draw `page_text` inside `candidate`'s rectangle, directly on the
    illustration, escalating from plain text to an outline to a translucent
    scrim only as far as needed for contrast against the artwork."""
    x = candidate.x * page_w
    w = candidate.w * page_w
    top_y = page_h - candidate.y * page_h
    h = candidate.h * page_h

    ink = colors.black if candidate.brightness > 128 else colors.white
    backdrop = colors.white if ink == colors.black else colors.black

    if candidate.variance <= CONTRAST_SAFE_VARIANCE:
        _draw_zone_text(c, page_text, font, candidate.font_size, x, top_y, w, "left", language, fill_color=ink)
    elif not candidate.requires_scrim:
        _draw_zone_text(
            c, page_text, font, candidate.font_size, x, top_y, w, "left", language,
            fill_color=backdrop, offset=(0.6, -0.6),
        )
        _draw_zone_text(c, page_text, font, candidate.font_size, x, top_y, w, "left", language, fill_color=ink)
    else:
        c.saveState()
        c.setFillColor(backdrop)
        c.setFillAlpha(SCRIM_OPACITY)
        c.rect(x, top_y - h, w, h, fill=1, stroke=0)
        c.restoreState()
        _draw_zone_text(c, page_text, font, candidate.font_size, x, top_y, w, "left", language, fill_color=ink)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_stage3_assemble.py -k "cover_fit or draw_caption_overlay" -v`
Expected: all 4 new tests PASS

- [ ] **Step 5: Run the full existing test file to confirm nothing else broke**

Run: `.venv/bin/pytest tests/test_stage3_assemble.py tests/test_zhuyin_render.py -v`
Expected: all PASS (nothing in this task touches `build_picture_book_pdf`/`_choose_layout_variant` yet — that's Task 8)

- [ ] **Step 6: Commit**

```bash
git add src/stages/stage3_assemble.py tests/test_stage3_assemble.py
git commit -m "feat(pipeline): ✨ add cover-fit image drawing and caption overlay rendering"
```

---

### Task 8: Wire `build_picture_book_pdf` to the new pipeline and remove the old zone layout

**Files:**
- Modify: `src/stages/stage3_assemble.py`
- Modify: `tests/test_stage3_assemble.py`

**Interfaces:**
- Consumes: `analyze_suitability`, `search_candidates`, `pick_best` (Tasks 3-5), `_cover_fit_image`, `_draw_caption_overlay` (Task 7)

- [ ] **Step 1: Remove the obsolete layout-selection code**

In `src/stages/stage3_assemble.py`, delete:
- The constants block (current lines 33-38): `TOP_BAND_FONT_SIZE`, `BOTTOM_CAPTION_FONT_SIZE`, `MAX_LINES_FOR_BOTTOM`, `TEXT_ZONE_PAD`, `INTER_ZONE_GAP`
- `_choose_layout_variant` (current lines 127-143)
- `_text_zone_height` (current lines 159-173)
- `_draw_picture_book_page` (current lines 213-249)

Add the new imports (combine with the Task 7 import line):

```python
from src.stages.text_placement import (
    CONTRAST_SAFE_VARIANCE,
    SCRIM_OPACITY,
    Candidate,
    analyze_suitability,
    pick_best,
    search_candidates,
)
```

- [ ] **Step 2: Rewrite `build_picture_book_pdf`**

Replace the current `build_picture_book_pdf` body (current lines 252-282) with:

```python
def build_picture_book_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
    page_size: tuple[float, float] = A4,
    language: str = "en",
) -> None:
    """Build a picture-book PDF: each page is a full-bleed illustration with
    its caption drawn directly on top, in a programmatically detected
    text-safe region (see text_placement.py) rather than a reserved zone
    outside the art.
    """
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    font = _resolve_cjk_font(font_path)
    page_w, page_h = page_size

    c = canvas.Canvas(output_path, pagesize=page_size)
    for img_path, page_text in zip(illustration_paths, pages):
        image = Image.open(img_path).convert("RGB")
        suitability = analyze_suitability(image)
        candidates = search_candidates(suitability, page_text, font, language, page_w, page_h)
        best = pick_best(candidates)

        _cover_fit_image(c, image, page_w, page_h)
        _draw_caption_overlay(c, best, page_text, font, page_w, page_h, language)
        c.showPage()
    c.save()
```

`build_spread_pdf` and `run_stage3` are unchanged — `build_spread_pdf` already just calls `build_picture_book_pdf` with a doubled page width.

- [ ] **Step 3: Remove/update the now-obsolete tests in `tests/test_stage3_assemble.py`**

Delete these tests entirely (they test functions removed in Step 1):
- `test_choose_layout_variant_picks_top_band_for_long_captions`
- `test_choose_layout_variant_picks_bottom_caption_for_short_captions`
- `test_choose_layout_variant_uses_spread_width`
- `test_text_zone_height_scales_with_longest_caption`
- `test_choose_layout_variant_uses_zhuyin_wrapping_for_zh_tw`
- `test_text_zone_height_larger_for_zhuyin_than_plain_at_same_width`

Rename (assertions unchanged, since full-bleed + overlay still produces a
non-trivial PDF file — only the misleading "bottom_caption"/"top_band" names
go away):
- `test_build_picture_book_pdf_creates_file_bottom_caption` → `test_build_picture_book_pdf_creates_file_short_caption`
- `test_build_picture_book_pdf_creates_file_top_band` → `test_build_picture_book_pdf_creates_file_long_caption`

- [ ] **Step 4: Run the full test suite**

Run: `.venv/bin/pytest tests/ -v`
Expected: all PASS. In particular, confirm these previously-passing integration
tests still pass unchanged: `test_build_picture_book_pdf_wrong_count_raises`,
`test_build_spread_pdf_creates_file`, `test_build_spread_pdf_zh_tw_creates_file`,
`test_build_spread_pdf_wrong_count_raises`, `test_build_spread_pdf_uses_double_width_page`,
`test_build_picture_book_pdf_zh_tw_creates_file`, `test_run_stage3_threads_language_from_stage1_result`,
`test_run_stage3_defaults_language_to_en_when_absent`, `test_build_picture_book_pdf_default_language_unaffected`.

- [ ] **Step 5: Commit**

```bash
git add src/stages/stage3_assemble.py tests/test_stage3_assemble.py
git commit -m "feat(pipeline): ✨ replace fixed text zone with detected overlay placement"
```
