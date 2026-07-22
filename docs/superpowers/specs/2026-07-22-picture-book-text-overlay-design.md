# Picture-book text overlay: text-safe region placement

## Problem

The current `picture_book` / `picture_book_spread` layouts
(`src/stages/stage3_assemble.py`, see
[2026-07-01-picture-book-layout-design.md](2026-07-01-picture-book-layout-design.md))
reserve a text zone *outside* the illustration — a fixed top-band or
bottom-caption band, same position every page — because the FLUX-generated
art has no guaranteed "quiet corner" to place text on.

The new `reference/3-6/` picture books (我學會等待, 正能量企鵝繪本夏日的朋友,
陶樂蒂的開學日 — aimed at an older age band than the `reference/0-2/` board
books the current design was based on) all do the opposite: text is drawn
directly on the illustration, positioned wherever that specific page's
background happens to be visually simple — sky, snow, whitespace — never a
fixed zone, and not always at the bottom. This spec redesigns
`picture_book`/`picture_book_spread` to do the same for our AI-generated
illustrations, by programmatically detecting a text-safe region per page
rather than relying on an illustrator having left one.

## Scope

- **Replaces** `picture_book` and `picture_book_spread` entirely. Both use
  the same per-image pipeline — `picture_book_spread` already generates one
  wide illustration per spread (`SPREAD_IMAGE_SIZE` in `illustrate_fal.py`),
  so no separate spread-specific logic is needed.
- **Unchanged**: `image_top_text_bottom` (`build_pdf`, the legacy layout used
  by ablation configs).
- **No config schema change.** `page_layout: picture_book` /
  `picture_book_spread` in `configs/*.yaml` keep working — only what happens
  inside them changes.
- The illustration is now drawn full-bleed, cover-fit (scaled and cropped to
  fill the whole page) instead of contain-fit into a sub-zone, since there is
  no outside zone left to reserve.

## Architecture

```
Illustration
     │
     ▼
Image Analysis  →  suitability map (saliency + edges + variance + brightness,
                    structured as independent channels so a future
                    foreground mask can be added as one more channel later)
     │
     ▼
Layout Optimization
     ├─ rectangle search over the suitability map → 3 shape candidates
     ├─ font-fit each candidate (largest size that fits, reject below min)
     └─ pick candidate maximizing (suitability score + font-size score)
     │
     ▼
Text Rendering
     ├─ sample brightness in the chosen rectangle → black or white ink
     ├─ contrast check → plain text, else outline/shadow, else scrim
```

New module `src/stages/text_placement.py` holds the analysis and
optimization phases, kept separate from `stage3_assemble.py`'s PDF-drawing
concerns:

```python
def analyze_suitability(image: Image.Image) -> SuitabilityMap: ...
def search_candidates(
    suitability: SuitabilityMap, caption: str, font: str, language: str
) -> list[Candidate]: ...
def pick_best(candidates: list[Candidate]) -> Candidate: ...
```

`stage3_assemble.py` calls these three functions per page, then draws the
image (cover-fit) and the caption at the position/size/color `pick_best`
returns — it does not know how the region was chosen, matching the existing
separation where `zhuyin_render.py` doesn't know overlay rendering exists.

## Image analysis: the suitability map

The illustration is downsampled to a small analysis grid (long side ~200px —
precision beyond a few cells doesn't matter for region search, and it keeps
every `cv2` op fast regardless of the real 1408×992+ resolution). Per grid
cell, three channels feed a badness score:

| Channel | Source | Meaning |
|---|---|---|
| saliency | `cv2.saliency.StaticSaliencySpectralResidual`, avg per cell | is this part of the *subject*? |
| edge density | Canny edge fraction per cell | how much detail/outline is here? |
| variance | std of grayscale intensity per cell, normalized | texture/brightness fluctuation |

```
badness(cell) = 0.5 * saliency + 0.3 * edge_density + 0.2 * variance_norm
```

Saliency is weighted highest: a visually uniform but salient object (e.g. a
character's plain-colored clothing) should still score badly, which texture
alone would miss. Mean brightness per cell is tracked separately — it is
*not* part of badness, and is only consulted later to choose ink color.

## Candidate rectangle search

Three fixed shape presets, matching the reference books' paragraph-block vs.
short-caption shapes:

| Preset | Width (fraction of page width) |
|---|---|
| narrow-tall | 0.28 |
| medium | 0.45 |
| wide-short | 0.65 |

For each preset, independently:

1. Try font sizes descending from `FONT_SIZE_MAX` (26pt) to `FONT_SIZE_MIN`
   (14pt), step 2pt.
2. At each size, wrap that page's caption to the preset's width (existing
   `_wrap_to_width` / `wrap_zhuyin`) → line count → rectangle height.
3. Slide that exact rectangle over the whole page via an integral image of
   the badness grid (O(1) badness-sum per position) → best position and its
   average badness.
4. Take the **largest** font size whose best-position badness is
   `<= SAFE_THRESHOLD` (0.35). If no size clears the threshold even at
   `FONT_SIZE_MIN`, keep the `FONT_SIZE_MIN` result anyway, flagged
   `requires_scrim=True`.

This produces exactly 3 `Candidate` objects per page
(`x, y, w, h, font_size, badness, requires_scrim`), each already fitted with
its own best achievable font size.

## Ranking

```
combined = 0.6 * (1 - badness)
         + 0.4 * (font_size - FONT_SIZE_MIN) / (FONT_SIZE_MAX - FONT_SIZE_MIN)
         - (0.25 if requires_scrim else 0)
```

`pick_best` returns the candidate with the highest `combined` score.
Suitability dominates the score; font size breaks ties toward the
better-reading shape; a scrim requirement is a real but not disqualifying
penalty (a candidate that needs a scrim can still win if its suitability and
font size are clearly better than the alternatives).

## Text rendering

1. Sample mean brightness in the winning rectangle. Ink color is black if
   brightness > 128, white otherwise.
2. If that rectangle's variance channel is low → draw plain text.
3. Else if `requires_scrim` is false → draw a thin outline first (a stroke
   pass: the same text drawn at a small offset in the opposite color, then
   the normal fill pass on top) for extra contrast.
4. Else (`requires_scrim` is true) → draw a translucent panel (opposite-of-ink
   color, ~55% opacity) covering the rectangle, then plain text on top.

A scrim's opacity makes contrast success guaranteed at that point, so there
is no further fallback step (e.g. "try the next candidate") after the scrim
— it would be unreachable code.

## Integration with existing code

- **`zhuyin_render.py`**: `draw_zhuyin_line` currently hardcodes
  `c.setFillColor(colors.black)` (line 85). Add a `fill_color` parameter
  instead. The outline step above is implemented purely by the caller
  invoking `draw_zhuyin_line` (or the plain-text draw path) twice — once at
  an offset in the outline color, once at the true position in the ink
  color — `zhuyin_render.py` itself gains no outline-specific branch and
  stays unaware that overlay rendering exists.
- **`stage3_assemble.py`**: `_draw_zone_text`'s hardcoded
  `c.setFillColor(colors.black)` gets the same `fill_color` parameter.
  `_choose_layout_variant`, `_text_zone_height`, and the now-unused
  constants `TOP_BAND_FONT_SIZE`, `BOTTOM_CAPTION_FONT_SIZE`,
  `MAX_LINES_FOR_BOTTOM`, `TEXT_ZONE_PAD`, `INTER_ZONE_GAP` are removed —
  they encoded the one-fixed-zone-per-book model this replaces.
  `_contain_fit_image` is replaced by a cover-fit equivalent for the
  full-bleed image. `_wrap_to_width` and `wrap_zhuyin` are unchanged and
  reused by the candidate search.
- **New constants**, in `src/stages/text_placement.py`:
  `FONT_SIZE_MIN = 14`, `FONT_SIZE_MAX = 26`, `FONT_SIZE_STEP = 2`,
  `SAFE_THRESHOLD = 0.35`, `SCRIM_OPACITY = 0.55`,
  `SHAPE_PRESET_WIDTHS = (0.28, 0.45, 0.65)`, the badness weights
  `(0.5, 0.3, 0.2)`, the ranking weights `(0.6, 0.4, 0.25)`, and the
  analysis-grid target size (~200px long side).
- **Dependency**: add `opencv-contrib-python` to `requirements.txt` (core
  dependency, CPU-only, no GPU) for `cv2.saliency` and `cv2.Canny`.

## Testing

Pure image-processing logic, fully offline and deterministic — no external
APIs, no GPU, so none of the existing mocking patterns are needed here:

- `test_analyze_suitability_scores_flat_region_low_and_textured_region_high`
  — synthetic image with a plain block and a noisy/checkerboard block;
  assert the plain block's badness is lower.
- `test_search_candidates_prefers_larger_font_when_suitability_ties` — two
  equally-clean regions of different size; assert the returned candidate
  list favors the larger one's font size.
- `test_search_candidates_flags_requires_scrim_when_nothing_clears_threshold`
  — an entirely noisy synthetic image; assert every candidate comes back
  flagged `requires_scrim=True`.
- `test_pick_best_selects_highest_combined_score` — hand-constructed
  `Candidate` objects (no image needed); assert the ranking formula picks
  the expected one.
- `test_render_cascade_uses_outline_for_high_variance_non_scrim_candidate`
  and `test_render_cascade_uses_scrim_for_requires_scrim_candidate` — assert
  the correct drawing path is taken, matching this repo's existing
  integration-test style (asserting behavior/output, not pixel-level PDF
  parsing).
- Update `tests/test_stage3_assemble.py`: remove/replace the existing
  `_choose_layout_variant` and `_text_zone_height` tests, since the
  functions they cover are removed; add coverage for the new full-bleed +
  overlay `build_picture_book_pdf` behavior (file creation, non-trivial
  size — same style as today).

## Out of scope

- Semantic foreground segmentation (SAM2/GroundingDINO or a hosted
  equivalent) — the suitability-map channel list is structured to accept it
  later as one more channel; nothing is built now.
- `image_top_text_bottom` (legacy `build_pdf`) — untouched.
- Page size/orientation — stays A4 portrait / doubled-width spread.
- Any `configs/*.yaml` schema change.
