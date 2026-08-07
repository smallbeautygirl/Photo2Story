# Picture-book layout redesign

## Problem

The current `full_bleed_caption` layout (`src/stages/stage3_assemble.py`) scales
the illustration to cover the whole page and overlays white caption text on a
semi-transparent black scrim at the bottom. None of the four 0-2 age reference
board books in `reference/0-2/` (*Brown Bear, Brown Bear, What Do You See?*,
*Dear Zoo*, *The Very Hungry Caterpillar*, *From Head to Toe*) do this: all
four keep the illustration uncropped within whitespace and place text in a
plain margin zone that never overlaps the art. Additionally, per-page layout
is not currently pinned to a consistent zone, so the reading rhythm can vary
page to page.

## Reference observations

Three of the four references (Brown Bear, Dear Zoo, Caterpillar) are PDFs
exported from an online flipbook viewer (FlipHTML5) rather than the original
book files — confirmed by leftover viewer UI chrome (an audio icon) baked
into the Caterpillar pages. `From Head to Toe` is a genuine Adobe Acrobat scan
and is treated as the most reliable source; the other three are corroborating
evidence, not primary.

- **The Very Hungry Caterpillar** and **From Head to Toe** (the two reliable,
  visually-confirmed sources) agree closely: full-scene illustrations fill
  most of the frame, and text sits in a bold, left-aligned block reserved
  *above* the illustration — a fixed zone, same position every page.
- **Dear Zoo**: a single caption sits *below* the illustration, centered
  under the image's full width (not a narrow corner) — e.g. "He was too big!
  I sent him back." under the elephant.
- **Brown Bear**: a different structure still — two short captions flank the
  illustration left and right (question on one side, answer on the other).
  This split-caption structure is a distinct pattern we are not replicating;
  it doesn't correspond to our one-caption-per-page pipeline output.
- In all cases text never overlaps the art — it occupies a zone the
  illustrator reserved, not empty space detected inside the art. Our
  FLUX-generated illustrations are always full-scene renders (style presets:
  ghibli/pixar/disney/crayon), never isolated-object/white-background
  portraits, so we cannot rely on the generated art having a built-in quiet
  corner. We reserve a text zone *outside* the image instead of trying to
  detect empty space inside it.

## Design

### 1. Rename `full_bleed_caption` → `picture_book`

The old name describes behavior (full-bleed + caption overlay) that no longer
applies. Rename the config value everywhere it appears:
`configs/demo.yaml`, the `Literal` type in `src/config.py`,
`docs/superpowers/specs/2026-06-28-storybook-improvements-design.md`. The legacy `image_top_text_bottom` path
(`build_pdf`) is unchanged — it's used by all ablation configs and is out of
scope for this change.

### 2. Shared geometry

Illustrations are square (`square_hd`, see `illustrate_fal.py`). For every
page in `build_picture_book_pdf` (replaces `build_caption_pdf`):

- The image is **contain-fit** into its zone (scaled to fit within both the
  available width and height, never cropped, centered) — replacing
  `_draw_full_bleed_image`'s cover/crop behavior.
- The page is split into a **text zone** and an **image zone**. The text
  zone's height is computed *once per book* — from the longest wrapped
  caption across *all* pages at that zone's width — not recomputed per page.
  Every page therefore reserves an identically-sized band and the image sits
  at an identical position throughout the book. This is what makes the
  layout consistent page to page.
- No scrim. Text is drawn directly on the plain page background, since it
  never overlaps the image.

### 3. Choosing top-band vs. bottom-caption (computed once per book)

Both variants now use the **same full page width** (margins only) — the
bottom variant is a single caption centered under the image's full width
(Dear Zoo-style), not a narrow corner column. Since both use the same width,
the choice can't be driven by narrow-vs-wide wrapping any more; instead we
measure caption length against a fixed yardstick:

Before rendering, wrap every page's caption at full page width **using the
top-band's font size** (~20pt) — this yardstick is fixed regardless of which
variant is ultimately chosen, so it's a pure text-length measurement.

- If any page's caption would need **more than 2 lines** at that yardstick →
  the whole book uses the **top-band** variant: full-width text block
  reserved above the image, left-aligned, ~20pt font. Matches
  Caterpillar/Head to Toe's longer descriptive sentences.
- Otherwise → the whole book uses the **bottom-caption** variant: image
  occupies the top of the page, a single caption centered under the full
  image width in the bottom margin, ~15pt font. Matches Dear Zoo's short
  punchy reactions ("He was too big! I sent him back.").

This is a deterministic, per-book (not per-page) decision, so it never
conflicts with the consistency requirement in §2. It requires no image
analysis — the input is caption text, which we already have before any
rendering happens.

Extract this as a pure function:

```python
def _choose_layout_variant(pages: list[str], font: str) -> Literal["top_band", "bottom_caption"]:
    ...
```

### 4. Text styling

- Top-band: ~20pt, left-aligned, full page width minus margins (matches
  Caterpillar/Head to Toe).
- Bottom-caption: ~15pt, centered, full page width minus margins (matches
  Dear Zoo).
- Regular-weight CJK font only (existing `_resolve_cjk_font`) — no bold
  font search added. The reference books' core pattern is text-in-whitespace,
  not text-weight; adding bold font discovery is unneeded complexity here.

### 5. Testing

`build_caption_pdf` currently has zero test coverage. Add:

- `test_choose_layout_variant_picks_top_band_for_long_captions`
- `test_choose_layout_variant_picks_bottom_caption_for_short_captions`
- `test_build_picture_book_pdf_creates_file` (covering both variants)
- `test_build_picture_book_pdf_wrong_count_raises`

Integration tests assert file creation and non-trivial size (matching the
existing style in `tests/test_stage3_assemble.py`), not pixel-level PDF
geometry — no new parsing dependency needed.

## Out of scope

- The legacy `image_top_text_bottom` layout (`build_pdf`) — untouched.
- Stage 1/2 changes (story generation, illustration prompts/composition).
- Page size/orientation (stays A4 portrait).
- Bold CJK font support.
