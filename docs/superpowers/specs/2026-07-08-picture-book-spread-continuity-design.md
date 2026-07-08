# Picture-book spread layout & narrative-continuity evaluation

## Problem

The 2026-07-01 `picture_book` layout design was derived entirely from 0-2 age
board books (`reference/0-2/`) and produces one square illustration per
single page. It has no reference basis for the 3-6 age band, and the
pipeline has no way to measure whether page-to-page narrative continuity
(the RQ3 causal-inference hypothesis in `docs/next_deliverables.md`) is
actually working beyond subjective A/B comparison.

## Reference observations (age 3-6)

Three books were captured as representative spreads (screenshots of pages
already legitimately viewed, not full-book reproductions) in
`reference/3-6/`: 陶樂蒂的開學日 (4 spreads), 正能量企鵝繪本夏日的朋友
(6 spreads), 我學會等待 (5 spreads). Each image is one printed **spread**
(two physical pages), confirmed by the page counters baked into the scans
(e.g. "5-6 / 37").

Unlike the four 0-2 board books, which converged on a single pattern (fixed
text zone that never overlaps the art), the 3-6 books show **three distinct
archetypes**:

- **陶樂蒂的開學日** — full-bleed illustration spans both pages
  continuously; the caption floats in whichever open pocket the composition
  leaves, and that position moves page to page (embedded in the crowd,
  top-left, top-right).
- **正能量企鵝繪本夏日的朋友** — a spot illustration on a wash background
  with generous white space; text sits in a fixed corner zone, close to the
  0-2 pattern despite the older age band.
- **我學會等待** — full-bleed spread again, but the caption is split into
  two mirrored blocks (top-left / top-right), one per physical page, each
  paired to that half's local action — a "split caption" structure, but for
  narrative continuity rather than the 0-2 Brown Bear-style Q&A split.

Given this divergence, the design does not try to replicate all three
archetypes faithfully. It adopts the constraints established by the
2026-07-01 spec (no image-content analysis; deterministic, per-book choices
driven by caption length) and picks the single archetype from each axis
(page unit, caption structure, layout rule) that is both consistent with
that philosophy and the smallest change to the existing pipeline. See
"Decisions" below for the specific choices and rationale.

## Decisions

1. **Page unit becomes a spread.** `stage0_select` still picks *k* photos;
   `stage1_story` still emits *k* captions (1:1 with photos, unchanged
   count/mapping); `stage2_illustrate` now renders each as a **landscape
   spread image** (~1.414:1, matching two A4 portrait pages side by side)
   instead of a square; `stage3_assemble` renders each as a double-page
   spread instead of a single page. The PDF becomes `2k` physical pages /
   *k* spreads.
2. **One caption per spread** (not the mirrored dual-caption structure of
   我學會等待). Matches the existing 1-caption-per-narrative-beat shape of
   `stage1_story.generate_story` — no change to its output contract.
3. **No opportunistic whitespace-detection caption placement** (陶樂蒂's
   style). Reuses the existing `_choose_layout_variant` top-band/
   bottom-caption rule from the 2026-07-01 spec, re-measured at spread
   width instead of page width — consistent with that spec's explicit
   choice to reserve a text zone outside the image rather than detect
   empty space inside it.
4. **Cross-spread visual consistency is a known gap, not solved
   pixel-perfectly.** Each spread's illustration is still an independent
   FLUX call. A short character/setting description — built once per book
   by concatenating and deduplicating the existing per-photo descriptions
   already produced by `stage0_select` (no new LLM call) — is injected into
   every spread's illustration prompt so descriptive/style continuity is
   maintained at the prompt level. Pixel-identical consistency (seed or
   reference-image conditioning) is out of scope.
5. **Cross-page relevance is evaluated within AND between spreads**:
   within-spread = does the caption match its own illustration; between-
   spread = does spread N narratively continue spread N-1. This is the
   thesis contribution — see "Evaluation" below.

## Architecture

### stage2_illustrate / illustrate_fal.py

- Aspect ratio changes from `square_hd` to a landscape ratio matching an A4
  spread (~1.414:1 width:height).
- Prompt construction gains a fixed character/setting description string,
  computed once per book, appended to every spread's illustration prompt.

### stage3_assemble.py

- `_choose_layout_variant` and the text-zone-height computation are
  re-measured at spread width instead of single-page width. Geometry logic
  (contain-fit image, no scrim, text drawn directly on background) is
  otherwise unchanged from the 2026-07-01 design.
- The PDF page size for a spread is two A4 portrait widths side by side,
  same height.

### stage1_story.py

- No change to `generate_story`'s output contract (still *k* strings, one
  per spread).
- `LLM_STORY_GENERATION`'s continuity rule ("Pages must connect naturally
  (reference what happened before)") is replaced with the specific
  techniques mined from the reference taxonomy (recurring character/object
  callback, consequence-of-prior-action framing, setting persistence) so
  the instruction is traceable to evidence rather than a generic platitude.

## Evaluation (the thesis contribution)

### Continuity taxonomy

A small taxonomy derived from the 7 reference books (0-2 + 3-6), used as a
scoring rubric:

| Category | Example from references |
| --- | --- |
| Recurring character/object | the watering can reappears across 企鵝 spreads |
| Consequence-of-prior-action | 我學會等待: "但是,某天早上,米亞沒有出來" continues yesterday's routine |
| Setting persistence | same schoolyard across 陶樂蒂's spreads |
| Emotional arc progression | crying → resolution across 陶樂蒂's spreads |

### Calibration set

The captions already visible in the 15 captured 3-6 spreads (plus the 0-2
references) are manually transcribed and hand-coded against the taxonomy
for each adjacent pair (~14 transitions total). This produces a
"professional baseline" continuity score to compare pipeline output
against — the calibration set lives in
`docs/eval/reference_continuity_calibration.md` (new file, plain
markdown table: book, spread pair, category, notes).

### New eval script: `src/eval/narrative_continuity_judge.py`

Extends the `llm_judge.py` script already planned in
`docs/next_deliverables.md`. For each adjacent caption pair in a pipeline
run, asks the judge model to classify continuity against the same taxonomy
used for calibration. Outputs per-transition category + a book-level
aggregate (e.g. "6 of 8 transitions show explicit continuity").

Mocks the judge LLM call in tests, per `.claude/rules/testing.md` (no real
HTTP calls in unit tests).

### Within-spread relevance

Reuses the `clip_score.py` script already planned in
`docs/next_deliverables.md`, applied per spread (caption vs. its own
illustration) instead of per page. No new script.

### Thesis claim this produces

A quantified, benchmarked comparison: the `causal` ablation's continuity
score against the taxonomy, versus the same taxonomy's score on real
published books, versus the `no_causal` baseline — replacing the current
purely subjective side-by-side comparison in `docs/next_deliverables.md`'s
RQ3 deliverable.

## Testing

- `test_choose_layout_variant_uses_spread_width` — layout decision measured
  at spread width, not single-page width.
- `test_build_spread_pdf_creates_file` — covering both top-band and
  bottom-caption variants at spread geometry.
- `test_build_spread_pdf_wrong_count_raises` — unchanged count-validation
  behavior.
- `test_illustrate_fal_uses_spread_aspect_ratio` — mocked FAL client,
  asserts the landscape aspect-ratio parameter is sent.
- `test_illustrate_fal_prompt_includes_character_description` — mocked FAL
  client, asserts the per-book character/setting string is present in every
  spread's prompt.
- `test_narrative_continuity_judge_classifies_adjacent_pairs` — mocked
  judge LLM call, asserts taxonomy category is returned per transition.
- `test_narrative_continuity_judge_aggregates_book_level_score` — pure
  function test over a list of per-transition classifications.

Integration tests assert file creation and non-trivial size (matching the
existing style in `tests/test_stage3_assemble.py`), not pixel-level PDF
geometry.

## Out of scope

- The legacy `image_top_text_bottom` layout (`build_pdf`) and the
  0-2 age `picture_book` single-page layout — both untouched, already
  shipped.
- Split/mirrored dual-caption structure (我學會等待 style).
- Opportunistic whitespace-detection caption placement (陶樂蒂 style).
- Pixel-identical character consistency (seed/reference-image
  conditioning, IP-Adapter, LoRA, etc.) — prompt-level consistency only.
- Changing the photo-count-to-spread-count mapping in `stage0_select`
  (stays 1:1).
- Bold CJK font support (carried over as out of scope from the 2026-07-01
  spec).
