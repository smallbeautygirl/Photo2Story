# Photo2Story

Turns a set of photos into an illustrated picture-book story: selecting photos, generating narrative text, illustrating pages, and assembling the result into a PDF.

## Language

**Badness**:
A per-cell (and per-candidate) score in `[0, 1]` describing how unsafe a region of an illustration is for placing text — `0` is ideal (quiet, low-detail), `1` is worst (busy, salient). Computed from saliency, edge density, and local variance. The canonical term for this axis everywhere in code, comments, and docs — including the type that carries it (`BadnessMap`) — regardless of whether a given expression reads it directly or as its complement.
_Avoid_: Suitability, suitability score, suitability map.

**Shape preset**:
One of three fixed rectangle width fractions (`narrow-tall` 0.28, `medium` 0.45, `wide-short` 0.65) that a Candidate's search is anchored to. Each preset is searched independently for its own best position and font size.

**Candidate**:
One fully-fitted text-placement rectangle for a page: a position, size, font size, and badness score, produced by searching one shape preset over a page's badness map. Identified by the shape preset that produced it. Its x/y/w/h are fractions of the final rendered page — i.e. of the illustration _after_ the cover-fit crop, not of the raw illustration file — so the badness map they're derived from must itself be analyzed over the post-crop visible region.

**Wordless page**:
A page whose caption is empty (or whitespace-only). Renders as a full-bleed illustration with no candidate search and no text overlay drawn — the badness map is never computed for it.

**Gutter**:
The vertical line at the physical binding fold of a `picture_book_spread` page, at `page_width / 2`. A hard exclusion for text candidates — a candidate whose x-range would cross it is never generated — but not a constraint on the illustration, which may span it freely. Does not apply to single-page `picture_book` layouts.

**Character reference image**:
One or more of the most recently generated spread illustrations (a rolling window of the previous 1-2 spreads, not the original source photos), passed to FLUX Kontext as image-conditioning input so the next spread's illustration visually matches the recurring character's actual rendered appearance rather than only a text description of it. Gated by `stage2.use_character_reference`; unavailable for the first spread of a book, which always falls back to text-only generation.
_Avoid_: character reference (ambiguous with the pre-existing text-only continuity string built by `_build_character_reference` in `illustrate_fal.py`), reference image (also used for the unrelated IP-Adapter style-image mechanism in the local SD 1.5 backend).
