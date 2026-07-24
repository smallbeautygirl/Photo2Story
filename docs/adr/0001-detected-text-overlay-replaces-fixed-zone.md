# Detected text-safe overlay replaces the fixed text zone

`picture_book` and `picture_book_spread` previously reserved a fixed top-band
or bottom-caption zone outside the illustration for the caption, guaranteeing
a safe place for text regardless of the art. We replaced this entirely with a
full-bleed illustration plus a programmatically detected text-safe region
(badness map + candidate search, see
[2026-07-22-picture-book-text-overlay-design.md](../superpowers/specs/2026-07-22-picture-book-text-overlay-design.md)),
matching the `reference/3-6/` picture books where text is drawn directly on
the art wherever that page happens to be visually simple.

We traded the old zone's unconditional reliability for closer fidelity to
real picture-book layout. The old zone-selection code
(`_choose_layout_variant`, `_text_zone_height`, and the fixed-zone font-size
constants) is removed, not deprecated — reverting means re-adding it, not
flipping a flag.

## Consequences

- No configuration fallback to the old fixed-zone behavior; if the detected
  overlay ever produces an unreadable result, the fix is in the detection
  logic, not a config switch back to the old layout.
- Adds `opencv-contrib-python` as a new core (CPU-only) dependency for
  `cv2.saliency` and `cv2.Canny`.
- `image_top_text_bottom` (the legacy `build_pdf` used by ablation configs)
  is untouched and still uses the old fixed layout — this decision applies
  only to `picture_book`/`picture_book_spread`.
