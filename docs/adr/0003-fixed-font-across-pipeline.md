# One fixed CJK typeface across the whole pipeline, not matched to illustration style

The RQ1–3 ablation configs manipulate photo-selection mode (`stage0.mode`) and
narrative generation (`context_mode`, `use_causal_inference`) as the study's
independent variables; illustration style and page layout are already held
constant across those conditions. Letting typography vary per illustration
style would introduce an uncontrolled confound on top of that — the same risk
the study already avoids elsewhere — so typography is treated the same way:
one fixed typeface everywhere, chosen once, not adapted per page.

This also surfaced (not caused) a real bug: `_resolve_cjk_font`'s candidate
list (`AR PL UMing TW`, `AR PL UKai TW`, `STHeiti`) pointed at filesystem
paths that don't exist in this environment, so every render — both the
legacy `image_top_text_bottom` layout and the new `picture_book`/
`picture_book_spread` overlay — was silently falling back to ReportLab's
built-in `STSong-Light`, a *simplified*-Chinese serif CID font, regardless of
the `zh-tw` language setting. **jf-openhuninn** (bundled at
`assets/fonts/jf-openhuninn.ttf`, OFL-1.1) replaces that fallback chain as
the project's one primary typeface, selected against: early-reader
legibility (sans-serif), native Bopomofo/tone-mark glyph coverage (a
functional requirement for `zhuyin_render.py`, not an aesthetic one), open
redistributable license, and descriptive consistency with `reference/3-6`'s
actual body-text style — decided during a grilling session comparing
rendered samples against `reference/3-6` (zh-tw) and `reference/0-2` (en).

## Considered options

- **Font matched to illustration style per page** — rejected: no clean,
  citable rule for the mapping, and it would confound the RQ1–3 comparison.
- **Taipei Sans TC Beta** (Source Han Sans-derived, TrueType rebuild) — the
  stronger citability/visual-fidelity case, but not picked as primary;
  reserved unwired for a possible future reader-preference study comparing
  it against jf-openhuninn. If that study happens, its font file must be
  re-sourced from JT Foundry's official distribution — the copy used to
  generate preview renders during this session came from a third-party npm
  mirror, not the canonical channel, and should not be reused as-is.
- **Iansui** — visually near-identical to jf-openhuninn (both descend from
  Japanese "maru gothic" designs); dropped rather than carried as a second
  comparison arm, since comparing two near-twin fonts wouldn't add
  information.

## Consequences

- Any future work that makes illustration style visible to stage3 (e.g. a
  style-conditioned layout) must not also condition font choice on it,
  without first revisiting this ADR.
- English captions still need their own fix: `_resolve_cjk_font` is used
  unconditionally regardless of `language`, so `en` pages currently render in
  a CJK typeface by accident. Tracked as a follow-up, not fixed by this ADR.
