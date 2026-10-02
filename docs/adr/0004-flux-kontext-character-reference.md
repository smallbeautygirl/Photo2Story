# Use FLUX Kontext image conditioning for cross-spread character consistency

Cross-spread character continuity was deliberately kept prompt-only in the
2026-07-08 picture-book-spread-continuity design
(`docs/superpowers/specs/2026-07-08-picture-book-spread-continuity-design.md`),
which explicitly scoped out "pixel-identical character consistency
(seed/reference-image conditioning, IP-Adapter, LoRA, etc.)". Production
output since then (`docs/reports/monthly_report_0729.md`) shows why
prompt-only isn't enough on its own: silhouette (hairstyle/face shape/outfit)
holds up under text-only conditioning, but skin tone visibly drifts page to
page — a failure mode text prompts can't fix because it comes from the
diffusion sampling's own stochasticity, not from a gap in the description.
Following the advisor's 2026-08-03 feedback, `stage2_illustrate`'s fal.ai
backend switches from `flux-lora`/`flux/dev` to `flux-kontext-lora`, passing
the 1-2 most recently generated spread illustrations as a [[character
reference image]] conditioning input alongside the existing style LoRA and
text prompt.

## Considered Options

- Keep prompt-only continuity, only enrich the text (e.g. have a VLM describe
  the previously *generated* illustration instead of reusing the original
  photo description) — rejected because it doesn't address the underlying
  cause: sampling variance, not description precision.
- Fixed single-anchor reference image (always the first spread) instead of a
  rolling window — rejected in favor of the rolling window because it matches
  the advisor's specific suggestion ("前幾頁", the previous few pages) and
  needs no separate "character sheet" generation step; revisit if drift
  compounds noticeably over longer books.

## Consequences

- `stage2.use_character_reference` config flag gates this, matching the
  existing `use_causal_inference` ablation-flag pattern — so a with/without
  image-conditioning comparison stays possible, and the pipeline can fall
  back to the old text-only path if Kontext's cost, latency, or quality
  doesn't hold up.
- The first spread of every book has no prior illustration to condition on,
  so it always falls back to the existing text-only generation.
- The pre-existing text-based mechanism (`_build_character_reference` in
  `illustrate_fal.py`, producing a continuity *string*) is a different
  mechanism from this image-conditioning approach. The name collision should
  be resolved (e.g. rename that function to `_build_character_description`)
  when this ships, to avoid confusing the two in code.
- Not yet implemented — this ADR records the decision reached in the
  2026-08-11 grilling session; the code change (`illustrate_fal.py`, config
  schema, tests) is a separate task.
