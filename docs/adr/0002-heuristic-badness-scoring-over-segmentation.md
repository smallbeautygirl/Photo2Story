# Badness scoring uses classical CV heuristics, not semantic segmentation

Text-safe region detection scores each cell of an illustration with saliency
+ edge density + local variance (`cv2.saliency`, `cv2.Canny`, local stddev) —
all CPU-only, no GPU, no model weights. We deliberately did not reach for a
semantic foreground-segmentation model (SAM2, GroundingDINO, or a hosted
equivalent) even though one would separate "the subject" from "the
background" more reliably than saliency alone (e.g. a character's
plain-colored clothing scores as safe by variance/edges but should still be
avoided).

We traded segmentation-grade accuracy for zero GPU dependency and low
latency, matching this pipeline's existing CPU-only assembly stage. This is
a scope decision, not an unfinished one: the `BadnessMap`'s channels
(saliency, edge density, variance) are structured as independent inputs to
the badness formula specifically so a foreground-mask channel can be added
later without restructuring the candidate search or ranking — see
[2026-07-22-picture-book-text-overlay-design.md](../superpowers/specs/2026-07-22-picture-book-text-overlay-design.md).

## Consequences

- Badness scoring can misjudge large, visually flat but semantically
  important regions (a character's plain shirt, a single-color prop) as
  text-safe, since nothing distinguishes "subject" from "background" beyond
  saliency's proxy signal.
- Adding real segmentation later is additive (one more weighted channel),
  not a rewrite — this was a design constraint on the current work, not a
  promise about when it will happen.
