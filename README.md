# Photo2Story

Turns a set of photos into an illustrated picture-book story: selecting representative photos, inferring their order, generating a coherent narrative, illustrating each page in a consistent style, and assembling the result into a downloadable PDF.

This is also the implementation behind the thesis *基於多模態大語言模型之個人化照片繪本自動生成系統* (A Personalized Photo-to-Picture-Book Generation System Based on Multimodal LLMs). The core research contribution is Stage 0's two-stage hybrid photo-selection algorithm, which is the first to bring a user's natural-language context description into the selection scoring function for personalized picture-book generation.

## Pipeline

```
N photos (+ optional context description, + style choice)
    → [Stage 0] photo analysis & selection   (src/stages/stage0_select.py)  — core research contribution
    → [Stage 1] story text generation        (src/stages/stage1_story.py)
    → [Stage 2] illustration generation      (src/stages/stage2_illustrate.py, illustrate_fal.py)
    → [Stage 3] PDF assembly & text overlay  (src/stages/stage3_assemble.py, text_placement.py)
```

Entry points: `python run_experiment.py --config configs/demo.yaml ...` (CLI pipeline runs) and `python gradio_app.py` (interactive demo UI).

## Where things live

| What | Where |
|---|---|
| Domain glossary (canonical terms: Badness, Candidate, Gutter, ...) | [`CONTEXT.md`](CONTEXT.md) |
| Design evolution — why the pipeline looks the way it does, in chronological order | [`docs/superpowers/specs/`](docs/superpowers/specs/) |
| Implementation plans matching each spec | [`docs/superpowers/plans/`](docs/superpowers/plans/) |
| Architectural decisions that were hard to reverse | [`docs/adr/`](docs/adr/) |
| Evaluation methodology — the narrative-continuity judge and its calibration set, the thesis's evaluation contribution | [`docs/eval/`](docs/eval/) |
| Progress artifacts — monthly/meeting reports, deliverables checklists | [`docs/reports/`](docs/reports/) |
| Reference picture books used for grounding the layout and continuity evaluation | `reference/0-2/`, `reference/3-6/` |

### Reading the design history in order

The specs and plans in `docs/superpowers/` narrate how the pipeline actually evolved — useful as source material when writing up the methodology:

1. `2026-04-28-photo2story-design` — original system design and research questions
2. `2026-06-28-storybook-improvements-design` — style scope narrowed to 4 styles, reading-level tiers, full-spread layout
3. `2026-07-01-picture-book-layout-design` — `picture_book` full-bleed layout (superseded by spread layout below, never shipped standalone)
4. `2026-07-08-picture-book-spread-continuity-design` — double-page-spread layout + the narrative-continuity evaluation that is the thesis's evaluation contribution
5. `2026-07-10-zhuyin-annotation-design` — Bopomofo annotation for `zh-tw` output
6. `2026-07-22-picture-book-text-overlay-design` — detected text-safe regions replace the fixed caption zone

`docs/adr/` records the decisions from steps 4–6 that were expensive to reverse (detected overlay over fixed zone, heuristic badness scoring over segmentation, one fixed font across the pipeline).
