# Bundled fonts

The pipeline's one fixed typeface per language (see
[docs/adr/0003-fixed-font-across-pipeline.md](../../docs/adr/0003-fixed-font-across-pipeline.md)) —
not matched to illustration style, not varied across RQ ablation conditions.

| File | Language | Source | License |
| --- | --- | --- | --- |
| `jf-openhuninn.ttf` | zh-tw | [justfont/open-huninn-font](https://github.com/justfont/open-huninn-font), release v2.1 | OFL-1.1 (`LICENSE-jf-openhuninn.txt`) |
| `OpenSans.ttf` | en | [google/fonts](https://github.com/google/fonts/tree/main/ofl/opensans) | OFL-1.1 (`LICENSE-OpenSans.txt`) |

Both are TrueType (glyf-outline) files, which ReportLab's `TTFont` can embed
directly — unlike CFF-outline fonts (e.g. Noto Sans CJK's `.ttc` releases),
which `TTFont` cannot load at all.

**Taipei Sans TC Beta** was the other finalist (stronger citability case —
derived from Adobe/Google's documented Source Han Sans — and a closer visual
match to `reference/3-6`'s body text) but isn't bundled here: it's reserved,
unwired, for a possible future reader-preference study against
`jf-openhuninn.ttf`. If that study happens, source it fresh from
[JT Foundry's official distribution](https://sites.google.com/view/jtfoundry/en/downloads) —
not from a third-party mirror.
