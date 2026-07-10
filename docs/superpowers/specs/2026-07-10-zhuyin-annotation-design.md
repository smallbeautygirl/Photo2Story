# Traditional Chinese Zhuyin (注音符號) annotation

## Problem

`configs/demo.yaml`-style runs can already request `--language zh-tw`, and
`stage1_story.py` will dutifully generate Traditional Chinese page text via
Gemini. But two things are broken or missing for that output to actually be
usable in a real children's book:

1. **Existing bug: `zh-tw` output does not render at all.**
   `stage3_assemble._resolve_cjk_font` only looks for CJK TrueType font files
   at fixed system paths (`AR PL UMing/UKai TW`, `STHeiti`), none of which
   exist on this machine or on most Linux deployments (confirmed: this
   machine only has Noto Sans/Serif CJK, which are OpenType/CFF outline
   fonts that ReportLab's `TTFont` wrapper cannot load — the existing code
   comment already notes this). The fallback is plain `Helvetica`, which has
   no CJK glyphs, so `zh-tw` captions currently render as missing/blank
   glyphs.
2. **No Zhuyin annotation**, which is the actual feature request: for early
   readers, Traditional Chinese children's books conventionally print the
   Bopomofo phonetic guide next to each character (see reference photos
   collected during brainstorming — `世代相傳`, `我不要上學！`, and a
   scanned page from 陶樂蒂的開學日).

## Decisions

1. **Fix the CJK font gap with ReportLab's built-in CID font support**,
   not a bundled/embedded font file. Confirmed via direct testing:
   `pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))` renders both
   Traditional Chinese hanzi and Bopomofo/tone-mark glyphs correctly with
   zero font files needed (relies on the PDF viewer's own CJK font
   substitution, which is universally supported). This becomes a new
   fallback tier in `_resolve_cjk_font`, tried after the existing file-based
   candidates (so an embedded font is still preferred when one is actually
   present) and before giving up to `Helvetica`. This fixes `zh-tw`
   rendering for every caller, not just the Zhuyin path.
2. **Zhuyin is always-on for `zh-tw`, no config toggle.** The whole point of
   choosing `zh-tw` output for a children's book is early readers; there's
   no current use case for `zh-tw` without it.
3. **Scope: only `zh-tw`, only the active `picture_book`/`picture_book_spread`
   layouts.** `zh-cn` conventionally uses Pinyin (not Bopomofo) in mainland
   China, so it is not touched. The legacy `image_top_text_bottom` /
   `build_pdf` ablation-baseline path is untouched — Zhuyin only affects the
   two current picture-book layouts.
4. **Layout, confirmed visually during brainstorming** (see
   `.superpowers/brainstorm/1279451-1783666259/content/zhuyin-layout-v5.html`
   for the approved mockup):
   - Each character gets extra trailing horizontal space to hold its own
     phonetic annotation (not overlapping the next character).
   - The main Bopomofo letters (initial/medial/final, excluding the tone
     mark) are stacked vertically, one per line, immediately after the
     character — built as literal stacked rows rather than CSS
     vertical-writing-mode, because that CSS property rendered inconsistently
     across fonts during mockup iteration; the PDF implementation draws each
     letter at an explicit y-coordinate for the same reason.
   - The tone mark is **not** part of that vertical stack. It's drawn beside
     the stack, vertically positioned by tone: 2nd tone (ˊ) upper-right, 3rd
     tone (ˇ) middle-right, 4th tone (ˋ) lower-right, neutral tone (˙) above
     the first letter. 1st tone (flat) has no mark at all.
   - Punctuation, spaces, and any non-Hanzi characters (e.g. an English word
     embedded mid-sentence) pass through with no annotation and no extra
     spacing.
5. **`pypinyin` is trusted as-is for heteronym disambiguation** (its default
   phrase-dictionary-based best guess). No extra context is fed to it, and
   its choice is not second-guessed.

## Architecture

### New: `src/utils/zhuyin.py`

Pure, ReportLab-independent conversion logic — Hanzi text in, one
`ZhuyinChar` per input character out. No PDF/canvas dependency, so it's
directly unit-testable.

```python
@dataclass(frozen=True)
class ZhuyinChar:
    char: str
    main: str        # stacked Bopomofo letters, tone mark excluded, "" if not Hanzi
    tone_mark: str    # one of "", "ˊ", "ˇ", "ˋ", "˙"

def annotate(text: str) -> list[ZhuyinChar]: ...
```

Implementation: `pypinyin.pinyin(text, style=Style.BOPOMOFO, heteronym=False)`
returns one entry per input character (confirmed by direct testing, including
for multi-character words — pypinyin's internal segmentation is only used to
disambiguate pronunciation, not to change the 1:1 output-to-input-character
alignment). For characters where the returned string is unchanged from the
input (punctuation, Latin, digits, spaces), emit `main=""`, `tone_mark=""`.
Otherwise, if the last character of the returned string is one of
`ˊˇˋ˙`, split it off as `tone_mark`; the rest is `main`. No trailing tone
mark means 1st tone (`tone_mark=""`, `main=` the full returned string).

### New: `src/stages/zhuyin_render.py`

ReportLab-specific: consumes `ZhuyinChar` lists, measures cell widths,
wraps lines, and draws.

```python
def wrap_zhuyin(zchars: list[ZhuyinChar], font: str, font_size: float, max_width: float) -> list[list[ZhuyinChar]]: ...
def draw_zhuyin_line(c: canvas.Canvas, line: list[ZhuyinChar], font: str, font_size: float, x: float, top_y: float, align: Literal["left", "center"]) -> None: ...
```

- Cell width per character = hanzi glyph width + a fixed extra gap sized to
  fit the widest possible Bopomofo stack (main letters are narrow;
  the gap is measured, not eyeballed, from `pdfmetrics.stringWidth` at the
  Zhuyin font size).
- Wrapping is character-based (matches the existing CJK branch of
  `_wrap_to_width` — no spaces to break on), using the wider cell width
  instead of the plain glyph width.
- Drawing places the hanzi glyph, then the stacked main letters (small font,
  each on its own explicit y-coordinate, top-aligned to the character), then
  the tone mark positioned per the rule in Decision 4.

### `src/stages/stage3_assemble.py`

- `_resolve_cjk_font` gains the `STSong-Light` CID-font fallback tier
  (Decision 1).
- `build_picture_book_pdf` / `_draw_picture_book_page` / `_draw_zone_text`
  gain a `language: str = "en"` parameter, threaded from `run_stage3`. When
  `language == "zh-tw"`, the text-drawing call delegates to
  `zhuyin_render.wrap_zhuyin` / `draw_zhuyin_line` instead of the plain
  `_wrap_to_width` / `_draw_zone_text` path. Any other language is
  byte-for-byte unaffected.
- `_text_zone_height` is unaffected: the confirmed layout keeps the Zhuyin
  stack within the character's own line height (vertically centered beside
  it), not adding a separate line above/below, so per-book text-zone height
  math doesn't change.
- `run_stage3` reads `language = stage1_result.get("language", "en")` and
  passes it through to whichever `build_*_pdf` function it dispatches to.

### `src/stages/stage1_story.py`

- `run_stage1`'s return dict gains `"language": language` (currently only
  `{"pages": ..., "narrative": ...}`). This is the only change needed to get
  `language` to `run_stage3` — `pipeline.py`'s call signature is untouched.

### `requirements.txt`

- Add `pypinyin` (no GPU/torch dependency; pure-Python + bundled data
  tables).

## Testing

- `tests/test_zhuyin.py`: real, unmocked tests of `annotate()` (it's a local
  deterministic dictionary lookup, not an external API — same rationale as
  not mocking `piexif`/`Pillow` elsewhere in this codebase). Covers: all 5
  tones extracted correctly, punctuation/Latin passthrough, multi-character
  phrase alignment (e.g. confirms `len(annotate(text)) == len(text)`).
- `tests/test_zhuyin_render.py`: `wrap_zhuyin` produces cell widths wider
  than plain-text wrapping for the same string (proves the extra gap is
  reserved); `draw_zhuyin_line` runs against a real canvas without error.
- `tests/test_stage3_assemble.py`: a `zh-tw` call to `build_picture_book_pdf`
  produces a valid, larger-than-trivial PDF (same file-exists/size-threshold
  rigor as the existing PDF tests — this codebase doesn't assert pixel
  positions in unit tests; visual correctness was already validated through
  the browser mockups during brainstorming).
- `tests/test_stage1_story.py`: `run_stage1`'s return dict includes the
  `language` key.

## Out of scope

- `zh-cn` (Pinyin, not Bopomofo, is the mainland-China convention).
- The legacy `image_top_text_bottom` / `build_pdf` ablation baseline.
- A config toggle to disable Zhuyin for `zh-tw` (Decision 2).
- Any change to `stage1_story.generate_story`'s output contract (still `k`
  plain-text strings; Zhuyin is a stage3 rendering concern only, not a
  story-generation concern).
