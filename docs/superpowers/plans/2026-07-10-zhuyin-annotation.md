# Zhuyin (注音符號) Annotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Bopomofo phonetic annotation to `zh-tw` picture-book output, and fix the existing bug where `zh-tw` text currently renders with missing glyphs.

**Architecture:** A new pure conversion module (`src/utils/zhuyin.py`, Hanzi → `ZhuyinChar` list via `pypinyin`) feeds a new ReportLab-specific rendering module (`src/stages/zhuyin_render.py`, character-cell-width measurement + wrapping + drawing). `stage3_assemble.py` threads a `language` value through its picture-book drawing path and delegates to `zhuyin_render.py` only when `language == "zh-tw"`; every other language is unaffected. `stage3_assemble._resolve_cjk_font` also gains a built-in CID-font fallback (`STSong-Light`), fixing `zh-tw` rendering for every caller, not just this feature.

**Tech Stack:** Python 3, ReportLab (PDF), `pypinyin` (Hanzi → Bopomofo, new dependency), pytest + pytest-mock.

## Global Constraints

- `from __future__ import annotations` at the top of every new/modified Python file.
- Built-in generics only: `list[str]`, `dict[str, float]`, `X | None` — never `List`/`Dict`/`Optional` from `typing` (`Literal` is the one exception already established in this codebase's `config.py`/`stage3_assemble.py` and is fine to keep using).
- f-strings for all string interpolation; `pathlib.Path` for all file paths.
- No `print()` — use `logging.getLogger(__name__)` with `extra={...}` kwargs, per `.claude/rules/logging.md`.
- `pypinyin` is a local, deterministic dictionary lookup with no network calls — per `.claude/rules/testing.md`'s spirit (mock *external* APIs only), it is used directly (unmocked) in tests, the same way this codebase already uses `piexif`/`Pillow` directly.
- Test files stay flat in `tests/` (matching this repo's existing convention, not `tests/<subpackage>/`).
- Ruff-compatible formatting: double quotes, 100-char line length.
- Commits follow Conventional Commits with this repo's established scopes (`pipeline` for stage/eval code) — see `230f6c3 feat(pipeline): ✨ add anatomy-quality prompt hint, bump caption font sizes` for direct recent precedent. End every commit body with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

**Note on a spec correction found while planning:** the design spec
(`docs/superpowers/specs/2026-07-10-zhuyin-annotation-design.md`) states
`_text_zone_height` is "unaffected" by this feature. That's wrong: Zhuyin
annotation makes each character wider (extra space for the phonetic column),
so a `zh-tw` caption wraps to *more* lines than plain-text wrapping would
predict. If `_text_zone_height`/`_choose_layout_variant` used plain wrapping
to measure `zh-tw` text, the reserved text zone could be too short, clipping
text. Task 5 below makes both functions (and `_draw_zone_text`)
`language`-aware for this reason — this plan is the corrected version.

---

### Task 1: `src/utils/zhuyin.py` — Hanzi → Bopomofo conversion

**Files:**
- Create: `src/utils/zhuyin.py`
- Modify: `requirements.txt`
- Test: `tests/test_zhuyin.py`

**Interfaces:**
- Produces: `ZhuyinChar` (frozen dataclass: `char: str`, `main: str`, `tone_mark: str`), `annotate(text: str) -> list[ZhuyinChar]`. Task 4's `zhuyin_render.py` consumes both.

- [ ] **Step 1: Add `pypinyin` to requirements**

In `requirements.txt`, add this line after `piexif>=1.1.3`:

```
pypinyin>=0.55.0
```

- [ ] **Step 2: Install it and write the failing test**

Run: `pip install pypinyin>=0.55.0`

Create `tests/test_zhuyin.py`:

```python
# tests/test_zhuyin.py
from src.utils.zhuyin import ZhuyinChar, annotate


def test_annotate_returns_one_entry_per_character():
    text = "陶樂蒂的開學日"
    result = annotate(text)
    assert len(result) == len(text)
    assert [zc.char for zc in result] == list(text)


def test_annotate_extracts_all_five_tones():
    # 媽(1st,no mark) 麻(2nd,ˊ) 馬(3rd,ˇ) 罵(4th,ˋ) 嗎(neutral,˙)
    result = annotate("媽麻馬罵嗎")
    assert result[0] == ZhuyinChar(char="媽", main="ㄇㄚ", tone_mark="")
    assert result[1] == ZhuyinChar(char="麻", main="ㄇㄚ", tone_mark="ˊ")
    assert result[2] == ZhuyinChar(char="馬", main="ㄇㄚ", tone_mark="ˇ")
    assert result[3] == ZhuyinChar(char="罵", main="ㄇㄚ", tone_mark="ˋ")
    assert result[4] == ZhuyinChar(char="嗎", main="ㄇㄚ", tone_mark="˙")


def test_annotate_passes_through_punctuation_and_latin():
    result = annotate("我愛Hi world的開學日123")
    assert len(result) == len("我愛Hi world的開學日123")
    non_hanzi = [zc for zc in result if zc.char in "Hi world123"]
    assert all(zc.main == "" and zc.tone_mark == "" for zc in non_hanzi)
    # the Hanzi around the passthrough run still got real readings
    assert result[0] == ZhuyinChar(char="我", main="ㄨㄛ", tone_mark="ˇ")


def test_annotate_empty_string_returns_empty_list():
    assert annotate("") == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_zhuyin.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.zhuyin'`

- [ ] **Step 4: Implement `zhuyin.py`**

Create `src/utils/zhuyin.py`:

```python
"""Hanzi -> Bopomofo (注音符號) conversion for Traditional Chinese captions.

Used by src/stages/zhuyin_render.py to annotate each character of a zh-tw
caption with its phonetic reading, for early readers.
"""

from __future__ import annotations

from dataclasses import dataclass

from pypinyin import Style, pinyin

_TONE_MARKS = {"ˊ", "ˇ", "ˋ", "˙"}


@dataclass(frozen=True)
class ZhuyinChar:
    """One character of input text, with its Bopomofo reading split out.

    `main` is the stacked initial/medial/final letters, tone mark excluded.
    `tone_mark` is one of "" (1st/flat tone), "ˊ", "ˇ", "ˋ", "˙" (neutral).
    Non-Hanzi characters (punctuation, Latin, digits, spaces) get
    `main=""`, `tone_mark=""` -- there is nothing to annotate.
    """

    char: str
    main: str
    tone_mark: str


def annotate(text: str) -> list[ZhuyinChar]:
    """Convert each character of `text` to its Bopomofo reading.

    pypinyin groups consecutive non-Hanzi characters into a single returned
    "reading" (e.g. the run "Hi world" comes back as one list entry, not one
    per character), so a naive zip(text, readings) misaligns as soon as any
    non-Hanzi text is present. This walks `text` by consuming exactly
    `len(reading)` characters per entry, re-expanding grouped passthrough
    runs back to one ZhuyinChar per original character.
    """
    if not text:
        return []

    readings = pinyin(text, style=Style.BOPOMOFO, heteronym=False)
    result: list[ZhuyinChar] = []
    pos = 0
    for (reading,) in readings:
        if text[pos : pos + len(reading)] == reading:
            # pypinyin returned this run unchanged: not Hanzi, no annotation.
            for ch in reading:
                result.append(ZhuyinChar(char=ch, main="", tone_mark=""))
            pos += len(reading)
        else:
            # A real Bopomofo reading, always for exactly one Hanzi character.
            char = text[pos]
            if reading and reading[-1] in _TONE_MARKS:
                result.append(ZhuyinChar(char=char, main=reading[:-1], tone_mark=reading[-1]))
            else:
                result.append(ZhuyinChar(char=char, main=reading, tone_mark=""))
            pos += 1
    return result
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_zhuyin.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt src/utils/zhuyin.py tests/test_zhuyin.py
git commit -m "feat(pipeline): ✨ add Hanzi-to-Bopomofo conversion

annotate() converts each character of a string to its Bopomofo
reading via pypinyin, splitting the trailing tone mark out from the
main stacked letters. Correctly re-expands pypinyin's grouped
non-Hanzi passthrough runs (e.g. embedded Latin text) back to one
ZhuyinChar per original character, avoiding a misalignment bug a
naive zip(text, readings) would have.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: CJK font fallback fix in `stage3_assemble.py`

**Files:**
- Modify: `src/stages/stage3_assemble.py:1-14` (imports), `:52-78` (`_resolve_cjk_font`)
- Test: `tests/test_stage3_assemble.py`

**Interfaces:**
- No signature change to `_resolve_cjk_font(font_path: str | None = None) -> str`. Existing callers (`build_picture_book_pdf`, `build_pdf`) are unaffected.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage3_assemble.py`:

```python
def test_resolve_cjk_font_falls_back_to_builtin_cid_font(monkeypatch):
    """When no embedded TTF font file is found, fall back to ReportLab's
    built-in STSong-Light CID font instead of plain Helvetica (which has no
    CJK glyphs at all)."""
    import src.stages.stage3_assemble as mod

    monkeypatch.setattr(mod, "_cjk_font_name", None)
    monkeypatch.setattr(mod, "_CJK_FONT_CANDIDATES", [])

    result = mod._resolve_cjk_font()

    assert result == "STSong-Light"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage3_assemble.py -k builtin_cid -v`
Expected: FAIL with `AssertionError: assert 'Helvetica' == 'STSong-Light'`

- [ ] **Step 3: Add the `UnicodeCIDFont` import**

In `src/stages/stage3_assemble.py`, replace line 13:

```python
from reportlab.pdfbase.ttfonts import TTFError, TTFont
```

with:

```python
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFError, TTFont
```

- [ ] **Step 4: Replace `_resolve_cjk_font`**

Replace the whole function (lines 52-78):

```python
def _resolve_cjk_font(font_path: str | None = None) -> str:
    """Register and return the first usable CJK TrueType font, or 'Helvetica' if none load.

    Result is cached: the font is registered once per process.
    """
    global _cjk_font_name
    if _cjk_font_name is not None:
        return _cjk_font_name

    candidates = list(_CJK_FONT_CANDIDATES)
    if font_path:
        candidates.insert(0, ("CustomCJK", font_path, 0))

    for name, path, index in candidates:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            _cjk_font_name = name
            logger.info("Registered CJK font", extra={"font": name, "path": path})
            return name
        except TTFError:
            continue

    logger.warning("No CJK TrueType font found; CJK text may not render in the PDF")
    _cjk_font_name = "Helvetica"
    return _cjk_font_name
```

with:

```python
def _resolve_cjk_font(font_path: str | None = None) -> str:
    """Register and return a usable CJK font.

    Tries an embedded TrueType font file first (an explicit `font_path`,
    then the fixed candidate list), then falls back to ReportLab's built-in
    'STSong-Light' CID font -- no font file needed, since it relies on the
    PDF viewer's own CJK font substitution (confirmed by direct testing to
    render Traditional Chinese and Bopomofo correctly). This fallback always
    succeeds: it ships as reportlab package data, not a filesystem lookup.

    Result is cached: the font is registered once per process.
    """
    global _cjk_font_name
    if _cjk_font_name is not None:
        return _cjk_font_name

    candidates = list(_CJK_FONT_CANDIDATES)
    if font_path:
        candidates.insert(0, ("CustomCJK", font_path, 0))

    for name, path, index in candidates:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            _cjk_font_name = name
            logger.info("Registered CJK font", extra={"font": name, "path": path})
            return name
        except TTFError:
            continue

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    _cjk_font_name = "STSong-Light"
    logger.info("Registered built-in CID CJK font", extra={"font": "STSong-Light"})
    return _cjk_font_name
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_stage3_assemble.py -v`
Expected: all PASS, including the new test and every pre-existing one (no other test asserts on `_resolve_cjk_font`'s specific return value, per `grep -n "_resolve_cjk_font\|Helvetica" tests/test_stage3_assemble.py` showing only literal `"Helvetica"` passed *as a test fixture argument*, never asserted as this function's output).

- [ ] **Step 6: Commit**

```bash
git add src/stages/stage3_assemble.py tests/test_stage3_assemble.py
git commit -m "fix(pipeline): 🐛 fall back to built-in CID font for CJK rendering

_resolve_cjk_font only checked for TrueType font files at fixed system
paths (AR PL UMing/UKai, STHeiti), none of which exist on this or most
Linux machines, so zh-tw output currently renders with missing glyphs
(silent fallback to Helvetica, which has no CJK glyphs at all).
ReportLab's built-in STSong-Light CID font needs no font file and
renders Traditional Chinese + Bopomofo correctly, confirmed by direct
testing. Fixes rendering for every caller of this function, not just
the zh-tw + Zhuyin path.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `run_stage1` returns the language used

**Files:**
- Modify: `src/stages/stage1_story.py:106-124`
- Test: `tests/test_stage1_story.py`

**Interfaces:**
- Produces: `run_stage1(...)` return dict gains `"language": str`. Task 5's `run_stage3` reads `stage1_result["language"]`. `pipeline.py`'s call signature is unaffected (it already passes `language` as a positional/keyword argument to `run_stage1` — only the *return value* changes).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage1_story.py`:

```python
def test_run_stage1_returns_language_used(mocker):
    stage0_result = {
        "ordered_paths": ["p1.jpg", "p2.jpg"],
        "descriptions": {"p1.jpg": "beach", "p2.jpg": "hotel"},
    }
    mocker.patch(
        "src.stages.stage1_story.generate_story", return_value=["Page 1.", "Page 2."]
    )
    config = {"model": "gemini-2.5-flash", "context_mode": "full", "use_causal_inference": False}

    result = run_stage1(
        stage0_result, context="trip", style="ghibli", language="zh-tw", config=config
    )

    assert result["language"] == "zh-tw"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage1_story.py -k returns_language_used -v`
Expected: FAIL with `KeyError: 'language'`

- [ ] **Step 3: Add `language` to the return dict**

In `src/stages/stage1_story.py`, replace line 124:

```python
    return {"pages": pages, "narrative": narrative}
```

with:

```python
    return {"pages": pages, "narrative": narrative, "language": language}
```

Also update the docstring at lines 109-112:

```python
    """
    Returns:
        {"pages": list[str], "narrative": str}
    """
```

with:

```python
    """
    Returns:
        {"pages": list[str], "narrative": str, "language": str}
    """
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage1_story.py -v`
Expected: all PASS, including the two pre-existing `test_run_stage1_*` tests (they don't assert on the full return dict shape, only `result["pages"]`/`result["narrative"]`, so the new key doesn't break them).

- [ ] **Step 5: Commit**

```bash
git add src/stages/stage1_story.py tests/test_stage1_story.py
git commit -m "feat(pipeline): ✨ thread language through run_stage1's return value

run_stage1 already receives language as a parameter but discarded it
after use. Returning it lets run_stage3 (Task 5) know which language
a book was generated in, without changing pipeline.py's call
signature to run_stage3 at all.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 4: `src/stages/zhuyin_render.py` — width measurement, wrapping, drawing

**Files:**
- Create: `src/stages/zhuyin_render.py`
- Test: `tests/test_zhuyin_render.py`

**Interfaces:**
- Consumes: `ZhuyinChar`, `annotate` (Task 1).
- Produces: `wrap_zhuyin(zchars: list[ZhuyinChar], font: str, font_size: float, max_width: float) -> list[list[ZhuyinChar]]`, `draw_zhuyin_line(c: canvas.Canvas, line: list[ZhuyinChar], font: str, font_size: float, x: float, y: float, w: float, align: Literal["left", "center"]) -> None`. Task 5's `stage3_assemble.py` calls both.

- [ ] **Step 1: Write the failing test**

Create `tests/test_zhuyin_render.py`:

```python
# tests/test_zhuyin_render.py
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.stages.zhuyin_render import draw_zhuyin_line, wrap_zhuyin
from src.utils.zhuyin import annotate


def test_cell_width_wider_than_plain_character():
    """A Zhuyin-annotated character must reserve more width than its plain
    glyph alone, since there needs to be room for the phonetic column."""
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    from src.stages.zhuyin_render import _cell_width

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    zc = annotate("陶")[0]
    plain_width = pdfmetrics.stringWidth("陶", "STSong-Light", 24)

    assert _cell_width(zc, "STSong-Light", 24) > plain_width


def test_wrap_zhuyin_wraps_at_max_width():
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    zchars = annotate("陶樂蒂的開學日")

    # A narrow width forces multiple lines; a very wide one fits on one line.
    narrow_lines = wrap_zhuyin(zchars, "STSong-Light", 24, max_width=60)
    wide_lines = wrap_zhuyin(zchars, "STSong-Light", 24, max_width=10_000)

    assert len(narrow_lines) > 1
    assert len(wide_lines) == 1
    # every character must appear exactly once across all lines, in order
    flat = [zc.char for line in narrow_lines for zc in line]
    assert flat == list("陶樂蒂的開學日")


def test_wrap_zhuyin_empty_input_returns_empty_list():
    assert wrap_zhuyin([], "Helvetica", 24, max_width=1000) == []


def test_draw_zhuyin_line_runs_without_error(tmp_path):
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    zchars = annotate("我不要上學！")
    lines = wrap_zhuyin(zchars, "STSong-Light", 24, max_width=10_000)

    out_path = str(tmp_path / "zhuyin_test.pdf")
    c = canvas.Canvas(out_path, pagesize=A4)
    for line in lines:
        draw_zhuyin_line(c, line, "STSong-Light", 24, x=50, y=700, w=400, align="center")
    c.save()

    from pathlib import Path

    assert Path(out_path).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_zhuyin_render.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.stages.zhuyin_render'`

- [ ] **Step 3: Implement `zhuyin_render.py`**

Create `src/stages/zhuyin_render.py`:

```python
"""ReportLab drawing for Zhuyin-annotated Traditional Chinese captions.

Consumes ZhuyinChar lists from src/utils/zhuyin.py. Each character reserves
extra width for its phonetic column (main letters stacked vertically,
tone mark positioned beside the stack), wraps character-by-character (no
spaces to break on, matching this codebase's existing CJK wrapping in
stage3_assemble._wrap_to_width), and draws accordingly.
"""

from __future__ import annotations

from typing import Literal

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen import canvas

from src.utils.zhuyin import ZhuyinChar

# Main Bopomofo letters render at this fraction of the base character's font size.
ZHUYIN_FONT_RATIO = 0.32
# Small fixed breathing room between a character's phonetic column and the next character.
ZHUYIN_GAP_PAD = 1.0


def _zhuyin_font_size(font_size: float) -> float:
    return font_size * ZHUYIN_FONT_RATIO


def _cell_width(zc: ZhuyinChar, font: str, font_size: float) -> float:
    """Total horizontal space one character (plus its annotation, if any) needs."""
    char_w = pdfmetrics.stringWidth(zc.char, font, font_size)
    if not zc.main and not zc.tone_mark:
        return char_w

    zy_size = _zhuyin_font_size(font_size)
    main_w = max((pdfmetrics.stringWidth(ch, font, zy_size) for ch in zc.main), default=0.0)
    tone_w = pdfmetrics.stringWidth(zc.tone_mark, font, zy_size) if zc.tone_mark else 0.0
    return char_w + main_w + tone_w + ZHUYIN_GAP_PAD


def wrap_zhuyin(
    zchars: list[ZhuyinChar], font: str, font_size: float, max_width: float
) -> list[list[ZhuyinChar]]:
    """Wrap Zhuyin-annotated characters to a pixel width, character-by-character."""
    lines: list[list[ZhuyinChar]] = []
    current: list[ZhuyinChar] = []
    current_width = 0.0
    for zc in zchars:
        w = _cell_width(zc, font, font_size)
        if current and current_width + w > max_width:
            lines.append(current)
            current = [zc]
            current_width = w
        else:
            current.append(zc)
            current_width += w
    if current:
        lines.append(current)
    return lines


def draw_zhuyin_line(
    c: canvas.Canvas,
    line: list[ZhuyinChar],
    font: str,
    font_size: float,
    x: float,
    y: float,
    w: float,
    align: Literal["left", "center"],
) -> None:
    """Draw one wrapped line, baseline at `y`.

    Each character is followed by its stacked main Bopomofo letters (small
    font, top-aligned within the character's height) and its tone mark
    positioned beside the stack: 2nd tone (ˊ) upper-right, 3rd tone (ˇ)
    mid-right, 4th tone (ˋ) lower-right, neutral tone (˙) above the first
    letter, 1st tone (no mark) drawn nowhere.
    """
    zy_size = _zhuyin_font_size(font_size)
    total_width = sum(_cell_width(zc, font, font_size) for zc in line)
    cursor_x = x if align == "left" else x + (w - total_width) / 2

    c.setFillColor(colors.black)
    for zc in line:
        cell_w = _cell_width(zc, font, font_size)
        c.setFont(font, font_size)
        c.drawString(cursor_x, y, zc.char)
        char_w = pdfmetrics.stringWidth(zc.char, font, font_size)

        if zc.main or zc.tone_mark:
            c.setFont(font, zy_size)
            letter_x = cursor_x + char_w + 1.0
            letter_y = y + font_size - zy_size
            for letter in zc.main:
                c.drawString(letter_x, letter_y, letter)
                letter_y -= zy_size * 1.05

            if zc.tone_mark:
                main_w = max(
                    (pdfmetrics.stringWidth(ch, font, zy_size) for ch in zc.main), default=0.0
                )
                tone_x = letter_x + main_w + 0.5
                if zc.tone_mark == "˙":
                    tone_y = y + font_size - zy_size
                elif zc.tone_mark == "ˊ":
                    tone_y = y + font_size - zy_size
                elif zc.tone_mark == "ˇ":
                    tone_y = y + (font_size - zy_size) / 2
                else:  # "ˋ"
                    tone_y = y
                c.drawString(tone_x, tone_y, zc.tone_mark)

        cursor_x += cell_w
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_zhuyin_render.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/stages/zhuyin_render.py tests/test_zhuyin_render.py
git commit -m "feat(pipeline): ✨ add Zhuyin wrapping and PDF drawing

wrap_zhuyin measures each character's total cell width (glyph plus
room for its phonetic column) and wraps character-by-character, the
same way stage3_assemble._wrap_to_width already handles plain CJK
text. draw_zhuyin_line renders the hanzi, its stacked main Bopomofo
letters, and the tone mark positioned beside the stack per tone
(confirmed layout from brainstorming mockups).

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Wire `zh-tw` + Zhuyin into `stage3_assemble.py`'s picture-book path

**Files:**
- Modify: `src/stages/stage3_assemble.py` (`_choose_layout_variant`, `_text_zone_height`, `_draw_zone_text`, `_draw_picture_book_page`, `build_picture_book_pdf`, `build_spread_pdf`, `run_stage3`)
- Test: `tests/test_stage3_assemble.py`

**Interfaces:**
- Consumes: `wrap_zhuyin`, `draw_zhuyin_line` (Task 4), `annotate` (Task 1), `stage1_result["language"]` (Task 3).
- Produces: every modified function above gains a `language: str = "en"` parameter (default preserves exact existing behavior for every current call site and test). `run_stage3` reads `stage1_result.get("language", "en")`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_stage3_assemble.py`:

```python
def test_choose_layout_variant_uses_zhuyin_wrapping_for_zh_tw():
    """A Hanzi string that fits in 2 plain-wrapped lines can need 3+ lines
    once Zhuyin annotation widens each character -- the decision must use
    the same wrapping the final draw will use, or the reserved zone could
    be too short. Verified numerically before writing this test: at
    TOP_BAND_FONT_SIZE against PAGE_W, this 33-character caption plain-wraps
    to 2 lines but Zhuyin-wraps to 3; at 20x the width it collapses to 1
    Zhuyin-wrapped line."""
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    caption = "陶樂蒂的開學日今天真是漂亮的一天大家都很開心呢" + "啊" * 10

    en_result = _choose_layout_variant([caption], "STSong-Light", PAGE_W, language="en")
    zh_result = _choose_layout_variant([caption], "STSong-Light", PAGE_W, language="zh-tw")
    zh_wide_result = _choose_layout_variant(
        [caption], "STSong-Light", PAGE_W * 20, language="zh-tw"
    )

    assert en_result == "bottom_caption"
    assert zh_result == "top_band"
    assert zh_wide_result == "bottom_caption"


def test_text_zone_height_larger_for_zhuyin_than_plain_at_same_width():
    from src.stages.stage3_assemble import _text_zone_height, TOP_BAND_FONT_SIZE, PAGE_W
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.pdfbase import pdfmetrics

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    caption = "陶樂蒂的開學日今天真是漂亮的一天大家都很開心呢" + "啊" * 10

    plain_h = _text_zone_height(
        [caption], "STSong-Light", TOP_BAND_FONT_SIZE, PAGE_W, language="en"
    )
    zhuyin_h = _text_zone_height(
        [caption], "STSong-Light", TOP_BAND_FONT_SIZE, PAGE_W, language="zh-tw"
    )

    assert zhuyin_h > plain_h


def test_build_picture_book_pdf_zh_tw_creates_file(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    pages = ["我不要上學！", "但是米亞不想動。", "世代相傳。"]
    out_path = str(tmp_path / "picture_book_zh.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_path, language="zh-tw")

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_picture_book_pdf_default_language_unaffected(tmp_path, tmp_images):
    """Omitting `language` must produce the same rendered content as
    language="en" -- English output takes the untouched plain-text path
    either way. Compares extracted text/page count via pypdf rather than
    raw bytes: reportlab embeds a unique per-build /ID in the PDF trailer
    (confirmed by direct testing -- two back-to-back builds from identical
    inputs differ at the trailer, nowhere in the actual content streams),
    so byte-for-byte comparison is never satisfiable even when nothing
    about the rendering differs."""
    import pypdf

    from src.stages.stage3_assemble import build_picture_book_pdf

    pages = ["Once upon a time.", "They had fun.", "The end."]
    out_no_lang = str(tmp_path / "no_lang.pdf")
    out_en = str(tmp_path / "en.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_no_lang)
    build_picture_book_pdf(tmp_images[:3], pages, out_en, language="en")

    reader_no_lang = pypdf.PdfReader(out_no_lang)
    reader_en = pypdf.PdfReader(out_en)

    assert len(reader_no_lang.pages) == len(reader_en.pages)
    assert [p.extract_text() for p in reader_no_lang.pages] == [
        p.extract_text() for p in reader_en.pages
    ]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage3_assemble.py -k "zhuyin or zh_tw" -v`
Expected: FAIL with `TypeError: _choose_layout_variant() got an unexpected keyword argument 'language'` (and similar for the other new tests).

- [ ] **Step 3: Add the import**

In `src/stages/stage3_assemble.py`, add to the existing local-import area at the top (after the `reportlab` imports, before `logger = logging.getLogger(__name__)`):

```python
from src.stages.zhuyin_render import draw_zhuyin_line, wrap_zhuyin
from src.utils.zhuyin import annotate
```

- [ ] **Step 4: Add a language-aware line-count helper**

Insert this new function immediately after `_wrap_to_width` (after line 102, before `_choose_layout_variant`):

```python
def _wrapped_line_count(text: str, font: str, font_size: float, max_width: float, language: str) -> int:
    """Number of lines `text` wraps to, using Zhuyin-aware wrapping for zh-tw
    (each character is wider once annotated) and plain wrapping otherwise."""
    if language == "zh-tw":
        return len(wrap_zhuyin(annotate(text), font, font_size, max_width))
    return len(_wrap_to_width(text, font, font_size, max_width))
```

- [ ] **Step 5: Make `_choose_layout_variant` language-aware**

Replace:

```python
def _choose_layout_variant(
    pages: list[str], font: str, page_width: float
) -> Literal["top_band", "bottom_caption"]:
    """Decide once per book: top-band for long captions, bottom-caption for short ones.

    Every page's caption is wrapped at `page_width` using the top-band font size as a
    fixed yardstick, regardless of which variant is ultimately chosen -- this keeps the
    decision a pure text-length measurement that never depends on the chosen variant.
    """
    max_width = page_width - 2 * MARGIN
    for page_text in pages:
        lines = _wrap_to_width(page_text, font, TOP_BAND_FONT_SIZE, max_width)
        if len(lines) > MAX_LINES_FOR_BOTTOM:
            return "top_band"
    return "bottom_caption"
```

with:

```python
def _choose_layout_variant(
    pages: list[str], font: str, page_width: float, language: str = "en"
) -> Literal["top_band", "bottom_caption"]:
    """Decide once per book: top-band for long captions, bottom-caption for short ones.

    Every page's caption is wrapped at `page_width` using the top-band font size as a
    fixed yardstick, regardless of which variant is ultimately chosen -- this keeps the
    decision a pure text-length measurement that never depends on the chosen variant.
    zh-tw captions use Zhuyin-aware wrapping, since each character is wider once
    annotated and would otherwise be under-measured.
    """
    max_width = page_width - 2 * MARGIN
    for page_text in pages:
        if _wrapped_line_count(page_text, font, TOP_BAND_FONT_SIZE, max_width, language) > MAX_LINES_FOR_BOTTOM:
            return "top_band"
    return "bottom_caption"
```

- [ ] **Step 6: Make `_text_zone_height` language-aware**

Replace:

```python
def _text_zone_height(pages: list[str], font: str, font_size: float, page_width: float) -> float:
    """Height of the text zone, sized to the longest wrapped caption across all pages.

    Computed once per book so every page reserves an identically sized band.
    """
    max_width = page_width - 2 * MARGIN
    line_height = font_size * 1.3
    max_lines = max(
        (len(_wrap_to_width(p, font, font_size, max_width)) or 1 for p in pages),
        default=1,
    )
    return 2 * TEXT_ZONE_PAD + max_lines * line_height
```

with:

```python
def _text_zone_height(
    pages: list[str], font: str, font_size: float, page_width: float, language: str = "en"
) -> float:
    """Height of the text zone, sized to the longest wrapped caption across all pages.

    Computed once per book so every page reserves an identically sized band.
    zh-tw captions use Zhuyin-aware wrapping (see _wrapped_line_count).
    """
    max_width = page_width - 2 * MARGIN
    line_height = font_size * 1.3
    max_lines = max(
        (_wrapped_line_count(p, font, font_size, max_width, language) or 1 for p in pages),
        default=1,
    )
    return 2 * TEXT_ZONE_PAD + max_lines * line_height
```

- [ ] **Step 7: Make `_draw_zone_text` language-aware**

Replace:

```python
def _draw_zone_text(
    c: canvas.Canvas,
    text: str,
    font: str,
    font_size: float,
    x: float,
    top_y: float,
    w: float,
    align: Literal["left", "center"],
) -> None:
    """Draw wrapped text with its first line's baseline just below `top_y`."""
    lines = _wrap_to_width(text, font, font_size, w) or [""]
    line_height = font_size * 1.3
    c.setFillColor(colors.black)
    c.setFont(font, font_size)
    cursor_y = top_y - font_size
    for line in lines:
        if align == "center":
            c.drawCentredString(x + w / 2, cursor_y, line)
        else:
            c.drawString(x, cursor_y, line)
        cursor_y -= line_height
```

with:

```python
def _draw_zone_text(
    c: canvas.Canvas,
    text: str,
    font: str,
    font_size: float,
    x: float,
    top_y: float,
    w: float,
    align: Literal["left", "center"],
    language: str = "en",
) -> None:
    """Draw wrapped text with its first line's baseline just below `top_y`.

    zh-tw text is drawn with Zhuyin annotation via zhuyin_render.py; every
    other language uses the plain wrap-and-draw path, unchanged.
    """
    line_height = font_size * 1.3
    cursor_y = top_y - font_size

    if language == "zh-tw":
        zhuyin_lines = wrap_zhuyin(annotate(text), font, font_size, w) or [[]]
        for zhuyin_line in zhuyin_lines:
            draw_zhuyin_line(c, zhuyin_line, font, font_size, x, cursor_y, w, align)
            cursor_y -= line_height
        return

    lines = _wrap_to_width(text, font, font_size, w) or [""]
    c.setFillColor(colors.black)
    c.setFont(font, font_size)
    for line in lines:
        if align == "center":
            c.drawCentredString(x + w / 2, cursor_y, line)
        else:
            c.drawString(x, cursor_y, line)
        cursor_y -= line_height
```

- [ ] **Step 8: Thread `language` through `_draw_picture_book_page`**

Replace:

```python
def _draw_picture_book_page(
    c: canvas.Canvas,
    img_path: str,
    page_text: str,
    font: str,
    variant: Literal["top_band", "bottom_caption"],
    text_h: float,
    page_w: float,
    page_h: float,
) -> None:
    """Draw one page/spread: a text zone (top or bottom) and a contain-fit image
    filling the rest, with a small gap so text never touches the art."""
    content_w = page_w - 2 * MARGIN
    if variant == "top_band":
        text_top_y = page_h - MARGIN
        _draw_zone_text(
            c, page_text, font, TOP_BAND_FONT_SIZE, MARGIN, text_top_y, content_w, "left"
        )
        image_h = page_h - MARGIN - text_h - INTER_ZONE_GAP - MARGIN
        _contain_fit_image(c, img_path, MARGIN, MARGIN, content_w, image_h)
    else:
        image_y = MARGIN + text_h + INTER_ZONE_GAP
        image_h = page_h - MARGIN - image_y
        _contain_fit_image(c, img_path, MARGIN, image_y, content_w, image_h)
        text_top_y = MARGIN + text_h
        _draw_zone_text(
            c, page_text, font, BOTTOM_CAPTION_FONT_SIZE, MARGIN, text_top_y, content_w, "center"
        )
```

with:

```python
def _draw_picture_book_page(
    c: canvas.Canvas,
    img_path: str,
    page_text: str,
    font: str,
    variant: Literal["top_band", "bottom_caption"],
    text_h: float,
    page_w: float,
    page_h: float,
    language: str = "en",
) -> None:
    """Draw one page/spread: a text zone (top or bottom) and a contain-fit image
    filling the rest, with a small gap so text never touches the art."""
    content_w = page_w - 2 * MARGIN
    if variant == "top_band":
        text_top_y = page_h - MARGIN
        _draw_zone_text(
            c, page_text, font, TOP_BAND_FONT_SIZE, MARGIN, text_top_y, content_w, "left", language
        )
        image_h = page_h - MARGIN - text_h - INTER_ZONE_GAP - MARGIN
        _contain_fit_image(c, img_path, MARGIN, MARGIN, content_w, image_h)
    else:
        image_y = MARGIN + text_h + INTER_ZONE_GAP
        image_h = page_h - MARGIN - image_y
        _contain_fit_image(c, img_path, MARGIN, image_y, content_w, image_h)
        text_top_y = MARGIN + text_h
        _draw_zone_text(
            c,
            page_text,
            font,
            BOTTOM_CAPTION_FONT_SIZE,
            MARGIN,
            text_top_y,
            content_w,
            "center",
            language,
        )
```

- [ ] **Step 9: Thread `language` through `build_picture_book_pdf` and `build_spread_pdf`**

Replace:

```python
def build_picture_book_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
    page_size: tuple[float, float] = A4,
) -> None:
    """Build a picture-book PDF: image contain-fit into a reserved zone, caption text
    in a plain-background zone outside the image that never overlaps it. The
    top-band-vs-bottom-caption choice and the text zone's height are both computed
    once per book, so every page/spread reserves an identically sized, positioned zone.
    """
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    font = _resolve_cjk_font(font_path)
    page_w, page_h = page_size
    variant = _choose_layout_variant(pages, font, page_w)
    font_size = TOP_BAND_FONT_SIZE if variant == "top_band" else BOTTOM_CAPTION_FONT_SIZE
    text_h = _text_zone_height(pages, font, font_size, page_w)

    c = canvas.Canvas(output_path, pagesize=page_size)
    for img_path, page_text in zip(illustration_paths, pages):
        _draw_picture_book_page(c, img_path, page_text, font, variant, text_h, page_w, page_h)
        c.showPage()
    c.save()


def build_spread_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
) -> None:
    """Build a picture-book PDF where each page is a double-page spread (one wide
    illustration spanning two A4 widths at one A4 height). Reuses the same
    top-band/bottom-caption geometry as `build_picture_book_pdf`, re-measured at
    the spread's doubled width.
    """
    build_picture_book_pdf(
        illustration_paths,
        pages,
        output_path,
        font_path,
        page_size=(SPREAD_PAGE_W, SPREAD_PAGE_H),
    )
```

with:

```python
def build_picture_book_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
    page_size: tuple[float, float] = A4,
    language: str = "en",
) -> None:
    """Build a picture-book PDF: image contain-fit into a reserved zone, caption text
    in a plain-background zone outside the image that never overlaps it. The
    top-band-vs-bottom-caption choice and the text zone's height are both computed
    once per book, so every page/spread reserves an identically sized, positioned zone.
    zh-tw captions are annotated with Zhuyin (see zhuyin_render.py); every other
    language takes the unchanged plain-text path.
    """
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    font = _resolve_cjk_font(font_path)
    page_w, page_h = page_size
    variant = _choose_layout_variant(pages, font, page_w, language)
    font_size = TOP_BAND_FONT_SIZE if variant == "top_band" else BOTTOM_CAPTION_FONT_SIZE
    text_h = _text_zone_height(pages, font, font_size, page_w, language)

    c = canvas.Canvas(output_path, pagesize=page_size)
    for img_path, page_text in zip(illustration_paths, pages):
        _draw_picture_book_page(
            c, img_path, page_text, font, variant, text_h, page_w, page_h, language
        )
        c.showPage()
    c.save()


def build_spread_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
    language: str = "en",
) -> None:
    """Build a picture-book PDF where each page is a double-page spread (one wide
    illustration spanning two A4 widths at one A4 height). Reuses the same
    top-band/bottom-caption geometry as `build_picture_book_pdf`, re-measured at
    the spread's doubled width.
    """
    build_picture_book_pdf(
        illustration_paths,
        pages,
        output_path,
        font_path,
        page_size=(SPREAD_PAGE_W, SPREAD_PAGE_H),
        language=language,
    )
```

- [ ] **Step 10: Wire `run_stage3` to read and pass `language`**

Replace:

```python
def run_stage3(
    stage1_result: dict,
    stage2_result: dict,
    descriptions: list[str],
    config: dict,
    output_path: str,
) -> dict:
    """Returns: {"pdf_path": str}"""
    layout = config.get("page_layout", "image_top_text_bottom")
    if layout == "picture_book":
        build_picture_book_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            output_path,
            config.get("font_path"),
        )
    elif layout == "picture_book_spread":
        build_spread_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            output_path,
            config.get("font_path"),
        )
    else:
        build_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            descriptions,
            output_path,
        )
    return {"pdf_path": output_path}
```

with:

```python
def run_stage3(
    stage1_result: dict,
    stage2_result: dict,
    descriptions: list[str],
    config: dict,
    output_path: str,
) -> dict:
    """Returns: {"pdf_path": str}"""
    layout = config.get("page_layout", "image_top_text_bottom")
    language = stage1_result.get("language", "en")
    if layout == "picture_book":
        build_picture_book_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            output_path,
            config.get("font_path"),
            language=language,
        )
    elif layout == "picture_book_spread":
        build_spread_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            output_path,
            config.get("font_path"),
            language=language,
        )
    else:
        build_pdf(
            stage2_result["illustration_paths"],
            stage1_result["pages"],
            descriptions,
            output_path,
        )
    return {"pdf_path": output_path}
```

- [ ] **Step 11: Run the full test suite**

Run: `pytest tests/ -v --ignore=tests/test_model_manager.py --ignore=tests/test_pipeline.py`
Expected: all PASS.

- [ ] **Step 12: Commit**

```bash
git add src/stages/stage3_assemble.py tests/test_stage3_assemble.py
git commit -m "feat(pipeline): ✨ render Zhuyin annotation for zh-tw picture books

_choose_layout_variant, _text_zone_height, and _draw_zone_text all
gain a language parameter (default \"en\", preserving every existing
call site's behavior exactly). zh-tw text uses Zhuyin-aware wrapping
and drawing throughout -- both the top-band/bottom-caption decision
and the reserved text-zone height now correctly account for the extra
width each character needs once annotated, not just the final draw
call, avoiding clipped/overlapping text.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage:**
- Decision 1 (CJK font fix) → Task 2.
- Decision 2 (always-on for zh-tw) → Task 5 (`language == "zh-tw"` check, no config toggle added anywhere).
- Decision 3 (scope: zh-tw only, picture_book/picture_book_spread only) → satisfied by construction (no `zh-cn` branch added; `build_pdf`/legacy layout untouched in every task).
- Decision 4 (layout: wider per-character gap, stacked main letters, tone mark beside per rule) → Task 4's `_cell_width`/`draw_zhuyin_line`.
- Decision 5 (trust pypinyin's heteronym default) → Task 1's `annotate()` uses `heteronym=False` with no extra disambiguation logic.
- Architecture's `src/utils/zhuyin.py` → Task 1. `src/stages/zhuyin_render.py` → Task 4. `stage3_assemble.py` changes → Tasks 2 and 5. `stage1_story.py` → Task 3. `requirements.txt` → Task 1.
- **Spec correction** (the design doc's "`_text_zone_height` unaffected" claim was wrong): documented in Global Constraints above and fixed by Task 5 making `_text_zone_height`/`_choose_layout_variant` language-aware, not just `_draw_zone_text`.
- **Alignment bug found during planning** (not in the spec, discovered by direct testing): naive `zip(text, pinyin(...))` misaligns as soon as non-Hanzi text is grouped by pypinyin into multi-character passthrough runs (confirmed: `'我愛Hi world的開學日123'` → 17 characters but only 8 pinyin entries). Task 1's `annotate()` walks the original text by consumed length instead of a naive zip, fixing this before it ships.

**Placeholder scan:** none found — every step has complete, runnable code or an exact command.

**Type consistency:** `ZhuyinChar` (Task 1: `char`, `main`, `tone_mark`) is consumed identically in Task 4's `_cell_width`/`draw_zhuyin_line` and Task 5's `_draw_zone_text`. `wrap_zhuyin`/`draw_zhuyin_line`'s signatures in Task 4 match their call sites in Task 5 exactly (parameter order: `c, line/zchars, font, font_size, x, y/max_width, w, align`). Every function gaining `language: str = "en"` in Task 5 defaults identically, so no call site outside this plan (e.g. `pipeline.py`, which is never modified) needs to change.
