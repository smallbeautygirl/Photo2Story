# Picture-Book Spread Layout & Continuity Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the never-shipped 2026-07-01 single-page `picture_book` layout as a foundation, then extend it to a double-page-spread architecture per the 2026-07-08 spec, and add the reference-grounded narrative-continuity evaluation that is this work's thesis contribution.

**Architecture:** `stage3_assemble.py` gains a shared contain-fit/text-zone geometry engine used by two public entry points — `build_picture_book_pdf` (single A4 page, the 2026-07-01 design) and `build_spread_pdf` (double-wide A4 spread, the 2026-07-08 extension) — selected via a new `picture_book` / `picture_book_spread` config value. `illustrate_fal.py` switches its FLUX aspect ratio to a landscape spread size and injects a per-book character/setting reference string into every prompt. `stage1_story.py`'s continuity instruction is sharpened with concrete techniques. Two new evaluation scripts (`src/eval/clip_score.py`, `src/eval/narrative_continuity_judge.py`) score pipeline output against the taxonomy calibrated in `docs/eval/reference_continuity_calibration.md`.

**Tech Stack:** Python 3, ReportLab (PDF), fal.ai FLUX.1-dev (illustration), Vertex AI Gemini (`src/utils/gemini_client.py`), open_clip (CLIPScore), pytest + pytest-mock.

## Global Constraints

- `from __future__ import annotations` at the top of every new/modified Python file.
- Built-in generics only: `list[str]`, `dict[str, float]`, `X | None` — never `List`/`Dict`/`Optional` from `typing` (`Literal` is the one exception already established in this codebase's `config.py` and is fine to keep using).
- f-strings for all string interpolation; `pathlib.Path` for all file paths.
- No `print()` — use `logging.getLogger(__name__)` with `extra={...}` kwargs, per `.claude/rules/logging.md`. Never log full LLM prompts/responses at INFO; DEBUG or omit.
- Mock all external APIs (fal, Gemini, open_clip/torch) in tests — no real HTTP calls, per `.claude/rules/testing.md`. Follow this repo's existing mocking idiom: `mocker.patch("module.path.function", return_value=...)` for Gemini calls (see `tests/test_stage1_story.py`), `patch.dict(sys.modules, {"open_clip": ...})` for CLIP (see `tests/test_stage0_select.py`), `monkeypatch.setattr("fal_client.run", ...)` for fal (see `tests/test_illustrate_fal.py`).
- Test files stay flat in `tests/` (matching this repo's existing convention, not `tests/<subpackage>/`).
- Ruff-compatible formatting: double quotes, 100-char line length.
- Commits follow Conventional Commits with this repo's actual established scopes (`pipeline` for stage/eval code, `config` for `config.py`/YAML, `docs` for markdown) — see `604bebf feat(pipeline): ✨ add FLUX.1 hosted backend and full-bleed caption layout` for direct precedent. End every commit body with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.

---

## Part A — Ship the 2026-07-01 `picture_book` layout (prerequisite, never implemented)

### Task 1: Rename `full_bleed_caption` → `picture_book`, add `picture_book_spread`

**Files:**
- Modify: `src/config.py:32-34`
- Modify: `configs/demo.yaml:24`
- Modify: `docs/storybook_improvements.md:90-118, 122-138`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `Stage3Config.page_layout` accepts `"image_top_text_bottom" | "picture_book" | "picture_book_spread"`. Tasks 3 and 4 dispatch on these values in `run_stage3`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (read the file first to match its existing style before appending):

```python
def test_stage3_accepts_picture_book_layout(tmp_path):
    import yaml
    from src.config import load_config

    cfg = {
        "stage0": {"mode": "random", "clip_model": "x", "clip_pretrained": "x",
                   "vlm_model": "x", "llm_model": "x", "k": 2},
        "stage1": {"model": "x", "context_mode": "none", "use_causal_inference": False},
        "stage2": {"base_model": "x", "use_ipadapter": False, "use_stylealigned": False,
                   "style_image_path": None},
        "stage3": {"output_format": "pdf", "page_layout": "picture_book_spread"},
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.dump(cfg))

    loaded = load_config(p)

    assert loaded["stage3"]["page_layout"] == "picture_book_spread"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_stage3_accepts_picture_book_layout -v`
Expected: this specific test actually passes already (the `Literal` type hint isn't enforced at runtime by `load_config`/`_validate`, which doesn't check `page_layout` at all). Confirm this by running it — it should PASS even before Step 3. This step exists to document that runtime behavior is unaffected by the type-hint-only rename in Step 3; the real regression check is Step 4.

- [ ] **Step 3: Update the type hint and config files**

In `src/config.py`, replace line 34:

```python
    page_layout: Literal["image_top_text_bottom", "full_bleed_caption"]
```

with:

```python
    page_layout: Literal["image_top_text_bottom", "picture_book", "picture_book_spread"]
```

In `configs/demo.yaml`, replace line 24:

```yaml
  page_layout: full_bleed_caption
```

with:

```yaml
  page_layout: picture_book_spread
```

In `docs/storybook_improvements.md`, replace lines 90-118:

```markdown
### 做法
新增 **full-bleed caption** 排版，取代舊的「image top / text bottom」：

- `_draw_full_bleed_image()`：把插圖等比例放大到**滿版覆蓋整頁**（多餘部分裁切到頁緣外）。
- `_draw_caption()`：在頁面底部畫一條**半透明黑色遮罩 (scrim)**，文字以白字置中疊在上面，確保任何底圖都讀得到字。

對應程式碼：[`src/stages/stage3_assemble.py`](../src/stages/stage3_assemble.py)
- `build_caption_pdf()`：滿版圖 + caption 疊字的新排版
- `build_pdf()`：保留舊的「上圖下字」排版（供 ablation baseline 對照）
- `run_stage3()` 依 `config.page_layout` 切換：`full_bleed_caption` vs `image_top_text_bottom`

排版常數（可調）：

| 常數 | 預設 | 意義 |
|---|---|---|
| `CAPTION_FONT_SIZE` | 18 | 疊字字級 |
| `CAPTION_LINE_HEIGHT` | 26 | 行高 |
| `SCRIM_ALPHA` | 0.5 | 底部遮罩透明度（越高字越清楚、圖越被壓暗） |
| `CAPTION_PAD_X / Y` | 1.5cm / 0.7cm | 文字與頁緣留白 |

```yaml
stage3:
  page_layout: full_bleed_caption   # 滿版插圖 + 底部疊字
```

> **後續可延伸**：目前疊字固定在底部 scrim。若要更接近繪本，可再支援
> 「對話框 / 特定圖案內的文字」——例如依場景把 caption 放到圖中留白區、或畫成對話泡泡。
> 這會是排版的下一步，但不在本次三項調整範圍內。
```

with:

```markdown
### 做法
新增 **picture_book** 排版，取代舊的「image top / text bottom」與已棄用的 full-bleed scrim 設計：

- `_contain_fit_image()`：把插圖等比例縮放到**留白區內完整顯示**（不裁切，置中）。
- 頁面切成**文字區**與**圖片區**兩塊；文字區高度整本書只算一次（取全書最長的斷行結果），確保每頁文字區大小、圖片位置一致。
- 不使用半透明遮罩：文字直接畫在素色背景上，因為文字區與圖片區不重疊。
- 文字區在上（top-band，~20pt，左靠）或在下（bottom-caption，~15pt，置中）由**全書**的 caption 長度決定，見 `_choose_layout_variant()`。

對應程式碼：[`src/stages/stage3_assemble.py`](../src/stages/stage3_assemble.py)
- `build_picture_book_pdf()`：單頁 picture_book 排版
- `build_spread_pdf()`：雙頁跨頁（spread）版本，共用同一套幾何邏輯，只是寬度加倍
- `build_pdf()`：保留舊的「上圖下字」排版（供 ablation baseline 對照）
- `run_stage3()` 依 `config.page_layout` 切換：`picture_book` / `picture_book_spread` / `image_top_text_bottom`

排版常數（可調）：

| 常數 | 預設 | 意義 |
|---|---|---|
| `TOP_BAND_FONT_SIZE` | 20 | top-band 字級 |
| `BOTTOM_CAPTION_FONT_SIZE` | 15 | bottom-caption 字級 |
| `MAX_LINES_FOR_BOTTOM` | 2 | 超過幾行就改用 top-band |
| `TEXT_ZONE_PAD` | 0.6cm | 文字區內邊距 |
| `INTER_ZONE_GAP` | 0.3cm | 文字區與圖片區的間隔 |

```yaml
stage3:
  page_layout: picture_book_spread   # 雙頁跨頁：滿版寬幅插圖 + 版位固定的文字區
```
```

In `docs/storybook_improvements.md`, replace lines 122-138:

```markdown
## 總結：一份 demo 設定就能開啟三項調整

```yaml
stage1:
  reading_level: simple             # ① 故事：一頁一句、用字最簡單
stage2:
  backend: fal                      # ② 畫風：4 種大師風格 (ghibli/pixar/disney/crayon) + FLUX LoRA
  use_ipadapter: true
stage3:
  page_layout: full_bleed_caption   # ③ 排版：滿版插圖 + 文字疊圖
```

| 調整 | 核心檔案 | 控制開關 |
|---|---|---|
| ① 故事內容 | `prompt_templates.py` / `stage1_story.py` | `stage1.reading_level` |
| ② 畫風 | `styles.py` / `illustrate_fal.py` | `style` 參數 + `stage2.backend: fal` |
| ③ Layout | `stage3_assemble.py` | `stage3.page_layout` |
```

with:

```markdown
## 總結：一份 demo 設定就能開啟三項調整

```yaml
stage1:
  reading_level: simple             # ① 故事：一頁一句、用字最簡單
stage2:
  backend: fal                      # ② 畫風：4 種大師風格 (ghibli/pixar/disney/crayon) + FLUX LoRA
  use_ipadapter: true
stage3:
  page_layout: picture_book_spread  # ③ 排版：雙頁跨頁 + 版位固定的文字區
```

| 調整 | 核心檔案 | 控制開關 |
|---|---|---|
| ① 故事內容 | `prompt_templates.py` / `stage1_story.py` | `stage1.reading_level` |
| ② 畫風 | `styles.py` / `illustrate_fal.py` | `style` 參數 + `stage2.backend: fal` |
| ③ Layout | `stage3_assemble.py` | `stage3.page_layout` |
```

- [ ] **Step 4: Run the full test suite to check for regressions**

Run: `pytest tests/ -v`
Expected: PASS (nothing currently depends on the literal string `"full_bleed_caption"` — confirmed by `grep -rn "full_bleed_caption" tests/` returning nothing).

- [ ] **Step 5: Commit**

```bash
git add src/config.py configs/demo.yaml docs/storybook_improvements.md tests/test_config.py
git commit -m "refactor(config): ♻️ rename full_bleed_caption to picture_book

Renames the stage3 layout config value ahead of implementing the
2026-07-01 picture_book design, and adds picture_book_spread for the
2026-07-08 double-page-spread extension.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 2: `_choose_layout_variant` — pure layout decision function

**Files:**
- Modify: `src/stages/stage3_assemble.py:1-2` (imports), `src/stages/stage3_assemble.py:28-33` (constants)
- Test: `tests/test_stage3_assemble.py`

**Interfaces:**
- Consumes: `_wrap_to_width(text: str, font: str, size: float, max_width: float) -> list[str]` (already exists at `stage3_assemble.py:76-97`, unchanged).
- Produces: `_choose_layout_variant(pages: list[str], font: str, page_width: float) -> Literal["top_band", "bottom_caption"]`. Task 3 calls this to decide `build_picture_book_pdf`'s layout; Task 4 calls it again at spread width.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage3_assemble.py`:

```python
def test_choose_layout_variant_picks_top_band_for_long_captions():
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W

    long_caption = (
        "This is a very long caption that will definitely need more than two "
        "lines when wrapped at the page width because it just keeps going and "
        "going and going."
    )
    result = _choose_layout_variant(["Short one.", long_caption], "Helvetica", PAGE_W)
    assert result == "top_band"


def test_choose_layout_variant_picks_bottom_caption_for_short_captions():
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W

    result = _choose_layout_variant(["Short one.", "Another short one."], "Helvetica", PAGE_W)
    assert result == "bottom_caption"


def test_choose_layout_variant_uses_spread_width():
    """The same long caption that top-bands at page width must bottom-caption
    at a much wider width, proving the decision is measured against the
    page_width argument rather than a hardcoded constant."""
    from src.stages.stage3_assemble import _choose_layout_variant, PAGE_W

    long_caption = (
        "This is a long caption that will wrap to several lines at normal "
        "page width because it just keeps going on and on with many more words."
    )
    narrow_result = _choose_layout_variant([long_caption], "Helvetica", PAGE_W)
    very_wide_result = _choose_layout_variant([long_caption], "Helvetica", PAGE_W * 10)

    assert narrow_result == "top_band"
    assert very_wide_result == "bottom_caption"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage3_assemble.py -k choose_layout_variant -v`
Expected: FAIL with `ImportError: cannot import name '_choose_layout_variant'`

- [ ] **Step 3: Add the `Literal` import and new constants**

In `src/stages/stage3_assemble.py`, replace line 1-4:

```python
from __future__ import annotations

import logging
from pathlib import Path
```

with:

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal
```

Replace lines 28-33:

```python
# --- Full-bleed caption layout constants ---
CAPTION_FONT_SIZE = 18
CAPTION_LINE_HEIGHT = 26
CAPTION_PAD_X = 1.5 * cm
CAPTION_PAD_Y = 0.7 * cm
SCRIM_ALPHA = 0.5
```

with:

```python
# --- picture_book layout constants (shared by single-page and spread variants) ---
TOP_BAND_FONT_SIZE = 20
BOTTOM_CAPTION_FONT_SIZE = 15
MAX_LINES_FOR_BOTTOM = 2
TEXT_ZONE_PAD = 0.6 * cm
INTER_ZONE_GAP = 0.3 * cm

# --- picture_book_spread geometry: one PDF page per spread (two A4 widths, one A4 height) ---
SPREAD_PAGE_W = 2 * PAGE_W
SPREAD_PAGE_H = PAGE_H
```

- [ ] **Step 4: Add `_choose_layout_variant`**

In `src/stages/stage3_assemble.py`, insert immediately after `_wrap_to_width` (after line 97, before the old `_draw_full_bleed_image` at line 100):

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

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_stage3_assemble.py -k choose_layout_variant -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/stages/stage3_assemble.py tests/test_stage3_assemble.py
git commit -m "feat(pipeline): ✨ add _choose_layout_variant for picture_book layout

Pure function deciding top-band vs bottom-caption once per book,
parameterized on page width so the spread variant (Task 4) can reuse
it at double width.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 3: `build_picture_book_pdf` — single-page contain-fit layout

**Files:**
- Modify: `src/stages/stage3_assemble.py:100-147` (delete legacy full-bleed functions, add new ones)
- Modify: `src/stages/stage3_assemble.py` `run_stage3` (dispatch)
- Test: `tests/test_stage3_assemble.py`

**Interfaces:**
- Consumes: `_choose_layout_variant` (Task 2), `_wrap_to_width`, `_resolve_cjk_font` (existing).
- Produces: `_contain_fit_image(c, img_path, x, y, w, h) -> None`, `_text_zone_height(pages, font, font_size, page_width) -> float`, `_draw_zone_text(c, text, font, font_size, x, top_y, w, align) -> None`, `_draw_picture_book_page(c, img_path, page_text, font, variant, text_h, page_w, page_h) -> None`, `build_picture_book_pdf(illustration_paths, pages, output_path, font_path=None, page_size=A4) -> None`. Task 4's `build_spread_pdf` calls `build_picture_book_pdf` with a different `page_size`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage3_assemble.py`:

```python
def test_text_zone_height_scales_with_longest_caption():
    from src.stages.stage3_assemble import _text_zone_height, TOP_BAND_FONT_SIZE, PAGE_W

    short_only = _text_zone_height(["Hi.", "Bye."], "Helvetica", TOP_BAND_FONT_SIZE, PAGE_W)
    with_long = _text_zone_height(
        ["Hi.", "This is a much longer caption that wraps across several lines of text."],
        "Helvetica",
        TOP_BAND_FONT_SIZE,
        PAGE_W,
    )
    assert with_long > short_only


def test_build_picture_book_pdf_creates_file_bottom_caption(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    pages = ["Once upon a time.", "They had fun.", "The end."]
    out_path = str(tmp_path / "picture_book_bottom.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_picture_book_pdf_creates_file_top_band(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    long_caption = (
        "This is a very long caption that will definitely need more than two "
        "lines when wrapped at the page width because it just keeps going and "
        "going."
    )
    pages = [long_caption, "Short.", "The end."]
    out_path = str(tmp_path / "picture_book_top.pdf")

    build_picture_book_pdf(tmp_images[:3], pages, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_picture_book_pdf_wrong_count_raises(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_picture_book_pdf

    with pytest.raises(AssertionError):
        build_picture_book_pdf(tmp_images[:2], ["Only one page."], str(tmp_path / "bad.pdf"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage3_assemble.py -k picture_book_pdf -v`
Expected: FAIL with `ImportError: cannot import name 'build_picture_book_pdf'`

- [ ] **Step 3: Delete the legacy full-bleed/scrim functions**

In `src/stages/stage3_assemble.py`, delete these three functions in full (originally at lines 100-147 before Task 2's edits shifted line numbers — locate by name, not line number, since Task 2 already inserted code above them):

```python
def _draw_full_bleed_image(c: canvas.Canvas, img_path: str) -> None:
    """Scale the illustration to cover the whole page (cropping overflow off the page edges)."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    scale = max(PAGE_W / img_w, PAGE_H / img_h)
    draw_w, draw_h = img_w * scale, img_h * scale
    x = (PAGE_W - draw_w) / 2
    y = (PAGE_H - draw_h) / 2
    c.drawImage(ImageReader(img), x, y, width=draw_w, height=draw_h)


def _draw_caption(c: canvas.Canvas, text: str, font: str) -> None:
    """Draw the story text in a semi-transparent scrim across the bottom of the page."""
    max_width = PAGE_W - 2 * CAPTION_PAD_X
    lines = _wrap_to_width(text, font, CAPTION_FONT_SIZE, max_width) or [""]
    scrim_h = 2 * CAPTION_PAD_Y + len(lines) * CAPTION_LINE_HEIGHT

    c.setFillColor(colors.black)
    c.setFillAlpha(SCRIM_ALPHA)
    c.rect(0, 0, PAGE_W, scrim_h, fill=1, stroke=0)

    c.setFillAlpha(1)
    c.setFillColor(colors.white)
    c.setFont(font, CAPTION_FONT_SIZE)
    y = scrim_h - CAPTION_PAD_Y - CAPTION_FONT_SIZE
    for line in lines:
        c.drawCentredString(PAGE_W / 2, y, line)
        y -= CAPTION_LINE_HEIGHT


def build_caption_pdf(
    illustration_paths: list[str],
    pages: list[str],
    output_path: str,
    font_path: str | None = None,
) -> None:
    """Build a picture-book PDF: each page is a full-bleed illustration with the story
    text overlaid in a caption band, the way a real storybook reads."""
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    font = _resolve_cjk_font(font_path)
    c = canvas.Canvas(output_path, pagesize=A4)
    for img_path, page_text in zip(illustration_paths, pages):
        _draw_full_bleed_image(c, img_path)
        _draw_caption(c, page_text, font)
        c.showPage()
    c.save()
```

- [ ] **Step 4: Add the contain-fit geometry engine and `build_picture_book_pdf`**

In their place, add:

```python
def _contain_fit_image(c: canvas.Canvas, img_path: str, x: float, y: float, w: float, h: float) -> None:
    """Scale the illustration to fit within (w, h) without cropping, centered in the zone."""
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    scale = min(w / img_w, h / img_h)
    draw_w, draw_h = img_w * scale, img_h * scale
    draw_x = x + (w - draw_w) / 2
    draw_y = y + (h - draw_h) / 2
    c.drawImage(ImageReader(img), draw_x, draw_y, width=draw_w, height=draw_h)


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
        _draw_zone_text(c, page_text, font, TOP_BAND_FONT_SIZE, MARGIN, text_top_y, content_w, "left")
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
```

- [ ] **Step 5: Wire `run_stage3` to dispatch to `picture_book`**

Replace the `run_stage3` function body's dispatch:

```python
    layout = config.get("page_layout", "image_top_text_bottom")
    if layout == "full_bleed_caption":
        build_caption_pdf(
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
    layout = config.get("page_layout", "image_top_text_bottom")
    if layout == "picture_book":
        build_picture_book_pdf(
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

(Task 4 adds the `picture_book_spread` branch to this same `if`/`elif`/`else`.)

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_stage3_assemble.py -v`
Expected: all PASS, including the pre-existing `test_build_pdf_creates_file` and `test_build_pdf_wrong_count_raises` (untouched — `build_pdf` was not modified).

- [ ] **Step 7: Commit**

```bash
git add src/stages/stage3_assemble.py tests/test_stage3_assemble.py
git commit -m "feat(pipeline): ✨ implement picture_book contain-fit layout

Ships the 2026-07-01 design: illustrations are contain-fit into a
reserved zone instead of cropped full-bleed, text lives in a plain
zone outside the image sized once per book, no scrim. Replaces
build_caption_pdf entirely.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Part B — Extend to double-page spreads (2026-07-08)

### Task 4: `build_spread_pdf` — double-wide spread layout

**Files:**
- Modify: `src/stages/stage3_assemble.py` (`run_stage3` dispatch)
- Test: `tests/test_stage3_assemble.py`

**Interfaces:**
- Consumes: `build_picture_book_pdf(..., page_size=...)` (Task 3), `SPREAD_PAGE_W`, `SPREAD_PAGE_H` (Task 2).
- Produces: `build_spread_pdf(illustration_paths, pages, output_path, font_path=None) -> None`. `run_stage3` dispatches `"picture_book_spread"` to it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage3_assemble.py`:

```python
def test_build_spread_pdf_creates_file(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_spread_pdf

    pages = ["Once upon a time.", "They had fun.", "The end."]
    out_path = str(tmp_path / "spread.pdf")

    build_spread_pdf(tmp_images[:3], pages, out_path)

    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000


def test_build_spread_pdf_wrong_count_raises(tmp_path, tmp_images):
    from src.stages.stage3_assemble import build_spread_pdf

    with pytest.raises(AssertionError):
        build_spread_pdf(tmp_images[:2], ["Only one page."], str(tmp_path / "bad.pdf"))


def test_build_spread_pdf_uses_double_width_page(tmp_path, tmp_images):
    """A spread PDF's page must be twice the width of a single picture_book page."""
    import pypdf

    from src.stages.stage3_assemble import PAGE_W, build_spread_pdf

    out_path = str(tmp_path / "spread_size.pdf")
    build_spread_pdf(tmp_images[:2], ["One.", "Two."], out_path)

    reader = pypdf.PdfReader(out_path)
    page_width_pt = float(reader.pages[0].mediabox.width)
    assert abs(page_width_pt - 2 * PAGE_W) < 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage3_assemble.py -k spread_pdf -v`
Expected: FAIL with `ImportError: cannot import name 'build_spread_pdf'` (and possibly `ModuleNotFoundError: No module named 'pypdf'` — if so, run `pip install pypdf` first; it is a test-only dependency for verifying page geometry, add it to `requirements.txt` under a `# test-only` comment if not already present).

- [ ] **Step 3: Add `pypdf` to requirements if missing**

Check: `grep pypdf requirements.txt`. If absent, add this line to `requirements.txt` (keep alphabetical-ish grouping with the other non-GPU deps):

```
pypdf>=4.2.0
```

- [ ] **Step 4: Implement `build_spread_pdf`**

In `src/stages/stage3_assemble.py`, add immediately after `build_picture_book_pdf`:

```python
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

- [ ] **Step 5: Wire `run_stage3`'s `picture_book_spread` branch**

Replace the dispatch block written in Task 3 Step 5:

```python
    layout = config.get("page_layout", "image_top_text_bottom")
    if layout == "picture_book":
        build_picture_book_pdf(
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

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_stage3_assemble.py -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/stages/stage3_assemble.py requirements.txt
git commit -m "feat(pipeline): ✨ add build_spread_pdf for double-page spreads

Reuses build_picture_book_pdf's contain-fit/text-zone geometry at
double width instead of duplicating layout logic, per the 2026-07-08
spread architecture spec.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 5: Landscape spread aspect ratio in `illustrate_fal.py`

**Files:**
- Modify: `src/stages/illustrate_fal.py:37` (constant), `illustrate_fal.py:108` (usage)
- Test: `tests/test_illustrate_fal.py`

**Interfaces:**
- Produces: `SPREAD_IMAGE_SIZE = {"width": 1408, "height": 992}` module constant, used as `generate_illustrations_fal`'s `image_size` argument (replacing `IMAGE_SIZE = "square_hd"`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_illustrate_fal.py`:

```python
def test_landscape_spread_aspect_ratio_used(fake_fal, tmp_path):
    generate_illustrations_fal(["a baby in a tub"], "crayon", {}, str(tmp_path))

    _, arguments = fake_fal[0]
    assert arguments["image_size"] == illustrate_fal.SPREAD_IMAGE_SIZE
    assert illustrate_fal.SPREAD_IMAGE_SIZE["width"] > illustrate_fal.SPREAD_IMAGE_SIZE["height"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_illustrate_fal.py -k landscape -v`
Expected: FAIL with `AttributeError: module 'src.stages.illustrate_fal' has no attribute 'SPREAD_IMAGE_SIZE'`

- [ ] **Step 3: Replace the square aspect ratio constant**

In `src/stages/illustrate_fal.py`, replace line 37:

```python
IMAGE_SIZE = "square_hd"
```

with:

```python
# Landscape ratio matching an A4 spread (two 210mm-wide portrait pages at 297mm
# height, ~1.4141:1) so one FLUX call renders a full spread instead of one page.
SPREAD_IMAGE_SIZE = {"width": 1408, "height": 992}
```

In `src/stages/illustrate_fal.py`, replace line 108:

```python
            "image_size": IMAGE_SIZE,
```

with:

```python
            "image_size": SPREAD_IMAGE_SIZE,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_illustrate_fal.py -v`
Expected: all PASS (existing tests don't assert on `IMAGE_SIZE`, so nothing else breaks).

- [ ] **Step 5: Commit**

```bash
git add src/stages/illustrate_fal.py tests/test_illustrate_fal.py
git commit -m "feat(pipeline): ✨ switch FLUX output to landscape spread ratio

Each illustration now renders as a full double-page spread
(~1.414:1 landscape) instead of a square single page, matching the
picture_book_spread PDF layout.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 6: Cross-spread character/setting continuity in illustration prompts

**Files:**
- Modify: `src/stages/illustrate_fal.py` (`_build_prompt`, `generate_illustrations_fal`)
- Test: `tests/test_illustrate_fal.py`

**Interfaces:**
- Produces: `_build_character_reference(descriptions: list[str]) -> str`, `_build_prompt(scene: str, preset: StylePreset, character_ref: str = "") -> str` (signature change — adds optional `character_ref` param, backward compatible since it defaults to `""`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_illustrate_fal.py`:

```python
def test_build_character_reference_dedupes_and_caps():
    from src.stages.illustrate_fal import CHARACTER_REF_WORD_CAP, _build_character_reference

    scenes = ["a red bike"] * 3 + [" ".join(f"word{i}" for i in range(100))]
    ref = _build_character_reference(scenes)

    assert ref.count("a red bike") == 1
    assert len(ref.split()) <= CHARACTER_REF_WORD_CAP


def test_prompt_includes_character_reference_from_other_pages(fake_fal, tmp_path):
    scenes = [
        "a girl in a red hat playing on a swing",
        "a girl in a red hat eating ice cream",
    ]
    generate_illustrations_fal(scenes, "crayon", {}, str(tmp_path))

    page_2_prompt = fake_fal[1][1]["prompt"]
    assert "eating ice cream" in page_2_prompt
    assert "playing on a swing" in page_2_prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_illustrate_fal.py -k character_reference -v`
Expected: FAIL with `ImportError: cannot import name '_build_character_reference'`

- [ ] **Step 3: Implement `_build_character_reference` and thread it through `_build_prompt`**

In `src/stages/illustrate_fal.py`, add near the top constants (after `LORA_SCALE = 1.0`):

```python
# Character/setting continuity: cap how much of the combined per-photo
# descriptions gets folded into every spread's prompt, so it stays a light
# continuity nudge rather than crowding out that spread's own scene.
CHARACTER_REF_WORD_CAP = 60
```

Add this function just above `_build_prompt`:

```python
def _build_character_reference(descriptions: list[str]) -> str:
    """Combine all photo descriptions into one continuity string, deduped and
    capped, so every spread's prompt carries the same characters/setting.

    No new LLM call: reuses the per-photo descriptions already produced by
    stage0_select and passed in as `scenes`.
    """
    seen: list[str] = []
    for description in descriptions:
        description = description.strip()
        if description and description not in seen:
            seen.append(description)
    combined = " ".join(seen)
    words = combined.split()
    if len(words) > CHARACTER_REF_WORD_CAP:
        combined = " ".join(words[:CHARACTER_REF_WORD_CAP])
    return combined
```

Replace `_build_prompt`:

```python
def _build_prompt(scene: str, preset: StylePreset) -> str:
    """Compose the FLUX prompt.

    FLUX uses a T5 encoder (~512 tokens), so unlike the SD 1.5 path we do not trim
    the scene. The FLUX-tuned style fragment is preferred; it falls back to the SD
    fragment for presets that have not been tuned for FLUX yet. Some LoRAs also
    require a trailing trigger sentence, appended via flux_prompt_suffix.
    """
    style_prompt = preset.flux_prompt or preset.sd_prompt
    prompt = SD_PROMPT_TEMPLATE.format(scene=scene, style_prompt=style_prompt)
    if preset.flux_prompt_suffix:
        prompt = f"{prompt} {preset.flux_prompt_suffix}"
    return prompt
```

with:

```python
def _build_prompt(scene: str, preset: StylePreset, character_ref: str = "") -> str:
    """Compose the FLUX prompt.

    FLUX uses a T5 encoder (~512 tokens), so unlike the SD 1.5 path we do not trim
    the scene. The FLUX-tuned style fragment is preferred; it falls back to the SD
    fragment for presets that have not been tuned for FLUX yet. Some LoRAs also
    require a trailing trigger sentence, appended via flux_prompt_suffix.

    `character_ref`, when given, is a book-wide continuity string (see
    `_build_character_reference`) appended so recurring characters/settings stay
    descriptively consistent across independently generated spreads.
    """
    style_prompt = preset.flux_prompt or preset.sd_prompt
    prompt = SD_PROMPT_TEMPLATE.format(scene=scene, style_prompt=style_prompt)
    if preset.flux_prompt_suffix:
        prompt = f"{prompt} {preset.flux_prompt_suffix}"
    if character_ref:
        prompt = f"{prompt} Recurring characters and setting across the book: {character_ref}."
    return prompt
```

In `generate_illustrations_fal`, replace:

```python
    saved_paths: list[str] = []
    for i, scene in enumerate(scenes):
        arguments: dict = {
            "prompt": _build_prompt(scene, preset),
```

with:

```python
    character_ref = _build_character_reference(scenes)
    saved_paths: list[str] = []
    for i, scene in enumerate(scenes):
        arguments: dict = {
            "prompt": _build_prompt(scene, preset, character_ref),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_illustrate_fal.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stages/illustrate_fal.py tests/test_illustrate_fal.py
git commit -m "feat(pipeline): ✨ inject cross-spread character continuity into prompts

Every spread's FLUX prompt now carries a deduped combination of all
the book's photo descriptions, so independently generated spreads
stay descriptively consistent on recurring characters/settings.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 7: Sharpen `stage1_story`'s continuity instruction

**Files:**
- Modify: `src/utils/prompt_templates.py:81`
- Test: `tests/test_stage1_story.py`

**Interfaces:**
- No signature changes — `LLM_STORY_GENERATION`'s formatted text changes only.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage1_story.py`:

```python
def test_generate_story_prompt_mentions_continuity_techniques(mocker):
    mock_generate = mocker.patch(
        "src.stages.stage1_story.generate_text",
        return_value='["A.", "B.", "C."]',
    )

    generate_story(
        descriptions={"p1.jpg": "a", "p2.jpg": "b", "p3.jpg": "c"},
        narrative="arc",
        context="ctx",
        style="ghibli",
        language="en",
        model_name="gemini-2.5-flash",
    )

    prompt = mock_generate.call_args[0][0]
    assert "recurring character" in prompt
    assert "consequence" in prompt
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_stage1_story.py -k continuity_techniques -v`
Expected: FAIL with `AssertionError: assert 'recurring character' in prompt`

- [ ] **Step 3: Replace the continuity rule**

In `src/utils/prompt_templates.py`, replace line 81:

```python
- Pages must connect naturally (reference what happened before)
```

with:

```python
- Pages must connect naturally: use techniques like a recurring character or object reappearing, a consequence following from the previous page's event, or the same setting carrying across pages -- vary which technique you use rather than repeating one every page
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_stage1_story.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/prompt_templates.py tests/test_stage1_story.py
git commit -m "feat(pipeline): ✨ sharpen stage1 continuity instruction with taxonomy

Replaces the generic 'reference what happened before' rule with the
specific techniques mined from docs/eval/reference_continuity_calibration.md
(recurring character/object, consequence-of-prior-action, setting
persistence), so the instruction is traceable to evidence.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Part C — Evaluation

### Task 8: `src/eval/clip_score.py` — within-spread relevance

**Files:**
- Create: `src/eval/__init__.py`
- Create: `src/eval/clip_score.py`
- Test: `tests/test_clip_score.py`

**Interfaces:**
- Produces: `compute_clip_score(image_path: str, caption: str, model_name: str = "ViT-B-32", pretrained: str = "openai") -> float`, `score_spreads(illustration_paths: list[str], pages: list[str], model_name: str = "ViT-B-32", pretrained: str = "openai") -> list[float]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_clip_score.py`:

```python
# tests/test_clip_score.py
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip entire file if torch not available


def _make_open_clip_mock(image_features: np.ndarray, text_features: np.ndarray):
    """Fake open_clip module returning fixed image/text feature vectors."""
    mock_preprocess = MagicMock(side_effect=lambda img: torch.zeros(3, 224, 224))
    mock_model = MagicMock()
    mock_model.encode_image.return_value = torch.tensor(image_features).unsqueeze(0)
    mock_model.encode_text.return_value = torch.tensor(text_features).unsqueeze(0)

    mock_tokenizer = MagicMock(side_effect=lambda texts: torch.zeros(len(texts), 77, dtype=torch.long))

    mock_oc = MagicMock()
    mock_oc.create_model_and_transforms.return_value = (mock_model, MagicMock(), mock_preprocess)
    mock_oc.get_tokenizer.return_value = mock_tokenizer
    return mock_oc


def test_compute_clip_score_identical_vectors_scores_one(tmp_images):
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(vec, vec)}):
        from src.eval.clip_score import compute_clip_score

        score = compute_clip_score(tmp_images[0], "a red square")

    assert score == pytest.approx(1.0, abs=1e-4)


def test_compute_clip_score_orthogonal_vectors_scores_zero(tmp_images):
    image_vec = np.zeros(512, dtype=np.float32)
    image_vec[0] = 1.0
    text_vec = np.zeros(512, dtype=np.float32)
    text_vec[1] = 1.0

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(image_vec, text_vec)}):
        from src.eval.clip_score import compute_clip_score

        score = compute_clip_score(tmp_images[0], "unrelated caption")

    assert score == pytest.approx(0.0, abs=1e-4)


def test_score_spreads_returns_one_score_per_page(tmp_images):
    vec = np.zeros(512, dtype=np.float32)
    vec[0] = 1.0

    with patch.dict(sys.modules, {"open_clip": _make_open_clip_mock(vec, vec)}):
        from src.eval.clip_score import score_spreads

        scores = score_spreads(tmp_images[:3], ["a.", "b.", "c."])

    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_score_spreads_wrong_count_raises(tmp_images):
    from src.eval.clip_score import score_spreads

    with pytest.raises(AssertionError):
        score_spreads(tmp_images[:2], ["only one caption"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_clip_score.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval'`

- [ ] **Step 3: Create the package and implementation**

Create `src/eval/__init__.py` (empty file).

Create `src/eval/clip_score.py`:

```python
"""CLIPScore: within-spread caption-image alignment.

Reuses the same CLIP model already used for photo selection in
src/stages/stage0_select.py's clip_cluster_select.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "ViT-B-32"
DEFAULT_CLIP_PRETRAINED = "openai"


def compute_clip_score(
    image_path: str,
    caption: str,
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> float:
    """Cosine similarity between an image and its caption, in [-1.0, 1.0]."""
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()

    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    text = tokenizer([caption])

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return float((image_features @ text_features.T).item())


def score_spreads(
    illustration_paths: list[str],
    pages: list[str],
    model_name: str = DEFAULT_CLIP_MODEL,
    pretrained: str = DEFAULT_CLIP_PRETRAINED,
) -> list[float]:
    """Within-spread relevance: one CLIPScore per spread (caption vs its own illustration)."""
    assert len(illustration_paths) == len(pages), (
        f"Mismatch: {len(illustration_paths)} illustrations vs {len(pages)} pages"
    )
    scores = [
        compute_clip_score(path, caption, model_name, pretrained)
        for path, caption in zip(illustration_paths, pages)
    ]
    logger.info(
        "Computed within-spread CLIPScores",
        extra={"count": len(scores), "mean": float(np.mean(scores)) if scores else None},
    )
    return scores
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_clip_score.py -v`
Expected: all PASS (or all SKIPPED if `torch` is not installed in this environment — acceptable per the existing `test_stage0_select.py` precedent).

- [ ] **Step 5: Commit**

```bash
git add src/eval/__init__.py src/eval/clip_score.py tests/test_clip_score.py
git commit -m "feat(pipeline): ✨ add clip_score for within-spread relevance

CLIPScore between each spread's caption and its own illustration,
per the 2026-07-08 evaluation design. Reuses the CLIP model already
used for photo selection in stage0_select.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

### Task 9: `src/eval/narrative_continuity_judge.py` — between-spread continuity

**Files:**
- Create: `src/eval/narrative_continuity_judge.py`
- Test: `tests/test_narrative_continuity_judge.py`

**Interfaces:**
- Consumes: `generate_text(prompt: str, model: str, max_output_tokens: int = 512, ...) -> str` (existing, `src/utils/gemini_client.py:55-87`).
- Produces: `TAXONOMY: list[str]`, `classify_transition(page_n: str, page_n_plus_1: str, model_name: str) -> str`, `score_book_continuity(pages: list[str], model_name: str) -> dict`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_narrative_continuity_judge.py`:

```python
# tests/test_narrative_continuity_judge.py
from src.eval.narrative_continuity_judge import (
    TAXONOMY,
    classify_transition,
    score_book_continuity,
)


def test_classify_transition_returns_taxonomy_category(mocker):
    mocker.patch(
        "src.eval.narrative_continuity_judge.generate_text",
        return_value="consequence_of_prior_action",
    )

    result = classify_transition("Page one.", "Page two.", model_name="gemini-2.5-flash")

    assert result == "consequence_of_prior_action"
    assert result in TAXONOMY


def test_classify_transition_accepts_none_classification(mocker):
    """A judge classifying a self-contained transition as 'none' must not be
    coerced into a positive category (docs/eval/reference_continuity_calibration.md)."""
    mocker.patch(
        "src.eval.narrative_continuity_judge.generate_text",
        return_value="none",
    )

    result = classify_transition("I am a giraffe.", "I am a buffalo.", model_name="gemini-2.5-flash")

    assert result == "none"


def test_classify_transition_defaults_to_none_on_unrecognized_output(mocker):
    mocker.patch(
        "src.eval.narrative_continuity_judge.generate_text",
        return_value="something the model made up",
    )

    result = classify_transition("Page one.", "Page two.", model_name="gemini-2.5-flash")

    assert result == "none"


def test_score_book_continuity_aggregates_transitions(mocker):
    mocker.patch(
        "src.eval.narrative_continuity_judge.classify_transition",
        side_effect=["recurring_character_object", "none", "consequence_of_prior_action"],
    )

    result = score_book_continuity(
        ["Page 1.", "Page 2.", "Page 3.", "Page 4."], model_name="gemini-2.5-flash"
    )

    assert result["transitions"] == [
        "recurring_character_object",
        "none",
        "consequence_of_prior_action",
    ]
    assert result["continuous_count"] == 2
    assert result["total_transitions"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_narrative_continuity_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.narrative_continuity_judge'`

- [ ] **Step 3: Implement the module**

Create `src/eval/narrative_continuity_judge.py`:

```python
"""LLM-as-judge scoring of page-to-page narrative continuity.

Classifies each adjacent caption pair against the taxonomy hand-calibrated
in docs/eval/reference_continuity_calibration.md.
"""

from __future__ import annotations

import logging

from src.utils.gemini_client import generate_text

logger = logging.getLogger(__name__)

TAXONOMY = [
    "recurring_character_object",
    "consequence_of_prior_action",
    "setting_persistence",
    "emotional_arc_progression",
    "none",
]

JUDGE_PROMPT = """\
You are scoring narrative continuity between two consecutive pages of a children's picture book.

Page N: "{page_n}"
Page N+1: "{page_n_plus_1}"

Classify the relationship between these two pages using EXACTLY ONE of these categories:
- recurring_character_object: the same character or object from page N reappears in page N+1
- consequence_of_prior_action: page N+1's event follows causally from page N's event
- setting_persistence: the same location or event context carries across both pages
- emotional_arc_progression: there is a clear emotional shift between the two pages
- none: there is no detectable continuity between the two pages

Reply with ONLY the category name, nothing else."""


def classify_transition(page_n: str, page_n_plus_1: str, model_name: str) -> str:
    """Classify one adjacent-page transition against the continuity taxonomy.

    Falls back to "none" if the judge returns anything outside the taxonomy,
    rather than silently accepting an unscored value.
    """
    prompt = JUDGE_PROMPT.format(page_n=page_n, page_n_plus_1=page_n_plus_1)
    raw = generate_text(prompt, model_name, max_output_tokens=16).strip().lower()
    if raw not in TAXONOMY:
        logger.warning(
            "Judge returned an unrecognized category; defaulting to 'none'",
            extra={"raw": raw[:64]},
        )
        return "none"
    return raw


def score_book_continuity(pages: list[str], model_name: str) -> dict:
    """Classify every adjacent transition in a book.

    Returns:
        {"transitions": list[str], "continuous_count": int, "total_transitions": int}
    """
    transitions = [
        classify_transition(pages[i], pages[i + 1], model_name) for i in range(len(pages) - 1)
    ]
    continuous_count = sum(1 for t in transitions if t != "none")
    logger.info(
        "Scored book continuity",
        extra={"continuous_count": continuous_count, "total_transitions": len(transitions)},
    )
    return {
        "transitions": transitions,
        "continuous_count": continuous_count,
        "total_transitions": len(transitions),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_narrative_continuity_judge.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/eval/narrative_continuity_judge.py tests/test_narrative_continuity_judge.py
git commit -m "feat(pipeline): ✨ add narrative_continuity_judge for between-spread scoring

LLM-as-judge classification of adjacent caption pairs against the
5-category taxonomy (including 'none') calibrated against 7 reference
books in docs/eval/reference_continuity_calibration.md. This is the
eval script the RQ3 causal-vs-no-causal ablation will run through.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes

**Spec coverage:**
- 2026-07-01 rename + `_choose_layout_variant` + `build_picture_book_pdf` + tests → Tasks 1-3.
- 2026-07-08 Decision 1 (spread unit, landscape illustration) → Tasks 4-5.
- Decision 2 (one caption per spread, no `stage1_story` output contract change) → satisfied by construction (Task 7 only edits prompt text, not `generate_story`'s signature/return shape).
- Decision 3 (reuse `_choose_layout_variant`, no whitespace-detection) → Task 4 (`build_spread_pdf` reuses Task 2/3 code as-is).
- Decision 4 (cross-spread character consistency, prompt-level only) → Task 6.
- Decision 5 + Evaluation section (taxonomy, calibration, judge, CLIPScore) → Tasks 8-9, calibration file already committed in `docs/eval/reference_continuity_calibration.md`.
- Testing section's named tests are all present: `test_choose_layout_variant_picks_top_band_for_long_captions`, `test_choose_layout_variant_picks_bottom_caption_for_short_captions` (Task 2); `test_build_picture_book_pdf_creates_file` (split into `_bottom_caption`/`_top_band` variants, Task 3), `test_build_picture_book_pdf_wrong_count_raises` (Task 3); `test_choose_layout_variant_uses_spread_width`, `test_build_spread_pdf_creates_file`, `test_build_spread_pdf_wrong_count_raises` (Task 4); `test_illustrate_fal_uses_spread_aspect_ratio` → `test_landscape_spread_aspect_ratio_used` (Task 5); `test_illustrate_fal_prompt_includes_character_description` → `test_prompt_includes_character_reference_from_other_pages` (Task 6); `test_narrative_continuity_judge_classifies_adjacent_pairs` → `test_classify_transition_returns_taxonomy_category` (Task 9); `test_narrative_continuity_judge_accepts_none_classification` → `test_classify_transition_accepts_none_classification` (Task 9); `test_narrative_continuity_judge_aggregates_book_level_score` → `test_score_book_continuity_aggregates_transitions` (Task 9).
- Out of scope items (split/mirrored captions, whitespace-detection placement, pixel-identical character consistency, `stage0_select` count mapping, bold CJK font) → correctly not implemented by any task.

**Placeholder scan:** none found — every step has complete code or an exact command.

**Type consistency:** `_choose_layout_variant` returns `Literal["top_band", "bottom_caption"]` in Task 2 and is consumed with that exact return type in Task 3's `build_picture_book_pdf` and `_draw_picture_book_page`. `build_picture_book_pdf`'s `page_size` parameter (Task 3) matches `build_spread_pdf`'s call site (Task 4). `_build_prompt`'s new `character_ref` parameter (Task 6) is optional with a default, so Task 5's unrelated `image_size` change and any other untouched call sites keep working. `TAXONOMY` (Task 9) values (`recurring_character_object`, `consequence_of_prior_action`, `setting_persistence`, `emotional_arc_progression`, `none`) match `docs/eval/reference_continuity_calibration.md`'s category names.
