# 繪本品質三大調整 (Storybook Quality — Three Improvements)

> 目標：讓 Photo2Story 的輸出更像「真正的兒童繪本」，而不是「圖片 + 一段說明文字」。
> 本文件說明三項調整的**動機、做法、以及對應的程式碼位置**。

---

## 0. 範圍收斂：只保留 4 種畫風 (Scope: 4 styles only)

在動畫風格上，我們把支援的畫風從 7 種收斂成 **4 種大師 / 經典風格**，每一種都綁定一個專屬的 hosted FLUX LoRA，確保整本書風格一致：

| Key | Label | 風格說明 | FLUX LoRA |
|---|---|---|---|
| `ghibli` | Studio Ghibli | 宮崎駿 / 吉卜力，手繪水彩背景、柔和暖光 | `openfree/flux-chatgpt-ghibli-lora` |
| `pixar` | 3D Pixar | 皮克斯 3D 動畫質感、柔和全域光、大眼角色 | `prithivMLmods/Canopus-Pixar-3D-Flux-LoRA` |
| `disney` | Classic Disney | 經典迪士尼手繪 2D 動畫 | `tubbymeatball/DisneyStyleLora` |
| `crayon` | Crayon drawing | 兒童蠟筆畫、蠟質筆觸、天真手繪感 | （FLUX 純提示詞，無需 LoRA） |

**移除**的風格：`watercolor`、`ink_wash`、`flat_pastel`（這三種沒有專屬 LoRA、且不屬於「大師風格」定位）。

對應程式碼：[`src/utils/styles.py`](../src/utils/styles.py)
- `StyleKey`（StrEnum）只保留 `GHIBLI / PIXAR / DISNEY / CRAYON` 四個成員
- `STYLE_PRESETS` 只保留這四個 preset
- `resolve_style()` 對未註冊的字串仍會 fallback 成 free-form passthrough，所以收斂不會讓舊呼叫直接崩潰

> 注意：[`assets/style_refs/`](../assets/style_refs/) 中對應的 IP-Adapter 參考圖也只需保留這 4 張。

---

## 1. 調整故事內容：更少的字、更簡單的詞 (Simpler story text)

### 為什麼
真正的繪本文字量很少、用字簡單——通常一頁只有一句話，用的是學齡前小孩就懂的常用字。
原本的故事生成預設是「每頁 2–3 句」，對繪本來說太長、太複雜。

### 做法
透過 **reading level（閱讀分級）** 控制故事的長度與用字難度。在
[`src/utils/prompt_templates.py`](../src/utils/prompt_templates.py) 的 `READING_LEVELS` 中定義：

| Level | 年齡 | 每頁句數 | 長度上限 | 用字 |
|---|---|---|---|---|
| `simple` | 3–5 歲 | **恰好一句短句** | 中文約 18 字 / 英文約 12 詞 | 只用學齡前最常見的日常字；無成語、無生難字、無子句 |
| `standard` | 6–8 歲 | 2–3 句短句 | 中文約 45 字 / 英文約 35 詞 | 簡單、溫暖、適合小孩的字彙 |

這些值會被注入到 `LLM_STORY_GENERATION` 提示詞的 `{sentences}` / `{length_hint}` / `{vocab}` / `{age}` 欄位。

### 串接位置
- [`src/stages/stage1_story.py`](../src/stages/stage1_story.py) 的 `generate_story()` 會讀 `reading_level`，
  從 `READING_LEVELS` 取出對應設定後填入提示詞。
- 在 [`configs/demo.yaml`](../configs/demo.yaml) 中以 `stage1.reading_level: simple` 啟用繪本級別的簡短文字。

```yaml
stage1:
  reading_level: simple   # 繪本：一頁一句、用字最簡單
```

---

## 2. 調整繪本的畫風：固定幾種大師風格 (Fixed master art styles)

### 為什麼
若每頁畫風漂移，就不像一本完整的繪本。固定成少數幾種「大師 / 經典」風格，
可以讓整本書視覺一致，也讓使用者選擇更直覺（宮崎駿 / 皮克斯 / 迪士尼 / 蠟筆）。

### 做法
- 風格集合收斂為第 0 節的 4 種（見上表）。
- 每個 preset 綁定專屬 FLUX LoRA 與調校過的自然語言提示詞 `flux_prompt`，
  在 hosted FLUX 後端 (`stage2.backend: fal`) 下重現該大師的筆觸。
- 風格的 **label** 也會餵進故事生成提示詞（`LLM_STORY_GENERATION` 的 `{style}`），
  讓文字語氣與畫風一致。

對應程式碼：
- 風格定義：[`src/utils/styles.py`](../src/utils/styles.py) → `STYLE_PRESETS`
- hosted FLUX 繪製：[`src/stages/illustrate_fal.py`](../src/stages/illustrate_fal.py)
- 風格如何進入文字：[`src/stages/stage1_story.py:64`](../src/stages/stage1_story.py#L64)（`resolve_style(style).label`）

```yaml
stage2:
  backend: fal            # hosted FLUX，吃 flux_prompt / flux_lora
  use_ipadapter: true     # 用 style_refs 參考圖維持整本一致
```

---

## 3. 調整最後的 Layout：文字疊在圖上，而非單獨一頁 (Caption overlay layout)

### 為什麼
舊版排版是「上圖、下方一大塊文字」（甚至文字像是獨立空白頁）。
真正的繪本是**滿版插圖**，文字直接疊在圖片上（caption band / 對話框），閱讀體驗完全不同。

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

---

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
</content>
</invoke>
