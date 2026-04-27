# Photo2Story 系統設計文件

**日期：** 2026-04-28  
**版本：** v1.0  
**論文題目：** 基於多模態大語言模型之個人化照片繪本自動生成系統

---

## 一、系統定位與核心貢獻

### 一句話定位
使用者輸入一批照片（例如旅遊 20 張）與情境描述，系統自動**篩選出代表性照片、推斷故事順序、生成連貫故事文字與風格一致插圖**，最終輸出可下載的個人化 PDF 繪本。

### 核心研究貢獻（論文 novelty）
**Stage 0 的兩階段混合選片演算法**——現有選片方法（MMR、DPP、Submodular）多用於影片摘要或一般圖集，本研究**首次將使用者自然語言情境描述引入選片評分函式**，並整合至個人化繪本生成流程，形成端到端可用系統。

---

## 二、研究問題（RQ）

- **RQ1：** 不同選片策略（隨機 / CLIP-only / LLM-only / 混合）對生成繪本敘事品質的影響？
- **RQ2：** 使用者情境描述（無 / 關鍵字 / 完整句）對選片關聯性與故事品質的影響？

---

## 三、系統 Pipeline

```
輸入：N 張照片（建議 10–30 張）
    + 使用者情境描述（可選，例如：「北海道六天五夜，男方爸媽和兩個兒子」）
    + 繪本風格（水彩 / 日系 / 歐式，或上傳參考圖）
         ↓
[Stage 0] 照片分析與自動選取  ← 核心研究貢獻
         ↓
[Stage 1] 故事文字生成
         ↓
[Stage 2] 插圖生成
         ↓
[Stage 3] PDF 組裝與輸出
```

---

## 四、各 Stage 詳細設計

### Stage 0：照片分析與自動選取

**目標：** 從 N 張照片中選出 K 張（K 由使用者指定，預設 4，範圍 3–6），並推斷敘事順序。

#### Phase A：多樣性過濾（CLIP + Clustering）
- 用 CLIP 將每張照片編碼為向量
- K-means Clustering 分成 K 群
- 每群選最靠近群心的 1 張照片
- **參考論文：** Radford et al., *CLIP*, ICML 2021；Gygli et al., *Submodular Summarization*, ECCV 2014；Carbonell & Goldstein, *MMR*, SIGIR 1998

#### Phase B：關聯性重排（VLM 描述 + LLM 評分）
- VLM 對 K 張候選照片各生成一段描述
- LLM 對照使用者情境描述，對每張照片的關聯性打分（0–1）
- 分數低於門檻者以同群次候選替換
- **參考論文：** Huang et al., *VIST*, NAACL 2016；Wang et al., *Qwen2-VL*, 2024

#### Phase C：敘事順序推斷
- 優先以 EXIF 時間戳排序
- 無 EXIF 時由 LLM 根據場景、光線、活動推斷順序
- 輸出：K 張有時序的照片 + 對應 VLM 描述（JSON）

---

### Stage 1：故事文字生成

**模型：** Qwen2.5-7B（4-bit 量化）  
**輸入：** K 張照片描述 + 使用者情境描述 + 風格提示  
**輸出：** K 頁故事文字（JSON，每頁 2–3 句）

**Prompt 結構：**
```
[系統] 你是繪本作家，請根據以下照片描述和旅遊情境，
       生成 {K} 頁連貫的繪本故事，每頁約 2–3 句話，
       風格：{風格}。
[照片描述] 第1張：{描述1} / 第2張：{描述2} / ...
[情境] {使用者輸入}
```

**參考論文：** AR-LDM (Pan et al., NeurIPS 2022)、Make-A-Story (Rahman et al., CVPR 2023)

---

### Stage 2：插圖生成

| 版本 | 模型 | 備註 |
|------|------|------|
| Demo 版 | SD 1.5 + IP-Adapter（風格條件） | ~5GB VRAM，快速 |
| 研究版 | SDXL + IP-Adapter + StyleAligned | ~8GB VRAM |

**跨頁風格一致性：** StyleAligned attention sharing（研究版）

**參考論文：** Ye et al., *IP-Adapter*, ICCV 2023；Hertz et al., *StyleAligned*, CVPR 2024

---

### Stage 3：PDF 組裝輸出

- 每頁排版：插圖（上）+ 故事文字（下）
- 工具：Pillow（排版）+ ReportLab（PDF 生成）
- Gradio UI：即時顯示每頁預覽 + PDF 下載按鈕

---

## 五、Config 驅動架構

### 目錄結構

```
photo2story/
├── configs/
│   ├── demo.yaml                  # 兩週 demo（輕量模型）
│   ├── baseline.yaml              # 論文 baseline（隨機選片，無情境，無 FaceID）
│   ├── ablation_rq1_clip.yaml     # RQ1：CLIP-only 選片
│   ├── ablation_rq1_llm.yaml      # RQ1：LLM-only 選片
│   ├── ablation_rq1_hybrid.yaml   # RQ1：混合選片（本方法）
│   ├── ablation_rq2_keyword.yaml  # RQ2：關鍵字情境
│   ├── ablation_rq2_full.yaml     # RQ2：完整句情境
│   └── full.yaml                  # 完整系統（RQ1 混合 + RQ2 完整情境）
│
├── src/
│   ├── pipeline.py                # StoryPipeline 主控類別
│   ├── model_manager.py           # VRAM load/unload 管理
│   ├── stages/
│   │   ├── stage0_select.py       # 照片選取（Phase A + B + C）
│   │   ├── stage1_story.py        # 故事文字生成
│   │   ├── stage2_illustrate.py   # 插圖生成
│   │   └── stage3_assemble.py     # PDF 組裝
│   └── utils/
│       ├── prompt_templates.py    # 所有 prompt 集中管理
│       └── image_utils.py
│
├── gradio_app.py                  # Demo UI 入口
├── run_experiment.py              # 論文實驗批次執行 CLI
└── evaluate.py                    # 指標計算
```

### Config Schema 範例

```yaml
# configs/demo.yaml
stage0:
  mode: hybrid
  clip_model: openai/clip-vit-base-patch32
  vlm_model: Qwen/Qwen2-VL-2B-Instruct
  llm_model: Qwen/Qwen2.5-7B-Instruct
  k: 4

stage1:
  model: Qwen/Qwen2.5-7B-Instruct
  context_mode: full        # none | keyword | full

stage2:
  base_model: runwayml/stable-diffusion-v1-5
  use_ipadapter: true
  use_faceid: false         # 研究版才開啟
  use_stylealigned: false

stage3:
  output_format: pdf
  page_layout: image_top_text_bottom
```

---

## 六、實驗計畫（Ablation Study）

| Config | Stage 0 選片 | 情境描述 | 對應 RQ |
|--------|------------|---------|---------|
| `baseline` | 隨機選 K 張 | 無 | 基準 |
| `ablation_rq1_clip` | CLIP-only | 無 | RQ1-A |
| `ablation_rq1_llm` | LLM-only | 有 | RQ1-B |
| `ablation_rq1_hybrid` | 混合（本方法） | 有 | **RQ1 主張** |
| `ablation_rq2_nokw` | 混合 | 無情境 | RQ2-A |
| `ablation_rq2_keyword` | 混合 | 關鍵字 | RQ2-B |
| `ablation_rq2_full` | 混合 | 完整句子 | RQ2-C |

---

## 七、評估指標

| 面向 | 指標 | 工具 | 方式 |
|------|------|------|------|
| 選片多樣性 | CLIP 特徵群內距離 | OpenCLIP | 自動 |
| 選片與情境關聯 | CLIP 文字-圖像相似度 | OpenCLIP | 自動 |
| 故事連貫性 | GPT-4-as-judge（1–5） | OpenAI API | 自動 |
| 故事與照片相符 | GPT-4-as-judge + 人工問卷 | 混合 | 混合 |
| 插圖風格一致性 | CLIP 特徵跨頁標準差 | OpenCLIP | 自動 |
| 整體滿意度 | 人工問卷（3 題 × 5 人） | Google Form | 人工 |

### 人工問卷（3 題，Google Form）
1. 這本繪本的故事讀起來連不連貫？（1 完全不連貫 ～ 5 非常連貫）
2. 故事內容和你看到的照片符合嗎？（1 完全不符 ～ 5 非常符合）
3. 整體來說你喜歡這本繪本嗎？（1 非常不喜歡 ～ 5 非常喜歡）

**受試者：** 5–8 人（實驗室同學或家人，無需正式招募）

---

## 八、硬體規格與記憶體管理

**目標硬體：** RTX 3060 12GB

| 模型 | VRAM 估算 |
|------|---------|
| Qwen2-VL-2B（4-bit） | ~2 GB |
| Qwen2-VL-7B（4-bit） | ~5 GB |
| Qwen2.5-7B（4-bit） | ~5 GB |
| SD 1.5 + IP-Adapter | ~5 GB |
| SDXL + IP-Adapter | ~8 GB |

**關鍵策略：** Stage 0/1（VLM/LLM）與 Stage 2（SD）分開載入，不同時佔用 VRAM，`model_manager.py` 負責 `del + torch.cuda.empty_cache()` 管理。

---

## 九、兩週 Demo 里程碑

```
Week 1：
  Day 1–2   建立 repo 結構 + config schema + model_manager 骨架
  Day 3–4   Stage 0（CLIP clustering + EXIF 排序 + 基本 LLM 評分）
  Day 5–7   Stage 1（Qwen2-VL-2B 描述 + Qwen2.5-7B 故事生成）

Week 2：
  Day 8–10  Stage 2（SD 1.5 + IP-Adapter 插圖生成）
  Day 11–12 Stage 3（Gradio UI + PDF 輸出）
  Day 13–14 端到端測試 + 準備 demo 照片組
```

**Demo 版限制（與研究版差異）：**
- Stage 0：完整 hybrid 流程（Phase A + B + C），但 VLM 用 Qwen2-VL-2B 而非 7B（加速）
- Stage 2：SD 1.5（非 SDXL），無 FaceID，無 StyleAligned
- 頁數：固定 K=4

---

## 十、使用的公開資料集

| 資料集 | 用途 |
|--------|------|
| VIST（Visual Storytelling） | Stage 0+1 評估基準（含人工故事標註） |
| FFHQ | 人臉身份訓練與評估（RQ3） |
| CelebA-HQ | 多人場景測試（RQ3） |
| Manga109 | 繪本插圖風格參考 |

---

## 十一、未來工作（Future Work）

**真實人臉植入插圖（FaceID）：** 使用 InsightFace 偵測照片中的人臉並建立 embedding，搭配 IP-Adapter FaceID 將真實人物外觀嵌入插圖，以 ArcFace 相似度評估跨頁人臉一致性。此方向因實作複雜度較高，列為後續研究延伸。

---

## 十二、倫理聲明

- 訓練資料使用公開授權資料集（FFHQ、CelebA-HQ、VIST）
- Demo 系統不儲存使用者上傳照片（session 結束後清除）
- 論文中聲明：系統不應用於未經本人同意的人臉場景生成
