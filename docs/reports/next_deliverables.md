# 下次 Meeting Deliverables Checklist

**Meeting 1 日期**：2026-05-19
**Meeting 2 目標日期**：（待定，預估 2 週後）
**Milestone**：完整 Ablation 結果報告

---

## 🎯 核心 Deliverable：三組 Ablation 對照表

下次 meeting 要交付的**主要產出**。每組對照表存成獨立 markdown，方便 meeting 時 share screen。

### Deliverable 1：RQ1 — 選圖策略對照表

**目標檔案**：`docs/ablation_rq1_selection.md`

- [ ] 跑 `configs/ablation_rq1_clip.yaml`（純 CLIP）
- [ ] 跑 `configs/ablation_rq1_llm.yaml`（純 LLM）
- [ ] 跑 `configs/ablation_rq1_hybrid.yaml`（hybrid）
- [ ] 加跑 `configs/baseline.yaml` 或自建 random 版本（基準線）
- [ ] **使用同一組照片**：`data/apple_swim/*.JPG`
- [ ] **使用同一個 context**：例如 `"family apple picking and swimming"`
- [ ] 填寫對照表（見下方模板）
- [ ] 標註每個變體**選中哪些照片**、**有沒有 outlier**

### Deliverable 2：RQ2 — Context 利用對照表

**目標檔案**：`docs/ablation_rq2_context.md`

- [ ] 跑 `configs/ablation_rq2_full.yaml`（完整 context）
- [ ] 跑 `configs/ablation_rq2_keyword.yaml`（關鍵字 context）
- [ ] **使用同一組照片**
- [ ] 比較**生成的故事文字**是否具體
- [ ] 比較**第 1 頁開場**是否切題

### Deliverable 3：RQ3 — Causal Inference 對照表 ⭐

**目標檔案**：`docs/ablation_rq3_causal.md`

- [ ] 跑 `configs/ablation_rq3_no_causal.yaml`
- [ ] 跑 `configs/ablation_rq3_causal.yaml`
- [ ] **逐頁並排**比較故事文字
- [ ] 標註頁與頁之間有沒有 reference（連貫性）
- [ ] 抓出 narrative arc 欄位給老師看（這是 causal 版本的「劇情骨架」）

---

## 📊 對照表 Markdown 模板

每張對照表至少要有：

```markdown
# RQ<X>: <主題> 對照表

## 實驗設定
- **照片**：`data/apple_swim/*.JPG`（N 張）
- **Context**：`"..."`
- **Style**：`watercolor`
- **Language**：`zh-tw`
- **共同變因**：除了 RQ<X> 涉及的參數，其他 config 完全相同

## 結果對照

| 變體 | 選中照片 | Effective Context | 觀察 |
|---|---|---|---|
| A | ... | ... | ... |
| B | ... | ... | ... |

## 逐頁對照（最關鍵的部分）

### Page 1
| A | B |
|---|---|
| 文字 / 插畫縮圖 | 文字 / 插畫縮圖 |

### Page 2
...

## 結論
- **觀察 1**：...
- **觀察 2**：...
- **支持的設計決策**：...
```

---

## 🛠️ 技術 Deliverables

### 必做（影響 meeting 2 的 demo 品質）

- [ ] **修正 PDF 中文掉字**：把 ReportLab 的 Helvetica 換成 Noto Sans CJK 或思源黑體
  - 影響檔案：`src/stages/stage3_assemble.py`
  - 驗證：跑一次 `--language zh-tw` 看 PDF 是否正常顯示
- [ ] **加上 per-stage timing log**：每個 stage 開始/結束記時間
  - 影響檔案：`src/pipeline.py`、各 stage 檔案
  - 目的：meeting 2 能回答「哪個 stage 是 bottleneck」
- [ ] **準備 backup demo run**：預先跑好一個成功的 run 並備份
  - 路徑：`output/demo_backup/`
  - 用途：meeting 2 demo 失敗時的 fallback

### 選做（時間夠就做）

- [ ] **加入 LLM-as-judge 評分腳本**（用 Gemini 評 coherence / creativity）
  - 新檔案：`src/eval/llm_judge.py`
  - 輸入：兩個 run 的 `result.json`
  - 輸出：score + 比較理由
- [ ] **CLIPScore 評估腳本**（量插畫-故事對齊度）
  - 新檔案：`src/eval/clip_score.py`
- [ ] **跑一組不同 style 測試**（watercolor / anime / Chinese ink wash）
  - 目的：為 meeting 3 的「風格矩陣」做暖身

---

## 📅 兩週內建議排程

| 週次 | 任務 |
|---|---|
| **Week 1 前半** | 修 PDF 中文字體 + per-stage timing + 跑完所有 ablation runs |
| **Week 1 後半** | 整理 RQ1、RQ2 對照表 |
| **Week 2 前半** | 整理 RQ3 對照表（最重要的一張）+ 主觀觀察 |
| **Week 2 後半** | 準備 meeting 2 講稿 + backup demo + 預習問答 |

---

## ❓ 開 meeting 2 前自我檢查

- [ ] 三張對照表都有具體的「結論」欄位（不只是並排截圖）
- [ ] 每張對照表的觀察至少有 **1 個量化指標**（例如「8 頁中有 6 頁有前後 reference」）
- [ ] 想清楚下下次（meeting 3）的 deliverable，主動提給教授
- [ ] 準備 **1-2 個新的選擇題** 請教教授

---

## 📌 Meeting 1 教授特別要求 / 反饋（meeting 後補填）

> *Meeting 結束後立刻把教授說過的話、要求、建議寫在這裡，避免兩週後忘記。*

- ...
- ...
- ...
