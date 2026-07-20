# 實驗資料核心（experiment_data_core）

本資料夾是**論文所有數據的單一真相來源快照**。檔案為**複製品**（非搬移）；
原始檔仍留在專案各處，以免弄壞寫死路徑的分析腳本（如 `geom_h4_h5.py`）。
複製日期：2026-07-21。

---

## 01_annotation_master —— 標註總檔（最重要）
- `framing_annotation_full.csv`（7,200 列）
  - 論文 Results 幾乎**每一個數字都由此檔算出**（已於 2026-07-21 逐項核對，見下）。
  - 一列 = 一則 AV 描述。關鍵欄位：`model / entity / lang / site / cell_type /
    frame_source(刺激框架) / d1_frame / d2_anchor / d3_construal / d4_ortho / d5_drift`。
  - 結構：Qwen 3,600 + Gemma 3,600；Taiwan 4,800 / Japan 1,200 / Iceland 1,200；
    en 3,600 / zh 3,600；baseline 6,720 / codeswitch 480；site A 3,600 / B 3,600。
  - Taiwan baseline = 4,320 列（H1/H2/H4/construal 主分析集）。

## 02_activations —— 殘差流向量（幾何 H4/H5、steering 來源）
- `activations_Qwen2.5-7B-Instruct.parquet`（L20）
- `activations_gemma-3-12b-it.parquet`（L32）
  - 供 `geom_h4_h5.py`（幾何 DiD、Design B）與 `steer.py`（因果注入）使用。
  - 兩模型維度/尺度不同，**絕不跨模型比**；幾何分析用 per-dim 標準化。

## 03_kappa_reliability_validity —— 信度與效度輸入（240 子集）
- `overlap_A/B/C.csv` —— 三位標註者對同 240 筆的標籤 → **Fleiss κ 信度**。
- `kappa_annotation_a_240.csv`、`kappa_annotation_b_240.csv` —— 兩位**人類**各標 240 → **效度天花板**。
- `llm_240.csv` —— LLM 對同 240 筆的標籤（效度比對用）。
- `kappa_blank_240.csv` —— 兩位人類所見的盲標輸入（av_text/原句）。
  - 註：效度 240 與信度 overlap 240 **交集為 0**（獨立樣本，無循環）。

## 04_corpus_minimal_pairs —— 語料來源
- `rq1_review_all.csv` —— 360 句 minimal-pair 語料（extract/steer 的輸入源）。

## 05_raw_annotator_shards —— 三人盲標原始輸入（provenance）
- `blind_A/B/C.csv` —— 合併成 `framing_annotation_full.csv` 之前的三份原始 shard。
  - 上游存檔，供追溯合併過程；平常分析用 01 的總檔即可。

---

## 資料可信度稽核結果（2026-07-21）

把論文 Results/Appendix 的 headline 數字**全部從 01 總檔重算並比對**：

- ✅ **吻合（分毫不差）**：結構計數、P1a 語言效應（PRC 14.3/3.0、any 17.8/6.0）、
  by-site（21.2/3.8、7.4/2.1）、by-model（Qwen 15.0 / Gemma 2.3）、2×2 PRC、
  控制實體 anchoring（≤0.2%）、construal（8.8/6.7、26.4/29.5）、
  模型 drift（47.6/18.7）、簡體（23.2/2.0）、8 個框架的 anchored%/PRC%/n、
  D4×D2 逐格計數、PRC 描述語言 350/23、PRC by ortho 48/16、
  GEOPOL-OTHER n=141、HALLUCINATED n=133。
- ❌ **唯一不吻合**：`geopolitical intrusion`（論文 Table lang：en 13.0 / zh 2.9；
  by-model Qwen 11.0 / Gemma 1.3）。用各種標準定義重算，最接近的是
  「非政治框架內 anchored」= en 10.1 / zh 1.5，**無任何定義能重現 13.0/2.9**。
  → 判斷為**定義未鎖定或數字過期**。建議：鎖定 intrusion 定義後重算並更新
  Table lang；或因 intrusion 本質上是 P1a 的重述，直接改報「非政治框架內的
  anchoring 率」（10.1% vs 1.5%，可重現）。此為次要支持統計，不影響 P1a 主結論。
(已修改論文，讓論文符合數據的呈現)
