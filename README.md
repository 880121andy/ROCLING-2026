# RQ1 實驗管線（Phase 02 抽取／自然語言化／忠實度 + Phase 03 分析骨架）

拿到人工審閱後的 **360 句語料**（`rq1_review_all.csv`）之後，本目錄的腳本把
`experiment_pipeline.md` 的 **02 模型環境建設** 與 **03 結果分析** 落成可執行程式，
Qwen2.5-7B（L20）與 Gemma-3-12B（L32）兩模型平行。方法框架依 NLA 官方 repo
（<https://github.com/kitft/natural_language_autoencoders>）的 AV／AR 推論介面。

## 資料流一覽

```
rq1_review_all.csv (360 句)
      │  extract_activations.py     360 句 × 2 site = 720 向量
      ▼
activations_*.parquet  (activation_vector: float32[720, d_model] + metadata)
      │  verbalize.py (AV, SGLang)  720 × k=5 = 3,600 則描述
      ▼
verbalizations.parquet
      │  score_roundtrip.py (AR)    每則描述 round-trip MSE = 2(1−cos)
      ▼
roundtrip.csv
      │  calibrate_gate.py          τ = MSE 最佳三分位 → pass_gate  ← Layer 0 閘門
      ▼
gate.gated.csv  +  gate.summary.md
      │  （人工/LLM 補 Layer 1–4 標註：construal, frames, china_anchor, …）
      ▼
analyze.py                          H1/H2/H4 迴歸、框架漂移、H3、幾何、H5
      ▼
analysis.report.md
```

## 腳本與 pipeline 階段對照

| 腳本 | experiment_pipeline.md 對應 | 產物 |
|---|---|---|
| `extract_activations.py` | 02 · Activation 抽取（720 向量） | `activations_*.parquet` |
| `verbalize.py` | 02 · AV 自然語言化（3,600 則） | `verbalizations.parquet` |
| `score_roundtrip.py` | 02 · AR round-trip 忠實度 | `roundtrip.csv` |
| `calibrate_gate.py` | 03 · Layer 0 忠實度閘門（τ） | `gate.gated.csv`, `gate.summary.md` |
| `analyze.py` | 03 · H1/H2/H4、漂移、H3、幾何、H5 | `analysis.report.md` |
| `run_pipeline.sh` | 02→03 單模型端到端 orchestration | 上述全部 |

## 前置環境（TWCC）

```bash
# 1) 相依套件
pip install -U "transformers>=4.50" torch pyarrow pandas numpy statsmodels scipy \
               httpx orjson pyyaml safetensors "sglang[all]"

# 2) NLA repo 與 SGLang input_embeds patch（AV 以 input_embeds 而非 input_ids 推論）
git clone https://github.com/kitft/natural_language_autoencoders
bash natural_language_autoencoders/patches/apply_sglang_patches.sh

# 3) 下載官方 checkpoints 到 $CKPT_ROOT
#    kitft/nla-qwen2.5-7b-L20-{av,ar}、kitft/nla-gemma3-12b-L32-{av,ar}
export HF_HOME=/work/$USER/hf_cache
huggingface-cli login          # Gemma 為 gated，需先於網頁同意授權
python natural_language_autoencoders/nla/scripts/pull_checkpoint.py ...
```

## 執行（兩模型平行）

```bash
# 開兩個 shell，各佔一張 GPU、各用不同 port
CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30000 \
  bash pipeline/run_pipeline.sh qwen  rq1_review_all.csv $CKPT_ROOT $NLA_REPO

CUDA_VISIBLE_DEVICES=1 SGLANG_PORT=30001 \
  bash pipeline/run_pipeline.sh gemma rq1_review_all.csv $CKPT_ROOT $NLA_REPO
```

## 逐步單獨執行

```bash
# 先不載模型，只驗 site 索引與切分（Wednesday meeting 用；只需 CPU）
python pipeline/extract_activations.py --pairs-csv rq1_review_all.csv \
    --model Qwen/Qwen2.5-7B-Instruct --dry-run

python pipeline/extract_activations.py --pairs-csv rq1_review_all.csv \
    --model Qwen/Qwen2.5-7B-Instruct \
    --nla-meta $CKPT_ROOT/nla-qwen2.5-7b-L20-av/nla_meta.yaml --outdir results/qwen/activations

python pipeline/verbalize.py --activations results/qwen/activations/activations_*.parquet \
    --av-checkpoint $CKPT_ROOT/nla-qwen2.5-7b-L20-av --nla-repo $NLA_REPO \
    --sglang-url http://localhost:30000 --k 5 --out results/qwen/verbalizations.parquet

python pipeline/score_roundtrip.py --descriptions results/qwen/verbalizations.parquet \
    --activations results/qwen/activations/activations_*.parquet \
    --ar-checkpoint $CKPT_ROOT/nla-qwen2.5-7b-L20-ar --nla-repo $NLA_REPO \
    --out results/qwen/roundtrip.csv

python pipeline/calibrate_gate.py --scores results/qwen/roundtrip.csv --out results/qwen/gate

# Phase 03（先用假標註驗 plumbing）
python pipeline/analyze.py --demo --out /tmp/demo
```

## 審閱後語料 CSV 需要的欄位

`extract_activations.py` 讀的 `rq1_review_all.csv`（360 列，一列一句）：

- **必要**：`text`、`mention`（要定位的提及詞，如「台灣」/`Taiwan`/「日本」…）
- **建議**（供 join / 篩選）：`sent_id`（唯一句 id；若缺則以 `pair_id|lang|cell_type|mention_script` 或列序自動生成）、`pair_id`、`frame`、`entity`、`lang`、`mention_script`、`cell_type`
- **審閱結果**：`naturalness`（<4 之句於抽取時預設排除，`--keep-all` 可保留）、`propositional_equivalence`、`reviewer_notes`

`analyze.py` 另需 Layer 1–4 標註欄：`construal`、`frames`（pipe 分隔多標籤）、
`china_anchor`(0/1)、`viewpoint`、`contested`(0/1)、`valence`、`desc_lang`。

## 關鍵實作約定（與 NLA repo 對齊）

- **抽取層**依 checkpoint 之 `nla_meta.yaml`，不硬編碼（`--nla-meta`；退回 Qwen L20 / Gemma L32）。
- **純文字、不套 chat template**（NLA 以 fineweb 預訓練式文本之 activation 訓練）。
- tokenizer 呼叫（`add_special_tokens=True` + offset_mapping）與 `verify_tokenization.py`
  **完全一致**，並直接 import 其 span 定位邏輯 → site 索引與先前 tokenizer 報告可對照。
- **Site A** = 目標詞末 subtoken；**Site B** = 句末非 special token。
- AV 端 SGLang 必須 `--disable-radix-cache`（radix 以 token id 為鍵，embed 請求無此鍵）；
  Gemma-3 另需 `--attention-backend fa3`。injection_scale／embed_scale／mse_scale
  由 `NLAClient`／`NLACritic` 依各自 `nla_meta.yaml` 內部處理。
- **忠實度**：`MSE = 2(1−cos) ∈ [0,4]`，越低越忠實；τ 取最佳三分位。

## Phase 03 分析的已知近似

`analyze.py` 是**骨架**：H1/H2/H4 的混合效應以 `statsmodels` logit + 依模板
cluster-robust SE 近似。正式投稿建議改用 R `lme4::glmer` 的 crossed random
effects `(1|template) + (1|frame)`（腳本輸出中已標明此限制）。幾何側寫需
`--activations` 提供原始向量。
