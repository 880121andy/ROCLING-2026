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

activations_*.parquet ──┐
rq1_review_all.csv    ──┴─ steer.py（因果驗證，不需 AV/AR）→ steer.report.md
```

## 腳本與 pipeline 階段對照

| 腳本 | experiment_pipeline.md 對應 | 產物 |
|---|---|---|
| `extract_activations.py` | 02 · Activation 抽取（720 向量） | `activations_*.parquet` |
| `verbalize.py` | 02 · AV 自然語言化（3,600 則） | `verbalizations.parquet` |
| `score_roundtrip.py` | 02 · AR round-trip 忠實度 | `roundtrip.csv` |
| `calibrate_gate.py` | 03 · Layer 0 忠實度閘門（τ） | `gate.gated.csv`, `gate.summary.md` |
| `analyze.py` | 03 · H1/H2/H4、漂移、H3、幾何、H5 | `analysis.report.md` |
| `steer.py` | 03 · **因果驗證**：Δ_lang activation steering | `steer.{scores,generations}.csv`, `steer.report.md` |
| `run_pipeline.sh` | 02→03 單模型端到端 orchestration | 上述全部 |

## 因果驗證（`steer.py`）——把相關性升級成因果

`analyze.py` 給的是「en 語境的向量描述較常落入 geopolitical frame」這種**相關**證據；
`steer.py` 直接把 Δ_lang ＝ mean(en) − mean(zh) 注入中文語境的殘差流，看下游行為是否偏移
（教授指示的最後一項）。只需原始 activations parquet 與語料 CSV，**不需要 AV/AR checkpoint、
不需要 SGLang**，因此可與 NLA 那條鏈分開跑。

主估計量是逐題配對的 delta-of-delta：

```
S_i(α) = mean logP(地緣政治續句 | 中文前綴 i, α) − mean logP(日常生活續句 | 中文前綴 i, α)
效應   = S_i(α) − S_i(0)          ← α=0 的組內相減抵消續句先天詞頻／長度偏誤
```

控制組（證明力來源）：劑量反應 α∈{−2,−1,0,1,2}、**反向注入 α<0 必須反向**、
同 norm 隨機方向（`--n-random`，預設 **5 個獨立 seed**——單一 seed 可能碰巧落在
geo／日常的 logit 軸上）、控制實體（日本／冰島）的 Δ、以及 Δ_台灣 投影掉
**span{各控制實體 Δ}** 後的**台灣特異殘差分量**（實測 cos(Δ_台灣, Δ_控制) ≈ 0.64(Qwen)／
0.79(Gemma)，故這條才是真正在做事的對照）；另有逐 token 平均 logP 的流暢度護欄，
避免把「模型被弄壞」誤讀成效應。去循環性以 LOFO 處理：測試前綴屬框架 f 時，
Δ 只用非 f 的句子估計。

⚠️ **norm-matching（`--match-norm`，預設開）**：各控制向量一律縮放到 ‖Δ_lang‖ 再注入。
殘差向量原始長度只有 ‖Δ_lang‖ 的 **0.77(Qwen)／0.61(Gemma)**，不縮放的話同一個 α 是
**較弱的推力**，弱結果就無法區分「沒有台灣特異效應」與「只是推得比較輕」。報告會列出
各條件縮放前後的 norm，故 per-unit-norm 的解讀仍可還原（`--no-match-norm` 保留舊行為）。

```bash
# 0) 上機第一件事：驗證注入點 == 抽取點（hidden_states[L] 由 layers[L-1] 產生）
python main_script/steer.py --verify-hook \
    --model Qwen/Qwen2.5-7B-Instruct --nla-meta $CKPT_ROOT/nla-qwen2.5-7b-L20-av/nla_meta.yaml

# 1) 主實驗
python main_script/steer.py \
    --activations results/qwen/activations/activations_Qwen2.5-7B-Instruct.parquet \
    --pairs-csv rq1_review_all.csv --model Qwen/Qwen2.5-7B-Instruct \
    --nla-meta $CKPT_ROOT/nla-qwen2.5-7b-L20-av/nla_meta.yaml --out results/qwen/steer

# Gemma 同理（--model google/gemma-3-12b-it，layers 在 language_model 下，腳本自動解析）
# 不載模型只看 Δ 與前綴：--dry-run ／ 無 torch 邏輯自測：--self-test

# 2) 最終投稿版：把樣本數拉滿（語料每框架 12–18 句中文 baseline，共 108 句可用）
python main_script/steer.py ... --per-frame 12 --gen-prefixes 24
```

主要旗標：`--per-frame`（每框架前綴數，預設 6 ＝ n 48；檢定的 n 是**前綴數**）、
`--gen-prefixes`（只對前 N 條前綴做貪婪生成，預設 12——生成是 GPU 主成本，打分仍用全部
前綴）、`--n-random`（隨機控制 seed 數，預設 5）、`--match-norm`／`--no-match-norm`、
`--resid-basis span|pooled`、`--alphas`、`--positions`。

⚠️ **α 的尺度不可跨模型直接比**：實測 ‖Δ_lang‖／平均‖h‖ 在 Qwen L20 約 **0.46**、
Gemma L32 約 **0.12**（Gemma 的 embed 有 √d 縮放，殘差流量級大）。同一個 α 在 Gemma 是
**弱得多的推力**，若 α=±1 看似無效應，先用更寬的 `--alphas -8 -4 -2 0 2 4 8` 確認是劑量
不足而非沒有效應；跨模型請比較**劑量反應曲線**，不要比單一 α 的點值（腳本啟動時會自動提醒）。

⚠️ **`--verify-hook` 不可略過**：`extract_activations.py` 取 `hidden_states[L]`，
而 `hidden_states[0]` 是 embedding 輸出 → 對應的是 `layers[L-1]` 的輸出。差一層，
整份因果證據就作廢。另注意 `hidden_states[n_layers]`（最後一層）已過 final norm，
與 `layers[n_layers-1]` 的輸出**不相等**——Qwen L20/28、Gemma L32/48 皆為中間層，不受影響。

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
  bash main_script/run_pipeline.sh qwen  rq1_review_all.csv $CKPT_ROOT $NLA_REPO

CUDA_VISIBLE_DEVICES=1 SGLANG_PORT=30001 \
  bash main_script/run_pipeline.sh gemma rq1_review_all.csv $CKPT_ROOT $NLA_REPO
```

## 逐步單獨執行

```bash
# 先不載模型，只驗 site 索引與切分（Wednesday meeting 用；只需 CPU）
python main_script/extract_activations.py --pairs-csv rq1_review_all.csv \
    --model Qwen/Qwen2.5-7B-Instruct --dry-run

python main_script/extract_activations.py --pairs-csv rq1_review_all.csv \
    --model Qwen/Qwen2.5-7B-Instruct \
    --nla-meta $CKPT_ROOT/nla-qwen2.5-7b-L20-av/nla_meta.yaml --outdir results/qwen/activations

python main_script/verbalize.py --activations results/qwen/activations/activations_*.parquet \
    --av-checkpoint $CKPT_ROOT/nla-qwen2.5-7b-L20-av --nla-repo $NLA_REPO \
    --sglang-url http://localhost:30000 --k 5 --out results/qwen/verbalizations.parquet

python main_script/score_roundtrip.py --descriptions results/qwen/verbalizations.parquet \
    --activations results/qwen/activations/activations_*.parquet \
    --ar-checkpoint $CKPT_ROOT/nla-qwen2.5-7b-L20-ar --nla-repo $NLA_REPO \
    --out results/qwen/roundtrip.csv

python main_script/calibrate_gate.py --scores results/qwen/roundtrip.csv --out results/qwen/gate

# Phase 03（先用假標註驗 plumbing）
python main_script/analyze.py --demo --out /tmp/demo
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
