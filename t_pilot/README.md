# t_pilot — τ 校準批小跑（上機前的品管閘門）

正式全量跑（`main_script/run_pipeline.sh`，360 句）之前，先用這裡的 20 句
mini-pipeline 把兩件事確認好：

1. **NLA 翻譯品質 OK 不 OK**（肉眼看 `preview.md`）
2. **τ 閾值的數量級合不合理**（`calibrate_gate.py` 算出的最佳三分位）

跑量小（20 句 × 2 site × k=5 ≈ 200 則描述），分鐘級。**這關過了才往全量走。**

---

## 檔案

| 檔案 | 作用 |
|---|---|
| `select_pilot.py` | 從 360 句抽 20 句「乾淨句」（reviewer_notes 空、跨 frame×lang 分層） |
| `rq1_pilot20.csv` | 已附的 20 句 pilot 語料（可直接用，不必重抽） |
| `run_pilot.sh` | 小跑 orchestration：抽取→AV→AR→τ→人眼檢視 |
| `pilot_preview.py` | 把描述 + 忠實度 join 成好讀的 `preview.md` |

> `run_pilot.sh` 會去 `../main_script/` 抓 `extract_activations.py`、`verbalize.py`、
> `score_roundtrip.py`、`calibrate_gate.py`（可用環境變數 `MAIN_SCRIPT_DIR` 覆寫）。

---

## 上機檢查清單（照順序，每關是下一關的前提）

### 0) 裝環境（見 `main_script/README.md` 前置環境）

```bash
pip install -U "transformers>=4.50" torch pyarrow pandas numpy statsmodels scipy \
               httpx orjson pyyaml safetensors "sglang[all]"
git clone https://github.com/kitft/natural_language_autoencoders
bash natural_language_autoencoders/patches/apply_sglang_patches.sh
export HF_HOME=/work/$USER/hf_cache
huggingface-cli login          # Gemma 為 gated，需先於網頁同意授權
export CKPT_ROOT=/path/to/checkpoints      # 內含 nla-qwen2.5-7b-L20-{av,ar} …
export NLA_REPO=/path/to/natural_language_autoencoders
```

### 1) `--dry-run`：先驗 site 索引與切分（只吃 CPU，不載模型）

```bash
python ../main_script/extract_activations.py --pairs-csv ../rq1_review_all.csv \
    --model Qwen/Qwen2.5-7B-Instruct --dry-run
```

看 site A（目標詞末 subtoken）／site B（句末非 special token）索引對不對。

### 2) 跑 pilot 小跑（GPU 0、Qwen）

```bash
CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30000 \
  bash run_pilot.sh qwen rq1_pilot20.csv $CKPT_ROOT $NLA_REPO
```

> 想重抽 pilot 語料時：
> `python select_pilot.py --pairs-csv ../rq1_review_all.csv --n 20 --core-only --out rq1_pilot20.csv`

### 3) 人眼驗收（**最重要的一關**）

打開 `results/pilot_qwen/preview.md`，逐句看 AV 描述是否 OK（判準見下節）。
順便看印出來的 `τ`、以及 `results/pilot_qwen/gate.summary.md` 的分層分布。

### 4) 沒問題才上全量

```bash
CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30000 \
  bash ../main_script/run_pipeline.sh qwen  ../rq1_review_all.csv $CKPT_ROOT $NLA_REPO
CUDA_VISIBLE_DEVICES=1 SGLANG_PORT=30001 \
  bash ../main_script/run_pipeline.sh gemma ../rq1_review_all.csv $CKPT_ROOT $NLA_REPO
```

---

## 「翻譯 OK」怎麼判？

忠實度指標：**`MSE = 2(1−cos) ∈ [0,4]`，越低越忠實**。但 pilot 階段是**質性肉眼判**
（量化 τ 留給全量），主要看三件事：

1. **不是亂碼／不跑題** — 描述若整段是無意義的 CJK 亂碼、或內容跟原句主題完全無關，
   通常代表**抽取或 scale 有誤（site 索引錯、injection/embed scale 沒對上）**，
   這是系統性 bug，不是個別句子問題，要先修 pipeline。
2. **抓到該位置該有的語意** — Site A 的描述應圍繞「目標實體／提及詞」附近的語意；
   Site B（句末）反映整句累積語意。描述抓到相關概念即算堪用，不要求逐字對應。
3. **MSE 分布合理且無系統性偏斜** — 看 `gate.summary.md`：若某語言（如 zh）整體 MSE
   明顯高於另一語言，**這本身可能是發現，不可靜默丟棄**（設計書 §9）；但若是「全部句子
   MSE 都爆高」則多半是 bug。

判斷口訣：**先分「系統性壞」還是「個別句差」**——前者（亂碼/全體爆高/整語言偏斜到不合理）
要回頭修抽取；後者（多數句合理、少數偏高）屬正常，正是 τ 閘門之後要篩掉的對象，放行即可。

---

## 關於 τ：pilot 的 τ 只是參考

`run_pilot.sh` 印的 τ 是 20 句的最佳三分位，**樣本太少、抽樣噪音大，只當數量級 sanity
check**。正式閘門的 τ 應在**全量 360 句跑完後用完整 MSE 分布重定**。若要沿用 pilot 校準值，
`calibrate_gate.py --calibrate-on pilot_ids.txt` 可「只在 pilot 子集校準 τ 再套用全體」。
