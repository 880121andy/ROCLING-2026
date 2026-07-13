#!/usr/bin/env bash
# ==========================================================================
# run_pilot.sh — τ 校準批小跑（20 句）：抽取→AV→AR→τ→人眼檢視
# ==========================================================================
# 用 select_pilot.py 抽出的小型語料（預設 20 句未標註者）跑一遍 mini-pipeline，
# 把 NLA 翻譯品質看清楚、並把 τ 閾值抓出來。跑量小（20×2×k），分鐘級。
#
# 用法：
#   1) 先抽 20 句：
#      python pipeline/select_pilot.py --pairs-csv rq1_review_all.csv \
#          --n 20 --core-only --out rq1_pilot20.csv
#   2) 小跑（GPU 0，Qwen）：
#      CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30000 \
#        bash pipeline/run_pilot.sh qwen rq1_pilot20.csv $CKPT_ROOT $NLA_REPO
#
# 環境變數：K（每向量描述數，預設 5）、SGLANG_PORT（預設 30000）
# --------------------------------------------------------------------------
set -euo pipefail

MODEL_KEY="${1:?需指定 qwen 或 gemma}"
PILOT_CSV="${2:?需指定 select_pilot.py 產出的 pilot CSV}"
CKPT_ROOT="${3:?需指定 checkpoint 根目錄}"
NLA_REPO="${4:?需指定 natural_language_autoencoders repo 路徑}"

K="${K:-5}"
SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_URL="http://localhost:${SGLANG_PORT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
# 共用腳本（extract/verbalize/score/calibrate）住在 ../main_script；pilot_preview.py 才在本目錄
MAIN="${MAIN_SCRIPT_DIR:-$(cd "$HERE/../main_script" && pwd)}"
PY="${PYTHON:-python}"

case "$MODEL_KEY" in
  qwen)  HF_MODEL="Qwen/Qwen2.5-7B-Instruct"; AV="$CKPT_ROOT/nla-qwen2.5-7b-L20-av"; AR="$CKPT_ROOT/nla-qwen2.5-7b-L20-ar"; EXTRA="" ;;
  gemma) HF_MODEL="google/gemma-3-12b-it";    AV="$CKPT_ROOT/nla-gemma3-12b-L32-av"; AR="$CKPT_ROOT/nla-gemma3-12b-L32-ar"; EXTRA="--attention-backend fa3" ;;
  *) echo "未知模型：$MODEL_KEY"; exit 1 ;;
esac

OUT="results/pilot_${MODEL_KEY}"
mkdir -p "$OUT" logs
echo "=== [PILOT $MODEL_KEY] $(wc -l < "$PILOT_CSV") 列 · k=$K · port=$SGLANG_PORT ==="

echo "--- 1/5 抽取 activations ---"
"$PY" "$MAIN/extract_activations.py" --pairs-csv "$PILOT_CSV" --model "$HF_MODEL" \
    --nla-meta "$AV/nla_meta.yaml" --outdir "$OUT/activations" --keep-all
ACT=$(ls "$OUT"/activations/activations_*.parquet | head -1)

echo "--- 2/5 launch AV SGLang ---"
"$PY" -m sglang.launch_server --model-path "$AV" --port "$SGLANG_PORT" \
    --disable-radix-cache $EXTRA > "logs/sglang_pilot_${MODEL_KEY}.log" 2>&1 &
SGLANG_PID=$!
trap 'kill $SGLANG_PID 2>/dev/null || true' EXIT
for i in $(seq 1 120); do
    curl -sf "$SGLANG_URL/health" >/dev/null 2>&1 && { echo "  server 就緒"; break; }
    sleep 5; [ "$i" = 120 ] && { echo "SGLang 逾時，見 logs/sglang_pilot_${MODEL_KEY}.log"; exit 1; }
done

echo "--- 3/5 AV verbalize (k=$K) ---"
"$PY" "$MAIN/verbalize.py" --activations "$ACT" --av-checkpoint "$AV" \
    --nla-repo "$NLA_REPO" --sglang-url "$SGLANG_URL" --k "$K" --temperature 0.8 \
    --out "$OUT/verbalizations.parquet"

echo "--- 4/5 AR round-trip ---"
"$PY" "$MAIN/score_roundtrip.py" --descriptions "$OUT/verbalizations.parquet" \
    --activations "$ACT" --ar-checkpoint "$AR" --nla-repo "$NLA_REPO" \
    --out "$OUT/roundtrip.csv"

echo "--- 5/5 τ 校準 + 人眼檢視 ---"
"$PY" "$MAIN/calibrate_gate.py" --scores "$OUT/roundtrip.csv" --out "$OUT/gate"
TAU=$("$PY" - "$OUT/gate.gated.csv" <<'PYEOF'
import csv, sys, statistics
rows=list(csv.DictReader(open(sys.argv[1],encoding="utf-8-sig")))
mses=sorted(float(r["mse"]) for r in rows)
i=int((1/3)*(len(mses)-1)); print(f"{mses[i]:.4f}")
PYEOF
)
"$PY" "$HERE/pilot_preview.py" --descriptions "$OUT/verbalizations.parquet" \
    --scores "$OUT/roundtrip.csv" --tau "$TAU" --out "$OUT/preview.md"

echo ""
echo "=========================================================="
echo " PILOT 完成（$MODEL_KEY）"
echo "   τ（MSE 最佳三分位）＝ $TAU"
echo "   τ 分層分布   -> $OUT/gate.summary.md"
echo "   人眼檢視翻譯 -> $OUT/preview.md   ← 先看這份判斷 NLA OK 不 OK"
echo "   忠實度明細   -> $OUT/roundtrip.csv"
echo "=========================================================="
