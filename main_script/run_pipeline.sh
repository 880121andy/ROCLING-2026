#!/usr/bin/env bash
# ==========================================================================
# run_pipeline.sh — RQ1 Phase 02→03 單模型端到端管線（TWCC / 國網中心）
# ==========================================================================
# 一個模型跑一條鏈：抽取 → 掛 AV SGLang → AV 自然語言化 → AR 忠實度 →
#                    τ 閘門 → Phase 03 分析（骨架）。
# Qwen 與 Gemma 兩模型「平行」：開兩個 shell、各指定不同 GPU 與 PORT 即可。
#
# 用法：
#   # GPU 0，Qwen（port 30000）
#   CUDA_VISIBLE_DEVICES=0 SGLANG_PORT=30000 \
#     bash pipeline/run_pipeline.sh qwen /path/to/rq1_review_all.csv /path/to/ckpt_root /path/to/nla_repo
#
#   # GPU 1，Gemma（port 30001），另一個 shell 同時跑
#   CUDA_VISIBLE_DEVICES=1 SGLANG_PORT=30001 \
#     bash pipeline/run_pipeline.sh gemma /path/to/rq1_review_all.csv /path/to/ckpt_root /path/to/nla_repo
#
# ckpt_root 下應有（用 nla/scripts/pull_checkpoint.py 事先下載）：
#   nla-qwen2.5-7b-L20-av/   nla-qwen2.5-7b-L20-ar/
#   nla-gemma3-12b-L32-av/   nla-gemma3-12b-L32-ar/
# --------------------------------------------------------------------------
set -euo pipefail

MODEL_KEY="${1:?需指定 qwen 或 gemma}"
PAIRS_CSV="${2:?需指定審閱後語料 CSV}"
CKPT_ROOT="${3:?需指定 checkpoint 根目錄}"
NLA_REPO="${4:?需指定 natural_language_autoencoders repo 路徑}"

SGLANG_PORT="${SGLANG_PORT:-30000}"
SGLANG_URL="http://localhost:${SGLANG_PORT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python}"

SGLANG_PID=""

# 關閉 SGLang server 及其所有 worker，等到 GPU 顯存真的釋放才返回。
# 單卡時必要：AV server（~68 GiB）要先退場，AR 步驟才有空間載入同一張卡。
# 只殺「本 server 的 process group」（步驟 2 用 setsid，PGID==leader PID），
# 平行跑的另一模型（不同 port/不同 PGID）不受影響。
stop_sglang() {
    [ -n "${SGLANG_PID:-}" ] || return 0
    local pid="$SGLANG_PID"; SGLANG_PID=""   # 先清空，避免 EXIT trap 重複進來
    echo "  釋放顯存：關閉 SGLang（PGID $pid）…"
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true

    # 等整個 process group 消失（持有顯存的是 worker，不能靠 cmdline 比對）
    local gone="" i
    for i in $(seq 1 24); do                 # 最長 ~120s
        pgrep -g "$pid" >/dev/null 2>&1 || { gone=1; break; }
        sleep 5
    done
    if [ -z "$gone" ]; then
        echo "  [警告] 120s 未完全退出，改送 SIGKILL。"
        kill -KILL -"$pid" 2>/dev/null || true
        sleep 5
    fi

    sleep 3                                   # 讓 driver 回收顯存
    if command -v nvidia-smi >/dev/null 2>&1; then
        local gpu="${CUDA_VISIBLE_DEVICES:-}"; gpu="${gpu%%,*}"
        echo -n "  顯存 free/total： "
        nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader \
            ${gpu:+-i "$gpu"} 2>/dev/null | head -1 || echo "(查詢失敗)"
    fi
}

case "$MODEL_KEY" in
  qwen)
    HF_MODEL="Qwen/Qwen2.5-7B-Instruct"
    AV_CKPT="$CKPT_ROOT/nla-qwen2.5-7b-L20-av"
    AR_CKPT="$CKPT_ROOT/nla-qwen2.5-7b-L20-ar"
    EXTRA_SGLANG=""
    ;;
  gemma)
    HF_MODEL="google/gemma-3-12b-it"
    AV_CKPT="$CKPT_ROOT/nla-gemma3-12b-L32-av"
    AR_CKPT="$CKPT_ROOT/nla-gemma3-12b-L32-ar"
    EXTRA_SGLANG="--attention-backend fa3"   # Gemma-3 需 fa3 以免 OOM
    ;;
  *) echo "未知模型：$MODEL_KEY（qwen|gemma）"; exit 1 ;;
esac

OUT="results/${MODEL_KEY}"
mkdir -p "$OUT" logs
echo "=== [$MODEL_KEY] $HF_MODEL  port=$SGLANG_PORT  out=$OUT ==="

# --- 步驟 1：抽取 720 向量 -------------------------------------------------
echo "--- 1/5 抽取 activations ---"
"$PY" "$HERE/extract_activations.py" \
    --pairs-csv "$PAIRS_CSV" --model "$HF_MODEL" \
    --nla-meta "$AV_CKPT/nla_meta.yaml" --outdir "$OUT/activations"
ACT_PARQUET=$(ls "$OUT"/activations/activations_*.parquet | head -1)

# --- 步驟 2：掛 AV SGLang server（背景），並在結束時關掉 --------------------
echo "--- 2/5 launch AV SGLang server ---"
setsid "$PY" -m sglang.launch_server \
    --model-path "$AV_CKPT" --port "$SGLANG_PORT" \
    --disable-radix-cache $EXTRA_SGLANG \
    > "logs/sglang_${MODEL_KEY}.log" 2>&1 &
SGLANG_PID=$!
trap stop_sglang EXIT

echo "等待 server 就緒（$SGLANG_URL）…"
for i in $(seq 1 120); do
    if curl -sf "$SGLANG_URL/health" >/dev/null 2>&1; then echo "  就緒。"; break; fi
    sleep 5
    [ "$i" = 120 ] && { echo "SGLang 啟動逾時，見 logs/sglang_${MODEL_KEY}.log"; exit 1; }
done

# --- 步驟 3：AV 自然語言化（720×5＝3600 則）-------------------------------
echo "--- 3/5 AV verbalize (k=5) ---"
"$PY" "$HERE/verbalize.py" \
    --activations "$ACT_PARQUET" --av-checkpoint "$AV_CKPT" \
    --nla-repo "$NLA_REPO" --sglang-url "$SGLANG_URL" \
    --k 5 --temperature 0.8 --out "$OUT/verbalizations.parquet"

# --- 步驟 3.5：AV 完成，關閉 server 釋放顯存給 AR（單卡必要）--------------
echo "--- AV 完成，關閉 server 釋放顯存 ---"
stop_sglang

# --- 步驟 4：AR round-trip 忠實度（不需 server）---------------------------
echo "--- 4/5 AR round-trip score ---"
"$PY" "$HERE/score_roundtrip.py" \
    --descriptions "$OUT/verbalizations.parquet" --activations "$ACT_PARQUET" \
    --ar-checkpoint "$AR_CKPT" --nla-repo "$NLA_REPO" \
    --out "$OUT/roundtrip.csv"

# --- 步驟 5：τ 閘門校準 ----------------------------------------------------
echo "--- 5/5 Layer 0 gate (τ) ---"
"$PY" "$HERE/calibrate_gate.py" --scores "$OUT/roundtrip.csv" --out "$OUT/gate"

cat <<EOF

完成 Phase 02（$MODEL_KEY）。產物：
  $ACT_PARQUET
  $OUT/verbalizations.parquet     (3600 則描述)
  $OUT/roundtrip.csv              (忠實度分數)
  $OUT/gate.gated.csv             (+ pass_gate)
  $OUT/gate.summary.md            (τ 與分層 MSE 分布)

Phase 03：待 Layer 1–4 標註併入 gate.gated.csv 後
  $PY $HERE/analyze.py --annotated <annotated.csv> --activations $ACT_PARQUET --gated-only --out $OUT/analysis
（先驗 plumbing：$PY $HERE/analyze.py --demo --out /tmp/demo）
EOF
