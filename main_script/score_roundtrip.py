#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
score_roundtrip.py — RQ1 Phase 02 步驟 3：AR round-trip 忠實度
==============================================================

對 verbalize.py 產出的每一則描述，用 AR（activation reconstructor）讀描述、
譯回向量，與該描述所本的**原始向量**比對，得 round-trip 分數：

    MSE = ((pred_n − gold_n)**2).mean() = 2(1 − cos)  ∈ [0, 4]（越低越忠實）
    cos = 方向餘弦相似度

實作使用 NLA 官方 `NLACritic.score(explanation, original) -> (mse, cos)`
（內部把 pred 與 gold 皆 L2-normalize 至 mse_scale 再算，維度無關）。

這是設計書 §9 Layer 0 忠實度閘門的原始分數；τ 閾值篩選見 calibrate_gate.py。
AR 在本機 in-process 跑（載 K+1 層 backbone），**不需** SGLang server。

用法：

    python pipeline/score_roundtrip.py \
        --descriptions verbalizations/qwen_av.parquet \
        --activations activations/qwen/activations_Qwen2.5-7B-Instruct.parquet \
        --ar-checkpoint /path/to/nla-qwen2.5-7b-L20-ar \
        --nla-repo /path/to/natural_language_autoencoders \
        --out scores/qwen_roundtrip.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_activations_index(path: Path):
    """vector_id -> 原始向量 np.ndarray[d]。"""
    import numpy as np
    import pyarrow.parquet as pq
    tbl = pq.read_table(path)
    ids = tbl.column("vector_id").to_pylist()
    flat = (tbl.column("activation_vector").combine_chunks().flatten()
            .to_numpy(zero_copy_only=False).astype(np.float32))
    vecs = flat.reshape(len(tbl), -1)
    return {vid: vecs[i] for i, vid in enumerate(ids)}


def load_descriptions(path: Path) -> list[dict]:
    import pyarrow.parquet as pq
    return pq.read_table(path).to_pylist()


def detect_desc_lang(text: str) -> str:
    """粗略描述語言標記（L4）：依 CJK 字元比例分 zh／en／mixed。"""
    if not text:
        return "empty"
    cjk = sum(1 for ch in text if "㐀" <= ch <= "鿿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    tot = cjk + latin
    if tot == 0:
        return "other"
    r = cjk / tot
    return "zh" if r > 0.6 else "en" if r < 0.15 else "mixed"


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--descriptions", type=Path, required=True)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--ar-checkpoint", required=True, help="AR checkpoint 目錄（含 nla_meta.yaml）")
    ap.add_argument("--nla-repo", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(args.nla_repo))
    from nla_inference import NLACritic  # type: ignore

    vec_of = load_activations_index(args.activations)
    descs = load_descriptions(args.descriptions)
    if args.limit:
        descs = descs[:args.limit]
    print(f"AR round-trip：{len(descs)} 則描述 → 忠實度分數")

    critic = NLACritic(args.ar_checkpoint, device=args.device)

    # 保留供分析的 metadata 欄（存在才寫）
    carry = ["desc_id", "vector_id", "sample_k", "sent_id", "pair_id", "frame",
             "entity", "lang", "mention_script", "cell_type", "site"]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_missing = 0
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        w = None
        for i, d in enumerate(descs):
            vid = d["vector_id"]
            gold = vec_of.get(vid)
            if gold is None:
                n_missing += 1
                continue
            mse, cos = critic.score(d["description"], gold.astype(np.float32))
            rec = {k: d.get(k, "") for k in carry}
            rec.update(mse=round(float(mse), 6), cos=round(float(cos), 6),
                       desc_lang=detect_desc_lang(d.get("description", "")),
                       desc_len=len(d.get("description", "") or ""))
            if w is None:
                w = csv.DictWriter(f, fieldnames=list(rec.keys()))
                w.writeheader()
            w.writerow(rec)
            if (i + 1) % 200 == 0:
                print(f"  ...{i + 1}/{len(descs)}")
    print(f"完成 → {args.out}（缺原始向量而略過 {n_missing} 則）")


if __name__ == "__main__":
    main()
