#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verbalize.py — RQ1 Phase 02 步驟 2：AV（activation verbaliser）自然語言化
=========================================================================

讀 extract_activations.py 產出的 activations parquet（720 向量），對每個
向量以 temperature 取樣 k=5 則自然語言描述（估計描述分布之穩定性），共
720×5 ＝ 3,600 則，輸出描述表。

AV 為 vector→text：把向量當單一 token embedding 注入固定 prompt，autoregress
出描述。實作直接使用 NLA 官方 `nla_inference.py` 的 `NLAClient`（處理
injection_scale、embed_scale、注入位置驗證與 <explanation> 解析），本腳本
只負責批次迴圈、metadata 對齊與續跑。

前置：SGLang server 須先掛起 AV checkpoint（見 run_pipeline.sh / README）：

    python -m sglang.launch_server \
        --model-path kitft/nla-qwen2.5-7b-L20-av \
        --disable-radix-cache            # 必要：radix 以 token id 為鍵，embed 請求無此鍵
        # Gemma-3 另需 --attention-backend fa3

用法：

    python pipeline/verbalize.py \
        --activations activations/qwen/activations_Qwen2.5-7B-Instruct.parquet \
        --av-checkpoint /path/to/nla-qwen2.5-7b-L20-av \
        --nla-repo /path/to/natural_language_autoencoders \
        --sglang-url http://localhost:30000 \
        --k 5 --temperature 0.8 \
        --out verbalizations/qwen_av.parquet

續跑：重跑同指令會略過 --out 中已完成的 (vector_id, sample_k)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def load_activations(path: Path):
    """回傳 (metadata rows: list[dict], vecs: np.ndarray[N, d])。"""
    import numpy as np
    import pyarrow.parquet as pq

    tbl = pq.read_table(path)
    cols = tbl.column_names
    vec_col = tbl.column("activation_vector")
    # FixedSizeList → [N, d]
    flat = vec_col.combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    n = len(tbl)
    vecs = flat.reshape(n, -1)
    meta_cols = [c for c in cols if c != "activation_vector"]
    df = tbl.select(meta_cols).to_pylist()
    return df, vecs


def _parts_dir(out_path: Path) -> Path:
    return out_path.with_suffix(out_path.suffix + ".parts")


def load_done(out_path: Path) -> set[tuple[str, int]]:
    """讀既有輸出（已合併檔 + 未合併 parts），回傳已完成 (vector_id, sample_k) 供續跑。"""
    import pyarrow.parquet as pq
    done: set[tuple[str, int]] = set()
    sources = ([out_path] if out_path.exists() else []) + \
              sorted(_parts_dir(out_path).glob("part-*.parquet"))
    for src in sources:
        t = pq.read_table(src, columns=["vector_id", "sample_k"]).to_pylist()
        done |= {(r["vector_id"], r["sample_k"]) for r in t}
    return done


def append_rows(out_path: Path, rows: list[dict]) -> None:
    """以 parquet dataset 方式增量寫（每批一檔），避免長工作中途遺失。"""
    if not rows:
        return
    import pyarrow as pa
    import pyarrow.parquet as pq
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 增量：寫到 {out}.parts/part-XXXX.parquet，最後 --finalize 合併
    parts = _parts_dir(out_path)
    parts.mkdir(parents=True, exist_ok=True)
    n = len(list(parts.glob("part-*.parquet")))
    pq.write_table(pa.Table.from_pylist(rows), parts / f"part-{n:05d}.parquet")


def finalize(out_path: Path) -> None:
    """合併「既有已合併檔 + 未合併 parts」成單一 parquet（依 desc_id 去重）。"""
    import pyarrow.parquet as pq
    import pyarrow as pa
    parts = _parts_dir(out_path)
    files = ([out_path] if out_path.exists() else []) + sorted(parts.glob("part-*.parquet"))
    if not files:
        print("  無資料可合併。")
        return
    tbl = pa.concat_tables([pq.read_table(f) for f in files], promote_options="default")
    df = tbl.to_pandas().drop_duplicates(subset=["desc_id"], keep="last")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), out_path)
    for f in parts.glob("part-*.parquet"):
        f.unlink()
    print(f"  合併 {len(files)} 個來源 → {out_path}（{len(df)} 列，已去重並清空 parts）")


def main() -> None:
    import numpy as np

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--activations", type=Path, required=True)
    ap.add_argument("--av-checkpoint", required=True, help="AV checkpoint 目錄（含 nla_meta.yaml）")
    ap.add_argument("--nla-repo", type=Path, required=True, help="natural_language_autoencoders repo 路徑")
    ap.add_argument("--sglang-url", default="http://localhost:30000")
    ap.add_argument("--k", type=int, default=5, help="每向量取樣描述數")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--finalize", action="store_true", help="只合併既有 parts，不再生成")
    ap.add_argument("--limit", type=int, default=None, help="只處理前 N 個向量（除錯）")
    args = ap.parse_args()

    if args.finalize:
        finalize(args.out)
        return

    sys.path.insert(0, str(args.nla_repo))
    from nla_inference import NLAClient  # type: ignore

    meta, vecs = load_activations(args.activations)
    if args.limit:
        meta, vecs = meta[:args.limit], vecs[:args.limit]
    done = load_done(args.out)
    print(f"AV 自然語言化：{len(vecs)} 向量 × k={args.k} = {len(vecs) * args.k} 則；"
          f"已完成 {len(done)} 則（續跑）")

    client = NLAClient(args.av_checkpoint, sglang_url=args.sglang_url)

    buf: list[dict] = []
    n_new = 0
    for row, v in zip(meta, vecs):
        vid = row["vector_id"]
        for k in range(args.k):
            if (vid, k) in done:
                continue
            desc = client.generate(
                v.astype(np.float32),
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                extract_explanation=True,
            )
            rec = dict(row)  # 帶入全部 metadata
            rec.update(desc_id=f"{vid}#k{k}", vector_id=vid, sample_k=k,
                       description=desc, vec_norm=float(np.linalg.norm(v)))
            buf.append(rec)
            n_new += 1
            if len(buf) >= 50:
                append_rows(args.out, buf)
                buf = []
                print(f"  ...已生成 {n_new} 則")
    append_rows(args.out, buf)
    print(f"完成本輪 {n_new} 則新描述。執行 --finalize 合併：")
    print(f"  python pipeline/verbalize.py --out {args.out} --finalize "
          f"--activations {args.activations} --av-checkpoint {args.av_checkpoint} "
          f"--nla-repo {args.nla_repo}")
    finalize(args.out)


if __name__ == "__main__":
    main()
