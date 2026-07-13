#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot_preview.py — 校準批的人眼檢視：句子 → AV 描述 → AR 忠實度
================================================================

把 verbalize.py 的描述表與 score_roundtrip.py 的忠實度分數，依原句 join 起來，
輸出一份好讀的 markdown：每句列出原文、兩個抽取位置、各 k 則 AV 描述及其
round-trip MSE／cos。用來目測「NLA 翻譯 OK 不 OK」，再決定 τ 是否合理。

用法：

    python pipeline/pilot_preview.py \
        --descriptions results/pilot_qwen/verbalizations.parquet \
        --scores results/pilot_qwen/roundtrip.csv \
        --out results/pilot_qwen/preview.md
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_descriptions(path: Path) -> dict[str, dict]:
    import pyarrow.parquet as pq
    return {r["desc_id"]: r for r in pq.read_table(path).to_pylist()}


def load_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        r["mse"] = float(r["mse"])
        r["cos"] = float(r.get("cos", "nan") or "nan")
    return rows


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--descriptions", type=Path, required=True)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--tau", type=float, default=None, help="標出低於 τ（通過閘門）的描述")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    desc = load_descriptions(args.descriptions)
    scores = load_scores(args.scores)

    # 依原句聚合：sent_id -> site -> list[(k, mse, cos, desc_lang, description)]
    by_sent: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    text_of: dict[str, str] = {}
    meta_of: dict[str, dict] = {}
    for s in scores:
        d = desc.get(s["desc_id"], {})
        sid = s.get("sent_id") or d.get("sent_id", "?")
        by_sent[sid][s.get("site", "?")].append(
            (s.get("sample_k", d.get("sample_k")), s["mse"], s["cos"],
             s.get("desc_lang", ""), d.get("description", "")))
        text_of.setdefault(sid, d.get("text", ""))
        meta_of.setdefault(sid, {k: (s.get(k) or d.get(k, "")) for k in
                                  ("frame", "lang", "entity", "cell_type")})

    all_mse = [s["mse"] for s in scores]
    md = ["# 校準批 AV／AR 人眼檢視", "",
          f"- 句數：{len(by_sent)}；描述數：{len(scores)}",
          f"- 全體 MSE：mean={mean(all_mse):.3f}  min={min(all_mse):.3f}  max={max(all_mse):.3f}",
          f"- 依語言：en mean={mean([s['mse'] for s in scores if s.get('lang')=='en']):.3f}；"
          f"zh mean={mean([s['mse'] for s in scores if s.get('lang')=='zh']):.3f}",
          (f"- τ＝{args.tau:.4f}（★ = 通過閘門）" if args.tau is not None else ""),
          "\n> MSE = 2(1−cos) ∈ [0,4]，越低越忠實；描述若跑題／全 CJK 亂碼＝抽取或 scale 有誤。\n"]

    for sid in sorted(by_sent):
        m = meta_of[sid]
        md.append(f"\n## {sid}　[{m['frame']} · {m['lang']} · {m['entity']}]")
        md.append(f"> {text_of[sid]}")
        for site in sorted(by_sent[sid]):
            md.append(f"\n**Site {site}**")
            md.append("| k | MSE | cos | 描述語言 | AV 描述 |")
            md.append("|---|---|---|---|---|")
            for k, mse, cos, dl, description in sorted(by_sent[sid][site], key=lambda x: x[1]):
                star = " ★" if (args.tau is not None and mse <= args.tau) else ""
                desc_cell = (description or "").replace("\n", " ").replace("|", "\\|")
                if len(desc_cell) > 160:
                    desc_cell = desc_cell[:157] + "…"
                md.append(f"| {k} | {mse:.3f}{star} | {cos:.3f} | {dl} | {desc_cell} |")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"人眼檢視 -> {args.out}")


if __name__ == "__main__":
    main()
