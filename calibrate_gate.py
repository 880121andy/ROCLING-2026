#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate_gate.py — RQ1 Phase 03 Layer 0：τ 忠實度閘門校準
==========================================================

讀 score_roundtrip.py 的分數表，依設計書 §9 Layer 0：

  * 主分析取 MSE **最佳三分位**（tercile）→ τ = MSE 的 33.3 百分位，pass_gate = (mse ≤ τ)。
  * **報告全體 MSE 分布**，分 語言 / 抽取位置 / cell_type / 框架 交叉呈現
    —— 若某語言之描述系統性較不忠實，本身即是發現（不可靜默丟棄）。

輸出：
  * {out}.gated.csv    原表 + pass_gate 欄
  * {out}.summary.md   τ 與分層分布摘要（可貼回報告）

用法：

    python pipeline/calibrate_gate.py \
        --scores scores/qwen_roundtrip.csv \
        --tercile 0.3333 \
        --out scores/qwen

τ 亦可只在 pilot 子集校準後套用到全體：--calibrate-on pilot_ids.txt
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def load_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        r["mse"] = float(r["mse"])
        r["cos"] = float(r.get("cos", "nan") or "nan")
    return rows


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo = int(i)
    frac = i - lo
    return s[lo] if lo + 1 >= len(s) else s[lo] * (1 - frac) + s[lo + 1] * frac


def dist_line(label: str, xs: list[float]) -> str:
    if not xs:
        return f"| {label} | 0 | – | – | – | – |"
    return (f"| {label} | {len(xs)} | {mean(xs):.3f} | {median(xs):.3f} | "
            f"{quantile(xs, 0.25):.3f} | {quantile(xs, 0.75):.3f} |")


def group_dist(rows: list[dict], key: str) -> list[str]:
    groups: dict[str, list[float]] = {}
    for r in rows:
        groups.setdefault(str(r.get(key, "")), []).append(r["mse"])
    lines = [f"\n### MSE 分布 × {key}\n",
             "| 組 | n | mean | median | Q1 | Q3 |", "|---|---|---|---|---|---|"]
    for g, xs in sorted(groups.items()):
        lines.append(dist_line(g or "(空)", xs))
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--tercile", type=float, default=1 / 3,
                    help="通過閘門之 MSE 分位（預設最佳三分位 0.333）")
    ap.add_argument("--calibrate-on", type=Path, default=None,
                    help="只在此檔列出的 desc_id（每行一個）上校準 τ，再套用全體")
    ap.add_argument("--out", type=Path, required=True, help="輸出前綴")
    args = ap.parse_args()

    rows = load_scores(args.scores)
    if not rows:
        raise SystemExit("[錯誤] 分數表為空。")

    calib = rows
    if args.calibrate_on and args.calibrate_on.exists():
        ids = {l.strip() for l in args.calibrate_on.read_text().splitlines() if l.strip()}
        calib = [r for r in rows if r.get("desc_id") in ids] or rows

    tau = quantile([r["mse"] for r in calib], args.tercile)
    for r in rows:
        r["pass_gate"] = int(r["mse"] <= tau)
    n_pass = sum(r["pass_gate"] for r in rows)

    gated = Path(f"{args.out}.gated.csv")
    with gated.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [f"# Layer 0 忠實度閘門摘要", "",
          f"- 分數檔：`{args.scores.name}`；描述數：{len(rows)}",
          f"- 校準集：{'pilot 子集 ' + str(len(calib)) + ' 則' if calib is not rows else '全體'}",
          f"- τ（MSE {args.tercile:.3f} 分位）＝ **{tau:.4f}**",
          f"- 通過閘門：{n_pass}/{len(rows)}（{n_pass / len(rows):.1%}）",
          "",
          "## 全體 MSE 分布",
          "| 全體 | n | mean | median | Q1 | Q3 |",
          "|---|---|---|---|---|---|",
          dist_line("all", [r["mse"] for r in rows])]
    for key in ("lang", "site", "cell_type", "frame", "entity", "desc_lang"):
        if key in rows[0]:
            md += group_dist(rows, key)

    Path(f"{args.out}.summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"τ = {tau:.4f}；通過 {n_pass}/{len(rows)}")
    print(f"  -> {gated}")
    print(f"  -> {args.out}.summary.md")


if __name__ == "__main__":
    main()
