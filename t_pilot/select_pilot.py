#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
select_pilot.py — 從 360 句抽 τ 校準批（pilot）
================================================

依設計書 §12：τ 閾值校準批建議 ≈ 核心語料 10% ≈ 20 句。本腳本從審閱表挑
**尚未被標註 reviewer_notes**（即審閱者未提出問題、視為乾淨）的句子，跨
框架×語言分層抽樣 N 句，輸出成一份小型語料 CSV，供 run_pilot.sh 小跑
AV／AR、把 τ 抓出來、順便目測 NLA 翻譯品質。

用法：

    python pipeline/select_pilot.py \
        --pairs-csv rq1_review_all.csv \
        --n 20 --out rq1_pilot20.csv

    # 只從台灣核心 baseline 抽（§12 之「核心語料 10%」）：
    python pipeline/select_pilot.py --pairs-csv rq1_review_all.csv \
        --n 20 --core-only --out rq1_pilot20.csv

自測（合成 360 列，無檔案）：

    python pipeline/select_pilot.py --self-test
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def is_unreviewed(row: dict) -> bool:
    """reviewer_notes 空（或無此欄）視為未標註／乾淨。"""
    return not (row.get("reviewer_notes") or "").strip()


def naturalness_ok(row: dict, min_score: float) -> bool:
    v = (row.get("naturalness") or "").strip()
    if not v:
        return True
    try:
        return float(v) >= min_score
    except ValueError:
        return True


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    """
    跨 (frame, lang) 分層、以 round-robin 抽 n 列，確保 τ 不偏於單一框架。
    決定性：以 seed 打散各層內順序與層的走訪順序。
    """
    import random
    rnd = random.Random(seed)
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[(r.get("frame", ""), r.get("lang", ""))].append(r)
    for b in buckets.values():
        rnd.shuffle(b)
    keys = list(buckets)
    rnd.shuffle(keys)

    picked: list[dict] = []
    while len(picked) < n and any(buckets[k] for k in keys):
        for k in keys:
            if buckets[k]:
                picked.append(buckets[k].pop())
                if len(picked) >= n:
                    break
    return picked


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("[錯誤] 沒有符合條件的句子可抽（reviewer_notes 全被標註？）")
    # 欄位取所有列鍵之聯集，保序以第一列為主
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def composition(rows: list[dict]) -> str:
    by = defaultdict(int)
    for r in rows:
        by[(r.get("frame", "?"), r.get("lang", "?"))] += 1
    return ", ".join(f"{f}/{l}:{c}" for (f, l), c in sorted(by.items()))


def select(rows: list[dict], n: int, core_only: bool, min_naturalness: float,
           seed: int) -> list[dict]:
    has_notes_col = any("reviewer_notes" in r for r in rows)
    elig = [r for r in rows if is_unreviewed(r) and naturalness_ok(r, min_naturalness)]
    if core_only:
        elig = [r for r in elig
                if (r.get("cell_type", "baseline") == "baseline")
                and (r.get("entity", "") in ("台灣", "Taiwan", "TW", ""))]
    if not has_notes_col:
        print("  [警告] CSV 無 reviewer_notes 欄，視全部為未標註。")
    if len(elig) < n:
        print(f"  [警告] 合格句只有 {len(elig)} < {n}，將全數採用。")
    return stratified_sample(elig, min(n, len(elig)), seed)


def self_test() -> None:
    rows = []
    frames = ["GEO", "POL-INT", "POL-DOM", "ECON", "CUL", "HIST", "LIFE", "TRAV"]
    for fi, frame in enumerate(frames):
        for t in range(12):
            for lang in ("zh", "en"):
                # 每 7 句放一則 reviewer_notes（不合格）
                note = "太翻譯腔" if (fi * 12 + t) % 7 == 0 else ""
                rows.append(dict(pair_id=f"{frame}-{t:02d}", frame=frame, entity="台灣",
                                 lang=lang, cell_type="baseline", mention_script="hanzi" if lang == "zh" else "latin",
                                 mention="台灣" if lang == "zh" else "Taiwan",
                                 text=f"{frame}-{t}-{lang}", naturalness="5",
                                 reviewer_notes=note))
    picked = select(rows, 20, core_only=True, min_naturalness=4.0, seed=42)
    assert len(picked) == 20, len(picked)
    assert all(is_unreviewed(r) for r in picked), "抽到被標註的句子"
    # 分層：至少涵蓋 6 個以上不同框架
    assert len({r["frame"] for r in picked}) >= 6, "分層不足"
    # 決定性
    again = select(rows, 20, core_only=True, min_naturalness=4.0, seed=42)
    assert [r["pair_id"] + r["lang"] for r in picked] == [r["pair_id"] + r["lang"] for r in again]
    print(f"SELF-TEST PASS ✅  抽到 20 句，組成：{composition(picked)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-csv", type=Path)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--core-only", action="store_true",
                    help="只從台灣核心 baseline 抽（§12 之核心語料 10%%）")
    ap.add_argument("--min-naturalness", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("rq1_pilot.csv"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.pairs_csv:
        ap.error("需 --pairs-csv（或 --self-test）")

    rows = load_csv(args.pairs_csv)
    picked = select(rows, args.n, args.core_only, args.min_naturalness, args.seed)
    write_csv(args.out, picked)
    print(f"抽出 {len(picked)} 句 → {args.out}")
    print(f"  組成 (frame/lang)：{composition(picked)}")
    print(f"  （{len(picked)} 句 × 2 site × k → 校準批向量；跑 run_pilot.sh 取 τ）")


if __name__ == "__main__":
    main()
