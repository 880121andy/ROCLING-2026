#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_annotation_sheet.py — 由 pipeline 產物組出框架標註表（對齊 framing_codebook_v1）
=====================================================================================

把 Phase 02 的兩個產物 join 起來，輸出 framing_codebook_v1.md §4 規定欄位的標註表：

    verbalize.py      → verbalizations.parquet   （含 AV 描述文字 `description`）
    calibrate_gate.py → gate.gated.csv           （含 mse, cos, pass_gate, desc_lang）

欄位對應（pipeline 名 → codebook 名）：
    description  → av_text
    pair_id      → item_id
    frame        → frame_source     ← 設計框架；標 D1–D4 時「不可看」！
    text         → 原句             ← 原刺激句；codebook 僅 D5 漂移判斷才看
    pass_gate    → gate

產物（兩個 CSV，皆 utf-8-sig，欄位順序＝ codebook §4）：
    {out}_sheet.csv    全池（所有 k 個 sample）→ 通過 κ 後交 LLM 批量標
    {out}_pilot.csv    κ pilot 批（每「句×site」取一個 sample）→ 兩人雙盲標、算 κ

用法：
    python main_script/build_annotation_sheet.py \
        --verbalizations results/gemma/verbalizations.parquet \
        --gated results/gemma/gate.gated.csv \
        --out annotations/gemma

    # κ pilot 批再依 frame×lang×site 分層抽樣到約 80 則（tractable 給兩位標註者）
    python main_script/build_annotation_sheet.py \
        --verbalizations results/gemma/verbalizations.parquet \
        --gated results/gemma/gate.gated.csv \
        --out annotations/gemma --pilot-n 80

接續：兩位標註者各自複製 {out}_pilot.csv，只填 d1_frame…d5_drift（D5 才看原句），
存成 A/B 兩檔後跑 compute_kappa.py 算 κ。
"""

from __future__ import annotations

import argparse
from pathlib import Path

# codebook §4 標註表交付欄位（順序固定；compute_kappa.py 靠 desc_id + d1..d5 讀）
SHEET_COLS = [
    "desc_id", "model", "item_id", "frame_source", "entity", "lang", "site",
    "cell_type", "mse", "cos", "gate", "desc_lang", "原句", "av_text",
    "annotator", "d1_frame", "d2_anchor", "d3_construal", "d4_ortho",
    "d5_drift", "genre_note", "notes",
]
# 交給標註者留空、由人/LLM 填的欄
BLANK_COLS = ["annotator", "d1_frame", "d2_anchor", "d3_construal",
              "d4_ortho", "d5_drift", "genre_note", "notes"]


def detect_desc_lang(text: str) -> str:
    """與 score_roundtrip.detect_desc_lang 同一套：依 CJK 比例分 zh/en/mixed。"""
    if not text:
        return "empty"
    cjk = sum(1 for ch in text if "㐀" <= ch <= "鿿")
    latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
    tot = cjk + latin
    if tot == 0:
        return "other"
    r = cjk / tot
    return "zh" if r > 0.6 else "en" if r < 0.15 else "mixed"


def load_verbalizations(path: Path):
    import pyarrow.parquet as pq
    df = pq.read_table(path).to_pandas()
    if "description" not in df.columns:
        raise SystemExit(f"[錯誤] {path} 無 `description` 欄。實際欄位：{list(df.columns)}")
    if "desc_id" not in df.columns:
        raise SystemExit(f"[錯誤] {path} 無 `desc_id` 欄（join 鍵）。")
    return df


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verbalizations", type=Path, required=True,
                    help="verbalize.py 產出的 parquet（含 description）")
    ap.add_argument("--gated", type=Path, default=None,
                    help="calibrate_gate.py 的 gate.gated.csv；缺則 mse/cos/gate 留空、"
                         "desc_lang 由 av_text 推定")
    ap.add_argument("--out", type=Path, required=True, help="輸出前綴（→ {out}_sheet.csv, {out}_pilot.csv）")
    ap.add_argument("--pilot-sample-k", type=int, default=1,
                    help="κ pilot 批每「句×site」取哪個 sample_k。"
                         "codebook §1 寫 1；注意 verbalize.py 為 0-indexed（第一個樣本＝0）")
    ap.add_argument("--pilot-n", type=int, default=None,
                    help="κ pilot 批再分層抽樣到約 N 則（依 frame×lang×site）；缺省＝全取")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    v = load_verbalizations(args.verbalizations)

    # --- 建立標註表基底（pipeline 欄名 → codebook 欄名）---
    out = pd.DataFrame()
    out["desc_id"] = v["desc_id"]
    out["model"] = v["model"] if "model" in v.columns else ""
    if "pair_id" in v.columns:
        out["item_id"] = v["pair_id"]
    elif "sent_id" in v.columns:
        out["item_id"] = v["sent_id"]
    else:
        out["item_id"] = ""
    out["frame_source"] = v["frame"] if "frame" in v.columns else ""
    for c in ("entity", "lang", "site", "cell_type"):
        out[c] = v[c] if c in v.columns else ""
    out["原句"] = v["text"] if "text" in v.columns else ""
    out["av_text"] = v["description"]
    out["_sample_k"] = v["sample_k"] if "sample_k" in v.columns else 0

    # --- join gate（mse / cos / pass_gate→gate / desc_lang）---
    if args.gated and args.gated.exists():
        g = pd.read_csv(args.gated, encoding="utf-8-sig")
        keep = [c for c in ("desc_id", "mse", "cos", "pass_gate", "desc_lang") if c in g.columns]
        out = out.merge(g[keep], on="desc_id", how="left")
        out = out.rename(columns={"pass_gate": "gate"})
    else:
        print("  [提示] 未提供 --gated：mse/cos/gate 留空，desc_lang 由 av_text 推定。")

    for c in ("mse", "cos", "gate", "desc_lang"):
        if c not in out.columns:
            out[c] = ""

    # desc_lang 補洞（gate 缺、或該列未被 score_roundtrip 打分）
    dl = out["desc_lang"].astype("string")
    need = dl.isna() | (dl.str.len() == 0)
    out.loc[need, "desc_lang"] = out.loc[need, "av_text"].fillna("").map(detect_desc_lang)

    # 標註欄留空
    for c in BLANK_COLS:
        out[c] = ""

    # --- 全池標註表 ---
    full = out[SHEET_COLS].copy()
    sheet_path = Path(f"{args.out}_sheet.csv")
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    full.to_csv(sheet_path, index=False, encoding="utf-8-sig")

    # --- κ pilot 批：每「句×site」（＝一個 vector）取一個 sample ---
    sk = out["_sample_k"].astype("Int64")
    avail = sorted(int(x) for x in sk.dropna().unique())
    chosen = args.pilot_sample_k
    if chosen not in avail:
        chosen = avail[0] if avail else 0
        print(f"  [提示] sample_k={args.pilot_sample_k} 不存在（實際 {avail}），改用 {chosen}。")
    pilot = out[sk == chosen].copy()

    # 可選：分層抽樣到約 pilot-n（依 frame×lang×site，讓兩人雙盲量 tractable）
    if args.pilot_n and len(pilot) > args.pilot_n:
        frac = args.pilot_n / len(pilot)
        parts = []
        for _, d in pilot.groupby(["frame_source", "lang", "site"], dropna=False):
            k = min(len(d), max(1, round(len(d) * frac)))
            parts.append(d.sample(k, random_state=args.seed))
        pilot = pd.concat(parts, ignore_index=True)

    pilot_path = Path(f"{args.out}_pilot.csv")
    pilot[SHEET_COLS].to_csv(pilot_path, index=False, encoding="utf-8-sig")

    # --- 摘要 ---
    print(f"全池標註表 -> {sheet_path}（{len(full)} 則）")
    gate_str = full["gate"].astype("string").fillna("")
    if (gate_str.str.len() > 0).any():
        n_pass = gate_str.isin(["1", "1.0", "True", "true"]).sum()
        print(f"  gate 通過：{n_pass}/{len(full)}")
    print(f"κ pilot 批 -> {pilot_path}（{len(pilot)} 則；每 vector 取 sample_k={chosen}）")
    print("  → 兩人各複製一份，只填 d1_frame…d5_drift，再跑：")
    print(f"     python main_script/compute_kappa.py {args.out}_pilot_A.csv {args.out}_pilot_B.csv")
    if chosen == 1:
        print("  ⚠ codebook §1 寫 sample_k=1，但 verbalize.py 0-indexed（第一個樣本＝0）；"
              "要改用第一個樣本加 --pilot-sample-k 0。")


if __name__ == "__main__":
    main()