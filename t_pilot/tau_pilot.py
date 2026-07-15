#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tau_pilot.py — τ 閾值校準批（pilot）一鍵完整腳本
==================================================

依 experiment_pipeline.md「01 · τ閾值校準批」與設計書 §12：從審閱後的
`rq1_review_all.csv`（360 句）抽 **20 句乾淨句**（無 reviewer_notes、
naturalness ≥ 4，跨 frame×lang 分層），小跑一遍 mini-pipeline：

    選句 → extract_activations（20×2 site＝40 向量）
         → SGLang AV server → verbalize（40×k＝200 則描述）
         → score_roundtrip（AR，MSE＝2(1−cos)）
         → calibrate_gate（τ＝MSE 最佳三分位）
         → preview.md（人眼檢視：NLA 翻譯 OK 不 OK）

跑量小（分鐘級），跑完看兩份東西：

  * `preview.md`  —— 每句原文 + k 則 AV 描述 + 各自 MSE/cos，目測 NLA 品質
  * `gate.summary.md` + 終端摘要 —— τ 數值與分層（lang/site）分布

之後全量跑 360 句時，可用本腳本輸出的 `pilot_desc_ids.txt` 把 pilot 上
校準到的 τ 套用到全體：

    python main_script/calibrate_gate.py --scores results/qwen/roundtrip.csv \
        --calibrate-on results/pilot_qwen/pilot_desc_ids.txt --out results/qwen/gate

用法（TWCC，GPU）：

    # Qwen（GPU 0）

    export PATH="/home/tyleryeh47/.conda/envs/rocling/bin:$PATH"
    PYTHONPATH=/home/tyleryeh47/ROCLING-2026/verify_script CUDA_VISIBLE_DEVICES=0 python tau_pilot.py --model qwen \
        --pairs-csv ../rq1_review_all.csv --ckpt-root checkpoint --nla-repo natural_language_autoencoders --force

    # Gemma（GPU 1、換 port）
    CUDA_VISIBLE_DEVICES=1 python t_pilot/tau_pilot.py --model gemma --port 30001 \
        --pairs-csv rq1_review_all.csv --ckpt-root $CKPT_ROOT --nla-repo $NLA_REPO

本機（無 GPU）先驗證選句與 site 索引：

    python t_pilot/tau_pilot.py --self-test
    python t_pilot/tau_pilot.py --model qwen --pairs-csv rq1_review_all.csv --select-only
    python t_pilot/tau_pilot.py --model qwen --pairs-csv rq1_review_all.csv --dry-run

中斷續跑：直接重跑同一指令即可 —— 已存在的中間產物會跳過
（verbalize 本身支援斷點續跑；要整批重來加 --force）。
"""

from __future__ import annotations

import argparse
import atexit
import csv
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MAIN = REPO / "main_script"
PY_SERVER = "/home/tyleryeh47/.conda/envs/rocling/bin/python"
PY_CLIENT = "/home/tyleryeh47/.conda/envs/nla_client/bin/python"

# 與 main_script/README.md、run_pipeline.sh 一致的模型設定
MODELS = {
    "qwen": dict(
        hf_model="Qwen/Qwen2.5-7B-Instruct",
        av="nla-qwen2.5-7b-L20-av",
        ar="nla-qwen2.5-7b-L20-ar",
        sglang_extra=[],
    ),
    "gemma": dict(
        hf_model="google/gemma-3-12b-it",
        av="nla-gemma3-12b-L32-av",
        ar="nla-gemma3-12b-L32-ar",
        sglang_extra=["--attention-backend", "fa3"],  # Gemma-3 需 fa3
    ),
}


# ──────────────────────────────────────────────────────────────────────
# 第 1 步：從 360 句抽 20 句校準批（跨 frame×lang 分層、只取乾淨句）
# ──────────────────────────────────────────────────────────────────────

def is_unreviewed(row: dict) -> bool:
    """reviewer_notes 空（或無此欄）視為審閱者未提問題、乾淨。"""
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
    """跨 (frame, lang) 分層 round-robin 抽 n 列，τ 不偏於單一框架；seed 決定性。"""
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


def select_pilot(pairs_csv: Path, n: int, core_only: bool,
                 min_naturalness: float, seed: int) -> list[dict]:
    with pairs_csv.open(encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    if not any("reviewer_notes" in r for r in rows):
        print("  [警告] CSV 無 reviewer_notes 欄，視全部為未標註。")

    elig = [r for r in rows if is_unreviewed(r) and naturalness_ok(r, min_naturalness)]
    if core_only:
        elig = [r for r in elig
                if r.get("cell_type", "baseline") == "baseline"
                and r.get("entity", "") in ("台灣", "Taiwan", "TW", "")]
    if len(elig) < n:
        print(f"  [警告] 合格句只有 {len(elig)} < {n}，將全數採用。")
    return stratified_sample(elig, min(n, len(elig)), seed)


def composition(rows: list[dict]) -> str:
    by = defaultdict(int)
    for r in rows:
        by[(r.get("frame", "?"), r.get("lang", "?"))] += 1
    return ", ".join(f"{f}/{l}:{c}" for (f, l), c in sorted(by.items()))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise SystemExit("[錯誤] 沒有符合條件的句子可抽（reviewer_notes 全被標註？）")
    fields = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ──────────────────────────────────────────────────────────────────────
# 第 2–5 步：呼叫 main_script 的四支腳本（子行程），並管理 SGLang server
# ──────────────────────────────────────────────────────────────────────

def run(cmd: list[str], **kw) -> None:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def launch_sglang(av_ckpt: Path, port: int, extra: list[str],
                  log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    proc = subprocess.Popen(
        [PY_SERVER, "-m", "sglang.launch_server",
         "--model-path", str(av_ckpt), "--port", str(port),
         "--disable-radix-cache", *extra],  # radix 以 token id 為鍵，embed 請求無此鍵
        stdout=log, stderr=subprocess.STDOUT)
    atexit.register(lambda: proc.poll() is None and proc.terminate())

    url = f"http://localhost:{port}/health"
    print(f"  等待 SGLang 就緒：{url}（log → {log_path}）")
    for i in range(120):
        if proc.poll() is not None:
            raise SystemExit(f"[錯誤] SGLang 啟動失敗，見 {log_path}")
        try:
            urllib.request.urlopen(url, timeout=3)
            print(f"  server 就緒（{(i + 1) * 5}s 內）")
            return proc
        except Exception:
            time.sleep(5)
    proc.terminate()
    raise SystemExit(f"[錯誤] SGLang 逾時（600s），見 {log_path}")


# ──────────────────────────────────────────────────────────────────────
# 第 6 步：τ 與人眼檢視報告
# ──────────────────────────────────────────────────────────────────────

def quantile(xs: list[float], q: float) -> float:
    """與 calibrate_gate.py 完全相同的線性內插分位數。"""
    if not xs:
        return float("nan")
    s = sorted(xs)
    i = q * (len(s) - 1)
    lo = int(i)
    frac = i - lo
    return s[lo] if lo + 1 >= len(s) else s[lo] * (1 - frac) + s[lo + 1] * frac


def load_scores(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for r in rows:
        r["mse"] = float(r["mse"])
        r["cos"] = float(r.get("cos", "nan") or "nan")
    return rows


def write_preview(descriptions: Path, scores: list[dict], tau: float,
                  out_md: Path) -> None:
    """每句：原文 + 兩個 site + k 則 AV 描述（依 MSE 排序、標 pass/fail）。"""
    import pyarrow.parquet as pq
    desc_of = {r["desc_id"]: r for r in pq.read_table(descriptions).to_pylist()}

    by_sent: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    meta_of: dict[str, dict] = {}
    for s in scores:
        d = desc_of.get(s["desc_id"], {})
        sid = s.get("sent_id") or d.get("sent_id", "?")
        site = s.get("site") or d.get("site", "?")
        by_sent[sid][site].append({**d, **s})
        meta_of.setdefault(sid, d)

    lines = ["# τ 校準批 · NLA 翻譯人眼檢視", "",
             f"- 描述總數：{len(scores)}；句數：{len(by_sent)}",
             f"- τ（MSE 最佳三分位）＝ **{tau:.4f}**；✅＝pass（mse ≤ τ）",
             f"- 忠實度：MSE＝2(1−cos)∈[0,4]，越低越忠實", ""]

    def sent_key(sid: str):
        m = meta_of.get(sid, {})
        return (m.get("frame", ""), m.get("pair_id", ""), m.get("lang", ""))

    for sid in sorted(by_sent, key=sent_key):
        m = meta_of[sid]
        lines += [f"## {sid}",
                  f"- frame=`{m.get('frame', '?')}` lang=`{m.get('lang', '?')}` "
                  f"cell=`{m.get('cell_type', '?')}` mention=`{m.get('mention', '?')}`",
                  f"- 原文：{m.get('text', '(activations parquet 未帶 text)')}", ""]
        for site in sorted(by_sent[sid]):
            tag = "提及詞末 subtoken" if site == "A" else "句末 token"
            lines.append(f"### Site {site}（{tag}）")
            lines += ["| gate | mse | cos | desc_lang | AV 描述 |",
                      "|---|---|---|---|---|"]
            for r in sorted(by_sent[sid][site], key=lambda x: x["mse"]):
                gate = "✅" if r["mse"] <= tau else "❌"
                d = (r.get("description") or "").replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {gate} | {r['mse']:.3f} | {r['cos']:.3f} "
                             f"| {r.get('desc_lang', '?')} | {d} |")
            lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  人眼檢視 → {out_md}")


def print_verdict(scores: list[dict], tau: float) -> None:
    """終端摘要：τ + 分層 pass rate，快速回答「NLA OK 不 OK」。"""
    print("\n" + "=" * 62)
    print(f" τ（MSE 最佳三分位）＝ {tau:.4f}")
    all_mse = [r["mse"] for r in scores]
    all_cos = [r["cos"] for r in scores]
    print(f" MSE  mean={mean(all_mse):.3f}  median={median(all_mse):.3f}  "
          f"Q1={quantile(all_mse, .25):.3f}  Q3={quantile(all_mse, .75):.3f}")
    print(f" cos  mean={mean(all_cos):.3f}")
    for key in ("lang", "site", "desc_lang"):
        if key not in scores[0]:
            continue
        groups: dict[str, list[float]] = defaultdict(list)
        for r in scores:
            groups[str(r.get(key, ""))].append(r["mse"])
        parts = [f"{g or '(空)'}: n={len(xs)} med={median(xs):.3f} "
                 f"pass={sum(x <= tau for x in xs) / len(xs):.0%}"
                 for g, xs in sorted(groups.items())]
        print(f" × {key:9s} " + " | ".join(parts))
    print("=" * 62)
    print(" 判讀提示：")
    print("  1. 看 preview.md —— 描述有沒有講到句子的實體／框架（NLA 翻譯 OK？）")
    print("  2. τ 附近（±0.05）的描述品質是否明顯分界（τ 合理？）")
    print("  3. zh/en 或 site A/B 的 pass rate 若系統性偏差，記入效度威脅，不可靜默丟棄")


# ──────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=sorted(MODELS), help="qwen 或 gemma")
    ap.add_argument("--pairs-csv", type=Path, help="審閱後 360 句 CSV（rq1_review_all.csv）")
    ap.add_argument("--ckpt-root", type=Path, help="NLA checkpoints 根目錄")
    ap.add_argument("--nla-repo", type=Path, help="natural_language_autoencoders repo 路徑")
    # 選句
    ap.add_argument("--n", type=int, default=20, help="校準批句數（預設 20 ≈ 核心 10%%）")
    ap.add_argument("--core-only", action="store_true",
                    help="只從台灣核心 baseline 抽（§12 之核心語料 10%%）")
    ap.add_argument("--min-naturalness", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=42)
    # 推論
    ap.add_argument("--k", type=int, default=5, help="每向量 AV 描述數（預設 5）")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--port", type=int, default=int(os.environ.get("SGLANG_PORT", 30000)))
    ap.add_argument("--tercile", type=float, default=1 / 3, help="τ 之 MSE 分位")
    ap.add_argument("--outdir", type=Path, default=None,
                    help="輸出目錄（預設 results/pilot_{model}）")
    # 模式
    ap.add_argument("--select-only", action="store_true", help="只抽 20 句，不跑模型")
    ap.add_argument("--dry-run", action="store_true",
                    help="抽 20 句 + site 索引驗證（CPU、不載模型權重）")
    ap.add_argument("--force", action="store_true", help="忽略既有中間產物，整批重跑")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not (args.model and args.pairs_csv):
        ap.error("需 --model 與 --pairs-csv（或 --self-test）")
    if not (args.select_only or args.dry_run) and not (args.ckpt_root and args.nla_repo):
        ap.error("完整跑需 --ckpt-root 與 --nla-repo（僅驗證可用 --select-only / --dry-run）")

    cfg = MODELS[args.model]
    outdir = args.outdir or REPO / "results" / f"pilot_{args.model}"
    outdir.mkdir(parents=True, exist_ok=True)
    if args.force:
        for p in ["verbalizations.parquet", "roundtrip.csv",
                  "gate.gated.csv", "gate.summary.md"]:
            (outdir / p).unlink(missing_ok=True)
        shutil.rmtree(outdir / "verbalizations.parquet.parts", ignore_errors=True)
        shutil.rmtree(outdir / "activations", ignore_errors=True)

    # ── 1/6 選句 ────────────────────────────────────────────────
    pilot_csv = outdir / f"rq1_pilot{args.n}.csv"
    print(f"--- 1/6 抽 τ 校準批（n={args.n}，seed={args.seed}）---")
    if pilot_csv.exists() and not args.force:
        print(f"  已存在 {pilot_csv}，沿用（--force 可重抽）")
    else:
        picked = select_pilot(args.pairs_csv, args.n, args.core_only,
                              args.min_naturalness, args.seed)
        write_csv(pilot_csv, picked)
        print(f"  抽出 {len(picked)} 句 → {pilot_csv}")
        print(f"  組成 (frame/lang)：{composition(picked)}")
    if args.select_only:
        return

    # ── 2/6 抽取 activations ────────────────────────────────────
    av_ckpt = (args.ckpt_root / cfg["av"]) if args.ckpt_root else None
    ar_ckpt = (args.ckpt_root / cfg["ar"]) if args.ckpt_root else None
    act_dir = outdir / "activations"
    print(f"--- 2/6 抽取 activations（{args.n}×2 site）---")
    if args.dry_run:
        run([PY_CLIENT, MAIN / "extract_activations.py",
             "--pairs-csv", pilot_csv, "--model", cfg["hf_model"],
             "--keep-all", "--dry-run"])
        print("dry-run 完成（site 索引 OK 即可上 GPU 跑完整 pilot）。")
        return
    acts = sorted(act_dir.glob("activations_*.parquet"))
    if acts and not args.force:
        print(f"  已存在 {acts[0]}，沿用")
    else:
        # --keep-all：選句階段已套過 naturalness 門檻，抽取端不再重複過濾
        run([PY_CLIENT, MAIN / "extract_activations.py",
             "--pairs-csv", pilot_csv, "--model", cfg["hf_model"],
             "--nla-meta", av_ckpt / "nla_meta.yaml",
             "--outdir", act_dir, "--keep-all"])
        acts = sorted(act_dir.glob("activations_*.parquet"))
    act = acts[0]

    # ── 3/6 SGLang AV server + verbalize ───────────────────────
    verb = outdir / "verbalizations.parquet"
    if verb.exists() and not args.force:
        print(f"--- 3/6 已存在 {verb}，跳過 AV ---")
    else:
        print(f"--- 3/6 launch SGLang + AV verbalize（k={args.k}）---")
        server = launch_sglang(av_ckpt, args.port, cfg["sglang_extra"],
                               outdir / "sglang.log")
        try:
            run([PY_CLIENT, MAIN / "verbalize.py",
                 "--activations", act, "--av-checkpoint", av_ckpt,
                 "--nla-repo", args.nla_repo,
                 "--sglang-url", f"http://localhost:{args.port}",
                 "--k", args.k, "--temperature", args.temperature,
                 "--out", verb])
        finally:
            server.terminate()  # 先釋放 VRAM，AR critic 才載得進同一張卡
            server.wait(timeout=60)

    # ── 4/6 AR round-trip ───────────────────────────────────────
    rt = outdir / "roundtrip.csv"
    if rt.exists() and not args.force:
        print(f"--- 4/6 已存在 {rt}，跳過 AR ---")
    else:
        print("--- 4/6 AR round-trip 忠實度 ---")
        run([PY_CLIENT, MAIN / "score_roundtrip.py",
             "--descriptions", verb, "--activations", act,
             "--ar-checkpoint", ar_ckpt, "--nla-repo", args.nla_repo,
             "--out", rt])

    # ── 5/6 τ 校準 ──────────────────────────────────────────────
    print("--- 5/6 τ 校準（calibrate_gate）---")
    run([PY_CLIENT, MAIN / "calibrate_gate.py",
         "--scores", rt, "--tercile", args.tercile, "--out", outdir / "gate"])

    scores = load_scores(rt)
    tau = quantile([r["mse"] for r in scores], args.tercile)
    ids_txt = outdir / "pilot_desc_ids.txt"
    ids_txt.write_text("\n".join(r["desc_id"] for r in scores) + "\n", encoding="utf-8")

    # ── 6/6 人眼檢視 ────────────────────────────────────────────
    print("--- 6/6 人眼檢視報告 ---")
    write_preview(verb, scores, tau, outdir / "preview.md")
    print_verdict(scores, tau)
    print(f"\n 產物目錄：{outdir}")
    print(f"   preview.md（先看這份）· gate.summary.md · roundtrip.csv")
    print(f"   pilot_desc_ids.txt —— 全量跑後供 calibrate_gate.py --calibrate-on 套 τ")


# ──────────────────────────────────────────────────────────────────────
# 自測（合成 360 列，不需檔案／GPU）
# ──────────────────────────────────────────────────────────────────────

def self_test() -> None:
    rows = []
    frames = ["GEO", "POL-INT", "POL-DOM", "ECON", "CUL", "HIST", "LIFE", "TRAV"]
    for fi, frame in enumerate(frames):
        for t in range(12):
            for lang in ("zh", "en"):
                note = "太翻譯腔" if (fi * 12 + t) % 7 == 0 else ""
                rows.append(dict(pair_id=f"{frame}-{t:02d}", frame=frame, entity="台灣",
                                 lang=lang, cell_type="baseline",
                                 mention_script="hanzi" if lang == "zh" else "latin",
                                 mention="台灣" if lang == "zh" else "Taiwan",
                                 text=f"{frame}-{t}-{lang}", naturalness="5",
                                 reviewer_notes=note))
    elig = [r for r in rows if is_unreviewed(r) and naturalness_ok(r, 4.0)]
    picked = stratified_sample(elig, 20, seed=42)
    assert len(picked) == 20, len(picked)
    assert all(is_unreviewed(r) for r in picked), "抽到被標註的句子"
    assert len({r["frame"] for r in picked}) >= 6, "分層不足"
    again = stratified_sample(elig, 20, seed=42)
    assert [r["pair_id"] + r["lang"] for r in picked] == \
           [r["pair_id"] + r["lang"] for r in again], "seed 不決定性"
    # τ 分位數與 calibrate_gate.py 對齊
    assert abs(quantile([1, 2, 3, 4], 1 / 3) - 2.0) < 1e-9
    assert abs(quantile(list(map(float, range(1, 10))), 1 / 3) - 3.6666666) < 1e-5
    print(f"SELF-TEST PASS ✅  抽到 20 句，組成：{composition(picked)}")


if __name__ == "__main__":
    main()
