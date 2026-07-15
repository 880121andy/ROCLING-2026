#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_activations.py — RQ1 Phase 02 步驟 1：residual stream activation 抽取
============================================================================

輸入：人工審閱後的 360 句語料 CSV（rq1_review_all.csv）。
輸出：每句 × 2 抽取位置 ＝ 720 個向量，寫成 NLA 推論所需的 parquet
      （必含 `activation_vector` 欄，float32，shape [N, d_model]），
      並附完整 metadata 供後續 join。

與設計書 §8 抽取規範對齊：
  * 純文字、**不套 chat template**（NLA 以 fineweb 預訓練式文本之 activation 訓練）。
  * 層位依 checkpoint 之 `nla_meta.yaml`（Qwen L20／Gemma-12B L32），不硬編碼；
    以 --nla-meta 讀取，缺省退回 --layer 或 per-model 預設。
  * Site A：目標詞**最後一個 subtoken**（自迴歸資訊匯聚處）。
  * Site B：句末（最後一個非 special）token。
  * tokenizer 呼叫方式（add_special_tokens=True、offset_mapping）與
    verify_tokenization.py **完全一致** —— 直接 import 其定位邏輯，
    確保 site 索引與先前 tokenizer 報告可對照。

用法（TWCC，單張 GPU 即可；360 句短序列，數十秒等級）：

    python pipeline/extract_activations.py \
        --pairs-csv rq1_review_all.csv \
        --model Qwen/Qwen2.5-7B-Instruct \
        --nla-meta /path/to/nla-qwen2.5-7b-L20-av/nla_meta.yaml \
        --outdir activations/qwen

    # Gemma（gated，需先 huggingface-cli login 並於網頁同意授權）
    python pipeline/extract_activations.py \
        --pairs-csv rq1_review_all.csv \
        --model google/gemma-3-12b-it \
        --nla-meta /path/to/nla-gemma3-12b-L32-av/nla_meta.yaml \
        --outdir activations/gemma

不載入模型、只驗證切分與 site 索引（Wednesday meeting 用；只需 CPU＋tokenizer）：

    python pipeline/extract_activations.py --pairs-csv rq1_review_all.csv \
        --model Qwen/Qwen2.5-7B-Instruct --dry-run

無網路邏輯自測：

    python pipeline/extract_activations.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

# 復用經稽核的 offset 定位邏輯（與 tokenizer 報告同一套）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from verify_tokenization import find_mention_char_spans, locate_token_span  # noqa: E402

# 每句需保留、供後續 join 的 metadata 欄位（不強制全部存在）
META_COLS = ["sent_id", "pair_id", "frame", "entity", "lang",
             "mention_script", "cell_type", "mention", "text"]

# 缺 --nla-meta 且無 --layer 時的退回層位（設計書 §2）
DEFAULT_LAYER = {
    "Qwen/Qwen2.5-7B-Instruct": 20,
    "google/gemma-3-12b-it": 32,
    "unsloth/gemma-3-12b-it": 32,
}


# ======================================================================
# 純邏輯層（與 torch 無關，可 mock 測試）
# ======================================================================

@dataclass
class SitePlan:
    """單句經 tokenizer 後、兩個抽取位置的索引規劃（尚未真正抽向量）。"""
    sent_id: str
    lang: str
    mention: str
    n_tokens: int
    site_a_idx: int          # 目標詞末 subtoken（含 special 之絕對索引）
    site_b_idx: int          # 句末非 special token
    mention_n_subtokens: int
    warnings: str = ""


def plan_sites(tok, text: str, mention: str, sent_id: str, lang: str) -> SitePlan:
    """
    以 offset_mapping 定位 mention span，回傳 Site A/B 的 token 索引。
    與 verify_tokenization.analyze_sentence 同源，但只保留抽取所需欄位。
    """
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids = list(enc["input_ids"])
    offsets = [tuple(o) for o in enc["offset_mapping"]]
    special = set(getattr(tok, "all_special_ids", []) or [])

    warns: list[str] = []
    spans = find_mention_char_spans(text, mention)
    if len(spans) != 1:
        warns.append(f"mention 出現 {len(spans)} 次（規範要求恰為 1）")
    char_span = spans[0] if spans else (0, 0)

    t0, t1 = locate_token_span(offsets, char_span)
    if t0 < 0:
        warns.append("offset 對齊失敗：找不到覆蓋 mention 的 token span")

    non_special = [i for i, x in enumerate(ids) if x not in special]
    site_b = non_special[-1] if non_special else -1
    return SitePlan(
        sent_id=sent_id, lang=lang, mention=mention,
        n_tokens=len(non_special),
        site_a_idx=(t1 - 1) if t0 >= 0 else -1,
        site_b_idx=site_b,
        mention_n_subtokens=max(t1 - t0, 0),
        warnings="; ".join(warns),
    )


def assign_sent_id(row: dict, idx: int) -> str:
    """
    產生唯一句 id。優先用 CSV 既有 sent_id；否則以能區分 360 列的鍵組合，
    仍不足則退回列序號（保證唯一）。
    """
    if row.get("sent_id"):
        return row["sent_id"].strip()
    parts = [row.get(k, "").strip() for k in ("pair_id", "lang", "cell_type", "mention_script")]
    key = "|".join(p for p in parts if p)
    return f"{key}#{idx:03d}" if key else f"row{idx:03d}"


# ======================================================================
# I/O
# ======================================================================

def load_pairs_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for k in ("text", "mention"):
        if rows and k not in rows[0]:
            raise SystemExit(f"[錯誤] CSV 缺必要欄位 {k!r}（實際欄位：{list(rows[0])}）")
    return rows


def row_passes_review(row: dict, min_naturalness: float) -> bool:
    """naturalness 門檻（<4 視為需改寫）。缺欄或空值則不擋（視為尚未評分）。"""
    v = (row.get("naturalness") or "").strip()
    if not v:
        return True
    try:
        return float(v) >= min_naturalness
    except ValueError:
        return True


def resolve_layer(model: str, nla_meta: Path | None, layer_arg: int | None) -> int:
    if layer_arg is not None:
        return layer_arg
    if nla_meta and nla_meta.exists():
        import yaml
        meta = yaml.safe_load(nla_meta.read_text(encoding="utf-8")) or {}
        for path in (("extraction", "layer"), ("layer",), ("model", "layer")):
            node = meta
            for k in path:
                node = node.get(k) if isinstance(node, dict) else None
            if isinstance(node, int):
                print(f"  層位取自 nla_meta.yaml：L{node}")
                return node
        print(f"  [警告] nla_meta.yaml 未見層位鍵，退回預設。")
    if model in DEFAULT_LAYER:
        return DEFAULT_LAYER[model]
    raise SystemExit(f"[錯誤] 無法決定層位：請給 --layer 或含層位的 --nla-meta（model={model}）")


def write_plan_csv(path: Path, plans: list[SitePlan]) -> None:
    if not plans:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(plans[0]).keys()))
        w.writeheader()
        w.writerows(asdict(p) for p in plans)


# ======================================================================
# 主抽取流程（需 torch＋transformers＋GPU）
# ======================================================================

def run_extract(rows: list[dict], model: str, layer: int, outdir: Path,
                dry_run: bool, batch_note: str = "") -> None:
    import numpy as np
    from transformers import AutoTokenizer

    tag = model.split("/")[-1]
    tok = AutoTokenizer.from_pretrained(model)
    if not getattr(tok, "is_fast", False):
        raise SystemExit("[錯誤] 需 fast tokenizer 才有 offset_mapping。")

    # (1) 先做 site 規劃（無論是否 dry-run 都算，供人工檢視）
    plans, keep = [], []
    for idx, row in enumerate(rows):
        sid = assign_sent_id(row, idx)
        p = plan_sites(tok, row["text"], row["mention"], sid, row.get("lang", ""))
        plans.append(p)
        keep.append(row)
        if p.warnings:
            print(f"  [警告] {sid}: {p.warnings}")
    outdir.mkdir(parents=True, exist_ok=True)
    write_plan_csv(outdir / f"site_plan_{tag}.csv", plans)
    seen = {p.sent_id for p in plans}
    if len(seen) != len(plans):
        print(f"  [警告] sent_id 不唯一（{len(plans)} 列 → {len(seen)} 唯一），"
              f"請在 CSV 補 sent_id 欄。")
    print(f"  site 規劃 -> {outdir}/site_plan_{tag}.csv（{len(plans)} 句）")

    if dry_run:
        print("  [dry-run] 未載入模型、未抽向量。")
        return

    # (2) 載入模型並抽向量
    import torch
    from transformers import AutoModelForCausalLM

    print(f"  載入 {model}（bf16, device_map=auto）…")
    net = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.bfloat16, device_map="auto",
        output_hidden_states=True,
    )
    net.eval()
    dev = next(net.parameters()).device

    records: list[dict] = []
    n_bad = 0
    with torch.no_grad():
        for row, p in zip(keep, plans):
            enc = tok(row["text"], return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            # hidden_states 為長度 n_layer+1 之 tuple；index L = 第 L 個 block 後之殘差流
            hs = net(**enc).hidden_states[layer][0]      # [seq, d_model]
            hs = hs.float().cpu().numpy()
            for site, idx in (("A", p.site_a_idx), ("B", p.site_b_idx)):
                if idx < 0 or idx >= hs.shape[0]:
                    n_bad += 1
                    continue
                vec = hs[idx].astype(np.float32)
                rec = {k: (row.get(k, "") if k != "sent_id" else p.sent_id) for k in META_COLS}
                rec.update(
                    vector_id=f"{p.sent_id}#site{site}",
                    site=site, site_idx=int(idx), layer=int(layer),
                    model=model, n_tokens=int(p.n_tokens),
                    mention_n_subtokens=int(p.mention_n_subtokens),
                    vec_norm=float(np.linalg.norm(vec)),
                    activation_vector=vec,
                )
                records.append(rec)

    _cfg = net.config
    _hidden = (getattr(_cfg, 'hidden_size', None)
               or getattr(getattr(_cfg, 'text_config', None), 'hidden_size', None))
    if _hidden is None:
        raise AttributeError(f'no hidden_size on {type(_cfg).__name__} (.hidden_size / .text_config.hidden_size)')
    d_model = int(_hidden)
    write_parquet(outdir / f"activations_{tag}.parquet", records, d_model)
    print(f"  抽取完成：{len(records)} 向量（跳過 {n_bad}）"
          f" -> {outdir}/activations_{tag}.parquet  d_model={d_model}")


def write_parquet(path: Path, records: list[dict], d_model: int) -> None:
    """activation_vector 存成 fixed-size list<float32>（NLA 推論端 flatten 讀取）。"""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not records:
        raise SystemExit("[錯誤] 無任何向量可寫出。")
    meta_cols = [k for k in records[0] if k != "activation_vector"]
    arrays = {c: pa.array([r[c] for r in records]) for c in meta_cols}
    mat = np.stack([r["activation_vector"] for r in records]).astype(np.float32)
    assert mat.shape[1] == d_model, (mat.shape, d_model)
    flat = pa.array(mat.reshape(-1), type=pa.float32())
    arrays["activation_vector"] = pa.FixedSizeListArray.from_arrays(flat, d_model)
    pq.write_table(pa.table(arrays), path)


# ======================================================================
# Self-test（mock tokenizer，無網路／無 torch）
# ======================================================================

def self_test() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from verify_tokenization import MockTokenizer

    tok = MockTokenizer()
    p = plan_sites(tok, "From the standpoint of tectonics, Taiwan sits on a boundary.",
                   "Taiwan", "T1", "en")
    assert p.mention_n_subtokens == 2, p
    assert p.site_a_idx >= 0 and p.site_b_idx > p.site_a_idx, p
    assert not p.warnings, p.warnings

    p2 = plan_sites(tok, "台灣很好，台灣真的很好。", "台灣", "T2", "zh")
    assert "恰為 1" in p2.warnings, p2

    # sent_id 分派：Design B 四格靠 (lang, mention_script) 區分
    base = dict(pair_id="GEO-01", frame="GEO", cell_type="codeswitch")
    a = assign_sent_id({**base, "lang": "zh", "mention_script": "latin"}, 0)
    b = assign_sent_id({**base, "lang": "en", "mention_script": "hanzi"}, 1)
    assert a != b and "GEO-01" in a

    assert row_passes_review({"naturalness": "3"}, 4.0) is False
    assert row_passes_review({"naturalness": "5"}, 4.0) is True
    assert row_passes_review({"naturalness": ""}, 4.0) is True   # 尚未評分不擋
    print("SELF-TEST PASS ✅（site 規劃、sent_id 分派、審閱門檻邏輯皆正確）")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs-csv", type=Path, help="審閱後 360 句語料 CSV")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--nla-meta", type=Path, default=None,
                    help="對應 AV/AR checkpoint 之 nla_meta.yaml（讀層位）")
    ap.add_argument("--layer", type=int, default=None, help="明確指定抽取層（覆蓋 meta）")
    ap.add_argument("--outdir", type=Path, default=Path("activations"))
    ap.add_argument("--min-naturalness", type=float, default=4.0,
                    help="低於此自然度分數之句子排除（缺分數不擋）")
    ap.add_argument("--keep-all", action="store_true", help="不套用自然度門檻，抽全部")
    ap.add_argument("--dry-run", action="store_true", help="只算 site 索引、不載模型")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.pairs_csv:
        ap.error("需 --pairs-csv（或用 --self-test）")

    rows = load_pairs_csv(args.pairs_csv)
    if not args.keep_all:
        before = len(rows)
        rows = [r for r in rows if row_passes_review(r, args.min_naturalness)]
        if len(rows) != before:
            print(f"  自然度門檻：{before} → {len(rows)} 句（排除 {before - len(rows)}）")

    layer = resolve_layer(args.model, args.nla_meta, args.layer)
    print(f"===== 抽取 {args.model} @ L{layer} =====")
    run_extract(rows, args.model, layer, args.outdir, args.dry_run)


if __name__ == "__main__":
    main()
