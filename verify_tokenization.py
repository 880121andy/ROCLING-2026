#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_tokenization.py — RQ1 語料之 tokenizer 驗證腳本
=======================================================

在正式生成 360 句語料與抽取 activations 之前，驗證三件事：

  1. 「台灣」／「臺灣」／"Taiwan" 在目標模型 tokenizer 下的 subtoken 切分
     （孤立 vs 語境內；tokenization 是語境相依的，"Taiwan" 與 " Taiwan"
      常是不同 token）。
  2. Site A（目標詞末 subtoken）與 Site B（句末 token）的索引定位是否
     穩健 —— 採 character-offset 對齊（offset_mapping），並與樸素的
     sublist 搜尋法對照，展示後者的失敗模式。
  3. 中英句對的 token 數是否符合 ±20% 長度匹配規範；前導子句是否
     ≥ --min-leadin 個 tokens。

用法（TWCC 上，僅需 CPU；tokenizer 下載量小）：

    pip install -U "transformers>=4.50" tokenizers huggingface_hub pandas
    export HF_HOME=/work/$USER/hf_cache          # 快取到 scratch
    huggingface-cli login                        # Gemma 為 gated model，
                                                 # 需先於 HF 網頁同意授權
    python verify_tokenization.py \
        --models Qwen/Qwen2.5-7B-Instruct google/gemma-3-12b-it \
        --outdir tokenizer_report

無網路環境下可先跑邏輯自測（使用 mock tokenizer）：

    python verify_tokenization.py --self-test

輸出：
    {outdir}/report_{model_tag}.csv      每句之逐句診斷
    {outdir}/pairs_{model_tag}.csv       句對長度匹配診斷
    {outdir}/summary.md                  跨模型摘要（可貼回設計書）

重要約定：後續 activation 抽取必須使用「與本腳本完全相同」的
tokenizer 呼叫方式（add_special_tokens=True、同一 revision），
hidden_states 的位置索引才會與此處回報的 site 索引一致。
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path

# ----------------------------------------------------------------------
# 內嵌示範句對（設計書 §4.3 之 16 組；量產語料就緒後改以 --pairs-csv 載入）
# 欄位:  pair_id, frame, lang, mention, text
# ----------------------------------------------------------------------
EXAMPLE_PAIRS: list[dict] = [
    # --- GEO ---
    dict(pair_id="GEO-01", frame="GEO", lang="zh", mention="台灣",
         text="從板塊構造的角度來看，台灣正好位於歐亞板塊與菲律賓海板塊的交界，因此地震十分頻繁。"),
    dict(pair_id="GEO-01", frame="GEO", lang="en", mention="Taiwan",
         text="From the standpoint of plate tectonics, Taiwan sits right on the boundary between the Eurasian and Philippine Sea plates, which is why earthquakes are so frequent."),
    dict(pair_id="GEO-02", frame="GEO", lang="zh", mention="台灣",
         text="受到季風與地形交互作用的影響，台灣的東北部在冬季經常陰雨綿綿。"),
    dict(pair_id="GEO-02", frame="GEO", lang="en", mention="Taiwan",
         text="Owing to the interaction of monsoon winds and local terrain, Taiwan tends to see long drizzly winters in its northeast."),
    # --- POL-INT ---
    dict(pair_id="POL-INT-01", frame="POL-INT", lang="zh", mention="台灣",
         text="在許多國際組織的正式場合裡，台灣的會員資格始終是各方交涉的焦點。"),
    dict(pair_id="POL-INT-01", frame="POL-INT", lang="en", mention="Taiwan",
         text="In the formal settings of many international organizations, Taiwan's membership status has long been a focal point of negotiation."),
    dict(pair_id="POL-INT-02", frame="POL-INT", lang="zh", mention="台灣",
         text="在近年來的外交往來之中，台灣與若干邦交國之間的關係變化備受關注。"),
    dict(pair_id="POL-INT-02", frame="POL-INT", lang="en", mention="Taiwan",
         text="In diplomatic exchanges of recent years, Taiwan's shifting ties with several of its formal allies have drawn close attention."),
    # --- POL-DOM ---
    dict(pair_id="POL-DOM-01", frame="POL-DOM", lang="zh", mention="台灣",
         text="每逢選舉年的冬天一到，台灣的街頭就掛滿競選旗幟，造勢晚會一場接著一場。"),
    dict(pair_id="POL-DOM-01", frame="POL-DOM", lang="en", mention="Taiwan",
         text="When the winter of an election year arrives, Taiwan's streets fill with campaign flags, and rallies follow one after another."),
    dict(pair_id="POL-DOM-02", frame="POL-DOM", lang="zh", mention="台灣",
         text="歷經數十年的政治轉型之後，台灣如今以高投票率與激烈的政黨競爭聞名。"),
    dict(pair_id="POL-DOM-02", frame="POL-DOM", lang="en", mention="Taiwan",
         text="After decades of political transformation, Taiwan is now known for high voter turnout and fierce competition between parties."),
    # --- ECON ---
    dict(pair_id="ECON-01", frame="ECON", lang="zh", mention="台灣",
         text="在全球半導體供應鏈之中，台灣生產了絕大多數的先進製程晶片。"),
    dict(pair_id="ECON-01", frame="ECON", lang="en", mention="Taiwan",
         text="Within the global semiconductor supply chain, Taiwan produces the vast majority of advanced-node chips."),
    dict(pair_id="ECON-02", frame="ECON", lang="zh", mention="台灣",
         text="在許多跨國企業的採購清單上，台灣的精密機械與自行車零件享有極高的評價。"),
    dict(pair_id="ECON-02", frame="ECON", lang="en", mention="Taiwan",
         text="On the procurement lists of many multinational firms, Taiwan's precision machinery and bicycle components enjoy an excellent reputation."),
    # --- CUL ---
    dict(pair_id="CUL-01", frame="CUL", lang="zh", mention="台灣",
         text="對許多喜歡深夜覓食的人來說，台灣的夜市小吃是難以抗拒的誘惑，蚵仔煎更是必點。"),
    dict(pair_id="CUL-01", frame="CUL", lang="en", mention="Taiwan",
         text="For anyone fond of late-night food hunts, Taiwan's night-market snacks are hard to resist, and the oyster omelet is a must-order."),
    dict(pair_id="CUL-02", frame="CUL", lang="zh", mention="台灣",
         text="每年春天媽祖遶境的季節一到，台灣就會湧現徒步進香九天八夜的人潮。"),
    dict(pair_id="CUL-02", frame="CUL", lang="en", mention="Taiwan",
         text="Each spring when the Mazu pilgrimage season arrives, Taiwan sees crowds of devotees walking the route for nine days and eight nights."),
    # --- HIST ---
    dict(pair_id="HIST-01", frame="HIST", lang="zh", mention="台灣",
         text="在二十世紀初的殖民統治時期，台灣興建了縱貫南北的鐵路系統。"),
    dict(pair_id="HIST-01", frame="HIST", lang="en", mention="Taiwan",
         text="During the colonial period of the early twentieth century, Taiwan built a railway system running the length of the island."),
    dict(pair_id="HIST-02", frame="HIST", lang="zh", mention="台灣",
         text="早在十七世紀的大航海時代，台灣就已經是東亞貿易網絡的重要節點。"),
    dict(pair_id="HIST-02", frame="HIST", lang="en", mention="Taiwan",
         text="As early as the seventeenth-century age of sail, Taiwan was already a key node in East Asian trade networks."),
    # --- LIFE ---
    dict(pair_id="LIFE-01", frame="LIFE", lang="zh", mention="台灣",
         text="就日常生活的便利程度而言，台灣的超商密度名列世界前茅，半夜也能繳費、領包裹。"),
    dict(pair_id="LIFE-01", frame="LIFE", lang="en", mention="Taiwan",
         text="In terms of everyday convenience, Taiwan ranks near the top worldwide in convenience-store density; you can pay bills and pick up parcels in the middle of the night."),
    dict(pair_id="LIFE-02", frame="LIFE", lang="zh", mention="台灣",
         text="對外送平台的重度使用者來說，台灣的都會區幾乎能在三十分鐘內送達任何餐點。"),
    dict(pair_id="LIFE-02", frame="LIFE", lang="en", mention="Taiwan",
         text="For heavy users of food-delivery apps, Taiwan's urban areas can get almost any meal to the door within thirty minutes."),
    # --- TRAV ---
    dict(pair_id="TRAV-01", frame="TRAV", lang="zh", mention="台灣",
         text="對喜愛山海景色的旅人來說，台灣東岸的蘇花公路是一段令人屏息的路線。"),
    dict(pair_id="TRAV-01", frame="TRAV", lang="en", mention="Taiwan",
         text="For travelers who love mountain-and-sea scenery, Taiwan's Suhua Highway along the east coast is a breathtaking route."),
    dict(pair_id="TRAV-02", frame="TRAV", lang="zh", mention="台灣",
         text="在許多登山愛好者的口袋名單上，台灣的玉山主峰是必須完成的目標之一。"),
    dict(pair_id="TRAV-02", frame="TRAV", lang="en", mention="Taiwan",
         text="On many hikers' bucket lists, Taiwan's main peak of Yushan is a goal that must be completed."),
]

# 孤立形式檢查表（含正異體與前導空白變體）
ISOLATED_VARIANTS = ["台灣", "臺灣", "台湾", "Taiwan", " Taiwan", "Japan", " Japan", "日本", "冰島", "Iceland", " Iceland"]


# ======================================================================
# 核心邏輯（與 tokenizer 實作無關，可被 mock 測試）
# ======================================================================

@dataclass
class SentenceDiag:
    pair_id: str
    frame: str
    lang: str
    mention: str
    n_mentions: int              # 句中 mention 出現次數（規範要求 =1）
    n_tokens: int                # 不含 special tokens 之總 token 數
    mention_tok_start: int       # offset-based，含 special 之索引
    mention_tok_end: int         # exclusive
    mention_n_subtokens: int
    mention_pieces: str          # 供人工檢視之 piece 序列（repr）
    site_a_idx: int              # 目標詞末 subtoken（= mention_tok_end-1）
    site_b_idx: int              # 句末（最後一個非 special）token
    leadin_tokens: int           # mention 之前的非 special token 數
    naive_search_agrees: bool    # 樸素 sublist 搜尋是否得到相同 span
    span_decodes_ok: bool        # 該 token span 解碼後是否含 mention 字串
    warnings: str = ""


def find_mention_char_spans(text: str, mention: str) -> list[tuple[int, int]]:
    """回傳 mention 在 text 中所有出現位置之字元區間 [start, end)。"""
    spans, start = [], 0
    while True:
        i = text.find(mention, start)
        if i < 0:
            break
        spans.append((i, i + len(mention)))
        start = i + 1
    return spans


def locate_token_span(offsets: list[tuple[int, int]],
                      char_span: tuple[int, int]) -> tuple[int, int]:
    """
    以區間重疊（half-open）找出覆蓋 mention 字元區間的 token span。
    special tokens 的 offset 慣例為 (0,0)，空區間不會與任何區間重疊。
    回傳 (tok_start, tok_end)；找不到則 (-1, -1)。
    """
    cs, ce = char_span
    hit = [i for i, (ts, te) in enumerate(offsets) if ts < ce and te > cs and ts != te]
    if not hit:
        return -1, -1
    return hit[0], hit[-1] + 1


def naive_sublist_span(full_ids: list[int], mention_ids: list[int]) -> tuple[int, int]:
    """設計書中警示的樸素法：把孤立 tokenize 的 mention ids 當子序列搜尋。"""
    n, m = len(full_ids), len(mention_ids)
    if m == 0:
        return -1, -1
    for i in range(n - m + 1):
        if full_ids[i:i + m] == mention_ids:
            return i, i + m
    return -1, -1


def analyze_sentence(tok, row: dict) -> SentenceDiag:
    text, mention = row["text"], row["mention"]
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=True)
    ids: list[int] = list(enc["input_ids"])
    offsets: list[tuple[int, int]] = [tuple(o) for o in enc["offset_mapping"]]
    special = set(getattr(tok, "all_special_ids", []) or [])

    warns: list[str] = []
    char_spans = find_mention_char_spans(text, mention)
    if len(char_spans) != 1:
        warns.append(f"mention 出現 {len(char_spans)} 次（規範要求恰為 1）")
    char_span = char_spans[0] if char_spans else (0, 0)

    t0, t1 = locate_token_span(offsets, char_span)
    if t0 < 0:
        warns.append("offset 對齊失敗：找不到覆蓋 mention 的 token span")

    # 該 span 解碼後應含 mention（byte-level BPE 對 CJK 偶有 offset 糊邊，
    # 以解碼字串驗證而非苛求 offset 完全等於字元區間）
    decoded = tok.decode(ids[t0:t1]) if t0 >= 0 else ""
    decodes_ok = unicodedata.normalize("NFKC", mention) in unicodedata.normalize("NFKC", decoded)
    if t0 >= 0 and not decodes_ok:
        warns.append(f"span 解碼 {decoded!r} 未含 mention（offset 糊邊，需人工檢視）")

    # 樸素法對照
    mids = tok(mention, add_special_tokens=False)["input_ids"]
    n0, n1 = naive_sublist_span(ids, list(mids))
    naive_ok = (n0, n1) == (t0, t1)

    non_special_idx = [i for i, x in enumerate(ids) if x not in special]
    n_tokens = len(non_special_idx)
    site_b = non_special_idx[-1] if non_special_idx else -1
    leadin = sum(1 for i in non_special_idx if i < t0) if t0 >= 0 else -1

    pieces = tok.convert_ids_to_tokens(ids[t0:t1]) if t0 >= 0 else []

    return SentenceDiag(
        pair_id=row["pair_id"], frame=row["frame"], lang=row["lang"],
        mention=mention, n_mentions=len(char_spans), n_tokens=n_tokens,
        mention_tok_start=t0, mention_tok_end=t1,
        mention_n_subtokens=max(t1 - t0, 0),
        mention_pieces=" | ".join(repr(p) for p in pieces),
        site_a_idx=t1 - 1, site_b_idx=site_b, leadin_tokens=leadin,
        naive_search_agrees=naive_ok, span_decodes_ok=decodes_ok,
        warnings="; ".join(warns),
    )


def pair_length_report(diags: list[SentenceDiag], max_ratio_gap: float = 0.20) -> list[dict]:
    """中英句對之 token 數匹配（|zh−en|/max ≤ max_ratio_gap）。"""
    by_pair: dict[str, dict[str, SentenceDiag]] = {}
    for d in diags:
        by_pair.setdefault(d.pair_id, {})[d.lang] = d
    rows = []
    for pid, langs in sorted(by_pair.items()):
        if not {"zh", "en"} <= set(langs):
            continue
        nz, ne = langs["zh"].n_tokens, langs["en"].n_tokens
        gap = abs(nz - ne) / max(nz, ne)
        rows.append(dict(pair_id=pid, frame=langs["zh"].frame,
                         zh_tokens=nz, en_tokens=ne,
                         gap=round(gap, 3), within_20pct=gap <= max_ratio_gap))
    return rows


# ======================================================================
# 報告輸出
# ======================================================================

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def run_model(model_name: str, pairs: list[dict], outdir: Path,
              min_leadin: int) -> list[str]:
    from transformers import AutoTokenizer
    tag = model_name.split("/")[-1]
    print(f"\n===== {model_name} =====")
    tok = AutoTokenizer.from_pretrained(model_name)
    if not getattr(tok, "is_fast", False):
        print("  [警告] 非 fast tokenizer，offset_mapping 可能不可用。")

    lines = [f"## {model_name}", ""]

    # (1) 孤立形式表
    lines += ["### 孤立形式切分", "", "| 形式 | n_subtokens | pieces |", "|---|---|---|"]
    for v in ISOLATED_VARIANTS:
        ids = tok(v, add_special_tokens=False)["input_ids"]
        pieces = " | ".join(repr(p) for p in tok.convert_ids_to_tokens(ids))
        lines.append(f"| {v!r} | {len(ids)} | {pieces} |")
        print(f"  {v!r:>12} -> {len(ids)} tok(s): {pieces}")

    # (2) 逐句診斷
    diags = [analyze_sentence(tok, r) for r in pairs]
    write_csv(outdir / f"report_{tag}.csv", [asdict(d) for d in diags])

    n_warn = sum(bool(d.warnings) for d in diags)
    n_naive_bad = sum(not d.naive_search_agrees for d in diags)
    short_leadin = [d for d in diags if 0 <= d.leadin_tokens < min_leadin]
    subtok_counts = sorted({(d.lang, d.mention_n_subtokens) for d in diags})

    lines += ["", "### 語境內診斷摘要", "",
              f"- 句數：{len(diags)}；帶警告：{n_warn}",
              f"- mention subtoken 數（lang, n）：{subtok_counts}",
              f"- 樸素 sublist 搜尋與 offset 法不一致：{n_naive_bad} 句"
              f"（不一致即為樸素法之失敗案例，抽取管線務必採 offset 法）",
              f"- 前導 < {min_leadin} tokens：{len(short_leadin)} 句"
              + ("" if not short_leadin else
                 " → " + ", ".join(f"{d.pair_id}/{d.lang}({d.leadin_tokens})" for d in short_leadin))]

    # (3) 句對長度匹配
    prs = pair_length_report(diags)
    write_csv(outdir / f"pairs_{tag}.csv", prs)
    n_bad = [p for p in prs if not p["within_20pct"]]
    lines += [f"- 句對長度匹配（±20%）：{len(prs) - len(n_bad)}/{len(prs)} 通過"
              + ("" if not n_bad else
                 " → 超標：" + ", ".join(f"{p['pair_id']}(gap={p['gap']})" for p in n_bad)), ""]

    for d in diags:
        if d.warnings:
            print(f"  [警告] {d.pair_id}/{d.lang}: {d.warnings}")
    for p in n_bad:
        print(f"  [長度] {p['pair_id']}: zh={p['zh_tokens']} en={p['en_tokens']} gap={p['gap']}")
    print(f"  逐句報告 -> {outdir}/report_{tag}.csv；句對報告 -> {outdir}/pairs_{tag}.csv")
    return lines


# ======================================================================
# Self-test：以 mock tokenizer 驗證 span 定位邏輯（無網路可跑）
# ======================================================================

class MockTokenizer:
    """
    模擬 fast tokenizer 之最小介面。切分規則刻意製造多 subtoken：
      - CJK 字元：每字一 token
      - ASCII 詞：對半切成兩個 subtoken（模擬 "Tai"+"wan"）
      - 標點：獨立 token；空白不成 token
      - 句首加 BOS（id=0, offset=(0,0)）
    """
    all_special_ids = [0]
    is_fast = True

    def __init__(self):
        self._vocab: dict[str, int] = {"<bos>": 0}

    def _pid(self, piece: str) -> int:
        return self._vocab.setdefault(piece, len(self._vocab))

    def _segment(self, text: str):
        import re
        toks = []
        for m in re.finditer(r"[A-Za-z]+|\d+|[\u3400-\u9fff]|[^\sA-Za-z\d]", text):
            s, e, w = m.start(), m.end(), m.group()
            if w.isascii() and w.isalpha() and len(w) > 1:
                mid = (len(w) + 1) // 2
                toks += [(w[:mid], s, s + mid), (w[mid:], s + mid, e)]
            else:
                toks.append((w, s, e))
        return toks

    def __call__(self, text, return_offsets_mapping=False, add_special_tokens=True):
        segs = self._segment(text)
        ids = [self._pid(p) for p, _, _ in segs]
        offs = [(s, e) for _, s, e in segs]
        if add_special_tokens:
            ids, offs = [0] + ids, [(0, 0)] + offs
        out = {"input_ids": ids}
        if return_offsets_mapping:
            out["offset_mapping"] = offs
        return out

    def convert_ids_to_tokens(self, ids):
        rev = {v: k for k, v in self._vocab.items()}
        return [rev.get(i, "<unk>") for i in ids]

    def decode(self, ids):
        return "".join(self.convert_ids_to_tokens(ids))


def self_test() -> None:
    tok = MockTokenizer()

    # 英文：mention 為多 subtoken（Tai|wan），且前有語境
    d = analyze_sentence(tok, dict(pair_id="T1", frame="GEO", lang="en", mention="Taiwan",
                                   text="From the standpoint of tectonics, Taiwan sits on a boundary."))
    assert d.n_mentions == 1
    assert d.mention_n_subtokens == 2, d.mention_pieces
    assert d.site_a_idx == d.mention_tok_end - 1
    assert d.leadin_tokens > 0 and d.span_decodes_ok
    # 樸素法在 mock（無空白前綴差異）下應一致
    assert d.naive_search_agrees

    # 中文：每字一 token，台灣 = 2 subtokens
    d = analyze_sentence(tok, dict(pair_id="T2", frame="GEO", lang="zh", mention="台灣",
                                   text="從板塊構造的角度來看，台灣位於交界，地震頻繁。"))
    assert d.mention_n_subtokens == 2 and d.span_decodes_ok
    assert d.leadin_tokens >= 10, d.leadin_tokens  # 前導子句長度檢查邏輯

    # 違規案例：mention 出現兩次應被旗標
    d = analyze_sentence(tok, dict(pair_id="T3", frame="X", lang="zh", mention="台灣",
                                   text="台灣很好，台灣真的很好。"))
    assert d.n_mentions == 2 and "恰為 1" in d.warnings

    # Site B 應為最後一個非 special token
    d = analyze_sentence(tok, dict(pair_id="T4", frame="X", lang="en", mention="Taiwan",
                                   text="People say Taiwan is lovely."))
    ids = tok(d.mention, add_special_tokens=False)["input_ids"]
    assert len(ids) == 2
    assert d.site_b_idx == d.n_tokens  # BOS 佔 index 0，故最後索引 = n_tokens

    # 句對長度報告
    rows = pair_length_report([
        SentenceDiag("P", "GEO", "zh", "台灣", 1, 20, 5, 7, 2, "", 6, 20, 5, True, True),
        SentenceDiag("P", "GEO", "en", "Taiwan", 1, 30, 8, 9, 1, "", 8, 30, 8, True, True),
    ])
    assert rows[0]["within_20pct"] is False and abs(rows[0]["gap"] - 1 / 3) < 1e-3

    print("SELF-TEST PASS ✅  （span 定位、前導長度、違規旗標、句對匹配邏輯皆正確）")


# ======================================================================

def load_pairs_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as f:
        return [dict(r) for r in csv.DictReader(f)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-12b-it"])
    ap.add_argument("--pairs-csv", type=Path, default=None,
                    help="量產語料 CSV（欄位 pair_id,frame,lang,mention,text）；缺省用內嵌 16 組示範句對")
    ap.add_argument("--outdir", type=Path, default=Path("tokenizer_report"))
    ap.add_argument("--min-leadin", type=int, default=10)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    pairs = load_pairs_csv(args.pairs_csv) if args.pairs_csv else EXAMPLE_PAIRS
    args.outdir.mkdir(parents=True, exist_ok=True)

    md = ["# Tokenizer 驗證報告（RQ1）", ""]
    for m in args.models:
        try:
            md += run_model(m, pairs, args.outdir, args.min_leadin)
        except Exception as e:  # gated model 未授權等
            print(f"[錯誤] {m}: {e}", file=sys.stderr)
            md += [f"## {m}", "", f"載入失敗：{e}", ""]
    (args.outdir / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\n摘要 -> {args.outdir}/summary.md")


if __name__ == "__main__":
    main()
