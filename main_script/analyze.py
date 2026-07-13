#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze.py — RQ1 Phase 03 統計分析骨架
=======================================

輸入：Layer 0 閘門後、且完成 Layer 1–4 標註的描述表（見設計書 §9）。
一列 = 一則 AV 描述，需含欄位：

  刺激 metadata：sent_id, pair_id(=模板), frame(刺激框架代碼), entity,
                  lang(語境語言 en/zh), mention_script, cell_type, site
  Layer 0：       mse, pass_gate
  Layer 1：       construal ∈ {PLACE,POLITY,PEOPLE,ECONOMY,CULTURE,UNDERSPEC}
  Layer 2：       frames（pipe 分隔多標籤，用擴充框架詞彙）
  Layer 3：       china_anchor(0/1), viewpoint, contested(0/1), valence
  Layer 4：       desc_lang

依 §10 分析計畫做五組檢定：
  H1/H2  geopolitical intrusion、china anchoring（logistic，語言主效應）
  H4     DiD：language × entity 交互作用（台灣特異性）
  漂移    刺激框架 × verbalized 框架矩陣，JS divergence + permutation test
  H3     construal 分布 × 語言（多項邏輯迴歸）
  幾何    各框架內兩語言均值向量 cosine、Δ_lang 跨框架一致性（需 --activations）
  H5     Design B：construal ~ context_language × mention_script

**這是骨架**：混合效應迴歸以 statsmodels 近似（logit + 依模板 cluster-robust SE）；
正式投稿建議改用 R lme4::glmer 的 crossed random effects (1|template)+(1|frame)。
腳本會逐項標明所用近似與限制。

先用假標註跑通全流程（不需真資料，驗證 plumbing）：

    python pipeline/analyze.py --demo --out analysis/demo

跑真資料：

    python pipeline/analyze.py \
        --annotated annotations/qwen_layer1-4.csv \
        --activations activations/qwen/activations_Qwen2.5-7B-Instruct.parquet \
        --gated-only --out analysis/qwen
"""

from __future__ import annotations

import argparse
from pathlib import Path

# 刺激框架代碼 → verbalized 框架名（frame match 用）
STIM2VERB = {
    "GEO": "Natural_geography", "POL-INT": "International_status",
    "POL-DOM": "Domestic_politics", "ECON": "Commerce_and_technology",
    "CUL": "Cultural_practice", "HIST": "History",
    "LIFE": "Everyday_life", "TRAV": "Tourism",
}
# 地緣政治侵入集合（H1 主檢定量）
GEO_INTRUSION = {"Sovereignty_dispute", "Military_conflict", "International_status"}
VERB_VOCAB = list(STIM2VERB.values()) + ["Sovereignty_dispute", "Military_conflict", "Media_discourse"]


# ======================================================================
# 派生測量
# ======================================================================

def parse_frames(s) -> list[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [x.strip() for x in s.replace(";", "|").split("|") if x.strip()]


def add_derived(df):
    import pandas as pd  # noqa
    df = df.copy()
    df["frames_list"] = df["frames"].apply(parse_frames)

    def geo_intr(row):
        stim = STIM2VERB.get(row["frame"])
        intruding = GEO_INTRUSION - ({stim} if stim else set())
        return int(bool(set(row["frames_list"]) & intruding))

    def frame_match(row):
        stim = STIM2VERB.get(row["frame"])
        return int(stim in row["frames_list"]) if stim else 0

    df["geo_intrusion"] = df.apply(geo_intr, axis=1)
    df["frame_match"] = df.apply(frame_match, axis=1)
    df["china_anchor"] = df.get("china_anchor", 0)
    df["is_polity"] = (df["construal"] == "POLITY").astype(int)
    return df


# ======================================================================
# H1/H2/H4：logistic + cluster-robust（依模板 pair_id）
# ======================================================================

def fit_logit(df, outcome: str, md: list[str], label: str, did: bool):
    md += [f"\n## {label}：`{outcome}`", ""]
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        md.append("_statsmodels 未安裝，略過（pip install statsmodels）。_")
        return
    sub = df.dropna(subset=[outcome]).copy()
    if sub[outcome].nunique() < 2:
        md.append(f"_結果變項無變異（全為 {sub[outcome].iloc[0]}），略過。_")
        return
    formula = f"{outcome} ~ C(lang)" + (" * C(entity)" if did else "")
    try:
        # cluster-robust SE by template（模板為隨機效應之近似）
        res = smf.logit(formula, data=sub).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": sub["pair_id"]})
    except Exception as e:  # 分離、完美預測等
        md.append(f"_擬合失敗：{e}_")
        return
    md += [f"公式：`{formula}`（cluster-robust by pair_id, n={len(sub)}）", "",
           "| 項 | coef | SE | z | p |", "|---|---|---|---|---|"]
    for name in res.params.index:
        md.append(f"| {name} | {res.params[name]:.3f} | {res.bse[name]:.3f} | "
                  f"{res.tvalues[name]:.2f} | {res.pvalues[name]:.4f} |")
    if did:
        inter = [n for n in res.params.index if ":" in n]
        md.append(f"\n**H4 DiD** 交互作用項：{inter or '（無，檢查 entity 水準）'} "
                  f"→ 顯著且與主效應同號即支持『台灣特異』。")


# ======================================================================
# 框架漂移矩陣 + JS divergence + permutation test
# ======================================================================

def frame_drift(df, md: list[str], n_perm: int = 2000):
    import numpy as np
    md += ["\n## 框架漂移矩陣 + JS divergence（H1 補充）", ""]
    R, C = len(STIM2VERB), len(VERB_VOCAB)
    rk = {k: i for i, k in enumerate(STIM2VERB)}
    ck = {k: i for i, k in enumerate(VERB_VOCAB)}

    # 預先攤平成 (row_i, stim_row, verb_col) 三元組，permutation 只重排 lang 遮罩
    d = df.reset_index(drop=True)
    lang_en = (d["lang"] == "en").to_numpy()
    pi, pr, pc = [], [], []
    for i, r in enumerate(d.itertuples(index=False)):
        ri = rk.get(r.frame)
        if ri is None:
            continue
        for fr in r.frames_list:
            if fr in ck:
                pi.append(i); pr.append(ri); pc.append(ck[fr])
    pi, pr, pc = np.array(pi), np.array(pr), np.array(pc)

    def matrix(mask_en_rows):
        sel = mask_en_rows[pi]
        M = np.zeros((R, C))
        np.add.at(M, (pr[sel], pc[sel]), 1)
        return M

    def js(p, q):
        p = p / p.sum() if p.sum() else p
        q = q / q.sum() if q.sum() else q
        m = 0.5 * (p + q)
        def kl(a, b):
            mask = a > 0
            return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
        return 0.5 * kl(p, m) + 0.5 * kl(q, m)

    Men, Mzh = matrix(lang_en), matrix(~lang_en)
    obs = js(Men.flatten(), Mzh.flatten())

    rng = np.random.default_rng(0)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(lang_en)
        ge += int(js(matrix(perm).flatten(), matrix(~perm).flatten()) >= obs)
    p = (ge + 1) / (n_perm + 1)
    md += [f"- 兩語言 verbalized-frame 分布 JS divergence（bits）＝ **{obs:.4f}**",
           f"- permutation test（{n_perm} 次打亂 lang）：p ＝ **{p:.4f}**",
           f"- 台灣核心語料，en n={int(lang_en.sum())} / zh n={int((~lang_en).sum())}。矩陣另存 CSV。"]
    return Men, Mzh


# ======================================================================
# H3：construal 分布 × 語言（多項邏輯迴歸）
# ======================================================================

def construal_dist(df, md: list[str]):
    import pandas as pd
    md += ["\n## H3：construal 分布 × 語言", ""]
    ct = pd.crosstab(df["construal"], df["lang"], normalize="columns")
    md += ["| construal | en | zh |", "|---|---|---|"]
    for c in ct.index:
        row = ct.loc[c]
        md.append(f"| {c} | {row.get('en', 0):.2%} | {row.get('zh', 0):.2%} |")
    try:
        import statsmodels.formula.api as smf
        sub = df.copy()
        sub["y"] = sub["construal"].astype("category").cat.codes
        res = smf.mnlogit("y ~ C(lang)", data=sub).fit(disp=0)
        md += ["", f"多項邏輯迴歸 `construal ~ C(lang)`：LLR p ＝ {res.llr_pvalue:.4g}"]
    except Exception as e:
        md.append(f"\n_MNLogit 略過：{e}_")


# ======================================================================
# 幾何側寫（需原始向量）
# ======================================================================

def geometry(df, activations: Path, md: list[str]):
    import numpy as np
    md += ["\n## 幾何側寫：Δ_lang 一致性（Site A）", ""]
    if not activations or not activations.exists():
        md.append("_未提供 --activations，略過幾何分析。_")
        return
    import pyarrow.parquet as pq
    tbl = pq.read_table(activations)
    ids = tbl.column("vector_id").to_pylist()
    flat = (tbl.column("activation_vector").combine_chunks().flatten()
            .to_numpy(zero_copy_only=False).astype(np.float32))
    vec = {v: flat.reshape(len(tbl), -1)[i] for i, v in enumerate(ids)}

    # 只取 Site A、台灣核心；每 (frame, lang) 求均值向量
    sub = df[(df.site == "A") & (df.entity.isin(["台灣", "Taiwan", "TW"]))]
    means: dict[tuple, np.ndarray] = {}
    for (frame, lang), g in sub.groupby(["frame", "lang"]):
        vs = [vec[v] for v in g["vector_id"].unique() if v in vec]
        if vs:
            means[(frame, lang)] = np.mean(vs, axis=0)

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    deltas = {}
    md += ["| frame | cos(en,zh) 均值向量 |", "|---|---|"]
    for frame in STIM2VERB:
        e, z = means.get((frame, "en")), means.get((frame, "zh"))
        if e is not None and z is not None:
            md.append(f"| {frame} | {cos(e, z):.3f} |")
            deltas[frame] = e - z
    if len(deltas) >= 2:
        ds = list(deltas.values())
        pair_cos = [cos(ds[i], ds[j]) for i in range(len(ds)) for j in range(i + 1, len(ds))]
        md.append(f"\n**Δ_lang 跨框架一致性**：{len(ds)} 個框架方向向量之平均兩兩 cosine "
                  f"＝ **{np.mean(pair_cos):.3f}**（高＝語言效應方向跨框架一致）。")


# ======================================================================
# H5：Design B
# ======================================================================

def design_b(df, md: list[str]):
    md += ["\n## H5：Design B（語境語言 × 提及詞形）", ""]
    # Design B 唯一可靠辨識：其模板才有 codeswitch 格 → 取這些模板的全部四格
    db_pairs = set(df.loc[df["cell_type"] == "codeswitch", "pair_id"])
    sub = df[df["pair_id"].isin(db_pairs) & df["mention_script"].isin(["hanzi", "latin"])].copy()
    if len(sub) < 8 or sub["mention_script"].nunique() < 2 or sub["lang"].nunique() < 2:
        md.append("_Design B 資料不足（需含 hanzi/latin 兩詞形），略過。_")
        return
    try:
        import statsmodels.formula.api as smf
        sub["context_language"] = sub["lang"]
        res = smf.logit("is_polity ~ C(context_language) * C(mention_script)",
                        data=sub).fit(disp=0)
        md += [f"公式：`is_polity ~ C(context_language)*C(mention_script)`（n={len(sub)}）", "",
               "| 項 | coef | p |", "|---|---|---|"]
        for n in res.params.index:
            md.append(f"| {n} | {res.params[n]:.3f} | {res.pvalues[n]:.4f} |")
        md.append("\n語境語言主效應 ≫ 詞形主效應 → 支持『語境主導』（H5 一支）；反之則詞形主導。")
    except Exception as e:
        md.append(f"_擬合失敗：{e}_")


# ======================================================================
# 假標註產生器（--demo）
# ======================================================================

def make_demo(seed: int = 0):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []

    def emit(sent_id, pair_id, frame, entity, lang, mention_script, cell_type, is_db):
        # 植入可偵測效應：英語語境 + 台灣 → 較高地緣侵入/china anchor/POLITY
        tw = entity in ("台灣", "Taiwan")
        base_intr = 0.10 + (0.35 if (lang == "en" and tw) else 0.05 if lang == "en" else 0.0)
        for k in range(5):
            for site in ("A", "B"):
                intr = rng.random() < base_intr
                anchor = rng.random() < (base_intr * 0.8)
                if intr:
                    construal = "POLITY"
                    frames = STIM2VERB.get(frame, "Everyday_life") + "|International_status"
                    if rng.random() < 0.4:
                        frames += "|Sovereignty_dispute"
                else:
                    construal = rng.choice(["PLACE", "PEOPLE", "ECONOMY", "CULTURE", "POLITY"],
                                           p=[0.35, 0.2, 0.2, 0.15, 0.10])
                    frames = STIM2VERB.get(frame, "Everyday_life")
                rows.append(dict(
                    desc_id=f"{sent_id}#site{site}#k{k}", vector_id=f"{sent_id}#site{site}",
                    sent_id=sent_id, pair_id=pair_id, frame=frame, entity=entity, lang=lang,
                    mention_script=mention_script, cell_type=cell_type, site=site,
                    mse=float(rng.gamma(2, 0.3)), pass_gate=1,
                    construal=construal, frames=frames,
                    china_anchor=int(anchor), viewpoint=rng.choice(["internal", "external", "neutral"]),
                    contested=int(intr and rng.random() < 0.3),
                    valence=rng.choice(["pos", "neu", "neg"]), desc_lang=lang))

    # 核心 192（8 框架×12 模板×2 語言，台灣）
    for fi, frame in enumerate(STIM2VERB):
        for t in range(12):
            pid = f"{frame}-{t:02d}"
            for lang, ms in (("zh", "hanzi"), ("en", "latin")):
                emit(f"{pid}|{lang}", pid, frame, "台灣", lang, ms, "baseline", False)
    # 控制 120（日、冰 × 5 框架 × 6 模板 × 2 語言）
    for entity in ("日本", "冰島"):
        for frame in ("GEO", "ECON", "CUL", "TRAV", "LIFE"):
            for t in range(6):
                pid = f"{frame}-{entity}-{t:02d}"
                for lang, ms in (("zh", "hanzi"), ("en", "latin")):
                    emit(f"{pid}|{lang}", pid, frame, entity, lang, ms, "baseline", False)
    # Design B 48（GEO,ECON × 6 模板 × 4 格）
    for frame in ("GEO", "ECON"):
        for t in range(6):
            pid = f"{frame}-DB-{t:02d}"
            for lang, ms, cell in (("zh", "hanzi", "baseline"), ("zh", "latin", "codeswitch"),
                                   ("en", "latin", "baseline"), ("en", "hanzi", "codeswitch")):
                emit(f"{pid}|{lang}|{ms}", pid, frame, "台灣", lang, ms, cell, True)

    import pandas as pd
    return pd.DataFrame(rows)


def main() -> None:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", type=Path, help="Layer 1–4 標註後描述表 CSV")
    ap.add_argument("--activations", type=Path, default=None, help="原始向量 parquet（幾何分析用）")
    ap.add_argument("--gated-only", action="store_true", help="只用 pass_gate==1 之描述做主分析")
    ap.add_argument("--demo", action="store_true", help="用假標註跑通全流程")
    ap.add_argument("--out", type=Path, default=Path("analysis/out"))
    args = ap.parse_args()

    if args.demo:
        df = make_demo()
        print(f"[demo] 產生 {len(df)} 則假描述")
    elif args.annotated:
        df = pd.read_csv(args.annotated, encoding="utf-8-sig")
    else:
        ap.error("需 --annotated 或 --demo")

    if args.gated_only and "pass_gate" in df:
        n = len(df)
        df = df[df["pass_gate"] == 1].copy()
        print(f"閘門篩選：{n} → {len(df)}")

    df = add_derived(df)
    core = df[df["entity"].isin(["台灣", "Taiwan", "TW"])]           # 台灣核心（H1/H2/H3/漂移）
    core_ctrl = df[df["cell_type"] == "baseline"]                    # 核心+控制（H4 DiD）

    md = ["# RQ1 Phase 03 分析報告（骨架）", "",
          f"- 描述數：{len(df)}（台灣核心 {len(core)}；核心+控制 {len(core_ctrl)}）",
          "- ⚠️ 混合效應以 logit + cluster-robust 近似；正式版建議 R glmer。"]

    fit_logit(core, "geo_intrusion", md, "H1 地緣政治侵入（語言主效應，僅台灣）", did=False)
    fit_logit(core_ctrl, "geo_intrusion", md, "H4 DiD：地緣侵入 language × entity", did=True)
    fit_logit(core, "china_anchor", md, "H2 China anchoring（語言主效應，僅台灣）", did=False)
    fit_logit(core_ctrl, "china_anchor", md, "H4 DiD：china anchor language × entity", did=True)
    Men, Mzh = frame_drift(core, md)
    construal_dist(core, md)
    geometry(df, args.activations, md)
    design_b(df, md)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rep = Path(f"{args.out}.report.md")
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    # 漂移矩陣另存
    import numpy as np
    np.savetxt(f"{args.out}.drift_en.csv", Men, delimiter=",", fmt="%.0f")
    np.savetxt(f"{args.out}.drift_zh.csv", Mzh, delimiter=",", fmt="%.0f")
    print(f"報告 -> {rep}")
    print(f"漂移矩陣 -> {args.out}.drift_{{en,zh}}.csv")


if __name__ == "__main__":
    main()
