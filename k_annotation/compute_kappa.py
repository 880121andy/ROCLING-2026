#!/usr/bin/env python3
"""
Compute inter-annotator Cohen's kappa per dimension for the framing annotation.

Usage:
    python3 compute_kappa.py annotatorA.csv annotatorB.csv

Both CSVs must be the SAME rows (joined on desc_id) with these columns filled:
    d1_frame, d2_anchor, d3_construal, d4_ortho, d5_drift

- D1, D4, D5 -> unweighted (nominal) Cohen's kappa
- D2, D3     -> linear-weighted kappa (ordered categories)
Gate: D1 >= 0.7 AND D2 >= 0.7 to proceed to mass annotation.
Stdlib only. Blank cells (either annotator) are skipped, per-dimension.
"""
import csv, sys
from collections import Counter, defaultdict

# dimension -> (column, ordered categories or None for nominal)
DIMS = {
    "D1 frame":      ("d1_frame",     None,  # nominal
        ["CUL","FOOD","TRAV","GEO","ECON","HIST","LIFE","POL-DOM","POL-INT","OTHER"]),
    "D2 anchor":     ("d2_anchor",    ["INGROUP","NEUTRAL","GEOPOL-OTHER","PRC"],  # ordinal
        ["INGROUP","NEUTRAL","GEOPOL-OTHER","PRC"]),
    "D3 construal":  ("d3_construal", ["EXPERIENTIAL","ANALYTIC","ACTOR"],         # ordinal
        ["EXPERIENTIAL","ANALYTIC","ACTOR"]),
    "D4 ortho":      ("d4_ortho",     None,
        ["TW-TRAD","PRC-SIMP","NA"]),
    "D5 drift":      ("d5_drift",     None,
        ["ON-TOPIC","DRIFTED","HALLUCINATED"]),
}
GATE_DIMS = {"D1 frame", "D2 anchor"}

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return {r["desc_id"]: r for r in csv.DictReader(f)}

def kappa(pairs, order):
    """pairs: list of (labelA, labelB). order=None -> nominal; else linear-weighted."""
    cats = order if order else sorted({x for p in pairs for x in p})
    idx = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    n = len(pairs)
    O = [[0]*k for _ in range(k)]
    for a, b in pairs:
        O[idx[a]][idx[b]] += 1
    rows = [sum(O[i]) for i in range(k)]
    cols = [sum(O[i][j] for i in range(k)) for j in range(k)]
    if order and k > 1:                       # linear weights
        w = lambda i, j: abs(i - j) / (k - 1)
    else:                                     # nominal: disagreement=1
        w = lambda i, j: 0.0 if i == j else 1.0
    num = sum(w(i, j) * O[i][j] for i in range(k) for j in range(k))
    den = sum(w(i, j) * rows[i] * cols[j] / n for i in range(k) for j in range(k))
    kap = 1 - num/den if den else float("nan")
    po = sum(O[i][i] for i in range(k)) / n   # raw agreement (exact match)
    return kap, po, O, cats

def main():
    if len(sys.argv) != 3:
        print(__doc__); sys.exit(1)
    A, B = load(sys.argv[1]), load(sys.argv[2])
    common = [d for d in A if d in B]
    print(f"joined desc_id: {len(common)} (A={len(A)}, B={len(B)})\n")

    gate_ok = True
    for name, (col, order, allowed) in DIMS.items():
        pairs, blank, bad = [], 0, Counter()
        for d in common:
            a, b = A[d].get(col, "").strip(), B[d].get(col, "").strip()
            if not a or not b:
                blank += 1; continue
            for who, lab in (("A", a), ("B", b)):
                if allowed and lab not in allowed:
                    bad[f"{who}:{lab}"] += 1
            pairs.append((a, b))
        weighted = "weighted" if order else "nominal "
        if not pairs:
            print(f"{name:14s} [{weighted}]  no annotated rows yet\n"); continue
        kap, po, O, cats = kappa(pairs, order)
        flag = ""
        if name in GATE_DIMS:
            ok = kap >= 0.7
            gate_ok &= ok
            flag = "  ✅GATE" if ok else "  ❌GATE(<0.7)"
        print(f"{name:14s} [{weighted}]  n={len(pairs):3d}  κ={kap:.3f}  raw-agree={po:.3f}{flag}")
        if blank: print(f"               ({blank} rows skipped: blank)")
        if bad:   print(f"               ⚠ off-codebook labels: {dict(bad)}")
        # disagreements (top) to aid adjudication
        dis = [(d, A[d][col].strip(), B[d][col].strip()) for d in common
               if A[d].get(col,'').strip() and B[d].get(col,'').strip()
               and A[d][col].strip()!=B[d][col].strip()]
        if dis:
            print(f"               {len(dis)} disagreements, e.g.:")
            for d, a, b in dis[:5]:
                print(f"                 {d}  A={a} / B={b}")
        print()

    print("="*60)
    print(f"PILOT GATE (D1 & D2 both κ≥0.7): {'PASS -> mass-annotate' if gate_ok else 'FAIL -> revise codebook, re-annotate fresh batch'}")

if __name__ == "__main__":
    main()
