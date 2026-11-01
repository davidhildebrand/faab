"""Medulla / lamina cable shares and synapse counts per PR class, proofread rows vs tracking candidates, from a
pr_classes_mesh_shares TSV - to set the bundle-tracking medulla-share check. Network-free.
Usage: me_share_by_class.py MESH_SHARES.tsv"""
import sys

import pandas as pd

M = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
for c in ("LA", "ME", "outside", "n_pre", "n_post"):
    M[c] = pd.to_numeric(M[c], errors="coerce")
M["proofread"] = M.confirmed.map({"True": "proofread", "False": "candidate"})
q = lambda s: s.quantile([0, 0.1, 0.5, 0.9, 1]).round(2).tolist()
for (cls, pr), g in M.groupby(["class", "proofread"]):
    print(f"{cls:6} {pr:9} n={len(g):3}  ME {q(g.ME)}  LA {q(g.LA)}  n_pre {q(g.n_pre)}")
for cls in ("R7", "R8"):
    g = M[(M["class"] == cls)].sort_values("ME", ascending=False).head(8)
    print(f"\n{cls} highest ME:\n" + g[["group", "root_now", "proofread", "LA", "ME", "n_pre", "n_post"]].to_string(index=False))
