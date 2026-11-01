"""Diagnostics of lmc_cand_classify.py (2026-11-01): what makes candidates 'unlike any typed LMC' (by side,
completeness, soma), and per call the candidates' key features next to the typed cells' (medians), to see whether the
calls look like their types. Network-free. Usage: lmc_cand_diag.py"""
import pandas as pd

from faab.config import fig

T = pd.read_csv(fig("lmc_cand_typing.tsv"), sep="\t", dtype={"root": str}).fillna({"soma_by": ""})
F = pd.read_csv(fig("lmc_cand_features.tsv"), sep="\t", dtype={"root": str})
F0 = pd.read_csv(fig("lmc_features.tsv"), sep="\t", dtype={"root": str}).fillna({"label": ""})
print("share typical, by side x complete:")
print(pd.crosstab(T.side, T.complete, values=T.typical, aggfunc="mean").round(2).to_string())
print("\nshare typical, by soma:")
print(T.groupby(T.soma_by.replace("", "none")).typical.mean().round(2).to_string())
cols = ["n_in_LA", "n_out_LA", "n_in_ME", "n_out_ME", "ME_out_depth_med", "ME_depth_max", "LA_spread_p90_um"]
C = F.merge(T[["root", "call", "confidence", "typical"]], on="root")
print("\nmedians: typed cells (label) vs candidates called that type (high / all)")
for t in ("L1", "L2", "L3", "L4", "L5"):
    a = F0[F0.label == t][cols].median()
    b = C[(C.call == t) & (C.confidence == "high")][cols].median()
    c = C[C.call == t][cols].median()
    print(f"{t}: " + "; ".join(f"{k} {a[k]:.2f}/{b[k]:.2f}/{c[k]:.2f}" for k in cols))
print("\nside differences among typical, high-confidence candidates (medians, L vs R):")
H = C[C.confidence == "high"]  # F already carries the side
print(H.groupby(["call", "side"])[["ME_out_depth_med", "ME_depth_max", "LA_spread_p90_um"]].median().round(2).to_string())
