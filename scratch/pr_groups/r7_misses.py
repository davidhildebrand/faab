"""Which right-side R7s does the bundles_verify.py rule miss, and why? (network-free)

Reads the newest figures/*_pr_classes_mesh_shares.tsv (r78_mesh_check.py) and the same local checkpoint's output
positions. Per output: distance to the LA_R and ME_R mesh surfaces (nearest vertex, signed: negative inside), so the
'outside both meshes' outputs can be placed (near the medulla surface, in the chiasm, or near the lamina). Prints every
R7 the rule does not call R7-like, with its group mates' calls, and scores margin variants of the medulla test."""
import glob
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from faab import ol_meshes, synsnap
from faab.config import FIGURES

VOX_UM = np.array([0.016, 0.016, 0.045])
LA_MIN, ME_MAX_LAMINA, R8_IN_PER_OUT = 0.6, 0.1, 0.2

path = sorted(glob.glob(os.path.join(FIGURES, "*_pr_classes_mesh_shares.tsv")))[-1]
g = pd.read_csv(path, sep="\t")
g = g[g.n_pre >= 10].copy()
g["in_per_out"] = (g.n_post / g.n_pre).round(3)


def rule(r):
    if r.LA >= LA_MIN and r.ME < ME_MAX_LAMINA:
        return "R8-like" if r.in_per_out >= R8_IN_PER_OUT else "R1-R6-like"
    return "R7-like" if r.LA < LA_MIN else "other"


g["call"] = [rule(r) for r in g.itertuples()]

S = synsnap.Snapshot()
pre = S.synapses(pre=sorted(set(g.root_now)), when="latest", positions=True)
pts = pre[["pre_x", "pre_y", "pre_z"]].to_numpy(float) * VOX_UM
reg = ol_meshes.assign(pts, ["LA_R", "ME_R"])
for lab in ("LA_R", "ME_R"):
    v, _ = ol_meshes.mesh(lab)
    d, _ = cKDTree(v / 1000.0).query(pts)
    pre[f"d_{lab}"] = np.where(reg == lab, -d, d)
pre["reg"] = reg


def per_cell(p):
    out = p[p.reg == "none"]
    return pd.Series({
        "ME5": np.mean(p.d_ME_R <= 5), "ME10": np.mean(p.d_ME_R <= 10),
        "far_LA10": np.mean(p.d_LA_R > 10), "far_LA20": np.mean(p.d_LA_R > 20),
        "none_dME_p50": out.d_ME_R.median() if len(out) else np.nan,
        "none_dLA_p50": out.d_LA_R.median() if len(out) else np.nan,
        "max_dLA": p.d_LA_R.max()})


feat = pre.groupby("pre_root").apply(per_cell)
g = g.join(feat, on="root_now")
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
cols = ["group", "class", "status", "root_now", "n_pre", "n_post", "in_per_out", "LA", "ME", "outside", "ME5", "ME10",
        "far_LA10", "far_LA20", "none_dME_p50", "none_dLA_p50", "max_dLA", "call"]

print(f"{os.path.basename(path)}; checkpoint {S.at('latest')}")
print("\nR7 calls (all / proofread-checked):")
print(pd.crosstab(g[g["class"] == "R7"].call, g[g["class"] == "R7"].confirmed))
miss = g[(g["class"] == "R7") & (g.call != "R7-like")]
print(f"\nR7s not called R7-like ({len(miss)}):")
print(miss[cols].round(2).to_string(index=False))
print("\ngroup mates of the proofread misses:")
mates = g[g.group.isin(miss[miss.confirmed].group)].sort_values(["group", "class"])
print(mates[cols].round(2).to_string(index=False))

print("\nper class, feature p10/50/90:")
for f in ("n_pre", "in_per_out", "LA", "ME", "outside", "ME5", "ME10", "far_LA10", "none_dME_p50", "none_dLA_p50"):
    print(f"  {f:13s} " + "; ".join(f"{c} {g[g['class'] == c][f].quantile([0.1, 0.5, 0.9]).round(2).tolist()}"
                                     for c in ("R1-R6", "R8", "R7")))
print("\ninputs per output, upper tail p90/p95/max: " + "; ".join(
    f"{c} {g[g['class'] == c].in_per_out.quantile([0.9, 0.95, 1.0]).round(2).tolist()}" for c in ("R1-R6", "R8")))
print("proofread: " + "; ".join(
    f"{c} {sorted(g[(g['class'] == c) & g.confirmed].in_per_out.round(2).tolist())}" for c in ("R8", "R7")))

WANT = (("R1-R6", "R1-R6-like"), ("R8", "R8-like"), ("R7", "R7-like"))


def score(name, f):
    s = g.assign(call=[f(r) for r in g.itertuples()])
    c = s[s.confirmed]
    hit = "; ".join(f"{k} {(c[c['class'] == k].call == v).sum()}/{(c['class'] == k).sum()}" for k, v in WANT)
    allhit = "; ".join(f"{k} {(s[s['class'] == k].call == v).sum()}/{(s['class'] == k).sum()}" for k, v in WANT)
    print(f"\n{name}\n  proofread: {hit}\n  all:       {allhit}")


def variant(me_key="ME", me_max=0.1, r7_in_per_out=None, me_is_r7=True):
    def f(r):
        me = getattr(r, me_key)
        if r.LA >= LA_MIN and me < me_max:
            if r7_in_per_out is not None and r.in_per_out >= r7_in_per_out:
                return "R7-like"
            return "R8-like" if r.in_per_out >= R8_IN_PER_OUT else "R1-R6-like"
        return "R7-like" if (r.LA < LA_MIN or (me_is_r7 and me >= me_max)) else "other"
    return f


score("current (LA >= 0.6 and ME < 0.1 -> lamina-ending; LA < 0.6 -> R7; else other)", variant(me_is_r7=False))
score("ME -> R7: any cell with ME >= 0.1 is R7-like", variant("ME", 0.1))
for k in ("ME5", "ME10"):
    score(f"medulla with margin: {k} >= 0.1 -> R7-like", variant(k, 0.1))
for t in (1.1, 1.3, 1.5):
    score(f"ME5 margin + lamina-ending with inputs per output >= {t} -> R7-like", variant("ME5", 0.1, t))
