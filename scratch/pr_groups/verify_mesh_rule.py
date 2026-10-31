"""Depth-window verdicts (bundles_verify.py) vs a mesh rule on the same counts (network-free).
Mesh rule: fragment < 10 outputs; R1-R6-like >= 40 outputs, >= 80% in the lamina mesh, inputs <= 100; R7/R8-like
>= 10 outputs, >= 20% in the medulla mesh, inputs <= 150, lamina share < 80%; else other.
Usage: verify_mesh_rule.py [--side right|left]"""
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from faab.config import FIGURES
from faab.pr.side import PREFIX, parse as parse_side
from faab.config import fig

side, _ = parse_side()
df = pd.read_csv(fig(f"{PREFIX[side]}bundles_roots_verified.tsv"), sep="\t", dtype={"root": str, "known": str})
df["known"] = df.known.fillna("")


def mesh_rule(r):
    if r.n_pre < 10:
        return "fragment"
    if r.n_pre >= 40 and r.la_frac >= 0.8 and r.n_post <= 100:
        return "R1-R6-like"
    if r.me_frac >= 0.2 and r.la_frac < 0.8 and r.n_post <= 150:
        return "R7/R8-like"
    if r.la_frac >= 0.8 and r.n_post <= 100:
        return "R1-R6-like (few sites)"
    return "other"


df["mesh_verdict"] = [mesh_rule(r) for r in df.itertuples()]
pd.set_option("display.width", 250)
print(f"side {side}: depth verdict (rows) vs mesh verdict (columns), all {len(df)} PR-like axons")
print(pd.crosstab(df.verdict, df.mesh_verdict, margins=True).to_string())
dis = df[df.verdict != df.mesh_verdict]
print(f"\ndisagreements {len(dis)}; n_post / depth / lamina share by pair:")
for (a, b), sub in dis.groupby(["verdict", "mesh_verdict"]):
    print(f"  {a} -> {b}: {len(sub)}; n_pre p50 {sub.n_pre.median():.0f}; n_post p50 {sub.n_post.median():.0f}; depth "
          f"p10/50/90 {sub.presyn_depth_um.quantile([0.1, 0.5, 0.9]).round(0).tolist()}; LA {sub.la_frac.median():.2f}; "
          f"ME {sub.me_frac.median():.2f}; clusters {sub.cluster.value_counts().head(5).to_dict()}")
g = df.groupby("cluster")
both = pd.DataFrame({"n": g.size(), "R16_depth": g.verdict.apply(lambda s: int(s.str.startswith("R1-R6").sum())),
                     "R16_mesh": g.mesh_verdict.apply(lambda s: int(s.str.startswith("R1-R6").sum())),
                     "R78_mesh": g.mesh_verdict.apply(lambda s: int((s == "R7/R8-like").sum())),
                     "frag": g.mesh_verdict.apply(lambda s: int((s == "fragment").sum())),
                     "other_mesh": g.mesh_verdict.apply(lambda s: int((s == "other").sum()))})
ch = both[(both.R16_depth != both.R16_mesh) & (both.n >= 4)].sort_values("n", ascending=False)
print(f"\nclusters (>= 4 axons) whose R1-R6 count changes: {len(ch)}")
print(ch.to_string())
