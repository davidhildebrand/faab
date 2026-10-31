"""Candidate synapse rules scored on the right-side classes of data/pr_groups.tsv (network-free; reads the newest
figures/*_pr_classes_mesh_shares.tsv from r78_mesh_check.py). Classes are geometric (R8 = central axon, R7 = ring
cell with medulla outputs / fewest lamina outputs), so they are an independent reference for synapse rules."""
import glob
import os

import numpy as np
import pandas as pd

from faab.config import FIGURES
from faab.config import fig

path = sorted(glob.glob(fig("*_pr_classes_mesh_shares.tsv")))[-1]
g = pd.read_csv(path, sep="\t")
g = g[g.n_pre >= 10].copy()
g["in_per_out"] = (g.n_post / g.n_pre).round(3)
print(f"{os.path.basename(path)}: {len(g)} cells with >= 10 outputs")
for cls, s in g.groupby("class"):
    print(f"  {cls}: inputs per output p10/25/50/75/90 {s.in_per_out.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(2).tolist()}; "
          f"outputs p10/50/90 {s.n_pre.quantile([0.1, 0.5, 0.9]).round(0).astype(int).tolist()}")


def score(name, f):
    out = g.assign(call=[f(r) for r in g.itertuples()])
    print(f"\n{name}")
    print(pd.crosstab(out["class"], out.call).to_string())
    c = out[out.confirmed]
    print("  proofread/checked only: " + "; ".join(f"{k}: {v}" for k, v in
                                                  c.groupby("class").call.apply(lambda s: s.value_counts().to_dict()).items()))


# current bundles_verify.py rule (position part only; all cells here have <= 150 inputs except a few)
score("current: lamina >= 0.8 -> R1-R6; medulla >= 0.2 and lamina < 0.8 -> 'R7/R8'; else other",
      lambda r: "R1-R6" if r.LA >= 0.8 else ("R7/R8" if r.ME >= 0.2 else "other"))
score("A: lamina >= 0.6 and medulla < 0.1 -> lamina-terminating (R1-R6 or R8); lamina < 0.6 -> R7; else other",
      lambda r: "lamina PR" if (r.LA >= 0.6 and r.ME < 0.1) else ("R7" if r.LA < 0.6 else "other"))
for thr in (0.1, 0.15, 0.2, 0.3):
    score(f"B: as A, lamina PRs split by inputs per output >= {thr} -> R8-like",
          lambda r, thr=thr: (("R8-like" if r.in_per_out >= thr else "R1-R6-like") if (r.LA >= 0.6 and r.ME < 0.1)
                              else ("R7" if r.LA < 0.6 else "other")))
