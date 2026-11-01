"""Calibrate ommatidia.py on known groups (2026-11-01): the proofread groups 1-8 and v1's clean groups (6 outer + R7 +
R8, no extras; figures/refine_<V1RUN>/summary.tsv). With the column-zone lines of an ommatidia.py run: per group the
members seen as straight lines, each member's lateral distance to the center line of the others (leave-one-out), and
the lateral distance between the center lines of neighboring groups. These set D_MAX (reach) and MERGE_UM.
Network-free. Usage: ommatidia_calibrate.py RUN V1RUN"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "pr_groups"))
import ommatidia as O  # noqa: E402
from faab.config import fig  # noqa: E402

run, v1 = sys.argv[1], sys.argv[2]
d = fig(f"ommatidia/{run}")
L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str})
L = L[L.angle <= O.MAX_ANGLE].reset_index(drop=True)
C = O.Lines(L)
tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
tab = tab[(tab.side == "right") & tab.cls.isin(["R1-R6", "R7", "R8", "R7/R8"]) & ~tab.notes.str.contains("OFF-BUNDLE|SECOND BUNDLE")]
S = pd.read_csv(os.path.join(fig(f"refine_{v1}"), "summary.tsv"), sep="\t").fillna("")
clean = {int(x.group) for x in S.itertuples(index=False) if x.status == "ok" and float(x.outer) == 6 and float(x.extras) == 0
         and x.R7 not in ("", "-") and x.R8 not in ("", "-")}
groups = {}
for g, t in tab.groupby(tab.group.astype(int)):
    if g <= 8 or g in clean:
        groups[g] = [C.i[r] for r in t.root_now if r in C.i]
seen = {g: len(m) for g, m in groups.items()}
print(f"{len(groups)} known groups (1-8 + {len(clean)} v1 clean); members seen as straight lines in the column zone: "
      f"{sum(seen.values())}/{sum(len(t) for g, t in tab.groupby(tab.group.astype(int)) if g in groups)}; "
      f"per group {pd.Series(seen).value_counts().sort_index().to_dict()}")
loo, cents = [], {}
for g, m in groups.items():
    if len(m) < 3:
        continue
    cents[g] = O.fit_center(C, m)
    for k in m:
        ce = O.centers_as_lines([O.fit_center(C, [q for q in m if q != k])])
        loo.append((g, C.roots[k], O.line_dist(C, k, ce, 0)))
D = pd.DataFrame(loo, columns=["group", "root", "dist"])
q = D.dist.quantile([0.5, 0.9, 0.95, 0.99]).round(2).tolist()
print(f"member -> center of the others: median {q[0]} um, p90 {q[1]}, p95 {q[2]}, p99 {q[3]}, max {D.dist.max():.2f}")
print(D.sort_values("dist", ascending=False).head(8).to_string(index=False))
F = pd.read_csv(os.path.join(d, "features.tsv"), sep="\t", dtype={"root": str}).set_index("root")
tl = []
for g, t in tab[tab.cls == "R1-R6"].groupby(tab.group.astype(int)):
    if g not in groups:
        continue
    pts = {r: F.loc[r, ["term_x", "term_y", "term_z"]].to_numpy(float) for r in t.root_now if r in F.index}
    pts = {r: p for r, p in pts.items() if np.isfinite(p).all()}
    for r, p in pts.items():
        if len(pts) >= 3:
            tl.append((g, r, float(np.linalg.norm(p - np.mean([q for k, q in pts.items() if k != r], axis=0)))))
TL = pd.DataFrame(tl, columns=["group", "root", "dist"])
print(f"R1-R6 terminal -> mean of the other terminals: median {TL.dist.median():.1f} um, p95 {TL.dist.quantile(0.95):.1f}, "
      f"p99 {TL.dist.quantile(0.99):.1f}, max {TL.dist.max():.1f} ({len(TL)} axons)")
print(TL.sort_values("dist", ascending=False).head(6).to_string(index=False))
gs = sorted(cents)
CL = O.centers_as_lines([cents[g] for g in gs])
nn = []
for a_, g in enumerate(gs):
    ds = [O.line_dist(CL, a_, CL, b_) for b_ in range(len(gs)) if b_ != a_
          and np.linalg.norm(cents[g]["p"] - cents[gs[b_]]["p"]) < 40]
    ds = [x for x in ds if np.isfinite(x)]
    if ds:
        nn.append((g, min(ds)))
N = pd.DataFrame(nn, columns=["group", "nearest"])
print(f"center -> nearest known center (lateral): p5 {N.nearest.quantile(0.05):.1f}, p10 {N.nearest.quantile(0.1):.1f}, "
      f"median {N.nearest.median():.1f}, p90 {N.nearest.quantile(0.9):.1f} um")
print(N.sort_values("nearest").head(8).to_string(index=False))
