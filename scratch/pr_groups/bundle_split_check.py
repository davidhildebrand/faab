"""Is a tracked group one bundle or two? (David 2026-11-01, group 19: 13 photoreceptor-like axons in one core.)
From a groups_refine.py run: the photoreceptor-like axons of the group (details_g<G>.tsv: listed, census, PR-like, and
the not-added medulla axons), their positions across the bundle in every tracked slice (track_g<G>.tsv, frame from
frames.pkl), the median distance of every pair over the slices they share, and an average-linkage split into two sets;
then the same for their lamina terminals (median of each axon's presynaptic sites, presyn.tsv, across the axis).
Network-free. Usage: bundle_split_check.py RUN G"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from faab.config import fig

run, g = sys.argv[1], int(sys.argv[2])
rdir = fig(f"refine_{run}")
F = pickle.load(open(os.path.join(rdir, "frames.pkl"), "rb"))[g]
D = pd.read_csv(os.path.join(rdir, f"details_g{g}.tsv"), sep="\t", dtype={"root": str}).fillna("")
keep = D[D.role.str.startswith(("listed R", "listed other", "census", "PR-like")) | D.role.str.contains("not this bundle's R7")]
role = {x.root: (x.assigned or ("ME, not added" if "not this bundle" in x.role else x.role.split(" (")[0]))
        for x in keep.itertuples(index=False)}
tr = pd.read_csv(os.path.join(rdir, f"track_g{g}.tsv"), sep="\t", dtype={"root": str})
P = tr[["x", "y", "z"]].to_numpy(float)
tr["pu"], tr["pv"] = P @ F["u"], P @ F["v"]
tr = tr[tr.root.isin(role)]
pos = {r: t.set_index("s_box")[["pu", "pv"]] for r, t in tr.groupby("root")}
roots = sorted(pos, key=lambda r: role[r])
n = len(roots)
M, S = np.full((n, n), np.nan), np.zeros((n, n), int)
for i in range(n):
    for j in range(i + 1, n):
        sh = pos[roots[i]].index.intersection(pos[roots[j]].index)
        S[i, j] = S[j, i] = len(sh)
        if len(sh) >= 3:
            M[i, j] = M[j, i] = float(np.median(np.hypot(*(pos[roots[i]].loc[sh] - pos[roots[j]].loc[sh]).to_numpy().T)))
print(f"group {g}: {n} photoreceptor-like axons in the track; slices {tr.s_box.nunique()}")
fill = np.where(np.isnan(M), np.nanmax(M) * 1.5, M)
np.fill_diagonal(fill, 0)
Z = linkage(squareform(fill), "average")
lab = fcluster(Z, 2, "maxclust")
order = dendrogram(Z, no_plot=True)["leaves"]
print("pairwise median distance (um; '-' = < 3 shared slices), in dendrogram order:")
print(" " * 26 + " ".join(f"{roots[j][-4:]:>5}" for j in order))
for i in order:
    print(f"{roots[i][-6:]} {role[roots[i]][:12]:12s}      " +
          " ".join("    ." if i == j else "    -" if np.isnan(M[i, j]) else f"{M[i, j]:5.1f}" for j in order))
for k in (1, 2):
    idx = [i for i in range(n) if lab[i] == k]
    w = [M[i, j] for i in idx for j in idx if i < j and not np.isnan(M[i, j])]
    print(f"\nset {k}: {len(idx)} axons; median pairwise distance {np.median(w) if w else float('nan'):.2f} um")
    for i in idx:
        print(f"  {roots[i]}  {role[roots[i]]}")
b = [M[i, j] for i in range(n) for j in range(n) if lab[i] == 1 and lab[j] == 2 and not np.isnan(M[i, j])]
print(f"\nbetween the sets: median pairwise distance {np.median(b) if b else float('nan'):.2f} um ({len(b)} pairs with >= 3 shared slices)")
# per slice: distance between the two sets' centers
cen = tr.assign(k=tr.root.map(dict(zip(roots, lab)))).groupby(["s_box", "k"])[["pu", "pv"]].median().unstack("k").dropna()
if len(cen):
    sep = np.hypot(cen[("pu", 1)] - cen[("pu", 2)], cen[("pv", 1)] - cen[("pv", 2)])
    print(f"center-to-center distance per slice: median {sep.median():.2f} um, range {sep.min():.2f}-{sep.max():.2f} "
          f"um over {len(sep)} slices")
# terminals: median presynaptic site of each axon, across the axis
pre = pd.read_csv(os.path.join(rdir, "presyn.tsv"), sep="\t", dtype={"pre": str})
term = {}
for r, t in pre[pre.pre.isin(roots)].groupby("pre"):
    Q = t[["pre_x", "pre_y", "pre_z"]].to_numpy(float)
    if len(Q) >= 10:
        term[r] = (np.median(Q @ F["u"]), np.median(Q @ F["v"]), np.median((Q - F["c"]) @ F["axis"]), len(Q))
print("\nterminals (median presynaptic site; u, v across the axis, s along it, um):")
for k in (1, 2):
    for r in [roots[i] for i in range(n) if lab[i] == k and roots[i] in term]:
        u, v, s, m = term[r]
        print(f"  set {k} {r}  u {u:7.1f}  v {v:7.1f}  s {s:6.1f}  ({m} sites)  {role[r]}")
for k in (1, 2):
    T = np.array([term[roots[i]][:2] for i in range(n) if lab[i] == k and roots[i] in term])
    if len(T):
        print(f"set {k} terminal spread: center ({T[:, 0].mean():.1f}, {T[:, 1].mean():.1f}); "
              f"median distance to center {np.median(np.hypot(*(T - T.mean(0)).T)):.1f} um")
