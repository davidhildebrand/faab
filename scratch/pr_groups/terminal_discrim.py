"""Can lamina terminals tell which ommatidium an R1-R6 axon belongs to? (David 2026-11-01.) On the clean groups of a
groups_refine.py run (6 outer + R7 + R8, no extras, every outer axon listed), each outer axon's terminal = the median
of its presynaptic sites (presyn.tsv, um); a group's terminal center = the mean of its six terminals. Per axon: the
distance to its own group's center without itself (leave-one-out) vs to the nearest other clean group's center, and
whether the nearest center is its own. For scale: the distance between neighboring groups' centers, and the same test
on bundle positions (each axon's median position in the tracked slices vs the group's bundle center there).
Network-free. Usage: terminal_discrim.py RUN"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

from faab.config import fig

run = sys.argv[1]
rdir = fig(f"refine_{run}")
S = pd.read_csv(os.path.join(rdir, "summary.tsv"), sep="\t").fillna("")
pre = pd.read_csv(os.path.join(rdir, "presyn.tsv"), sep="\t", dtype={"pre": str})
term = {r: np.median(t[["pre_x", "pre_y", "pre_z"]].to_numpy(float), axis=0) for r, t in pre.groupby("pre") if len(t) >= 10}
outer = {}
for x in S.itertuples(index=False):
    if x.status != "ok" or float(x.outer) != 6 or float(x.extras) != 0 or x.R7 in ("", "-") or x.R8 in ("", "-"):
        continue
    D = pd.read_csv(os.path.join(rdir, f"details_g{x.group}.tsv"), sep="\t", dtype={"root": str}).fillna("")
    o = D[(D.assigned == "outer") & D.role.str.startswith("listed")].root.tolist()
    if len(o) == 6 and all(r in term for r in o):
        outer[int(x.group)] = o
cen = {g: np.mean([term[r] for r in o], axis=0) for g, o in outer.items()}
print(f"{len(outer)} clean groups (6 listed outer axons with >= 10 presynaptic sites each)")
G = sorted(cen)
C = np.array([cen[g] for g in G])
nn = [np.sort(np.linalg.norm(C - C[i], axis=1))[1] for i in range(len(G))]
print(f"terminal centers: nearest other group's center median {np.median(nn):.1f} um (p10-p90 "
      f"{np.percentile(nn, 10):.1f}-{np.percentile(nn, 90):.1f})")
own_d, oth_d, hit = [], [], 0
for g in G:
    for r in outer[g]:
        loo = np.mean([term[q] for q in outer[g] if q != r], axis=0)
        d_own = float(np.linalg.norm(term[r] - loo))
        d_oth = min(float(np.linalg.norm(term[r] - cen[h])) for h in G if h != g)
        own_d.append(d_own)
        oth_d.append(d_oth)
        hit += d_own < d_oth
n = len(own_d)
print(f"terminals: {n} outer axons; distance to own group's center (leave-one-out) median {np.median(own_d):.1f} um "
      f"(p90 {np.percentile(own_d, 90):.1f}); to the nearest other group's center median {np.median(oth_d):.1f} um; "
      f"own center nearest for {hit}/{n} ({100 * hit / n:.0f}%)")

# the same test on the bundle: every axon's median position across the axis in its group's tracked slices
frames = pickle.load(open(os.path.join(rdir, "frames.pkl"), "rb"))
b_own, b_oth, b_hit, b_n = [], [], 0, 0
for g in G:
    F = frames[g]
    tr = pd.read_csv(os.path.join(rdir, f"track_g{g}.tsv"), sep="\t", dtype={"root": str})
    P = tr[["x", "y", "z"]].to_numpy(float)
    tr["pu"], tr["pv"] = P @ F["u"], P @ F["v"]
    others = {h: [r for r in outer[h]] for h in G if h != g}
    for r in outer[g]:
        mine = tr[tr.root == r].set_index("s_box")[["pu", "pv"]]
        rest = tr[tr.root.isin([q for q in outer[g] if q != r])].groupby("s_box")[["pu", "pv"]].mean()
        sh = mine.index.intersection(rest.index)
        if len(sh) < 5:
            continue
        d_own = float(np.median(np.hypot(*(mine.loc[sh] - rest.loc[sh]).to_numpy().T)))
        # other clean groups' axons that show up in this group's track
        d_oth = []
        for h, o in others.items():
            oc = tr[tr.root.isin(o)].groupby("s_box")[["pu", "pv"]].mean()
            s2 = mine.index.intersection(oc.index)
            if len(s2) >= 5:
                d_oth.append(float(np.median(np.hypot(*(mine.loc[s2] - oc.loc[s2]).to_numpy().T))))
        b_own.append(d_own)
        b_n += 1
        if d_oth:
            b_oth.append(min(d_oth))
            b_hit += d_own < min(d_oth)
        else:
            b_hit += 1
print(f"bundle: {b_n} outer axons; distance to own bundle center (leave-one-out) median {np.median(b_own):.1f} um "
      f"(p90 {np.percentile(b_own, 90):.1f}); a neighbor clean group was in the track for {len(b_oth)}, its center "
      f"median {np.median(b_oth) if b_oth else float('nan'):.1f} um away; own center nearest for {b_hit}/{b_n}")
