"""Flattened-bundle view of one PR group (David's test, 2026-10-13): along the bundle axis, in 2-um slices, every
axon's position is taken relative to the mean position of all axons present in that slice, which removes the common
bends of the bundle. Shows (A) each axon's angle around the bundle centre vs depth (is the cyclic order constant? where
do R7/R8 sit?), (B) its distance from the centre vs depth, (C) flattened cross-sections at several depths with the
cells named from the terminal analysis, (D) the mirror-pair test in the flattened frame: for each pair the mean of the
two partners' offsets vs depth (a true mirror pair about a plane through the centre keeps that mean on the plane).
Network-free (cached skeletons, roots of 2026-09-30). Usage: python faab_pr_bundle_flatten.py [group]
Names for group 1 come from faab_pr_pairing_demo.py (2026-10-13); other groups from figures/faab_pr_groups_pairs.tsv."""
import os
import sys
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.pr.groups_mirror import load_groups, prep, SCACHE, OUT
from faab.pr.pairing_demo import presyn_centroid

warnings.filterwarnings("ignore")
PAIR_COL = {"R2/R5": "#7F77DD", "R1/R6": "#D85A30", "R3/R4": "#1D9E75", "R7": "#A32D2D", "R8": "#0F6E56"}
STEP = 2.0


def main():
    import pickle
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    cells = load_groups()[g]
    sk = pickle.load(open(SCACHE, "rb"))
    names = {}
    pairs = pd.read_csv(os.path.join(OUT, "faab_pr_groups_pairs.tsv"), sep="\t", dtype=str)
    for r in pairs[pairs.group == str(g)].itertuples():
        names[r.root_used] = r.fly_pair
    for r, c in cells:
        names.setdefault(r, c)
    neu = {r: prep(sk[int(r)]) for r, _ in cells}
    outer = [r for r, c in cells if c.startswith("R1")]
    pts = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in outer])
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    P = presyn_centroid(g)
    if np.median((P - c) @ axis) < 0:
        axis = -axis
    u = np.cross(axis, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(axis, u)
    s_end = max(((neu[r].nodes[["x", "y", "z"]].values - c) @ axis).max() for r in outer)
    s_lo = min(((neu[r].nodes[["x", "y", "z"]].values - c) @ axis).min() for r in outer)
    edges = np.arange(s_lo, s_end + STEP, STEP)
    mids = edges[:-1] + STEP / 2
    pos = {}  # root -> (n_slices, 2) mean (u, v) per slice, nan where absent
    for r, _ in cells:
        p = neu[r].nodes[["x", "y", "z"]].values - c
        s = p @ axis
        uv = np.c_[p @ u, p @ v]
        m = np.full((len(mids), 2), np.nan)
        for k in range(len(mids)):
            sel = (s >= edges[k]) & (s < edges[k + 1])
            if sel.sum():
                m[k] = uv[sel].mean(0)
        pos[r] = m
    allp = np.stack([pos[r] for r, _ in cells])            # cells x slices x 2
    n_here = (~np.isnan(allp[:, :, 0])).sum(0)
    centre = np.nanmean(allp, axis=0)
    good = n_here == len(cells)                              # slices where all eight axons are present
    off = allp - centre[None]
    ang = np.degrees(np.arctan2(off[:, :, 1], off[:, :, 0]))
    rad = np.hypot(off[:, :, 0], off[:, :, 1])
    print(f"group {g}: {good.sum()} of {len(mids)} slices hold all {len(cells)} axons (s {mids[good].min():.0f} to {mids[good].max():.0f} um)")

    fig, ax = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    (A, B), (C, D) = ax
    for i, (r, cl) in enumerate(cells):
        lab = names.get(r, cl)
        col = PAIR_COL.get(lab, "#888780")
        ls = "-" if r == [x for x, _ in cells if names.get(x, "") == lab][0] else "--"
        a = np.where(good, ang[i], np.nan)
        A.plot(mids, a, ls, color=col, lw=1.4, label=f"{lab} ...{r[-6:]}")
        B.plot(mids, np.where(good, rad[i], np.nan), ls, color=col, lw=1.4)
    for axx in (A, B):
        axx.axvspan(s_end - 35, s_end, color="#EF9F27", alpha=0.15)
        axx.set_xlabel("s (um) along the bundle axis, lamina to the right (shaded: terminal zone)")
    A.set_ylabel("angle around the bundle centre (deg)")
    A.set_title(f"A  group {g}: each axon's angle around the per-slice centre of all {len(cells)} axons")
    A.legend(fontsize=8, ncol=2)
    B.set_ylabel("distance from the bundle centre (um)")
    B.set_title("B  distance from the centre: who runs in the middle?")

    depths = [q for q in np.percentile(mids[good], [5, 30, 55, 80]) ] + [s_end - 20]
    for q, mk in zip(depths, ["o", "s", "^", "D", "*"]):
        k = int(np.argmin(np.abs(mids - q)))
        for i, (r, cl) in enumerate(cells):
            lab = names.get(r, cl)
            C.scatter(off[i, k, 0], off[i, k, 1], s=70, marker=mk, color=PAIR_COL.get(lab, "#888780"),
                      edgecolor="k", lw=0.5, alpha=0.85)
            if mk == "o":
                C.annotate(lab, (off[i, k, 0], off[i, k, 1]), xytext=(5, 5), textcoords="offset points", fontsize=8)
        C.scatter([], [], marker=mk, color="k", label=f"s = {mids[k]:.0f} um")
    C.axhline(0, color="#aaa", lw=0.6); C.axvline(0, color="#aaa", lw=0.6)
    C.set_aspect("equal")
    C.set_xlabel("u (um) offset from the bundle centre"); C.set_ylabel("v (um)")
    C.set_title("C  flattened cross-sections at five depths (global u, v frame; labels at the shallowest)")
    C.legend(fontsize=8)

    for lab in ("R1/R6", "R2/R5", "R3/R4"):
        idx = [i for i, (r, cl) in enumerate(cells) if names.get(r, "") == lab]
        if len(idx) == 2:
            mid = (off[idx[0]] + off[idx[1]]) / 2
            D.plot(mids, np.where(good, np.hypot(mid[:, 0], mid[:, 1]), np.nan), color=PAIR_COL[lab], lw=1.6, label=f"{lab} pair midpoint")
            D.plot(mids, np.where(good, np.hypot(*(off[idx[0]] - off[idx[1]]).T) / 2, np.nan), color=PAIR_COL[lab], lw=1, ls=":",
                   label=f"{lab} half-separation")
    D.axvspan(s_end - 35, s_end, color="#EF9F27", alpha=0.15)
    D.set_xlabel("s (um)"); D.set_ylabel("um")
    D.set_title("D  mirror pairs in the flattened frame: midpoint distance from the centre (solid) and half-separation (dotted)")
    D.legend(fontsize=8)
    fig.suptitle(f"Group {g}: bundle flattened slice by slice (offsets from the mean of all axons in each {STEP:.0f}-um slice)")
    png = os.path.join(OUT, f"pr_bundle_flatten_group{g}.png")
    fig.savefig(png, dpi=120)
    print("saved", png)
    # numbers: cyclic order at a few depths
    for q in depths:
        k = int(np.argmin(np.abs(mids - q)))
        order = sorted(range(len(cells)), key=lambda i: ang[i, k])
        print(f"  s = {mids[k]:5.0f} um: order by angle = " + " > ".join(names.get(cells[i][0], cells[i][1]) for i in order)
              + "; radii " + ", ".join(f"{names.get(cells[i][0], cells[i][1])} {rad[i, k]:.1f}" for i in order))


if __name__ == "__main__":
    main()
