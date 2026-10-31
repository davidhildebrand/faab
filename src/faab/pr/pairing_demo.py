"""Demonstration of the pairing frame on ONE real group (default group 1), network-free.

Repeats the terminal-zone analysis of faab_pr_bundles_pairs.analyse() on the cached skeletons
(figures/pr_partner_skels.pkl, roots of 2026-09-30): bundle axis (PCA of the pooled R1-R6 nodes), lamina side
(from the R1-R6 presynaptic sites saved in figures/pr_groups_out_raw.tsv), terminal zone (last 35 um), mirror
plane search (reflect + NBLAST), mutual pairs, t/d, fly names. Then draws the real node clouds in that frame:
 A  side view face-on to the plane (s along the axis, t in the plane), terminal zone shaded, presynaptic sites
 B  side view edge-on to the plane (s, d): the plane is the horizontal line d = 0
 C  cross-section of the terminal zone (t, d): nodes, per-cell means, the three pairs, R7/R8 positions
 D  the reflection test for one pair: cell A, A reflected through the plane, and its partner B
Output: figures/pr_pairing_demo_group<g>.png and a self-contained figures/pr_pairing_demo_group<g>.html.
Usage: python faab_pr_pairing_demo.py [group]"""
from faab.config import REPO
import base64
import json
import os
import pickle
import sys
import warnings

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.pr.groups_mirror import load_groups, prep, reflect, THETAS, SCACHE, OUT
from faab.config import fig

warnings.filterwarnings("ignore")
navis.set_pbars(hide=True)
HERE = REPO
TMP = os.path.join(HERE, "..", "tmp")
ZONE = 35.0
PAIR_COL = {"R2/R5": "#7F77DD", "R1/R6": "#D85A30", "R3/R4": "#1D9E75"}
R7_COL, R8_COL = "#A32D2D", "#0F6E56"


def presyn_centroid(group):
    """lamina reference from the saved synapse table (roots of 2026-10-10 mapped through the pasted order)"""
    pasted = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    roots = json.load(open(fig("pr_footprint_roots.json")))["root_used"]
    sel = {int(r) for r, g, c in zip(roots, pasted.group, pasted["class"]) if int(g) == group and c.startswith("R1")}
    syn = pd.read_csv(fig("pr_groups_out_raw.tsv"), sep="\t")
    P = syn[syn.pre.isin(sel)][["pre_x", "pre_y", "pre_z"]].to_numpy(float)
    return P


def analyse_group(g, verbose=True):
    """frame + pairing of one group on the cached skeletons; returns a dict with everything the figures need"""
    import builtins
    _print = print if verbose else (lambda *a, **k: None)
    cells = load_groups()[g]
    sk = pickle.load(open(SCACHE, "rb"))
    outer = [r for r, c in cells if c.startswith("R1")]
    inner = {r: c for r, c in cells if not c.startswith("R1")}
    neu = {r: prep(sk[int(r)]) for r, _ in cells}
    pts = np.vstack([neu[r].nodes[["x", "y", "z"]].values for r in outer])
    c = pts.mean(0)
    axis = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    P = presyn_centroid(g)
    sign = 1.0 if np.median((P - c) @ axis) > 0 else -1.0
    axis = axis * sign  # +s = lamina side from here on
    s_end = max(((neu[r].nodes[["x", "y", "z"]].values - c) @ axis).max() for r in outer)
    u = np.cross(axis, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(axis, u)

    def zone(n, lamina_only=False):
        s = (n.nodes[["x", "y", "z"]].values - c) @ axis
        m = (s > s_end - ZONE) & ((s <= s_end + 5) if lamina_only else True)
        keep = n.nodes.node_id.values[m]
        return navis.subset_neuron(n, keep, inplace=False) if len(keep) >= 3 else None
    term = {r: zone(neu[r]) for r in outer}
    for r in inner:
        t78 = zone(neu[r], lamina_only=True)
        if t78 is not None:
            term[r] = t78
    tpts = np.vstack([term[r].nodes[["x", "y", "z"]].values for r in outer])
    ct = tpts.mean(0)
    dps = navis.NeuronList([navis.make_dotprops(term[r], k=5) for r in outer])
    best = None
    for th in THETAS:
        nrm = np.cos(np.radians(th)) * u + np.sin(np.radians(th)) * v
        refl = navis.NeuronList([navis.make_dotprops(reflect(term[r], ct, nrm), k=5) for r in outer])
        S = navis.nblast(refl, dps, scores="mean", normalized=True, n_cores=1, progress=False).values
        np.fill_diagonal(S, np.nan)
        score = np.nanmax(S, axis=1).mean()
        if best is None or score > best[0]:
            best = (score, th, S, nrm)
    score, th, S, nrm = best
    that = np.cross(axis, nrm); that /= np.linalg.norm(that)
    print(f"group {g}: plane theta = {th} deg, mean best match {score:.2f}, s_end = {s_end:.1f} um", flush=True)

    def frame(n, origin):
        p = n.nodes[["x", "y", "z"]].values - origin
        return p @ axis, p @ that, p @ nrm  # s, t, d
    pos = {r: tuple(float(x.mean()) for x in frame(term[r], ct)[1:]) for r in term}
    bm = {i: int(np.nanargmax(S[i])) for i in range(len(outer))}
    pairs = sorted({tuple(sorted((i, j))) for i, j in bm.items() if bm.get(j) == i})
    left = [i for i in range(len(outer)) if not any(i in p for p in pairs)]
    if len(left) == 2:
        pairs.append((left[0], left[1]))
    names = {}
    r7 = [r for r, k in inner.items() if k == "R7" and r in term]
    t7 = np.mean([pos[r][0] for r in r7]) if r7 else np.nan
    if len(pairs) == 3:
        dmean = {p: (abs(pos[outer[p[0]]][1]) + abs(pos[outer[p[1]]][1])) / 2 for p in pairs}
        lat = max(pairs, key=lambda p: dmean[p])
        ends = [p for p in pairs if p != lat]
        tmean = {p: (pos[outer[p[0]]][0] + pos[outer[p[1]]][0]) / 2 for p in ends}
        near = min(ends, key=lambda p: abs(tmean[p] - t7))
        for k in lat:
            names[k] = "R2/R5"
        for k in near:
            names[k] = "R1/R6"
        for k in [p for p in ends if p != near][0]:
            names[k] = "R3/R4"
    for i, r in enumerate(outer):
        pr = next(p for p in pairs if i in p)
        j = pr[1] if pr[0] == i else pr[0]
        print(f"  {r}: t {pos[r][0]:+.1f}, d {pos[r][1]:+.1f}, partner {outer[j]} (S {S[i, j]:.2f}), {names.get(i, '')}")
    for r in inner:
        if r in pos:
            print(f"  {inner[r]} {r}: t {pos[r][0]:+.1f}, d {pos[r][1]:+.1f} (lamina-zone nodes)")

    return dict(g=g, cells=cells, outer=outer, inner=inner, neu=neu, term=term, pos=pos, pairs=pairs, names=names, S=S,
                th=th, score=score, s_end=s_end, P=P, c=c, ct=ct, axis=axis, that=that, nrm=nrm, frame=frame)


def main():
    g = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    R = analyse_group(g)
    cells, outer, inner, neu, term, pos, pairs, names, S = (R[k] for k in
                                                           ("cells", "outer", "inner", "neu", "term", "pos", "pairs", "names", "S"))
    th, score, s_end, P, c, ct, frame = (R[k] for k in ("th", "score", "s_end", "P", "c", "ct", "frame"))
    # ---------------- figure ----------------
    col = {outer[i]: PAIR_COL.get(names.get(i, ""), "#888780") for i in range(len(outer))}
    mk = {}
    for p in pairs:
        mk[outer[p[0]]], mk[outer[p[1]]] = "o", "s"
    tag = {r: "abcdef"[i] for i, r in enumerate(outer)}
    fig, ax = plt.subplots(2, 2, figsize=(16, 13), constrained_layout=True)
    (A, B), (C, D) = ax
    s_pre = (P - c) @ axis
    rng = np.random.default_rng(0)
    sub = rng.choice(len(P), size=min(250, len(P)), replace=False)
    A.scatter(s_pre[sub], ((P - c) @ that)[sub], s=10, color="#BA7517", marker="x", alpha=0.35, zorder=1,
              label=f"R1-R6 presynaptic sites ({len(P)}, 250 drawn)")
    B.scatter(s_pre[sub], ((P - c) @ nrm)[sub], s=10, color="#BA7517", marker="x", alpha=0.35, zorder=1)
    for r in outer:
        s, t, d = frame(neu[r], c)
        A.scatter(s, t, s=3, color=col[r], alpha=0.8, zorder=3, label=f"{tag[r]}: {names.get(outer.index(r), '')} ...{r[-6:]}")
        B.scatter(s, d, s=3, color=col[r], alpha=0.8, zorder=3)
    for r, k in inner.items():
        s, t, d = frame(neu[r], c)
        cc = R7_COL if k == "R7" else R8_COL
        A.scatter(s, t, s=3, color=cc, alpha=0.8, zorder=2, label=f"{k} ...{r[-6:]}")
        B.scatter(s, d, s=3, color=cc, alpha=0.8, zorder=2)
    for axx, yl in ((A, "t (um): position along the plane"), (B, "d (um): distance from the plane")):
        axx.axvspan(s_end - ZONE, s_end, color="#EF9F27", alpha=0.15, zorder=0,
                    label="terminal zone (last 35 um)" if axx is A else None)
        axx.axvline(s_end, color="#BA7517", lw=0.8, ls="--")
        axx.set_xlabel("s (um) along the bundle axis, lamina to the right")
        axx.set_ylabel(yl)
    B.axhline(0, color="#378ADD", lw=2, label=f"mirror plane seen edge-on (theta = {th} deg)")
    A.set_title(f"A  group {g} skeletons seen face-on to the mirror plane (retina at left, R7 bends off to the medulla)")
    B.set_title("B  the same skeletons seen edge-on: the plane is the blue line, d is the height above it")
    A.legend(fontsize=8, loc="upper left", markerscale=3)
    B.legend(fontsize=8, loc="upper left")

    for r in outer:
        s, t, d = frame(term[r], ct)
        C.scatter(t, d, s=5, color=col[r], alpha=0.3, zorder=1)
    for p in pairs:
        a, b = pos[outer[p[0]]], pos[outer[p[1]]]
        C.plot([a[0], b[0]], [a[1], b[1]], ls="--", color=col[outer[p[0]]], lw=1.5, zorder=4)
    for r in outer:
        C.scatter(*pos[r], s=260, color=col[r], marker=mk[r], edgecolor="k", zorder=5)
        C.text(pos[r][0], pos[r][1], tag[r], ha="center", va="center", fontsize=10, fontweight="bold", color="w", zorder=6)
    for r, k in inner.items():
        if r in pos:
            cc = R7_COL if k == "R7" else R8_COL
            C.scatter(*pos[r], s=260, marker="D", color=cc, edgecolor="k", zorder=6)
            C.text(pos[r][0], pos[r][1], k[1], ha="center", va="center", fontsize=9, fontweight="bold", color="w", zorder=7)
    C.axhline(0, color="#378ADD", lw=2, zorder=2)
    C.axvline(0, color="#888780", lw=0.6, zorder=2)
    lines = [f"{tag[r]}  {names.get(outer.index(r), ''):5s} ...{r[-6:]}  t {pos[r][0]:+5.1f}  d {pos[r][1]:+5.1f}" for r in outer]
    lines += [f"{k}  (lamina-zone nodes)        t {pos[r][0]:+5.1f}  d {pos[r][1]:+5.1f}" for r, k in inner.items() if r in pos]
    lines += ["", "pairs = mutual best mirror match; each straddles the plane",
              "largest mean |d| -> R2/R5; nearer the R7 in t -> R1/R6; other -> R3/R4"]
    C.text(0.02, 0.98, "\n".join(lines), transform=C.transAxes, va="top", fontsize=8.5, family="monospace",
           bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.9), zorder=8)
    C.set_xlabel("t (um) along the plane")
    C.set_ylabel("d (um) from the plane")
    C.set_aspect("equal")
    C.set_title("C  looking down the axis at the terminal zone: nodes (faint), cell means (big), mutual pairs (dashed)")

    demo = max(pairs, key=lambda p: S[p[0], p[1]])
    ra, rb = outer[demo[0]], outer[demo[1]]
    sa, ta, da = frame(term[ra], ct)
    sb, tb, db = frame(term[rb], ct)
    D.scatter(ta, da, s=14, color=col[ra], alpha=0.6, label=f"cell {tag[ra]} ...{ra[-6:]}")
    D.scatter(ta, -da, s=28, color=col[ra], alpha=0.8, marker="x", label=f"cell {tag[ra]} reflected through the plane (d -> -d)")
    D.scatter(tb, db, s=14, color="k", alpha=0.45, label=f"its partner {tag[rb]} ...{rb[-6:]}   NBLAST(reflected {tag[ra]}, {tag[rb]}) = {S[demo[0], demo[1]]:.2f}")
    D.axhline(0, color="#378ADD", lw=2)
    D.set_xlabel("t (um)")
    D.set_ylabel("d (um)")
    D.set_title(f"D  the mirror test for the {names.get(demo[0], '')} pair: reflected {tag[ra]} should overlay {tag[rb]}")
    D.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Group {g}: the pairing frame on the real skeletons (roots of 2026-09-30; lamina side from the presynaptic sites)",
                 fontsize=13)
    png = fig(f"pr_pairing_demo_group{g}.png")
    fig.savefig(png, dpi=130)
    b64 = base64.b64encode(open(png, "rb").read()).decode()
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>Pairing demo group {g}</title></head><body "
            f"style='font-family:Helvetica,Arial,sans-serif;max-width:1500px;margin:20px auto'>"
            f"<h2>Group {g}: the pairing frame on the real skeletons</h2>"
            f"<p>Plane angle {th} deg, mean best mirror match {score:.2f}. s is the coordinate along the bundle axis "
            f"(lamina side from {len(P)} R1-R6 presynaptic sites), the terminal zone is the last {ZONE:.0f} um, "
            f"t and d are the means of each cell's terminal-zone nodes in the plane frame.</p>"
            f"<img style='width:100%' src='data:image/png;base64,{b64}'></body></html>")
    open(fig(f"pr_pairing_demo_group{g}.html"), "w").write(html)
    print("saved", png)


if __name__ == "__main__":
    main()
