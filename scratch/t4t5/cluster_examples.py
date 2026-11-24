"""Example cells for the four partner-defined T4/T5 clusters (David 2026-11-24: "give examples for the four T4 T5
clusters in ng link form or image form for illustration purposes").

Clusters: tmp/t4t5_partner_clusters.tsv (subtype_partner_clusters.py, k = 4 on output-partner fractions), shown in
lobula-plate depth order (deepest on the lobula side first; tentative a-like ... d-like). One lobula-plate spot is
chosen where all eight cluster x type classes have a cell close by (anchor = the cell LOP-output centroid minimizing the
largest nearest-class distance); per class the NPER nearest cells whose branch direction is within 60 deg of that
class's local circular mean (cells with >= MIN_DEN dendrite inputs, high / annotated confidence). Directions are only
comparable inside one spot (local frames).
Writes, in tmp/: <stamp>_t4t5_clusters_all_spec.json (one cell per class, colored by cluster) and
<stamp>_t4t5_cluster<k>_spec.json (NPER T4 + NPER T5 per cluster); figures/t4t5/<stamp>_t4t5_cluster_examples.png
(dendrite skeletons in the local plane, base at the origin, arrow = branch-growth direction, + rose of the spot's
directions per cluster) and <stamp>_t4t5_cluster_examples.tsv. Read-only toward CAVE (one is_latest_roots call,
throttled get_latest_roots for stale ids). Usage: cluster_examples.py [CLUSTERS.tsv]"""
import glob
import json
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from faab.clients.cave import client as cave_client, throttle  # noqa: E402
from faab.config import TMP, fig  # noqa: E402
from faab.partners.sheet import utc_stamp  # noqa: E402

OUT = fig("t4t5")
VOX = np.array([0.016, 0.016, 0.045])
NPER, MIN_DEN, NEAR_NODE, SPOT = 3, 30, 3.0, 40.0
CL_COL = ["#E69F00", "#56B4E9", "#CC79A7", "#009E73"]  # Okabe-Ito, by depth order
TYPE_COL = {"T4": "#0072B2", "T5": "#E69F00"}


def newest(pat):
    return sorted(glob.glob(os.path.join(OUT, pat)))[-1]


def circ_mean(deg):
    a = np.radians(np.asarray(deg, float))
    return float(np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())))


def adiff(a, b):
    return np.abs((np.asarray(a, float) - b + 180) % 360 - 180)


def main():
    stamp = utc_stamp()
    G = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else os.path.join(TMP, "t4t5_partner_clusters.tsv"), sep="\t",
                    dtype={"pre_root": str})
    G = G.rename(columns={"pre_root": "root"})
    out = pd.read_parquet(newest("*_t4t5ld_out.parquet"))
    inp = pd.read_parquet(newest("*_t4t5ld_in.parquet"))
    lop = out[out.region == "LOP_R"].groupby("pre_root")[["x", "y", "z"]].mean()
    lop.index = lop.index.astype(str)
    G = G[G.root.isin(lop.index) & G.bangle.notna() & (G.n_den >= MIN_DEN)].copy()
    G[["lx", "ly", "lz"]] = lop.loc[G.root].to_numpy()
    order = G.groupby("cl").depth.median().sort_values(ascending=False).index.tolist()
    name = {c: f"cluster {c} ({'abcd'[i]}-like, depth {G[G.cl == c].depth.median():.2f})" for i, c in enumerate(order)}
    print("clusters by lobula-plate depth: " + "; ".join(f"{name[c]}: T4 {int(((G.cl == c) & (G.type == 'T4')).sum())},"
                                                          f" T5 {int(((G.cl == c) & (G.type == 'T5')).sum())}" for c in order))

    # ---- the spot: every class has a cell near
    classes = [(c, t) for c in order for t in ("T4", "T5")]
    trees = {k: cKDTree(G[(G.cl == k[0]) & (G.type == k[1])][["lx", "ly", "lz"]].to_numpy()) for k in classes}
    A = G[["lx", "ly", "lz"]].to_numpy()
    worst = np.max([trees[k].query(A)[0] for k in classes], axis=0)
    anchor = A[int(np.argmin(worst))]
    print(f"spot (LOP, um): {np.round(anchor, 1).tolist()}; every class within {worst.min():.1f} um")

    picks, rose = [], {}
    for c, t in classes:
        q = G[(G.cl == c) & (G.type == t)].copy()
        q["dist"] = np.linalg.norm(q[["lx", "ly", "lz"]].to_numpy() - anchor, axis=1)
        local = q[q.dist <= SPOT]
        m = circ_mean(local.bangle) if len(local) >= 3 else circ_mean(q.nsmallest(5, "dist").bangle)
        rose[(c, t)] = local.bangle.to_numpy()
        ok = q[adiff(q.bangle, m) <= 60].nsmallest(NPER, "dist")
        for r in ok.itertuples():
            picks.append(dict(cl=c, cluster=name[c], type=t, root=r.root, dist_um=round(r.dist, 1),
                              bangle=round(r.bangle, 1), local_mean=round(m, 1), depth=round(r.depth, 2),
                              n_den=int(r.n_den), confidence=r.confidence))
    P = pd.DataFrame(picks)

    # ---- live roots for the links (the table's roots are from this morning's gather)
    c = cave_client()
    last = throttle(0)
    ok = np.asarray(c.chunkedgraph.is_latest_roots([int(x) for x in P.root])).astype(bool)
    live = []
    for x, g in zip(P.root, ok):
        if g:
            live.append(x)
            continue
        last = throttle(last)
        live.append(";".join(str(int(y)) for y in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(x)))))
    P["root_now"] = live
    print(f"{len(P)} example cells, {int((~ok).sum())} edited since the gather")
    P.to_csv(os.path.join(OUT, f"{stamp}_t4t5_cluster_examples.tsv"), sep="\t", index=False)

    ids = lambda q: [y for v in q.root_now for y in v.split(";")]  # noqa: E731
    pos = np.round(anchor / VOX, 0).tolist()
    base = {"meshes": [{"kind": "brain"}, {"kind": "OL"}], "layout": "3d", "image": True, "compact": True,
            "position": pos, "scale": 6000}
    first = P.groupby(["cl", "type"], sort=False).head(1)
    neurons = [{"name": f"{name[cl]} {t}", "color": CL_COL[order.index(cl)], "ids": ids(first[(first.cl == cl) &
                                                                                              (first.type == t)])}
               for cl in order for t in ("T4", "T5")]
    specs = {"all": dict(base, neurons=[n for n in neurons if n["ids"]], select=neurons[0]["name"])}
    for cl in order:
        specs[f"cluster{cl}"] = dict(base, neurons=[{"name": f"{name[cl]} {t} ({len(P[(P.cl == cl) & (P.type == t)])})",
                                                      "color": TYPE_COL[t], "ids": ids(P[(P.cl == cl) & (P.type == t)])}
                                                     for t in ("T4", "T5")])
    for k, sp in specs.items():
        sp["out"] = os.path.join(TMP, f"{stamp}_t4t5_{k}_ng.txt")
        json.dump(sp, open(os.path.join(TMP, f"{stamp}_t4t5_{k}_spec.json"), "w"), indent=1)
        print(f"spec {stamp}_t4t5_{k}_spec.json")

    # ---- figure: rose per cluster (spot), then dendrites T4 / T5
    sk = pickle.load(open(fig("t4t5_tab_skels.pkl"), "rb"))
    fig_, ax = plt.subplots(3, 4, figsize=(13, 10), dpi=150)
    for j, cl in enumerate(order):
        a = fig_.add_subplot(3, 4, j + 1, projection="polar")
        ax[0, j].set_axis_off()
        for t, ls in (("T4", "-"), ("T5", "--")):
            v = rose[(cl, t)]
            if len(v):
                h, e = np.histogram(np.radians(v), bins=12, range=(-np.pi, np.pi))
                a.step(np.r_[e[:-1], e[-1]], np.r_[h, h[0]], where="post", color=TYPE_COL[t], lw=1.5, ls=ls,
                       label=f"{t} (n {len(v)})")
        a.set_title(name[cl], fontsize=9, color=CL_COL[j], fontweight="bold")
        a.tick_params(labelsize=7)
        a.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.25, -0.2), frameon=False)
        for i, t in enumerate(("T4", "T5")):
            axp = ax[i + 1, j]
            q = P[(P.cl == cl) & (P.type == t)]
            for k, r in enumerate(q.itertuples()):
                gr = G[G.root == r.root].iloc[0]
                S = sk.get(r.root)
                if S is None:
                    continue
                n = gr[["n_x", "n_y", "n_z"]].to_numpy(float)
                e1 = np.array([0.0, 1.0, 0.0]) - n[1] * n
                e1 /= np.linalg.norm(e1)
                e2 = np.cross(n, e1)
                b = gr[["base_x", "base_y", "base_z"]].to_numpy(float)
                nd = S.nodes
                xyz = nd[["x", "y", "z"]].to_numpy(float) / 1000.0
                reg = "ME_R" if t == "T4" else "LO_R"
                Pd = inp[(inp.post_root == int(r.root)) & (inp.region == reg)][["x", "y", "z"]].to_numpy(float)
                near = cKDTree(Pd).query(xyz)[0] <= NEAR_NODE
                idx = {nid: m for m, nid in enumerate(nd.node_id)}
                shade = 1.0 - 0.25 * k
                for cid, pid in zip(nd.node_id, nd.parent_id):
                    if pid in idx and near[idx[cid]] and near[idx[pid]]:
                        u, w = xyz[idx[cid]] - b, xyz[idx[pid]] - b
                        axp.plot([u @ e1, w @ e1], [u @ e2, w @ e2], color=CL_COL[j], lw=1.2, alpha=shade)
                d = np.array([gr.b1, gr.b2])
                d = 10 * d / np.linalg.norm(d)
                axp.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.4,
                                                                     alpha=shade))
                axp.text(0.02, 0.97 - 0.06 * k, f"{r.root} ({r.bangle:.0f} deg)", transform=axp.transAxes,
                         fontsize=6.5, va="top", color="#333333")
            axp.plot(0, 0, "o", color="#333333", ms=3)
            axp.set_xlim(-22, 22)
            axp.set_ylim(-22, 22)
            axp.set_aspect("equal")
            axp.tick_params(labelsize=7)
            axp.grid(color="#dddddd", lw=0.5)
            if j == 0:
                axp.set_ylabel(f"{t} dendrite ({'medulla' if t == 'T4' else 'lobula'})\num, perpendicular", fontsize=8)
            if i == 1:
                axp.set_xlabel("um, along dataset y (projected)", fontsize=8)
    fig_.suptitle("Aedes T4/T5 output-partner clusters: one lobula-plate spot, nearest typical cells per cluster\n"
                  "top: branch directions of the spot's cells (within 40 um); below: dendrite skeletons in the local "
                  "plane, base at the dot, arrow = branch-growth direction", fontsize=10)
    fig_.tight_layout(rect=(0, 0, 1, 0.95))
    png = os.path.join(OUT, f"{stamp}_t4t5_cluster_examples.png")
    fig_.savefig(png)
    print(f"written {png}")


if __name__ == "__main__":
    main()
