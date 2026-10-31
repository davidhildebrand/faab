"""Where do the truncated left-side PR pieces end? (network-free: cached skeletons, figures/pr_left_helpers.tsv)

Per piece near cartridges 1/2 (lamina point within 40 um): end of the axon stub = the leaf with the largest path
distance from the node nearest the arbor's skeleton centroid. If the stub ends share one section (z) or one plane, the
truncation is a data boundary, not unproofread cells. Compared with the R1-6 of cartridge 8 (long axons)."""
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import networkx as nx
import numpy as np
import pandas as pd

from faab.config import FIGURES
from faab.config import fig

known = pd.concat([pd.read_csv(fig(f), sep="\t", dtype={"r16_ids": str, "r78_ids": str})
                   for f in ("pr_left_lattice_known.tsv", "pr_left_lattice_noaxis.tsv")]).sort_values("cartridge").reset_index(drop=True)
h = pd.read_csv(fig("pr_left_helpers.tsv"), sep="\t")
sk = pickle.load(open(fig("pr_left_skels.pkl"), "rb"))
lam12 = known[known.cartridge.isin(["L1", "L2"])][["lam_x", "lam_y", "lam_z"]].values.mean(0)


def stub_end(s):
    X = s.nodes[["x", "y", "z"]].values / 1000.0
    ids = s.nodes.node_id.values
    pos = {int(i): k for k, i in enumerate(ids)}
    G = nx.Graph()
    for a, b in s.nodes[["node_id", "parent_id"]].values:
        if b >= 0:
            G.add_edge(int(a), int(b), weight=float(np.linalg.norm(X[pos[int(a)]] - X[pos[int(b)]])))
    c = int(ids[np.argmin(np.linalg.norm(X - X.mean(0), axis=1))])
    dist = nx.single_source_dijkstra_path_length(G, c, weight="weight")
    end = max((n for n in G.nodes if G.degree(n) == 1), key=lambda n: dist.get(n, -1))
    return X[pos[end]], X.min(0), X.max(0)


near = h[np.linalg.norm(h[["lam_x", "lam_y", "lam_z"]].values - lam12, axis=1) < 40]
E, lo, hi = [], [], []
for r in near.itertuples():
    e, a, b = stub_end(sk[int(r.root)])
    E.append(e)
    lo.append(a)
    hi.append(b)
E, lo, hi = np.array(E), np.array(lo), np.array(hi)
print(f"pieces within 40 um of cartridges 1/2: {len(near)} (length p50 {near.length_um.median():.0f} um)")
print(f"lamina points 1/2 mean (um) {lam12.round(1).tolist()} -> z section {lam12[2] / 0.045:.0f}")
for k, a in enumerate("xyz"):
    vox = 0.045 if a == "z" else 0.016
    print(f"  stub-end {a}: p5/25/50/75/95 {np.percentile(E[:, k], [5, 25, 50, 75, 95]).round(1).tolist()} um; "
          f"skeleton {a}-min p5/50/95 {np.percentile(lo[:, k], [5, 50, 95]).round(1).tolist()}, "
          f"{a}-max p5/50/95 {np.percentile(hi[:, k], [5, 50, 95]).round(1).tolist()}")
c = E - E.mean(0)
w, U = np.linalg.eigh(np.cov(c.T))
print(f"stub-end cloud: sd along principal axes {np.sqrt(w).round(1).tolist()} um; thinnest axis {U[:, 0].round(2).tolist()}")
zs = np.round(hi[:, 2] / 0.045).astype(int)
vals, cnt = np.unique(zs, return_counts=True)
top = np.argsort(-cnt)[:8]
print("most common skeleton z-max sections (section: pieces): " + ", ".join(f"{vals[i]}: {cnt[i]}" for i in top))
zs = np.round(lo[:, 2] / 0.045).astype(int)
vals, cnt = np.unique(zs, return_counts=True)
top = np.argsort(-cnt)[:8]
print("most common skeleton z-min sections (section: pieces): " + ", ".join(f"{vals[i]}: {cnt[i]}" for i in top))
