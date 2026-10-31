"""Axon direction of truncated R1-6 pieces from their axon stub (network-free, cached L2 skeletons).

Per cell: node nearest the skeleton centroid (inside the terminal arbor) -> the leaf with the largest path distance
from it = end of the axon stub; direction = stub end - arbor node (and the stub's last 15 um of path, PCA). Pooled
per cartridge; validated on cartridge 8, whose cells are long (their stub end lies in the retina-side axon)."""
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
sk = pickle.load(open(fig("pr_left_skels.pkl"), "rb"))
ax_k = {r.cartridge: np.array([r.ax_x, r.ax_y, r.ax_z]) for r in known.itertuples()}


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip(a @ b / np.linalg.norm(a) / np.linalg.norm(b), -1, 1))))


for r in known.itertuples():
    dirs = []
    for rid in r.r16_ids.split(","):
        s = sk[int(rid)]
        nd = s.nodes.set_index("node_id")
        X = nd[["x", "y", "z"]].values / 1000.0
        ids = nd.index.values
        G = nx.Graph()
        for a, b in s.nodes[["node_id", "parent_id"]].values:
            if b >= 0:
                G.add_edge(int(a), int(b), weight=float(np.linalg.norm(X[ids == a][0] - X[ids == b][0])))
        c = int(ids[np.argmin(np.linalg.norm(X - X.mean(0), axis=1))])
        dist = nx.single_source_dijkstra_path_length(G, c, weight="weight")
        leaves = [n for n in G.nodes if G.degree(n) == 1]
        end = max(leaves, key=lambda n: dist.get(n, -1))
        path = nx.shortest_path(G, c, end, weight="weight")
        P = np.array([X[ids == n][0] for n in path])
        along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        tail = P[along >= along[-1] - 15.0]
        v = P[-1] - P[0]
        t_ax = np.linalg.svd(tail - tail.mean(0), full_matrices=False)[2][0] if len(tail) >= 3 else v
        t_ax = t_ax if t_ax @ v > 0 else -t_ax
        dirs.append(v / np.linalg.norm(v))
        print(f"{r.cartridge} {rid}: stub end {dist[end]:.0f} um of path from the arbor, straight {np.linalg.norm(v):.0f} um; "
              f"arbor->end vs cartridge axis (retina->lamina, so expect ~180) {ang(v, ax_k[r.cartridge]):.0f} deg; "
              f"last-15-um direction vs arbor->end {ang(t_ax, v):.0f} deg")
    D = np.array(dirs)
    m = D.mean(0)
    m /= np.linalg.norm(m)
    print(f"  {r.cartridge} pooled stub direction (lamina -> retina) {m.round(3).tolist()}; spread "
          f"{np.round([ang(d, m) for d in D], 0).tolist()} deg; vs -axis of L8 {ang(m, -ax_k['L8']):.0f} deg, vs -{r.cartridge} "
          f"axis in the site table {ang(m, -ax_k[r.cartridge]):.0f} deg")
