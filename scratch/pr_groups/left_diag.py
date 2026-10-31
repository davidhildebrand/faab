"""Diagnostics for left_sites.py (network-free): skeleton extent of each left-side SeaTable R1-6 / R7-8 cell along the
cartridge 8 axis, and the LA_L sheet normal at growing radii vs the plane through the three lamina points."""
import os
import pickle
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from faab import ol_meshes
from faab.config import FIGURES
from faab.config import fig

known = pd.concat([pd.read_csv(fig(f), sep="\t", dtype={"r16_ids": str, "r78_ids": str})
                   for f in ("pr_left_lattice_known.tsv", "pr_left_lattice_noaxis.tsv")]).sort_values("cartridge").reset_index(drop=True)
sk = pickle.load(open(fig("pr_left_skels.pkl"), "rb"))
k8 = known.set_index("cartridge").loc["L8"]
a8 = np.array([k8.ax_x, k8.ax_y, k8.ax_z])
print("per R1-6 skeleton: nodes, cable (um), extent along the L8 axis (um), max pairwise-extent proxy (bbox diagonal, um)")
for r in known.itertuples():
    lam = np.array([r.lam_x, r.lam_y, r.lam_z])
    for rid in r.r16_ids.split(","):
        s = sk.get(int(rid))
        if s is None:
            print(f"  {r.cartridge} {rid}: no skeleton")
            continue
        X = s.nodes[["x", "y", "z"]].values / 1000.0
        t = (X - lam) @ a8
        print(f"  {r.cartridge} {rid}: {len(X)} nodes, cable {float(s.cable_length) / 1000.0:.0f}, along L8 axis "
              f"{t.min():.0f}..{t.max():.0f}, bbox {np.linalg.norm(X.max(0) - X.min(0)):.0f}")

L = known[["lam_x", "lam_y", "lam_z"]].values
n3 = np.cross(L[1] - L[0], L[2] - L[0])
n3 /= np.linalg.norm(n3)
print(f"\nplane through L1, L2, L8: normal {n3.round(3).tolist()}; angle to L8 axis {np.degrees(np.arccos(abs(n3 @ a8))):.1f} deg")
V = ol_meshes.mesh("LA_L")[0] / 1000.0
for r in known.itertuples():
    lam = np.array([r.lam_x, r.lam_y, r.lam_z])
    out = []
    for rad in (40, 60, 80, 100, 150):
        near = V[np.linalg.norm(V - lam, axis=1) < rad]
        w, U = np.linalg.eigh(np.cov((near - near.mean(0)).T))
        n = U[:, 0]
        out.append(f"r{rad}: thin {np.sqrt(w[0] / w[1]):.2f}, to L8 axis {np.degrees(np.arccos(abs(n @ a8))):.0f}, "
                   f"to 3-pt plane {np.degrees(np.arccos(abs(n @ n3))):.0f}")
    print(f"{r.cartridge}: " + " | ".join(out))
