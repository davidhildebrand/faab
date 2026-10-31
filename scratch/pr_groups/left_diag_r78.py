"""Do the left-side SeaTable R7-8 cells keep their retina-side axon? L2 skeletons (cached with the R1-6 ones in
figures/pr_left_skels.pkl, one throttled fetch), extent along the cartridge 8 axis, and the PCA axis of each cell's
retina-side part (nodes > 20 um retina-ward of its cartridge's lamina point) vs the L8 axis."""
import os
import pickle
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from fafbseg import flywire

from faab.clients.cave import DATASTACK, throttle
from faab.config import FIGURES
from faab.config import fig

SCACHE = fig("pr_left_skels.pkl")
known = pd.concat([pd.read_csv(fig(f), sep="\t", dtype={"r16_ids": str, "r78_ids": str})
                   for f in ("pr_left_lattice_known.tsv", "pr_left_lattice_noaxis.tsv")]).sort_values("cartridge").reset_index(drop=True)
sk = pickle.load(open(SCACHE, "rb"))
need = sorted({int(x) for v in known.r78_ids for x in v.split(",")} - set(sk))
t = time.time()
for r in need:  # one root at a time, >= 5 s apart
    t = throttle(t)
    for s in flywire.get_l2_skeleton([r], dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False):
        sk[int(s.id)] = s
if need:
    pickle.dump(sk, open(SCACHE, "wb"))
k8 = known.set_index("cartridge").loc["L8"]
a8 = np.array([k8.ax_x, k8.ax_y, k8.ax_z])
for r in known.itertuples():
    lam = np.array([r.lam_x, r.lam_y, r.lam_z])
    for rid in r.r78_ids.split(","):
        s = sk.get(int(rid))
        if s is None:
            print(f"{r.cartridge} {rid}: no skeleton")
            continue
        X = s.nodes[["x", "y", "z"]].values / 1000.0
        t = (X - lam) @ a8
        up = X[t < -20]
        msg = ""
        if len(up) >= 5:
            ax = np.linalg.svd(up - up.mean(0), full_matrices=False)[2][0]
            msg = f"; retina-side part {len(up)} nodes, its axis {np.degrees(np.arccos(abs(ax @ a8))):.0f} deg from the L8 axis"
        print(f"{r.cartridge} {rid}: cable {float(s.cable_length) / 1000.0:.0f} um, along L8 axis {t.min():.0f}..{t.max():.0f}{msg}")
