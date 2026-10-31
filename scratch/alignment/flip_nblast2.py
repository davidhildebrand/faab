"""Plain NBLAST of one aedes cell vs the L-R MIRRORED image of another, using the
lab's symmetric mirror registration (aedes_aedesSym_1000_tps) reproduced in navis.

A = 648518347638241763
B = 648518347495933144   (mirrored to the opposite side, then compared)

Mirror = aedes_mirror(method="tps"): TPS map original->symmetric space, flip X at
mirrorAxisSize=734.144 um, TPS map back. refmat/tarmat (986 landmark pairs, um)
were dumped from the .rds; navis.TPStransform is the same engine nat/Morpho uses.

Plain NBLAST (position-dependent, fast) is appropriate here BECAUSE the mirror puts
B onto A's side -> no CPD alignment needed. A vs B *un*mirrored is included as a
control (should score near the floor: they sit on opposite sides).

Dataset: aedes (wclee_aedes_brain). Read-only, throttled.
"""
from faab.config import REPO
import os
import sys
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):        # morphops (navis TPS backend) predates numpy 2.0
    np.row_stack = np.vstack
import pandas as pd
import navis
from navis.transforms.thinplate import TPStransform
from fafbseg import flywire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.clients.cave import DATASTACK, latest_roots, throttle
from faab.config import fig

OUT = os.path.join(REPO, "figures")
A_IN = 648518347638241763
B_IN = 648518347495933144
MIRROR_AXIS = 734.144            # um; nat::mirror mirrorAxisSize for aedes symmetric


def build_mirror():
    ref = pd.read_csv(fig("mirror_refmat.csv")).values.astype(float)  # original space
    tar = pd.read_csv(fig("mirror_tarmat.csv")).values.astype(float)  # symmetric space
    fwd = TPStransform(ref, tar)     # original -> symmetric
    bwd = TPStransform(tar, ref)     # symmetric -> original

    def mirror_um(pts):
        pts = np.asarray(pts, float)
        s = fwd.xform(pts)
        s[:, 0] = MIRROR_AXIS - s[:, 0]     # flip across X in symmetric space
        return bwd.xform(s)
    return mirror_um


def dp(skel, name):
    d = navis.make_dotprops(skel, k=5, resample=False)
    d.units = "um"
    d.name = name
    d.id = name
    return d


def main():
    print("fetching skeletons ...", flush=True)
    roots, _ = latest_roots([A_IN, B_IN])
    A_root, B_root = roots[0], roots[1]
    t = throttle(time.time())
    sk = flywire.get_l2_skeleton([A_root, B_root], dataset=DATASTACK,
                                 max_threads=1, omit_failures=True, progress=False)
    sk_by = {}
    for s in sk:
        s2 = s / 1000.0
        s2.units = "um"
        sk_by[int(s.id)] = s2
    skA, skB = sk_by[A_root], sk_by[B_root]
    print(f"A {A_IN}->{A_root}: {len(skA.nodes):,} nodes,  cable {skA.cable_length:.0f} um", flush=True)
    print(f"B {B_IN}->{B_root}: {len(skB.nodes):,} nodes,  cable {skB.cable_length:.0f} um", flush=True)

    # ---- mirror B through the lab's symmetric registration ----
    mirror_um = build_mirror()
    skBm = skB.copy()
    skBm.nodes.loc[:, ["x", "y", "z"]] = mirror_um(skB.nodes[["x", "y", "z"]].values)
    skBm._clear_temp_attr()

    cA = skA.nodes[["x", "y", "z"]].values.mean(0)
    cB = skB.nodes[["x", "y", "z"]].values.mean(0)
    cBm = skBm.nodes[["x", "y", "z"]].values.mean(0)
    print(f"\ncentroid A       (um): {cA.round(1)}")
    print(f"centroid B raw   (um): {cB.round(1)}   |A - B_raw|    = {np.linalg.norm(cA-cB):.1f} um")
    print(f"centroid B mirror(um): {cBm.round(1)}   |A - B_mirror| = {np.linalg.norm(cA-cBm):.1f} um")

    # ---- plain NBLAST (position-dependent, no alignment) ----
    dA, dBraw, dBm = dp(skA, "A"), dp(skB, "B_raw"), dp(skBm, "B_mirror")
    nl = navis.NeuronList([dA, dBraw, dBm])
    S = navis.nblast(nl, nl, scores="mean", normalized=True, progress=False)
    s_raw = S.loc["A", "B_raw"]
    s_mir = S.loc["A", "B_mirror"]
    print("\n=== plain NBLAST (normalized, symmetric 'mean'; 1.0 = identical) ===")
    print(f"  A  vs  B (un-mirrored, opposite sides)  : {s_raw:+.3f}   <- control, expect low")
    print(f"  A  vs  B (L-R mirrored onto A's side)   : {s_mir:+.3f}")

    # ---- figure ----
    NAVY, GREEN, ORANGE = "#1f4e8c", "#2e8b57", "#e2571e"
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3)
    for c, (neuron, col, title) in enumerate([
            (skA, NAVY, f"A  {str(A_root)[-6:]}"),
            (skB, GREEN, f"B  {str(B_root)[-6:]}  (raw)"),
            (skBm, ORANGE, f"B  {str(B_root)[-6:]}  (mirrored)")]):
        ax = fig.add_subplot(gs[0, c])
        navis.plot2d(neuron, ax=ax, method="2d", view=("x", "y"), color=col, lw=1.3)
        ax.set_title(title, fontsize=12); ax.set_aspect("equal"); ax.axis("off")

    axL = fig.add_subplot(gs[1, 0])
    navis.plot2d([skA, skB], ax=axL, method="2d", view=("x", "y"), color=[NAVY, GREEN], lw=1.1)
    axL.set_title(f"A + B raw (same coords)\nopposite sides -> NBLAST {s_raw:.3f}", fontsize=11)
    axL.set_aspect("equal"); axL.axis("off")

    axR = fig.add_subplot(gs[1, 1:3])
    navis.plot2d([skA, skBm], ax=axR, method="2d", view=("x", "y"), color=[NAVY, ORANGE], lw=1.2)
    axR.set_title(f"A (navy) + B mirrored onto A's side (orange), SAME coords, no alignment\n"
                  f"plain NBLAST = {s_mir:.3f}", fontsize=12)
    axR.set_aspect("equal"); axR.axis("off")

    fig.suptitle(f"A={A_root}  vs  L-R mirrored  B={B_root}   (dataset: aedes; lab TPS mirror)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = fig("flip_nblast.png")
    fig.savefig(out, dpi=190)
    print(f"\nsaved {out}", flush=True)
    pickle.dump({"A_root": A_root, "B_root": B_root, "s_raw": float(s_raw),
                 "s_mirror": float(s_mir), "S": S,
                 "centroid_dist_mirror_um": float(np.linalg.norm(cA-cBm))},
                open(fig("flip_nblast.pkl"), "wb"))


if __name__ == "__main__":
    main()
