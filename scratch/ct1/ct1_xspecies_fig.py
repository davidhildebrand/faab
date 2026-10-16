"""Side-by-side figure: Drosophila CT1 (male-CNS) vs nblast_align'd Aedes putative
CT1 (A, B), with the cross-species nblast_align scores annotated.

Aligns CT1_R, A, B onto CT1_L with the SAME rigid+scale alignment nblast_align used,
then renders real (resampled) skeleton morphology. Read-only, throttled CAVE.
"""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import navis
from navis.transforms.align import align_rigid
from navis.interfaces import neuprint as npn
from fafbseg import flywire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.sources.cave import DATASTACK, latest_roots, throttle
from faab.sources.neuprint import client as npclient, MALE_CNS

OUT = os.path.join(REPO, "figures")
AEDES = {"A": 648518347638241763, "B": 648518347495933144}
FLY = {"CT1_L": 10009, "CT1_R": 10157}
NODES = 2500


def resample(n):
    return navis.resample_skeleton(n, resample_to=max(n.cable_length / NODES, 0.5),
                                   inplace=False)


def main():
    M = pickle.load(open(os.path.join(OUT, "ct1_xspecies_nblast.pkl"), "rb"))
    lab = M["labels"]; Mx = M["M"]
    idx = {n: i for i, n in enumerate(lab)}

    scache = os.path.join(OUT, "ct1_xspecies_skels.pkl")
    if os.path.exists(scache):
        raw = pickle.load(open(scache, "rb"))
    else:
        c = npclient(MALE_CNS)
        fly = npn.fetch_skeletons(list(FLY.values()), client=c, heal=True)
        raw = {}
        for n in fly:
            n.units = "8 nm"
            bid = int(n.id)
            key = "CT1_L" if bid == FLY["CT1_L"] else "CT1_R"
            raw[key] = n.convert_units("um")
        roots, _ = latest_roots(list(AEDES.values()))
        t = throttle(time.time())
        sk = flywire.get_l2_skeleton(roots, dataset=DATASTACK, max_threads=1,
                                     omit_failures=True, progress=False)
        for s, nm in zip(sk, ["A", "B"]):
            s2 = s / 1000.0; s2.units = "um"; raw[nm] = s2
        pickle.dump(raw, open(scache, "wb"))
    print("skeletons ready:", {k: int(v.cable_length) for k, v in raw.items()}, flush=True)

    rs = {k: resample(v) for k, v in raw.items()}
    ref = rs["CT1_L"]
    aligned = {"CT1_L": ref}
    for k in ("CT1_R", "A", "B"):
        a = align_rigid(navis.NeuronList(rs[k]), target=ref, scale=True, progress=False)
        aligned[k] = (a[0] if isinstance(a, tuple) else a)[0]
        print(f"aligned {k}", flush=True)

    # common limits (x,y) across all aligned cells
    allpts = np.vstack([aligned[k].nodes[["x", "y"]].values for k in aligned])
    lo, hi = allpts.min(0), allpts.max(0)
    pad = 0.05 * (hi - lo)
    xlim, ylim = (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - pad[1], hi[1] + pad[1])

    FLYC, AC, BC, RC = "#333333", "#1f4e8c", "#e2571e", "#1b9e77"
    col = {"CT1_L": FLYC, "CT1_R": RC, "A": AC, "B": BC}
    sc = {k: Mx[idx[k], idx["CT1_L"]] for k in ("CT1_R", "A", "B")}

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4)

    def draw(ax, items, title):
        for nm, c in items:
            navis.plot2d(aligned[nm], ax=ax, method="2d", view=("x", "y"), color=c, lw=0.7)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=12)

    # row 1: each cell alone, in the common (CT1_L) frame
    draw(fig.add_subplot(gs[0, 0]), [("CT1_L", FLYC)], "Drosophila CT1_L (reference)")
    draw(fig.add_subplot(gs[0, 1]), [("CT1_R", RC)], "Drosophila CT1_R")
    draw(fig.add_subplot(gs[0, 2]), [("A", AC)], f"Aedes A  (aligned)\nvs CT1_L = {sc['A']:.3f}")
    draw(fig.add_subplot(gs[0, 3]), [("B", BC)], f"Aedes B  (aligned)\nvs CT1_L = {sc['B']:.3f}")

    # row 2: overlays on the fly CT1_L reference
    draw(fig.add_subplot(gs[1, 0]), [("CT1_L", FLYC), ("CT1_R", RC)],
         f"CT1_L + CT1_R  (fly same-type ref)\nnblast_align = {Mx[idx['CT1_L'], idx['CT1_R']]:.3f}")
    draw(fig.add_subplot(gs[1, 1]), [("CT1_L", FLYC), ("A", AC)],
         f"CT1_L + Aedes A\nnblast_align = {sc['A']:.3f}")
    draw(fig.add_subplot(gs[1, 2]), [("CT1_L", FLYC), ("B", BC)],
         f"CT1_L + Aedes B\nnblast_align = {sc['B']:.3f}")

    axs = fig.add_subplot(gs[1, 3]); axs.axis("off")
    txt = ("cross-species nblast_align (rigid+scale)\n\n"
           f"  A     vs CT1_L : {Mx[idx['A'], idx['CT1_L']]:.3f}\n"
           f"  A     vs CT1_R : {Mx[idx['A'], idx['CT1_R']]:.3f}\n"
           f"  B     vs CT1_L : {Mx[idx['B'], idx['CT1_L']]:.3f}\n"
           f"  B     vs CT1_R : {Mx[idx['B'], idx['CT1_R']]:.3f}\n"
           f"  CT1_L vs CT1_R : {Mx[idx['CT1_L'], idx['CT1_R']]:.3f}  (fly ref)\n"
           f"  A     vs B     : {Mx[idx['A'], idx['B']]:.3f}  (rigid; chirality-\n"
           "                    suppressed; TPS-mirror plain NBLAST = 0.641)")
    axs.text(0.0, 0.95, txt, va="top", ha="left", fontsize=11, family="monospace")

    fig.suptitle("Drosophila CT1 (male-CNS) vs nblast_align'd Aedes putative CT1  (A, B)",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, "ct1_xspecies_sidebyside.png")
    fig.savefig(out, dpi=170)
    print("saved", out, "\nDONE", flush=True)


if __name__ == "__main__":
    main()
