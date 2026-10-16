"""Side-by-side skeleton gallery for re-check #2: the 2 Group-1 + 1 Group-2 LC->MeTu4
candidates vs a MeTu4 exemplar and an LC exemplar. Real L2 skeletons, rigid+scale
aligned to the MeTu4 exemplar. Read-only, throttled."""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import navis
from navis.transforms.align import align_rigid
from fafbseg import flywire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.sources.cave import DATASTACK, throttle

OUT = os.path.join(REPO, "figures")
CANDS = [("648518347700868494", "G1"), ("648518347660289039", "G1"),
         ("648518347475132830", "G2")]


def main():
    sc = pd.read_csv(os.path.join(OUT, "metu_lc2_scores.csv"))
    sc["root"] = sc["root"].astype(str)
    simM = dict(zip(sc.root, sc.sim_MeTu4)); simL = dict(zip(sc.root, sc.sim_LC))
    # exemplars: most-central MeTu4 (max sim_MeTu4) and most-central LC (max sim_LC), well-traced
    m4 = sc[(sc.label == "MeTu4") & (sc.npts >= 120)].sort_values("sim_MeTu4", ascending=False)
    lcx = sc[(sc.label == "LC") & (sc.npts >= 120)].sort_values("sim_LC", ascending=False)
    metu_ex = str(m4.iloc[0].root); lc_ex = str(lcx.iloc[0].root)
    print(f"MeTu4 exemplar {metu_ex} (sim_M {simM[metu_ex]:.3f}); LC exemplar {lc_ex} "
          f"(sim_L {simL[lc_ex]:.3f})", flush=True)

    roots = [metu_ex, lc_ex] + [c for c, _ in CANDS]
    t = throttle(time.time())
    sk = flywire.get_l2_skeleton([int(r) for r in roots], dataset=DATASTACK,
                                 max_threads=1, omit_failures=True, progress=False)
    raw = {}
    for s in sk:
        s2 = s / 1000.0; s2.units = "um"; raw[str(int(s.id))] = s2
    print(f"skeletons: {sorted(raw)}", flush=True)

    ref = raw[metu_ex]
    aln = {metu_ex: ref}
    for r in roots:
        if r == metu_ex or r not in raw:
            continue
        a = align_rigid(navis.NeuronList(raw[r]), target=ref, scale=True, progress=False)
        aln[r] = (a[0] if isinstance(a, tuple) else a)[0]

    allpts = np.vstack([aln[r].nodes[["x", "y"]].values for r in aln])
    lo, hi = allpts.min(0), allpts.max(0); pad = 0.05 * (hi - lo)
    xlim = (lo[0]-pad[0], hi[0]+pad[0]); ylim = (lo[1]-pad[1], hi[1]+pad[1])

    BLUE, RED, AMBER, GREY = "#1f4e8c", "#e6194b", "#e08a1e", "#555555"
    ccol = {"G1": RED, "G2": AMBER}

    def draw(ax, items, title):
        for r, c in items:
            if r in aln:
                navis.plot2d(aln[r], ax=ax, method="2d", view=("x", "y"), color=c, lw=1.1)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(title, fontsize=10)

    fig = plt.figure(figsize=(22, 9))
    gs = fig.add_gridspec(2, 5)
    # row 0 — individuals
    draw(fig.add_subplot(gs[0, 0]), [(metu_ex, BLUE)], f"MeTu4 exemplar\n{metu_ex[-6:]}")
    for k, (c, grp) in enumerate(CANDS):
        draw(fig.add_subplot(gs[0, 1+k]), [(c, ccol[grp])],
             f"{grp} candidate {c[-6:]}\nsim_MeTu4 {simM[c]:.3f} / sim_LC {simL[c]:.3f}")
    draw(fig.add_subplot(gs[0, 4]), [(lc_ex, GREY)], f"LC exemplar\n{lc_ex[-6:]}")
    # row 1 — overlays on the MeTu4 exemplar (blue), candidate coloured; last = candidate vs LC
    draw(fig.add_subplot(gs[1, 0]), [(metu_ex, BLUE), (lc_ex, GREY)],
         "reference contrast\nMeTu4 (blue) + LC (grey)")
    for k, (c, grp) in enumerate(CANDS):
        draw(fig.add_subplot(gs[1, 1+k]), [(metu_ex, BLUE), (c, ccol[grp])],
             f"{c[-6:]} (col) on MeTu4 (blue)")
    c0 = CANDS[0][0]
    draw(fig.add_subplot(gs[1, 4]), [(lc_ex, GREY), (c0, RED)],
         f"{c0[-6:]} (red) on LC (grey)")

    fig.suptitle("LC->MeTu4 re-check #2: candidate skeletons vs MeTu4 and LC exemplars "
                 "(rigid+scale aligned to MeTu4)", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT, "metu_lc2_gallery.png")
    fig.savefig(out, dpi=170)
    print(f"saved {out}\nDONE", flush=True)


if __name__ == "__main__":
    main()
