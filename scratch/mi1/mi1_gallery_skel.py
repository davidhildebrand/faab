"""Re-render the Mi1 gallery with ACTUAL cell morphology (L2 skeletons from the
current segmentation), and render the pairwise nblast_align score matrix.
Dataset: aedes. Read-only. Reuses cached ids + score matrix."""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
import pandas as pd
import navis
import caveclient
from navis.transforms.align import align_rigid
from fafbseg import flywire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.clients.cave import DATASTACK, throttle
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def main():
    by = {int(d.id): d for d in pickle.load(open(fig("mi1_dotprops.pkl"), "rb"))}
    mi1 = list(by)
    nuc = pickle.load(open(fig("nuclei_cache.pkl"), "rb"))
    soma = set(int(x) for x in nuc["pt_root_id"] if pd.notna(x))
    mi1n = [r for r in mi1 if r in soma]
    M = pickle.load(open(fig("mi1_align_matrix.pkl"), "rb"))
    assert M.shape[0] == len(mi1n), (M.shape, len(mi1n))
    mean_sim = np.nanmean(M, axis=1)
    order = list(np.argsort(-mean_sim))
    print(f"soma-bearing Mi1: {len(mi1n)}")

    # ---- fetch REAL morphology (L2 skeletons, current segmentation) ----
    scache = fig("mi1_skeletons.pkl")
    if os.path.exists(scache):
        sk_by = pickle.load(open(scache, "rb"))
    else:
        client = caveclient.CAVEclient(DATASTACK)
        t = throttle(time.time())
        sk = flywire.get_l2_skeleton(mi1n, dataset=DATASTACK, max_threads=1,
                                     omit_failures=True, progress=False)
        sk_by = {}
        for s in sk:
            s2 = s / 1000.0
            s2.units = "um"
            sk_by[int(s.id)] = s2
        pickle.dump(sk_by, open(scache, "wb"))
    print(f"skeletons fetched: {len(sk_by)}")

    # align each skeleton to the medoid skeleton (common frame for comparison)
    medoid = mi1n[order[0]]
    ref = sk_by[medoid]
    aligned = {}
    for r in mi1n:
        if r not in sk_by:
            continue
        a = align_rigid(navis.NeuronList(sk_by[r]), target=ref, progress=False)
        aligned[r] = (a[0] if isinstance(a, tuple) else a)[0]

    # ---- gallery of real morphologies ----
    fig, axes = plt.subplots(5, 8, figsize=(20, 13))
    for ax in axes.flat:
        ax.axis("off")
    lo, hi = np.nanmin(mean_sim), np.nanmax(mean_sim)
    for k, i in enumerate(order):
        if k >= 40:
            break
        r = mi1n[i]
        if r not in aligned:
            continue
        ax = axes.flat[k]
        col = plt.cm.plasma(0.10 + 0.62 * (mean_sim[i] - lo) / (hi - lo))
        navis.plot2d(aligned[r], ax=ax, method="2d", view=("x", "y"), color=col, lw=1.4)
        ax.set_title(f"{str(r)[-6:]}  {mean_sim[i]:.2f}", fontsize=10)
        ax.set_aspect("equal")
    fig.suptitle("Soma-bearing Mi1 — actual morphology (L2 skeletons), aligned to medoid, "
                 "sorted by consistency", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(fig("mi1_gallery.png"), dpi=200)
    print("saved mi1_gallery.png (real morphology)")

    # ---- pairwise nblast_align score matrix (sorted by consistency) ----
    Ms = M[np.ix_(order, order)].copy()
    labels = [str(mi1n[i])[-6:] for i in order]
    fig2, ax2 = plt.subplots(figsize=(13, 11))
    cmap = plt.cm.magma.copy(); cmap.set_bad("0.85")
    im = ax2.imshow(np.ma.masked_invalid(Ms), cmap=cmap, vmin=np.nanmin(M), vmax=np.nanmax(M))
    ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, rotation=90, fontsize=6)
    ax2.set_yticks(range(len(labels))); ax2.set_yticklabels(labels, fontsize=6)
    fig2.colorbar(im, ax=ax2, shrink=0.8, label="nblast_align score")
    ax2.set_title("Pairwise nblast_align scores between soma-bearing Mi1 "
                  "(sorted by consistency; diagonal masked)", fontsize=12)
    fig2.tight_layout()
    fig2.savefig(fig("mi1_score_matrix.png"), dpi=180)
    print("saved mi1_score_matrix.png")


if __name__ == "__main__":
    main()
