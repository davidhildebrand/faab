"""NBLAST comparison of one aedes cell against the L-R MIRRORED image of another.

A = 648518347638241763
B = 648518347495933144   (compared as-is AND L-R flipped)

Why flip: nblast_align aligns by rigid rotation+translation (+scale), which
preserves handedness. Two contralateral homologues are ~mirror images, so a
rigid alignment cannot overlay them -> low score. Reflecting one cell first
(any single-axis flip reverses chirality) lets the alignment overlay them, so
a high flipped score / low unflipped score is the signature of a true L-R pair.
The reflection PLANE is irrelevant here because the subsequent rigid alignment
absorbs any rotation/translation; only chirality matters, so we negate x.

Dataset: aedes (wclee_aedes_brain). Read-only, throttled.
"""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
import navis
from navis.transforms.align import align_rigid
from fafbseg import flywire
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from faab.clients.cave import DATASTACK, throttle

from faab.morph import fetch_dotprops_um

OUT = os.path.join(REPO, "figures")
A_IN = 648518347638241763
B_IN = 648518347495933144
FLIP = np.array([-1, 1, 1])          # reflection (det = -1) -> reverses handedness


def prep(nl_or_neuron, name):
    n = nl_or_neuron.copy()
    n.name = name
    n.id = name
    n.units = "um"
    return n


def main():
    # ---- fetch dotprops (for NBLAST) + skeletons (for morphology figure) ----
    dp, roots = fetch_dotprops_um([A_IN, B_IN], update_roots=True)
    dp_by = {}
    for d in dp:
        dp_by[int(d.id)] = d
    A_root, B_root = roots[0], roots[1]
    print(f"A: {A_IN} -> {A_root}   B: {B_IN} -> {B_root}")

    t = throttle(time.time())
    sk = flywire.get_l2_skeleton([A_root, B_root], dataset=DATASTACK,
                                 max_threads=1, omit_failures=True, progress=False)
    sk_by = {}
    for s in sk:
        s2 = s / 1000.0
        s2.units = "um"
        sk_by[int(s.id)] = s2

    A_dp = prep(dp_by[A_root], "A")
    B_dp = prep(dp_by[B_root], "B")
    Bf_dp = prep(dp_by[B_root] * FLIP, "B_flip")

    # ---- confirm the two cells sit on opposite sides ----
    ca = np.asarray(A_dp.points).mean(0)
    cb = np.asarray(B_dp.points).mean(0)
    print(f"centroid A (um): {ca.round(1)}")
    print(f"centroid B (um): {cb.round(1)}")
    print(f"centroid difference |A-B| per axis (um): {np.abs(ca - cb).round(1)}  "
          f"(largest separation axis = {'xyz'[int(np.argmax(np.abs(ca - cb)))]})")

    # ---- nblast_align: A vs B (raw) and A vs B_flip ----
    nls = navis.NeuronList([A_dp, B_dp, Bf_dp])
    S = navis.nblast_align(nls, nls, align_method="rigid", two_way_align=False,
                           n_cores=1, progress=False, align_kwargs={"scale": True})

    def sym(i, j):
        return 0.5 * (S.loc[i, j] + S.loc[j, i])

    s_raw = sym("A", "B")
    s_flip = sym("A", "B_flip")
    print("\n=== nblast_align scores (symmetrised, 1.0 = identical) ===")
    print(f"  A  vs  B         (as-is, chirality mismatch): {s_raw:+.3f}")
    print(f"  A  vs  B_flip    (L-R mirrored)             : {s_flip:+.3f}")
    print(f"  gain from flipping                          : {s_flip - s_raw:+.3f}")

    # ---- figure: real morphology + aligned overlays ----
    aA = align_rigid(navis.NeuronList(sk_by[A_root]), target=sk_by[A_root], progress=False)
    refA = (aA[0] if isinstance(aA, tuple) else aA)[0]
    skB = sk_by[B_root]
    skBf = skB * FLIP
    aB = align_rigid(navis.NeuronList(skB), target=sk_by[A_root], progress=False)
    aBf = align_rigid(navis.NeuronList(skBf), target=sk_by[A_root], progress=False)
    B_on_A = (aB[0] if isinstance(aB, tuple) else aB)[0]
    Bf_on_A = (aBf[0] if isinstance(aBf, tuple) else aBf)[0]

    NAVY, GREEN, ORANGE = "#1f4e8c", "#2e8b57", "#e2571e"
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 3)
    panels = [(0, 0, sk_by[A_root], NAVY, f"A  {str(A_root)[-6:]}"),
              (0, 1, skB, GREEN, f"B  {str(B_root)[-6:]}  (raw)"),
              (0, 2, skBf, ORANGE, f"B  {str(B_root)[-6:]}  (L-R flipped)")]
    for r, c, neuron, col, title in panels:
        ax = fig.add_subplot(gs[r, c])
        navis.plot2d(neuron, ax=ax, method="2d", view=("x", "y"), color=col, lw=1.5)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal"); ax.axis("off")

    axL = fig.add_subplot(gs[1, 0:2])
    navis.plot2d([refA, B_on_A], ax=axL, method="2d", view=("x", "y"),
                 color=[NAVY, GREEN], lw=1.4)
    axL.set_title(f"A (navy) + B raw, rigid-aligned onto A (green)\n"
                  f"nblast_align = {s_raw:.3f}  — chirality mismatch, poor overlay",
                  fontsize=12)
    axL.set_aspect("equal"); axL.axis("off")

    axR = fig.add_subplot(gs[1, 2])
    navis.plot2d([refA, Bf_on_A], ax=axR, method="2d", view=("x", "y"),
                 color=[NAVY, ORANGE], lw=1.4)
    axR.set_title(f"A (navy) + B FLIPPED, aligned onto A (orange)\n"
                  f"nblast_align = {s_flip:.3f}", fontsize=12)
    axR.set_aspect("equal"); axR.axis("off")

    fig.suptitle(f"A={A_root}  vs  B={B_root} raw and L-R flipped  (dataset: aedes)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(OUT, "flip_nblast.png")
    fig.savefig(out, dpi=190)
    print(f"\nsaved {out}")
    pickle.dump({"A_root": A_root, "B_root": B_root, "s_raw": float(s_raw),
                 "s_flip": float(s_flip), "S": S},
                open(os.path.join(OUT, "flip_nblast.pkl"), "wb"))


if __name__ == "__main__":
    main()
