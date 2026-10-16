"""Compare rigid vs PCA alignment for a small set of aedes neurons, and
visualize raw vs rigid-aligned vs pca-aligned overlays.

Run from the repo so `faab_morph` is importable:
    .venv/bin/python faab/align_compare_viz.py
"""
from faab.config import REPO
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import navis
from navis.transforms.align import align_rigid
from sklearn.decomposition import PCA

from faab.morph import fetch_dotprops_um

# Proper rotations (det=+1) of the PCA frame: resolve principal-axis sign ambiguity
# without mirroring (chirality-preserving).
PCA_FLIPS = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]


def pca_align(dp, ref_idx=0, k=5):
    """Reorient each neuron onto its 3D principal axes (centered), then pick the
    sign-flip that best matches the reference neuron by NBLAST. Returns a NeuronList.
    Unlike navis.align_pca (which projects to 1D), this preserves 3D shape."""
    def mkdp(co):
        d = navis.make_dotprops(np.asarray(co), k=k)
        d.units = "um"                      # coords are already microns
        return d

    coords = [PCA(n_components=3).fit_transform(d.points) for d in dp]  # centered+rotated
    ref = mkdp(coords[ref_idx])
    out = []
    for i, co in enumerate(coords):
        if i == ref_idx:
            out.append(ref)
            continue
        best, best_s = None, -np.inf
        for f in PCA_FLIPS:
            cand = mkdp(co * np.array(f))
            s = float(navis.nblast(cand, ref, n_cores=1, progress=False).iloc[0, 0])
            if s > best_s:
                best, best_s = cand, s
        out.append(best)
    nl = navis.NeuronList(out)
    for n, d in zip(nl, dp):
        n.id = d.id
    return nl

IDS = [648518347611904256, 648518347550912741, 648518347638273763]
COLORS = ["#1f77b4", "#d62728", "#2ca02c"]      # one per neuron, consistent across panels
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)


def offdiag_mean(m):
    a = m.values.astype(float).copy()
    np.fill_diagonal(a, np.nan)
    return np.nanmean(a)


def short(ids):
    return [str(i)[-6:] for i in ids]


def main():
    # ---- fetch (only CAVE access; read-only, throttled). Cache so plot iteration
    #      does not keep hitting CAVE. ----
    import pickle
    cache = os.path.join(OUT, "dp_cache.pkl")
    if os.path.exists(cache):
        dp, roots = pickle.load(open(cache, "rb"))
        print("loaded cached dotprops:", [str(r)[-6:] for r in roots])
    else:
        dp, roots = fetch_dotprops_um(IDS)
        pickle.dump((dp, roots), open(cache, "wb"))
        print("fetched + cached dotprops")
    labels = short(roots)
    for d, lab in zip(dp, labels):
        d.name = lab
    print("fetched", len(dp), "dotprops (microns):", labels)

    # ---- apply each alignment method (same neurons, different pose handling) ----
    raw = dp
    rigid = align_rigid(dp, target=dp[0], progress=False)      # all -> neuron 0 frame (CPD)
    rigid = rigid[0] if isinstance(rigid, tuple) else rigid
    pca = pca_align(dp, ref_idx=0)                             # 3D principal-axis frame, flip-resolved

    # ---- #1: score each with the SAME metric (nblast_allbyall) for a fair compare ----
    s_raw = navis.nblast_allbyall(raw, n_cores=1, progress=False)
    s_rigid = navis.nblast_allbyall(rigid, n_cores=1, progress=False)
    s_pca = navis.nblast_allbyall(pca, n_cores=1, progress=False)
    for m in (s_raw, s_rigid, s_pca):
        m.index = labels
        m.columns = labels

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n--- NBLAST on RAW (pose-dependent) ---"); print(s_raw.round(3))
    print("\n--- NBLAST after RIGID align (to neuron %s) ---" % labels[0]); print(s_rigid.round(3))
    print("\n--- NBLAST after PCA align (per-neuron) ---"); print(s_pca.round(3))
    print(f"\nmean off-diagonal:  raw={offdiag_mean(s_raw):.3f}  "
          f"rigid={offdiag_mean(s_rigid):.3f}  pca={offdiag_mean(s_pca):.3f}")

    # ---- coordinate diagnostics (debug the PCA panel) ----
    for nm, nl in (("raw", raw), ("rigid", rigid), ("pca", pca)):
        sp = [tuple((d.points.max(0) - d.points.min(0)).round(1)) for d in nl]
        print(f"{nm:5s} per-neuron xyz spans (um):", sp)

    panels = [("Raw (brain coords)", raw),
              ("Rigid -> neuron %s" % labels[0], rigid),
              ("PCA (per-neuron)", pca)]
    views = [(("x", "y"), "XY"), (("x", "z"), "XZ")]

    fig, axes = plt.subplots(len(views), len(panels), figsize=(15, 9))
    for r, (view, vname) in enumerate(views):
        for c, (title, nl) in enumerate(panels):
            ax = axes[r, c]
            navis.plot2d(nl, color=COLORS, ax=ax, method="2d", view=view,
                         lw=1.2, alpha=0.9)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(title, fontsize=11)
            ax.set_ylabel(vname if c == 0 else "")
            ax.set_xticks([]); ax.set_yticks([])
    # legend
    handles = [plt.Line2D([0], [0], color=COLORS[i], lw=2, label=labels[i])
               for i in range(len(dp))]
    fig.suptitle("Aedes neurons: raw vs rigid vs PCA (3D) alignment", y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.legend(handles=handles, loc="lower center", ncol=len(dp), frameon=False,
               bbox_to_anchor=(0.5, 0.0))

    out = os.path.join(OUT, "align_compare.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nsaved figure ->", out)


if __name__ == "__main__":
    main()
