"""Template-vs-each-neuron view: full L2 skeletons rigidly aligned to a template,
with the pairwise NBLAST score annotated.

Neuron 0 is the template; each subsequent neuron is rigidly aligned onto it and
shown overlaid. Read-only CAVE access (skeleton fetch only), cached to disk.

    .venv/bin/python faab/template_pairwise_viz.py
"""
from faab.config import REPO
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import navis
from navis.transforms.align import align_rigid
import caveclient
from fafbseg import flywire

from faab.clients.cave import latest_roots, throttle, DATASTACK

IDS = [648518347611904256, 648518347550912741, 648518347638273763]
OUT = os.path.join(REPO, "figures")
os.makedirs(OUT, exist_ok=True)

TEMPLATE_COLOR = "#777777"
PAIR_COLORS = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def short(i):
    return str(int(i))[-6:]


def fetch_skeletons_um():
    """Read-only L2 skeleton fetch (refined), returned in microns. Cached."""
    cache = os.path.join(OUT, "skel_cache.pkl")
    if os.path.exists(cache):
        sk, roots = pickle.load(open(cache, "rb"))
        print("loaded cached skeletons:", [short(r) for r in roots])
        return sk, roots
    client = caveclient.CAVEclient(DATASTACK)
    roots, t = latest_roots(IDS, client=client)
    t = throttle(t)
    sk = flywire.get_l2_skeleton(roots, dataset=DATASTACK, max_threads=1, progress=False)
    sk = sk / 1000.0                      # L2 coords are nm -> microns
    for s in sk:
        s.units = "um"
    pickle.dump((sk, roots), open(cache, "wb"))
    print("fetched + cached skeletons:", [short(r) for r in roots])
    return sk, roots


def dp(neuron, k=5):
    d = navis.make_dotprops(neuron, k=k)
    d.units = "um"
    return d


def nblast_mean(a, b):
    s1 = float(navis.nblast(a, b, n_cores=1, progress=False).iloc[0, 0])
    s2 = float(navis.nblast(b, a, n_cores=1, progress=False).iloc[0, 0])
    return 0.5 * (s1 + s2)


def main():
    sk, roots = fetch_skeletons_um()
    labels = [short(r) for r in roots]
    spans = [tuple((s.nodes[["x", "y", "z"]].max() - s.nodes[["x", "y", "z"]].min()).round(1))
             for s in sk]
    print("skeleton xyz spans (um):", spans)

    tmpl = sk[0]
    tmpl_dp = dp(tmpl)

    pairs = list(range(1, len(sk)))
    views = [(("x", "y"), "XY"), (("x", "z"), "XZ")]
    fig, axes = plt.subplots(len(views), len(pairs),
                             figsize=(6 * len(pairs), 9), squeeze=False)

    scores = {}
    for c, i in enumerate(pairs):
        aligned = align_rigid(navis.NeuronList(sk[i]), target=tmpl, progress=False)
        if isinstance(aligned, tuple):
            aligned = aligned[0]
        aligned = aligned[0]                       # NeuronList -> single neuron
        score = nblast_mean(dp(aligned), tmpl_dp)
        scores[labels[i]] = score
        col = PAIR_COLORS[c % len(PAIR_COLORS)]
        for r, (view, vname) in enumerate(views):
            ax = axes[r][c]
            navis.plot2d(tmpl, color=TEMPLATE_COLOR, ax=ax, method="2d",
                         view=view, lw=1.6, alpha=0.65)
            navis.plot2d(aligned, color=col, ax=ax, method="2d",
                         view=view, lw=1.3, alpha=0.9)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_ylabel(vname if c == 0 else "")
            if r == 0:
                ax.set_title(f"template {labels[0]}  +  {labels[i]}\n"
                             f"NBLAST = {score:.3f}", fontsize=12)

    handles = [plt.Line2D([0], [0], color=TEMPLATE_COLOR, lw=2.5,
                          label=f"template {labels[0]}")]
    handles += [plt.Line2D([0], [0], color=PAIR_COLORS[c % len(PAIR_COLORS)], lw=2.5,
                           label=labels[i]) for c, i in enumerate(pairs)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.suptitle("Full morphologies rigidly aligned to template (with NBLAST scores)",
                 y=0.99, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out = os.path.join(OUT, "template_pairwise.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("\nscores vs template:", {k: round(v, 3) for k, v in scores.items()})
    print("saved figure ->", out)


if __name__ == "__main__":
    main()
