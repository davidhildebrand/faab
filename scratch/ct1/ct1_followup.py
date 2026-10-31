"""CT1 follow-ups: brain-mesh ng link (verify alignment), fly CT1<->Mi1 numbers +
count (male-CNS), and a Drosophila-vs-FAAB CT1 side-by-side figure.
"""
from faab.config import REPO
import os
import json
import time
import urllib.parse
import numpy as np
import pandas as pd
import caveclient
import navis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fafbseg import flywire
from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from navis.interfaces import neuprint as nvneu
from faab.clients.neuprint import client as npclient, MALE_CNS
from faab.clients.cave import throttle, DATASTACK
from faab.config import fig

OUT = os.path.join(REPO, "figures")
NET = 648518347700703374
FAAB = [648518347700703374, 648518347534000721, 648518347595752117]  # net + LOME + 2nd net
MESH_BBOX = (np.array([40865., 29642., 0.]), np.array([522417., 659470., 330615.]))


def ng_link(segids, mesh=True):
    img = "precomputed://gs://zetta_lee_mosquito_001_alignment_central/img/v2_sharded"
    seg = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain"
    layers = [{"type": "image", "source": img, "name": "img"}]
    if mesh:
        layers.append({"type": "segmentation",
                       "source": "precomputed://gs://aedes_whole-brain/meshes",
                       "segments": ["1"], "objectAlpha": 0.18, "name": "brain_mesh"})
    layers.append({"type": "segmentation", "source": seg,
                   "segments": [str(s) for s in segids], "name": "CT1-like"})
    return "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(
        json.dumps({"layers": layers, "layout": "3d"}, separators=(",", ":")))


def main():
    # ---- FAAB morphologies (one CAVE fetch) ----
    client = caveclient.CAVEclient(DATASTACK)
    t = time.time()
    dp_nm = flywire.get_l2_dotprops(FAAB, dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False)
    by = {int(d.id): d for d in dp_nm}
    nb = by[NET].points
    print(f"[align] net cell {NET} bbox(nm): {nb.min(0).round(0)} -> {nb.max(0).round(0)}")
    inside = bool((nb.min(0) >= MESH_BBOX[0] - 5000).all() and (nb.max(0) <= MESH_BBOX[1] + 5000).all())
    print(f"[align] neuron inside brain-mesh bbox? {inside}  (-> mesh {'ALIGNED' if inside else 'MAY BE MISALIGNED'})")

    print("\n[1] ng link WITH brain mesh:")
    print(ng_link([NET], mesh=True))

    # ---- fly CT1 count + CT1<->Mi1 (male-CNS) ----
    c = npclient(MALE_CNS)
    ct1 = fetch_neurons(NC(type="CT1"), client=c)[0]
    print(f"\n[3] (male-CNS) CT1 count: {len(ct1)} (per-brain)  bodyIds {list(ct1['bodyId'])}")
    d = fetch_simple_connections(NC(type="CT1"), NC(type="Mi1"), client=c)
    u = fetch_simple_connections(NC(type="Mi1"), NC(type="CT1"), client=c)
    n_mi1 = fetch_neurons(NC(type="Mi1"), client=c)[0].shape[0]
    print(f"   total Mi1 in male-CNS: {n_mi1}")
    print(f"   CT1->Mi1: {len(d)} connections to {d['bodyId_post'].nunique()} Mi1, "
          f"median {d['weight'].median():.0f} syn (range {d['weight'].min()}-{d['weight'].max()})")
    print(f"   Mi1->CT1: {len(u)} connections from {u['bodyId_pre'].nunique()} Mi1, "
          f"median {u['weight'].median():.0f} syn")
    touched = len(set(d['bodyId_post']) | set(u['bodyId_pre']))
    print(f"   distinct Mi1 touched by CT1 (either dir): {touched}/{n_mi1} ({touched/n_mi1*100:.0f}%)")

    # ---- side-by-side figure: fly CT1 (2) + FAAB putative (3) ----
    ct1_sk = nvneu.fetch_skeletons(list(ct1["bodyId"]), client=c)
    fly = []
    for s in ct1_sk:
        s2 = s.copy(); s2.nodes[["x", "y", "z"]] = s2.nodes[["x", "y", "z"]] * 8 / 1000.0  # 8nm vox -> um
        fly.append((f"fly CT1\n{int(s.id)}", s2))
    faabn = [(f"FAAB {str(i)[-6:]}", by[i] / 1000.0) for i in FAAB if i in by]
    panels = fly + faabn
    views = [(("x", "y"), "XY"), (("x", "z"), "XZ")]
    fig, axes = plt.subplots(2, len(panels), figsize=(2.6 * len(panels), 6), squeeze=False)
    for col, (title, n) in enumerate(panels):
        color = "navy" if col < len(fly) else "firebrick"
        for r, (view, vn) in enumerate(views):
            ax = axes[r][col]
            navis.plot2d(n, ax=ax, method="2d", view=view, color=color, lw=0.6)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(title, fontsize=8)
            if col == 0: ax.set_ylabel(vn)
    fig.suptitle("CT1: Drosophila male-CNS (navy) vs FAAB putative (red)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig("ct1_sidebyside.png"), dpi=140)
    print("\n[2] saved ct1_sidebyside.png")


if __name__ == "__main__":
    main()
