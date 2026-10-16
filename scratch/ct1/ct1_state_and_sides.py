"""Decode the user's neuroglancer state to get the aligned brain-mesh layer;
check whether 700703374 and 595752117 are on opposite sides; re-render the
CT1 side-by-side larger.
"""
from faab.config import REPO
import os
import json
import time
import urllib.parse
import pickle
import numpy as np
import pandas as pd
import caveclient
import navis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fafbseg import flywire
from neuprint import NeuronCriteria as NC, fetch_neurons
from navis.interfaces import neuprint as nvneu
from faab.sources.neuprint import client as npclient, MALE_CNS
from faab.sources.cave import throttle, DATASTACK

OUT = os.path.join(REPO, "figures")
STATE_ID = 6364293577572352
A, B = 648518347700703374, 648518347595752117
FAAB = [A, B, 648518347534000721]


def main():
    client = caveclient.CAVEclient(DATASTACK)

    # ---- (re3-link) decode the shared state, find the brain-mesh layer ----
    t = time.time()
    state = client.state.get_state_json(STATE_ID)
    print("layers in shared state:")
    mesh_layers = []
    for L in state.get("layers", []):
        src = L.get("source")
        srcs = src if isinstance(src, str) else (src.get("url") if isinstance(src, dict) else str(src))
        print(f"  - {L.get('name')}: type={L.get('type')} source={str(srcs)[:80]}")
        if "aedes_whole-brain" in str(src) or (L.get("type") == "mesh") or "mesh" in str(L.get("name", "")).lower():
            mesh_layers.append(L)
    print(f"\nbrain-mesh layer(s) found: {[L.get('name') for L in mesh_layers]}")
    if mesh_layers:
        print("  mesh layer JSON (verbatim, for reuse):")
        print("  " + json.dumps(mesh_layers[0])[:400])

    # ---- (re2-sides) centroids of A, B vs the Mi1 cloud ----
    t = throttle(t)
    dp = flywire.get_l2_dotprops(FAAB, dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False)
    by = {int(d.id): d.points / 1000.0 for d in dp}     # microns
    mi1 = pickle.load(open(os.path.join(OUT, "mi1_dotprops.pkl"), "rb"))
    mi1_all = np.vstack([d.points for d in mi1])         # already microns
    mi1_c = mi1_all.mean(0)
    print("\n=== positions (microns) ===")
    print(f"  Mi1 cloud centroid: {mi1_c.round(0)}  extent: {(mi1_all.max(0)-mi1_all.min(0)).round(0)}")
    for r in FAAB:
        c = by[r].mean(0)
        print(f"  {r}: centroid {c.round(0)}  | dist to Mi1 centroid {np.linalg.norm(c-mi1_c):.0f}um")
    ca, cb = by[A].mean(0), by[B].mean(0)
    print(f"\n  A-B centroid separation: {np.abs(ca-cb).round(0)} um (per axis), total {np.linalg.norm(ca-cb):.0f}um")

    # ---- build aligned brain-mesh link (reuse mesh layer from the state) ----
    img = "precomputed://gs://zetta_lee_mosquito_001_alignment_central/img/v2_sharded"
    seg = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain"
    layers = [{"type": "image", "source": img, "name": "img"}]
    for L in mesh_layers:
        layers.append(L)
    layers.append({"type": "segmentation", "source": seg,
                   "segments": [str(A), str(B)], "name": "CT1-like"})
    url = "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(
        json.dumps({"layers": layers, "layout": "3d"}, separators=(",", ":")))
    open(os.path.join(OUT, "ct1_mesh_link.txt"), "w").write(url + "\n")
    print("\nsaved aligned brain-mesh link -> figures/ct1_mesh_link.txt")

    # ---- (re1) larger side-by-side ----
    c = npclient(MALE_CNS)
    ct1 = fetch_neurons(NC(type="CT1"), client=c)[0]
    sk = nvneu.fetch_skeletons(list(ct1["bodyId"]), client=c)
    fly = []
    for s in sk:
        s2 = navis.resample_skeleton(s, 2000)            # coarser so it isn't a solid blob (8nm units)
        s2.nodes[["x", "y", "z"]] = s2.nodes[["x", "y", "z"]] * 8 / 1000.0
        fly.append((f"fly CT1 {int(s.id)}", s2, "navy"))
    faabn = [(f"FAAB {str(i)[-6:]}", navis.make_dotprops(by[i], k=5), "firebrick") for i in FAAB]
    panels = fly + faabn
    views = [(("x", "y"), "XY"), (("x", "z"), "XZ")]
    fig, axes = plt.subplots(2, len(panels), figsize=(5 * len(panels), 11), squeeze=False)
    for col, (title, n, color) in enumerate(panels):
        for r, (view, vn) in enumerate(views):
            ax = axes[r][col]
            navis.plot2d(n, ax=ax, method="2d", view=view, color=color, lw=0.7, alpha=0.7)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(title, fontsize=13)
            if col == 0: ax.set_ylabel(vn, fontsize=12)
    fig.suptitle("CT1: Drosophila male-CNS (navy) vs FAAB putative (red)", fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "ct1_sidebyside_large.png"), dpi=130)
    print("saved ct1_sidebyside_large.png")


if __name__ == "__main__":
    main()
