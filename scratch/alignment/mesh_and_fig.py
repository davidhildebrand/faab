"""List available aedes neuropil meshes (esp. optic lobe), and re-render the CT1
side-by-side with IDENTICAL rendering (point scatter) for fly vs FAAB.
"""
from faab.config import REPO
import os
import time
import json
import numpy as np
import requests
import caveclient
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fafbseg import flywire
from neuprint import NeuronCriteria as NC, fetch_neurons
from navis.interfaces import neuprint as nvneu
from faab.clients.neuprint import client as npclient, MALE_CNS
from faab.clients.cave import throttle, DATASTACK

OUT = os.path.join(REPO, "figures")
CT1 = 648518347700703374


def list_neuropils():
    base = "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/aedes_brain"
    try:
        info = requests.get(base + "/info", timeout=30).json()
        sp = info.get("segment_properties")
        spi = requests.get(f"{base}/{sp}/info", timeout=30).json()
        inl = spi["inline"]
        ids = inl["ids"]
        labels = next(p["values"] for p in inl["properties"] if p.get("type") == "label")
        print(f"aedes_neuropil regions ({len(ids)}):")
        ol = []
        for i, l in zip(ids, labels):
            mark = ""
            if any(k in str(l).lower() for k in ["medulla", "lobula", "optic", "lop", "me", "lo", "la"]):
                mark = "  <-- optic-lobe?"
                ol.append((i, l))
            print(f"  id {i}: {l}{mark}")
        return ol
    except Exception as e:
        print("neuropil list err:", str(e)[:150])
        return []


def scatter_panel(ax, pts, color, view):
    a, b = {"x": 0, "y": 1, "z": 2}[view[0]], {"x": 0, "y": 1, "z": 2}[view[1]]
    ax.scatter(pts[:, a], pts[:, b], s=1.5, c=color, alpha=0.5, linewidths=0)
    ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])


def main():
    ol = list_neuropils()

    client = caveclient.CAVEclient(DATASTACK)
    t = time.time()
    dp = flywire.get_l2_dotprops([CT1], dataset=DATASTACK, max_threads=1, omit_failures=True, progress=False)
    aedes_pts = dp[0].points / 1000.0

    c = npclient(MALE_CNS)
    ct1 = fetch_neurons(NC(type="CT1"), client=c)[0]
    sk = nvneu.fetch_skeletons(list(ct1["bodyId"]), client=c)
    fly_pts = [(int(s.id), s.nodes[["x", "y", "z"]].values * 8 / 1000.0) for s in sk]

    # IDENTICAL rendering (point scatter) for all; fly = navy, aedes = red
    panels = [(f"fly CT1 {bid}", p, "navy") for bid, p in fly_pts]
    panels.append((f"FAAB {str(CT1)[-6:]} (CT1 candidate)", aedes_pts, "firebrick"))
    views = [(("x", "y"), "XY"), (("x", "z"), "XZ")]
    fig, axes = plt.subplots(2, len(panels), figsize=(7 * len(panels), 13), squeeze=False)
    for col, (title, pts, color) in enumerate(panels):
        for r, (view, vn) in enumerate(views):
            ax = axes[r][col]
            scatter_panel(ax, pts, color, view)
            if r == 0:
                ax.set_title(f"{title}\n{len(pts)} pts, {np.linalg.norm(pts.max(0)-pts.min(0)):.0f}um", fontsize=15)
            if col == 0:
                ax.set_ylabel(vn, fontsize=14)
    fig.suptitle("CT1 morphology, same rendering: Drosophila male-CNS (navy) vs FAAB candidate (red)",
                 fontsize=17, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "ct1_compare.png"), dpi=110)
    print("\nsaved ct1_compare.png")


if __name__ == "__main__":
    main()
