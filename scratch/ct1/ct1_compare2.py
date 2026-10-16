"""CT1 comparison v2: dense aedes MESH vertices (not coarse dotprops), matched
um scale across all panels. fly CT1 (male-CNS) vs aedes candidate (FAAB)."""
from faab.config import REPO
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neuprint import NeuronCriteria as NC, fetch_neurons
from navis.interfaces import neuprint as nvneu
from faab.clients.neuprint import client as npclient, MALE_CNS

OUT = os.path.join(REPO, "figures")
CT1 = 648518347700703374


def aedes_mesh_pts():
    import cloudvolume as cv
    tok = json.load(open(os.path.expanduser("~/.cloudvolume/secrets/cave-secret.json")))["token"]
    vol = cv.CloudVolume("graphene://https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain",
                         use_https=True, secrets={"token": tok}, progress=False)
    m = vol.mesh.get(CT1)
    m = m[CT1] if isinstance(m, dict) else m
    v = np.asarray(m.vertices) / 1000.0
    if len(v) > 45000:
        v = v[np.random.RandomState(0).choice(len(v), 45000, replace=False)]
    return v


def main():
    av = aedes_mesh_pts()
    print(f"aedes mesh: {len(av)} vertices (sampled), extent {np.round(av.max(0)-av.min(0))}um")

    c = npclient(MALE_CNS)
    ct1 = fetch_neurons(NC(type="CT1"), client=c)[0]
    sk = nvneu.fetch_skeletons(list(ct1["bodyId"]), client=c)
    fly = [(int(s.id), s.nodes[["x", "y", "z"]].values * 8 / 1000.0) for s in sk]

    panels = [(f"fly CT1 {b}", p, "navy") for b, p in fly] + [(f"FAAB {str(CT1)[-6:]}", av, "firebrick")]
    W = max(np.linalg.norm(p.max(0) - p.min(0)) for _, p, _ in panels) / 2 * 1.05   # matched half-window
    print(f"matched window: +/-{W:.0f}um")
    views = [(("x", "y"), "XY"), (("x", "z"), "XZ")]
    ax_i = {"x": 0, "y": 1, "z": 2}
    fig, axes = plt.subplots(2, len(panels), figsize=(6.5 * len(panels), 12), squeeze=False)
    for col, (title, pts, color) in enumerate(panels):
        ctr = pts.mean(0)
        for r, (view, vn) in enumerate(views):
            a, b = ax_i[view[0]], ax_i[view[1]]
            ax = axes[r][col]
            ax.scatter(pts[:, a], pts[:, b], s=1.0, c=color, alpha=0.35, linewidths=0)
            ax.set_xlim(ctr[a] - W, ctr[a] + W); ax.set_ylim(ctr[b] - W, ctr[b] + W)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"{title}\n{len(pts)} pts | {np.linalg.norm(pts.max(0)-pts.min(0)):.0f}um", fontsize=14)
            if col == 0:
                ax.set_ylabel(vn, fontsize=13)
    fig.suptitle(f"CT1: fly male-CNS (navy) vs FAAB candidate (red) — matched scale (+/-{W:.0f}um)",
                 fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUT, "ct1_compare2.png"), dpi=110)
    print("saved ct1_compare2.png")


if __name__ == "__main__":
    main()
