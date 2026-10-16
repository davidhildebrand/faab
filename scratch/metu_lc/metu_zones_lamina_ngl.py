"""(2) Prototype: cluster aedes MeTu by AOTU output-synapse position (target zone)
        - a typing-free, fragmentation-free subtyping approach.
(3) Build a neuroglancer/Spelunker URL showing the annotated lamina (L1-L5) cells.

Read-only. CAVE calls serialized + throttled.
"""
from faab.config import REPO
import os
import json
import time
import urllib.parse

import numpy as np
import pandas as pd
import caveclient
import seaserpent as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster

from faab.sources.cave import throttle, DATASTACK

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
OUT = os.path.join(REPO, "figures")
BAD = {"duplicate", "fragment", "tiny", "bad_nucleus", "not_a_neuron"}


def to_int64(s):
    return s.map(lambda x: int(str(x)) if str(x).strip().lstrip("-").isdigit() else pd.NA).astype("Int64")


def main():
    # ---- metadata ----
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    meta["rid"] = to_int64(meta["root_id"])
    meta = meta[meta["rid"].notna()]
    metu = meta[(meta["subclass"] == "MeTu") & ~meta["status"].isin(BAD)]["rid"].astype("int64").tolist()
    lamina = meta[(meta["type"].astype(str) == "L1-L5") & ~meta["status"].isin(BAD)]["rid"].astype("int64").tolist()
    print(f"MeTu={len(metu)}  lamina(L1-L5)={len(lamina)}")

    client = caveclient.CAVEclient(DATASTACK)
    t = time.time()
    img = client.info.image_source()
    seg = client.info.segmentation_source()
    site = client.info.viewer_site() or "https://spelunker.cave-explorer.org/"

    # ---- (3) lamina neuroglancer link (update roots to latest first) ----
    t = throttle(t)
    lam_latest = [int(x) for x in client.chunkedgraph.get_latest_roots(lamina)]
    seg_url = seg.replace("graphene://", "graphene://middleauth+") if seg.startswith("graphene://") and "middleauth+" not in seg else seg
    state = {
        "layers": [
            {"type": "image", "source": img, "name": "img"},
            {"type": "segmentation", "source": seg_url,
             "segments": [str(r) for r in lam_latest], "name": "lamina_L1-L5"},
        ],
        "selectedLayer": {"layer": "lamina_L1-L5", "visible": True},
        "layout": "xy-3d",
    }
    url = site.rstrip("/") + "/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    with open(os.path.join(OUT, "lamina_ngl_url.txt"), "w") as f:
        f.write(url + "\n")
    print(f"\n[3] lamina neuroglancer URL ({len(lam_latest)} segs) -> figures/lamina_ngl_url.txt")
    print("    seg source:", seg_url[:70], "...")

    # ---- (2) MeTu output-synapse AOTU zone clustering ----
    t = throttle(t)
    syn = client.materialize.query_table(
        "synapses_v2", filter_in_dict={"pre_pt_root_id": metu}, split_positions=True)
    syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
    pos = syn[["ctr_pt_position_x", "ctr_pt_position_y", "ctr_pt_position_z"]].astype(float).values / 1000.0  # um
    syn = syn.assign(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2])
    # restrict to the AOTU core (dense region of MeTu output): within 60 um of global median
    gmed = np.median(pos, axis=0)
    d = np.linalg.norm(pos - gmed, axis=1)
    core = syn[d < 60]
    print(f"\n[2] MeTu output synapses: {len(syn):,} | in AOTU core(<60um): {len(core):,} "
          f"({len(core)/len(syn)*100:.0f}%)")
    cen = core.groupby("pre_pt_root_id")[["x", "y", "z"]].median()
    cen = cen[cen.index.isin(metu)]
    print(f"MeTu with AOTU output centroid: {len(cen)} | AOTU extent um: "
          f"{(cen.max() - cen.min()).round(0).to_dict()}")

    Z = linkage(cen.values, method="ward")
    for k in (3, 4, 6):
        sizes = sorted(pd.Series(fcluster(Z, k, "maxclust")).value_counts().tolist(), reverse=True)
        print(f"  AOTU-zone clusters k={k}: {sizes}")
    cl = fcluster(Z, 4, "maxclust")

    # scatter of AOTU centroids colored by zone-cluster (2 views)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (a, b, an, bn) in zip(axes, [("x", "y", "x", "y"), ("x", "z", "x", "z")]):
        sc = ax.scatter(cen[a], cen[b], c=cl, cmap="tab10", s=18)
        ax.set_xlabel(an + " (um)"); ax.set_ylabel(bn + " (um)"); ax.set_aspect("equal")
    fig.suptitle("Aedes MeTu AOTU output-synapse centroids, colored by zone-cluster (k=4)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "metu_aotu_zones.png"), dpi=150)
    pd.DataFrame({"root_id": cen.index.astype("int64"), "zone_k4": cl,
                  "x": cen["x"].values, "y": cen["y"].values, "z": cen["z"].values}
                 ).to_csv(os.path.join(OUT, "metu_aotu_zones.csv"), index=False)
    print("  saved metu_aotu_zones.png + .csv")


if __name__ == "__main__":
    main()
