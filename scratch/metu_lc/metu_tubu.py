"""Subtype FAAB (aedes) MeTu by MeTu->TuBu connectivity via spectral co-clustering
(method validated on male-CNS at ARI ~0.94, no TuBu subtype labels needed).
Dataset: aedes (wclee_aedes_brain) + metadata from FlyTable. Read-only.
"""
from faab.config import REPO
import os
import time
import numpy as np
import pandas as pd
import caveclient
import seaserpent as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralCoclustering

from faab.clients.cave import throttle, DATASTACK
from faab.config import fig

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
OUT = os.path.join(REPO, "figures")
BAD = {"duplicate", "fragment", "tiny", "bad_nucleus", "not_a_neuron"}


def to_int64(s):
    return s.map(lambda x: int(str(x)) if str(x).strip().lstrip("-").isdigit() else pd.NA).astype("Int64")


def main():
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    meta["rid"] = to_int64(meta["root_id"])
    meta = meta[meta["rid"].notna()]
    metu = meta[(meta["subclass"] == "MeTu") & ~meta["status"].isin(BAD)]["rid"].astype("int64").tolist()
    tubu = meta[(meta["subclass"] == "TuBu") & ~meta["status"].isin(BAD)]["rid"].astype("int64").tolist()
    print(f"(aedes) MeTu={len(metu)}  TuBu={len(tubu)}")

    client = caveclient.CAVEclient(DATASTACK)
    t = time.time()
    syn = client.materialize.query_table(
        "synapses_v2", filter_in_dict={"pre_pt_root_id": metu, "post_pt_root_id": tubu},
        split_positions=True)
    conn = syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size().rename("w").reset_index()
    mat = conn.pivot_table(index="pre_pt_root_id", columns="post_pt_root_id",
                           values="w", aggfunc="sum", fill_value=0)
    mat = mat.loc[mat.sum(1) > 0, mat.sum(0) > 0]
    M = mat.values.astype(float)
    print(f"(aedes) MeTu->TuBu connectivity matrix: {M.shape} "
          f"({len(metu) - M.shape[0]} MeTu have no TuBu output)")
    print(f"(aedes) total MeTu->TuBu synapses: {int(M.sum())}")

    # block-diagonality (fraction of synapses inside co-clusters) measures clustering quality
    for k in (3, 4, 5, 6):
        co = SpectralCoclustering(n_clusters=k, random_state=0).fit(M)
        within = sum(M[np.ix_(co.row_labels_ == i, co.column_labels_ == i)].sum() for i in range(k))
        sizes = sorted(pd.Series(co.row_labels_).value_counts().tolist(), reverse=True)
        print(f"  k={k}: MeTu cluster sizes {sizes} | within-block synapses {within/M.sum()*100:.0f}%")

    k = 4
    co = SpectralCoclustering(n_clusters=k, random_state=0).fit(M)
    ro = np.argsort(co.row_labels_)
    cocol = np.argsort(co.column_labels_)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(np.log1p(M[np.ix_(ro, cocol)]), aspect="auto", cmap="magma")
    ax.set_xlabel("TuBu (reordered by co-cluster)")
    ax.set_ylabel("MeTu (reordered by co-cluster)")
    ax.set_title("(aedes) MeTu x TuBu connectivity, spectral co-clustered (k=4)")
    fig.tight_layout(); fig.savefig(fig("faab_metu_tubu_coclust.png"), dpi=150)

    pd.DataFrame({"root_id": mat.index.astype("int64"), "metu_tubu_cluster_k4": co.row_labels_}
                 ).to_csv(fig("faab_metu_tubu_clusters.csv"), index=False)
    print("saved faab_metu_tubu_coclust.png + faab_metu_tubu_clusters.csv")


if __name__ == "__main__":
    main()
