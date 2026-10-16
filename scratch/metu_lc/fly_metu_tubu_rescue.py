"""Can we recover MeTu1-4 WITHOUT TuBu subtype labels, by finding TuBu groups
de novo from the MeTu x TuBu connectivity matrix? Test on Drosophila male-CNS.
If yes -> directly applicable to FAAB (where TuBu are unsubtyped).
Dataset: male-CNS (neuPrint).
"""
import re
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralCoclustering

from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.sources.neuprint import client, MALE_CNS


def major(t):
    m = re.match(r"(MeTu\d)", str(t))
    return m.group(1) if m else str(t)


def cluster_ari(X, labels, k):
    X = X / (X.sum(1, keepdims=True) + 1e-9)
    Z = linkage(pdist(X, metric="cosine"), method="ward")
    return adjusted_rand_score(labels, fcluster(Z, k, "maxclust"))


def main():
    c = client(MALE_CNS)
    mt, _ = fetch_neurons(NC(type="MeTu.*", regex=True), client=c)
    mt = mt[mt["instance"].astype(str).str.endswith("_R")]
    conn = fetch_simple_connections(NC(type="MeTu.*", regex=True),
                                    NC(type="TuBu.*", regex=True), client=c)
    conn = conn[conn["bodyId_pre"].isin(set(mt["bodyId"]))]
    id2maj = dict(zip(mt["bodyId"], mt["type"].map(major)))
    mat = conn.pivot_table(index="bodyId_pre", columns="bodyId_post",
                           values="weight", aggfunc="sum", fill_value=0)
    M = mat.values.astype(float)
    labels = [id2maj[i] for i in mat.index]
    k = len(set(labels))
    print(f"(male-CNS) MeTu x TuBu-cell matrix {M.shape}, true MeTu groups k={k}")

    # rescue A: spectral co-clustering (jointly blocks MeTu rows + TuBu cols)
    co = SpectralCoclustering(n_clusters=k, random_state=0).fit(M)
    print(f"  [A] SpectralCoclustering(k={k}) -> ARI {adjusted_rand_score(labels, co.row_labels_):.3f}")

    # rescue B: cluster TuBu de novo by their MeTu-input profile, then aggregate
    Xt = M.T / (M.T.sum(1, keepdims=True) + 1e-9)
    Zt = linkage(pdist(Xt, metric="cosine"), method="ward")
    for ktubu in (6, 10, 14):
        tg = fcluster(Zt, ktubu, "maxclust")
        agg = np.column_stack([M[:, tg == g].sum(1) for g in range(1, ktubu + 1)])
        print(f"  [B] de-novo TuBu groups={ktubu:2d} -> MeTu ARI {cluster_ari(agg, labels, k):.3f}")

    print("  (reference: TuBu-subtype labels 0.994 | individual cells 0.297 | morphology 0.10)")


if __name__ == "__main__":
    main()
