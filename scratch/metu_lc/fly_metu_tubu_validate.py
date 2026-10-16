"""(a) Validate the TuBu-connectivity subtyping idea in fly: cluster MeTu by their
MeTu->TuBu connectivity profile and check if it recovers the known MeTu1-4 labels
(where morphology NBLAST failed at ARI ~0.10). male-CNS via neuPrint.
"""
import re
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.metrics import adjusted_rand_score

from neuprint import NeuronCriteria as NC, fetch_neurons, fetch_simple_connections
from faab.clients.neuprint import client, MALE_CNS


def major(t):
    m = re.match(r"(MeTu\d)", str(t))
    return m.group(1) if m else str(t)


def main():
    c = client(MALE_CNS)
    mt, _ = fetch_neurons(NC(type="MeTu.*", regex=True), client=c)
    if "instance" in mt.columns:
        mt = mt[mt["instance"].astype(str).str.endswith("_R")]      # right hemisphere
    print(f"male-CNS MeTu (right): {len(mt)}")

    conn = fetch_simple_connections(NC(type="MeTu.*", regex=True),
                                    NC(type="TuBu.*", regex=True), client=c)
    conn = conn[conn["bodyId_pre"].isin(set(mt["bodyId"]))]
    mat = conn.pivot_table(index="bodyId_pre", columns="type_post",
                           values="weight", aggfunc="sum", fill_value=0)
    print(f"MeTu with TuBu output: {mat.shape[0]} | distinct TuBu types: {mat.shape[1]}")

    id2maj = dict(zip(mt["bodyId"], mt["type"].map(major)))
    labels = [id2maj[i] for i in mat.index]
    X = mat.values.astype(float)
    X = X / (X.sum(1, keepdims=True) + 1e-9)            # fraction of TuBu output per type
    Z = linkage(pdist(X, metric="cosine"), method="ward")
    k = len(set(labels))
    cl = fcluster(Z, k, "maxclust")
    ari = adjusted_rand_score(labels, cl)
    print(f"\n=== MeTu->TuBu connectivity clustering ===")
    print(f"ARI (MeTu1-4, k={k}): {ari:.3f}   [morphology NBLAST was ~0.10 = random]")
    print(pd.crosstab(pd.Series(labels, name="true"), pd.Series(cl, name="cluster")).to_string())


if __name__ == "__main__":
    main()
