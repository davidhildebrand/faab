"""Does MeTu->TuBu subtyping require TuBu SUBTYPE labels, or do individual TuBu
CELLS suffice? Test both on Drosophila male-CNS (where MeTu1-4 labels are known).
If individual-cell features also recover subtypes -> the FAAB approach (TuBu not
subtyped) is viable. Dataset: male-CNS (neuPrint).
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


def ari_for(mat, labels):
    X = mat.values.astype(float)
    X = X / (X.sum(1, keepdims=True) + 1e-9)
    Z = linkage(pdist(X, metric="cosine"), method="ward")
    k = len(set(labels))
    return adjusted_rand_score(labels, fcluster(Z, k, "maxclust"))


def main():
    c = client(MALE_CNS)
    mt, _ = fetch_neurons(NC(type="MeTu.*", regex=True), client=c)
    mt = mt[mt["instance"].astype(str).str.endswith("_R")]
    conn = fetch_simple_connections(NC(type="MeTu.*", regex=True),
                                    NC(type="TuBu.*", regex=True), client=c)
    conn = conn[conn["bodyId_pre"].isin(set(mt["bodyId"]))]
    id2maj = dict(zip(mt["bodyId"], mt["type"].map(major)))

    print("(male-CNS) MeTu->TuBu subtyping — feature comparison:")
    for col, fname in [("type_post", "TuBu SUBTYPE labels"),
                       ("bodyId_post", "individual TuBu CELLS (no labels)")]:
        mat = conn.pivot_table(index="bodyId_pre", columns=col, values="weight",
                               aggfunc="sum", fill_value=0)
        labels = [id2maj[i] for i in mat.index]
        ari = ari_for(mat, labels)
        print(f"  features = {fname:35s} -> shape {mat.shape}  ARI(MeTu1-4) = {ari:.3f}")
    print("  (morphology NBLAST was ARI ~0.10)")


if __name__ == "__main__":
    main()
