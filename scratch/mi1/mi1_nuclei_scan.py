"""Nuclei-table scan for Mi1 candidates (dataset: aedes). One query.
Validate: how many known Mi1 have a soma. Size the medulla-cortex pool by
distance of every nucleus to the known-Mi1 soma layer. Read-only.
"""
from faab.config import REPO
import os
import time
import pickle
import numpy as np
import pandas as pd
import caveclient
from scipy.spatial import cKDTree
from faab.clients.cave import DATASTACK
from faab.config import fig

OUT = os.path.join(REPO, "figures")


def main():
    client = caveclient.CAVEclient(DATASTACK)
    for attempt in range(5):
        try:
            nuc = client.materialize.query_table("nuclei_v1_aedes", split_positions=True)
            break
        except Exception as e:
            if attempt == 4:
                raise
            print(f"  (retry {attempt+1}/5: {str(e)[:60]})"); time.sleep(6)
    nuc["pt_root_id"] = nuc["pt_root_id"].map(lambda x: int(x) if pd.notna(x) else 0).astype("int64")
    nuc = nuc[nuc["pt_root_id"] > 0]
    pos = nuc[["pt_position_x", "pt_position_y", "pt_position_z"]].astype(float).values / 1000.0  # nm->um
    print(f"nuclei table: {len(nuc)} nuclei with a root")

    mi1 = [int(d.id) for d in pickle.load(open(fig("mi1_dotprops.pkl"), "rb"))]
    mi1set = set(mi1)
    is_mi1 = nuc["pt_root_id"].isin(mi1set).values
    soma = pos[is_mi1]
    print(f"known Mi1 with a soma in the table: {nuc.loc[is_mi1,'pt_root_id'].nunique()} / {len(mi1)}")
    if len(soma):
        print(f"Mi1 soma-layer extent (um): {(soma.max(0)-soma.min(0)).round(0)}")

    # distance of every nucleus to the known-Mi1 soma layer
    tree = cKDTree(soma)
    d, _ = tree.query(pos, k=1)
    print("\nmedulla-cortex pool sizing (nuclei near the Mi1 soma layer):")
    for tol in (5, 10, 15, 20, 30):
        pool = nuc[d < tol]
        print(f"  within {tol:2d} um: {len(pool):5d} nuclei  "
              f"(includes {pool['pt_root_id'].isin(mi1set).sum()} known-Mi1 somata)")

    # save a working pool (20um) for the next stage
    keep = d < 20
    pool_ids = nuc.loc[keep, "pt_root_id"].unique().tolist()
    pickle.dump({"pool_ids": pool_ids,
                 "pool_pos_um": pos[keep]},
                open(fig("mi1_nuclei_pool.pkl"), "wb"))
    print(f"\nsaved pool (<20um): {len(pool_ids)} candidate cells -> mi1_nuclei_pool.pkl")


if __name__ == "__main__":
    main()
