"""Build the Dm2-candidate pool (a) and fetch MeTu morphologies for subtyping (b).

Connectivity-based candidate selection (NO completeness filter): presynaptic
partners of MeTu whose connection pattern matches the fly Dm2->MeTu1 signature
(converges onto several MeTu, few synapses per connection, moderate total).
The MeTu these candidates target = the MeTu1-equivalent aedes subset.

Read-only. CAVE calls serialized + throttled (>=5 s, max_threads=1). Saves
dotprops caches + candidate table for downstream local clustering.
"""
from faab.config import REPO
import os
import time
import pickle

import numpy as np
import pandas as pd
import caveclient
import seaserpent as ss
from fafbseg import flywire

from faab.clients.cave import throttle, DATASTACK
from faab.config import fig

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
OUT = os.path.join(REPO, "figures")
ANNO = ["superclass", "class", "subclass", "type"]
BAD = {"duplicate", "fragment", "tiny", "bad_nucleus", "not_a_neuron"}  # bad-id hygiene only, not completeness
MAX_CAND = 400


def to_int64(s):
    def f(x):
        try:
            return int(str(x))
        except Exception:
            return pd.NA
    return s.map(f).astype("Int64")


def dps_um(dp):
    dp = dp / 1000.0                      # L2 dotprops nm -> microns
    for d in dp:
        d.units = "um"
    return dp


def main():
    # ---- MeTu ids + annotation map (SeaTable) ----
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    meta["rid"] = to_int64(meta["root_id"])
    meta = meta[meta["rid"].notna()]
    metu = meta[(meta["subclass"] == "MeTu") & ~meta["status"].isin(BAD)]
    metu_roots = metu["rid"].astype("int64").unique().tolist()
    root2 = (meta.drop_duplicates("rid")
             .set_index(meta.drop_duplicates("rid")["rid"].astype("int64"))[ANNO])
    print("MeTu:", len(metu_roots))

    # ---- upstream synapses onto MeTu -> per-(partner,MeTu) connection table ----
    client = caveclient.CAVEclient(DATASTACK)
    t = time.time()
    syn = client.materialize.query_table(
        "synapses_v2", filter_in_dict={"post_pt_root_id": metu_roots}, split_positions=True)
    syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
    conn = (syn.groupby(["pre_pt_root_id", "post_pt_root_id"]).size()
            .rename("w").reset_index())
    g = (conn.groupby("pre_pt_root_id")
         .agg(n_metu=("post_pt_root_id", "nunique"), total=("w", "sum"),
              med_w=("w", "median"), max_w=("w", "max")).reset_index())
    g = g.join(root2, on="pre_pt_root_id")
    g["annotated"] = g[ANNO].notna().any(axis=1)

    # Dm2 connectivity signature (fly: ~6 MeTu1/Dm2, median ~4-5 syn, total ~30)
    sig = g[(~g["annotated"]) & (g["n_metu"] >= 3) & (g["med_w"] <= 10)
            & (g["max_w"] <= 25) & g["total"].between(15, 80)]
    sig = sig.sort_values(["n_metu", "total"], ascending=False)
    print(f"unannotated partners: {(~g['annotated']).sum()} | Dm2-signature candidates: {len(sig)}")
    sig.to_csv(fig("dm2_candidates.csv"), index=False)

    cand_ids = sig["pre_pt_root_id"].astype("int64").head(MAX_CAND).tolist()
    # MeTu targeted by these candidates = MeTu1-equivalent subset
    tgt = conn[conn["pre_pt_root_id"].isin(cand_ids)]["post_pt_root_id"].value_counts()
    tgt.rename("n_candidate_inputs").to_csv(fig("metu1_equiv_targets.csv"))
    print(f"MeTu targeted by candidates (MeTu1-equiv subset): {len(tgt)} / {len(metu_roots)}")

    # ---- fetch MeTu dotprops (b) ----
    t = throttle(t)
    metu_dp = dps_um(flywire.get_l2_dotprops(metu_roots, dataset=DATASTACK,
                                             max_threads=1, omit_failures=True, progress=False))
    pickle.dump((metu_dp, metu_roots), open(fig("metu_dotprops.pkl"), "wb"))
    print("fetched MeTu dotprops:", len(metu_dp))

    # ---- fetch candidate dotprops (a) ----
    t = throttle(t)
    cand_dp = dps_um(flywire.get_l2_dotprops(cand_ids, dataset=DATASTACK,
                                             max_threads=1, omit_failures=True, progress=False))
    pickle.dump((cand_dp, cand_ids), open(fig("candidate_dotprops.pkl"), "wb"))
    spans = [float((d.points.max(0) - d.points.min(0)).max()) for d in cand_dp]
    print(f"fetched candidate dotprops: {len(cand_dp)} | span um median {np.median(spans):.0f} "
          f"| >150um (large, drop later): {sum(s > 150 for s in spans)}")


if __name__ == "__main__":
    main()
