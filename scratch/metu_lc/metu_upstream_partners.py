"""Presynaptic (upstream) partners of MeTu cells in the aedes brain.

MeTu == aedes_main.subclass == 'MeTu' (medulla -> anterior optic tubercle).
Read-only: SeaTable for metadata, one CAVE synapse query (throttled).

    .venv/bin/python faab/metu_upstream_partners.py
"""
import time

import pandas as pd
import caveclient
import seaserpent as ss

from faab.clients.cave import throttle, DATASTACK

SERVER = "https://flytable.mrc-lmb.cam.ac.uk/"
# Exclude only clearly non-usable segments; keep partially-traced real neurons.
BAD_STATUS = {"duplicate", "fragment", "tiny", "bad_nucleus", "not_a_neuron"}
ANNO_COLS = ["superclass", "class", "subclass", "type", "side"]


def to_int64(s):
    """Parse 18-digit root_ids without float precision loss -> nullable Int64."""
    def f(x):
        try:
            return int(str(x))
        except Exception:
            return pd.NA
    return s.map(f).astype("Int64")


def main():
    # ---- 1) metadata (SeaTable, read-only) ----
    meta = ss.Table("aedes_main", server=SERVER).to_frame()
    meta.columns = [str(c) for c in meta.columns]
    meta["rid"] = to_int64(meta["root_id"])
    meta = meta[meta["rid"].notna()]

    metu = meta[meta["subclass"] == "MeTu"]
    metu_good = metu[~metu["status"].isin(BAD_STATUS)]
    metu_roots = metu_good["rid"].astype("int64").unique().tolist()
    print(f"MeTu rows: {len(metu)} | usable: {len(metu_good)} | unique roots: {len(metu_roots)}")
    print("MeTu by side:", metu_good["side"].value_counts().to_dict())

    # root_id -> annotation map (dedupe roots; keep first)
    root2 = (meta.drop_duplicates("rid").set_index(meta.drop_duplicates("rid")["rid"]
             .astype("int64"))[ANNO_COLS])

    # ---- 2) currency check + upstream synapse query (CAVE, read-only, throttled) ----
    client = caveclient.CAVEclient(DATASTACK)
    t = time.time()
    is_latest = client.chunkedgraph.is_latest_roots(metu_roots)
    print(f"MeTu roots current at HEAD: {int(sum(is_latest))}/{len(metu_roots)} "
          f"(mat v{client.materialize.most_recent_version()})")

    t = throttle(t)
    syn = client.materialize.query_table(
        "synapses_v2", filter_in_dict={"post_pt_root_id": metu_roots}, split_positions=True)
    syn = syn[syn["pre_pt_root_id"] != syn["post_pt_root_id"]]
    print(f"\nupstream synapses onto MeTu: {len(syn):,} | "
          f"distinct presynaptic partners: {syn['pre_pt_root_id'].nunique():,}")

    # ---- 3) partner-level counts + annotation ----
    partners = syn.groupby("pre_pt_root_id").size().rename("syn").reset_index()
    partners = partners.join(root2, on="pre_pt_root_id")
    ann = int(partners["class"].notna().sum())
    print(f"partners annotated in aedes_main: {ann}/{len(partners)} "
          f"({ann / len(partners) * 100:.1f}%)")

    # ---- 4) upstream input summarized BY partner type ----
    pd.set_option("display.width", 160)
    for col in ["class", "subclass"]:
        g = (partners.assign(k=partners[col].fillna("(unannotated)"))
             .groupby("k").agg(synapses=("syn", "sum"),
                               n_partners=("pre_pt_root_id", "nunique"))
             .sort_values("synapses", ascending=False))
        print(f"\n=== upstream input by partner {col} (top 20) ===")
        print(g.head(20).to_string())

    print("\n=== top 25 individual presynaptic partners ===")
    top = partners.sort_values("syn", ascending=False).head(25).copy()
    top["pre_pt_root_id"] = top["pre_pt_root_id"].astype(str)
    print(top[["pre_pt_root_id", "syn"] + ANNO_COLS].to_string(index=False))

    out = "metu_upstream_partners.csv"
    partners.sort_values("syn", ascending=False).to_csv(out, index=False)
    print(f"\nwrote partner table -> faab/{out}")


if __name__ == "__main__":
    main()
