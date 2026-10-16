"""Confirm the L1->Mi1 anchor in fly (is it as strong/clean as Dm2->MeTu1?).
mCNS via neuPrint; FAFB via fafbseg get_adjacency (right hemisphere). Read-only.
"""
import time
import pandas as pd
from neuprint import NeuronCriteria as NC, fetch_simple_connections
from fafbseg import flywire
from faab.clients.neuprint import client, MALE_CNS

pd.set_option("display.width", 140)


def summarize(pre, post, w, name):
    per = post.groupby(post.name)  # placeholder; computed below
    print(f"\n=== {name} L1 -> Mi1 ===")


def mcns():
    c = client(MALE_CNS)
    conn = fetch_simple_connections(NC(type="L1"), NC(type="Mi1"), client=c)
    w = conn["weight"]
    per = conn.groupby("bodyId_post")["bodyId_pre"].nunique()
    print("\n=== male-CNS L1 -> Mi1 ===")
    print(f"connections {len(conn)} | distinct L1 {conn['bodyId_pre'].nunique()} | Mi1 {conn['bodyId_post'].nunique()}")
    print(f"synapses/connection: median {w.median():.0f} mean {w.mean():.1f} range {w.min()}-{w.max()}")
    print(f"L1 inputs per Mi1: mean {per.mean():.1f} range {per.min()}-{per.max()}")


def fafb():
    l1 = flywire.search_annotations("L1", exact=True, dataset="public", verbose=False)
    mi1 = flywire.search_annotations("Mi1", exact=True, dataset="public", verbose=False)
    l1r = l1[(l1["cell_type"] == "L1") & (l1["side"] == "right")]["root_id"].astype("int64").tolist()
    mi1r = mi1[(mi1["cell_type"] == "Mi1") & (mi1["side"] == "right")]["root_id"].astype("int64").tolist()
    print(f"\nFAFB right: L1={len(l1r)} Mi1={len(mi1r)} -> adjacency ...")
    e = flywire.get_adjacency(l1r, mi1r, square=False, dataset="public", progress=False)
    e = e[e["weight"] > 0]
    w = e["weight"].astype(float)
    per = e.groupby("target")["source"].nunique()
    print("=== FAFB L1 -> Mi1 (right) ===")
    print(f"connections {len(e)} | distinct L1 {e['source'].nunique()} | Mi1 {e['target'].nunique()}")
    print(f"synapses/connection: median {w.median():.0f} mean {w.mean():.1f} range {int(w.min())}-{int(w.max())}")
    print(f"L1 inputs per Mi1: mean {per.mean():.1f} range {per.min()}-{per.max()}")


if __name__ == "__main__":
    mcns()
    fafb()
    print("\n(compare: Dm2->MeTu1 was 'many weak' = ~30-43 inputs, median 4-5 syn)")
