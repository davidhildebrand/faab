"""Build a lineage lookup for the 78 cached downstream partners in
figures/faab_pr_top10_xyz.tsv, so any hand-edited root id from the tracking sheet can be
mapped back to its original row.

  build (default):  one get_lineage_graph per cached id (read-only, 5 s spacing) ->
                    figures/pr_partner_lineage.json  {cached_id: [every root id in lineage]}
  match <file>:     file with one root id per line (as pasted from the sheet) ->
                    prints  sheet_id -> cached downstream id (PR, rank), or 'no match'
"""
from faab.config import REPO
import json
import os
import sys
import time

import pandas as pd
import caveclient
from faab.clients.cave import DATASTACK, throttle

OUT = os.path.join(REPO, "figures")
TSV = os.path.join(OUT, "faab_pr_top10_xyz.tsv")
LIN = os.path.join(OUT, "pr_partner_lineage.json")


def build():
    df = pd.read_csv(TSV, sep="\t", dtype=str)
    cached = sorted({int(x) for x in df.downstream})
    client = caveclient.CAVEclient(DATASTACK)
    lin = json.load(open(LIN)) if os.path.exists(LIN) else {}
    t = throttle(time.time())
    for i, cid in enumerate(cached, 1):
        t = throttle(t)
        try:
            g = client.chunkedgraph.get_lineage_graph(cid, as_nx_graph=True)
            lin[str(cid)] = sorted(int(n) for n in g.nodes)
        except Exception as e:
            print(f"  {cid}: lineage failed ({str(e)[:50]})", flush=True)
            lin[str(cid)] = [cid]
        if i % 10 == 0 or i == len(cached):
            print(f"  {i}/{len(cached)} lineages", flush=True)
            json.dump(lin, open(LIN, "w"))
    json.dump(lin, open(LIN, "w"))
    n = sum(len(v) for v in lin.values())
    print(f"saved {LIN}: {len(lin)} cached ids, {n} lineage roots in total\nDONE")


def match(path):
    df = pd.read_csv(TSV, sep="\t", dtype=str)
    where = {}
    for _, r in df.iterrows():
        where.setdefault(r["downstream"], []).append((r["upstream_PR"], r["#"]))
    lin = json.load(open(LIN))
    owner = {}
    for cid, nodes in lin.items():
        for n in nodes:
            owner.setdefault(str(n), set()).add(cid)
    ids = [ln.strip() for ln in open(path) if ln.strip().isdigit()]
    for sid in ids:
        hits = sorted(owner.get(sid, []))
        if not hits:
            print(f"{sid} -> no match in any cached lineage")
        for cid in hits:
            rows = ", ".join(f"PR {pr} #{rk}" for pr, rk in where[cid])
            print(f"{sid} -> cached {cid}  ({rows})")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "match":
        match(sys.argv[2])
    else:
        build()
