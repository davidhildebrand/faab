"""Do the proposed cutouts hold any R1-R6 / R8 synapses? (David 2026-11-01.) For each box of a registry
(propose_targets.py), every synapse inside it (live spatial query on synapses_v2, >= 2 s apart, read-only), and how many
of them are outputs / inputs of the R1-R6 / R8 axons (PRs tab classes R1-R6 / R8 / R1-R6? at the 2026-11-01 roots, and
the column-zone axons with an R1-R6 / R8 synapse verdict). The snapshot catch-up version hung for hours; this one is
14 small queries. Usage: cutouts_pr_check.py REGISTRY.tsv"""
import datetime
import os
import sys
import time

import numpy as np
import pandas as pd

from faab.clients.cave import client as cave_client, throttle
from faab.config import fig
from faab.partners.top10_xyz import VOX

R = pd.read_csv(sys.argv[1], sep="\t")
od = fig("ommatidia/20261101")
tab = pd.read_csv(os.path.join(od, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
F = pd.read_csv(os.path.join(od, "features.tsv"), sep="\t", dtype={"root": str})
prs = set(tab[tab.cls.isin(["R1-R6", "R8", "R1-R6?"])].root_now) | \
    set(F[F.verdict.isin(["R1-R6-like", "R1-R6-like (few sites)", "R8-like"])].root)
prs = {int(r) for r in prs if r not in ("", "0")}

client, now, t = cave_client(), datetime.datetime.now(datetime.timezone.utc), 0.0
for x in R.itertuples(index=False):
    lo = (np.array([float(v) for v in x.lo_xyz.split(",")]) * VOX).tolist()
    hi = (np.array([float(v) for v in x.hi_xyz.split(",")]) * VOX).tolist()
    t = throttle(t)
    a = client.materialize.live_query("synapses_v2", timestamp=now, filter_spatial_dict={"ctr_pt_position": [lo, hi]})
    t = time.time()
    a = a[a.pre_pt_root_id != a.post_pt_root_id]
    out, inp = a.pre_pt_root_id.isin(prs), a.post_pt_root_id.isin(prs)
    print(f"{x.box_id}\t{len(a)} synapses\tR1-R6/R8 outputs {int(out.sum())} from {a[out].pre_pt_root_id.nunique()} axons"
          f"\tinputs {int(inp.sum())} onto {a[inp].post_pt_root_id.nunique()} axons", flush=True)
print(f"({len(prs)} R1-R6/R8 axons at the 2026-11-01 roots)")
