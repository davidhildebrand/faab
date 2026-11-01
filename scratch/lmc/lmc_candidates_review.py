"""Review of lmc_candidates.py (2026-11-01): the known LMCs it missed (their photoreceptor synapses in the lamina, from
the local snapshot), the input/output profile of known LMCs per type vs other SeaTable cells, and how the cells not
in SeaTable spread over lamina-input x medulla-output bins (to set the LMC-like rule).
Network-free apart from the snapshot catch-up. Usage: lmc_candidates_review.py SEATABLE.tsv RUN"""
import os
import sys

import numpy as np
import pandas as pd
import caveclient

from faab import synsnap
from faab.clients.cave import DATASTACK
from faab.config import fig

S = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
T = pd.read_csv(fig("lmc_candidates.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
known = S[S["class"] == "LMC"].copy()
known["label"] = known[["subclass", "subsubclass", "type"]].agg(" ".join, axis=1).str.split().str[-1].fillna("")
missed = known[~known.root_now.isin(set(T.root))]
snap = synsnap.Snapshot(log=lambda *_: None)
e = snap.edges(post=[int(r) for r in missed.root_now], when="now", client=caveclient.CAVEclient(DATASTACK))
od = fig(f"ommatidia/{sys.argv[2]}")
tab = pd.read_csv(os.path.join(od, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
prs = set(tab.root_now)
e["pr"] = e.pre.astype(str).isin(prs)
n_in = e.groupby("post").n.sum()
n_pr = e[e.pr].groupby("post").n.sum()
print(f"known LMCs missed by the photoreceptor net: {len(missed)}")
for x in missed.itertuples(index=False):
    r = int(x.root_now) if x.root_now else 0
    print(f"  {x.root_now} {x.side} {x.label:6s} {x.status:10s} inputs {int(n_in.get(r, 0)):5d}, from listed "
          f"photoreceptors {int(n_pr.get(r, 0))}; notes: {x.notes[:60]}")
cols = ["n_pr", "n_in_LA", "n_out_LA", "n_in_ME", "n_out_ME"]
lab = dict(zip(known.root_now, known.label))
K = T[T.root.isin(set(known.root_now))].assign(label=lambda d: d.root.map(lab))
q = lambda s: f"{int(s.quantile(0.1))}/{int(s.median())}/{int(s.quantile(0.9))}"
print("\nknown LMCs caught, p10/p50/p90 per type:")
print(K.groupby("label")[cols].agg(q).to_string())
O = T[(T.seatable != "") & ~T.root.isin(set(known.root_now))]
print("\nother SeaTable cells caught (p10/p50/p90):")
print(O.groupby(O.seatable.str.split(" / ").str[0])[cols].agg(q).to_string())
N = T[T.seatable == ""]
bins_in = pd.cut(N.n_in_LA, [-1, 9, 19, 49, 99, 1e9], labels=["<10", "10-19", "20-49", "50-99", ">=100"])
bins_out = pd.cut(N.n_out_ME, [-1, 0, 4, 19, 99, 1e9], labels=["0", "1-4", "5-19", "20-99", ">=100"])
print(f"\ncells not in SeaTable: {len(N)} ({N.side.value_counts().to_dict()}); lamina inputs (rows) x medulla outputs (cols):")
print(pd.crosstab(bins_in, bins_out).to_string())
print("\nsame, cells with >= 20 photoreceptor synapses:")
M = N[N.n_pr >= 20]
print(pd.crosstab(bins_in[M.index], bins_out[M.index]).to_string())
