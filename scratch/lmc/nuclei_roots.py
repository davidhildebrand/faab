"""Nuclei (soma positions) by today's roots (2026-11-01, LMC search: L5 vs C2/C3-like cells by soma side). The nucleus
table copy tmp/nuclei_v1_aedes_v542.tsv (pt_position in nm, pt_supervoxel_id) -> roots now (get_roots, batches of 50k,
>= 2 s apart; read-only) -> figures/lmc/nuclei_roots_now.tsv (nucleus id, root, x/y/z um, volume). A root with several
nuclei is a merge. Usage: nuclei_roots.py"""
import datetime
import os
import time

import numpy as np
import pandas as pd

from faab.clients.cave import client as cave_client, throttle
from faab.config import TMP, fig

N = pd.read_csv(os.path.join(TMP, "nuclei_v1_aedes_v542.tsv"), sep="\t")
N = N[N.valid.astype(str) == "True"].copy()
P = np.array([[float(v) for v in s.strip("[]").split()] for s in N.pt_position]) / 1000.0
N[["x", "y", "z"]] = P
svs = N.pt_supervoxel_id.astype(np.int64).to_numpy()
client, t, roots = cave_client(), 0.0, np.zeros(len(svs), np.int64)
when = datetime.datetime.now(datetime.timezone.utc)
for k in range(0, len(svs), 50000):
    t = throttle(t)
    roots[k:k + 50000] = np.asarray(client.chunkedgraph.get_roots(svs[k:k + 50000], timestamp=when)).astype(np.int64)
    t = time.time()
    print(f"  {min(k + 50000, len(svs))}/{len(svs)} mapped", flush=True)
N["root"] = roots.astype(str)
N["n_nuclei_on_root"] = N.groupby("root").root.transform("size")
N[["id", "root", "x", "y", "z", "volume", "n_nuclei_on_root"]].to_csv(fig("nuclei_roots_now.tsv"), sep="\t", index=False)
print(f"{len(N)} nuclei on {N.root.nunique()} roots at {when:%Y-%m-%d %H:%M} UTC; roots with > 1 nucleus "
      f"{int((N.groupby('root').size() > 1).sum())}; written {fig('nuclei_roots_now.tsv')}")
