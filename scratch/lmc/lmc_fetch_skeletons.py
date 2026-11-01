"""Fetch missing L2 skeletons (um) into figures/lmc/lmc_skeletons.pkl for the roots of a table (2026-11-01: the L5
search candidates, for their points on the primary neurite). CAVE read-only, one call per root >= 2 s apart; failures
are listed and can be retried by running again. Usage: lmc_fetch_skeletons.py TABLE.tsv [--l5]  (--l5: only rows with
the L5 profile that are not known L5s, as in lmc_l5_candidates.tsv)"""
import os
import pickle
import sys
import time

import navis
import pandas as pd
from fafbseg import flywire

from faab.clients.cave import DATASTACK, throttle
from faab.config import fig

navis.set_pbars(hide=True)
T = pd.read_csv(sys.argv[1], sep="\t", dtype={"root": str})
if "--l5" in sys.argv:
    T = T[T.l5_profile & ~T.known_l5]
path = fig("lmc_skeletons.pkl")
sk = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
todo = [r for r in T.root if r not in sk]
print(f"{len(T)} roots; skeletons cached {len(T) - len(todo)}; to fetch {len(todo)} (~{len(todo) * 3 / 60:.0f} min)", flush=True)
t, failed = 0.0, []
for i, r in enumerate(todo):
    t = throttle(t)
    try:
        sk[r] = flywire.get_l2_skeleton(int(r), dataset=DATASTACK, progress=False, max_threads=1) / 1000.0
    except Exception as ex:
        failed.append(r)
        print(f"  {r}: {str(ex)[:80]}", flush=True)
    t = time.time()
    if (i + 1) % 25 == 0 or i + 1 == len(todo):
        pickle.dump(sk, open(path, "wb"))
        print(f"  {i + 1}/{len(todo)} fetched", flush=True)
print(f"done; failed {len(failed)}: {' '.join(failed)}")
