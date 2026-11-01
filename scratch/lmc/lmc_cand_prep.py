"""Data for subtyping the LMC search candidates (2026-11-01): the LMC-like cells of figures/lmc/lmc_candidates.tsv
(>= MIN_LA inputs in the lamina, >= MIN_ME outputs in the medulla of their side, at most MAX_LA lamina inputs - the
wide-field visual projection cells have ~1,500), minus cells already in the typing set (lmc_cells.tsv). Their L2
skeletons (um; cache figures/lmc/lmc_skeletons.pkl shared with lmc_data.py) and synapses with positions (local
snapshot caught up to now) -> figures/lmc/lmc_cand_cells.tsv and lmc_cand_synapses.parquet, the inputs of
lmc_features.py --set cand. CAVE read-only: one get_l2_skeleton per missing cell, >= 2 s apart, plus the snapshot
catch-up. Usage: lmc_cand_prep.py [--limit N]"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

from faab import synsnap
from faab.clients.cave import DATASTACK, client as cave_client, throttle
from faab.config import VOX, fig

MIN_LA, MIN_ME, MAX_LA = 20, 20, 800
T = pd.read_csv(fig("lmc_candidates.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
done = set(pd.read_csv(fig("lmc_cells.tsv"), sep="\t", dtype=str).root)
C = T[(T.n_in_LA >= MIN_LA) & (T.n_out_ME >= MIN_ME) & (T.n_in_LA <= MAX_LA) & ~T.root.isin(done)].copy()
if "--limit" in sys.argv:
    C = C.head(int(sys.argv[sys.argv.index("--limit") + 1]))
C["label"] = C.seatable.str.extract(r"\b(L[1-5](?:-L5)?)$", expand=False).fillna("")
cells = C.rename(columns={"seatable": "notes"})[["serial_id", "root", "side", "label", "notes"]].assign(
    status="", proofreader="", point_xyz="")
cells.to_csv(fig("lmc_cand_cells.tsv"), sep="\t", index=False)
print(f"{len(cells)} candidates ({cells.side.value_counts().to_dict()}); SeaTable labels "
      f"{cells.label.replace('', 'none').value_counts().to_dict()}", flush=True)

import navis
from fafbseg import flywire
navis.set_pbars(hide=True)
path = fig("lmc_skeletons.pkl")
sk = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
todo = [r for r in cells.root if r not in sk]
print(f"skeletons cached {len(cells) - len(todo)}; to fetch {len(todo)} (~{len(todo) * 3 / 60:.0f} min)", flush=True)
t = 0.0
for i, r in enumerate(todo):
    t = throttle(t)
    try:
        sk[r] = flywire.get_l2_skeleton(int(r), dataset=DATASTACK, progress=False, max_threads=1) / 1000.0
    except Exception as ex:
        print(f"  {r}: {str(ex)[:80]}", flush=True)
    t = time.time()
    if (i + 1) % 25 == 0 or i + 1 == len(todo):
        pickle.dump(sk, open(path, "wb"))
        print(f"  {i + 1}/{len(todo)} fetched", flush=True)

snap, ids, parts = synsnap.Snapshot(log=lambda *_: None), [int(r) for r in cells.root], []
for kw, who in (({"pre": ids}, "pre"), ({"post": ids}, "post")):
    d = snap.synapses(**kw, when="now", positions=True, client=cave_client())
    d = d[d.pre_root != d.post_root]
    d["lmc"] = d[f"{who}_root"].astype(str)
    d["role"] = "out" if who == "pre" else "in"
    d[["x", "y", "z"]] = d[[f"{who}_x", f"{who}_y", f"{who}_z"]].to_numpy(float) * np.array(VOX) / 1000.0
    parts.append(d[["id", "lmc", "role", "pre_root", "post_root", "x", "y", "z"]])
syn = pd.concat(parts, ignore_index=True)
syn.to_parquet(fig("lmc_cand_synapses.parquet"))
print(f"synapses: {syn.groupby('role').size().to_dict()}; written {fig('lmc_cand_synapses.parquet')}")
