"""LMC typing data (David, 2026-10-31: check the L1..L5 groups and subdivide LRS's 'L1-L5' cells). From a
seatable_fetch TSV: every class-LMC cell, its L2 skeleton (um; cached in figures/lmc/lmc_skeletons.pkl, reusing
figures/glia/fragment_skeletons.pkl) and its synapses with positions (local snapshot caught up to now, faab.synsnap).
CAVE read-only: one get_l2_skeleton per missing cell, >= 2 s apart, plus the snapshot catch-up.
Usage: lmc_data.py SEATABLE.tsv [--extra ROOT:SIDE[:NOTE] ...]  (extra = cells not typed LMC in SeaTable, to be called)"""
import argparse
import os
import pickle
import re
import time

import numpy as np
import pandas as pd

from faab import synsnap
from faab.clients.cave import client as cave_client, throttle
from faab.config import DATASTACK, VOX, fig

ap = argparse.ArgumentParser()
ap.add_argument("seatable")
ap.add_argument("--extra", nargs="*", default=[])
args = ap.parse_args()
S = pd.read_csv(args.seatable, sep="\t", dtype=str).fillna("")
typed = S[["subclass", "subsubclass", "type"]].agg(" ".join, axis=1)
# class LMC, plus optic-lobe cells typed L1..L5 without the class (2 'ol_intrinsic / visual / L5'); the central-complex
# EPG/PEG/PEN/PGE rows carry protocerebral-bridge glomeruli L1..L5 and are not LMCs
L = S[(S["class"] == "LMC") | ((S.superclass == "ol_intrinsic") & typed.str.contains(r"\bL[1-5]\b"))].copy()
lab = L[["subclass", "subsubclass", "type"]].agg(" ".join, axis=1)
L["label"] = [("L1-L5" if re.search(r"\bL1\s*-\s*L?5\b", t) else
               (re.findall(r"\bL[1-5]\b", t) or [""])[0]) for t in lab]
L = L.rename(columns={"root_now": "root"})[["serial_id", "root", "side", "label", "status", "proofreader", "point_xyz", "notes"]]
L = L.drop_duplicates("root")
for e in args.extra:
    r, sd, *note = e.split(":")
    if r not in set(L.root):
        L.loc[len(L) + 10 ** 6] = dict(serial_id="", root=r, side=sd, label="", status="", proofreader="", point_xyz="",
                                       notes="extra: " + ":".join(note))
L.to_csv(fig("lmc_cells.tsv"), sep="\t", index=False)
print(f"{len(L)} LMC cells; labels {L.label.value_counts().to_dict()}; sides {L.side.value_counts().to_dict()}")
print(pd.crosstab(L.label, L.side).to_string())

# ---- L2 skeletons (um)
import navis
from fafbseg import flywire
navis.set_pbars(hide=True)
path = fig("lmc_skeletons.pkl")
sk = pickle.load(open(path, "rb")) if os.path.exists(path) else {}
old = fig("fragment_skeletons.pkl")
if os.path.exists(old):
    for r, s in pickle.load(open(old, "rb")).items():
        if str(r) in set(L.root) and str(r) not in sk:
            sk[str(r)] = s
todo = [r for r in L.root if r not in sk]
print(f"skeletons cached {len(L) - len(todo)}; to fetch {len(todo)} (~{len(todo) * 3 / 60:.0f} min)", flush=True)
t = 0.0
for i, r in enumerate(todo):
    t = throttle(t)
    try:
        sk[r] = flywire.get_l2_skeleton(int(r), dataset=DATASTACK, progress=False, max_threads=1) / 1000.0
    except Exception as ex:
        print(f"  {r}: {str(ex)[:80]}", flush=True)
    t = time.time()
    if (i + 1) % 20 == 0 or i + 1 == len(todo):
        pickle.dump(sk, open(path, "wb"))
        print(f"  {i + 1}/{len(todo)} fetched", flush=True)
print("nodes per skeleton:", pd.Series({r: len(s.nodes) for r, s in sk.items() if r in set(L.root)}).describe().round(0).to_dict())

# ---- synapses (positions um), self-contacts removed
c = cave_client()
snap = synsnap.Snapshot(log=lambda *_: None)
ids = [int(r) for r in L.root]
parts = []
for kw, who in (({"pre": ids}, "pre"), ({"post": ids}, "post")):
    d = snap.synapses(**kw, when="now", positions=True, client=c)
    d = d[d.pre_root != d.post_root]
    d["lmc"] = d[f"{who}_root"].astype(str)
    d["role"] = "out" if who == "pre" else "in"
    w = "pre" if who == "pre" else "post"
    d[["x", "y", "z"]] = d[[f"{w}_x", f"{w}_y", f"{w}_z"]].to_numpy(float) * np.array(VOX) / 1000.0
    parts.append(d[["id", "lmc", "role", "pre_root", "post_root", "x", "y", "z"]])
syn = pd.concat(parts, ignore_index=True)
syn.to_parquet(fig("lmc_synapses.parquet"))
print(f"synapses: {syn.groupby('role').size().to_dict()}; written {fig('lmc_synapses.parquet')}")
