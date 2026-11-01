"""Spelunker links (README scene base) for the LMC typing (lmc_typing.tsv, lmc_cartridge_calls.tsv):
  calls     the called cells by putative type (cartridge assignment where there is one, else the consensus), one layer
            per type, plus the SeaTable-typed cells per type as hidden comparison layers
  outliers  typed cells whose consensus differs from their label, with the SeaTable cells of both types (hidden)
Network-free; upload with scripts/tools/ng_short_link.py. Usage: lmc_links.py STAMP"""
import sys

import numpy as np
import pandas as pd

from faab import nglink
from faab.config import fig

TYPES = ["L1", "L2", "L3", "L4", "L5"]
COL = {"L1": "#e6194b", "L2": "#3cb44b", "L3": "#4363d8", "L4": "#f58231", "L5": "#911eb4"}
stamp = sys.argv[1]
T = pd.read_csv(fig("lmc_typing.tsv"), sep="\t", dtype={"root": str, "serial_id": str}).fillna("")
K = pd.read_csv(fig("lmc_cartridge_calls.tsv"), sep="\t", dtype={"root": str}).fillna("")
call = dict(zip(T.root, T.consensus))
call.update({r: a for r, a in zip(K.root, K.assigned) if a})
U = T[~T.label.isin(TYPES)]
OL = [{"kind": "OL", "segments": ["11", "14", "21", "24"], "alpha": 0.12}]
neurons = []
for t in TYPES:
    ids = [r for r in U.root if call.get(r) == t]
    if ids:
        neurons.append({"name": f"putative {t} ({len(ids)})", "color": COL[t], "ids": ids})
for t in TYPES:
    ids = T[T.label == t].root.tolist()
    neurons.append({"name": f"SeaTable {t} ({len(ids)})", "color": COL[t], "ids": ids, "visible": False})
spec = {"meshes": OL, "neurons": neurons, "layout": "3d", "select": neurons[0]["name"]}
out = fig(f"{stamp}_lmc_calls_ng.txt")
open(out, "w").write(nglink.to_url(nglink.build(spec)) + "\n")
print("written", out)

bad = T[(T.label.isin(TYPES)) & (T.consensus != T.label)]
if len(bad):
    types = sorted(set(bad.label) | set(bad.consensus))
    neurons = [{"name": f"SeaTable {l}, typed {c} here ({len(g)})", "color": "#ff00ff", "ids": g.root.tolist()}
               for (l, c), g in bad.groupby(["label", "consensus"])]
    for t in types:
        ids = [r for r in T[T.label == t].root if r not in set(bad.root)]
        neurons.append({"name": f"SeaTable {t} ({len(ids)})", "color": COL[t], "ids": ids, "visible": False})
    xyz = np.array([[float(v) for v in s.split(",")] for s in bad.point_xyz if s])
    spec = {"meshes": OL, "neurons": neurons, "layout": "xy-3d", "select": neurons[0]["name"], "scale": 9000}
    if len(xyz):
        spec["position"] = xyz[0].tolist()
    out = fig(f"{stamp}_lmc_outliers_ng.txt")
    open(out, "w").write(nglink.to_url(nglink.build(spec)) + "\n")
    print("written", out)
