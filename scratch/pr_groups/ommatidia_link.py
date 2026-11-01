"""Spelunker link (README scene base) for one PRs-tab group in an ommatidia.py run (2026-11-01): one layer per version 2
ommatidium holding any of the group's axons (all its members, by slot), and a layer of the group's axons that are in
none of them (with why, from compare.tsv). Network-free; upload with scripts/tools/ng_short_link.py.
Usage: ommatidia_link.py RUN STAMP G"""
import os
import re
import sys

import numpy as np
import pandas as pd

from faab import nglink
from faab.config import fig

run, stamp, g = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = fig(f"ommatidia/{run}")
Om = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
Cmp = pd.read_csv(os.path.join(d, "compare.tsv"), sep="\t").fillna("")
tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
mine = tab[tab.group.astype(int) == g]
x = Cmp[Cmp.group == g].iloc[0]
oms = [o for o in [x.ommatidium] + str(x.other_om).split(", ") if o]
COL = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4"]
neurons, placed = [], set()
for k, om in enumerate(oms):
    m = Om[Om.ommatidium == om]
    placed |= set(m.root)
    slots = ", ".join(f"{(m.slot == s).sum()} {s}" for s in ("R1-R6", "R7", "R8"))
    neurons.append({"name": f"{om} ({slots})", "color": COL[k % len(COL)], "ids": m.root.tolist()})
rest = sorted(set(mine.root_now) - placed - {"", "0"})
if rest:
    neurons.append({"name": f"group {g} axons in none of these ({len(rest)})", "color": "#ffffff", "ids": rest})
xyz = np.array([[float(v) for v in s.split(",")] for s in mine.xyz if s.count(",") == 2])
spec = {"meshes": [{"kind": "OL", "segments": ["11", "14", "21", "24"], "alpha": 0.12}], "neurons": neurons,
        "layout": "xy-3d", "select": neurons[0]["name"], "scale": 9000}
if len(xyz):
    spec["position"] = np.median(xyz, axis=0).tolist()
out = fig(f"{stamp}_ommatidia_g{g}_ng.txt")
open(out, "w").write(nglink.to_url(nglink.build(spec)) + "\n")
print("written", out)
for n in neurons:
    print(f"  {n['name']}: {', '.join(n['ids'])}")
print("why not in an ommatidium:", "; ".join(w for w in re.split(r";\s*", x.lost) if any(r in w for r in rest)) or "-")
