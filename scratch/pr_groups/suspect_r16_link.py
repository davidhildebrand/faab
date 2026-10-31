"""Link (README scene base) for R1-R6 candidates with >= 30% of their cable in the medulla (an R1-R6 ends in the
lamina), from a pr_classes_mesh_shares TSV. Network-free; upload with scripts/tools/ng_short_link.py.
Usage: suspect_r16_link.py MESH_SHARES.tsv OUT_ng.txt"""
import sys

import numpy as np
import pandas as pd

from faab import nglink

M = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
M["ME"] = pd.to_numeric(M.ME, errors="coerce")
B = M[(M["class"] == "R1-R6") & (M.ME >= 0.3)].drop_duplicates("root_now").sort_values("ME", ascending=False)
pal = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6"]
xyz = np.array([[float(v) for v in s.split(",")] for s in B.xyz])
spec = {"meshes": [{"kind": "OL", "segments": ["21", "22", "23", "24"], "alpha": 0.15}],
        "neurons": [{"name": f"R1-R6 candidates >= 30% in ME ({len(B)})",
                     "colors": dict(zip(B.root_now, pal)), "ids": B.root_now.tolist()}],
        "position": np.median(xyz, axis=0).round().tolist(), "scale": 20000, "layout": "xy-3d"}
spec["select"] = spec["neurons"][0]["name"]
open(sys.argv[2], "w").write(nglink.to_url(nglink.build(spec)) + "\n")
for (r, g, me), c in zip(B[["root_now", "group", "ME"]].itertuples(index=False), pal):
    print(f"  {c} group {g} {r} ME {me:.2f}")
print("written", sys.argv[2])
