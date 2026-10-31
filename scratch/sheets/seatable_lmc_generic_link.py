"""Spelunker link (README scene base) for the SeaTable cells typed only L1-L5, one layer per side.
Network-free; upload with scripts/tools/ng_short_link.py. Usage: seatable_lmc_generic_link.py L1-5_ONLY.tsv"""
import sys

import pandas as pd

from faab import nglink

T = pd.read_csv(sys.argv[1], sep="\t", dtype=str).fillna("")
spec = {"meshes": [{"kind": "OL"}, {"kind": "neuropil"}],
        "neurons": [{"name": f"SeaTable L1-L5 only, side {s} ({len(g)})", "color": c, "ids": g.root_now.tolist()}
                    for (s, g), c in zip(T.groupby("side"), ["#e6194b", "#1f77d6"])],
        "layout": "xy-3d"}
spec["select"] = spec["neurons"][0]["name"]
out = sys.argv[1].replace(".tsv", "_ng.txt")
open(out, "w").write(nglink.to_url(nglink.build(spec)) + "\n")
print("written", out)
