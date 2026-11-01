"""Why two ommatidia of an ommatidia.py run do not merge (2026-11-01 debugging): their centers' lateral distance, and
each member's distance to the merged center (R1-R6 core), against MERGE_MAX and D_MAX.
Network-free. Usage: ommatidia_merge_check.py RUN OM_A OM_B"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "pr_groups"))
import ommatidia as O  # noqa: E402
from faab.config import fig  # noqa: E402

run, a, b = sys.argv[1:4]
d = fig(f"ommatidia/{run}")
L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str}).reset_index(drop=True)
C = O.Lines(L)
Om = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
ga = [(C.i[x.root], x.slot) for x in Om[Om.ommatidium == a].itertuples()]
gb = [(C.i[x.root], x.slot) for x in Om[Om.ommatidium == b].itertuples()]
ca, cb = (O.centers_as_lines([O.fit_center(C, O.core_of(g))]) for g in (ga, gb))
print(f"center distance {O.line_dist(ca, 0, cb, 0):.2f} um (MERGE_MAX {O.MERGE_MAX})")
g = ga + gb
ce = O.centers_as_lines([O.fit_center(C, O.core_of(g))])
for i, ty in g:
    print(f"  {C.roots[i]} {ty:6s} {O.line_dist(C, i, ce, 0):.2f} um to the merged center (D_MAX {O.D_MAX})")
