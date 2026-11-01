"""Nearest ommatidia of some axons in an ommatidia.py run (2026-11-01): each root's column-zone line against the
center line (R1-R6 core) of every ommatidium nearby, with the PRs-tab groups of each ommatidium's members.
Network-free. Usage: ommatidia_nearest.py RUN ROOT [ROOT ...]"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "pr_groups"))
import ommatidia as O  # noqa: E402
from faab.config import fig  # noqa: E402

run, roots = sys.argv[1], sys.argv[2:]
d = fig(f"ommatidia/{run}")
L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str}).reset_index(drop=True)
C = O.Lines(L)
Om = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
grp = tab.groupby("root_now").group.apply(lambda s: ",".join(sorted(set(s), key=int))).to_dict()
cents, names = [], []
for om, t in Om[Om.ommatidium != ""].groupby("ommatidium"):
    g = [(C.i[r], s) for r, s in zip(t.root, t.slot) if r in C.i]
    if g:
        cents.append(O.fit_center(C, O.core_of(g)))
        names.append(f"{om} (tab groups {', '.join(sorted({grp.get(r, '-') for r in t.root}))})")
CL = O.centers_as_lines(cents)
for r in roots:
    if r not in C.i:
        print(f"{r}: no column-zone line")
        continue
    k = C.i[r]
    ds = sorted((O.line_dist(C, k, CL, c), names[c]) for c in range(len(cents)) if np.linalg.norm(cents[c]["p"] - C.p[k]) < 25)
    print(f"{r}: " + "; ".join(f"{dd:.1f} um {n}" for dd, n in ds[:4] if np.isfinite(dd)))
