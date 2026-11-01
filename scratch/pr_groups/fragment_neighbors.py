"""Which listed axons run next to some roots in a groups_refine.py track (2026-11-01, group 94 fold): per root, the slices
it is in, and the tab rows (group, class) whose axons lie closest in those slices (median distance, um).
Network-free. Usage: fragment_neighbors.py RUN GROUP BOOK ROOT [ROOT ...]"""
import os
import sys

import numpy as np
import pandas as pd

from faab.config import fig
from faab.pr import prs_tab as T

run, g, book, roots = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4:]
S = pd.read_csv(os.path.join(fig(f"refine_{run}"), f"track_g{g}.tsv"), sep="\t", dtype={"root": str})
ws = T.load(book)[1]
tab = {}
for b in T.blocks(ws):
    for r in b.rows:
        for k in ("root_id_curr", "root_id_orig"):
            tab.setdefault(T.get(ws, b, r, k), []).append(f"g{b.group} {T.get(ws, b, r, 'class')}")
P = S.set_index(["s_box", "root"])[["x", "y", "z"]]
for root in roots:
    mine = S[S.root == root]
    if mine.empty:
        print(f"{root}: not in the track of group {g}")
        continue
    d = {}
    for x in mine.itertuples(index=False):
        here = S[(S.s_box == x.s_box) & (S.root != root)]
        for y in here.itertuples(index=False):
            d.setdefault(y.root, []).append(float(np.linalg.norm(np.array([x.x - y.x, x.y - y.y, x.z - y.z]))))
    near = sorted(((np.median(v), r, len(v)) for r, v in d.items() if r in tab), key=lambda t: t[0])[:6]
    print(f"{root}: {len(mine)} slices (s {mine.s_box.min():.0f} to {mine.s_box.max():.0f} um); listed: "
          f"{', '.join(tab.get(root, ['not in the tab']))}")
    for m, r, n in near:
        print(f"   {m:5.2f} um  {r}  {', '.join(tab[r])}  ({n} shared slices)")
