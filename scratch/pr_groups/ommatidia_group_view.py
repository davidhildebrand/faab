"""One PRs-tab group in an ommatidia.py run (2026-11-01 debugging): its members' column-zone lines (depths seen, angle
to the local axis, wobble, verdict, ommatidium) and their lateral offsets from the members' mean at four heights along
the shared axis, plus the raw per-depth census positions (tracks.tsv) as offsets from the per-depth mean.
Network-free. Usage: ommatidia_group_view.py RUN G"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "pr_groups"))
import ommatidia as O  # noqa: E402
from faab.config import fig  # noqa: E402

run, g = sys.argv[1], int(sys.argv[2])
d = fig(f"ommatidia/{run}")
tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
t = tab[tab.group.astype(int) == g]
L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str})
F = pd.read_csv(os.path.join(d, "features.tsv"), sep="\t", dtype={"root": str}).set_index("root")
Om = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
om = dict(zip(Om.root, Om.ommatidium + " " + Om.slot))
tr = pd.read_csv(os.path.join(d, "tracks.tsv"), sep="\t", dtype={"root": str})
C = O.Lines(L.reset_index(drop=True))
mem = [r for r in t.root_now if r in C.i]
print(f"group {g}: {len(t)} rows, {len(mem)} with a column-zone line")
for x in t.itertuples(index=False):
    r = x.root_now
    li = L[L.root == r]
    desc = (f"depths {li.depths.iloc[0]}, angle {li.angle.iloc[0]:.0f}, wobble {li.wobble.iloc[0]:.1f}, pieces>1 "
            f"{li.multi.iloc[0]}" if len(li) else "no line")
    print(f"  {x.cls:6s} {r}  {desc}; {F.loc[r, 'verdict'] if r in F.index else '-'}; {om.get(r, '-')}")
ks = [C.i[r] for r in mem]
ax = O.unit(C.ax[ks].sum(0))
lo = max(min(C.trange(k, ax)[0] for k in ks), -1e9)
hi = max(C.trange(k, ax)[1] for k in ks)
ts = np.linspace(lo, hi, 4)
P = np.stack([C.at(k, ts, ax) for k in ks])  # member x height x 3
mean = P.mean(0)
print("lateral offset from the members' mean (um) at four heights along the axis (retina side last):")
for k, r in zip(ks, mem):
    off = P[ks.index(k)] - mean
    lat = np.linalg.norm(off - np.outer(off @ ax, ax), axis=1)
    print(f"  {r}  " + "  ".join(f"{v:5.1f}" for v in lat))
print("raw census positions: offset from the per-depth mean of the members (um)")
sub = tr[tr.root.isin(mem)]
for dp, s in sub.groupby("depth"):
    Q = s[["x", "y", "z"]].to_numpy(float)
    off = Q - Q.mean(0)
    lat = np.linalg.norm(off - np.outer(off @ ax, ax), axis=1)
    print(f"  depth {dp:.0f}: " + ", ".join(f"{r[-6:]} {v:.1f}" for r, v in zip(s.root, lat)))
