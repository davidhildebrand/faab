"""Ring check of photoreceptor groups (2026-11-01): a real ommatidium has its six R1-R6 around the bundle center (the
largest angular gap between neighbors on the ring is the R1/R6 gap holding the R7, ~76 deg in group 1); a group made of
halves of two bundles has its R1-R6 on one side (largest gap >= 180 deg). For the PRs-tab groups (version 1) and the
ommatidia of an ommatidia.py run (version 2), on the column-zone lines: the R1-R6 positions across the axis at the
middle of their shared stretch, relative to their center; largest angular gap, mean radius. Reference: proofread
groups 1-8. For every lead group the two versions disagree on: the gap of the tab group vs the gaps of the ommatidia
that hold its members.
Network-free. Usage: ommatidia_ring_check.py RUN"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "pr_groups"))
import ommatidia as O  # noqa: E402
from faab.config import fig  # noqa: E402

run = sys.argv[1]
d = fig(f"ommatidia/{run}")
L = pd.read_csv(os.path.join(d, "lines.tsv"), sep="\t", dtype={"root": str})
L = L[L.n_points >= 2].reset_index(drop=True)
C = O.Lines(L)
tab = pd.read_csv(os.path.join(d, "tab_roots.tsv"), sep="\t", dtype=str).fillna("")
tab = tab[(tab.side == "right") & ~tab.notes.str.contains("OFF-BUNDLE|SECOND BUNDLE")]
Om = pd.read_csv(os.path.join(d, "ommatidia.tsv"), sep="\t", dtype={"root": str}).fillna("")
Cmp = pd.read_csv(os.path.join(d, "compare.tsv"), sep="\t").fillna("")


def ring(r16):
    """largest angular gap (deg) and mean radius (um) of R1-R6 lines around their center, or None"""
    ks = [C.i[r] for r in r16 if r in C.i]
    if len(ks) < 4:
        return None
    ce = O.fit_center(C, ks)
    ax = ce["ax"]
    t = np.median([np.mean(C.trange(k, ax)) for k in ks])
    cl = O.centers_as_lines([ce])
    c0 = cl.at(0, np.array([t]), ax)[0]
    e1 = O.unit(np.cross(ax, [0.0, 0.0, 1.0]))
    e2 = np.cross(ax, e1)
    P = np.vstack([C.at(k, np.array([t]), ax)[0] for k in ks]) - c0
    ang = np.sort(np.degrees(np.arctan2(P @ e2, P @ e1)) % 360)
    gaps = np.diff(np.r_[ang, ang[0] + 360])
    return float(gaps.max()), float(np.linalg.norm(P - np.outer(P @ ax, ax), axis=1).mean()), len(ks)


rows = []
for g, t in tab.groupby(tab.group.astype(int)):
    x = ring(t[t.cls == "R1-R6"].root_now)
    if x:
        rows.append(dict(version="tab", id=f"g{g}", section=t.section.iloc[0], gap=x[0], radius=x[1], n=x[2]))
for om, t in Om[Om.ommatidium != ""].groupby("ommatidium"):
    x = ring(t[t.slot == "R1-R6"].root)
    if x:
        rows.append(dict(version="v2", id=om, section="", gap=x[0], radius=x[1], n=x[2]))
R = pd.DataFrame(rows)
ref = R[(R.version == "tab") & (R.section == "targets")]
print(f"proofread groups 1-8: largest gap {ref.gap.round(0).astype(int).tolist()} deg; radius "
      f"{ref.radius.round(2).tolist()} um")
for v, s in (("tab leads", R[(R.version == "tab") & (R.section == "leads")]), ("v2 ommatidia", R[R.version == "v2"])):
    print(f"{v}: {len(s)} with >= 4 R1-R6 lines; largest gap p50 {s.gap.median():.0f}, p90 {s.gap.quantile(0.9):.0f} deg; "
          f">= 180 deg: {int((s.gap >= 180).sum())}; radius p50 {s.radius.median():.2f} um")
gap = dict(zip(R.id, R.gap))
dis = Cmp[(Cmp.section == "leads") & (Cmp.jaccard < 1) & (Cmp.ommatidium != "")]
print(f"\nlead groups the versions disagree on ({len(dis)}): tab group gap -> gaps of the ommatidia holding its members")
better = worse = 0
for x in dis.itertuples():
    oms = [x.ommatidium] + [o for o in str(x.other_om).split(", ") if o]
    tg = gap.get(f"g{x.group}")
    og = [gap.get(o) for o in oms]
    known = [v for v in og if v is not None]
    if tg is not None and known:
        better += max(known) < tg
        worse += max(known) > tg
    print(f"  g{x.group}: {'-' if tg is None else f'{tg:.0f}'} -> " +
          ", ".join(f"{o} {'-' if v is None else f'{v:.0f}'}" for o, v in zip(oms, og)))
print(f"tab group's gap larger than every ommatidium's: {better}; smaller than the largest: {worse}")
