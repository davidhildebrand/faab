"""For each MISMATCH (a row's root_id_curr is current but its xyz lies on another segment): how big is the segment at
the xyz, and where is the nearest point on the intended cell (nearest L2 chunk representative point, verified with a
root lookup)? Also the last-edit time of the roots of the Zohar pair. CAVE read-only, >= 5 s between calls.
Usage: mismatch_points.py"""
import numpy as np

from faab.clients.cave import client as cave_client, parse_xyz, roots_at_xyz, throttle
from faab.config import VOX

CASES = [  # (rows, intended root, xyz in the sheet)
    ("PRg1hop2 35/160/286 (Kristo T. moved the xyz)", 648518347463102525, "7418, 18627, 4397"),
    ("PRg1hop2 20/33, PRg1toT4T5 59", 648518347605671615, "9562, 17822, 4958"),
    ("PRg1hop2 210 (Zohar L.)", 648518347509939307, "10697, 20234, 5090"),
    ("PRg1toT4T5 58", 648518347546546302, "9048, 17693, 5083"),
]
EDIT_TIMES = [648518347596617050, 648518347583692884, 648518347513169067, 648518347509936491, 648518347463102525,
              648518347463231049]
c = cave_client()
t = throttle(0)
at = roots_at_xyz([x for _, _, x in CASES])
for (rows, want, xyz), got in zip(CASES, at):
    p = np.array(parse_xyz(xyz)) * np.array(VOX)
    t = throttle(t)
    n_got = len(c.chunkedgraph.get_leaves(int(got), stop_layer=2))
    t = throttle(t)
    l2 = c.chunkedgraph.get_leaves(int(want), stop_layer=2)
    t = throttle(t)
    data = c.l2cache.get_l2data([int(x) for x in l2], attributes=["rep_coord_nm"])
    pts = np.array([v["rep_coord_nm"] for v in data.values() if "rep_coord_nm" in v], float)
    dist = np.linalg.norm(pts - p, axis=1)
    order = np.argsort(dist)[:5]
    cand = [np.round(pts[i] / np.array(VOX)).astype(int) for i in order]
    t = throttle(t)
    ok = roots_at_xyz([list(v) for v in cand])
    best = next(((v, dist[i]) for v, i, r in zip(cand, order, ok) if r == want), None)
    print(f"{rows}: xyz {xyz} lies on {got} ({n_got} L2 chunks); intended {want} ({len(l2)} L2 chunks), nearest point "
          f"on it {dist[order[0]] / 1000:.2f} um away; verified point: "
          + (f"{best[0][0]}, {best[0][1]}, {best[0][2]} ({best[1] / 1000:.2f} um)" if best else "none of the 5 nearest"))
t = throttle(t)
ts = c.chunkedgraph.get_root_timestamps(EDIT_TIMES)
for r, w in zip(EDIT_TIMES, ts):
    print(f"root {r} created (last edit) {w:%Y-%m-%d %H:%M} UTC")
