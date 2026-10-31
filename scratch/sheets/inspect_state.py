"""What does a shared Spelunker state show? Layers, selected segments (current or not, and their current roots),
annotation points (in 16 x 16 x 45 nm voxels) with the root under each, and the view position. CAVE read-only, >= 5 s
between calls. Usage: inspect_state.py STATE_ID"""
import json
import sys

import numpy as np

from faab.clients.cave import client as cave_client, roots_at_xyz, throttle

c = cave_client()
t = throttle(0)
state = c.state.get_state_json(int(sys.argv[1]))
dims = state.get("dimensions", {})
scale = np.array([dims.get(k, [1, "m"])[0] / s for k, s in zip("xyz", (16e-9, 16e-9, 45e-9))]) if dims else np.ones(3)
print(f"dimensions {json.dumps(dims)}; position {state.get('position')} (voxels {np.round(np.array(state.get('position', [0, 0, 0])) * scale).astype(int).tolist()})")
segs, points = [], []
for L in state.get("layers", []):
    kind, name = L.get("type"), L.get("name")
    if kind in ("segmentation", "segmentation_with_graph"):
        ids = [str(s) for s in L.get("segments", [])]
        hidden = [str(s)[1:] for s in ids if str(s).startswith("!")]
        shown = [s for s in ids if not s.startswith("!")]
        print(f"segmentation layer '{name}': shown {shown}; hidden {hidden}")
        segs += [s for s in shown + hidden if s.isdigit() and s.startswith("648")]
    elif kind == "annotation":
        for a in L.get("annotations", []):
            for key in ("point", "pointA", "pointB", "center"):
                if key in a:
                    p = np.round(np.array(a[key], float) * scale).astype(int)
                    points.append((name, f"{a.get('type')} {key}" + (f": {a['description']}" if a.get("description") else ""), p))
        print(f"annotation layer '{name}': {len(L.get('annotations', []))} annotations")
    else:
        print(f"{kind} layer '{name}'")
if segs:
    t = throttle(t)
    ok = np.asarray(c.chunkedgraph.is_latest_roots([int(s) for s in segs])).astype(bool)
    for s, good in zip(segs, ok):
        if not good:
            t = throttle(t)
            now = [int(x) for x in np.atleast_1d(c.chunkedgraph.get_latest_roots(int(s)))]
            print(f"  segment {s} is no longer current; now {now}")
    print(f"segments current: {int(ok.sum())} of {len(segs)}")
if points:
    t = throttle(t)
    under = roots_at_xyz([list(p) for _, _, p in points])
    for (name, desc, p), r in zip(points, under):
        print(f"point in '{name}' {p[0]}, {p[1]}, {p[2]}{' (' + desc + ')' if desc else ''}: root {r}")
