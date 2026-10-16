"""Inventory of existing cutouts from shared neuroglancer states (CAVE state service, read-only).
Input tmp/cutout_state_ids.txt (one state id or spelunker URL per line). For every state: dimensions,
annotation layers and their bounding boxes (voxel corners, size in um), number of point annotations,
and the names/descriptions that identify the cutout. Then compare with our proposed group 1 cubes
(figures/cutout_group1_*um_state.json): size match and centre-to-centre / gap distance.
Output figures/cutout_inventory.tsv. One call per 5 s."""
from faab.config import REPO
import glob
import json
import os
import re
import time

import numpy as np
import pandas as pd
import caveclient
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
IDS = os.path.join(HERE, "..", "tmp", "cutout_state_ids.txt")


def main():
    ids = []
    for ln in open(IDS):
        m = re.search(r"(\d{10,})", ln)
        if m and m.group(1) not in ids:
            ids.append(m.group(1))
    client = caveclient.CAVEclient(DATASTACK)
    t = throttle(time.time())
    rows = []
    for k, sid in enumerate(ids):
        t = throttle(t)
        try:
            st = client.state.get_state_json(int(sid))
        except Exception as e:
            rows.append(dict(state=sid, error=str(e)[:80]))
            print(f"{sid}: failed {str(e)[:60]}", flush=True)
            continue
        dims = st.get("dimensions", {})
        res = [dims.get(a, [None])[0] for a in ("x", "y", "z")]
        names = [L.get("name", "") for L in st.get("layers", [])]
        nbox = 0
        for L in st.get("layers", []):
            if L.get("type") != "annotation":
                continue
            anns = L.get("annotations", [])
            pts = sum(1 for a in anns if a.get("type") == "point")
            for a in anns:
                if a.get("type") != "axis_aligned_bounding_box":
                    continue
                A, B = np.array(a["pointA"], float), np.array(a["pointB"], float)
                lo, hi = np.minimum(A, B), np.maximum(A, B)
                size_um = (hi - lo) * VOX / 1000.0
                nbox += 1
                rows.append(dict(state=sid, layer=L.get("name", ""), layers="; ".join(names), res_nm=res,
                                 box_desc=a.get("description", ""), lo=lo.round(1).tolist(), hi=hi.round(1).tolist(),
                                 size_um=size_um.round(2).tolist(), n_points=pts, position=st.get("position")))
        if nbox == 0:
            rows.append(dict(state=sid, layer="", layers="; ".join(names), res_nm=res, box_desc="", lo=None, hi=None,
                             size_um=None, n_points=0, position=st.get("position")))
        print(f"[{k + 1}/{len(ids)}] {sid}: {nbox} boxes; layers {names}", flush=True)
    inv = pd.DataFrame(rows)
    inv.to_csv(os.path.join(OUT, "cutout_inventory.tsv"), sep="\t", index=False)

    # ---- compare with our proposals ----
    ours = {}
    for p in sorted(glob.glob(os.path.join(OUT, "cutout_group1_*um_state.json"))):
        st = json.load(open(p))
        for L in st["layers"]:
            if L.get("type") == "annotation":
                cube = next(a for a in L["annotations"] if a["id"] == "cube")
                ours[os.path.basename(p)] = (np.array(cube["pointA"], float), np.array(cube["pointB"], float))
    boxes = inv[inv.lo.notna()]
    print(f"\n{len(boxes)} existing boxes in {inv.state.nunique()} states; sizes (um):")
    print(boxes.size_um.astype(str).value_counts().to_string())
    for name, (A, B) in ours.items():
        c = (A + B) / 2
        print(f"\nours {name}: corners {A.round(1).tolist()} -> {B.round(1).tolist()}, size "
              f"{((B - A) * VOX / 1000).round(2).tolist()} um")
        for r in boxes.itertuples():
            lo, hi = np.array(r.lo), np.array(r.hi)
            gap = np.maximum(0, np.maximum(lo - B, A - hi)) * VOX / 1000.0  # per-axis gap in um (0 = overlap on axis)
            cc = np.linalg.norm(((lo + hi) / 2 - c) * VOX) / 1000.0
            flag = "OVERLAP" if np.all(gap == 0) else ""
            if cc < 60 or flag:
                print(f"  {r.state} '{r.layer}' {r.box_desc}: centre distance {cc:.1f} um, axis gaps {gap.round(1).tolist()} {flag}")
    near = []
    for r in boxes.itertuples():
        for name, (A, B) in ours.items():
            cc = np.linalg.norm(((np.array(r.lo) + np.array(r.hi)) / 2 - (A + B) / 2) * VOX) / 1000.0
            near.append((cc, r.state, r.layer, name))
    near.sort()
    print("\nclosest existing boxes to our proposals:")
    for cc, s, l, n in near[:6]:
        print(f"  {cc:7.1f} um  state {s}  layer '{l}'  vs {n}")
    print("DONE")


if __name__ == "__main__":
    main()
