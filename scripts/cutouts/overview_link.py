"""One link: EM image, the group's photoreceptors, the proposed cutout (yellow, with its in-box PR
presynaptic sites) and every existing cutout from figures/cutout_inventory.tsv as a second annotation
layer (cyan). Network-free. Usage: python faab_cutout_overview_link.py figures/cutout_<name>_state.json"""
from faab.config import REPO
import ast
import json
import os
import sys
import urllib.parse

import numpy as np
import pandas as pd
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")
DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}


def main():
    path = sys.argv[1]
    st = json.load(open(path))
    import faab.nglink as nglink
    layers = [L for L in st["layers"] if L.get("type") != "annotation"]
    assert layers and layers[0]["type"] == "image", "EM layer must come first"
    layers[0]["name"] = "EM"
    if not any(L.get("name") == "brain mesh" for L in layers):
        layers.insert(1, nglink.mesh_layer("brain", alpha=0.15))
    for L in st["layers"]:
        if L.get("type") != "annotation":
            continue
        boxes = [a for a in L["annotations"] if a["type"] == "axis_aligned_bounding_box" and a["id"] == "cube"]
        lo, hi = np.array(boxes[0]["pointA"]), np.array(boxes[0]["pointB"])
        pts = [a for a in L["annotations"] if a["type"] == "point"
               and np.all(np.array(a["point"]) >= lo) and np.all(np.array(a["point"]) <= hi)]
        layers.append({"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                       "annotations": boxes + pts, "name": "proposed cutout (group 1)", "annotationColor": "#ffff00"})
    inv = pd.read_csv(fig("cutout_inventory.tsv"), sep="\t", dtype={"state": str})
    inv = inv[inv.lo.notna()]
    seen, anns = set(), []
    for r in inv.itertuples():
        lo, hi = ast.literal_eval(r.lo), ast.literal_eval(r.hi)
        key = tuple(np.round(lo, 0)) + tuple(np.round(hi, 0))
        if key in seen:
            continue
        seen.add(key)
        anns.append({"type": "axis_aligned_bounding_box", "id": f"existing_{r.state}", "pointA": lo, "pointB": hi,
                     "description": f"existing cutout, state {r.state}"})
    layers.append({"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                   "annotations": anns, "name": f"existing cutouts ({len(anns)})", "annotationColor": "#00e5ff"})
    state = dict(st)
    state["layers"] = layers
    state["selectedLayer"] = {"layer": "proposed cutout (group 1)", "visible": True, "size": 400}
    url = "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    out = path.replace("_state.json", "_overview_ng.txt")
    open(out, "w").write(url + "\n")
    json.dump(state, open(path.replace("_state.json", "_overview_state.json"), "w"), indent=1)
    print(f"{len(anns)} distinct existing cutouts; {len(url)} chars -> {os.path.basename(out)}")
    print(url)


if __name__ == "__main__":
    main()
