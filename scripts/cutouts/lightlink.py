"""Shareable link for a cutout state: keep the EM/segmentation layers, the box annotations and only the
point annotations that fall inside the proposed cube (the full state with every PR site stays in the JSON).
Usage: python faab_cutout_lightlink.py figures/cutout_<name>_state.json"""
import json
import os
import sys
import urllib.parse

import numpy as np


def main():
    path = sys.argv[1]
    st = json.load(open(path))
    for L in st["layers"]:
        if L.get("type") != "annotation":
            continue
        boxes = [a for a in L["annotations"] if a["type"] == "axis_aligned_bounding_box"]
        cube = next(a for a in boxes if a["id"] == "cube")
        lo, hi = np.array(cube["pointA"]), np.array(cube["pointB"])
        pts = [a for a in L["annotations"] if a["type"] == "point"
               and np.all(np.array(a["point"]) >= lo) and np.all(np.array(a["point"]) <= hi)]
        L["annotations"] = boxes + pts
        print(f"kept {len(boxes)} boxes and {len(pts)} points inside the cube")
    url = "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(json.dumps(st, separators=(",", ":")))
    out = path.replace("_state.json", "_ng_light.txt")
    open(out, "w").write(url + "\n")
    print(f"{len(url)} chars -> {os.path.basename(out)}")
    print(url)


if __name__ == "__main__":
    main()
