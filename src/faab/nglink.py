#!/usr/bin/env python3
"""Reusable, NETWORK-FREE builder for aedes (FAAB) Spelunker/neuroglancer links.

No caveclient, no fafbseg, no network — pure stdlib. Mesh sources/transforms and
the standard view are hardcoded from memory (faab-neuroglancer-mesh-sources), so
this runs locally and doesn't need a permissions popup for network access.

Usage:
    python3 nglink.py spec.json        # prints the URL (and writes spec["out"] if set)

spec.json format (all fields optional except at least one of meshes/neurons):
{
  "meshes":  [ {"kind":"brain","alpha":1.0}, {"kind":"AL","alpha":0.45} ],
                 # kind in: brain, AL, MB, LH, CX   (alpha/color optional -> defaults)
  "neurons": [ {"name":"putative Mi1","color":"#9467bd","ids_file":"figures/mi1_ids.txt"},
               {"name":"A + B","colors":{"648...":"#1f77d6","648...":"#2ecc71"},
                "ids":["648...","648..."]} ],
                 # ids: inline list  OR  ids_file: path to newline-separated IDs
                 # color: single default color  OR  colors: {id: "#hex", ...}
  "view":   "anterior",         # "anterior" | "posterior" | [x,y,z,w] quaternion
  "layout": "3d",               # "3d" | "4panel" | "xy-3d"
  "select": "A + B",            # layer name to open in the right-side panel
  "image":  false,              # include the EM image layer (for cross-sections)
  "out":    "figures/foo_ng.txt"
}
"""
import sys
import json
import urllib.parse

SEG = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain"
IMG = "precomputed://gs://zetta_lee_mosquito_001_alignment_central/img/v2_sharded"
_OUTDIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}

# standard view (whole-brain framing from reference state 6364293577572352)
DIMS = _OUTDIMS
POS = [26870.783203125, 18121.6953125, 5377.9716796875]
PSCALE = 31785.313376655453
VIEWS = {"anterior": [0, 1, 0, 0],    # antennal lobes head-on, dorsal up
         "posterior": [0, 0, 0, 1]}   # identity = posterior face

# neuropil mesh templates (segmentation nm space; only the brain needs a transform)
_BRAIN_SRC = {"url": "precomputed://gs://aedes_whole-brain/meshes",
              "transform": {"matrix": [[0.12187, 0.99254, 0, 0],
                                       [-0.99254, 0.12187, 0, 30600],
                                       [0, 0, 1, 0]],
                            "outputDimensions": _OUTDIMS}}
MESHES = {
    "brain": {"source": _BRAIN_SRC, "segments": ["1"], "name": "brain mesh",
              "color": "#ffffff", "alpha": 1.0, "silhouette": 4.1},
    "AL": {"source": "gs://aedes_whole-brain/meshes/al_meshes_2/|neuroglancer-precomputed:",
           "segments": ["10000", "10001"], "name": "AL mesh", "color": "#f0a030", "alpha": 0.45},
    "MB": {"source": "gs://aedes_whole-brain/meshes/mb_meshes/|neuroglancer-precomputed:",
           "segments": ["1", "2"], "name": "MB mesh", "color": "#8fbc8f", "alpha": 0.4},
    "LH": {"source": "gs://aedes_whole-brain/meshes/lh_meshes_02_13_2026/|neuroglancer-precomputed:",
           "segments": ["1", "0"], "name": "LH mesh", "color": "#c39bd3", "alpha": 0.4},
    "CX": {"source": "gs://aedes_whole-brain/meshes/CX_mesh/|neuroglancer-precomputed:",
           "segments": ["1", "2", "3", "4", "5", "6", "7", "8"], "name": "CX mesh",
           "color": "#7fb3d5", "alpha": 0.4},
}


def mesh_layer(kind, alpha=None, color=None):
    t = MESHES[kind]
    layer = {"type": "segmentation", "source": t["source"], "segments": list(t["segments"]),
             "name": t["name"], "visible": True,
             "objectAlpha": t["alpha"] if alpha is None else alpha,
             "segmentDefaultColor": color or t["color"]}
    if "silhouette" in t:
        layer["meshSilhouetteRendering"] = t["silhouette"]
    return layer


def _load_ids(spec):
    ids = spec.get("ids")
    if not ids and spec.get("ids_file"):
        with open(spec["ids_file"]) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    return [str(i) for i in (ids or [])]


def neuron_layer(spec):
    layer = {"type": "segmentation", "source": SEG, "segments": _load_ids(spec),
             "name": spec.get("name", "neurons"), "visible": True}
    if spec.get("colors"):
        layer["segmentColors"] = {str(k): v for k, v in spec["colors"].items()}
    elif spec.get("color"):
        layer["segmentDefaultColor"] = spec["color"]
    return layer


def annotation_layer(spec):
    """spec: {"name", "color": "#hex", "lines": [[[x,y,z],[x,y,z]], ...], "points": [[x,y,z], ...]}
    coordinates in 16 x 16 x 45 nm voxels (the viewer's dimensions)"""
    ann = []
    for k, (a, b) in enumerate(spec.get("lines", [])):
        ann.append({"type": "line", "pointA": [float(v) for v in a], "pointB": [float(v) for v in b], "id": f"l{k}"})
    for k, p in enumerate(spec.get("points", [])):
        ann.append({"type": "point", "point": [float(v) for v in p], "id": f"p{k}"})
    return {"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
            "annotations": ann, "name": spec.get("name", "annotations"), "annotationColor": spec.get("color", "#ffff00"),
            "visible": True}


def build(spec):
    layers = []
    if spec.get("image", True):  # user rule (2026-09-25): the EM data layer is always present
        layers.append({"type": "image", "source": IMG, "name": "v2_sharded"})
    layers += [mesh_layer(m["kind"], m.get("alpha"), m.get("color")) for m in spec.get("meshes", [])]
    layers += [neuron_layer(n) for n in spec.get("neurons", [])]
    layers += [annotation_layer(a) for a in spec.get("annotations", [])]
    view = spec.get("view", "anterior")
    orient = VIEWS.get(view, view)
    # optional "position": [x, y, z] in 16x16x45 nm voxels and "scale": projectionScale
    # (whole brain = 31785; ~6000 frames a single cell) to open the view on one cell
    state = {"dimensions": DIMS, "position": spec.get("position", POS), "projectionOrientation": orient,
             "projectionScale": spec.get("scale", PSCALE), "layers": layers,
             "showSlices": bool(spec.get("image", False)),
             "showDefaultAnnotations": False, "layout": spec.get("layout", "3d")}
    if spec.get("select"):
        state["selectedLayer"] = {"layer": spec["select"], "visible": True, "size": 361}
    return state


def to_url(state):
    return "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(
        json.dumps(state, separators=(",", ":")))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python3 nglink.py spec.json")
    with open(sys.argv[1]) as f:
        spec = json.load(f)
    url = to_url(build(spec))
    if spec.get("out"):
        with open(spec["out"], "w") as f:
            f.write(url + "\n")
    print(url)


if __name__ == "__main__":
    main()
