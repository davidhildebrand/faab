#!/usr/bin/env python3
"""Reusable, NETWORK-FREE builder for aedes (FAAB) Spelunker/neuroglancer links.

No caveclient, no fafbseg, no network — pure stdlib. Every link starts from the
proofreading scene recommended in the FAAB README (state 5971265676902400, kept in
data/ng_readme_scene.json; refresh with scripts/tools/ng_base_scene.py): its
segmentation source carries the live SeaTable annotation source, so a viewer sees
at once whether a neuron is annotated (serial_id, side, ...) (David, 2026-10-31).
Neuron layers use that same source list (the first group goes into the scene's own
segmentation layer, which would otherwise overlay every cell on the EM); the scene's
mesh layers are reused (those a spec does not ask for stay in the list, hidden).

Usage:
    python3 nglink.py spec.json        # prints the URL (and writes spec["out"] if set)

spec.json format (all fields optional except at least one of meshes/neurons):
{
  "meshes":  [ {"kind":"brain","alpha":1.0}, {"kind":"AL","alpha":0.45},
               {"kind":"OL","segments":["21","22","23","24"]} ],
                 # kind in: brain, AL, MB, LH, CX, OL (optic-lobe neuropils), neuropil (whole-brain neuropil)
                 # (alpha/color/segments optional -> defaults)
  "neurons": [ {"name":"putative Mi1","color":"#9467bd","ids_file":"figures/mi1_ids.txt"},
               {"name":"A + B","colors":{"648...":"#1f77d6","648...":"#2ecc71"},
                "ids":["648...","648..."]} ],
                 # ids: inline list  OR  ids_file: path to newline-separated IDs
                 # color: single default color  OR  colors: {id: "#hex", ...}; "visible": false = hidden layer
  "view":   "anterior",         # "anterior" | "posterior" | [x,y,z,w] quaternion
  "layout": "3d",               # "3d" | "4panel" | "xy-3d"
  "select": "A + B",            # layer name to open in the right-side panel
  "image":  false,              # include the EM image layer (for cross-sections)
  "out":    "figures/foo_ng.txt"
}
"""
import copy
import os
import sys
import json
import urllib.parse

BASE_SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "ng_readme_scene.json")
BASE_SEG = "wclee_aedes_brain"  # the README scene's segmentation layer (graphene + live SeaTable annotation source)
BASE_MESH = {"brain": "brain mesh", "AL": "al_meshes3", "MB": "mb_meshes", "LH": "lh_meshes", "CX": "CX",
             "OL": "aedes_ol_meshes", "neuropil": "aedes_neuropil"}
SEG ="graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain"
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
    # optic-lobe neuropils (David's standard link, 2026-10-27; ids -> labels in faab.ol_meshes.LABELS):
    # 1 CB, 2 OL_L, 3 OL_R, 11-14 ME/LO/LOP/LA_L, 21-24 ME/LO/LOP/LA_R; default = the eight OL neuropils
    "OL": {"source": {"url": "https://flyemdev.mrc-lmb.cam.ac.uk/flyconnectome/rclone/aedes_ol_meshes/|neuroglancer-precomputed:",
                      "subsources": {"default": True, "properties": True}, "enableDefaultSubsources": False},
           "segments": ["11", "12", "13", "14", "21", "22", "23", "24"], "name": "aedes_ol_meshes",
           "color": None, "alpha": 0.22},
    # whole-brain neuropil surface (standard link: segment 6 shown, alpha 0.13, light grey)
    "neuropil": {"source": "https://flyem.mrc-lmb.cam.ac.uk/flyconnectome/aedes_brain/|neuroglancer-precomputed:",
                 "segments": ["6"], "name": "aedes_neuropil", "color": "#e0e0e0", "alpha": 0.13},
}


def mesh_layer(kind, alpha=None, color=None, segments=None):
    t = MESHES[kind]
    layer = {"type": "segmentation", "source": t["source"], "segments": [str(s) for s in (segments or t["segments"])],
             "name": t["name"], "visible": True,
             "objectAlpha": t["alpha"] if alpha is None else alpha}
    if color or t["color"]:
        layer["segmentDefaultColor"] = color or t["color"]
    if "silhouette" in t:
        layer["meshSilhouetteRendering"] = t["silhouette"]
    return layer


def _load_ids(spec):
    ids = spec.get("ids")
    if not ids and spec.get("ids_file"):
        with open(spec["ids_file"]) as f:
            ids = [ln.strip() for ln in f if ln.strip()]
    return [str(i) for i in (ids or [])]


def _seg_url(src):
    first = src[0] if isinstance(src, list) and src else src
    return str(first.get("url") if isinstance(first, dict) else first).rstrip("/")


def clean_state(state):
    """(1) drop source entries commented out with a leading '#' (the README scene has '#.../aedes_infos/|...';
    Spelunker rejects it: 'scheme does not support | URL pipelines'); (2) the README scene's segmentation layer with no
    segments overlays EVERY cell on the EM (David, 2026-10-31), so the first neuron layer on the same segmentation is
    folded into it (it keeps the scene's sources and settings, takes that layer's name, segments and colors).
    Returns the state, changed in place."""
    layers = state.get("layers", [])
    for L in layers:
        if isinstance(L.get("source"), list):
            L["source"] = [s for s in L["source"] if not (isinstance(s, str) and s.startswith("#"))]
    base = next((L for L in layers if L.get("name") == BASE_SEG and L.get("type") == "segmentation"), None)
    if base is not None and not base.get("segments"):
        first = next((L for L in layers if L is not base and L.get("type") == "segmentation" and L.get("segments")
                      and _seg_url(L["source"]) == _seg_url(base["source"])), None)
        if first is not None:
            base.update({k: v for k, v in first.items() if k != "source"})
            layers.remove(first)
            if state.get("selectedLayer", {}).get("layer") == BASE_SEG:
                state["selectedLayer"]["layer"] = base["name"]
    return state


def base_scene():
    """the README proofreading scene (a fresh copy, '#' sources dropped)"""
    if not os.path.exists(BASE_SCENE):
        sys.exit(f"missing {BASE_SCENE}: run scripts/tools/ng_base_scene.py once")
    with open(BASE_SCENE) as f:
        st = json.load(f)
    st.pop("_faab_scene_id", None)
    return clean_state(st)


def seg_source():
    """the README scene's segmentation source list (graphene + live SeaTable annotations)"""
    return next(L for L in base_scene()["layers"] if L.get("name") == BASE_SEG)["source"]


def neuron_layer(spec, source=None):
    layer = {"type": "segmentation", "source": copy.deepcopy(source) if source else seg_source(), "segments": _load_ids(spec),
             "name": spec.get("name", "neurons"), "visible": spec.get("visible", True)}
    if spec.get("colors"):
        layer["segmentColors"] = {str(k): v for k, v in spec["colors"].items()}
    elif spec.get("color"):
        layer["segmentDefaultColor"] = spec["color"]
    return layer


def annotation_layer(spec):
    """spec: {"name", "color": "#hex", "lines": [[[x,y,z],[x,y,z]], ...], "points": [[x,y,z], ...],
    "boxes": [[[x,y,z],[x,y,z]], ...]} (boxes = axis-aligned bounding boxes, corner to corner, as the synapse cutouts
    are drawn); coordinates in 16 x 16 x 45 nm voxels (the viewer's dimensions)"""
    ann = []
    for k, (a, b) in enumerate(spec.get("lines", [])):
        ann.append({"type": "line", "pointA": [float(v) for v in a], "pointB": [float(v) for v in b], "id": f"l{k}"})
    for k, (a, b) in enumerate(spec.get("boxes", [])):
        ann.append({"type": "axis_aligned_bounding_box", "pointA": [float(v) for v in a], "pointB": [float(v) for v in b],
                    "id": f"b{k}"})
    for k, p in enumerate(spec.get("points", [])):
        ann.append({"type": "point", "point": [float(v) for v in p], "id": f"p{k}"})
    return {"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
            "annotations": ann, "name": spec.get("name", "annotations"), "annotationColor": spec.get("color", "#ffff00"),
            "visible": True}


def build(spec):
    state = base_scene()
    base = state["layers"]
    seg = next(L for L in base if L.get("name") == BASE_SEG)
    asked = {BASE_MESH[m["kind"]]: m for m in spec.get("meshes", [])}
    layers = []
    for L in base:
        if L["type"] == "image":
            if not spec.get("image", True):  # user rule (2026-09-25): the EM data layer is always present
                continue
        elif L is not seg:  # mesh layers: shown when the spec asks for them, else kept hidden
            m = asked.pop(L["name"], None)
            L["visible"] = m is not None
            if m:
                if m.get("segments"):
                    L["segments"] = [str(s) for s in m["segments"]]
                if m.get("alpha") is not None:
                    L["objectAlpha"] = m["alpha"]
                if m.get("color"):
                    L["segmentDefaultColor"] = m["color"]
                    L.pop("segmentColors", None)
        layers.append(L)
    layers += [mesh_layer(m["kind"], m.get("alpha"), m.get("color"), m.get("segments")) for m in asked.values()]
    layers += [neuron_layer(n, seg["source"]) for n in spec.get("neurons", [])]
    layers += [annotation_layer(a) for a in spec.get("annotations", [])]
    view = spec.get("view", "anterior")
    orient = VIEWS.get(view, view)
    # optional "position": [x, y, z] in 16x16x45 nm voxels and "scale": projectionScale
    # (whole brain = 31785; ~6000 frames a single cell) to open the view on one cell
    state.update({"dimensions": DIMS, "position": spec.get("position", POS), "projectionOrientation": orient,
                  "projectionScale": spec.get("scale", PSCALE), "layers": layers,
                  "showSlices": bool(spec.get("image", False)),
                  "showDefaultAnnotations": False, "layout": spec.get("layout", "3d")})
    if spec.get("select"):
        state["selectedLayer"] = {"layer": spec["select"], "visible": True, "size": 361}
    if spec.get("cross_scale"):  # 2D panel zoom (nm per screen pixel / voxel; ~0.5 frames a 2 um cutout)
        state["crossSectionScale"] = spec["cross_scale"]
    return clean_state(state)


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
