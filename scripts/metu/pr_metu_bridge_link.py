"""Spelunker link for the strongest PR -> bridge -> MeTu paths (network-free, from the saved tables of
faab_pr_metu_bridge.py). Bridge cells with path_strength >= MIN_STRENGTH in red, the PRs they receive
from (R1-R6 orange, R7/R8 magenta), the MeTus they feed in blue; EM layer first, brain mesh faint.
Also prints, per bridge cell, which PRs and MeTus it connects. Output figures/pr_metu_bridge_ng.txt"""
from faab.config import REPO
import json
import os
import sys
import urllib.parse

import numpy as np
import pandas as pd
import faab.nglink as nglink
from faab.partners.top10_xyz import VOX
from faab.config import fig

HERE = REPO
OUT = os.path.join(HERE, "figures")
MIN_STRENGTH = 3


DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}
# Body axes in dataset coordinates, taken from the recorded standard views (anterior view = antennal lobes
# head-on with dorsal up; identity orientation = posterior face): dorsal = -y, anterior = +z. Lateral for the
# right optic lobe = -x (midline at x ~ 26870). TO BE VERIFIED ONCE IN THE VIEWER; flip a sign here if wrong.
# (letter at the + end, letter at the - end, unit direction in dataset nm space, colour)
AXES = [("D", "V", np.array([0, -1, 0.0]), "#ffffff"),   # dorsal assumed -y
        ("A", "P", np.array([0, 0, 1.0]), "#7fff00"),    # anterior assumed +z
        ("L", "M", np.array([-1, 0, 0.0]), "#ffa500")]   # lateral = -x for the right optic lobe
# letter strokes in a unit box (u right 0..0.7, v up 0..1)
GLYPHS = {"D": [((0, 0), (0, 1)), ((0, 1), (0.4, 1)), ((0.4, 1), (0.7, 0.75)), ((0.7, 0.75), (0.7, 0.25)), ((0.7, 0.25), (0.4, 0)), ((0.4, 0), (0, 0))],
          "V": [((0, 1), (0.35, 0)), ((0.35, 0), (0.7, 1))],
          "A": [((0, 0), (0.35, 1)), ((0.35, 1), (0.7, 0)), ((0.15, 0.4), (0.55, 0.4))],
          "P": [((0, 0), (0, 1)), ((0, 1), (0.5, 1)), ((0.5, 1), (0.7, 0.8)), ((0.7, 0.8), (0.5, 0.55)), ((0.5, 0.55), (0, 0.55))],
          "L": [((0, 1), (0, 0)), ((0, 0), (0.7, 0))],
          "M": [((0, 0), (0, 1)), ((0, 1), (0.35, 0.5)), ((0.35, 0.5), (0.7, 1)), ((0.7, 1), (0.7, 0))]}


def rotation(q):
    """3x3 rotation matrix of a neuroglancer quaternion [x, y, z, w]"""
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def axis_legend(centre_vox, orientation, scale, offset_frac=(0.55, 0.36), half_frac=0.05, letter_frac=0.04):
    """Legend off to the side of the field of view (user convention, 2026-10-01): per body axis one layer with
    a line and ONE letter, drawn as line strokes in the screen plane at the positive end; no balls. Sizes and
    the offset from the view centre scale with the zoom: `scale` is the viewer's projectionScale, i.e. the
    field height in x-voxels, so the legend lands toward the upper right edge at any zoom."""
    R = rotation(orientation)
    right, up = R @ np.array([1.0, 0, 0]), R @ np.array([0, -1.0, 0])  # identity view: x right, y down
    height_nm = float(scale) * VOX[0]
    origin_nm = np.asarray(centre_vox, float) * VOX + (right * offset_frac[0] + up * offset_frac[1]) * height_nm
    half_nm, letter_nm = half_frac * height_nm, letter_frac * height_nm
    layers = []
    for plus, minus, direction, colour in AXES:
        a = origin_nm - direction * half_nm
        b = origin_nm + direction * half_nm
        anns = [{"type": "line", "id": f"{plus}{minus}", "pointA": (a / VOX).round(1).tolist(), "pointB": (b / VOX).round(1).tolist(),
                 "description": f"{plus}-{minus} axis, {plus} at the lettered end"}]
        c = b + direction * letter_nm * 0.9  # letter centre just beyond the positive end
        for k, (p0, p1) in enumerate(GLYPHS[plus]):
            pa = c + right * (p0[0] - 0.35) * letter_nm + up * (p0[1] - 0.5) * letter_nm
            pb = c + right * (p1[0] - 0.35) * letter_nm + up * (p1[1] - 0.5) * letter_nm
            anns.append({"type": "line", "id": f"{plus}_{k}", "pointA": (pa / VOX).round(1).tolist(),
                         "pointB": (pb / VOX).round(1).tolist(), "description": plus})
        layers.append({"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                       "annotations": anns, "name": f"legend {plus}-{minus}", "annotationColor": colour})
    return layers


def main():
    bridge = pd.read_csv(fig("pr_metu_bridge.tsv"), sep="\t", dtype={"root": str})
    down = pd.read_csv(fig("pr_metu_down_synapses.tsv"), sep="\t", dtype={"pre_pt_root_id": str, "post_pt_root_id": str})
    up = pd.read_csv(fig("pr_metu_up_synapses.tsv"), sep="\t", dtype={"pre_pt_root_id": str, "post_pt_root_id": str})
    cells = pd.read_csv(fig("pr_metu_link_cells.tsv"), sep="\t", dtype={"root": str})
    pr_class = {r.root: r.cls for r in cells[cells.kind == "PR"].itertuples()}
    # name grouped photoreceptors by group and class; FlyTable ones by their cartridge label
    gtab = pd.read_csv(os.path.join(HERE, "..", "tmp", "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    pr_name = {r.root_id_curr: f"g{r.group} {r._3}" for r in gtab.itertuples()}
    for r in cells[cells.kind == "PR"].itertuples():
        pr_name.setdefault(r.root, f"SeaTable {str(r.flytable).split(' / ')[-2]} {str(r.flytable).split(' / ')[-1]}"
                           if " / " in str(r.flytable) else "SeaTable PR")
    strong = bridge[bridge.path_strength >= MIN_STRENGTH]
    prs, metus, pts = set(), set(), []
    for b in strong.itertuples():
        d = down[down.post_pt_root_id == b.root]
        u = up[up.pre_pt_root_id == b.root]
        p_ids = sorted(set(d.pre_pt_root_id))
        m_ids = sorted(set(u.post_pt_root_id))
        prs |= set(p_ids)
        metus |= set(m_ids)
        for s in d.pre_pt_position:
            pts.append([float(v) for v in str(s).strip("[]").split()])
        print(f"bridge {b.root} (strength {b.path_strength}): from "
              f"{', '.join(f'{p} [{pr_name.get(p, pr_class.get(p, '?'))}] {int((d.pre_pt_root_id == p).sum())} syn' for p in p_ids)}"
              f" -> to {len(m_ids)} MeTu: {', '.join(f'{m} ({int((u.post_pt_root_id == m).sum())} syn)' for m in m_ids)}")
    centre = np.median(np.array(pts), axis=0) / VOX if pts else None
    # photoreceptor layers: grouped cells carry a definite class (R7 / R8 / R1-R6 from David's table); SeaTable
    # cells are typed only R1-6 or R7-8, so R7-8 ones are listed separately as unresolved
    gclass = {r.root_id_curr: r._3 for r in gtab.itertuples()}
    r7s = sorted(p for p in prs if gclass.get(p) == "R7")
    r8s = sorted(p for p in prs if gclass.get(p) == "R8")
    r78_unres = sorted(p for p in prs if pr_class.get(p) == "R7/R8" and p not in gclass)
    others = sorted(p for p in prs if p not in r7s and p not in r8s and p not in r78_unres)  # all R1-R6
    r7_fed = sorted(strong[strong.syn_from_R7R8 > strong.syn_from_R1R6].root)
    r16_fed = sorted(strong[strong.syn_from_R1R6 >= strong.syn_from_R7R8].root)
    spec = {"meshes": [{"kind": "brain", "alpha": 0.15}],
            "neurons": [{"name": f"R7-fed bridge cells (nsyn >= {MIN_STRENGTH})", "color": "#e6194b", "ids": r7_fed},    # red
                        {"name": f"R1-R6-fed bridge cells (nsyn >= {MIN_STRENGTH})", "color": "#ffe119", "ids": r16_fed},  # yellow
                        {"name": "R7 and likely R7", "color": "#ff9f1c", "ids": sorted(r7s + r78_unres)},  # orange: grouped R7 + SeaTable R7-8
                        {"name": "R8s", "color": "#a0522d", "ids": r8s},                                   # brown (only if any)
                        {"name": "R1-R6 PRs", "color": "#e040fb", "ids": others},                          # magenta
                        {"name": "MeTu cells", "color": "#1f77d6", "ids": sorted(metus)}],
            "view": "anterior", "layout": "xy-3d", "image": True, "select": f"R7-fed bridge cells (nsyn >= {MIN_STRENGTH})",
            "scale": 8000}
    if centre is not None:
        spec["position"] = [int(round(v)) for v in centre]
    spec["neurons"] = [n for n in spec["neurons"] if n["ids"]]  # no empty layers (e.g. R8s when none feeds a bridge)
    state = nglink.build(spec)
    state["layers"][0]["name"] = "EM"
    # camera from the user's saved view (tmp/user_view_state.json, written by tmp/read_state_view.py) if present
    view_path = os.path.join(HERE, "..", "tmp", "user_view_state.json")
    if os.path.exists(view_path):
        uv = json.load(open(view_path))
        for k in ("position", "projectionOrientation", "projectionScale", "layout"):
            if uv.get(k) is not None:
                state[k] = uv[k]
        state["showSlices"] = bool(uv.get("showSlices", False))
        print(f"view taken from the saved state: position {np.round(state['position']).astype(int).tolist()}, "
              f"scale {state['projectionScale']:.0f}, layout {state['layout']}")
    if "--legend" in sys.argv:  # axis legend only on request (user, 2026-10-01); recipe kept in axis_legend()
        state["layers"] += axis_legend(np.array(state["position"], float), state["projectionOrientation"], state["projectionScale"])
    url = nglink.to_url(state)
    open(fig("pr_metu_bridge_ng.txt"), "w").write(url + "\n")
    json.dump(state, open(fig("pr_metu_bridge_state.json"), "w"), indent=1)  # paste into the JSON editor if the URL fails
    print(f"\n{len(strong)} bridge cells, {len(prs)} PRs, {len(metus)} MeTus; {len(url)} chars")
    print(url)


if __name__ == "__main__":
    main()
