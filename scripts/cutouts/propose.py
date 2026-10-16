"""Propose a synapse-annotation cutout (subvolume) around photoreceptor terminals and draw it.

Convention from creating_synapse_cut_outs.ipynb: cutouts are axis-aligned boxes in the 16x16x45 nm
voxel frame, drawn as neuroglancer bounding-box annotations over the EM image; the notebook used
2 um cubes (+-1000 nm around a point).

Steps (read-only, throttled):
  1. live synapse query, pre_pt_root_id in the group's current roots -> presynaptic sites (nm);
  2. densest neighbourhood: the R1-R6 site with the most other R1-R6 sites within the cube half-size
     -> cube centre (rounded to voxels); also the box enclosing all R1-R6 sites of the group (5-95 pct);
  3. every predicted synapse inside the cube (spatial query on ctr_pt_position) -> counts;
  4. Spelunker link: EM image, the 8 PRs (R1-R6 orange, R7 magenta, R8 cyan), an annotation layer with
     the cube, the enclosing box, and the PR presynaptic sites as points; xy-3d layout centred on the cube.
Usage: python faab_cutout_propose.py <group_tsv> [cube_um]
  group_tsv: tab-separated root_id_orig, class, root_id_curr, xyz (no header)
Outputs: figures/cutout_<name>_state.json (paste into neuroglancer if the link is too long),
         figures/cutout_<name>_ng.txt, figures/cutout_<name>_synapses.tsv
"""
from faab.config import REPO
import datetime
import json
import os
import sys
import time
import urllib.parse

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from scipy.spatial import cKDTree
from faab.sources.cave import DATASTACK, throttle
from faab.partners.top10_xyz import VOX

HERE = REPO
OUT = os.path.join(HERE, "figures")
SEG = "graphene://middleauth+https://cave.fanc-fly.com/segmentation/table/wclee_aedes_brain"
IMG = "precomputed://gs://zetta_lee_mosquito_001_alignment_central/img/v2_sharded"
DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}
COLORS = {"R1-R6": "#ff9f1c", "R1–R6": "#ff9f1c", "R7": "#e040fb", "R8": "#00c2d1"}


def vox(p_nm):
    return [round(float(v), 1) for v in (np.asarray(p_nm, float) / VOX)]


def main():
    path = sys.argv[1]
    cube_um = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    name = os.path.splitext(os.path.basename(path))[0] + f"_{cube_um:g}um"
    prs = pd.read_csv(path, sep="\t", header=None, dtype=str, names=["root_id_orig", "class", "root_id_curr", "xyz"]).fillna("")
    prs["class"] = prs["class"].str.replace("–", "-")
    roots = [int(r) for r in prs.root_id_curr]
    client = caveclient.CAVEclient(DATASTACK)
    now = datetime.datetime.now(datetime.timezone.utc)
    t = throttle(time.time())
    latest = client.chunkedgraph.is_latest_roots(roots, timestamp=now)
    for r, ok in zip(list(roots), latest):
        if not ok:
            t = throttle(t)
            new = [int(x) for x in np.atleast_1d(client.chunkedgraph.get_latest_roots(r))]
            print(f"root {r} expired -> {new} (using the first)", flush=True)
            roots[roots.index(r)] = new[0]
    prs["root_used"] = [str(r) for r in roots]
    cls_of = dict(zip(roots, prs["class"]))

    t = throttle(t)
    syn = client.materialize.live_query("synapses_v2", timestamp=now, filter_in_dict={"pre_pt_root_id": roots})
    syn = syn[syn.pre_pt_root_id != syn.post_pt_root_id].copy()
    syn["class"] = syn.pre_pt_root_id.map(cls_of)
    pts = np.vstack(syn.pre_pt_position.values).astype(float)
    print(f"{len(syn)} presynaptic sites of the {len(roots)} PRs at {now:%Y-%m-%d %H:%M} UTC: "
          f"{syn['class'].value_counts().to_dict()}", flush=True)

    # existing cutouts (inventory 2026-09-25) are 125 x 125 x 50 voxels = 2.0 x 2.0 x 2.25 um:
    # x/y half = cube_um * 500 nm; z half = cube_um * 12.5 sections * 45 nm (25 sections for 2 um)
    half = np.array([cube_um * 500.0, cube_um * 500.0, cube_um * 12.5 * VOX[2]])
    r16 = syn["class"] == "R1-R6"
    p16 = pts[r16.values]
    tree = cKDTree(p16 / half)  # scale so the box becomes a unit cube -> Chebyshev radius 1
    counts = np.array([len(tree.query_ball_point(p / half, 1.0, p=np.inf)) for p in p16])
    best = p16[np.argmax(counts)]
    centre_nm = np.round(best / VOX) * VOX
    lo, hi = centre_nm - half, centre_nm + half
    inside = np.all((pts >= lo) & (pts <= hi), axis=1)
    print(f"cube {cube_um} um: centre voxel {vox(centre_nm)} ({int(counts.max())} R1-R6 sites within, "
          f"{int(inside.sum())} PR sites of any class): {syn[inside]['class'].value_counts().to_dict()}", flush=True)
    enc_lo = np.percentile(p16, 5, axis=0)
    enc_hi = np.percentile(p16, 95, axis=0)
    print(f"box enclosing 5-95 pct of all R1-R6 sites: {vox(enc_lo)} -> {vox(enc_hi)}, "
          f"size {np.round((enc_hi - enc_lo) / 1000, 1).tolist()} um", flush=True)

    # every predicted synapse inside the cube
    t = throttle(t)
    all_in = None
    try:
        all_in = client.materialize.live_query("synapses_v2", timestamp=now,
                                               filter_spatial_dict={"ctr_pt_position": [lo.tolist(), hi.tolist()]})
        c = np.vstack(all_in.ctr_pt_position.values).astype(float)
        keep = np.all((c >= lo) & (c <= hi), axis=1)
        all_in = all_in[keep]
        n_pr = int(all_in.pre_pt_root_id.isin(roots).sum())
        print(f"all predicted synapses inside the cube: {len(all_in)} ({n_pr} with a group PR presynaptic, "
              f"{all_in.pre_pt_root_id.nunique()} distinct presynaptic roots)", flush=True)
        all_in.to_csv(os.path.join(OUT, f"cutout_{name}_synapses.tsv"), sep="\t", index=False)
    except Exception as e:
        print("spatial synapse query failed:", str(e)[:120], flush=True)

    # ---- neuroglancer state ----
    seg_layer = {"type": "segmentation", "source": SEG, "segments": [str(r) for r in roots],
                 "segmentColors": {str(r): COLORS.get(c, "#ffffff") for r, c in cls_of.items()},
                 "name": f"PRs {name}", "selectedAlpha": 0.3, "objectAlpha": 0.9}
    anns = [{"type": "axis_aligned_bounding_box", "id": "cube", "pointA": vox(lo), "pointB": vox(hi),
             "description": f"proposed cutout {cube_um:g} x {cube_um:g} x {cube_um * 1.125:g} um "
                            f"({int(round(cube_um * 62.5))} x {int(round(cube_um * 62.5))} x {int(round(cube_um * 25))} voxels)"},
            {"type": "axis_aligned_bounding_box", "id": "enclosing", "pointA": vox(enc_lo), "pointB": vox(enc_hi),
             "description": "5-95 pct box of all R1-R6 presynaptic sites"}]
    for i, (p, c) in enumerate(zip(pts, syn["class"])):
        anns.append({"type": "point", "id": f"s{i}", "point": vox(p), "description": f"{c} presynaptic site"})
    ann_layer = {"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                 "annotations": anns, "name": "cutout + PR presynaptic sites", "annotationColor": "#ffff00"}
    from faab import nglink  # brain mesh layer (faint) after the EM layer, as in every other FAAB link
    state = {"dimensions": DIMS, "position": vox(centre_nm), "crossSectionScale": 1.0, "projectionScale": 20000,
             "layers": [{"type": "image", "source": IMG, "name": "EM"}, nglink.mesh_layer("brain", alpha=0.15),
                        seg_layer, ann_layer],
             "showSlices": False, "layout": "xy-3d",
             "selectedLayer": {"layer": "cutout + PR presynaptic sites", "visible": True, "size": 400}}
    json.dump(state, open(os.path.join(OUT, f"cutout_{name}_state.json"), "w"), indent=1)
    url = "https://spelunker.cave-explorer.org/#!" + urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    open(os.path.join(OUT, f"cutout_{name}_ng.txt"), "w").write(url + "\n")
    print(f"link length {len(url)} chars; state json in figures/cutout_{name}_state.json")
    print(url)
    print("DONE")


if __name__ == "__main__":
    main()
