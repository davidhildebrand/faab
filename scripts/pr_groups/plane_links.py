"""Spelunker links that DRAW the pairing frame on real groups: the bundle axis, the mirror plane (as a rectangle of
line annotations containing the axis), the plane normal, and one line per mutual pair crossing the plane, with the
R1-R6 cells colored by fly pair, R7 red, R8 green. Also a 'center' marker (per-slice bundle center is not drawn; the
axis line IS the center line of the compact bundle). Network-free: frame from faab_pr_pairing_demo.analyse_group on
the cached skeletons; segment ids = the roots resolved at the sheet xyz on 2026-10-10 (figures/pr_footprint_roots.json),
mapped onto the 2026-09-30 skeleton roots by their position in tmp/pr_groups_pasted.tsv.
Output: figures/pr_plane_ng_group<g>.txt per group. Usage: python faab_pr_plane_links.py 1 3"""
from faab.config import REPO
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from faab.pr.pairing_demo import analyse_group, PAIR_COL, R7_COL, R8_COL, OUT, TMP
from faab.nglink import build, to_url
from faab.config import fig

VOX = np.array([16.0, 16.0, 45.0])
HERE = REPO


def current_roots(g):
    pasted = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    now = json.load(open(fig("pr_footprint_roots.json")))["root_used"]
    rows = [(r.root_id_curr, str(n), r._3) for r, n in zip(pasted.itertuples(), now) if int(r.group) == g]
    return rows  # (skeleton-era root, current root, class) in sheet order


def vox(p_um):
    return (np.asarray(p_um, float) * 1000.0 / VOX).tolist()


def main():
    groups = [int(a) for a in sys.argv[1:]] or [1, 3]
    for g in groups:
        R = analyse_group(g, verbose=False)
        axis, that, nrm, c, ct, s_end = (R[k] for k in ("axis", "that", "nrm", "c", "ct", "s_end"))
        outer, inner, names, pairs, pos = (R[k] for k in ("outer", "inner", "names", "pairs", "pos"))
        s_lo = min(((R["neu"][r].nodes[["x", "y", "z"]].values - c) @ axis).min() for r in outer)
        # axis line through the bundle, from the retina end to the lamina end
        a0, a1 = c + s_lo * axis, c + (s_end + 5) * axis
        # plane rectangle: contains the axis; spans the terminal zone along s and +-18 um along t
        T = 18.0
        corners = [ct + (s - (ct - c) @ axis) * axis + t * that for s, t in
                   ((s_end - 35, -T), (s_end + 8, -T), (s_end + 8, T), (s_end - 35, T))]
        plane_lines = [[vox(corners[i]), vox(corners[(i + 1) % 4])] for i in range(4)]
        plane_lines.append([vox(ct), vox(ct + 10 * nrm)])  # normal arrow (d direction)
        axis_lines = [[vox(a0), vox(a1)]]
        pair_lines = []
        for i, j in pairs:
            pa = ct + pos[outer[i]][0] * that + pos[outer[i]][1] * nrm
            pb = ct + pos[outer[j]][0] * that + pos[outer[j]][1] * nrm
            pair_lines.append([vox(pa), vox(pb)])
        rows = current_roots(g)
        old2new = {o: n for o, n, _ in rows}
        colors = {}
        for i, r in enumerate(outer):
            colors[old2new.get(r, r)] = PAIR_COL.get(names.get(i, ""), "#888888")
        for r, k in inner.items():
            colors[old2new.get(r, r)] = R7_COL if k == "R7" else R8_COL
        spec = {"image": True, "layout": "3d", "view": "anterior", "position": vox(ct), "scale": 4500,
                "meshes": [{"kind": "brain", "alpha": 0.08}],
                "neurons": [{"name": f"group {g} PRs (pair colors; R7 red, R8 green)", "ids": list(colors), "colors": colors}],
                "annotations": [{"name": "bundle axis (center line)", "color": "#ffffff", "lines": axis_lines},
                                {"name": "mirror plane (+ normal)", "color": "#2a6fd6", "lines": plane_lines},
                                {"name": "mutual pairs", "color": "#ffd700", "lines": pair_lines,
                                 "points": [vox(ct)]}],
                "select": "mirror plane (+ normal)"}
        url = to_url(build(spec))
        out = fig(f"pr_plane_ng_group{g}.txt")
        open(out, "w").write(url + "\n")
        print(f"group {g}: plane theta {R['th']} deg, pairs {[(names.get(i, ''), names.get(j, '')) for i, j in pairs]}; "
              f"link {len(url)} chars -> {out}")


if __name__ == "__main__":
    main()
