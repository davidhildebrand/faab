"""EM cross-sections of a PR group's axon bundle with the eight axons colored (comparison point for Brammer 1970's
retina cross-sections). For the requested positions along the bundle axis (s in um, lamina-positive; default the
compact bundle at -30 and -55 um) a WIN x WIN um window of the EM image (mip closest to 16 nm) and of the segmentation
(supervoxels -> current roots, one throttled get_roots call) is read in the xy plane through the bundle's skeleton
center; the eight roots are tinted (pair colors from the terminal analysis, R7 red, R8 green) and labeled.
Output figures/pr_bundle_em_group<g>_s<s>.png per slice. Read-only, <= 1 CAVE call / 5 s.
Usage: python faab_pr_bundle_em_slice.py <group> [s1 s2 ...]"""
import json
import os
import sys
import time

import numpy as np
if not hasattr(np, "row_stack"):
    np.row_stack = np.vstack
import pandas as pd
import caveclient
from cloudvolume import CloudVolume
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from faab.clients.cave import DATASTACK, throttle
from faab.pr.pairing_demo import analyse_group, PAIR_COL, R7_COL, R8_COL, OUT, TMP
from faab.nglink import IMG
from faab.config import fig

VOX = np.array([16.0, 16.0, 45.0])
WIN_UM = 7.0


def main():
    g = int(sys.argv[1])
    S = [float(a) for a in sys.argv[2:]] or [-30.0, -55.0]
    R = analyse_group(g, verbose=False)
    axis, c, outer, inner, names = (R[k] for k in ("axis", "c", "outer", "inner", "names"))
    pasted = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    now = json.load(open(fig("pr_footprint_roots.json")))["root_used"]
    old2new = {r.root_id_curr: int(n) for r, n in zip(pasted.itertuples(), now) if int(r.group) == g}
    label, color = {}, {}
    for i, r in enumerate(outer):
        rt = old2new.get(r, int(r))
        label[rt] = names.get(i, "R1-R6")
        color[rt] = PAIR_COL.get(label[rt], "#888888")
    for r, k in inner.items():
        rt = old2new.get(r, int(r))
        label[rt], color[rt] = k, (R7_COL if k == "R7" else R8_COL)
    allpts = np.vstack([R["neu"][r].nodes[["x", "y", "z"]].values for r in outer + list(inner)])
    s_all = (allpts - c) @ axis

    client = caveclient.CAVEclient(DATASTACK)
    seg = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=0)
    img = CloudVolume(IMG.replace("precomputed://", ""), use_https=True, progress=False, fill_missing=True, bounded=False)
    res_img = [tuple(sc["resolution"]) for sc in img.info["scales"]]
    mip_img = int(np.argmin([abs(r[0] - 16) for r in res_img]))
    img = CloudVolume(IMG.replace("precomputed://", ""), use_https=True, progress=False, fill_missing=True, bounded=False, mip=mip_img)
    rimg = np.array(res_img[mip_img], float)
    print(f"image mip {mip_img} at {rimg} nm; segmentation at {seg.resolution} nm", flush=True)
    half_seg = np.round(WIN_UM * 500 / np.array(seg.resolution)[:2]).astype(int)
    half_img = np.round(WIN_UM * 500 / rimg[:2]).astype(int)

    t = throttle(time.time())
    panels = []
    for s in S:
        sel = np.abs(s_all - s) <= 2.0
        if sel.sum() < 4:
            print(f"  s = {s}: no skeleton nodes nearby, skipped", flush=True)
            continue
        ctr_nm = allpts[sel].mean(0) * 1000.0
        cz = int(round(ctr_nm[2] / seg.resolution[2]))
        cs = np.round(ctr_nm[:2] / np.array(seg.resolution)[:2]).astype(int)
        ci = np.round(ctr_nm[:2] / rimg[:2]).astype(int)
        czi = int(round(ctr_nm[2] / rimg[2]))
        S2 = np.asarray(seg[cs[0] - half_seg[0]:cs[0] + half_seg[0], cs[1] - half_seg[1]:cs[1] + half_seg[1], cz:cz + 1]).squeeze()
        E = np.asarray(img[ci[0] - half_img[0]:ci[0] + half_img[0], ci[1] - half_img[1]:ci[1] + half_img[1], czi:czi + 1]).squeeze()
        svs = [int(v) for v in np.unique(S2) if v != 0]
        t = throttle(t)
        roots = np.asarray(client.chunkedgraph.get_roots(svs)).astype(np.int64)
        sv2root = dict(zip(svs, roots.tolist()))
        Rm = np.vectorize(lambda v: sv2root.get(int(v), 0))(S2) if S2.size else S2
        panels.append((s, E, Rm, ctr_nm, cz))
        found = sorted({label[r] for r in np.unique(Rm) if r in label})
        print(f"  s = {s:+.0f} um: z slice {cz}, {len(svs)} supervoxels, cells found: {found}", flush=True)

    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 5.4 * len(panels)), squeeze=False)
    for (s, E, Rm, ctr_nm, cz), (A, B) in zip(panels, axes):
        ext = [-WIN_UM / 2, WIN_UM / 2, WIN_UM / 2, -WIN_UM / 2]
        A.imshow(E.T, cmap="gray", extent=ext)
        B.imshow(E.T, cmap="gray", extent=ext)
        over = np.zeros(Rm.T.shape + (4,), float)
        for rt, col in color.items():
            m = (Rm.T == rt)
            if m.any():
                over[m] = list(to_rgb(col)) + [0.45]
                ys, xs = np.nonzero(m)
                x = (xs.mean() / Rm.T.shape[1] - 0.5) * WIN_UM
                y = (ys.mean() / Rm.T.shape[0] - 0.5) * WIN_UM
                B.text(x, y, label[rt], color="w", fontsize=8, ha="center", va="center", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.9))
        B.imshow(over, extent=ext)
        for axx in (A, B):
            axx.set_xticks([]); axx.set_yticks([])
        A.set_title(f"group {g}, s = {s:+.0f} um: EM (xy, z = {cz})", fontsize=10)
        B.set_title("the eight axons tinted (pair colors; R7 red, R8 green)", fontsize=10)
        A.plot([-WIN_UM / 2 + 0.4, -WIN_UM / 2 + 2.4], [WIN_UM / 2 - 0.4] * 2, color="w", lw=2)
        A.text(-WIN_UM / 2 + 1.4, WIN_UM / 2 - 0.75, "2 um", color="w", fontsize=8, ha="center")
    fig.suptitle(f"FAAB group {g}: axon bundle cross-sections (xy plane through the skeleton center; axis ~30 deg off z)", fontsize=11)
    fig.tight_layout()
    for s, *_ in panels:
        pass
    png = fig(f"pr_bundle_em_group{g}.png")
    fig.savefig(png, dpi=150)
    print("saved", png)


if __name__ == "__main__":
    main()
