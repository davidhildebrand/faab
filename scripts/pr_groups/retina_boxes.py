"""Follow a PR group from the top of its axon bundle up into the retina with LARGE cutouts (David, 2026-10-14: the
extended check toward the retina). Retinula cell bodies are 5-8 um wide, so the 6-um tracking boxes cannot hold the
eight cells there; here each step reads a BOX_UM cube (mip 1) centered on the centroid of the group's cells found in
the previous box, moving STEP_UM toward the retina along the direction between the last two centroids (start: the
last complete slice of the bundle tracking, direction = bundle axis). Per box: centroid, voxel count and position of
each of the eight current roots in the bundle frame (t, d), the ring order about the per-box mean, and an EM panel
(xy slice through the box center, cells tinted). Stops when fewer than 4 cells are found twice in a row.
Outputs figures/pr_retina_boxes_group<g>.tsv and figures/pr_retina_boxes_group<g>.png. Read-only; <= 1 CAVE call / 5 s.
Usage: python faab_pr_retina_boxes.py <group> [n_steps]"""
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

from faab.sources.cave import DATASTACK, throttle
from faab.pr.bundles import read_box
from faab.pr.pairing_demo import analyse_group, PAIR_COL, R7_COL, R8_COL, OUT, TMP
from faab.nglink import IMG

RES1 = np.array([32.0, 32.0, 45.0])
BOX_UM, STEP_UM = 14.0, 6.0


def main():
    g = int(sys.argv[1])
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    R = analyse_group(g, verbose=False)
    axis, that, nrm, c, outer, inner, names = (R[k] for k in ("axis", "that", "nrm", "c", "outer", "inner", "names"))
    pasted = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    now = json.load(open(os.path.join(OUT, "pr_footprint_roots.json")))["root_used"]
    old2new = {r.root_id_curr: int(n) for r, n in zip(pasted.itertuples(), now) if int(r.group) == g}
    label, color = {}, {}
    for i, r in enumerate(outer):
        rt = old2new.get(r, int(r))
        label[rt] = names.get(i, "R1-R6")
        color[rt] = PAIR_COL.get(label[rt], "#888888")
    for r, k in inner.items():
        rt = old2new.get(r, int(r))
        label[rt], color[rt] = k, (R7_COL if k == "R7" else R8_COL)
    # start: centroid of the last complete, compact slice of the bundle tracking (frame coords -> xyz)
    tr = pd.read_csv(os.path.join(OUT, f"pr_bundle_track_group{g}.tsv"), sep="\t")
    comp = tr[tr.complete].groupby("s_box").filter(lambda s: s.radius.max() <= 2.5)
    s_start = float(comp.s_box.min())
    sl = comp[comp.s_box == s_start]
    start = c + sl.s.mean() * axis + sl.t.mean() * that + sl.d.mean() * nrm
    direction = -axis.copy()  # toward the retina
    print(f"group {g}: starting at the bundle slice s = {s_start:.0f} um, {BOX_UM:.0f}-um boxes every {STEP_UM:.0f} um", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    seg = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    res_img = [tuple(sc["resolution"]) for sc in CloudVolume(IMG.replace("precomputed://", ""), use_https=True).info["scales"]]
    mip_img = int(np.argmin([abs(r[0] - 32) for r in res_img]))
    img = CloudVolume(IMG.replace("precomputed://", ""), use_https=True, progress=False, fill_missing=True, bounded=False, mip=mip_img)
    rimg = np.array(res_img[mip_img], float)
    half = np.round(BOX_UM * 500 / RES1).astype(int)
    t = throttle(time.time())
    sv2root, rows, panels = {}, [], []
    pos, last = start.copy(), start.copy()
    weak = 0
    for k in range(n_steps + 1):
        s_here = float((pos - c) @ axis)
        b = read_box(seg, pos, half, RES1)
        svs = sorted(sv for sv in b if sv not in sv2root)
        for i in range(0, len(svs), 50000):
            t = throttle(t)
            rt = np.asarray(client.chunkedgraph.get_roots(svs[i:i + 50000])).astype(np.int64)
            sv2root.update(dict(zip(svs[i:i + 50000], rt.tolist())))
        per = {}
        for sv, (n, cm) in b.items():
            rt = sv2root.get(sv)
            if rt in label:
                per.setdefault(rt, []).append((n, np.array(cm)))
        cents = {}
        for rt, lst in per.items():
            w = np.array([n for n, _ in lst], float)
            cents[rt] = (np.sum([n * p for n, p in lst], axis=0) / w.sum(), int(w.sum()))
        if cents:
            allc = np.mean([p for p, _ in cents.values()], axis=0)
            for rt, (p, n) in cents.items():
                q = p - allc
                rows.append(dict(step=k, s_um=round(s_here, 1), root=rt, cell=label[rt], vox=n,
                                 u=float(q @ that), v=float(q @ nrm), along=float(q @ axis),
                                 x_vox=int(p[0] * 1000 / 16), y_vox=int(p[1] * 1000 / 16), z_vox=int(p[2] * 1000 / 45)))
        # EM panel and the 2-D ring order from the single xy slice through the box center (3-D centroids of a
        # 14-um cube are smeared along the oblique axons; the slice is a true cross-section)
        cz = int(round(pos[2] * 1000 / rimg[2]))
        ci = np.round(pos[:2] * 1000 / rimg[:2]).astype(int)
        hi = np.round(BOX_UM * 500 / rimg[:2]).astype(int)
        E = np.asarray(img[ci[0] - hi[0]:ci[0] + hi[0], ci[1] - hi[1]:ci[1] + hi[1], cz:cz + 1]).squeeze()
        czs = int(round(pos[2] * 1000 / RES1[2]))
        cs = np.round(pos[:2] * 1000 / RES1[:2]).astype(int)
        S2 = np.asarray(seg[cs[0] - half[0]:cs[0] + half[0], cs[1] - half[1]:cs[1] + half[1], czs:czs + 1]).squeeze()
        Rm = np.vectorize(lambda v: sv2root.get(int(v), 0))(S2)
        c2d = {}
        for rt in label:
            ys, xs = np.nonzero(Rm == rt)
            if len(xs) >= 20:
                c2d[rt] = np.array([xs.mean(), ys.mean()]) * RES1[:2] / 1000.0
        if c2d:
            m2 = np.mean(list(c2d.values()), axis=0)
            order2 = sorted(c2d, key=lambda rt: np.degrees(np.arctan2(*(c2d[rt] - m2)[::-1])))
            for rt in c2d:
                rows.append(dict(step=k, s_um=round(s_here, 1), root=rt, cell=label[rt], vox=-1, u=float((c2d[rt] - m2)[0]),
                                 v=float((c2d[rt] - m2)[1]), along=np.nan, x_vox=-1, y_vox=-1, z_vox=czs, slice2d=True))
        else:
            order2 = []
        print(f"  step {k}: s = {s_here:+.0f} um, {len(cents)} cells in the box, {len(c2d)} in the central slice; "
              f"slice order " + " > ".join(label[rt] for rt in order2), flush=True)
        panels.append((k, s_here, E, Rm, len(c2d)))
        if len(cents) < 4:
            weak += 1
            if weak >= 2:
                break
        else:
            weak = 0
            last = allc.copy()
            # keep walking along the bundle axis (direction fixed), re-centered laterally on the cells found
            lateral = allc - ((allc - pos) @ direction) * direction - pos
            pos = pos + lateral + STEP_UM * direction
            continue
        pos = pos + STEP_UM * direction
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, f"pr_retina_boxes_group{g}.tsv"), sep="\t", index=False)

    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 5.4 * len(panels)), squeeze=False)
    ext = [-BOX_UM / 2, BOX_UM / 2, BOX_UM / 2, -BOX_UM / 2]
    for (k, s_here, E, Rm, n), (A, B) in zip(panels, axes):
        A.imshow(E.T, cmap="gray", extent=ext)
        B.imshow(E.T, cmap="gray", extent=ext)
        over = np.zeros(Rm.T.shape + (4,), float)
        for rt, col in color.items():
            m = Rm.T == rt
            if m.any():
                over[m] = list(to_rgb(col)) + [0.45]
                ys, xs = np.nonzero(m)
                B.text((xs.mean() / Rm.T.shape[1] - 0.5) * BOX_UM, (ys.mean() / Rm.T.shape[0] - 0.5) * BOX_UM, label[rt],
                       color="w", fontsize=8, ha="center", va="center", fontweight="bold",
                       bbox=dict(boxstyle="round,pad=0.15", fc=col, ec="none", alpha=0.9))
        B.imshow(over, extent=ext)
        for axx in (A, B):
            axx.set_xticks([]); axx.set_yticks([])
        A.set_title(f"step {k}: s = {s_here:+.0f} um (EM xy, {BOX_UM:.0f} um window)", fontsize=10)
        B.set_title(f"{n} of 8 cells in the box", fontsize=10)
        A.plot([-BOX_UM / 2 + 0.6, -BOX_UM / 2 + 5.6], [BOX_UM / 2 - 0.6] * 2, color="w", lw=2)
        A.text(-BOX_UM / 2 + 3.1, BOX_UM / 2 - 1.3, "5 um", color="w", fontsize=8, ha="center")
    fig.suptitle(f"FAAB group {g}: from the top of the bundle into the retina", fontsize=11)
    fig.tight_layout()
    png = os.path.join(OUT, f"pr_retina_boxes_group{g}.png")
    fig.savefig(png, dpi=130)
    print("saved", png)


if __name__ == "__main__":
    main()
