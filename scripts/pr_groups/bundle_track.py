"""High-resolution tracking of the eight PR axons of a group through the compact bundle (David, 2026-10-13: 'do the
flattening test the right way').

Along the group's bundle axis (frame from faab_pr_pairing_demo.analyse_group, cached skeletons) a 6 x 6 x 6 um mip-1
cutout (32 x 32 x 45 nm voxels) is read every STEP um from the retina end to the lamina end. In each cutout every
supervoxel is mapped to its root (throttled get_roots in batches) and for each of the group's eight CURRENT roots
(figures/pr_footprint_roots.json, 2026-10-10) the voxel-weighted centroid is taken. Centroids are expressed in the
frame (s along the axis, t in the plane, d from the plane) and, flattened, relative to the mean of the roots present
in that slice. Outputs figures/pr_bundle_track_group<g>.tsv (per slice per cell: s, t, d, u, v offsets, angle,
radius, voxels) and figures/pr_bundle_track_group<g>.png. Read-only; CAVE <= 1 call / 5 s.
Usage: python faab_pr_bundle_track.py [group] [--step 2]"""
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

from faab.sources.cave import DATASTACK, throttle
from faab.pr.bundles import read_box
from faab.pr.pairing_demo import analyse_group, PAIR_COL, OUT, TMP

VOX = np.array([16.0, 16.0, 45.0])
RES1 = np.array([32.0, 32.0, 45.0])      # mip-1 voxel size (nm)
HALF_UM = 3.0                              # cutout half-size: 6 x 6 x 6 um
STEP = 2.0


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    g = int(args[0]) if args else 1
    step = float(sys.argv[sys.argv.index("--step") + 1]) if "--step" in sys.argv else STEP
    R = analyse_group(g, verbose=False)
    axis, that, nrm, c, s_end, outer, inner, names = (R[k] for k in ("axis", "that", "nrm", "c", "s_end", "outer", "inner", "names"))
    s_lo = min(((R["neu"][r].nodes[["x", "y", "z"]].values - c) @ axis).min() for r in outer)
    pasted = pd.read_csv(os.path.join(TMP, "pr_groups_pasted.tsv"), sep="\t", dtype=str)
    now = json.load(open(os.path.join(OUT, "pr_footprint_roots.json")))["root_used"]
    rows = [(r.root_id_curr, int(n), r._3) for r, n in zip(pasted.itertuples(), now) if int(r.group) == g]
    old2new = {o: n for o, n, _ in rows}
    label = {}
    for i, r in enumerate(outer):
        label[old2new.get(r, int(r))] = names.get(i, "R1-R6")
    for r, k in inner.items():
        label[old2new.get(r, int(r))] = k
    roots = list(label)
    print(f"group {g}: tracking {len(roots)} roots from s = {s_lo:.0f} to {s_end:.0f} um every {step} um", flush=True)

    client = caveclient.CAVEclient(DATASTACK)
    vol = CloudVolume(client.info.segmentation_source(), use_https=True, agglomerate=False, progress=False,
                      fill_missing=True, bounded=False, mip=1)
    half_vox = np.round(HALF_UM * 1000.0 / RES1).astype(int)
    S = np.arange(s_lo + 3, s_end - 1, step)
    # box centers follow the bundle: mean skeleton position of the cells within +-step of each s (the global axis
    # is straight, the bundle bends by ~10 um, a 6-um box centered on the axis misses it)
    allpts = np.vstack([R["neu"][r].nodes[["x", "y", "z"]].values for r in outer + list(inner)])
    s_all = (allpts - c) @ axis
    centers = {}
    for s in S:
        sel = np.abs(s_all - s) <= step
        centers[float(s)] = allpts[sel].mean(0) if sel.sum() >= 4 else c + s * axis
    # --extend <um>: continue toward the retina beyond the skeletons, along the direction of the first 12 um of
    # skeleton centers (David, 2026-10-14: "extended checks towards the retina")
    extend = float(sys.argv[sys.argv.index("--extend") + 1]) if "--extend" in sys.argv else 0.0
    if extend > 0:
        s0 = float(S[0])
        s1 = float(S[min(6, len(S) - 1)])
        direction = centers[s1] - centers[s0]
        direction /= np.linalg.norm(direction)
        ext = [s0 - k * step for k in range(1, int(extend / step) + 1)]
        for s in ext:
            centers[float(s)] = centers[s0] - (s0 - s) * direction
        S = np.concatenate([sorted(ext), S])
        print(f"  extended {extend:.0f} um toward the retina along the bundle direction (s = {S[0]:.0f} .. {s0:.0f})", flush=True)
    boxes = {}
    t0 = time.time()
    t = throttle(time.time())
    sv2root = {}
    if extend > 0 and "--adaptive" in sys.argv:
        # adaptive: walk toward the retina one box at a time, re-centering each box on the centroid of the group's
        # cells found in the previous one (one throttled get_roots per box); stop after 3 empty boxes
        ext_s = [s for s in S if s < float(S[0]) + extend + 1e-6 and s < min(centers) + extend + 1e-6]
        S_main = [s for s in S if s not in set(ext_s)]
        s0 = float(S_main[0])
        pos = centers[s0].copy()
        direction = centers[float(S_main[min(6, len(S_main) - 1)])] - centers[s0]
        direction /= np.linalg.norm(direction)
        empty, s = 0, s0
        last_cen = centers[s0].copy()
        S_ext = []
        while s > s0 - extend and empty < 3:
            s -= step
            pos = pos - step * direction
            b = read_box(vol, pos, half_vox, RES1)
            svs = sorted(sv for sv in b if sv not in sv2root)
            if svs:
                t = throttle(t)
                rt = np.asarray(client.chunkedgraph.get_roots(svs)).astype(np.int64)
                sv2root.update(dict(zip(svs, rt.tolist())))
            pts = [(n, np.array(cm)) for sv, (n, cm) in b.items() if sv2root.get(sv) in label]
            if pts:
                w = np.array([n for n, _ in pts], float)
                cen = np.sum([n * p for n, p in pts], axis=0) / w.sum()
                d_new = last_cen - cen  # direction the found cells are heading (toward the retina)
                if np.linalg.norm(d_new) > 0.5:
                    direction = 0.5 * direction + 0.5 * d_new / np.linalg.norm(d_new)
                    direction /= np.linalg.norm(direction)
                last_cen, pos = cen, cen.copy()
                empty = 0
            else:
                empty += 1
            boxes[float(s)] = b
            centers[float(s)] = pos.copy()
            S_ext.append(float(s))
            print(f"  adaptive s = {s:.0f}: {len(pts)} cell fragments found", flush=True)
        S = np.array(sorted(S_ext) + list(S_main))
    for k, s in enumerate(S):
        if float(s) in boxes:
            continue
        boxes[float(s)] = read_box(vol, centers[float(s)], half_vox, RES1)
        if k % 10 == 9:
            print(f"  {k + 1}/{len(S)} cutouts, {time.time() - t0:.0f} s", flush=True)
    svs = sorted({sv for b in boxes.values() for sv in b if sv not in sv2root})
    print(f"{len(svs)} supervoxels to map", flush=True)
    for i in range(0, len(svs), 50000):
        chunk = svs[i:i + 50000]
        t = throttle(t)
        rt = np.asarray(client.chunkedgraph.get_roots(chunk)).astype(np.int64)
        sv2root.update(dict(zip(chunk, rt.tolist())))
    rec = []
    for s, b in boxes.items():
        per = {}
        for sv, (n, cm) in b.items():
            rt = sv2root.get(sv)
            if rt in label:
                per.setdefault(rt, []).append((n, np.array(cm)))
        for rt, lst in per.items():
            w = np.array([n for n, _ in lst], float)
            cm = np.sum([n * p for n, p in lst], axis=0) / w.sum()
            p = cm - c
            rec.append(dict(s_box=s, root=rt, cell=label[rt], vox=int(w.sum()), s=float(p @ axis), t=float(p @ that),
                            d=float(p @ nrm)))
    df = pd.DataFrame(rec)
    # flatten: offsets from the mean of the cells present in each slice (only slices with all cells)
    full = df.groupby("s_box").root.nunique()
    df["complete"] = df.s_box.map(full) == len(roots)
    m = df.groupby("s_box")[["t", "d"]].transform("mean")
    df["u"], df["v"] = df.t - m.t, df.d - m.d
    df["angle"] = np.degrees(np.arctan2(df.v, df.u))
    df["radius"] = np.hypot(df.u, df.v)
    tag_ext = "_ext" if extend > 0 else ""
    df.to_csv(os.path.join(OUT, f"pr_bundle_track_group{g}{tag_ext}.tsv"), sep="\t", index=False)
    print(f"{df.s_box.nunique()} slices, {int(df.complete.sum() / max(len(roots), 1))} with all {len(roots)} cells; "
          f"{time.time() - t0:.0f} s", flush=True)

    fig, ax = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    (A, B), (C, D) = ax
    col = {rt: PAIR_COL.get(label[rt], "#888780") for rt in roots}
    seen = {}
    for rt in roots:
        sub = df[(df.root == rt) & df.complete].sort_values("s_box")
        lab = label[rt]
        ls = "-" if lab not in seen else "--"
        seen[lab] = True
        A.plot(sub.s_box, sub.angle, ls, color=col[rt], lw=1.4, label=f"{lab} ...{str(rt)[-6:]}")
        B.plot(sub.s_box, sub.radius, ls, color=col[rt], lw=1.4)
        C.plot(sub.u, sub.v, ls, color=col[rt], lw=1, alpha=0.8)
        if len(sub):
            C.scatter(sub.u.iloc[0], sub.v.iloc[0], s=60, color=col[rt], edgecolor="k", zorder=5)
            C.annotate(lab, (sub.u.iloc[0], sub.v.iloc[0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    for axx in (A, B):
        axx.axvspan(s_end - 35, s_end, color="#EF9F27", alpha=0.15)
        axx.set_xlabel("s (um) along the bundle axis, lamina to the right (shaded: terminal zone)")
    A.set_ylabel("angle around the per-slice bundle center (deg)")
    A.set_title(f"A  group {g}, cutout centroids every {step:g} um: angle of each axon around the bundle center")
    A.legend(fontsize=8, ncol=2)
    B.set_ylabel("distance from the bundle center (um)")
    B.set_title("B  distance from the bundle center")
    C.axhline(0, color="#aaa", lw=0.6); C.axvline(0, color="#aaa", lw=0.6)
    C.set_aspect("equal")
    C.set_xlabel("u (um)"); C.set_ylabel("v (um)")
    C.set_title("C  flattened tracks (offset from the slice center), dot = retina end")
    # D: cyclic order per slice as a raster: cells sorted by angle
    comp = df[df.complete]
    order_rows = []
    for s, sub in comp.groupby("s_box"):
        sub = sub.sort_values("angle")
        order_rows.append((s, list(sub.cell), list(sub.root)))
    for s, cells, rts in order_rows:
        for pos_i, (cl, rt) in enumerate(zip(cells, rts)):
            D.scatter(s, pos_i, s=40, marker="s", color=col[rt])
    D.set_yticks(range(len(roots)))
    D.set_ylabel("rank by angle (counterclockwise)")
    D.set_xlabel("s (um)")
    D.set_title("D  cyclic order around the center at each slice (colors = fly pair; red R7, green R8)")
    fig.suptitle(f"Group {g}: eight axons tracked with 6-um mip-1 cutouts along the bundle axis (frame of the pairing analysis)")
    png = os.path.join(OUT, f"pr_bundle_track_group{g}{tag_ext}.png")
    fig.savefig(png, dpi=120)
    print("saved", png)
    for s, cells, rts in order_rows[::max(1, len(order_rows) // 8)]:
        print(f"  s = {s:5.0f}: " + " > ".join(cells))


if __name__ == "__main__":
    main()
