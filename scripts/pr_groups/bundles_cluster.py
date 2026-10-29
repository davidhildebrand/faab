"""Step 3b of the photoreceptor-group finder: cluster the bundle-zone census into ommatidial bundles (network-free).

Input: figures/pr_bundle_boxes.pkl written by faab_pr_bundles.py (per site/depth cutouts: supervoxel voxel counts
and centroids; supervoxel -> root map), figures/pr_lattice_known.tsv (known groups), pr_lattice_seatable_prs.tsv.
Per root and depth (+45/+60/+75/+90 um above the lamina) the voxel count and centroid are recomputed; a root is a
PR-like axon if it is present at >= MIN_DEPTHS depths, its centroid track runs within MAX_ANGLE_DEG of the mean
cartridge axis, and its slab volume lies in [LOW_VOX, HIGH_VOX] (window calibrated on the known PRs, printed).
Bundles: average-linkage clustering on the TRACK distance (mean centroid distance over shared depths, >= 2 shared),
which keeps axons that run in parallel together and separates bundles that only touch at one depth. The threshold
is calibrated on the known groups (intra-group vs nearest-other-group track distances are printed).
Outputs: figures/pr_bundles_roots.tsv (one row per PR-like root: cluster, xyz at the reference depth, known/SeaTable
labels), pr_bundles_clusters.tsv, pr_bundles_ng.txt / pr_bundles_state.json.
Left eye: --left reads/writes the pr_left_* files instead.
Usage: python faab_pr_bundles_cluster.py [LINK_UM] [--left]
"""
from faab.config import REPO
import datetime
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from faab.partners.top10_xyz import VOX
import faab.nglink as nglink

HERE = REPO
OUT = os.path.join(HERE, "figures")
LEFT = "--left" in sys.argv
PFX = "pr_left_" if LEFT else "pr_"
KNOWN_NAME = "known PRs (LRS left cartridges)" if LEFT else "known PRs (8 groups + LRS cartridges)"
CACHE = os.path.join(OUT, f"{PFX}bundle_boxes.pkl")
DEPTHS_UM = [45.0, 60.0, 75.0, 90.0]
REF_DEPTH = 75.0
MIN_HIT_VOX = 2000
MIN_DEPTHS = 2
MAX_ANGLE_DEG = 25.0
# per-slab volume window for one photoreceptor axon. Right: known PRs p5 47k, p10 70k, p90 203k. Left (cartridge 8,
# 2026-10-29): thinner, p5 21k, p50 38k, p90 68k -> same ratio to p5 (~0.53) gives 11k
LOW_VOX, HIGH_VOX = (11000, 600000) if LEFT else (25000, 600000)
_num = [a for a in sys.argv[1:] if not a.startswith("--")]
# left: 5 um put cartridge 8's R1-R6 into a 29-axon cluster, 3 um into 13, 2.5 um into 8 (its six + 2) - left bundles
# are packed tighter; right: 6 um merged adjacent known groups, 5 um did not
LINK_UM = float(_num[0]) if _num else (2.5 if LEFT else 5.0)
FAR = 1000.0
DIMS = {"x": [1.6e-08, "m"], "y": [1.6e-08, "m"], "z": [4.5e-08, "m"]}
PALETTE = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
           "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9"]


def main():
    cache = pickle.load(open(CACHE, "rb"))
    boxes, sv2root = cache["boxes"], cache["sv2root"]
    known = pd.read_csv(os.path.join(OUT, f"{PFX}lattice_known.tsv"), sep="\t", dtype={"r16_ids": str, "r78_ids": str})
    st_path = os.path.join(OUT, f"{PFX}lattice_seatable_prs.tsv")
    st = pd.read_csv(st_path, sep="\t", dtype=str) if os.path.exists(st_path) else pd.DataFrame(columns=["root", "type", "cart"])
    st_type = {r.root: f"SeaTable {r.type}" + (f" cart {r.cart}" if isinstance(r.cart, str) and r.cart not in ("", "nan") else "")
               for r in st.itertuples()}
    root_group, group_of = {}, {}
    for r in known.itertuples():
        for col, cls in (("r16_ids", "R1-R6"), ("r78_ids", "R7/R8")):
            for x in str(getattr(r, col)).split(","):
                if x and x != "nan":
                    root_group[x] = f"{r.cartridge} {cls}"
                    group_of[x] = r.cartridge
    axes = known[["ax_x", "ax_y", "ax_z"]].values
    mean_axis = axes.mean(0)
    mean_axis /= np.linalg.norm(mean_axis)
    # local axis: that of the nearest site (known or predicted, interpolated axes) at the reference depth
    pred = pd.read_csv(os.path.join(OUT, f"{PFX}lattice_predicted.tsv"), sep="\t")
    site_axes = np.vstack([axes, pred[["ax_x", "ax_y", "ax_z"]].values])
    site_axes /= np.linalg.norm(site_axes, axis=1, keepdims=True)
    site_lam = np.vstack([known[["lam_x", "lam_y", "lam_z"]].values, pred[["lam_x", "lam_y", "lam_z"]].values])
    site_ref = site_lam - site_axes * REF_DEPTH

    def local_axis(p):
        return site_axes[np.argmin(np.linalg.norm(site_ref - p, axis=1))]

    # ---- per root and depth ----
    per_depth = {d: {} for d in DEPTHS_UM}
    for (name, d), bx in boxes.items():
        per_depth[d].update(bx)
    acc = {}
    for d, svd in per_depth.items():
        for sv, (n, c) in svd.items():
            r = sv2root.get(sv, 0)
            if r:
                e = acc.setdefault((r, d), [0, np.zeros(3)])
                e[0] += n
                e[1] += n * np.array(c)
    tracks = {}
    for (r, d), (n, sc) in acc.items():
        if n >= MIN_HIT_VOX:
            tracks.setdefault(str(r), {})[d] = (n, sc / n)

    kn_vox = [n for r, dd in tracks.items() if r in root_group for n, _ in dd.values()]
    print(f"roots present: {len(tracks)}; known PRs present {sum(r in root_group for r in tracks)}/{len(root_group)}; known per-slab "
          f"voxels min/p5/p10/p50/p90/p95/max {np.percentile(kn_vox, [0, 5, 10, 50, 90, 95, 100]).round(0).astype(int).tolist()}")

    rows = []
    for r, dd in tracks.items():
        ds = sorted(dd)
        vox = [dd[d][0] for d in ds]
        cents = np.array([dd[d][1] for d in ds])
        if len(ds) >= 2:
            v = cents[-1] - cents[0]
            ax_loc = local_axis(cents.mean(0))
            ang = float(np.degrees(np.arccos(np.clip(abs(v @ ax_loc) / np.linalg.norm(v), -1, 1))))
        else:
            ang = np.nan
        med = float(np.median(vox))
        if len(ds) >= MIN_DEPTHS and ang <= MAX_ANGLE_DEG and LOW_VOX <= med <= HIGH_VOX:
            cls = "PR-like"
        elif len(ds) >= MIN_DEPTHS and ang <= MAX_ANGLE_DEG and med > HIGH_VOX:
            cls = "too big"
        elif len(ds) >= MIN_DEPTHS and ang <= MAX_ANGLE_DEG:
            cls = "too thin"
        elif len(ds) >= MIN_DEPTHS:
            cls = "oblique"
        else:
            cls = "single depth"
        ref = REF_DEPTH if REF_DEPTH in dd else min(ds, key=lambda x: abs(x - REF_DEPTH))
        rc = dd[ref][1]
        rows.append(dict(root=r, cls=cls, n_depths=len(ds), depths=",".join(f"+{int(d)}" for d in ds),
                         vox_per_depth=",".join(map(str, vox)), vox_med=int(med), angle_deg=round(ang, 1) if len(ds) >= 2 else np.nan,
                         ref_depth=int(ref), ref_x=rc[0], ref_y=rc[1], ref_z=rc[2],
                         known=root_group.get(r, ""), seatable=st_type.get(r, "")))
    df = pd.DataFrame(rows)
    for a_, v_ in zip("xyz", VOX):
        df[f"ref_vox_{a_}"] = (df[f"ref_{a_}"] * 1000.0 / v_).round(0).astype(int)
    df["xyz_ref"] = df.ref_vox_x.astype(str) + ", " + df.ref_vox_y.astype(str) + ", " + df.ref_vox_z.astype(str)
    kn = df[df.known != ""]
    print(f"classes: {df.cls.value_counts().to_dict()}; known PRs: {kn.cls.value_counts().to_dict()}")
    print("known PRs not PR-like: " + "; ".join(f"{r.known} {r.root} {r.cls} ({r.depths}; vox {r.vox_per_depth}; {r.angle_deg} deg)"
                                                for r in kn[kn.cls != "PR-like"].itertuples()))

    # ---- track distances among PR-like roots ----
    pr = df[df.cls == "PR-like"].reset_index(drop=True)
    n = len(pr)
    C = np.full((n, len(DEPTHS_UM), 3), np.nan)
    for i, r in enumerate(pr.root):
        for k, d in enumerate(DEPTHS_UM):
            if d in tracks[r]:
                C[i, k] = tracks[r][d][1]
    D = np.full((n, n), FAR)
    for k in range(len(DEPTHS_UM)):
        pass
    # mean over shared depths (vectorised per depth)
    S = np.zeros((n, n))
    N = np.zeros((n, n))
    for k in range(len(DEPTHS_UM)):
        has = ~np.isnan(C[:, k, 0])
        idx = np.where(has)[0]
        P = C[idx, k]
        dk = np.linalg.norm(P[:, None] - P[None], axis=2)
        S[np.ix_(idx, idx)] += dk
        N[np.ix_(idx, idx)] += 1
    ok = N >= 2
    D[ok] = S[ok] / N[ok]
    np.fill_diagonal(D, 0.0)
    D = np.minimum(D, D.T)

    # calibration: intra-group vs nearest other group
    gi = {g: [i for i, r in enumerate(pr.root) if group_of.get(r) == g] for g in sorted(set(group_of.values()))}
    gi = {g: v for g, v in gi.items() if len(v) >= 2}
    print("\ncalibration (track distance, um): per known group intra-member median / max, and min distance to a member of another group")
    for g, idx in gi.items():
        sub = D[np.ix_(idx, idx)]
        intra = sub[np.triu_indices(len(idx), 1)]
        others = [j for g2, v in gi.items() if g2 != g for j in v]
        inter = D[np.ix_(idx, others)].min() if others else np.nan
        print(f"  {g:3s}: n {len(idx)}; intra median {np.median(intra[intra < FAR]):.1f}, max {intra[intra < FAR].max() if (intra < FAR).any() else np.nan:.1f}; "
              f"nearest other group {inter:.1f}")

    # ---- clustering ----
    Z = linkage(squareform(D, checks=False), method="average")
    pr["cluster"] = fcluster(Z, t=LINK_UM, criterion="distance")
    cl = pr.groupby("cluster").agg(n_roots=("root", "size"), n_known=("known", lambda s: int((s != "").sum())),
                                   known_groups=("known", lambda s: ", ".join(sorted({x.split(" ")[0] for x in s if x}))),
                                   n_seatable=("seatable", lambda s: int((s != "").sum())),
                                   cx=("ref_x", "mean"), cy=("ref_y", "mean"), cz=("ref_z", "mean"),
                                   n_depths_med=("n_depths", "median"))
    cl["n_new"] = cl.n_roots - cl.n_known
    for c, sub in pr.groupby("cluster"):
        P = sub[["ref_x", "ref_y", "ref_z"]].values
        cl.loc[c, "spread_um"] = round(float(np.linalg.norm(P - P.mean(0), axis=1).max()), 1)
    for a_, v_ in zip("xyz", VOX):
        cl[f"c_vox_{a_}"] = (cl[f"c{a_}"] * 1000.0 / v_).round(0).astype(int)
    cl["xyz_centroid"] = cl.c_vox_x.astype(str) + ", " + cl.c_vox_y.astype(str) + ", " + cl.c_vox_z.astype(str)
    cl = cl.sort_values(["n_new", "n_roots"], ascending=False)
    pr = pr.merge(cl[["n_roots", "n_known", "n_new"]], left_on="cluster", right_index=True)
    pr.to_csv(os.path.join(OUT, f"{PFX}bundles_roots.tsv"), sep="\t", index=False)
    df.to_csv(os.path.join(OUT, f"{PFX}bundles_all_roots.tsv"), sep="\t", index=False)
    cl.to_csv(os.path.join(OUT, f"{PFX}bundles_clusters.tsv"), sep="\t")
    pd.set_option("display.width", 250)
    print(f"\naverage-linkage clusters at {LINK_UM} um: {len(cl)}; sizes {cl.n_roots.value_counts().sort_index().to_dict()}")
    print("clusters containing known PRs (calibration):")
    print(cl[cl.n_known > 0][["n_roots", "n_known", "n_new", "known_groups", "spread_um", "xyz_centroid"]].to_string())
    new = cl[(cl.n_known == 0) & (cl.n_roots >= 4)]
    print(f"\nnew bundles (no known PR, >= 4 axons): {len(new)}")
    print(new[["n_roots", "n_seatable", "spread_um", "n_depths_med", "xyz_centroid"]].to_string())

    # ---- link ----
    neurons = [{"name": KNOWN_NAME, "color": "#808080", "ids": sorted(root_group)}]
    top = new.index.tolist()[:len(PALETTE)]
    for k, c in enumerate(top):
        ids = sorted(pr[pr.cluster == c].root)
        neurons.append({"name": f"new bundle {k + 1} ({len(ids)} axons)", "color": PALETTE[k], "ids": ids})
    rest = sorted(pr[(pr.known == "") & ~pr.cluster.isin(top)].root)
    if rest:
        neurons.append({"name": "other new PR-like axons", "color": "#ffffff", "ids": rest})
    spec = {"meshes": [{"kind": "brain", "alpha": 0.15}], "neurons": neurons, "view": "anterior", "layout": "3d", "image": True,
            "scale": 12000, "position": [int(v) for v in (pr[["ref_x", "ref_y", "ref_z"]].values.mean(0) * 1000.0 / VOX)]}
    state = nglink.build(spec)
    state["layers"][0]["name"] = "EM"
    anns = [{"type": "point", "id": f"b{c}", "point": [int(cl.loc[c, "c_vox_x"]), int(cl.loc[c, "c_vox_y"]), int(cl.loc[c, "c_vox_z"])],
             "description": f"bundle {c}: {int(cl.loc[c, 'n_roots'])} axons, {int(cl.loc[c, 'n_known'])} known ({cl.loc[c, 'known_groups']})"}
            for c in cl.index if cl.loc[c, "n_roots"] >= 3]
    state["layers"].append({"type": "annotation", "source": {"url": "local://annotations", "transform": {"outputDimensions": DIMS}},
                            "annotations": anns, "name": f"bundle centroids (+{int(REF_DEPTH)} um above the lamina)", "annotationColor": "#ffe119"})
    url = nglink.to_url(state)
    open(os.path.join(OUT, f"{PFX}bundles_ng.txt"), "w").write(url + "\n")
    json.dump(state, open(os.path.join(OUT, f"{PFX}bundles_state.json"), "w"), indent=1)
    print(f"\nlink {len(url)} chars saved to figures/{PFX}bundles_ng.txt ({datetime.datetime.now(datetime.timezone.utc):%Y-%m-%d %H:%M} UTC)")


if __name__ == "__main__":
    main()
